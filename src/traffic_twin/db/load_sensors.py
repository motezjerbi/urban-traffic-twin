"""Load cleaned sensor rows into Supabase `sensor_hourly` (upsert on iu_ac, ts).

Needs SUPABASE_DB_URL (direct Postgres connection string) in the environment or .env.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from traffic_twin.config import load_config

log = logging.getLogger(__name__)

UPSERT_SQL = """
insert into public.sensor_hourly
    (iu_ac, libelle, zone, ts, flow_veh_h, occupancy_pct, traffic_state, lat, lon)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
on conflict (iu_ac, ts) do update set
    flow_veh_h = excluded.flow_veh_h,
    occupancy_pct = excluded.occupancy_pct,
    traffic_state = excluded.traffic_state
"""


def _read_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def to_rows(df: pd.DataFrame) -> list[tuple]:
    cols = [
        "iu_ac",
        "libelle",
        "zone",
        "ts",
        "flow_veh_h",
        "occupancy_pct",
        "traffic_state",
        "lat",
        "lon",
    ]
    clean = df[cols].astype(object).where(df[cols].notna(), None)
    return [tuple(r) for r in clean.itertuples(index=False, name=None)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--batch", type=int, default=5000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _read_dotenv()

    import psycopg  # optional dependency: pip install -e ".[db]"

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SystemExit("SUPABASE_DB_URL is not set (see .env.example)")

    cfg = load_config(args.config)
    df = pd.read_parquet(Path(cfg["paths"]["processed"]) / "sensor_hourly.parquet")
    rows = to_rows(df)
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        for i in range(0, len(rows), args.batch):
            cur.executemany(UPSERT_SQL, rows[i : i + args.batch])
            log.info("upserted %d / %d", min(i + args.batch, len(rows)), len(rows))


if __name__ == "__main__":
    main()
