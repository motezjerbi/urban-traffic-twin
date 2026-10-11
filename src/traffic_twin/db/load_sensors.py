"""Load cleaned sensor rows into Supabase `sensor_hourly` (upsert on iu_ac, ts).

Needs SUPABASE_DB_URL (direct Postgres connection string) in the environment or .env.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from traffic_twin.config import DATA_KEYS, load_config
from traffic_twin.data.clean import SENSOR_HOURLY_COLUMNS
from traffic_twin.db.env import connect
from traffic_twin.paths import DATA_CONFIG, SENSOR_HOURLY_FILE

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


def to_rows(df: pd.DataFrame) -> list[tuple]:
    cols = SENSOR_HOURLY_COLUMNS
    clean = df[cols].astype(object).where(df[cols].notna(), None)
    return [tuple(r) for r in clean.itertuples(index=False, name=None)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DATA_CONFIG)
    parser.add_argument("--batch", type=int, default=5000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config, DATA_KEYS)
    df = pd.read_parquet(Path(cfg["paths"]["processed"]) / SENSOR_HOURLY_FILE)
    rows = to_rows(df)
    with connect() as conn, conn.cursor() as cur:
        for i in range(0, len(rows), args.batch):
            cur.executemany(UPSERT_SQL, rows[i : i + args.batch])
            log.info("upserted %d / %d", min(i + args.batch, len(rows)), len(rows))


if __name__ == "__main__":
    main()
