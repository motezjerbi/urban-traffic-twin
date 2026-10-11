"""Load finished simulation runs into Supabase: `run`, `kpi_run` and `trip` rows in
Postgres, and the full state log as Parquet in the private Storage bucket `sim-logs`.

    python -m traffic_twin.db.load_run data/results/baseline_fixed_seed42 [...]

Needs SUPABASE_DB_URL, and SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY for the upload.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests

from traffic_twin.db.env import connect

log = logging.getLogger(__name__)
BUCKET = "sim-logs"

RUN_SQL = """
insert into public.run (run_id, scenario, seed, code_version, config, started_at, duration_s,
                        state_log_path)
values (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
on conflict (run_id) do nothing
"""
KPI_SQL = """
insert into public.kpi_run (run_id, scenario, seed, user_type, kpi, value)
values (%s, %s, %s, %s, %s, %s)
on conflict (run_id, user_type, kpi) do update set value = excluded.value
"""
TRIP_SQL = """
insert into public.trip (run_id, id, type, depart_s, arrival_s, duration_s, time_loss_s,
                         waiting_s, stops)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
on conflict (run_id, id) do nothing
"""
TRIP_COLUMNS = ["run_id", "id", "type", "depart_s", "arrival_s", "duration_s", "time_loss_s",
                "waiting_s", "stops"]  # fmt: skip


def storage_path(meta: dict) -> str:
    return f"{meta['scenario']}/{meta['run_id']}/state_log.parquet"


def trip_rows(trips: pd.DataFrame) -> list[tuple]:
    df = trips.reindex(columns=TRIP_COLUMNS)
    df = df.astype(object).where(df.notna(), None)
    return [tuple(r) for r in df.itertuples(index=False, name=None)]


def upload_state_log(path: Path, object_path: str) -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        log.warning("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set: state log not uploaded")
        return
    r = requests.post(
        f"{url.rstrip('/')}/storage/v1/object/{BUCKET}/{object_path}",
        data=path.read_bytes(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/octet-stream",
            "x-upsert": "true",
        },
        timeout=300,
    )
    r.raise_for_status()


def load_run_dir(run_dir: Path, conn) -> str:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    kpis = pd.read_csv(run_dir / "kpi_run.csv")
    trips = pd.read_parquet(run_dir / "trip.parquet")
    obj = storage_path(meta)
    with conn.cursor() as cur:
        cur.execute(
            RUN_SQL,
            (meta["run_id"], meta["scenario"], meta["seed"], meta["code_version"],
             json.dumps(meta["config"]), meta["started_at"], meta["duration_s"], obj),
        )  # fmt: skip
        cur.executemany(
            KPI_SQL,
            [(r.run_id, r.scenario, int(r.seed), r.user_type, r.kpi, float(r.value))
             for r in kpis.itertuples(index=False)],
        )  # fmt: skip
        cur.executemany(TRIP_SQL, trip_rows(trips))
    conn.commit()
    upload_state_log(run_dir / "state_log.parquet", obj)
    log.info("%s: run %s, %d kpis, %d trips", run_dir, meta["run_id"], len(kpis), len(trips))
    return meta["run_id"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with connect() as conn:
        for d in args.run_dirs:
            load_run_dir(d, conn)


if __name__ == "__main__":
    main()
