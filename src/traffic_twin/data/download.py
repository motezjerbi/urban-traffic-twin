"""Download City of Paris permanent sensor data around each candidate zone.

Uses the Opendatasoft Explore v2.1 export endpoint, filtered server side by distance
and date, so only the arcs near a candidate intersection are pulled.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import requests

from traffic_twin.config import load_config

log = logging.getLogger(__name__)


def build_where(lat: float, lon: float, radius_m: int, since: date) -> str:
    """ODSQL filter: arcs within radius_m of the point, measured since `since`."""
    return (
        f"within_distance(geo_point_2d, geom'POINT({lon} {lat})', {radius_m}m)"
        f" AND t_1h >= date'{since.isoformat()}'"
    )


def export_url(base_url: str, dataset: str, fmt: str = "parquet") -> str:
    return f"{base_url.rstrip('/')}/{dataset}/exports/{fmt}"


def download_zone(
    base_url: str,
    dataset: str,
    zone: dict,
    since: date,
    out_path: Path,
    session: requests.Session | None = None,
) -> Path:
    session = session or requests.Session()
    params = {"where": build_where(zone["lat"], zone["lon"], zone["radius_m"], since)}
    url = export_url(base_url, dataset)
    log.info("GET %s where=%s", url, params["where"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    src = cfg["source"]
    since = date.today() - timedelta(days=30 * src["months"])
    raw_dir = Path(cfg["paths"]["raw"])
    for name, zone in cfg["zones"].items():
        path = download_zone(
            src["base_url"], src["dataset"], zone, since, raw_dir / f"{name}.parquet"
        )
        log.info("%s -> %s (%.1f MB)", name, path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
