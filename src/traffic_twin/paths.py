"""Default file locations. Paths are relative to the repo root, which is the working
directory of every CLI (`make` targets, `python -m traffic_twin...`)."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("configs")
SCENARIO_CONFIG = CONFIG_DIR / "scenario_base.yaml"
DATA_CONFIG = CONFIG_DIR / "data.yaml"
INTERSECTIONS_CONFIG = CONFIG_DIR / "intersections.yaml"
VALIDATION_CONFIG = CONFIG_DIR / "validation.yaml"  # pre-registered validation settings
CALIBRATION_DIR = CONFIG_DIR / "calibration"  # generated demand calibration files
PITIE_FIXED_CONFIG = CONFIG_DIR / "pitie_fixed.yaml"  # S0 on the real junction
PITIE_ACTUATED_CONFIG = CONFIG_DIR / "pitie_actuated.yaml"  # S1 on the real junction

REFERENCE_DIR = Path("data") / "reference"  # versioned sensor extracts (no data/processed)
FIELD_FILE = Path("data/field/pitie_measurements.yaml")  # site-visit measurements

SUMO_DIR = Path("sumo")
PLAIN_NET_DIR = SUMO_DIR / "network" / "plain"
NET_FILE = SUMO_DIR / "network" / "intersection.net.xml"
DEMAND_DIR = SUMO_DIR / "demand"

# File names inside the `paths.processed` directory of configs/data.yaml
SENSOR_HOURLY_FILE = "sensor_hourly.parquet"
HOURLY_PROFILE_FILE = "hourly_profile.parquet"
ARC_COVERAGE_FILE = "arc_coverage.csv"
