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
PITIE_RULES_CONFIG = CONFIG_DIR / "pitie_rules.yaml"  # S2 rule based (Part 4)
EVALUATION_CONFIG = CONFIG_DIR / "evaluation.yaml"  # pre-registered S0/S1/S2 evaluation
# Parts 5-6 (pre-registered D37-D67)
ETA_CONFIG = CONFIG_DIR / "eta.yaml"  # ETA dataset, model and evaluation (Part 5)
RL_CONFIG = CONFIG_DIR / "rl.yaml"  # RL environment and training (Part 6)
API_CONFIG = CONFIG_DIR / "api.yaml"  # API v1 and mock screens (Part 5)
PITIE_RL_CONFIG = CONFIG_DIR / "pitie_rl.yaml"  # RL controller on the real junction
INCIDENT_SITES = CONFIG_DIR / "pitie" / "incident_sites.yaml"  # lane-slowdown sites (D45)

REFERENCE_DIR = Path("data") / "reference"  # versioned sensor extracts (no data/processed)
FIELD_FILE = Path("data/field/pitie_measurements.yaml")  # site-visit measurements

SUMO_DIR = Path("sumo")
PLAIN_NET_DIR = SUMO_DIR / "network" / "plain"
NET_FILE = SUMO_DIR / "network" / "intersection.net.xml"
DEMAND_DIR = SUMO_DIR / "demand"

EVAL_DIR = Path("data") / "results" / "evaluation"  # evaluation.yaml outputs.dir

# Parts 5-6 outputs (gitignored, D64); never written into data/results
ETA_DIR = Path("data") / "eta"  # ETA datasets, data/eta/<dataset_id>/
MODELS_DIR = Path("data") / "models"  # ETA bundles (models/eta) and the selected RL model
DEMO_DIR = Path("data") / "demo"  # demo / replay runs (seeds 4000-4009) with ETA files
RL_RUNS_DIR = Path("data") / "rl_runs"  # RL training runs and dev-test runs

# File names inside the `paths.processed` directory of configs/data.yaml
SENSOR_HOURLY_FILE = "sensor_hourly.parquet"
HOURLY_PROFILE_FILE = "hourly_profile.parquet"
ARC_COVERAGE_FILE = "arc_coverage.csv"
