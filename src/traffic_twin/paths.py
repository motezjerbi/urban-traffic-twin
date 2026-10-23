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
# Part 7 (pre-registered D68-D90); rl.yaml and pitie_rl.yaml stay the Part 6 record
RL7_CONFIG = CONFIG_DIR / "rl7.yaml"  # Part 7 pre-registration (extends rl.yaml)
RL7_CONFIG_DIR = CONFIG_DIR / "rl7"  # Part 7 run configs (t0, t1, v1, v2, p1, sensitivity)
PITIE_RL_S3_CONFIG = CONFIG_DIR / "pitie_rl_s3.yaml"  # S3: the selected Part 7 policy (D86)
# Parts 8-10 (pre-registered D91-D120)
PITIE_RL_S4_CONFIG = CONFIG_DIR / "pitie_rl_s3_nopreempt.yaml"  # S4: S3 without preemption
PITIE_AMB_RL_S3_CONFIG = CONFIG_DIR / "pitie_amb_rl_s3.yaml"  # ambulance-probe S3
PITIE_AMB_RL_S4_CONFIG = CONFIG_DIR / "pitie_amb_rl_s3_nopreempt.yaml"  # ambulance-probe S4
DEMO_CONFIG = CONFIG_DIR / "demo.yaml"  # demo script, ports and prerequisites (D116)
USERTESTS_CONFIG = CONFIG_DIR / "usertests.yaml"  # user-test protocol (D111)

REFERENCE_DIR = Path("data") / "reference"  # versioned sensor extracts (no data/processed)
FIELD_FILE = Path("data/field/pitie_measurements.yaml")  # site-visit measurements

SUMO_DIR = Path("sumo")
PLAIN_NET_DIR = SUMO_DIR / "network" / "plain"
NET_FILE = SUMO_DIR / "network" / "intersection.net.xml"
DEMAND_DIR = SUMO_DIR / "demand"

RESULTS_DIR = Path("data") / "results"  # run folders; seeds 1-10 behind the run lock (D92)
EVAL_DIR = RESULTS_DIR / "evaluation"  # evaluation.yaml outputs.dir
# Part 8 final outputs (D98), written by evaluation.final through `evaluate`
EVAL_FINAL_TABLE = EVAL_DIR / "final_table.csv"
EVAL_A06 = EVAL_DIR / "a06.json"
EVAL_SERIES = EVAL_DIR / "series.csv"
EVAL_DIGEST = EVAL_DIR / "digest.json"

# Parts 5-6 outputs (gitignored, D64); never written into data/results
ETA_DIR = Path("data") / "eta"  # ETA datasets, data/eta/<dataset_id>/
MODELS_DIR = Path("data") / "models"  # ETA bundles (models/eta) and the selected RL model
DEMO_DIR = Path("data") / "demo"  # demo / replay runs (seeds 4000-4009) with ETA files
RL_RUNS_DIR = Path("data") / "rl_runs"  # RL training runs and dev-test runs
RL7_DIR = Path("data") / "rl7"  # Part 7 runs, campaign, study, sensitivity (gitignored, D85)
MODELS_RL_DIR = Path("models") / "rl"  # committed S3 model, models/rl/<model_version>/ (D86)

# Parts 9-10 (D111-D117): demo, user tests, report and slides
DEMO_CHECK_DIR = Path("data") / "demo_check"  # rehearsal reports and pids (gitignored)
DEMO_DOCS_DIR = Path("docs") / "demo"  # runbook, rehearsal.json, rehearsal_log.yaml
DEMO_PACK = DEMO_DOCS_DIR / "demo_pack.zip"  # demo runs + evaluation outputs (D116)
USERTESTS_DIR = Path("docs") / "usertests"  # results stay under docs/ (never dirty a run)
USERTESTS_RESULTS = USERTESTS_DIR / "results.yaml"  # filled by hand after each session
USERTESTS_DOC = Path("docs") / "usertests.md"  # generated synthesis page
USERTESTS_KIT = USERTESTS_DIR / "usertest_kit.pdf"  # printable kit
USERTESTS_SYNTHESIS = USERTESTS_DIR / "synthesis.json"  # machine-readable synthesis (A-10)
REPORT_DIR = Path("docs") / "report"  # report.md, report.pdf, facts.json, build.json
REPORT_SRC = REPORT_DIR / "src"  # hand-written templates NN_*.md
REPORT_FACTS = REPORT_DIR / "facts.json"  # every result the report and slides quote
SLIDES_PPTX = Path("docs") / "slides" / "urban_traffic_twin.pptx"
APP_FIGURES = Path("docs") / "figures" / "app"  # app screenshots (report, slides)

# File names inside the `paths.processed` directory of configs/data.yaml
SENSOR_HOURLY_FILE = "sensor_hourly.parquet"
HOURLY_PROFILE_FILE = "hourly_profile.parquet"
ARC_COVERAGE_FILE = "arc_coverage.csv"
