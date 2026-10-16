# Windows has no `python3` (or only the Store stub): use `python` there.
# No make on Windows? `python tasks.py <target>` runs the same targets.
ifeq ($(OS),Windows_NT)
PY ?= python
else
PY ?= python3
endif
CONFIG ?= configs/data.yaml

# Part 3 targets run on the Pitié configs unless SCENARIO is given: S0 for most of them,
# S1 for `compare` (tasks.py does the same); `part4` checks S2. Decided before SCENARIO
# gets its default.
ifeq ($(origin SCENARIO),undefined)
P3_S0 = configs/pitie_fixed.yaml
P3_S1 = configs/pitie_actuated.yaml
P4_S2 = configs/pitie_rules.yaml
else
P3_S0 = $(SCENARIO)
P3_S1 = $(SCENARIO)
P4_S2 = $(SCENARIO)
endif
BASELINE ?= configs/pitie_fixed.yaml
CALIB_SEEDS ?=

.PHONY: install data explore score db-load net demand sim sim-gui db-runs plots baseline
.PHONY: osm tls-check calib-data field-check field-sheets calibrate compare validate plots-pitie part3
.PHONY: evaluate plots-eval part4 provenance rebaseline
.PHONY: eta-daytypes eta-data eta-train eta-data-test eta-eval demo-runs api api-bench part5
.PHONY: rl-timing rl-train rl-select rl-eval part6 reproduce56
.PHONY: rl7-campaign rl7-holdout rl7-select rl7-publish rl7-stress rl7-sens rl7-devtest rl7-docs
.PHONY: part7 reproduce7
.PHONY: lint format test test-fast check train eval app

## Every extra the tasks and the tests use; PyTorch first from the CPU index (a plain
## install pulls the CUDA toolkit on Linux)
install:
	$(PY) -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev,db,sim,ml,api,rl]"

## Part 1: download Paris sensors near the candidate zones, clean them, plot hourly profiles
data:
	$(PY) -m traffic_twin.data.download --config $(CONFIG)
	$(PY) -m traffic_twin.data.clean --config $(CONFIG)

explore:
	$(PY) -m traffic_twin.data.explore --config $(CONFIG)

## Rank candidate intersections with the spec's weighted grid (section 2.2)
score:
	$(PY) -m traffic_twin.data.intersection_score configs/intersections.yaml

## Push cleaned sensor rows to Supabase (needs SUPABASE_DB_URL in .env)
db-load:
	$(PY) -m traffic_twin.db.load_sensors --config $(CONFIG)

## Part 2: SUMO network, demand and the fixed-time baseline
SCENARIO ?= configs/scenario_base.yaml
SEEDS ?= 1 2 3

## The grid build rewrites sumo/network/intersection.net.xml
net:
	$(PY) -m traffic_twin.sim.network --config $(SCENARIO)

demand:
	$(PY) -m traffic_twin.sim.demand --config $(SCENARIO)

sim:
	$(PY) -m traffic_twin.sim.run_scenario --config $(SCENARIO) --seed $(SEEDS)

sim-gui:
	$(PY) -m traffic_twin.sim.run_scenario --config $(SCENARIO) --seed 1 --gui

## Load finished runs into Supabase (tables + state logs in Storage)
db-runs:
	$(PY) -m traffic_twin.db.load_run data/results/*/

## KPI figures from data/results into docs/figures
plots:
	$(PY) -m traffic_twin.viz.plots

## Rebuild the fixed-time baseline: network, every seed, figures
baseline: net sim plots

## Part 3: Pitié-Salpêtrière junction (S0 fixed time, S1 actuated)
## OSM extract -> curated plain XML under sumo/network/pitie
osm:
	$(PY) -m traffic_twin.sim.osm_import --config $(P3_S0)

## Signal program checks: foe greens, intergreens, crossing red (exit 1 on errors)
tls-check:
	$(PY) -m traffic_twin.sim.tls_check --config $(P3_S0)

## Reference CSV and demand calibration file from data/raw
calib-data:
	$(PY) -m traffic_twin.data.calibrate --config $(CONFIG) --from-raw

field-check:
	$(PY) -m traffic_twin.validation.field --check --scenario $(P3_S0)

## Printable field-visit sheets (docs/field/pitie_field_sheets.pdf)
field-sheets:
	$(PY) -m traffic_twin.validation.field_sheets --scenario $(P3_S0)

## Simulation-in-the-loop demand scaling (CALIB_SEEDS, else the calibration seeds)
calibrate:
	$(PY) -m traffic_twin.validation.calibration_loop --scenario $(P3_S0)$(if $(CALIB_SEEDS), --seeds $(CALIB_SEEDS))

## Paired S0 (BASELINE) vs S1 on the evaluation seeds, running the missing ones
compare:
	$(PY) -m traffic_twin.evaluation.compare --baseline-config $(BASELINE) --scenario-config $(P3_S1) --run-missing

## Sim vs reality tables and docs/validation.md, running the missing seeds
validate:
	$(PY) -m traffic_twin.validation.report --scenario $(P3_S0) --run-missing --docs docs/validation.md

plots-pitie:
	$(PY) -m traffic_twin.viz.plots --scenario pitie_fixed pitie_actuated --out docs/figures/pitie

## The whole Part 3 chain: S0 = BASELINE, S1 = SCENARIO (default configs/pitie_actuated.yaml)
part3: P3_S0 = $(BASELINE)
part3: tls-check field-check calibrate compare validate plots-pitie

## Part 4: rule-based control (S2), Safety Guard, per-user evaluation of S0, S1 and S2
## Pre-registered evaluation (configs/evaluation.yaml) and docs/evaluation.md, running the
## missing seeds; exit 1 on a pairing error or an A-04 failure
evaluate:
	$(PY) -m traffic_twin.evaluation.evaluate --config configs/evaluation.yaml --run-missing --docs docs/evaluation.md

plots-eval:
	$(PY) -m traffic_twin.viz.evaluation --config configs/evaluation.yaml --out docs/figures/pitie/eval

## The whole Part 4 chain: tls-check on S2 (SCENARIO, default configs/pitie_rules.yaml)
part4: P3_S0 = $(P4_S2)
part4: tls-check evaluate plots-eval

## Provenance of the published Pitié runs (in-process steps of tasks.py, stdlib only):
## exit 1 unless every evaluation run and docs page comes from the clean HEAD
provenance:
	$(PY) tasks.py provenance

## After a commit: move the evaluation runs aside, re-make them from HEAD
## (compare validate part4) and require identical KPI rows
rebaseline:
	$(PY) tasks.py rebaseline

## Part 5: ETA dataset, quantile model, A-07 evaluation, API v1 and mock screens.
## Seeds, splits and budgets come from configs/eta.yaml (pre-registered, D38).
## ALLOW_DIRTY=1 builds a development dataset (id suffixed -dev) from an uncommitted tree.
ALLOW_DIRTY ?=

eta-daytypes:
	$(PY) -m traffic_twin.ml.daytypes --check

eta-data:
	$(PY) -m traffic_twin.ml.eta_dataset --config configs/eta.yaml --splits train val calib$(if $(ALLOW_DIRTY), --allow-dirty)

## Fit, conformal step and bundle, then freeze it before any test world exists (D51)
eta-train:
	$(PY) -m traffic_twin.ml.train_eta --config configs/eta.yaml
	$(PY) -m traffic_twin.ml.train_eta --config configs/eta.yaml --freeze

## Test worlds 2900-2999: generated and evaluated once, after the freeze (D50)
eta-data-test:
	$(PY) -m traffic_twin.ml.eta_dataset --config configs/eta.yaml --splits test$(if $(ALLOW_DIRTY), --allow-dirty)

eta-eval:
	$(PY) -m traffic_twin.ml.evaluate_eta --config configs/eta.yaml --split test --docs docs/eta.md --figures docs/figures/eta

## Demo / replay runs S0, S1, S2 on seed 4000 into data/demo, with ETA predictions
demo-runs:
	$(PY) -m traffic_twin.ml.predict demo --seeds 4000

api:
	$(PY) -m uvicorn traffic_twin.api.main:app --no-access-log --host 127.0.0.1 --port 8000

## Latency report (ENF-02) written between the latency markers of docs/api.md
api-bench:
	$(PY) -m traffic_twin.api.bench --results data/results data/demo --docs docs/api.md

part5: eta-daytypes eta-data eta-train eta-data-test eta-eval demo-runs api-bench

## Part 6: Gymnasium environment, first RL trainings, episode timing (configs/rl.yaml).
## rl-train reads train.s6 of configs/rl.yaml, so tasks.py runs it; ALGO, SEED, STEPS and
## N_ENVS narrow or override it.
ALGO ?=
SEED ?=
STEPS ?=
N_ENVS ?=

rl-timing:
	$(PY) -m traffic_twin.rl.evaluate timing

rl-train:
	$(PY) tasks.py rl-train$(if $(ALGO), --algo $(ALGO))$(if $(SEED), --seed $(SEED))$(if $(STEPS), --steps $(STEPS))$(if $(N_ENVS), --n-envs $(N_ENVS))

rl-select:
	$(PY) -m traffic_twin.rl.evaluate select

## Exploratory comparison on the development-test seeds 3020-3029, then docs/rl.md
rl-eval:
	$(PY) -m traffic_twin.rl.evaluate compare --seeds 3020 3021 3022 3023 3024 3025 3026 3027 3028 3029
	$(PY) -m traffic_twin.rl.evaluate kpi --seeds 3020 3021 3022 3023 3024 3025 3026 3027 3028 3029
	$(PY) -m traffic_twin.rl.evaluate docs --docs docs/rl.md --figures docs/figures/rl

part6: rl-timing rl-train rl-select rl-eval

## After a commit (D65): move the Parts 5-6 outputs aside, rerun part5 part6 from HEAD,
## require identical metrics and model/policy sha256 (in-process steps: tasks.py)
reproduce56:
	$(PY) tasks.py reproduce56

## Part 7: full RL training, tuning, weight sensitivity, S3 selection (configs/rl7.yaml,
## pre-registered D68-D90). Outputs under data/rl7/ (not versioned). SLOTS overrides
## protocol.slots; ALLOW_DIRTY=1 trains from an uncommitted tree (recorded, D80).
SLOTS ?=

rl7-campaign:
	$(PY) -m traffic_twin.rl.campaign --config configs/rl7.yaml --stages T S F$(if $(SLOTS), --slots $(SLOTS))$(if $(ALLOW_DIRTY), --allow-dirty)

## The finished runs are listed when the step runs (in-process step of tasks.py)
rl7-holdout:
	$(PY) tasks.py rl7-holdout

rl7-select:
	$(PY) -m traffic_twin.rl.study select

## Writes models/rl/<model_version>/ and prints the configs/pitie_rl_s3.yaml lines
rl7-publish:
	$(PY) -m traffic_twin.rl.study publish

rl7-stress:
	$(PY) -m traffic_twin.rl.study stress

rl7-sens:
	$(PY) -m traffic_twin.rl.sensitivity run
	$(PY) -m traffic_twin.rl.sensitivity tables

## The single look at the dev-test seeds 3020-3029 (refuses a second one)
rl7-devtest:
	$(PY) -m traffic_twin.rl.study devtest

rl7-docs:
	$(PY) -m traffic_twin.rl.report7 --docs docs/rl_training.md --figures docs/figures/rl7

## Stops after rl7-publish until configs/pitie_rl_s3.yaml exists (in-process steps)
part7:
	$(PY) tasks.py part7$(if $(SLOTS), --slots $(SLOTS))$(if $(ALLOW_DIRTY), --allow-dirty)

## After a commit (D82): rerun t1 seed 0, the S3 golden and docs/rl_training.md into
## data/rl7/_reproduce/<HEAD>/ and compare (in-process steps: tasks.py)
reproduce7:
	$(PY) tasks.py reproduce7

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest -q

## Tests that do not need the SUMO binaries
test-fast:
	$(PY) -m pytest -q -m "not sumo"

## What CI runs
check: lint test

# Later parts of the roadmap
train eval app:
	@echo "'make $@' arrives in a later part of the roadmap"; exit 1
