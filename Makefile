# Windows has no `python3` (or only the Store stub): use `python` there.
# No make on Windows? `python tasks.py <target>` runs the same targets.
ifeq ($(OS),Windows_NT)
PY ?= python
else
PY ?= python3
endif
CONFIG ?= configs/data.yaml

# Part 3 targets run on the Pitié configs unless SCENARIO is given: S0 for most of them,
# S1 for `compare` (tasks.py does the same). Decided before SCENARIO gets its default.
ifeq ($(origin SCENARIO),undefined)
P3_S0 = configs/pitie_fixed.yaml
P3_S1 = configs/pitie_actuated.yaml
else
P3_S0 = $(SCENARIO)
P3_S1 = $(SCENARIO)
endif
BASELINE ?= configs/pitie_fixed.yaml
CALIB_SEEDS ?=

.PHONY: install data explore score db-load net demand sim sim-gui db-runs plots baseline
.PHONY: osm tls-check calib-data field-check calibrate compare validate plots-pitie part3
.PHONY: lint format test test-fast check train eval app

install:
	$(PY) -m pip install -e ".[dev,db,sim]"

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
