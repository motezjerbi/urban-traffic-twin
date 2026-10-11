# Windows has no `python3` (or only the Store stub): use `python` there.
# No make on Windows? `python tasks.py <target>` runs the same targets.
ifeq ($(OS),Windows_NT)
PY ?= python
else
PY ?= python3
endif
CONFIG ?= configs/data.yaml

.PHONY: install data explore score db-load net demand sim sim-gui db-runs plots baseline
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

net:
	$(PY) -m traffic_twin.sim.network

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
