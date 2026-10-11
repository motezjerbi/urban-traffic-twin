PY ?= python3
CONFIG ?= configs/data.yaml

.PHONY: install data explore score db-load lint test sim train eval app

install:
	$(PY) -m pip install -e ".[dev,db]"

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

lint:
	ruff check src tests

test:
	pytest -q

# Later parts of the roadmap
sim train eval app:
	@echo "'make $@' arrives in a later part of the roadmap"; exit 1
