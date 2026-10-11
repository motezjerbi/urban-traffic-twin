# Urban Traffic Twin

Digital twin of **one signalised Paris intersection** in Eclipse SUMO, calibrated on City of
Paris permanent sensor data (hourly flow and occupancy). It compares four signal controllers
(fixed time, actuated, rule based, reinforcement learning) and shows the effect for five kinds
of users (student, ambulance, bus, pedestrian, operator). The spec in `docs/` is the source of
truth for scope.

Storage is **Supabase**: the spec's tables (section 9.1) live in Postgres, and large raw
simulation logs go to the private Storage bucket `sim-logs` as Parquet.

## Roadmap

Built in 10 parts, one demoable PR each. This branch is **Part 1**: repo setup, Supabase
schema, Paris sensor download and exploration, and the intersection choice grid.

| Part | Content |
| --- | --- |
| 1 | Repo, Supabase schema, sensor data exploration, intersection choice |
| 2 | SUMO network (signals, crossings, bus), provisional demand, fixed-time baseline, first KPIs |
| 3 | Demand calibration, sim vs reality validation, actuated baseline |
| 4 | Rule-based controller (preemption, bus TSP, pedestrians), Safety Guard, per-user KPIs |
| 5 | ETA model (XGBoost quantiles), API v1 |
| 6-7 | Gymnasium env and RL training |
| 8 | Final evaluation (10 seeds) |
| 9 | App views and replay, user tests |
| 10 | Report and demo |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install          # pip install -e ".[dev,db]"
make data             # download sensors around the 3 candidate zones, clean them
make explore          # coverage table + hourly profile figures in docs/figures/
make score            # rank candidates with the spec's weighted grid
make test lint
```

`make data` calls the opendata.paris.fr Explore API (no key needed) and keeps only arcs
within `radius_m` of each zone in `configs/data.yaml`.

## Choosing the intersection

1. `make data && make explore` and look at `data/processed/arc_coverage.csv` and the figures.
   A candidate needs a sensor with at least ~80% coverage on its main approaches.
2. Update the `sensor` score of each candidate in `configs/intersections.yaml`
   (0 means no usable sensor, which eliminates it), and adjust the other scores after
   checking Street View and the bus map.
3. `make score` prints the ranking and the chosen intersection.

## Supabase

```bash
cp .env.example .env    # fill SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_DB_URL
supabase db push        # or apply supabase/migrations/*.sql from the SQL editor
make db-load            # upsert cleaned sensor rows into sensor_hourly
```

`.mcp.json` registers the Supabase MCP server for Claude Code in this project.
All tables have row level security on with no policies, so only the service role
(backend and API) can read or write them.

## Layout

```
configs/            data.yaml, intersections.yaml, scenario_base.yaml, rl.yaml
data/               raw/, processed/, results/ (not versioned)
sumo/               network/, demand/ (Part 2)
src/traffic_twin/
  data/             download, clean, explore, intersection_score
  db/               Supabase loaders
  sim/ control/ rl/ ml/ api/ app/   later parts
supabase/migrations/
tests/
docs/
```
