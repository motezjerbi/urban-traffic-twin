# Urban Traffic Twin

Digital twin of **one signalised Paris intersection** in Eclipse SUMO, calibrated on City of
Paris permanent sensor data (hourly flow and occupancy). It compares four signal controllers
(fixed time, actuated, rule based, reinforcement learning) and shows the effect for five kinds
of users (student, ambulance, bus, pedestrian, operator). The spec in `docs/` is the source of
truth for scope.

Storage is **Supabase**: the spec's tables (section 9.1) live in Postgres, and large raw
simulation logs go to the private Storage bucket `sim-logs` as Parquet.

## Roadmap

Built in 10 parts, one demoable PR each. Part 1 (repo, Supabase schema, sensor
exploration, intersection grid) is merged; this branch adds **Part 2**: the SUMO
network, provisional demand, the fixed-time baseline and the first KPIs.

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
make net sim          # build the SUMO network, run the fixed-time baseline (seeds 1 2 3)
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

## Simulation (Part 2)

`sumo/network/plain/` describes a **provisional** 4-arm signalised intersection: a main
axis (W-E) with 2 lanes plus a left-turn lane on each approach, a cross street (N-S)
with 2 approach lanes, sidewalks and zebra crossings on every arm, a bus line B1 with a
stop, and one ambulance trip. It stands in for the real Paris intersection, which
replaces it (OSM extract) once Part 1's choice is made; the code does not depend on
the geometry beyond `configs/scenario_base.yaml`.

- `make net` runs netconvert. The signal program has 4 green stages (EW, EW_LEFT, NS,
  NS_LEFT), each followed by pedestrian clearance, yellow and all-red phases.
- `make sim SEEDS="1 2 3"` writes demand from the config, runs SUMO headless through
  libsumo (`make sim-gui` to watch it), and writes per run in `data/results/<scenario>_seed<n>/`:
  `kpi_run.csv` (spec 9.4 format), `trip.parquet`, `state_log.parquet` and `run.json`.
- Controllers implement `control.base.Controller.decide(obs) -> stage`; `sim.signal.SignalPlan`
  always runs the clearance phases before a new green, so no controller can skip a yellow.
- Flows use per-second probabilities, so every scenario sees the same expected demand and
  each seed gives a different draw (spec 13.3).

Fixed-time baseline (cycle 100 s, provisional demand 07:30-09:30, mean of seeds 1-3):

| User | KPI | Value |
| --- | --- | --- |
| Car | mean time loss | 32.9 s |
| Car | wait p95 | 61 s |
| Bus | mean delay | 28.3 s |
| Bus | headway std | 28.8 s |
| Pedestrian | mean wait (time lost) | 56.8 s |
| Pedestrian | max wait | 137 s (spec target 90 s) |
| Ambulance | time loss | 62.5 s |

The ambulance and pedestrian numbers are what the rule-based controller in Part 4 targets.

## Supabase

```bash
cp .env.example .env    # fill SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_DB_URL
supabase db push        # or apply supabase/migrations/*.sql from the SQL editor
make db-load            # upsert cleaned sensor rows into sensor_hourly
make db-runs            # load run, kpi_run, trip rows; upload state logs to sim-logs
```

`.mcp.json` registers the Supabase MCP server for Claude Code in this project.
All tables have row level security on with no policies, so only the service role
(backend and API) can read or write them.

## Layout

```
configs/            data.yaml, intersections.yaml, scenario_base.yaml, rl.yaml
data/               raw/, processed/, results/ (not versioned)
sumo/               network/ (plain XML + generated .net.xml), demand/ (generated)
src/traffic_twin/
  data/             download, clean, explore, intersection_score
  db/               Supabase loaders (sensors, runs)
  sim/              network, demand, signal sequencing, run_scenario, kpi
  control/          Controller interface, fixed time
  rl/ ml/ api/ app/ later parts
supabase/migrations/
tests/
docs/
```
