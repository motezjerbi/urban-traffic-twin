# Architecture

Developer guide to `src/traffic_twin/`. The README covers setup and commands; the spec
(`docs/cahier_des_charges.docx`) covers scope. All CLIs run from the repo root: the paths
in `paths.py` and in the configs are relative to it.

## Module map

| Module | Role |
| --- | --- |
| `paths.py` | Default file locations (configs, SUMO network, demand dir, processed file names) |
| `config.py` | `load_config` / `load_scenario` with required dotted keys, `get(cfg, "a.b")`, `ConfigError` |
| `data/download`, `data/clean` | Paris sensor API to `data/raw/`, cleaned hourly rows to `data/processed/` |
| `data/explore`, `data/intersection_score` | Coverage table, hourly profile figures, candidate ranking |
| `db/load_sensors`, `db/load_run` | Upserts into Supabase; state logs uploaded to the `sim-logs` bucket |
| `sim/network` | netconvert from `sumo/network/plain/` to `intersection.net.xml` |
| `sim/demand` | Scenario config to `sumo/demand/<scenario>.rou.xml` and additional file |
| `sim/sumo` | Finds the binaries, picks libsumo (headless) or TraCI (GUI), builds the command line |
| `sim/signal` | `SignalPlan`: stage requests to SUMO phases, clearance and min green |
| `sim/session` | `SimSession`: one SUMO run driven step by step |
| `sim/results` | SUMO outputs to `kpi_run.csv`, `trip.parquet`, `state_log.parquet`, `run.json` |
| `sim/kpi` | KPIs per user type in the spec 9.4 long format |
| `sim/run_scenario` | CLI: controller loop over seeds, writes results, prints a summary |
| `control/base` | `Observation`, `Controller` |
| `control/registry` | `@register(name)`, `make_controller(cfg)` |
| `control/fixed` | Fixed-time baseline (`controller: fixed`) |
| `viz/plots`, `viz/style` | KPI figures from `data/results/`, shared figure style |

## Data flow

```
Part 1 (sensors)
  opendata.paris.fr --download--> data/raw/ --clean--> data/processed/sensor_hourly.parquet
                                                         |-- explore --> docs/figures/profile_*.png
                                                         '-- load_sensors --> Supabase sensor_hourly

Part 2 (simulation)
  configs/scenario_base.yaml
     |-- demand.write_demand --> sumo/demand/*.rou.xml
     |-- registry.make_controller --> Controller
     v
  SimSession (libsumo or TraCI) <--- stage --- Controller.decide(Observation)
     |  tripinfo.xml, stops.xml, state_rows
     v
  results.collect_results --> kpi.compute_kpis
     v
  data/results/<scenario>_seed<n>/ {kpi_run.csv, trip.parquet, state_log.parquet, run.json}
     |-- viz.plots --> docs/figures/{kpi_by_user,wait_distribution,queue_timeline}.png
     '-- db.load_run --> Supabase run, kpi_run, trip  +  state_log.parquet in sim-logs
```

`run.json` holds the run id, seed, git short hash, the full scenario config and wall
time, so every row in Supabase can be traced back to the config that produced it.

## Configuration

`load_scenario` checks the keys in `config.SCENARIO_KEYS` (absent or empty values count as
missing) and raises `ConfigError`. Code that needs more uses `get(cfg, "dotted.key")`,
which raises `ConfigError` naming the key; controllers raise it too for bad or unknown
names. `run_scenario` turns `ConfigError` into `config error: ...` and a non-zero exit,
without a traceback.

`sumo_command` adds `--gui-settings-file sumo/gui/view.xml` only for GUI runs. The view
file is display only (colours, viewport, delay); headless runs never read it.

## Stepping loop

`SimSession` is the single place that talks to SUMO. `run_scenario` and the future
Gymnasium env use the same three calls:

```python
with SimSession(cfg, seed, out_dir) as sim:       # writes demand, starts SUMO
    controller.reset(seed)
    while not sim.done:                            # demand window, then drain
        sim.step(controller.decide(sim.observe()))
```

- `observe()` returns an `Observation` (time since scenario start, current stage, time in
  stage, `in_transition`, halting vehicles per approach). It is computed once per step.
- `step(stage)` passes the request to `SignalPlan`, sets the resulting phase, advances
  SUMO by `step_length_s` and returns the phase shown.
- `done` is true once the demand window is over and the network is empty, or after
  `DRAIN_S` (1800 s) of drain. The controller keeps running during the drain; the state
  log only covers the demand window.
- If SUMO goes away (sumo-gui window closed), any call raises `SimulationClosed`.
  `run_scenario.run` removes SUMO's partial outputs and re-raises; the CLI logs a warning,
  writes no results and exits with code 1.

A Gymnasium env (Parts 6-7) wraps the same calls; one action may hold for several steps:

```python
def reset(self, seed=None, options=None):
    if self.sim: self.sim.close()
    self.sim = SimSession(self.cfg, seed, self.out_dir, log_state=False).start()
    return self.encode(self.sim.observe()), {}

def step(self, action):
    for _ in range(self.steps_per_action):
        self.sim.step(int(action))
        if self.sim.done: break
    obs = self.sim.observe()
    return self.encode(obs), self.reward(obs), self.sim.done, False, {}
```

## Controllers

Contract (`control/base.py`): `decide(obs) -> int` returns the index of the wanted green
stage, in `network.stages` order. Controllers never pick SUMO phases and never handle
yellow or all-red; `reset(seed)` clears state between runs; `from_config(cfg)` builds the
controller from the scenario config, usually from a section named after it.

`control/registry.py` maps names to classes. `make_controller(cfg)` reads `controller:`
and imports every module of `traffic_twin.control` on first lookup, so a new controller
needs no other wiring. Adding the Part 3 actuated controller:

1. Create `src/traffic_twin/control/actuated.py`:

   ```python
   from traffic_twin.config import get
   from traffic_twin.control.base import Controller, Observation
   from traffic_twin.control.registry import register

   @register("actuated")
   class ActuatedController(Controller):
       def __init__(self, max_green_s: float, gap_s: float): ...

       @classmethod
       def from_config(cls, cfg):
           return cls(get(cfg, "actuated.max_green_s"), get(cfg, "actuated.gap_s"))

       def reset(self, seed: int) -> None: ...
       def decide(self, obs: Observation) -> int: ...
   ```

2. Add a scenario config (e.g. `configs/scenario_actuated.yaml`) with
   `scenario: baseline_actuated`, `controller: actuated` and an `actuated:` section; keep
   `demand` identical to the fixed-time scenario so the comparison is fair.
3. `python tasks.py sim --scenario configs/scenario_actuated.yaml`, then
   `python -m traffic_twin.viz.plots --scenario baseline_fixed baseline_actuated`.

Keep heavy imports (torch, xgboost) inside functions: the registry imports every module
in the package. Detector data beyond queues goes into `Observation.extras`, filled in
`SimSession.observe()`.

## Invariants

- **Reproducible baseline.** Same config and seed give identical KPI values (only
  `run_id`, timestamps and wall time differ). A refactor of `sim/` or `control/` must
  leave the baseline table in the README unchanged; check by diffing `kpi_run.csv` values
  for seeds 1-3 before and after.
- **Same demand everywhere.** Flows use per-second probabilities, so every scenario sees
  the same expected demand and the seed only changes the draw (`tests/test_demand.py`).
- **Clearance is never skipped.** `SignalPlan.request` always queues the current stage's
  clearance phases (pedestrian clearance, yellow, all-red) before a new green, and
  ignores further requests while a transition runs.
- **Minimum green.** A switch requested before the green has lasted
  `safety.min_green_s` is held; the controller asks again on a later step. The fixed
  green times must be at least that value (`tests/test_signal.py`).
- **Stage order.** Controllers must serve the stages in the configured cycle order
  (`SignalPlan` does not enforce it): each LEFT stage's clearance is what ends that
  axis's permissive left turns.
- **No partial results.** A run either writes all four result files or none.
