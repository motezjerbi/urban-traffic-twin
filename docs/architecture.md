# Architecture

Developer guide to `src/traffic_twin/`. The README covers setup and commands; the spec
(`docs/cahier_des_charges.docx`) covers scope. All CLIs run from the repo root: the paths
in `paths.py` and in the configs are relative to it.

Two scenario families share the code: the Part 2 **grid** (`configs/scenario_base.yaml`,
frozen) and the Part 3 **Pitié-Salpêtrière junction** (`configs/pitie_fixed.yaml` = S0,
`configs/pitie_actuated.yaml` = S1). Decision numbers (D1-D18) refer to
[Part 3 decisions](#part-3-decisions).

## Module map

| Module | Role |
| --- | --- |
| `paths.py` | Default file locations (configs, SUMO network, demand dir, processed file names, validation config, field file, reference dir) |
| `config.py` | `load_config` / `load_scenario` with `extends`, required dotted keys, `get(cfg, "a.b")`, `ConfigError`; `config_sources`, `approach_in_edges` / `approach_out_edges`, `config_hash`, `demand_hash` |
| `data/download`, `data/clean` | Paris sensor API to `data/raw/`, cleaned hourly rows to `data/processed/` (`ts` = published end-of-period stamp; profiles on `hour_start`) |
| `data/explore`, `data/intersection_score` | Coverage table, hourly profile figures, candidate ranking |
| `data/calendar` | One typical-day filter for calibration, validation and the profile figures: period start from the stamp (D3), Tue-Thu school term without public holidays inside `calendar.typical_period` (other such days are hold-out days), robust MAD filter, uniform weights inside each sensor hour (D4, D5). The filter is shared, the day set is not: completeness and MAD are judged per window, so a validation slot can keep other days than the calibration's 07:30-09:30 window |
| `data/calibrate` | Versioned reference extract `data/reference/pitie_302_hours.csv` (from `data/raw/`) and the calibration file `configs/calibration/pitie_302.yaml` (slot flows per arm, turn shares) |
| `db/load_sensors`, `db/load_run` | Upserts into Supabase; state logs uploaded to the `sim-logs` bucket |
| `sim/osm_import` | OSM extract to curated plain XML (`sumo/network/pitie/pitie.{nod,edg,con}.xml`): semantic edge ids, explicit connections and crossings |
| `sim/network` | netconvert: the grid from `sumo/network/plain/` (legacy build), or the scenario's `network.build` (plain XML + `.tll.xml` + options) |
| `sim/tls_check` | Static checks of a signal program: foe greens, intergreens from the geometry, worst-case red of each crossing; CLI exits 1 on errors |
| `sim/demand` | Scenario config to `sumo/demand/<scenario>.rou.xml` and `<scenario>.stops.add.xml`: legacy grid schema, or calibrated slot flows, bus lines and pinned walks. With `demand.bus.counted_by_sensors` an arm's car demand is its sensor flow minus its scheduled buses (sensors and E1 stations both count every vehicle) |
| `sim/detectors` | Passive detectors of a run: control loops (actuated), station loops (validation), E2 queue detectors, signal switch times; parsers for their outputs |
| `sim/sumo` | Finds the binaries, picks libsumo (headless) or TraCI (GUI), builds the command line |
| `sim/signal` | `SignalPlan`: stage requests to SUMO phases, clearance and min green; `SignalState` snapshot for the guard |
| `sim/session` | `SimSession`: one SUMO run driven step by step, guard included |
| `sim/audit` | Audit of the phases shown during a run (A-04 evidence) |
| `sim/results` | SUMO outputs to `kpi_run.csv`, `trip.parquet`, `state_log.parquet`, `run.json`, and the measurement tables |
| `sim/kpi` | KPIs per user type in the spec 9.4 long format; SUMO statistics for the health block |
| `sim/run_scenario` | CLI: controller loop over seeds, writes results, prints a summary and a health line |
| `control/base` | `Observation`, `Controller`, `EXTRA_GAP_S`, `EXTRA_CALLS` |
| `control/registry` | `@register(name)`, `make_controller(cfg)` |
| `control/fixed` | Fixed-time baseline (`controller: fixed`) |
| `control/actuated` | S1: SUMO-style gap-based actuated control (`controller: actuated`) |
| `control/safety` | `SafetyGuard` between every controller and `SignalPlan`; `safe_transitions` |
| `validation/__init__` | Pre-registered settings (`configs/validation.yaml`), clock helpers |
| `validation/reference` | Real flows per station and slot from the versioned extract (typical days) |
| `validation/field` | Field file loading, schema checks, inventory of what is measured |
| `validation/flows`, `validation/signal`, `validation/queues` | Flow (A-02, GEH), signal timing and queue tables |
| `validation/calibration_loop` | Simulation-in-the-loop demand scaling on the calibration seeds; `docs/calibration_log.yaml` |
| `validation/report` | Validation CSVs, `summary.json` and `docs/validation.md` |
| `evaluation/stats` | Exact Wilcoxon signed-rank, paired t CI, Hodges-Lehmann, Holm, GEH (numpy only) |
| `evaluation/compare` | Paired S0 vs S1 comparison over the evaluation seeds, equity flags |
| `viz/plots`, `viz/style` | KPI figures from `data/results/`, shared figure style |
| `viz/validation` | Validation flow figure, queue figure, paired differences figure |

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

```
Part 3 (Pitié-Salpêtrière junction)
  network (run once; outputs versioned)
    sumo/network/osm/pitie.osm --osm_import--> sumo/network/pitie/pitie.{nod,edg,con}.xml
    + pitie.tll.xml --network --config--> pitie.net.xml --tls_check--> program report
  demand calibration (outputs versioned)
    data/raw/pitie_salpetriere.parquet --calibrate --from-raw--> data/reference/pitie_302_hours.csv
    data/reference/pitie_302_hours.csv --calibrate--> configs/calibration/pitie_302.yaml
  runs
    configs/pitie_fixed.yaml (extends configs/pitie/*.yaml), configs/pitie_actuated.yaml
       |-- demand.write_demand --> sumo/demand/<scenario>.rou.xml, <scenario>.stops.add.xml
       |-- detectors.write_detectors --> <run dir>/detectors.add.xml
       v
    SimSession: Controller.decide --> SafetyGuard.apply --> SignalPlan --> SUMO
       v
    data/results/<scenario>_seed<n>/ {kpi_run.csv, trip, state_log, run.json,
                                      detector.parquet, queue.parquet, signal.parquet}
  evaluation
    calibration_loop (S0, seeds 101-103) --> docs/calibration_log.yaml
    evaluation.compare (S0 vs S1, seeds 1-10) --> data/results/comparison/<a>__vs__<b>.{csv,md}
                                                  docs/figures/pitie/comparison_<a>_<b>.png
    validation.report (runs + data/reference + data/field) --> data/results/validation/<scenario>/
                                                  docs/validation.md, docs/figures/pitie/validation_flows.png
    viz.plots --> docs/figures/pitie/{kpi_by_user,wait_distribution,queue_timeline}.png
```

The calibration file is always rebuilt from the versioned CSV, and validation reads the
same CSV, so the tests and CI need neither `data/raw/` nor `data/processed/`.

## Configuration

`load_scenario` checks the keys in `config.SCENARIO_KEYS` (absent or empty values count as
missing) and raises `ConfigError`. Code that needs more uses `get(cfg, "dotted.key")`,
which raises `ConfigError` naming the key; controllers raise it too for bad or unknown
names. `run_scenario` turns `ConfigError` into `config error: ...` and a non-zero exit,
without a traceback. A scenario needs `demand.approaches` (grid) or `demand.calibration`
(Pitié).

**`extends` and fragments.** A config may build on others with `extends:` (one path or a
list, relative to the including file's folder). The bases are merged in order, then the
including file on top: mappings merge key by key, lists and scalars are replaced. The
`extends` key is removed, a cycle raises `ConfigError`, and required keys are checked after
the merge. `config_sources(path)` lists the files in merge order; `run.json` records them
as `config_sources`.

| File | Sections |
| --- | --- |
| `configs/pitie/network.yaml` | `network` (net file, TLS id, GUI view, `build`, `osm`, `stages`, `approaches` with in and out edges, `crossings` with walk positions, `bus_stops`, `signal_groups`, `reference_group`, `assumptions`), `fixed.green_s` |
| `configs/pitie/demand.yaml` | `demand` (calibration file, `arm_scale`, bus lines, pedestrians, assumptions), `emergency` |
| `configs/pitie/control.yaml` | `detectors.control`, `actuated`, `safety` |
| `configs/pitie/measurement.yaml` | `measurement` (station loops, queue detectors, signal switches) |
| `configs/pitie_fixed.yaml` | S0: `extends` the four fragments; scenario, window, controller, SUMO options, outputs |
| `configs/pitie_actuated.yaml` | S1: `extends: pitie_fixed.yaml`; scenario name and controller only |
| `configs/validation.yaml` | Not a scenario: pre-registered validation settings (reference, seeds, slots, stations, thresholds, calibration loop, headline KPIs, field file) |

`approach_in_edges` / `approach_out_edges` accept the grid form (`{W: w_in}`) and the
Pitié form (`{W: {in_edges: [...], out_edges: [...]}}`).

**Hashes.** `config_hash` is the sha256 of the merged config as canonical JSON, without
`outputs`. `demand_hash` covers what the demand depends on: `begin_s`, `duration_s`,
`step_length_s`, `demand`, `emergency`, `network.{net_file, approaches, crossings,
bus_stops}`, plus the sha256 of the calibration file and of the net file. S0 and S1 share
it by construction; a comparison refuses runs whose `demand_hash` differs.

`sumo_command` appends `sumo.options` (Pitié: `--collision.check-junctions true`) and adds
`--statistic-output` for the health block. It adds `--gui-settings-file` only for GUI runs:
`network.gui_settings` (Pitié: `sumo/gui/pitie_view.xml`), default `sumo/gui/view.xml`. The
view file is display only (colours, viewport, delay); headless runs never read it.

## Run outputs

Per run in `<outputs.dir>/<scenario>_seed<n>/`:

| File | Written | Content |
| --- | --- | --- |
| `kpi_run.csv` | always | KPIs per user type, spec 9.4 long format |
| `trip.parquet` | always | Vehicle trips and pedestrian walks, with `depart_delay_s` |
| `state_log.parquet` | always | Intersection, vehicle and pedestrian state every `outputs.state_log_every_s` |
| `run.json` | always | Run id, seed, code version, merged config, `config_hash`, `demand_hash`, `config_sources`, `health`, `safety` |
| `detector.parquet` | `measurement.stations` | `run_id, det_id, station, lane, begin_s, end_s, n_veh, occupancy_pct, mean_speed` per station loop and period |
| `queue.parquet` | `measurement.queues` | `run_id, t, approach, jam_veh, jam_m, halting_veh` per sample |
| `signal.parquet` | `measurement.tls_switch` | `run_id, link, begin_s, end_s, duration_s`: green intervals per TLS link |

Times are relative to the scenario start. SUMO's raw outputs stay next to them:
`tripinfo.xml`, `stops.xml`, `statistics.xml`, and with detectors `detectors.add.xml`,
`e1.xml`, `tls_switch.xml`. The grid has no `measurement` section, so it writes only the
first four files.

- `health`: `teleports`, `collisions`, `emergency_stops`, `emergency_braking`,
  `vehicles_waiting_end` (from `statistics.xml`), `depart_delay_mean`, `depart_delay_max`,
  `drain_timeout`, plus the person counts of `statistics.xml`.
- `safety`: `interventions` per guard rule (`min_green`, `order`, `max_green`,
  `starvation`) and `audit` (`sim.audit.AUDIT_KEYS`).
- KPI rows added in Part 3 (existing rows unchanged): `depart_delay_mean` and
  `depart_delay_max` per vehicle type; bus `dwell_mean`; pedestrian `ped_wait_p95`,
  `ped_wait_single_mean`, `ped_wait_single_p95`, `ped_wait_single_max`. Bus `headway_std`
  is the mean over (bus stop, line) groups with at least 3 arrivals, which equals the old
  value for one line and one stop.

## Stepping loop

`SimSession` is the single place that talks to SUMO. `run_scenario` and the future
Gymnasium env use the same three calls:

```python
with SimSession(cfg, seed, out_dir) as sim:  # writes demand, starts SUMO
    controller.reset(seed)
    while not sim.done:  # demand window, then drain
        sim.step(controller.decide(sim.observe()))
```

- `start()` writes the demand and the detector file, starts SUMO, reads the program's
  phase states, and builds the `SignalPlan` and the `SafetyGuard` (`SafetyGuard.from_config`
  checks the config: link-safe cycle order, greens under the caps, `max_red_s` above the
  longest legal cycle).
- `observe()` returns an `Observation` (time since scenario start, current stage, time in
  stage, `in_transition`, halting vehicles per approach summed over its in-edges, and
  `extras` when control loops exist). It is computed once per step.
- `step(stage)` passes the request through `SafetyGuard.apply(stage, plan.state(t))`, then
  to `SignalPlan`, sets the resulting phase and advances SUMO by `step_length_s`. It records
  the phase shown (`phases_shown`, audited after the run), each pedestrian's longest single
  wait (`person.getWaitingTime`) and, on state-log ticks, the E2 queue detectors. It returns
  the phase shown.
- `done` is true once the demand window is over and the network is empty, or after
  `DRAIN_S` (1800 s) of drain. The controller keeps running during the drain; the state
  log only covers the demand window.
- If SUMO goes away (sumo-gui window closed), any call raises `SimulationClosed`.
  `run_scenario.run` removes SUMO's partial outputs and re-raises; the CLI logs a warning,
  writes no results and exits with code 1.

A Gymnasium env (Parts 6-7) wraps the same calls; one action may hold for several steps:

```python
def reset(self, seed=None, options=None):
    if self.sim:
        self.sim.close()
    self.sim = SimSession(self.cfg, seed, self.out_dir, log_state=False).start()
    return self.encode(self.sim.observe()), {}


def step(self, action):
    for _ in range(self.steps_per_action):
        self.sim.step(int(action))
        if self.sim.done:
            break
    obs = self.sim.observe()
    return self.encode(obs), self.reward(obs), self.sim.done, False, {}
```

The guard sits inside `step`, so an RL policy cannot bypass it either.

## Controllers

Contract (`control/base.py`): `decide(obs) -> int` returns the index of the wanted green
stage, in `network.stages` order. Controllers never pick SUMO phases and never handle
yellow or all-red; `reset(seed)` clears state between runs; `from_config(cfg)` builds the
controller from the scenario config, usually from a section named after it. During a
transition `obs.stage` is the stage being left.

Detector data beyond queues goes into `Observation.extras`, filled by
`SimSession.observe()` when the scenario has `detectors.control`. Keys (tuples indexed by
stage):

- `EXTRA_GAP_S` (`"gap_s"`): minimum time since detection over the stage's control loops,
  `inf` for a stage without loops;
- `EXTRA_CALLS` (`"calls"`): a vehicle hit one of the stage's loops since its green last
  ended.

Control loops follow SUMO's actuated placement: one E1 per incoming vehicle lane with a
TLS link, `min(detector_gap * v, (min_green / passing_time + 0.5) * 7.5)` before the stop
line. A loop serves a stage when every link of its lane is green there, at least one with
priority (`G`). `ActuatedController` (S1) extends a green while `gap_s` is under
`actuated.max_gap_s`, between `actuated.min_green_s` and `actuated.max_green_s` (by stage
name), and serves the stages in cycle order; `tests/test_actuated_native.py` checks it
against SUMO's native actuated logic, decision by decision.

`control/registry.py` maps names to classes. `make_controller(cfg)` reads `controller:`
and imports every module of `traffic_twin.control` on first lookup, so a new controller
needs no other wiring. Adding the Part 4 rule-based controller:

1. Create `src/traffic_twin/control/rules.py`:

   ```python
   from traffic_twin.config import get
   from traffic_twin.control.base import Controller, Observation
   from traffic_twin.control.registry import register


   @register("rules")
   class RuleController(Controller):
       def __init__(self, max_green_s: list[float]): ...

       @classmethod
       def from_config(cls, cfg):
           return cls(get(cfg, "rules.max_green_s"))

       def reset(self, seed: int) -> None: ...
       def decide(self, obs: Observation) -> int: ...
   ```

2. Add a scenario config (e.g. `configs/pitie_rules.yaml`) with `extends: pitie_fixed.yaml`,
   `scenario: pitie_rules`, `controller: rules` and a `rules:` section. Do not touch
   `demand`: S0 and the new scenario must share one `demand_hash`.
3. `python tasks.py compare --scenario configs/pitie_rules.yaml`, then
   `python -m traffic_twin.viz.plots --scenario pitie_fixed pitie_rules --out docs/figures/pitie`.

Keep heavy imports (torch, xgboost) inside functions: the registry imports every module
in the package.

## Safety guard and signal audit

`SafetyGuard.apply(proposed, state)` returns the stage to request instead of `proposed`.
Part 3 rules, in order (D13):

1. **min green**: no switch before the green has lasted `safety.min_green_s`;
2. **anti-starvation**: once a stage has been red for `safety.max_red_s`, head for it;
3. **stage order**: a switch goes to a link-safe stage; with `safety.stage_order: strict`
   (the default) that is the next stage in cycle order;
4. **max green**: a green ends after `safety.max_green_s`.

`SignalPlan` still runs every clearance phase. A rule that fires counts one intervention
per green, however many steps it acts on. The grid has no caps and no `max_red_s`. Part 4
adds pedestrian forcing and emergency pre-emption.

`sim/audit.audit_phases` replays the phases shown and counts: a vehicle link from green to
red without yellow, a pedestrian link from green to red into a phase that is not a
clearance phase of the stage being left, yellow to green, a complete green shorter than
the min green, and a green served out of order. A safe run has every count at 0.

`sim/tls_check` complements both statically: it checks the program itself (foe greens,
intergreens with the fixed greens and with the min greens, worst-case red of each crossing
with the fixed greens). `tests/test_actuated.py` checks the same crossing bound with the
actuated max greens.

## Validation and comparison

- **Settings** come from `configs/validation.yaml`, pre-registered before any calibrated
  result: a threshold changes only with a dated note.
- **Real side** (`validation.reference`): typical-day flows per station and slot from
  `data/reference/pitie_302_hours.csv`, with the same `data.calendar` filter as the demand
  calibration, applied per slot (so a slot's days can differ from the calibration's
  `day_filter.days_used`). Fewer than `min_days` days gives NO_DATA.
- **Field side** (`validation.field`): `data/field/pitie_measurements.yaml`. A missing file
  or a null value is NOT_MEASURED, never an error; schema problems raise one `ConfigError`
  listing all of them. Model inputs that do not come from the field are ASSUMED and listed
  from `network.assumptions` and `demand.assumptions`.
- **Tables**: `flows` (station loops from `detector.parquet`, A-02 PASS / FAIL on inbound
  arms and A-02 slots, ok / off on information rows, GEH share over distinct station
  groups), `signal` (cycle and per-head greens from `signal.parquet`, yellow and all-red
  from the program; PASS / FAIL for S0 only), `queues` (`queue.parquet` around each observed
  instant, capped to the distance the camera sees).
- **Report** (`validation.report`) writes the CSVs, `summary.json` and `docs/validation.md`;
  `--run-missing` runs the absent evaluation seeds first.
- **Calibration loop** (`validation.calibration_loop`) runs S0 on the calibration seeds in
  a temporary directory with `demand.arm_scale` changed in memory, and never edits a
  config: the recommended `arm_scale` is copied by hand into `configs/pitie/demand.yaml`.
  An arm whose flow stops following its demand is frozen and reported as capacity-limited.
- **Comparison** (`evaluation.compare`) loads both scenarios' runs for the evaluation seeds,
  refuses unless the seed sets match and all runs share one `demand_hash`, and tests the
  paired per-seed differences (`evaluation.stats`); Holm applies over the headline KPIs of
  `compare.headline` only, and `count` KPIs are not tested.

## Part 3 decisions

Fixed in Phase 0 of Part 3 (spec 17); the code and configs refer to them by number. Values
in brackets are the implemented ones.

| ID | Decision |
| --- | --- |
| D1 | Site: node 302, Bd de l'Hôpital x Bd Saint-Marcel. General legs N, S, W; bus-only legs NB, WB. Junction and TLS id `C`. The hospital gate on the NB bus road is not a leg. |
| D2 | Grid frozen: grid files are never edited; `tests/test_grid_regression.py` compares the grid KPIs of seeds 1-3 with a fixture, exact on the fixture's platform, 5 % elsewhere. |
| D3 | The published `t_1h` (cleaned `ts`) is the END of the hourly period: `hour_start = (ts - 1 h).hour`. `sensor_hourly.ts` stays as published. |
| D4 | Typical days: Tue-Thu, zone C school term, no public holiday, then a per-arc robust MAD filter (k = 3.5). One implementation (`data/calendar.py`), one config (`configs/data.yaml: calendar`). |
| D5 | Window 07:30-09:30 (`begin_s` 27000, 7200 s), no warm-up. Demand piecewise constant per sensor hour in 900 s slots; the real side of validation uses the same uniform-within-hour rule. |
| D6 | Reference arcs: N in 508; 510 (copy of 505) sensitivity only; S in 511 (509 = 511 = 5071, counted once); W in 513 (one station copied along about 700 m); outbound 509, 512, 514 information only. |
| D7 | Turn shares, ASSUMED until the video count: N to S 0.63, W 0.37; S to N 0.90, W 0.10; W to N 0.85, S 0.15. |
| D8 | Flow levels are recomputed by `data.calibrate`, never typed in [07:30-09:30 means N 626, S 370, W 206 veh/h]. |
| D9 | A-02: each inbound arm within ±15 % of the real flow in 07:30-08:30 and 08:30-09:30 (sim = mean of the 10 evaluation seeds, E1 loops). GEH < 5 on at least 85 % of distinct counts is reported. Control slots 08:00-09:00 and 07:30-09:30. |
| D10 | Signal: cycle ±2 s, green ±2 s, yellow and all-red ±0.5 s, pass/fail for S0 only. Queue: PASS inside the simulated p10-p90 or within 2 vehicles of the median, sample ±300 s around the observation over the seeds. |
| D11 | Seeds: evaluation 1-10, calibration 101-103. S1 parameters are derived analytically, never tuned on seeds 1-10. |
| D12 | S1 = the Python `ActuatedController` emulating SUMO's `type="actuated"` (max-gap 3, detector-gap 2, passing-time 1.9); native SUMO actuated is only a test oracle. Max greens keep the worst-case crossing red at or under `ped_max_wait_s - ped_red_margin_s` = 88 s [HOP 38, STM 22, others 6]. |
| D13 | SafetyGuard: min green, strict stage order, max-green cap, anti-starvation. Always on; 0 interventions for fixed and actuated. Part 4 adds pedestrian forcing and pre-emption. |
| D14 | The 90 s pedestrian target is judged on `ped_wait_single_max` (max over time of `person.getWaitingTime`); the walk time-loss KPIs are kept. |
| D15 | Provisional network and program, ASSUMED: no N crossing, `s_in` lane 1 shared by buses and straight cars, 3 m sidewalks, bus 57N only in S_LEFT, program from the lead's sketch v7 [adapted: cycle 123 s, HOP 36 s, STM 20 s]. |
| D16 | Statistics: exact Wilcoxon signed-rank (midranks, zero differences dropped), paired t CI, Hodges-Lehmann, Holm over the 7 headline KPIs only; count KPIs untested. Equity flag: a KPI more than 5 % worse. |
| D17 | `eclipse-sumo`, `libsumo`, `traci`, `sumolib` pinned to 1.28.0. |
| D18 | Missing or null field value: NOT_MEASURED. Model input not from the field: ASSUMED. Validation never fails because field data is missing. |

## Invariants

- **Reproducible runs.** Same config and seed give identical KPI values (only `run_id`,
  timestamps and wall time differ).
- **Frozen grid.** `configs/scenario_base.yaml`, `sumo/network/plain/`,
  `sumo/network/intersection.net.xml`, `sumo/gui/view.xml` and the grid figures are never
  edited, and the grid demand files are byte-identical (sha256 test). A refactor of `sim/`
  or `control/` must keep `tests/test_grid_regression.py` green; new KPI rows may be added,
  existing ones must not change.
- **Same demand everywhere.** Flows use per-second probabilities (cars, pedestrians) or a
  period (buses), so every scenario sees the same expected demand and the seed only changes
  the draw (`tests/test_demand.py`). Compared scenarios must share one `demand_hash` and the
  same seeds; `evaluation.compare` refuses otherwise.
- **Route file sorted by departure.** Every top-level element of a calibrated route file
  is sorted by departure time: SUMO ignores out-of-order elements (the ambulance would be
  dropped).
- **Clearance is never skipped.** `SignalPlan.request` always queues the current stage's
  clearance phases (pedestrian clearance, yellow, all-red) before a new green, and
  ignores further requests while a transition runs.
- **Minimum green.** A switch requested before the green has lasted
  `safety.min_green_s` is held; the controller asks again on a later step. The fixed
  green times must be at least that value (`tests/test_signal.py`).
- **Stage order is enforced by the guard.** With `safety.stage_order: strict` the
  `SafetyGuard` only lets the next stage in cycle order start: each LEFT stage's clearance
  is what ends that axis's permissive left turns. `SignalPlan` itself does not check order.
- **The guard is always on and silent for the baselines.** Every request goes through
  `SafetyGuard.apply`; the fixed and actuated controllers cause 0 interventions.
- **Signal audit = 0 on every run.** Every run's `safety.audit` counts are 0.
- **Detectors are passive.** Control loops, station loops, queue detectors and switch
  logging never change what is simulated (`tests/test_detectors_passive.py`).
- **Missing field data is NOT_MEASURED,** never an error; model inputs not measured on
  site are ASSUMED and listed in `docs/validation.md`.
- **Versioned inputs only.** The calibration file, the validation and every test read
  versioned files (`data/reference/`, `configs/calibration/`, `tests/data/`), never
  `data/processed/`; only `calib-data` (`data.calibrate --from-raw`) reads `data/raw/`. The
  calibration file and `docs/calibration_log.yaml` are deterministic (no wall-clock time).
- **No partial results.** A run either writes all its result files (four, plus three with a
  `measurement` section) or none.
