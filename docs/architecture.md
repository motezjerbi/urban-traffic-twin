# Architecture

Developer guide to `src/traffic_twin/`. The README covers setup and commands; the spec
(`docs/cahier_des_charges.docx`) covers scope. All CLIs run from the repo root: the paths
in `paths.py` and in the configs are relative to it.

Two networks share the code: the Part 2 **grid** (`configs/scenario_base.yaml`, frozen)
and the **Pitié-Salpêtrière junction**: `configs/pitie_fixed.yaml` = S0 and
`configs/pitie_actuated.yaml` = S1 (Part 3), `configs/pitie_rules.yaml` = S2 with its
ablation and the ambulance-probe configs (Part 4). Decision numbers refer to
[Part 3 decisions](#part-3-decisions) (D1-D18) and [Part 4 decisions](#part-4-decisions)
(D19-D36).

## Module map

| Module | Role |
| --- | --- |
| `paths.py` | Default file locations (configs, SUMO network, demand dir, processed file names, validation config, field file, reference dir, S2 config, evaluation config and output dir) |
| `config.py` | `load_config` / `load_scenario` with `extends`, required dotted keys, `get(cfg, "a.b")`, `ConfigError`; `config_sources`, `approach_in_edges` / `approach_out_edges`, `config_hash`, `demand_hash` |
| `data/download`, `data/clean` | Paris sensor API to `data/raw/`, cleaned hourly rows to `data/processed/` (`ts` = published end-of-period stamp; profiles on `hour_start`) |
| `data/explore`, `data/intersection_score` | Coverage table, hourly profile figures, candidate ranking |
| `data/calendar` | One typical-day filter for calibration, validation and the profile figures: period start from the stamp (D3), Tue-Thu school term without public holidays inside `calendar.typical_period` (other such days are hold-out days), robust MAD filter, uniform weights inside each sensor hour (D4, D5). The filter is shared, the day set is not: completeness and MAD are judged per window, so a validation slot can keep other days than the calibration's 07:30-09:30 window |
| `data/calibrate` | Versioned reference extract `data/reference/pitie_302_hours.csv` (from `data/raw/`) and the calibration file `configs/calibration/pitie_302.yaml` (slot flows per arm, turn shares) |
| `db/load_sensors`, `db/load_run` | Upserts into Supabase; state logs uploaded to the `sim-logs` bucket |
| `sim/osm_import` | OSM extract to curated plain XML (`sumo/network/pitie/pitie.{nod,edg,con}.xml`): semantic edge ids, explicit connections and crossings |
| `sim/network` | netconvert: the grid from `sumo/network/plain/` (legacy build), or the scenario's `network.build` (plain XML + `.tll.xml` + options) |
| `sim/tls_check` | Static checks of a signal program: foe greens, intergreens from the geometry, worst-case red of each crossing; CLI exits 1 on errors. For `controller: rules` it reports the worst-case red at max greens plus the bus extension and requires the pedestrian shield when it exceeds θ; `cached_requirements` feeds the audit |
| `sim/demand` | Scenario config to `sumo/demand/<scenario>.rou.xml` and `<scenario>.stops.add.xml`: legacy grid schema, or calibrated slot flows, bus lines and pinned walks. With `demand.bus.counted_by_sensors` an arm's car demand is its sensor flow minus its scheduled buses (sensors and E1 stations both count every vehicle). `emergency.trips` (ambulance probe) writes `ambulance_0..k` sorted by departure; without it the single-ambulance output is byte-identical |
| `sim/detectors` | Passive detectors of a run: control loops (actuated), station loops (validation), E2 queue detectors, signal switch times; parsers for their outputs |
| `sim/sumo` | Finds the binaries, picks libsumo (headless) or TraCI (GUI), builds the command line |
| `sim/signal` | `SignalPlan`: stage requests to SUMO phases, clearance and min green; `SignalState` snapshot for the guard (with the shield's pedestrian waits and the preemption target); `link_stages` |
| `sim/session` | `SimSession`: one SUMO run driven step by step, guard included; on a junction with `network.crossings` it also samples pedestrian waits, runs the emergency and bus detectors, the preemption manager and the junction monitor |
| `sim/emergency` | `EmergencyDetector`: vClass `emergency` vehicles on an approach (`vehicle.getNextTLS`) or still pending insertion, as `EmergencyCall`s (link, target stage, distance, first detection) |
| `sim/transit` | `BusDetector`: buses on an approach before the stop line, as `BusCall`s (line, link, stages, distance, speed, remaining dwell) |
| `sim/junction` | `edge_links`, `lane_links` (TLS link maps) and `JunctionMonitor`: box, exit-lane and crossing observer (blocked exits for box clearing, junction and crossing KPIs); read only |
| `sim/audit` | Audit of the phases shown during a run (A-04 evidence): danger keys and information keys |
| `sim/results` | SUMO outputs to `kpi_run.csv`, `trip.parquet`, `state_log.parquet`, `run.json`, and the measurement tables |
| `sim/kpi` | KPIs per user type in the spec 9.4 long format (`compute_kpis`, unchanged since Part 3) plus the Part 4 rows of `extra_kpis` (KPI version 2) and the curb waits (KPI version 3; `KPI_VERSION` = 3); SUMO statistics for the health block |
| `sim/run_scenario` | CLI: controller loop over seeds, writes results, prints a summary and a health line |
| `control/base` | `Observation`, `Controller` (with `report()` and events), the `EXTRA_*` keys, `EmergencyCall`, `BusCall`, `Event` |
| `control/registry` | `@register(name)`, `make_controller(cfg)` |
| `control/fixed` | Fixed-time baseline (`controller: fixed`) |
| `control/actuated` | S1: SUMO-style gap-based actuated control (`controller: actuated`) |
| `control/rules` | S2: gap logic with relaxed max greens, bus priority (TSP), box clearing, pedestrian anticipation (`controller: rules`) |
| `control/safety` | `SafetyGuard` between every controller and `SignalPlan` (D20 rule order, preemption, shield); `safe_transitions` |
| `control/pedestrian` | `PedestrianShield`: time until each crossing's next green and the stay / jump checks against θ (D21) |
| `control/preemption` | `next_hop` (cheapest link-safe path to a target stage) and `PreemptionManager` (first come first served requests, served / time-out bookkeeping); no TraCI |
| `validation/__init__` | Pre-registered settings (`configs/validation.yaml`), clock helpers |
| `validation/reference` | Real flows per station and slot from the versioned extract (typical days) |
| `validation/field` | Field file loading, schema checks, inventory of what is measured |
| `validation/flows`, `validation/signal`, `validation/queues` | Flow (A-02, GEH), signal timing and queue tables |
| `validation/calibration_loop` | Simulation-in-the-loop demand scaling on the calibration seeds; `docs/calibration_log.yaml` |
| `validation/report` | Validation CSVs, `summary.json` and `docs/validation.md` |
| `evaluation/stats` | Exact Wilcoxon signed-rank, paired t CI, Hodges-Lehmann, Holm, GEH (numpy only) |
| `evaluation/runs` | Loads the runs of a scenario and checks they are current (`config_hash`, `demand_hash`, `kpi_version`); runs the missing or stale seeds |
| `evaluation/compare` | Paired comparison of two scenarios over the evaluation seeds, equity flags; a qualified KPI takes the direction and unit of its base name |
| `evaluation/evaluate` | Pre-registered N-scenario evaluation (`configs/evaluation.yaml`): pairing, person-weighted delay, pairs, equity, A-03 to A-05, `docs/evaluation.md` |
| `viz/plots`, `viz/style` | KPI figures from `data/results/`, shared figure style |
| `viz/validation` | Validation flow figure, queue figure, paired differences figure |
| `viz/evaluation` | The five Part 4 evaluation figures (`docs/figures/pitie/eval/`) |

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

```
Part 4 (S2, Safety Guard, per-user evaluation)
  configs/evaluation.yaml (pre-registered: families, pairs, A-04 / A-05 / equity rules)
     main family:            S0 pitie_fixed, S1 pitie_actuated, S2 pitie_rules,
                             S2_noP pitie_rules_nopreempt                (one demand_hash)
     ambulance_probe family: S0 pitie_amb_fixed, S2 pitie_amb_rules,
                             S2_noP pitie_amb_rules_nopreempt            (one demand_hash)
  runs (seeds 1-10; run_missing reruns absent, stale or kpi_version < 3 runs)
    SimSession.observe: pedestrian waits, EmergencyDetector -> PreemptionManager (target),
                        BusDetector, JunctionMonitor (blocked exits)
    Controller.decide --> SafetyGuard.apply (shield, preemption target) --> SignalPlan --> SUMO
       v
    data/results/<scenario>_seed<n>/ {..., run.json (kpi_version, safety, preemption,
                                      control), events.parquet}
  evaluation.evaluate --> data/results/evaluation/<family>/{kpis.csv, pairs/, equity.csv, a04.csv}
                          data/results/evaluation/{a05.json, summary.json, provenance.json}
                          docs/evaluation.md
  viz.evaluation --> docs/figures/pitie/eval/{kpi_by_user, paired_vs_s0, equity_heatmap,
                                              a05_ambulance, ped_wait_by_crossing}.png
```

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
| `configs/pitie/rules.yaml` | S2 fragment: `rules` (gap logic, max greens, TSP, box, pedestrian anticipation), `preemption`, `safety.max_red_s` 180; with the derivation of the S2 crossing reds and cycle |
| `configs/pitie_rules.yaml` | S2: `extends: [pitie_fixed.yaml, pitie/rules.yaml]`, controller `rules` |
| `configs/pitie_rules_nopreempt.yaml` | S2_noP: S2 with `preemption.enabled: false` |
| `configs/pitie/emergency_probe.yaml` | `emergency.trips`: 7 ambulances (N to S and W to S) |
| `configs/pitie_amb_{fixed,rules,rules_nopreempt}.yaml` | Ambulance-probe family: S0, S2 and S2_noP plus `pitie/emergency_probe.yaml` |
| `configs/validation.yaml` | Not a scenario: pre-registered validation settings (reference, seeds, slots, stations, thresholds, calibration loop, headline KPIs, field file) |
| `configs/evaluation.yaml` | Not a scenario: the pre-registered Part 4 evaluation (seeds, families and pairs, A-04 / A-05 / equity rules, occupancies, outputs); it changes only through dated `changes` entries |

`approach_in_edges` / `approach_out_edges` accept the grid form (`{W: w_in}`) and the
Pitié form (`{W: {in_edges: [...], out_edges: [...]}}`).

**Hashes.** `config_hash` is the sha256 of the merged config as canonical JSON, without
`outputs`. `demand_hash` covers what the demand depends on: `begin_s`, `duration_s`,
`step_length_s`, `demand`, `emergency`, `network.{net_file, approaches, crossings,
bus_stops}`, plus the sha256 of the calibration file and of the net file. S0, S1, S2 and
S2_noP share it by construction (the S2 fragment never touches `demand`, `emergency` or
`network`); the ambulance-probe configs change `emergency` and share their own. A
comparison refuses runs whose `demand_hash` differs. Adding `safety.ped_forcing` in Part 4
changed the `config_hash` of S0 and S1, so their Part 3 runs were rerun once: the 35 Part 3
KPI rows of every run are identical.

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
| `run.json` | always | Run id, seed, code version, merged config, `config_hash`, `demand_hash`, `config_sources`, `health`, `safety`, `kpi_version`, `preemption`, `control` |
| `detector.parquet` | `measurement.stations` | `run_id, det_id, station, lane, begin_s, end_s, n_veh, occupancy_pct, mean_speed` per station loop and period |
| `queue.parquet` | `measurement.queues` | `run_id, t, approach, jam_veh, jam_m, halting_veh` per sample |
| `signal.parquet` | `measurement.tls_switch` | `run_id, link, begin_s, end_s, duration_s`: green intervals per TLS link |
| `events.parquet` | any event | `run_id, t, source, rule, stage_from, stage_to, detail`: controller rules, guard interventions and `ped_yield`, preemption request / served / timeout |

Times are relative to the scenario start. SUMO's raw outputs stay next to them:
`tripinfo.xml`, `stops.xml`, `statistics.xml`, and with detectors `detectors.add.xml`,
`e1.xml`, `tls_switch.xml`. The grid has no `measurement` section, so it writes only the
first four files.

- `health`: `teleports`, `collisions`, `emergency_stops`, `emergency_braking`,
  `vehicles_waiting_end` (from `statistics.xml`), `depart_delay_mean`, `depart_delay_max`,
  `drain_timeout`, plus the person counts of `statistics.xml`.
- `safety`: `interventions` per guard rule (`min_green`, `order`, `max_green`,
  `starvation`, `ped_wait`), `audit` (`sim.audit.AUDIT_KEYS`), `danger` (the sum of
  `DANGER_KEYS`) and `ped_forcing` (the shield mode, `off` without a shield).
- `preemption`: `enabled`, the manager's `requests`, `served`, `timeouts`, `active_s` and
  the guard's `hold_s`, `jumps`, `ped_yield`; zeros when disabled.
- `control`: `controller.report()`; for `rules`: `ped_protect`, `box_gapouts` and a `tsp`
  block (bus calls, extended greens, extension seconds, per line, truncations).
- `kpi_version`: 3 since the curb-wait fix of Part 4 (2 before it). `evaluation.runs`
  counts a run without it as version 1, and the evaluation treats a run below
  `configs/evaluation.yaml` `kpi_version` (3) as stale.
- KPI rows added in Part 3 (existing rows unchanged): `depart_delay_mean` and
  `depart_delay_max` per vehicle type; bus `dwell_mean`; pedestrian `ped_wait_p95`,
  `ped_wait_single_mean`, `ped_wait_single_p95`, `ped_wait_single_max`. Bus `headway_std`
  is the mean over (bus stop, line) groups with at least 3 arrivals, which equals the old
  value for one line and one stop.
- KPI rows added in Part 4 (KPI version 2, `kpi.extra_kpis`, appended after the unchanged
  `compute_kpis` rows): car `throughput_vph`, `delay_total_mean` (time loss + insertion
  delay) and per arm (`__arm_N`); bus `delay_total_mean`, `delay_mean__flow_*`,
  `headway_std__line_*`; ambulance `travel_time_total` (arrival - planned departure),
  `stop_duration_mean`; pedestrian `crossing_complete_rate`, `crossing_red_entries`,
  `ped_wait_single_{mean,max}__crossing_*`; junction `box_halt_red_s`, `box_foe_onsets`,
  `exit_blocked_s`, `exit_blocked_green_s` (from `JunctionMonitor.summary()`). A qualified
  KPI is named `<base>__<qualifier>`.
- KPI rows added by KPI version 3 (Pitié only, appended after the version 2 rows):
  pedestrian `ped_curb_wait_mean`, `ped_curb_wait_p95`, `ped_curb_wait_max` and
  `ped_curb_wait_{mean,max}__crossing_*`, from `SimSession.ped_curb_wait_max` (the longest
  curb wait per person). SUMO resets `person.getWaitingTime` whenever a waiting pedestrian
  creeps faster than 0.1 m/s, so the single wait (`ped_wait_single_*`, Part 3, unchanged)
  undercounts the wait at a red curb; the curb wait keeps a per-person clock through such
  creeps and is never below the single wait. A-04 judges `ped_curb_wait_max`. The person-weighted delay is computed by the
  evaluation from `trip.parquet`, never written to `kpi_run.csv`. Existing rows never
  change (the grid regression and the Part 3 rerun check it).

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
  `extras` when control loops exist). It is computed once per step. With
  `network.crossings` (Pitié) the extras also carry the Part 4 keys: the pedestrian waits
  and counts per crossing sampled after the previous step, the shield slack, the emergency
  calls and the preemption target (`PreemptionManager.update`, called here once per step),
  the bus calls, the blocked exits and `gap_s_open`. On the grid the extras are exactly
  those of Part 3.
- `step(stage)` builds the guard state `plan.state(t)` with the shield's pedestrian waits
  (`ped_wait`: detected waits, or seconds since green in `red` mode) and the preemption
  target, passes the request through `SafetyGuard.apply(stage, state)`, then to
  `SignalPlan`, sets the resulting phase and advances SUMO by `step_length_s`. It records
  the phase shown (`phases_shown`, audited after the run), each pedestrian's longest single
  wait (`person.getWaitingTime`) and longest curb wait (a clock that starts when the
  person waits at a crossing that is not green and keeps running through creeps until the
  person's next edge changes or the crossing turns green), the per-crossing curb waits
  (the shield's w_c), the junction monitor sample and,
  on state-log ticks, the E2 queue detectors. It returns the phase shown.
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

On a junction with `network.crossings` (Pitié, Part 4) `observe()` adds: `gap_s_open`
(like `gap_s`, over the loops whose lane still has an exit that is not blocked),
`ped_wait` and `ped_count` per crossing (in `network.crossings` order), `ped_slack` (θ
minus the shield's wait plus time to green, per crossing), `emergency` (`EmergencyCall`s),
`preempt` (the active preemption target or None), `bus_calls` (`BusCall`s) and
`exit_blocked` (exit lanes). A controller may read them; only the guard acts on the
pedestrian bound and on preemption.

Explainability: `Controller.report()` goes to run.json `control`, and
`Controller.log_event(t, rule, stage_from, stage_to, detail)` appends an `Event` that ends
in `events.parquet` with the guard's and the preemption manager's events. Fixed and
actuated keep the empty defaults.

Control loops follow SUMO's actuated placement: one E1 per incoming vehicle lane with a
TLS link, `min(detector_gap * v, (min_green / passing_time + 0.5) * 7.5)` before the stop
line. A loop serves a stage when every link of its lane is green there, at least one with
priority (`G`). `ActuatedController` (S1) extends a green while `gap_s` is under
`actuated.max_gap_s`, between `actuated.min_green_s` and `actuated.max_green_s` (by stage
name), and serves the stages in cycle order; `tests/test_actuated_native.py` checks it
against SUMO's native actuated logic, decision by decision.

`RuleBasedController` (S2, `controller: rules`) decides in this order: hold during a
transition or before its min green; the gap logic over `gap_s_open` when box clearing is
on (`box_gapout` when only the box rule ends the green); paid TSP truncation (off at
Pitié); TSP extension of a bus-serving stage about to end, by up to
`rules.tsp.extension_max_s`, while a called bus can reach the stop line in time
(`tsp_extend`); pedestrian anticipation, leaving the green as soon as `min(ped_slack)` < 1
step (`ped_protect`); then stay or move to the cycle successor. With S1's max greens and
every rule off it takes exactly the decisions of `ActuatedController`
(`tests/test_rules.py`).

`control/registry.py` maps names to classes. `make_controller(cfg)` reads `controller:`
and imports every module of `traffic_twin.control` on first lookup, so a new controller
needs no other wiring. Adding a controller (as `rules` was added in Part 4):

1. Create `src/traffic_twin/control/<name>.py`:

   ```python
   from traffic_twin.config import get
   from traffic_twin.control.base import Controller, Observation
   from traffic_twin.control.registry import register


   @register("my_controller")
   class MyController(Controller):
       def __init__(self, max_green_s: list[float]): ...

       @classmethod
       def from_config(cls, cfg):
           return cls(get(cfg, "my_controller.max_green_s"))

       def reset(self, seed: int) -> None:
           self.clear_events()

       def decide(self, obs: Observation) -> int: ...
   ```

2. Add a scenario config with `extends: pitie_fixed.yaml` (or a fragment list, as
   `configs/pitie_rules.yaml`), its `scenario` name, `controller: my_controller` and its
   section. Do not touch `demand`, `emergency` or `network`: S0 and the new scenario must
   share one `demand_hash`. `SafetyGuard.from_config` checks its greens against the caps
   and the pedestrian bound.
3. `python tasks.py compare --scenario configs/<new>.yaml` for a pairwise comparison with
   S0. A scenario enters the pre-registered evaluation (`configs/evaluation.yaml`) only
   through a dated entry in its `changes`.

Keep heavy imports (torch, xgboost) inside functions: the registry imports every module
in the package.

## Safety guard and signal audit

`SafetyGuard.apply(proposed, state)` returns the stage to request instead of `proposed`.
Rule order (D20; R4 holds the Part 3 rules of D13):

- **R0** an unknown stage or preemption target raises `ValueError`;
- **R1** during a transition the current stage is returned: `SignalPlan` runs every
  clearance phase in full;
- **R2 min green**: no switch before the green has lasted `safety.min_green_s`; hard, also
  for preemption (counted as an intervention only without preemption);
- **R3 preemption**, when `state.preempt_target` is set: the first hop on the cheapest
  link-safe path to the target (`control.preemption.next_hop`, cost clearance + min green
  per hop, also under strict order), held once there. R4 is suspended. A hop that the
  shield refuses becomes the cycle successor (`ped_yield`);
- **R4 Part 3 rules**: anti-starvation (once a stage has been red for `safety.max_red_s`,
  head for it), stage order (a link-safe stage; with `safety.stage_order: strict`, the
  default, the next stage in cycle order), max green (`safety.max_green_s`);
- **R5 pedestrian shield** (`control/pedestrian.py`, D21), hard and last: a stay, or a
  jump past the successor, that would let a crossing's wait exceed θ becomes the
  successor (`ped_wait`).

**Shield.** θ = `safety.ped_max_wait_s` - `safety.ped_red_margin_s` (Pitié: 90 - 2 = 88 s).
For each crossing c, w_c is the longest current curb wait (creeps at the curb ignored) of
a person whose next edge is the crossing (`safety.ped_forcing: detected`, ideal
detection, ASSUMED; `red` uses the time
since the crossing last showed green), and T_c the time until c shows green if the
current stage ends its min green now and every following stage runs at min green with full
clearance, in cycle order. A green may last one more step only if w_c + T_c + dt <= θ for
every crossing; a jump j only if w_c + T_via(c, j) <= θ. The successor is always allowed,
since the worst red of every crossing at min greens is at most θ (`from_config` checks
it). `SafetyGuard.from_config` also refuses a plan whose worst-case crossing red exceeds θ
without a shield, a `max_red_s` not above the longest legal cycle of the largest plan
(S2: 177 s, so 180 s), and `preemption.respects_ped_max_wait: false`.

**Preemption flow.** `EmergencyDetector.calls(t)` lists every `emergency` vehicle on an
approach edge before the stop line, and with `preemption.detection.pending` every one
still waiting for insertion (`dist_m` inf). `PreemptionManager.update(t, calls)` keeps the
requests first come first served on (first detection, id), marks a request served when its
vehicle is no longer called (it passed the stop line), and drops it after `max_hold_s`
(120 s) without the vehicle getting `progress_m` (1 m) closer to the stop line; a dropped
vehicle is never requested again. The target goes to the guard through
`SignalState.preempt_target`. The time-out clock was amended at the Part 4 gate (dated
change of 2026-10-12 in `configs/evaluation.yaml`): a clock from the first detection
alone dropped a moving ambulance whose hold the shield had cut (see the
`control/preemption.py` docstring).

`SignalPlan` still runs every clearance phase. A rule that fires counts one intervention
per green, however many steps it acts on, and logs one guard event. The grid has no caps,
no `max_red_s`, no shield and no preemption, so its guard behaves as in Part 3.

`sim/audit.audit_phases` replays the phases shown. Danger keys (`DANGER_KEYS`, every count
0 in a safe run): a vehicle link from green to red without yellow, a pedestrian link from
green to red into a phase that is not a clearance phase of the stage being left, yellow to
green, a complete green shorter than the min green, `clearance_incomplete` (the phases
between two greens are not exactly the leaving stage's clearance phases at their
programmed durations), `intergreen_short` (from `tls_check.intergreen_requirements`),
`foe_vehicle_greens`, `illegal_transition` (not in the guard's link-safe graph) and
`unknown_phase`. Information key (`INFO_KEYS`): `out_of_order_green`, since preemption may
skip a stage. The keys that need the program's durations, intergreens or foes are passed
only for Pitié; on the grid they stay 0.

`sim/tls_check` complements both statically: it checks the program itself (foe greens,
intergreens with the fixed greens and with the min greens, worst-case red of each crossing
with the fixed greens). `tests/test_actuated.py` checks the same crossing bound with the
actuated max greens. For `controller: rules` it reports the worst-case red at the S2 max
greens plus the TSP extension (S 120 s, W 126 s on the ASSUMED program) and accepts it
only with the shield on.

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
- **Evaluation** (`evaluation.evaluate`, Part 4) reads `configs/evaluation.yaml`. Per
  family it runs the missing or stale seeds (`--run-missing`; a run with `kpi_version` < 3
  is stale), checks the pairing (same seeds, current hashes, one `demand_hash`, the same
  (user, KPI) set in every scenario, which is A-03), adds the person-weighted delay per run
  from `trip.parquet` (occupancies of the config, plus the bus sensitivity values), calls
  `compare()` once per pre-registered pair (a Holm family per pair), and builds the equity
  table with the 5 % flags and the hidden-degradation flag. A-04 is checked over every run
  of both families (exit 1 on failure), A-05 is the pre-registered one-sided exact
  Wilcoxon (a FAIL is reported, exit 0). It writes `data/results/evaluation/` and
  `docs/evaluation.md`; `viz.evaluation` draws the figures from those outputs.

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
| D13 | SafetyGuard: min green, strict stage order, max-green cap, anti-starvation. Always on; 0 interventions for fixed and actuated. Part 4 adds the pedestrian shield and preemption (D20-D22). |
| D14 | The 90 s pedestrian target is judged on `ped_wait_single_max` (max over time of `person.getWaitingTime`); the walk time-loss KPIs are kept. Amended in Part 4 (dated change of 2026-10-12): `getWaitingTime` resets when a waiting pedestrian creeps at the curb, so from KPI version 3 the target is judged on the curb wait `ped_curb_wait_max` (A-04, D27); the Part 3 KPI is kept unchanged and reported. |
| D15 | Provisional network and program, ASSUMED: no N crossing, `s_in` lane 1 shared by buses and straight cars, 3 m sidewalks, bus 57N only in S_LEFT, program from the lead's sketch v7 [adapted: cycle 123 s, HOP 36 s, STM 20 s]. |
| D16 | Statistics: exact Wilcoxon signed-rank (midranks, zero differences dropped), paired t CI, Hodges-Lehmann, Holm over the 7 headline KPIs only; count KPIs untested. Equity flag: a KPI more than 5 % worse. |
| D17 | `eclipse-sumo`, `libsumo`, `traci`, `sumolib` pinned to 1.28.0. |
| D18 | Missing or null field value: NOT_MEASURED. Model input not from the field: ASSUMED. Validation never fails because field data is missing. |

## Part 4 decisions

Fixed in Phase 0 of Part 4, before any S2 run on the evaluation seeds, with
`configs/evaluation.yaml` (pre-registered 2026-10-12). Values in brackets are the
implemented ones.

| ID | Decision |
| --- | --- |
| D19 | S2 = controller `rules` (`RuleBasedController`), `configs/pitie_rules.yaml` = `extends: [pitie_fixed.yaml, pitie/rules.yaml]`. The fragment sets only `rules`, `preemption` and `safety.max_red_s`, so S0, S1 and S2 share one `demand_hash`. Ablation S2_noP = `preemption.enabled: false`. |
| D20 | Guard rule order: R0 unknown stage, R1 in transition, R2 min green (hard, also for preemption), R3 preemption, R4 the Part 3 rules (suspended during preemption), R5 pedestrian shield (hard, last; also checks the preemption hops). |
| D21 | Pedestrian shield, θ = `ped_max_wait_s` - `ped_red_margin_s` [88 s]: a stay needs w_c + T_c + dt <= θ, a jump w_c + T_via <= θ, for every crossing; otherwise the successor (`ped_wait`). w_c = the longest current curb wait (creeps at the curb ignored) from TraCI, ideal detection, ASSUMED; `safety.ped_forcing: detected` for every Pitié controller, absent (off) on the grid. |
| D22 | S0 and S1 behave as in Part 3: 0 interventions on every key, `ped_wait` included. S1 sits exactly on the bound (66 + 21 + 1 = 88). |
| D23 | S2 parameters, tuned on seeds 101-103 only (development seeds 201-207 used once by the TSP analysis): max-gap 3 s, min greens 6 s, max greens [HOP 50, STM 30, others 6]; TSP extension up to 10 s on S_LEFT, BUS_91, BUS_57 for lines 91, 24, 57, ETA = distance / 7 m/s + remaining dwell + 4 s if halted or dwelling, from ideal bus detection (TraCI position, line and exact remaining scheduled dwell, standing for AVL or a check-in loop; ASSUMED); paid truncation off (penalty cap HOP 38, STM 22 kept configurable); box exit zone 15 m, halted below 0.5 m/s, read from TraCI (ideal exit-blockage detection, no modelled detector; ASSUMED); pedestrian anticipation on; `safety.max_red_s` 180 in S2 only (longest legal cycle 177 s). |
| D24 | Preemption: detection by `vehicle.getNextTLS` for vClass `emergency` plus pending vehicles (upstream / V2I detection, ASSUMED); first come first served; target = the stage showing `G` on the link, else `g`; Dijkstra over link-safe transitions (cost clearance + min green); held until the stop line; time-out 120 s. The ambulance obeys the signals: no bluelight device, no rescue lane (ASSUMED). Amended at the Part 4 gate (dated change of 2026-10-12): a request is dropped after 120 s without the ambulance getting closer to the stop line; the 120 s clock restarts when the ambulance gets 1 m closer to the stop line (`preemption.progress_m`), so a stationary ambulance still times out 120 s after its first detection and a moving one is not dropped. |
| D25 | Demand frozen: the main family keeps the Part 3 demand (1 ambulance N to S at 08:10, Part 3 bus headways, no jitter). The ambulance-probe family adds 7 trips (`configs/pitie/emergency_probe.yaml`; N to S and W to S) with its own `demand_hash`. Bus headway updates and jitter are left for a later sensitivity family. |
| D26 | A-05 primary: main family, S2 vs S0, ambulance `travel_time_total` (arrival - planned departure), exact one-sided Wilcoxon, α 0.05, n 10; PASS iff p < α. Hodges-Lehmann, CI, `travel_time`, `stops`, `stop_duration_mean` reported. Secondary (reported only): probe family S2 vs S2_noP and S2 vs S0 on per-seed means. |
| D27 | A-04 over every run of both families: every `DANGER_KEYS` count 0, 0 collisions, `ped_wait_single_max` <= 90 s, 0 interventions for fixed and actuated. `evaluate` exits 1 on a failure; S2 interventions are reported (expected 0). Amended (dated change of 2026-10-12, a correctness fix found on the calibration seeds before any regenerated seed 1-10 run): the pedestrian condition is `ped_curb_wait_max` <= the scenario's `safety.ped_max_wait_s` (90 s); `ped_wait_single_max` is reported alongside. |
| D28 | Statistics: a Holm family per pair over the 7 Part 3 headline KPIs; pairs S0-S1 (part3), S0-S2 and S1-S2 (confirmatory), S2_noP-S2 (exploratory); families not pooled; new KPIs secondary (unadjusted p). |
| D29 | Equity: change vs S0 of every KPI per user; flag above 5 % worse (rates: 1 point). Hidden degradation: `all.person_delay_mean` significantly better while a primary KPI (car `time_loss_mean`, bus `delay_mean`, pedestrian `ped_wait_single_max`, ambulance `travel_time_total`) or a subgroup KPI (per car arm, bus flow, crossing) is flagged. Amended (dated change of 2026-10-12, same curb-wait fix): the pedestrian primary is `ped_curb_wait_max` and the pedestrian subgroups are `ped_curb_wait_mean__crossing_*`. Occupancies ASSUMED: car 1.3, bus 25 (20 and 30 sensitivity), pedestrian 1, ambulance 1. |
| D30 | KPI version 2: rows appended by `extra_kpis`; `compute_kpis` unchanged. Person-weighted KPIs only in the evaluation. KPI version 3 (2026-10-12) appends the curb-wait rows `ped_curb_wait_*`; every earlier row is unchanged. |
| D31 | Audit: five new danger keys (`clearance_incomplete`, `intergreen_short`, `foe_vehicle_greens`, `illegal_transition`, `unknown_phase`); `out_of_order_green` becomes information. |
| D32 | Seeds: evaluation 1-10 (run once, by the Phase 3 integration only), calibration and tuning 101-103, development 201-207 (used once), RL training from 1000. |
| D33 | `configs/evaluation.yaml` pre-registered 2026-10-12; changes only as dated `changes` entries (6 so far, listed in `docs/evaluation.md`). |
| D34 | Outside preemption, starvation is bounded per stage (`safety.max_red_s`: once a stage has been red that long the guard heads for it); every approach is served by at least one stage, so this bounds each approach's red too. During preemption R4 is suspended (D20): only min green, clearance and the pedestrian shield stay hard, and a hold ends when the ambulance passes the stop line or after 120 s without 1 m of progress (D24), so a slowly moving ambulance or back-to-back ambulances can keep a stage red beyond `max_red_s`. The evaluation runs never triggered the starvation rule. |
| D35 | No all-red "clear junction" hold: `SignalPlan` cannot hold a clearance. Box clearing = SUMO keep-clear for drivers plus S2 ignoring loops that only feed blocked exits. |
| D36 | Provisional: everything rests on the ASSUMED 123 s program. Every Pitié number (hold caps, maxima, 177 s cycle, crossing reds) is derived from config and program in code and tests, never hard-coded in logic. `tasks.py part3 part4` reruns everything after the field visit. |

## Invariants

- **Reproducible runs.** Same config and seed give identical KPI values (only `run_id`,
  timestamps and wall time differ).
- **Published results name a commit.** A run records `code_version` (`<sha>-dirty` from an
  uncommitted tree), and `docs/validation.md` and `docs/evaluation.md` print it. Staleness
  (`evaluation.runs.stale_reason`) ignores the code version on purpose, so after a commit
  `tasks.py rebaseline` re-makes the evaluation runs from it and requires identical KPI
  rows; `tasks.py provenance` and `tests/test_docs.py` (on a clean tree) refuse a page that
  names an uncommitted tree.
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
- **The guard is always on and never overrides a controller of the study.** Every
  request goes through `SafetyGuard.apply`; fixed and actuated must cause 0 interventions
  (overrides of the controller's request) on every key, `ped_wait` included (D13, D22),
  and rules is expected to cause 0 (the controller leaves a green at the step where the
  shield would otherwise force it; the counts per run are in `docs/evaluation.md`). The
  guard is not idle, though: its own preemption hops and holds are shield-checked, and a
  refused hop or a cut hold is counted as `ped_yield`, not as an intervention.
- **Danger audit = 0 on every run.** Every `DANGER_KEYS` count of `safety.audit` is 0;
  `out_of_order_green` is information only (preemption may skip a stage).
- **Pedestrian bound.** On Pitié every pedestrian curb wait (creeps at the curb
  included, hence also every single wait) stays at or under θ = 88 s (`ped_max_wait_s` 90
  minus the 2 s margin), preemption included: the shield is hard and last in the guard,
  reads the curb wait, and the worst crossing red at min greens is at most θ.
- **Demand frozen per family.** S0, S1, S2 and S2_noP share one `demand_hash`, the
  ambulance-probe trio another; a controller config never touches `demand`, `emergency`
  or `network`.
- **Evaluation pre-registered.** `configs/evaluation.yaml` (seeds, pairs, Holm families,
  A-04 / A-05 / equity rules, occupancies) changes only through dated `changes` entries;
  S2 parameters are tuned on seeds 101-103 (the TSP analysis also used the development
  seeds 201-207 once, D23), and seeds 1-10 tune nothing.
- **KPI rows only grow.** A new KPI version appends rows; existing rows keep their values
  (the grid regression and the Part 3 rerun check it).
- **Detectors are passive.** Control loops, station loops, queue detectors and switch
  logging never change what is simulated (`tests/test_detectors_passive.py`).
- **Missing field data is NOT_MEASURED,** never an error; model inputs not measured on
  site are ASSUMED and listed in `docs/validation.md` (Part 3: network and demand) and in
  `docs/evaluation.md` and the README Part 4 section (Part 4: detection, ambulance
  behaviour, probe routes, occupancies).
- **Versioned inputs only.** The calibration file, the validation and every test read
  versioned files (`data/reference/`, `configs/calibration/`, `tests/data/`), never
  `data/processed/`; only `calib-data` (`data.calibrate --from-raw`) reads `data/raw/`. The
  calibration file and `docs/calibration_log.yaml` are deterministic (no wall-clock time).
- **No partial results.** A run either writes all its result files (four, plus three with a
  `measurement` section, plus `events.parquet` when any event was logged) or none.
