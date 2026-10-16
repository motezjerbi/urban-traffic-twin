# API v1 and mock screens (Part 5)

The API serves precomputed simulation runs of the Pitié-Salpêtrière junction to the five
user views of spec 12. It implements spec 9.2 (common message format) and every GET
endpoint of spec 9.3, plus `/health`. It is a FastAPI app (`src/traffic_twin/api/`) that
reads local Parquet, CSV and JSON files through DuckDB. It never talks to SUMO and never
imports the ETA model: ETA predictions are precomputed per run (D51).

**Everything it serves is simulated.** The signal program is ASSUMED until the field visit
(3-5 Nov 2026), so every response says `provisional: true`.

## Running it

```text
python tasks.py api        # uvicorn traffic_twin.api.main:app --no-access-log --host 127.0.0.1 --port 8000
```

- Settings: `configs/api.yaml`. Two environment variables override it:
  - `TT_API_CONFIG`: another settings file;
  - `TT_RESULTS_DIRS`: run folders to index, separated by `;` on Windows and `:` elsewhere. The evaluation CSVs are then read from `<first dir>/evaluation`.
- Interactive schema: `http://127.0.0.1:8000/docs`.
- Mock screens: `http://127.0.0.1:8000/mock` (also `/`), when `dev.mock: true`.
- Fake data for the mock screens, without any simulation:

  ```text
  python -m traffic_twin.api.fixtures --out data/fake
  TT_API_CONFIG=data/fake/api.yaml uvicorn traffic_twin.api.main:app --no-access-log
  ```

  This writes three synthetic runs (S0, S1, S2 labels, seed 4000), the main family's evaluation CSVs, FAKE ETA predictions (`model_version: fake-0`), a FAKE reference bundle (`fake-ref-0`) and the settings that point at them. It refuses to write under `data/results`.

## What it reads

| Source | Content |
|---|---|
| `results_dirs` (`data/results`, `data/demo`) | Run index: a one-level glob `*/run.json`. Nested folders (`_rebaseline/`, `_part3_backup/`) are skipped. |
| `<run>/state_log.parquet` | Replay state every 5 s, queried with DuckDB at each request |
| `<run>/signal.parquet`, `events.parquet`, `trip.parquet`, `queue.parquet` | Green intervals per TLS link, controller and preemption events, trips, queues. Cached by path, size and mtime. |
| `<run>/eta_prediction.parquet`, `eta_snapshots.parquet` | Precomputed ETA (Supabase `eta_prediction` columns) and the recorder's feature rows (`pending_<arm>`) |
| `data/models/eta/LATEST` → `<version>/reference.parquet` | ETA reference table: the degraded mode (D49) |
| `data/results/evaluation/<family>/{kpis,equity}.csv`, `configs/evaluation.yaml` | KPI tables and the S0…S2_noP labels |
| `configs/pitie_fixed.yaml` and its net file, `configs/calibration/pitie_302.yaml`, `data/field/pitie_measurements.yaml`, `configs/eta.yaml` | Static site: stages, phase map and program, crossing lengths, bus lines, junction centre, `calibrated_on`, `provisional`, trajets and student routes |

Raw SUMO XMLs are never read, since some runs no longer have them. The default replay run
is the first `default_run` folder that is indexed: `data/demo/pitie_rules_seed4000` (the
demo run with ETA predictions), else `data/results/pitie_rules_seed1`. When neither is
indexed, it is the first run in (scenario, seed) order.

## Common envelope (spec 9.2)

Every JSON response, errors included, carries these fields. The message's own fields sit
next to them, as in the spec's `bus_status` example.

| Field | Content |
|---|---|
| `data_source` | Always `"simulation"` (ENF-05) |
| `calibrated_on` | Last sensor day used by the demand calibration (`max(day_filter.days_used)` = 2026-10-08). The calibration file has no generation date, and adding one would change `demand_hash`. |
| `timestamp` | The instant the data describes, ISO 8601 in Europe/Paris. Replay instants are `replay_date` (today when null) + `begin_s` + t. |
| `data_age_s` | Replay data: query time minus the time of the data used (e.g. the last 5 s state sample). Aggregates: the wall-clock age of the file. |
| `stale` | `data_age_s` above the type's limit (ENF-04). The notice then says "dernière valeur connue". |
| `type` | `health`, `scenario_list`, `kpi_table`, `replay_state`, `crossing_status`, `bus_status`, `ambulance_route`, `eta` |
| `status` | `ok`, `degraded` (approximate or partial answer) or `unavailable` (no data) |
| `notice` | Human-readable reason (French, for the screens) or null |
| `provisional` | True while the field file has no measured signal timing or no measured queue |
| `run` | `{run_id, scenario, seed, controller, code_version, current}` or null. `current`: the run's `config_hash` and `demand_hash` equal those of its scenario's current config. |
| `sim_t_s` | Replay time (s since the scenario start) or null |
| `error` | `{code, message}` on errors, null otherwise |

**Freshness limits** (`api.yaml freshness_s`):
- `replay_state` 10 s, `crossing_status` 2 s, `ambulance_route` 10 s, `eta` 300 s;
- bus: 1.5 × the line's scheduled headway, because a fixed limit would flag nearly every bus status of a 6-10 min line.

## Replay clock

Time-dependent endpoints take `run_id` (default: the default run; `/replay/default` is the
alias) and `t`:
- seconds since the scenario start (`t=2700`);
- or a clock time `HH:MM[:SS]` (`t=08:15`), converted with the run's `begin_s`.

Without `t`, `clock.mode` decides:
- `fixed` (the default) always uses `clock.default_t` (08:15);
- `loop` folds the wall-clock time of day onto the run (demos).

A time before the start of the run is `bad_t`. A time past its end gives the last known
value, marked `stale`, up to one day (`clock.MAX_PAST_END_S`, 86400 s) past the end; a later
time (such as `t=1e12`) is `bad_t` too.

## Endpoints

| Endpoint | `type` | Fields besides the envelope |
|---|---|---|
| `GET /health` | `health` | `runs_indexed`, `eta_model_version` (the `LATEST` bundle) or null, `default_run` |
| `GET /scenarios` | `scenario_list` | `scenarios[{scenario, label, family, controller, runs[{run_id, seed, current, has_eta, dir}]}]` |
| `GET /kpi?scenario=&user=&family=main` | `kpi_table` | `scenario`, `label`, `family`, `source` (`evaluation` or `runs`), `rows[{user, kpi, mean, std, n, rel_change_vs_s0_pct, p_value, worse_flag}]` |
| `GET /replay/{run_id\|default}?t=&positions=false` | `replay_state` | `phase` (SUMO), `stage`, `stage_name`, `in_clearance`, `queue_total`, `queues{N,S,W}`, `counts{car,bus,pedestrian,ambulance}`, `positions[]` (with `positions=true`) |
| `GET /crossing/{NB\|S\|WB\|W}?speed=&t=&run_id=` | `crossing_status` | `signal`, `green_remaining_s`, `remaining_basis`, `next_green_in_s`, `length_m`, `speed_mps`, `walk_time_s`, `can_cross`, `obstructed` |
| `GET /bus/{line\|flow}/status?t=&run_id=` | `bus_status` | `line`, `flow`, `vehicle_id`, `delay_s`, `delay_basis`, `headway_s`, `scheduled_headway_s`, `priority_available`, `priority_active`, `message` |
| `GET /ambulance/next-intersections?t=&run_id=` | `ambulance_route` | `vehicle_id`, `approach`, `intersections[{id, name, modelled, status, distance_m, passed, reason}]`, `eta_destination_s` (null) |
| `GET /eta?trajet=&t=&run_id=&deadline=&student=false` | `eta` | `trajet`, `eta_median_s`, `eta_low_s`, `eta_high_s`, `approximate`, `model_version`, `queue_beyond_zone`, `rest_of_trip_s`, `advice{code, message, arrival_median, arrival_high, margin_s}` |
| `GET /mock` | HTML | Mock screens (dev only) |

**Endpoint rules:**
- **`/kpi`.**
  - `scenario` is an evaluation label of the family (`S0`, `S1`, `S2`, `S2_noP`) or a scenario name.
  - Mean, std and n over the seeds come from `kpis.csv`. The change versus S0, the p-value and the worse flag come from `equity.csv`.
  - Without evaluation outputs, the endpoint falls back to the runs' `kpi_run.csv`: `status: degraded`, `source: runs`, no test and no comparison.
- **`/replay`.** The phase is the logged SUMO phase index. Stage and clearance come from the run's own `network.stages`.
- **`/crossing`.**
  - `speed` is `normal` 1.2, `slow` 1.0, `reduced` 0.8 m/s (ASSUMED), or a number of m/s within 0.3-3.0.
  - `walk_time_s` = crossing length (from the net) / speed.
  - `can_cross` = the walk ends before the green does (conservative; the intergreen is not counted).
  - `signal: clearance` = the crossing's green just ended and the junction is in that stage's intergreen.
  - The answer comes from `signal.parquet` at 1 s resolution. Without it, the colour comes from the 5 s state log and the program (`status: degraded`, remaining time unknown).
- **`/bus`.**
  - The id is a flow (`91_E`) or a line (`91`; its latest bus in either direction).
  - Only buses that have arrived by t are used, so there is no future knowledge.
  - `delay_s` = time loss + insertion delay of the last bus.
  - `headway_s` = gap to the previous bus of the same flow.
  - `priority_available` = S2 rules with TSP enabled for the line. `priority_active` = a `tsp_extend` of the line whose window [t_event, t_event + allow] holds t.
- **`/ambulance`.** C's status follows the ambulance's own state, in this order:
  - `passed: true` (the ambulance is past the stop line of its approach in-edge, from the net's lane shapes): `clear`, "carrefour franchi". The junction is behind it.
  - `green_wave`, "préemption en cours, feu vert": a preemption request of this ambulance is open (no later served or timeout) and its own TLS link is green in `signal.parquet` at t. The link is the one logged in the request, else the net's link of its (in-edge, next edge) movement.
  - Otherwise the queue rule: `blocked` when its approach has at least `ambulance_blocked_halting` (10) halting vehicles, `clear` below that, `unknown` without data. An open request whose link is not green adds "préemption demandée, feu pas encore vert" to the reason.
  - An ambulance stopped (speed < 0.1 m/s) while its link is not green is `blocked`, "ambulance arrêtée au feu rouge", whatever the queue.
  - The ambulance shown is the nearest one still before its stop line, else the nearest one. The approach is that of its movement, else of the next (or last) planned ambulance.
  - `distance_m` is the straight line to the junction centre (it does not say on which side); `passed` says whether the junction is behind (null: no ambulance or no geometry). The mock shows "FRANCHI" and "Carrefour franchi" then.
- **`/eta`.**
  - `eta_*` are the junction-zone ETA (`y_zone_s`, D39) in whole seconds.
  - With `student=true` (routes N_S and W_S only), `rest_of_trip_s` (120 s, ASSUMED) is added in the advice's arrival times.
  - `advice` follows spec 11.5 (`ml.advice`, D52) when `deadline` (`HH:MM` or ISO) is given.
  - `queue_beyond_zone` is true when the run's `eta_snapshots` show cars waiting to enter that arm (`pending_<arm>` > 0); null without snapshots.

## Error policy (ENF-03)

| Situation | HTTP | Body |
|---|---|---|
| Unknown id or bad parameter | 404 | Envelope, `status: unavailable`, `error.code` (below) |
| FastAPI validation error (422) | 404 | Remapped: `error.code = bad_request` |
| Unknown path / wrong method | 404 / 405 | Envelope, `not_found` / `method_not_allowed` |
| Missing or unreadable data for a valid request | 200 | `status: degraded` or `unavailable` with a notice |
| Anything unexpected | 503 | Envelope, `error.code = internal`, "données temporairement indisponibles", no traceback |

**404 codes:** `unknown_run`, `unknown_crossing`, `unknown_bus`, `unknown_trajet`,
`bad_speed`, `bad_t`, `bad_deadline`, `bad_request`, `not_found`. The API never answers 500.
An unreadable Parquet or CSV file is logged and treated as absent.

## Degraded modes

| Case | Answer |
|---|---|
| ETA: the run has no prediction file (e.g. the evaluation runs), an unseen controller (`rl`), an unseen day type, or no row yet at t | The bundle's reference table (mean, p10, p90 per trajet, hour and day type; else the trajet's fallback row) in whole seconds: `approximate: true`, `status: degraded`, notice "estimation approximative (reason)" (spec 11.4). The advice message gets " (estimation approximative)". |
| ETA: no reference table either | `status: unavailable`, no values |
| Crossing without `signal.parquet` | Colour from the 5 s state log; remaining time unknown; `degraded` |
| Ambulance | Always `degraded`: "1 carrefour modélisé sur 3" |
| KPI without evaluation outputs | Means of the runs' `kpi_run.csv`; `degraded` |
| A run of another network (the grid baseline) | Crossing, bus, ambulance and ETA: `unavailable` ("autre réseau"); replay and KPI still answer |
| No runs at all | `/health` and `/scenarios` `degraded`; time-dependent endpoints `unknown_run` |

## Honesty labels (D53)

- **`provisional: true`** while the signal timing or the queues are not measured in the field (`data/field/pitie_measurements.yaml`).
- **`remaining_basis`**:
  - `program` for the fixed-time S0, whose future greens are known;
  - `replay` for adaptive controllers (S1, S2, RL). Their replayed countdown is future knowledge that a live system would not have; it could only guarantee the minimum green;
  - `unknown` without a signal log.
- **`delay_basis: junction_time_loss`**: the simulation has no timetable, so bus "delay" is the time lost crossing the junction area against free flow.
- **Ambulance corridor:** only C is modelled. The two other junctions are listed as `unknown`. Their names are ASSUMED and are to be confirmed at the field visit.
- **`obstructed: null`**: crossing obstruction is not logged in v1.
- **`eta_destination_s: null`**: the trip beyond the junction is not modelled for the ambulance.
- **`queue_beyond_zone`**: the junction ETA starts at physical entry into the modelled approach; the insertion backlog is flagged, not added.

**ENF-07:**
- no accounts, no cookies;
- CORS allows GET only;
- run uvicorn with `--no-access-log`, so client IPs are not logged;
- simulated vehicle ids are not personal data.

## Mock screens (S5, D54)

`src/traffic_twin/app/mock.html` is plain HTML and JS with no dependency, served at `/mock`.
- **Header:** the five profiles in one click (Piéton by default, then Étudiant, Ambulance, Bus, Opérateur), a run selector and a replay-clock slider with play.
- **Footer:** always "Données simulées", the calibration date, and whether the signal program is provisional (ENF-05).
- **Each view:** one main item and at most two secondary items. Every view handles the 12.1 states: loading, stale (last known value), degraded (approximate with a mention), unavailable, and error (a plain message, no technical detail).
- **Pedestrian view** (ENF-06, usable in under 5 s): one big verdict ("Vous pouvez traverser" / "Attendez"), the green left or the wait, and the walking time; the crossing and the speed are one click each.
- **Ambulance view:** large high-contrast text, no interaction.

The Streamlit app (spec 8.6) comes in Part 9.

## Latency (ENF-02)

ENF-02 asks for answers in less than 500 ms on precomputed results.

- `python -m traffic_twin.api.bench --results data/results data/demo --n 200 --docs docs/api.md` measures every endpoint in process (TestClient) and over HTTP (uvicorn and httpx), then rewrites the block below.
- `tests/test_api_latency.py` guards p95 < 500 ms on the synthetic fixtures in CI.

<!-- latency:begin -->
Measured 2026-10-16 02:51 (Europe/Paris) by `python -m traffic_twin.api.bench`: code `8fd1026-dirty`, Python 3.12.10, Windows 11 AMD64. 76 runs indexed from `data/results`, `data/demo`; default run `pitie_rules` seed 4000; ETA model `eta1-56926796`.

200 requests per endpoint after one warm-up request, parameters drawn with seed 4000. Cold start (app, run index, site, first request): 526 ms.

| Endpoint | In-process p50 (ms) | p95 | max | HTTP p50 (ms) | p95 | max | Answers |
|---|---|---|---|---|---|---|---|
| health | 4.9 | 6.6 | 21.1 | 1.6 | 2.2 | 3.1 | 200 ok: 200 |
| scenarios | 16.8 | 19.9 | 24.1 | 12.4 | 15.2 | 17.6 | 200 ok: 200 |
| kpi | 13.3 | 15.6 | 17.7 | 9.3 | 11.3 | 13.3 | 200 ok: 200 |
| replay (default run) | 15.8 | 18.7 | 66.6 | 12.7 | 14.8 | 16.8 | 200 ok: 200 |
| replay (positions) | 16.4 | 19.2 | 22.0 | 13.5 | 15.9 | 17.7 | 200 ok: 200 |
| replay (random run) | 17.4 | 44.4 | 56.0 | 13.7 | 18.7 | 27.0 | 200 ok: 200 |
| crossing | 7.1 | 20.6 | 33.0 | 3.7 | 8.2 | 65.8 | 200 ok: 191, 200 unavailable: 9 |
| bus | 9.4 | 42.3 | 98.5 | 6.0 | 21.1 | 36.9 | 200 degraded: 4, 200 ok: 184, 200 unavailable: 12 |
| ambulance | 16.0 | 21.9 | 50.4 | 12.7 | 16.8 | 20.0 | 200 degraded: 189, 200 unavailable: 11 |
| eta | 20.8 | 23.8 | 84.2 | 17.3 | 20.6 | 25.1 | 200 ok: 200 |
| error (404) | 3.7 | 5.4 | 6.5 | 1.3 | 2.0 | 2.9 | 404 unavailable: 200 |

**ENF-02 (< 500 ms on precomputed results): PASS**, worst p95 44.4 ms.
<!-- latency:end -->

## Tests

- `tests/test_api.py`: the envelope on every endpoint, the clock and age, crossing remaining green and the speed verdict, `remaining_basis`, bus past-only data, the TSP window and freshness, the ambulance corridor (green wave only on its own green, waiting at red, past the junction), scenario labels, `/kpi` from the evaluation and the fallback, ETA predictions and advice, `/mock`, CORS.
- `tests/test_api_degraded.py`: the ETA cascade, corrupt files, an empty results folder, every 404 code (including absurd replay times), the 422 remap, the 503 without traceback, unseen controller and day type, other networks, settings and clock parsing.
- `tests/test_api_latency.py`: the ENF-02 guard and the docs block rewrite.

All of them run on synthetic run folders (`api.fixtures`) and never read `data/results`.
