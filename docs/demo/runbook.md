# Demo runbook (Part 10)

The live demo of the Pitié-Salpêtrière digital twin: what to start, what to show in which
order, what to say, and what to do when something fails. The script itself is
[`configs/demo.yaml`](../../configs/demo.yaml). The automated rehearsal checks every step
of this runbook (`python tasks.py demo-rehearsal`) and records the A-08 / A-09 evidence in
[`rehearsal.json`](rehearsal.json). Log each human rehearsal by hand in
[`rehearsal_log.yaml`](rehearsal_log.yaml).

**Every value on screen is simulated.** The app shows "données simulées", the calibration
date and "programme de feux provisoire" (the signal program is ASSUMED until the field
visit). Say this in the first minute. Never quote a number that is not on screen or in
[`docs/evaluation.md`](../evaluation.md).

## 1. Before the demo (day before, then 30 min before)

| When | Action | Expected |
|---|---|---|
| Day before | `python tasks.py demo-rehearsal` | `PASS`, or `PENDING` with only Part 8 items. Fix any `FAIL` with the hint it prints. |
| Day before | One full human rehearsal with this runbook, then an entry in `rehearsal_log.yaml` | `manual_code_edits: 0` |
| Day before | Open the slides at slides 12 and 15 (the fallback) | The screenshots load |
| 30 min before | Close every other server; `python tasks.py demo-clean` | Ports 8000 and 8501 free |
| 30 min before | `python tasks.py demo` | Prerequisites listed, "API up", "Operator up", the quick check, then the home page opens |
| 30 min before | Click through the 8 steps once (section 3) | Each view loads in under 2 s |

On a fresh clone (another laptop), first run `python tasks.py bootstrap`. It restores the
demo pack (`docs/demo/demo_pack.zip`: demo runs, evaluation outputs, ETA reference table).
Then `python tasks.py demo-readiness` checks that a clean checkout runs the demo with no
manual step.

`tasks.py demo` refuses to start when port 8000 holds another server. In particular
`run_mock.py` serves invented data and must never be shown. It also refuses when a
prerequisite is missing, and prints the fix.

## 2. Two-screen setup

- **Projector (audience):** the browser with the user app, `http://127.0.0.1:8000/mock?view=home`,
  full screen (F11), zoom 110-125 %.
- **Laptop (presenter):** this runbook, the terminal running `tasks.py demo`, and a second
  browser tab with the Operator (Streamlit), `http://127.0.0.1:8501`. Drag it to the
  projector at step 8.
- One laptop only: two browser tabs (user app, Operator), switched with Ctrl+Tab.
- No network is needed. Everything runs on 127.0.0.1, and no simulation runs during the
  demo (spec 7.1).

## 3. The walkthrough (about 12 minutes)

Each step has a deep link (scenario + seed, never a run id), so a step can be reopened
directly if a click goes wrong. The links below are for the default ports.

| # | Step | Deep link | Show | Say | Time |
|---|---|---|---|---|---|
| 1 | Home | `/mock?view=home` | The 5 profile cards, Relecture, Opérateur; the footer | One twin, five users; the footer: simulated data, calibration date, provisional program | 1 min |
| 2 | Pedestrian | `/mock?view=pedestrian&scenario=pitie_rules&seed=4000&t=1805&crossing=S&speed=normal` | The verdict, the green time left, the walking time; switch the speed to "réduite" | The answer is at most 2 clicks for any crossing and speed (ENF-06); obstruction is not measured | 1.5 min |
| 3 | Student | `/mock?view=student&scenario=pitie_rules&seed=4000&t=2700&trajet=N_S&exam=08:18` | The median ETA, the 80 % interval, the departure advice | The interval comes from the conformal ETA model (A-07); "non disponible" when the model is missing | 1.5 min |
| 4 | Bus | `/mock?view=bus&scenario=pitie_rules&seed=4000&t=1305&line=91` | The delay at the junction, the headway, the signal priority | Delay basis = junction time loss; the relay message can be copied for the driver | 1 min |
| 5 | Ambulance | `/mock?view=ambulance&scenario=pitie_rules&seed=4000&t=2440&kiosk=1` | The 3 junctions; the modelled one in "onde verte" | Only one junction is modelled: the other two say "non modélisé", and the view is honestly "degraded" | 1 min |
| 6 | Replay | `/mock?view=replay&scenario=pitie_rl_s3&seed=4000&t=2440&play=0` | The 2D/3D map of a precomputed S3 run; press play ×5; the event ticker | A replay of a simulated run (EF-21), not live traffic. The S3 ETA is "approximative": the ETA model never saw the RL controller | 2 min |
| 7 | Operator card | `/mock?view=operator` | Person delay per scenario, ablations dimmed | Values come from the 10-seed evaluation (`/kpi`); the link opens the full Operator | 0.5 min |
| 8 | Operator (Streamlit) | `http://127.0.0.1:8501` | Tabs Comparaison, Tests et paires (A-06), Équité, Sécurité et ambulance, Courbes, Sensibilité, Exports | See section 4 | 3 min |

The instants were checked on `pitie_rules` seed 4000: bus 91 priority is active at 1305 s,
and the ambulance's green wave is at 2440 s.

## 4. What to say about the results

- **Simulated data.** The twin is calibrated on City of Paris sensor counts, but the signal
  program is ASSUMED until the field visit (3-5 Nov 2026). Every comparison is
  simulation-against-simulation under that program.
- **Main solution: S2 (rule-based with preemption).** The choice was fixed at week 7 by the
  Part 7 interim verdict (FALLBACK: the RL policy did not meet the spec 10.6 criteria on the
  dev-test seeds). A-06 does not reopen this choice.
- **A-06 (RL policy S3 against S0, S1 and S2, seeds 1-10).** This is the pre-registered,
  confirmatory test of record. It is presented whatever its direction. Read the sentence and
  the numbers on the "Tests et paires" tab, or in `docs/evaluation.md`. Do not paraphrase
  them with other numbers.
- **The 10.6 check on seeds 1-10** is reported, not gating. Say whether it "confirms" or "does
  not confirm" the interim FALLBACK, as the page states.
- **S4 (S3 without preemption)** is out of distribution: the policy never decided with an
  ambulance present. It shows degradation, not a design.
- **Ambulance (A-05).** One ambulance per run gives limited power. The ambulance-probe family
  is the robust evidence.
- **User tests (A-10)** are PENDING until at least 3 real sessions are logged. Do not claim
  results.

## 5. When something fails

| Symptom | What to do (in order) |
|---|---|
| A view shows "Chargement" for more than 4 s | Wait for the 4 s timeout: the view then shows the last known value or an error. Reload the deep link (F5). |
| "Erreur : Données temporairement indisponibles" | The API stopped. Show it: this is the spec 12.1 error state, never a blank page. Restart with `python tasks.py demo`, then reload. |
| `tasks.py demo` says port 8000 is held by another server | Stop that server (often `run_mock.py`, which serves invented data), or run `python tasks.py demo-clean`, then retry. |
| `tasks.py demo` lists a missing prerequisite | Run the printed fix: `bootstrap` from the pack, or the named task. Do not edit code. |
| The 3D map does not render (graphics driver) | Switch the map to 2D, or continue with the slides fallback. |
| The Operator (Streamlit) does not load | Show its screenshots on slide 12, and the A-06 sentence from `docs/evaluation.md`. |
| The laptop or browser fails | **Fallback:** slides 12 (app screenshots of every view, the replay and the Operator tabs) and 15 (this runbook and the fallback). The replay is a precomputed run, so the screenshots show exactly what the live replay would show. |

After the demo, stop with Ctrl+C in the terminal. Both servers stop, and the terminal says
"Stopped; ports freed". Log the rehearsal or demo in `rehearsal_log.yaml`.

## 6. Commands

| Command | Does |
|---|---|
| `python tasks.py demo` | Prerequisites, impostor and port checks, API (8000) + Operator (8501), quick check, opens the home page; Ctrl+C stops both |
| `python tasks.py demo-check` | Walks every step against the running API (API calls + DOM checks) |
| `python tasks.py demo-rehearsal` | Headless rehearsal on the real data (add `--record` through `python -m traffic_twin.demo rehearsal --record` for the A-08 evidence) |
| `python -m traffic_twin.demo rehearsal --fixtures --no-browser` | The CI rehearsal on FAKE data |
| `python tasks.py demo-clean` | Stops demo leftovers and removes old reports (never `data/results`, `data/demo` or `models`) |
| `python tasks.py demo-pack` | Writes `docs/demo/demo_pack.zip` + `MANIFEST.json` |
| `python tasks.py bootstrap` | Restores the pack into a fresh clone (never rebuilds the evaluation) |
| `python tasks.py demo-readiness` | Fresh `git clone --local` + bootstrap + rehearsal (A-09) |

Reports go to `data/demo_check/<timestamp>-<kind>/report.{json,md}` (gitignored).
