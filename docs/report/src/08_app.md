## 8. API and application

### API (spec 9)

A FastAPI service reads the precomputed run folders and evaluation outputs. It has no live connection, no account and no access log, and answers GET requests only (ENF-07).

- **Envelope.** Every answer, errors included, carries the spec 9.2 envelope: data source (`simulation`), calibration date, timestamp, data age, stale flag, type, status (`ok`, `degraded` or `unavailable`), notice, `provisional`, the run and the replay time.
- **Errors (ENF-03, ENF-04).** Unknown ids and bad parameters give 404 with a code. Missing data give a `degraded` or `unavailable` answer with a French notice. Unexpected errors give 503 without a traceback.
- **Endpoints (spec 9.3).**
  - Part 5: health, scenarios, KPIs, replay, crossing, bus status, ambulance corridor and ETA.
  - Parts 8 and 9: the final evaluation, the pair comparisons, the queue series, the weight-sensitivity table and the replay events.
  - A label that is registered but not evaluated yet answers `unavailable` ("pas encore évalué"), never an error page.

<!-- include: docs/api.md#Latency (ENF-02) | table: 1 -->

**ENF-02 {{api.bench.verdict}}**: worst p95 {{api.bench.worst_p95}} on precomputed results (measured {{api.bench.date}}).

### Two front ends, one API

The application has two front ends, and both read only the API:

- **The user application** (`/mock`, Motez Jerbi's design) has a home screen with five profiles, plus views for the pedestrian, the student, the bus driver, the ambulance and the replay. A deep link opens any view of any run at any time (`?view=…&scenario=…&seed=…&t=…`), and the ambulance screen has a kiosk mode.
- **The Operator** (Streamlit) compares the scenarios. It has tabs for the final table, tests and pairs, equity, safety and the ambulance, queue curves, and weight sensitivity, and it offers CSV exports.

The application invents no value. A value the API does not give is shown as a degraded or unavailable state with its reason. Every view follows these rules:

- one main item and at most two secondary items (spec 12);
- the states of spec 12.1: loading, ok, degraded, stale, unavailable and error;
- a footer with the data source ("données simulées"), the calibration date, "programme de feux provisoire" while the program is ASSUMED, and the data timestamp and age (ENF-04, ENF-05);
- every answer within two clicks of the home screen (ENF-06).

**Replay (EF-21).** Motez's 2D and 3D renderers replay any simulated run, with the vehicle positions, the signal states and an event ticker from the API.

![Home screen with the five profiles](../../figures/app/home.png)

![Pedestrian view](../../figures/app/pedestrian.png)

![Student view](../../figures/app/student.png)

![Bus view](../../figures/app/bus.png)

![Ambulance view (kiosk)](../../figures/app/ambulance.png)

![Replay of the S3 run on the demo seed](../../figures/app/replay.png)

![Streamlit Operator](../../figures/app/streamlit.png)

![Error state of a bad deep link](../../figures/app/error.png)

**A-08 {{acceptance.A-08}}.** The evidence is the recorded rehearsal (`docs/demo/rehearsal.json`). It checks every view through the API, the footer, and the loading, error and degraded states.
