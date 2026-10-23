## 3. Data and calibration

**Sensor data.** The demand comes from the hourly flows of the Paris permanent sensors. Each value is stamped at the end of its hour (D3). Typical days are Tuesday to Thursday in school term, without public holidays. A robust per-arc MAD filter then removes outliers. The extract covers 2026-07-13 to 2026-10-10, which leaves 16 to 18 typical days per arc and slot.

**Reference arcs (D6).** One reference arc is used per inbound arm:

- N inbound: arc 508, upstream of Rue Poliveau, the nearest station with its own flow;
- S inbound: arc 511;
- W inbound: arc 513.

Several Paris arcs copy one station, so the twin has about three independent inbound counts. The extract is versioned in `data/reference/pitie_302_hours.csv`, so the calibration rebuilds without the raw download.

**Demand.** The demand is built per 900-second slot from the typical-day means:

- car flows per movement, using ASSUMED turn shares;
- bus lines 91, 24 and 57 with their stop;
- pedestrian flows along every crossing;
- one ambulance (N to S at 08:10).

Flows use per-second probabilities. Every scenario therefore sees the same expected demand, and each seed gives a different draw (spec 13.3).

**Calibration loop.** `tasks.py calibrate` runs S0 on the calibration seeds 101-103, never on the evaluation seeds. It scales each inbound arm until the arm's loop flow matches the sensor flow within the calibration tolerance. An arm that stops following its demand would be reported as capacity-limited instead of being inflated. Result: **{{val.calibration}}** (`docs/calibration_log.yaml`); {{val.calibration.rescaled}}.

The validation thresholds were written in `configs/validation.yaml` on {{val.preregistered}}, before any calibrated result (spec 17). They are:

- a flow error of at most ±{{val.flow_threshold_pct}} per inbound arm and slot;
- GEH below 5 on at least {{val.geh_target_share_pct}} of the distinct counts.

The checks themselves are in section 10.
