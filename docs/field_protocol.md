# Field protocol: Pitié-Salpêtrière junction (node 302)

Site: Bd de l'Hôpital × Bd Saint-Marcel, Paris 13e (TLS `C` in the model). This visit
turns the ASSUMED model inputs (signal program, lane use, turn shares) into measurements
and gives the independent checks of the validation (spec 13.4): cycle timing, queues
and turning counts. Results go into `data/field/pitie_measurements.yaml`, then into
`docs/validation.md` through `tasks.py part3`.

## When

- A typical morning: **Tuesday, Wednesday or Thursday in school term, 07:30-09:30**.
  - Planned: Tue 3, Wed 4 or Thu 5 Nov 2026, after the Toussaint holidays (17 Oct-1 Nov;
    the October window was not used). Wed 11 Nov is a public holiday.
- Within a week after the visit, re-download the sensor data (the dataset is rolling)
  so the same-day values of arcs 508, 511 and 513 exist. The 08:00-09:00 video count
  should match the sensor value stamped 09:00 (stamps mark the END of the hour).
  The download covers 12 months (`source.months`), but calibration and the A-02
  reference only use the autumn 2026 term (`calendar.typical_period` in
  `configs/data.yaml`). Typical days outside it, such as the spring term, are hold-out
  days: listed in `configs/calibration/pitie_302.yaml` (`day_filter.hold_out_days`),
  never mixed into the demand.

## Equipment and set-up

- Print the field kit: `tasks.py field-sheets` writes `docs/field/pitie_field_sheets.pdf`
  (12 A4 pages: checklist and schedule, site map, timing sheets, counting grids, queues,
  lane use, bus dwell, sheet-to-YAML guide). Bring a second copy of the timing pages.

- One or two phones on small tripods, filming the stop lines and the signal heads.
  Note each camera's position and what it sees (`sessions[].positions`, `video`).
- Synchronise every clock to phone time and film the phone clock at the start; record
  any camera offset in `camera_offset_s`. All times in the file are phone clock times.
- A tally counter or a counting app with 15-minute bins.

## What to measure

| Priority | What | How | File section |
|---|---|---|---|
| **MUST** | Signal timing | Over at least 5 consecutive cycles (ideally 10) at 08:00-09:00, note per head the start of green, yellow and red, to 1 s: vehicle heads on Bd de l'Hôpital N and S and on Bd Saint-Marcel, the bus heads (R11j) on both bus roads, the pedestrian heads of each crossing. Note the stage order, whether greens change between cycles, whether bus stages appear every cycle, and any pedestrian-only stage. | `signal` |
| **MUST** | Turning counts | 15-minute bins, at least 08:00-09:00 (07:30-09:30 better), for the 6 movements N→S, N→W, S→N, S→W, W→N, W→S. Count buses by line and two-wheelers separately. The N approach total decides 508 vs 510 as the N reference: about 165 vs about 340 vehicles per 15 min. | `counts` |
| **MUST** | Queues | At 2-3 instants per approach (N, S, W): vehicles queued at the end of red (`cycle_max`), with the exact clock time and the distance the camera sees (`visible_m`). | `queues` |
| **MUST** | Lane use and crossings | Lanes and arrows at each stop line; where the S bus lane starts and whether cars use its lane straight ahead; whether S→W left is allowed, and protected or not; whether a crossing exists on the N arm at the junction; whether the W and WB crossings are signalised or zebras, their distance from the junction, length and refuge island. | `geometry` |
| SHOULD | Pedestrian counts | Per crossing and direction, 2 × 15 min. | `counts[].pedestrians` |
| SHOULD | Buses | Arrival and departure per line and direction at Saint-Marcel - La Pitié (dwell), 08:00-09:00. | `bus_stops` |
| SHOULD | Context flows | Rue Poliveau inflow and right turns into Rue des Wallons, 15 min each. | `counts[].poliveau_in`, `wallons_right` |
| SHOULD | Loop locations | Saw-cut loops visible in the pavement per lane, and their distance to the stop line. | `loops` |
| NICE | Saturation headways | N straight lane and the S lanes, 3-5 cycles, from the 4th vehicle after green. | `saturation` |
| NICE | Ambulance access | Which gate the ambulances use. | `ambulance` |
| NICE | Conditions | Weather, incidents, roadworks. | `sessions[].weather`, `notes` |

### Timing sheet (one line per cycle)

| Cycle | Head | Green start | Yellow start | Red start | Note |
|---|---|---|---|---|---|
| 1 | N_straight | 08:02:10 | 08:02:43 | 08:02:46 | |

From the sheet: `cycle_s` = time between two green starts of `N_straight` (the reference
head); a head's `green_s` = yellow start - green start (one value per green interval);
`yellow_s` = red start - yellow start; `all_red_after_s` = from the head's red start to
the next green start of any other vehicle head.

**A head that turns green more than once per cycle** gets one list per stage, never one
mixed list. In the model this is `ped_NB` (the crossing over the bus road towards the
hospital), green with Bd de l'Hôpital (`HOP`) and again with Bd Saint-Marcel (`STM`):

```yaml
# example values
ped_NB: {green_s: {HOP: [36, 35, 36], STM: [20, 21, 20]}, flashing_s: ~}
```

Tag each green interval with the stage running at its green start (`HOP`: the Hôpital
straight heads are green; `STM`: the Saint-Marcel head `W` is green). The validation
compares each stage's green on its own, so a total (56 s) or a mean of the two (28 s)
would be wrong. `tasks.py field-check` rejects a plain list for such a head. If another
head turns out to be green twice per cycle on site, record it the same way: the extra
stage shows as NOT_IN_SIM until the program is updated.

### Head names (`signal.heads`)

`N_straight`, `N_right`, `S_straight`, `S_left`, `W` (Saint-Marcel), `bus_91_E`,
`bus_91_W`, `bus_57_N`, `bus_57_S`, `ped_S`, `ped_W`, `ped_WB`, `ped_NB` (as
`network.signal_groups` in `configs/pitie/network.yaml`).

## Filling the file

- Leave a value `~` (null) or a list `[]` until it is measured: the validation then shows
  NOT_MEASURED. **Never type an estimate.** A value that was not timed on site but
  copied from elsewhere goes with `signal.source: assumption` and shows as ASSUMED.
- Clock times are quoted strings: `"08:15:20"` (an unquoted 08:15 is read as a number).
- Names must match the model (heads, approaches N/S/W/NB/WB, crossings S/W/WB/NB, bus
  lines 91_E, 91_W, 24_E, 24_W, 57_N, 57_S). Unknown names, negative values, bad times and
  times outside the session are all reported together by `tasks.py field-check`.
- Queue instants in the first 5 minutes of the simulated window (before 07:35) or after
  09:30 are OUT_OF_WINDOW: the model has no warm-up before 07:30.

## After the visit

1. Fill `data/field/pitie_measurements.yaml` and run `tasks.py field-check` until it
   reports no problem.
2. If the stage structure differs from the model (`signal.stage_order`), update
   `sumo/network/pitie/pitie.tll.xml`, `network.stages`, `signal_groups`, `crossings` and
   `osm.lane_use`, then `tasks.py net tls-check --scenario configs/pitie_fixed.yaml`
   (without `--scenario`, `net` rebuilds the grid network). If a measured all-red is
   shorter than the geometric requirement, keep the minimal safe value in the model and
   let the signal table show the difference.
3. Set `fixed.green_s` so that each head's green (green phase plus the clearance phases
   where the head stays green) matches the measured mean.
4. Put the counts into the calibration (`tasks.py calib-data`: turn shares become
   `source: field`). Check in its output that the arms still use 16-18 autumn days and
   that any new spring days are listed as hold-out days.
5. Rerun `tasks.py part3`. Signal and queue rows become PASS or FAIL. A failure is fixed
   by adjusting the demand or the driving parameters, and documented (spec 13.4).
   - The calibration step stops the chain (exit 1) when an arm is capacity-limited, the
     loop did not converge, or the recommended `demand.arm_scale` differs from the config:
     copy the printed scale into `configs/pitie/demand.yaml` and rerun; never inflate the
     demand of a capacity-limited arm.
   - `compare` and `validate` rerun every evaluation run made with the old program, demand,
     calibration or net (its `config_hash` / `demand_hash` differs from the current
     config), and refuse to publish stale runs, so no result of the provisional program
     survives the update.
