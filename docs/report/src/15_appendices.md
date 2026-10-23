## Appendix A. Decision index

The numbered decisions are recorded in `docs/architecture.md`:

- **D1-D18**, Part 3: the network, the calibration and the validation.
- **D19-D36**, Part 4: the Safety Guard, the shield, preemption, bus priority and the evaluation.
- **D37-D67**, Parts 5 and 6: the ETA model, the API and the RL environment.
- **D68-D90**, Part 7: the RL campaign, the selection and the interim spec 10.6 verdict.
- **D91-D120**, Parts 8 to 10: the final evaluation, A-06, the app, the user tests, this report and the demo.

## Appendix B. Reproduction

Every result rebuilds from the repository with these commands, run from the repo root. Seeds 1-10 are run only by the registered Part 8 step.

```
.venv\Scripts\python.exe tasks.py part3        # calibration, validation (A-02)
.venv\Scripts\python.exe tasks.py part8        # rebaseline S0-S2, the S3/S4 runs, A-06 (seeds 1-10, once)
.venv\Scripts\python.exe tasks.py part9        # demo runs, API bench, user-test kit and synthesis
.venv\Scripts\python.exe tasks.py report-facts # docs/report/facts.json from the local outputs
.venv\Scripts\python.exe tasks.py report       # report.md and report.pdf from facts.json
.venv\Scripts\python.exe tasks.py slides       # docs/slides/urban_traffic_twin.pptx
.venv\Scripts\python.exe tasks.py demo         # API + Operator for the live demo
```

## Appendix C. Glossary

| Term | Definition |
|---|---|
| TraCI | The interface through which a Python program reads and drives a SUMO simulation step by step. |
| timeLoss | The time a vehicle loses compared with an unhindered trip. |
| Headway | The time between two successive buses of the same line. |
| TSP | Transit signal priority: signal priority for public transport. |
| Preemption | Taking over the signals to clear the way for an emergency vehicle. |
| Box clearing | The rule that forbids entering the junction when the exit is blocked. |
| Actuated | Standard adaptive signals that extend the green on vehicle detection. |
| Occupancy | The share of time a sensor detects a vehicle; it shows saturation. |
| Digital twin | A simulation model calibrated on real data that reproduces a system's behaviour. |
| Replay | Playing back a simulation in the application. |
| Safety Guard | The layer that every controller's request goes through and that enforces the signal safety rules. |
| Hodges-Lehmann (HL) | The median of the pairwise averages of the per-seed differences: the paired shift estimate. |
| Holm | A step-down correction of p-values for several tests. |
| ASSUMED | A model input that is not measured on site yet. |
| NOT_MEASURED | A validation row that waits for a field measurement. |
| PENDING | A result that does not exist yet; it is never estimated. |

## Appendix D. Provenance of the numbers

Every result the report quotes, number or result sentence, is a fact of `docs/report/facts.json`; the template lint (D113) refuses a typed decimal, percentage, p-value or result claim. The integers typed in the templates are seed counts, rule thresholds and design constants of the configurations (the ASSUMED signal program among them). Each fact records its source file, that file's sha256 and the code version the source records. The table below groups the facts by source.

<!-- generate: provenance -->

## Appendix E. PENDING items

These facts and sections do not exist yet. Each one says what it waits for.

<!-- generate: pending -->
