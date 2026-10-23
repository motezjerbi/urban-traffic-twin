## 10. Validation against reality

Spec 13.4 asks that the simulation match reality before any comparison, on three checks:

- the flow per arm;
- the signal timing;
- the queues.

`tasks.py validate` writes `docs/validation.md` from S0 on seeds {{val.seeds}}. The simulated flows come from E1 loops that stand in for the Paris sensors.

**Flows: A-02 {{val.a02.verdict}}.** {{val.a02.pass}} of {{val.a02.rows}} inbound arm × slot rows are within ±{{val.flow_threshold_pct}} of the real flow. GEH is under 5 on {{val.geh.n_below}} of {{val.geh.n_counts}} distinct counts ({{val.geh.share_pct}}, target {{val.geh_target_share_pct}}).

<!-- include: docs/validation.md#Flows (A-02) | rows: N_in 508, S_in 511, W_in 513 -->

![Simulated against real flows](../../figures/pitie/validation_flows.png)

**The inbound check is close to circular.** The car demand is built from the same sensors, so A-02 mainly shows that the network carries the demand in the right time slots. The independent checks are these:

- the outbound flows, which are information only because they depend on the ASSUMED turn shares;
- the signal timing ({{val.signal_not_measured}} rows NOT_MEASURED);
- the queues ({{val.queue_not_measured}} rows NOT_MEASURED).

The signal timing and the queues wait for the field visit.

**After the field visit ({{meta.field_visit}}).** The measurements go into `data/field/pitie_measurements.yaml`, an empty template in which a missing value stays NOT_MEASURED and no estimate is ever typed. If the measured program or cycle differs beyond the validation thresholds, these steps follow (D119):

1. A dated change goes into the validation and evaluation configurations.
2. Part 3 is rerun: calibration, validation and the signal and queue rows.
3. The evaluation is rerun: S0 to S2, plus S3 with its unchanged model, labelled "trained on the ASSUMED program".
4. The ETA model is retrained.
5. The report becomes version 2.

Retraining RL is optional.
