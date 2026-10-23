## 9. Evaluation protocol and results

### Pre-registration

Spec 17 asks that the targets be fixed before the controllers are evaluated, so this section states the pre-registration before any result.

**The evaluation configuration.** `configs/evaluation.yaml` was pre-registered on **{{eval.preregistered}}**, before any S2 run on the evaluation seeds. It changes only through dated entries; there are {{eval.changes_n}} of them, listed below.

**The Part 8 keys.** S3 and S4, their pairs, the three new A-05 secondary tests, the A-06 rule, the spec 10.6 replication and the final-table layout were written in a dated change before any S3 or S4 run on seeds 1-10. A run lock in `evaluate` refuses any run on those seeds from an uncommitted tree or before that registration is committed. The record is:

- registration commit `{{a06.registration.commit}}`, committed {{a06.registration.committed_at}};
- first S3 run on the evaluation seeds started {{a06.registration.first_s3_started_at}};
- registration before the runs: **{{a06.registration.order}}**.

<!-- include: docs/evaluation.md#Pre-registration | table: 1 | cols: Date,Change -->

**Method (spec 13.2, 13.3).**

- **Families.** The main family runs S0, S1, S2, S2_noP, S3 and S4 on one demand with one ambulance per run. The ambulance-probe family runs seven ambulances per run to attribute the ambulance gain.
- **Pairs.** Every pair is compared per seed on seeds {{eval.seeds}} with the exact paired Wilcoxon test. Holm's correction applies over the headline KPIs within each pair; the other KPIs are secondary (unadjusted).
- **Equity.** A KPI more than {{eval.equity_pct}} worse than S0 in its bad direction is flagged. A hidden degradation is a person-weighted delay that improves significantly while a primary or subgroup KPI is flagged.
- **Person-weighted delay.** It counts car occupants at {{eval.occupancy.car}} and bus riders at {{eval.occupancy.bus}} per vehicle (ASSUMED).

### Final comparison table (spec S8)

<!-- generate: final_table -->

![Mean delay per person per controller, seeds 1-10](../../figures/report/person_delay.png)

### A-06: S3 against the baselines

**The rule (D94).** The test is the exact two-sided paired Wilcoxon on `all/person_delay_mean`, with the difference d = S3 − reference per seed. It reports the Hodges-Lehmann estimate and its confidence interval, against S1 (the primary), S2 and S0, with Holm's correction across the three.

**The status is procedural.** A-06 is PASS when every comparison has ten paired seeds, every S3 run is current with its pinned model and inputs, the family shares one demand, and A-03 passes. Otherwise it is INVALID, with the reasons. It is never FAIL because a result is unfavourable.

**The outcome** of each comparison is "S3 better" or "S3 worse" when Holm's p is under alpha, and "no significant difference" otherwise.

<!-- generate: a06 -->

**A-06 {{acceptance.A-06}}.** Primary comparison: {{a06.primary_sentence}}. Problems: {{a06.problems}}.

![A-06 forest plot](../../figures/pitie/eval/a06_forest.png)

### Spec 10.6 check on seeds 1-10 (reported, not gating)

The Part 7 criteria C1 to C3 are repeated exactly on seeds 1-10 against S1; C4 is cited from Part 7.

<!-- generate: rep106 -->

{{rep106.text}} Seeds 1-10 **{{rep106.confirms}}** the interim FALLBACK. Whatever this check gives, the main solution stays **{{rep106.main_solution}}**: the choice was fixed at week 7 (spec 15), and A-06 does not reopen it.

### KPIs per user (S0 to S2, main family)

<!-- include: docs/evaluation.md#KPIs per user and scenario | rows: car time_loss_mean, car wait_p95, car throughput_vph, bus delay_mean, bus headway_std, pedestrian ped_curb_wait_mean, pedestrian ped_curb_wait_max, ambulance travel_time_total, all person_delay_mean | cols: User,KPI,Unit,S0,S1,S2,S2_noP -->

![Headline KPI per user type](../../figures/pitie/eval/kpi_by_user.png)

### Ambulance (A-05)

The primary test is pre-registered: main family, S2 against S0, ambulance travel time from the planned departure, exact one-sided Wilcoxon. The result is a Hodges-Lehmann shift of {{eval.a05.hl}} {{eval.a05.ci}}, from {{eval.a05.mean_s0}} to {{eval.a05.mean_s2}}, with p {{eval.a05.p}} on n = {{eval.a05.n}}: **A-05 {{eval.a05.verdict}}**.

The ambulance-probe family runs seven ambulances per run, which gives the robust evidence:

- S2 against S0: HL {{eval.a05.probe.hl}}, p {{eval.a05.probe.p}};
- S2 against S2 without preemption: HL {{eval.a05.probe_nop.hl}}, p {{eval.a05.probe_nop.p}}.

This second comparison isolates the effect of preemption.

**Power note.** With one ambulance per run, the main-family test has limited power, so the probe family carries the attribution. Nothing was retuned after seeing a result.

![Ambulance travel time per seed](../../figures/pitie/eval/a05_ambulance.png)

### Safety (A-04)

<!-- include: docs/evaluation.md#A-04 | table: 1 | cols: Family,Scenario,Controller,Runs,Collisions,Teleports,Danger audit,Guard interventions,Worst curb ped wait (s),Bound (s),A-04 -->

Across every evaluated run, the longest pedestrian curb wait is {{eval.a04.worst_curb_wait}}; the bound is {{eval.a04.bound}} (A-04 {{eval.a04.verdict}}). The guard interventions per controller are in the table above.

### Equity and hidden degradation (spec 13.3)

<!-- include: docs/evaluation.md#Equity and hidden degradation (13.3) | table: 1 -->

Hidden degradation is flagged for: {{eval.hidden_degradation}}. {{eval.equity.sentence}}. Spec 13.3 asks for this check because an average gain can hide a user group that pays for it.

![Change against S0 per KPI and scenario](../../figures/pitie/eval/equity_heatmap.png)

### S4, out of distribution

S4 is S3 with preemption switched off. The policy never decided with an ambulance present without preemption, so S4 shows how far a learned policy degrades outside its training conditions. It is not an alternative design. S4 is exploratory, its pairs are reported in `docs/evaluation.md`, and it is never part of A-06.
