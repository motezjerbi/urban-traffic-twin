# {{meta.title}}

## {{meta.subtitle}}

**Final report, {{meta.version}}**

{{meta.site}}

Authors: {{meta.authors}}<br>
Institution: {{meta.institution}} · Supervisor: {{meta.supervisor}}<br>
Date: {{meta.date}} · Code version: `{{meta.code}}`

> Every number in this report comes from simulation. The twin is calibrated on City of Paris sensor data, but its signal program is ASSUMED until the field visit ({{meta.field_visit}}). Until then the signal timing and queue checks are NOT_MEASURED and every result is provisional. A value that does not exist yet is printed **PENDING**, never estimated.

<!-- toc -->

## Abstract

This report presents a digital twin of one signalised Paris junction, Bd de l'Hôpital × Bd Saint-Marcel next to the Pitié-Salpêtrière hospital. It is built in Eclipse SUMO and calibrated on the City of Paris permanent traffic sensors. On the twin, four signal controllers are compared for four kinds of users: car drivers, bus drivers, an ambulance and pedestrians. The controllers are fixed time (S0), actuated (S1), rule based (S2) and reinforcement learning (S3). The results are shown in an application with one view per user profile.

**Calibration.** The calibrated demand reproduces the real inbound flows: **A-02 {{val.a02.verdict}}**, with {{val.a02.pass}} of {{val.a02.rows}} arm × slot rows within ±{{val.flow_threshold_pct}} of the sensors.

**Results on seeds {{eval.seeds}}.** Every controller runs on the same demand and the same seeds. Against fixed time, the rules controller S2 {{eval.person_delay.s2_verb}} the mean delay per person from {{eval.person_delay.s0}} to {{eval.person_delay.s2}} ({{eval.person_delay.rel_s2_vs_s0}}). With emergency preemption, S2 against S0 {{eval.a05.sentence}} (**A-05 {{eval.a05.verdict}}**).

**Reinforcement learning.** The selected RL policy S3 was looked at once in the Part 7 development test, on four pre-registered criteria: {{rl7.criteria_sentence}}. The pedestrian curb wait changed by {{rl7.c2.ped_rel}} against S1 (limit {{rl7.c2.limit}}). The interim spec 10.6 verdict is **{{rl7.verdict}}**. {{rl7.verdict_sentence}}. A-06, the pre-registered test of record on seeds 1-10, reads: {{a06.primary_sentence}} (**A-06 {{acceptance.A-06}}**).

**Travel time and application.** The served {{eta.interval_pct}} travel-time interval covers {{eta.a07.coverage_pct}} of the test trips (**A-07 {{eta.a07.verdict}}**). The application answers through one API in every view and labels its data as simulated. The user tests (A-10) are **{{acceptance.A-10}}** until real sessions are run.
