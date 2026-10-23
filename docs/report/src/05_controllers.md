## 5. Controllers S0-S3 and the Safety Guard

Every controller runs on the same SUMO session, the same demand and the same seeds. Its stage requests go through one Safety Guard.

- **S0, fixed time** (`configs/pitie_fixed.yaml`) runs the provisional program of section 4.
- **S1, actuated** (`configs/pitie_actuated.yaml`, EF-06) emulates SUMO's gap-based actuated logic through the common controller interface. Each green lasts at least 6 s, is extended while a vehicle crossed one of the stage's loops in the last 3 s, and ends at its max green. Its max greens keep every crossing's worst-case red at or under 88 s. `tests/test_actuated_native.py` checks, step by step, that it takes the same decisions as SUMO's own actuated logic.
- **S2, rules** (`configs/pitie_rules.yaml`, EF-07 to EF-09) adds four rules on top of the S1 gap logic, with longer max greens:
  - *Emergency preemption.* Every emergency vehicle on an approach, or waiting to enter, is detected, served first come first served and held until it passes the stop line. A request is dropped after 120 s without progress.
  - *Bus priority.* A bus green is extended by up to 10 s for a bus of lines 91, 24 or 57 that can reach the stop line in time.
  - *Box clearing.* A loop whose lane only feeds a blocked exit does not extend the green.
  - *Pedestrian protection.* The controller leaves a green itself before the shield would force it.
- **S2_noP** is S2 without preemption, an ablation that attributes the ambulance gain.
- **S3, reinforcement learning** (`configs/pitie_rl_s3.yaml`) plays the DQN policy selected in Part 7 (section 6). It chooses between staying in the current stage and advancing to the next one, under strict stage order. The guard and the preemption keep priority over it.
- **S4** is S3 with preemption switched off, the spec 13.1 option. It is **out of distribution**: the policy never decided with an ambulance present without preemption. Its results describe a degradation, not a design, and they never enter A-06.

**Safety Guard (EF-11, ENF-11).** One guard (`control/safety.py`) filters every request of every controller (D20). It applies these rules in order:

1. an unknown stage is an error;
2. clearance (yellow and all-red) is never cut;
3. the min green of 6 s holds, even for an ambulance;
4. emergency preemption moves to the next link-safe stage towards the ambulance's movement;
5. anti-starvation, strict stage order and the max green apply, all suspended during preemption;
6. last and hard, the **pedestrian shield**.

The shield keeps every crossing's curb wait under 88 s. A green may continue only while, for every crossing, the longest current curb wait plus the time to that crossing's next green stays within the bound. The curb wait is a per-person clock that keeps running when a waiting pedestrian creeps a few centimetres, which SUMO's own waiting time does not do.

Every run also audits the phases it showed: a green ending without yellow, a pedestrian green cut outside clearance, a yellow turning green, a short green, an incomplete clearance, an intergreen that is too short, conflicting greens, illegal transitions. The counts go to the run's safety record and are judged by A-04.
