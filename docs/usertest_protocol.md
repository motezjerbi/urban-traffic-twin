# User-test protocol: the Urban Traffic Twin app (spec 12.2, A-10)

The app is tested with **3 to 5 people**, each playing one of the five users of spec 12
(pedestrian, student, bus driver, ambulance driver, operator). The tests check that each
view gives its answer in the time and the clicks of ENF-06, that people understand the
data are simulated (ENF-05), and they feed a fix loop. The tasks, instants, targets and
codes are pre-registered in `configs/usertests.yaml` (registered 2026-10-23, before any
participant; later changes only through its dated `changes` list). The results go into
`docs/usertests/results.yaml`, and `tasks.py usertest-docs` turns them into
`docs/usertests.md` and `docs/usertests/synthesis.json`.

**A-10 stays PENDING until at least 3 complete sessions with real participants are
recorded.** It becomes DONE for 3 to 5 sessions and is never reported as PASS. Nobody
writes a session that did not take place; the synthesis estimates nothing.

## What is frozen for the campaign

- **The run**: the S2 demo run `pitie_rules`, seed 4000, resolved by scenario and seed
  through `/scenarios`. The printable kit records its run_id and the sha256 of its
  `signal.parquet`.
- **The app**: one git commit for every session. `tasks.py usertest-serve` refuses a
  modified tree, a run whose run_id or signal sha256 differ from the kit, a clock that is
  not fixed, and a busy port. It prints the app version (git sha) to write on each sheet.
- **The answers**: `tasks.py usertest-kit` reads every expected answer from the API at the
  task's instant and refuses to build when one no longer holds, or when S3 is missing from
  `/kpi` (T-OP would have no answer before the Part 8 run).

If the demo runs have to be remade (for example after the field visit), the instants are
re-checked and any change goes in a dated `changes` entry **before** the next session.

## Recruitment

- 3 to 5 participants, ideally one per profile. People who do not hold the role play it
  and are recorded as `profile_match: proxy`; people who hold it are `real`. The report
  says which.
- Two rounds when time allows: **round 1** with P1-P3, then the fixes, then **round 2**
  with P4-P5 on the fixed app, which also verifies the fixes. With only 3 participants,
  fixes are verified by an automated test and reported as "fixed, not re-tested with
  users".
- About 27 minutes per session: book 30-minute slots. One laptop, Chrome full screen at
  100 % zoom.

## Consent and privacy (ENF-07)

- The consent text (kit page 3, French) is read aloud: simulated data, what is noted, no
  recording, the right to stop and to withdraw. Oral consent is ticked on the sheet. If the
  school requires written consent, the paper stays with the facilitator and is never
  scanned into the repository.
- Participants are **P1..P5** only. No name, contact details, employer, age or health
  information is written anywhere. **Reduced mobility is health data and is never
  recorded**: only the role played and `profile_match: real | proxy`.
- No audio or video recording.
- The participant keeps their P-id slip, which lets them withdraw their answers before
  the report date.
- `tasks.py usertest-check` refuses any key outside the schema, any session id other than
  `P<n>`, and any text that looks like an e-mail address or a phone number. It cannot catch
  a description that identifies someone, so notes describe what happened **on the
  screen**, never who the person is.

## Tasks

Instants are seconds since 07:30 on the simulated morning. The facilitator opens a start
link that sets the run and the instant; it never preselects the view, except the
ambulance screen, which needs no interaction (spec 12). `python -m traffic_twin.usertests
links` prints the links.

| Task | View | Instant | The participant is asked | Limit |
|---|---|---|---|---|
| `T-PED` | Piéton | 08:00:05 (1805 s) | whether they can cross now at crossing S, at normal speed | 30 s |
| `T-PMR` | Piéton | 08:00:05 (1805 s) | the same, walking slowly with a cane (reduced speed) | 30 s |
| `T-STU` | Étudiant | 08:15:00 (2700 s) | when to leave for an exam at 08:18 | 60 s |
| `T-BUS` | Bus | 07:51:45 (1305 s) | whether bus 91 is late at the junction and whether signal priority is active | 60 s |
| `T-AMB` | Ambulance | 08:10:40 (2440 s) | the state of the next junction, after a 3 s glance | 20 s |
| `T-OP` | Opérateur | none | which scenario has the lowest mean delay per person, according to the app | 90 s |
| `T-REPLAY` | Relecture (optional) | 08:40 | how many vehicles queue on the North approach under the rules (S2) | 120 s |

- The cards are French, like the app. `T-OP` asks a factual question: its answer is read
  from `/kpi` when the kit is built and is not a recommendation (the main solution stays
  S2, Part 7).
- **Order:** `T-PED` comes first for everyone, so it measures first use (ENF-06: pedestrian
  view usable in under 5 s). The other required tasks rotate by participant number;
  `T-REPLAY` comes last. The kit prints each participant's order.

## Session script

1. **Setup:** `tasks.py usertest-serve`; the participant's observation sheet with P-id,
   profile, real / proxy, round, app version; the cards in the sheet's order; a phone
   stopwatch to 0.1 s.
2. **Consent** (3 min) and **introduction** (2 min): the French lines of the kit are read
   word for word ("on teste l'application, pas vous ; données simulées ; répondez à voix
   haute ; je ne peux pas aider").
3. **Each task:** open the start link with the screen turned away. The participant reads
   the card and says "prêt". Turn the screen, say "Allez-y" and start the stopwatch.
   Count the clicks (a time entry counts as 1). Stop at the first complete spoken answer
   and write it word for word. For `T-AMB`, show the screen for 3 s, cover it, then ask.
   Then the SEQ (1-7) and 20 s of "qu'est-ce qui vous a aidé, ou gêné ?".
4. **Hint:** one scripted hint, only after 10 s of silence ("regardez la carte de votre
   profil sur l'écran d'accueil"). It caps the outcome at partial.
5. **Close:** the transparency question ("réelles, simulées, ou vous ne savez pas ?",
   recorded as `simulated_understood`), the optional SUS-10, a short debrief.
6. **Same day:** copy the sheet into `docs/usertests/results.yaml`, run
   `tasks.py usertest-check` until it is clean, log the problems as issues, then
   `tasks.py usertest-docs`.

**Scoring** (enforced by `usertest-check`):

- success: correct, no hint, within the time limit and at most 2 clicks;
- partial: correct after the hint, or over the click limit;
- fail: wrong answer or timeout (the time is then the limit);
- skipped: not run; excluded from n. Every required task appears in each session, as
  `skipped` if needed.

Error codes: `wrong_view`, `wrong_answer`, `misread`, `term`, `hesitation`, `help`,
`slider`, `gave_up`, `bug`, `other`.

**Measures:** time from "Allez-y" to the first full answer (phone stopwatch, about
± 0.3 s near the 5 s threshold, so every raw value is reported), clicks, outcome, error
codes, SEQ per task, the transparency answer, and an optional SUS-10. The SUS items in
the kit are our own French wording: the score is indicative with 3 to 5 people and is not
the validated F-SUS.

## Observer

The facilitator is a developer of the app, which biases the results. A second person who
times and notes the clicks is advised; when present, the timer's value is recorded.

## Fix loop

1. **Expert walkthrough before P1:** the defects found by walking through the app are
   logged as `U01`, `U02`, ... with `found_by: walkthrough` and no session (the
   current ones are in `docs/usertests/results.yaml`). The ones the spec needs are fixed
   before the first participant.
2. **During the tests:** each problem seen becomes an issue with `found_by: test`, the
   tasks and sessions where it appeared, a severity (4 blocks the task, 3 major, 2 minor,
   1 cosmetic) and a kind.
3. **Triage** after each round: issues of severity 3 or more are fixed first. A fix commit
   names the issue id ("fix(app): U03 ...").
4. **Closing an issue:** `status: fixed` needs the sha of the fixing commit, which
   `usertest-check` looks up with git. It is **verified** either by a later session whose
   app version contains the fix, or by an automated test node
   (`tests/test_x.py::test_name`) that exists. `wontfix` issues stay listed with a reason
   in `fix`.

## Synthesis

`docs/usertests.md` lists, per task: n, successes, success or partial, the median time
with every raw value, clicks (median, maximum, attempts within 2 clicks), hints, error
codes and the SEQ median; the results per round when both rounds exist; each
participant's success on their own profile's tasks; the issues and their fixes; and the
limitations. The ENF-06 lines (T-PED under 5 s, at most 2 clicks, the 3 s ambulance
glance) and the ENF-05 line print **PENDING (n<3)** until 3 sessions are complete, so one
or two sessions never read as a verdict.

**Limitations:** 3 to 5 participants give qualitative evidence, not statistics; proxies
are marked; the facilitator is a developer; the data are simulated (calibrated
2026-10-08) and the signal program is ASSUMED until the field visit.
