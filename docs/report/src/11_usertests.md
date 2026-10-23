## 11. User tests (spec 12.2, A-10)

**Status: {{usertests.status}}** (complete sessions: {{usertests.n}}). A-10 is **{{acceptance.A-10}}**.

The protocol was pre-registered before any session (`configs/usertests.yaml`, `docs/usertest_protocol.md`):

- **Tasks.** Each participant does six core tasks on the frozen app and on one frozen run (the rules controller on the demo seed): the pedestrian verdict, the pedestrian with reduced mobility, the student's departure advice, the bus delay, the ambulance corridor and the operator's comparison. The pedestrian task comes first for everyone, the other tasks rotate, and a replay task is optional.
- **Measures.** Each task is timed with a stopwatch and scored as success, partial, fail or skipped. The participant answers the single ease question after each task. The SUS questionnaire is optional and indicative.
- **Privacy (ENF-07).** Participants are identified only as P1, P2, and so on. There is no recording, and no e-mail or phone number. Reduced mobility is never recorded (the participant is marked as a real user or a proxy), and paper consent stays offline.
- **Fix loop.** Each issue gets an id (U01, U02, and so on). A fix needs a commit and is verified by a later session or by an automated test.

The results file `docs/usertests/results.yaml` is filled in by hand after each real session. The synthesis page `docs/usertests.md` is then generated. A-10 is DONE with three to five complete sessions and is never reported as PASS. Below three sessions every rate stays PENDING. No session has been invented: until real users are tested, this section reports PENDING.
