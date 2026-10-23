## 13. Limitations

- **The twin learns the simulator.** One junction, one day type (the school-term Tuesday-Thursday peak) and ten evaluation seeds: the results compare controllers on the modelled junction. They describe the real junction only once the field visit has validated the program.
- **The signal program is ASSUMED.** So are every number derived from it: max greens, hold caps, crossing reds. The signal-timing and queue checks of spec 13.4 are NOT_MEASURED until the visit.
- **The inbound flow check is close to circular.** The demand comes from the same sensors. Several Paris arcs copy one station, which leaves about three independent inbound counts.
- **Detection is ideal.** Pedestrians (camera or push button), buses (positions on the approach) and the ambulance (detected before it enters the network) are all seen perfectly. The ambulance obeys red lights; no rescue lane forms.
- **S3 does not observe pedestrians.** Its pedestrian cost is that of a policy that cannot see them, and only the shield bounds their wait. S4 is out of distribution.
- **One ambulance per run** gives the main A-05 test limited power; the probe family carries the attribution.
- **Occupancies are ASSUMED.** The person-weighted delay depends on the assumed car and bus occupancies; the evaluation reports its sensitivity to the bus occupancy.
- **Insertion delay is not in the delay KPIs.** Cars waiting to enter a capacity-limited arm are outside the delay KPIs, so those KPIs understate the gap between controllers on the seeds where an arm saturates.
- **The application shows simulated data only.** The replay plays simulated runs, not a recorded real day.
- **User tests and rehearsal depend on people.** The user tests (A-10, **{{acceptance.A-10}}**) and the human demo rehearsal ({{demo.rehearsed}}) each stay PENDING until they happen.
