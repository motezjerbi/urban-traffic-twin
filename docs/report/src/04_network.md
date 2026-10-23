## 4. Network and signal program

**Network pipeline.** The network is built from OpenStreetMap:

1. `osm_import` crops the OSM extract around the junction and joins the signal nodes into one node.
2. It names each leg, and writes explicit lane-to-lane connections and explicit crossings.
3. It opens the bus lanes to emergency vehicles.
4. `netconvert` builds the SUMO network from these curated plain files.

The process is deterministic and its output is versioned. SUMO is pinned, because the link order of the traffic light depends on its version.

**Signal program check.** `tasks.py tls-check` checks the program against the geometry:

- no two conflicting vehicle links are green together;
- every intergreen covers the path to the conflict point;
- the worst-case red of every crossing is at most {{net.red_bound}}, the 90 s pedestrian target minus a 2 s margin.

The check is recomputed from the committed S0 program when the report is built. It reads **{{net.tls_check}}**: {{net.phases}} phases and a cycle of {{net.cycle}}, {{net.foe_greens}} conflicting greens and {{net.intergreen_violations}} intergreen violations. The worst-case reds are {{net.worst_red}}.

**Provisional program (ASSUMED until the field visit).** The cycle is {{net.cycle}}: {{net.green_total}} of green and {{net.clearance}} of clearance. The stage greens below are the design values of `configs/pitie/network.yaml` (`fixed.green_s`), not measurements:

| Stage | S0 green (s) | Served |
|---|---|---|
| HOP | 36 | N straight, S straight, crossings NB, WB, W |
| S_LEFT | 6 | S straight and left, bus 57 north, crossing WB |
| STM | 20 | W (Saint-Marcel), N right, crossings NB, S, WB |
| BUS_91 | 6 | bus 91 and 24 both ways, N right, crossing S |
| BUS_57 | 6 | bus 57 south, bus 91 and 24 west, N right |

The program is a lead sketch adapted to the junction; spec EF-05 asks for the phases measured on site. Other inputs are also ASSUMED until the visit:

- lane use at the stop lines;
- the turn shares;
- bus headways and dwell times;
- pedestrian flows;
- the absence of a crossing on the N arm.

The network is cropped to the junction, so the neighbouring signals are not modelled.
