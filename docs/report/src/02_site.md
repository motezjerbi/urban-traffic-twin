## 2. Site choice

Paris was chosen for its data (spec 2.1). The City of Paris publishes the hourly flow and occupancy of its permanent loop sensors, for free and without a key. Flow and occupancy together separate free-flowing traffic from saturated traffic. The site can also be filmed to validate the simulation.

The junction was chosen with the weighted grid of spec 2.2. A usable sensor near each main approach is an eliminating criterion. The other criteria are: signals with three or four legs, a bus line, a busy pedestrian crossing, a hospital within a kilometre, a nearby university or station, a simple geometry, and access for filming.

The three candidate zones of spec 2.3 were scored with `tasks.py score`. **Bd de l'Hôpital × Bd Saint-Marcel (Pitié-Salpêtrière, Paris 13e, sensor node 302)** has the best score. It is the only candidate whose sensors were assessed: Lariboisière and Porte de Pantin keep a sensor score of 0 and are eliminated.

The junction has three general legs:

- N: Bd de l'Hôpital towards Austerlitz;
- S: Bd de l'Hôpital towards Place d'Italie;
- W: Bd Saint-Marcel.

It also has two bus-only legs (the Bd de l'Hôpital east bus road and the Bd Saint-Marcel bus road) and four signalised crossings. Bus lines 91, 24 and 57 stop at Saint-Marcel - La Pitié. The hospital gate on the bus road gives a realistic ambulance trip through the junction.

![Hourly profile of the busiest sensor arcs near Pitié-Salpêtrière](../../figures/profile_pitie_salpetriere.png)

The profile above shows the hourly flow and occupancy of the four busiest sensor arcs near the site, averaged over school-term Tuesday-Thursday days. The shaded band is the simulated 07:30-09:30 peak.
