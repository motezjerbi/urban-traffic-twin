## 1. Context and objectives

### Context

Picture one morning peak and four people caught in the same congestion, each needing something different:

- students who must not be late for an exam;
- an ambulance driver who has to reach a patient;
- a bus driver who has to keep to the timetable;
- a pedestrian stuck at a busy crossing.

Fixed-cycle signals treat all four the same. The central question of the project (spec 1.2) is whether adaptive signal control, learned from real traffic data and tested in a faithful simulation, can cut the delay of every user category without sacrificing one of them for the others.

The product is a prototype digital twin of one junction (spec 1.3). It has four parts:

- a calibrated simulation;
- several signal strategies that can be compared with each other;
- a travel-time prediction model;
- an application that shows each user the information that matters to them.

The application's data come from the simulation, which is calibrated on real data. They are not a live feed from the city. The interface states this limit on every screen.

The project follows five guiding principles (spec 1.4):

- every screen supports one decision;
- there is no account and no needless notification;
- uncertainty is shown honestly (a travel time always comes with its interval);
- the comparison is fair (every scenario runs on the same demand and the same random seeds);
- every ambitious component has a simpler fallback that already works.

![The six layers of the twin](../../figures/report/pipeline.png)

### Objectives

The specific objectives of spec 3.2 are listed below. The result column is generated from the evidence files: a result that does not exist yet reads PENDING.

<!-- generate: objectives -->

### Scope and deviations

The scope is one signalised, real and calibrated junction. The users are cars, buses, one ambulance and pedestrians, and the 07:30-09:30 morning peak is simulated. The application has five profiles: student, ambulance, bus, pedestrian and operator. These are the project's deviations from the spec, each decided and recorded (D91):

- **User views.** The views are an HTML and JavaScript client of the API, built on Motez Jerbi's design. The Operator is a Streamlit page; its existing Altair charts are kept, and Plotly is used only for the new charts.
- **Replay (EF-21).** Only simulated runs are replayed. A recorded real day is not available.
- **What is not built.** The simulation-on-demand endpoint (EF-22) is not built, and the export (EF-23) is CSV only. Both are priority C in a frozen scope (spec 16).
