import pandas as pd
import pytest

from traffic_twin.sim.kpi import (
    BUS_STOP_COLUMNS,
    KPI_COLUMNS,
    bus_lines,
    compute_kpis,
    read_bus_stop_arrivals,
    read_bus_stops,
    read_statistics,
    read_tripinfo,
    user_type_of,
)

NO_ARRIVALS = pd.Series(dtype=float)


def _trip(id_: str, vtype: str, time_loss: float = 0.0, waiting: float = 0.0) -> dict:
    return {"id": id_, "type": user_type_of(vtype), "depart_s": 0.0, "arrival_s": 60.0,
            "duration_s": 60.0, "time_loss_s": time_loss, "waiting_s": waiting,
            "stops": int(waiting > 0)}  # fmt: skip


def _kpi_index(kpis: pd.DataFrame) -> pd.Series:
    return kpis.set_index(["user_type", "kpi"])["value"]


def test_kpis_from_sumo_outputs(sumo_outputs):
    tripinfo, stops = sumo_outputs
    trips, walks = read_tripinfo(tripinfo)
    arrivals = read_bus_stop_arrivals(stops)
    kpis = compute_kpis(trips, walks, arrivals, "s", 1)
    assert list(kpis.columns) == KPI_COLUMNS
    assert set(kpis["scenario"]) == {"s"} and set(kpis["seed"]) == {1}
    get = _kpi_index(kpis)
    assert get["car", "time_loss_mean"] == 10
    assert get["car", "count"] == 2
    assert get["bus", "delay_mean"] == 30
    assert get["bus", "headway_std"] == pytest.approx(pd.Series([400, 440]).std())
    assert get["pedestrian", "ped_wait_mean"] == 45
    assert not kpis["value"].isna().any()


def test_missing_stop_file_gives_no_headway(tmp_path):
    assert read_bus_stop_arrivals(tmp_path / "nope.xml").empty


def test_stop_arrivals_are_sorted(write_file):
    stops = '<stops><stopinfo started="900"/><stopinfo started="100"/><other/></stops>'
    assert read_bus_stop_arrivals(write_file("s.xml", stops)).tolist() == [100.0, 900.0]


def test_empty_tripinfo_gives_empty_kpis(write_file):
    trips, walks = read_tripinfo(write_file("trip.xml", "<tripinfos/>"))
    assert trips.empty and walks.empty
    kpis = compute_kpis(trips, walks, NO_ARRIVALS, "s", 3)
    assert kpis.empty
    assert list(kpis.columns) == KPI_COLUMNS


def test_missing_user_types_produce_no_rows():
    trips = pd.DataFrame([_trip("c0", "car", 5, 2), _trip("c1", "car", 15, 0)])
    kpis = compute_kpis(trips, pd.DataFrame(), NO_ARRIVALS, "s", 1)
    assert set(kpis["user_type"]) == {"car"}
    assert "delay_mean" not in set(kpis["kpi"])  # bus only
    get = _kpi_index(kpis)
    assert get["car", "count"] == 2
    assert get["car", "stops"] == 0.5


def test_pedestrians_only():
    walks = pd.DataFrame(
        {"id": ["p0", "p0", "p1"], "type": "pedestrian", "depart_s": 0.0, "arrival_s": 1.0,
         "duration_s": [100.0, 50.0, 30.0], "time_loss_s": [10.0, 20.0, 60.0]}
    )  # fmt: skip
    kpis = compute_kpis(pd.DataFrame(), walks, NO_ARRIVALS, "s", 1)
    get = _kpi_index(kpis)
    assert set(kpis["user_type"]) == {"pedestrian"}
    assert get["pedestrian", "count"] == 2  # persons, not walks
    assert get["pedestrian", "ped_wait_max"] == 60
    assert get["pedestrian", "ped_wait_mean"] == pytest.approx(30)


@pytest.mark.parametrize("arrivals", [[], [100.0], [100.0, 520.0]])
def test_bus_headway_needs_three_arrivals(arrivals):
    trips = pd.DataFrame([_trip("b0", "bus", 30)])
    kpis = compute_kpis(trips, pd.DataFrame(), pd.Series(arrivals, dtype=float), "s", 1)
    assert "headway_std" not in set(kpis["kpi"])
    assert _kpi_index(kpis)["bus", "delay_mean"] == 30


def test_regular_buses_have_zero_headway_std():
    arrivals = pd.Series([0.0, 420.0, 840.0, 1260.0])
    kpis = compute_kpis(pd.DataFrame(), pd.DataFrame(), arrivals, "s", 1)
    assert _kpi_index(kpis)["bus", "headway_std"] == 0


def test_single_trip_has_finite_kpis():
    kpis = compute_kpis(pd.DataFrame([_trip("a", "ambulance")]), pd.DataFrame(), NO_ARRIVALS,
                        "s", 1)  # fmt: skip
    assert set(kpis["user_type"]) == {"ambulance"}
    assert not kpis["value"].isna().any()


@pytest.mark.parametrize(
    ("vtype", "user"),
    [("bus", "bus"), ("ambulance", "ambulance"), ("car", "car"), ("", "car"), ("truck", "car")],
)
def test_user_type_of(vtype, user):
    assert user_type_of(vtype) == user


# -- Part 3 additions ---------------------------------------------------------------------
TRIPS_WITH_DELAY = """<tripinfos>
  <tripinfo id="c.0" vType="car" depart="10" departDelay="0.00" arrival="70" duration="60"
            timeLoss="5" waitingTime="0" waitingCount="0"/>
  <tripinfo id="c.1" vType="car" depart="30" departDelay="12.00" arrival="80" duration="50"
            timeLoss="9" waitingTime="4" waitingCount="1"/>
  <tripinfo id="bus_91_E.0" vType="bus" depart="0" departDelay="3.00" arrival="90"
            duration="90" timeLoss="30" waitingTime="20" waitingCount="1"/>
</tripinfos>"""

# Two lines at one stop (91: 3 arrivals, 24: 2 arrivals) and line 91 at a second stop
STOP_EVENTS = """<stops>
  <stopinfo id="bus_91_E.0" busStop="stop_SM_E" started="100" ended="120"/>
  <stopinfo id="bus_91_E.1" busStop="stop_SM_E" started="460" ended="480"/>
  <stopinfo id="bus_91_E.2" busStop="stop_SM_E" started="880" ended="910"/>
  <stopinfo id="bus_24_E.0" busStop="stop_SM_E" started="300" ended="320"/>
  <stopinfo id="bus_24_E.1" busStop="stop_SM_E" started="900" ended="920"/>
  <stopinfo id="bus_91_W.0" busStop="stop_SM_W" started="200" ended="220"/>
  <stopinfo id="bus_91_W.1" busStop="stop_SM_W" started="500" ended="520"/>
  <stopinfo id="bus_91_W.2" busStop="stop_SM_W" started="860" ended="880"/>
</stops>"""

ROUTES = """<routes>
  <vType id="bus" vClass="bus"/>
  <flow id="bus_91_E" type="bus" line="91" begin="0" end="9" period="360"/>
  <flow id="bus_91_W" type="bus" line="91" begin="0" end="9" period="360"/>
  <flow id="bus_24_E" type="bus" line="24" begin="0" end="9" period="600"/>
  <flow id="car_N_S_00" type="car" begin="0" end="9" probability="0.1"/>
</routes>"""


def test_depart_delay_is_parsed_and_reported(write_file):
    trips, _ = read_tripinfo(write_file("t.xml", TRIPS_WITH_DELAY))
    assert trips["depart_delay_s"].tolist() == [0.0, 12.0, 3.0]
    get = _kpi_index(compute_kpis(trips, pd.DataFrame(), NO_ARRIVALS, "s", 1))
    assert get["car", "depart_delay_mean"] == 6.0 and get["car", "depart_delay_max"] == 12.0
    assert get["bus", "depart_delay_max"] == 3.0


def test_depart_delay_absent_in_old_tripinfo(sumo_outputs):
    trips, _ = read_tripinfo(sumo_outputs[0])
    assert trips["depart_delay_s"].isna().all()
    kpis = compute_kpis(trips, pd.DataFrame(), NO_ARRIVALS, "s", 1)
    assert "depart_delay_mean" not in set(kpis["kpi"])  # NaN rows are dropped


def test_bus_lines_from_routes(write_file):
    assert bus_lines(write_file("r.rou.xml", ROUTES)) == {
        "bus_91_E": "91", "bus_91_W": "91", "bus_24_E": "24",
    }  # fmt: skip
    assert bus_lines(None) == {}


def test_read_bus_stops_takes_the_line_from_the_routes(write_file):
    path = write_file("stops.xml", STOP_EVENTS)
    stops = read_bus_stops(path, bus_lines(write_file("r.rou.xml", ROUTES)))
    assert list(stops.columns) == BUS_STOP_COLUMNS
    assert stops.groupby(["busStop", "line"]).size().to_dict() == {
        ("stop_SM_E", "24"): 2, ("stop_SM_E", "91"): 3, ("stop_SM_W", "91"): 3,
    }  # fmt: skip
    assert set(read_bus_stops(path)["line"]) == {"bus_91_E", "bus_24_E", "bus_91_W"}  # flow ids
    assert read_bus_stops(write_file("none.xml", "<stops/>")).empty


def test_headway_std_is_the_mean_over_stop_and_line_groups(write_file):
    stops = read_bus_stops(write_file("stops.xml", STOP_EVENTS), {"bus_24_E": "24"})
    get = _kpi_index(compute_kpis(pd.DataFrame(), pd.DataFrame(), stops, "s", 1))
    sd_e = pd.Series([360.0, 420.0]).std()  # 91 at stop_SM_E; line 24 has only 2 arrivals
    sd_w = pd.Series([300.0, 360.0]).std()
    assert get["bus", "headway_std"] == pytest.approx((sd_e + sd_w) / 2)
    assert get["bus", "dwell_mean"] == pytest.approx(170 / 8)


def test_headway_std_for_one_line_equals_the_pooled_value(sumo_outputs):
    stops = sumo_outputs[1]
    legacy = compute_kpis(pd.DataFrame(), pd.DataFrame(), read_bus_stop_arrivals(stops), "s", 1)
    grouped = compute_kpis(pd.DataFrame(), pd.DataFrame(), read_bus_stops(stops), "s", 1)
    assert _kpi_index(grouped)["bus", "headway_std"] == _kpi_index(legacy)["bus", "headway_std"]


def test_pedestrian_single_waits_and_p95():
    walks = pd.DataFrame(
        {"id": ["p0", "p1", "p2"], "type": "pedestrian", "depart_s": 0.0, "arrival_s": 1.0,
         "duration_s": 60.0, "time_loss_s": [10.0, 20.0, 60.0]}
    )  # fmt: skip
    single = pd.Series({"p0": 0.0, "p1": 30.0, "p2": 90.0})
    kpis = compute_kpis(pd.DataFrame(), walks, NO_ARRIVALS, "s", 1, ped_single_waits=single)
    get = _kpi_index(kpis)
    assert get["pedestrian", "ped_wait_p95"] == pytest.approx(walks["time_loss_s"].quantile(0.95))
    assert get["pedestrian", "ped_wait_single_mean"] == 40.0
    assert get["pedestrian", "ped_wait_single_p95"] == pytest.approx(single.quantile(0.95))
    assert get["pedestrian", "ped_wait_single_max"] == 90.0
    # new rows come after the existing ones of the group
    ped = kpis[kpis["user_type"] == "pedestrian"]["kpi"].tolist()
    assert ped[:4] == ["count", "ped_wait_mean", "ped_wait_max", "travel_time"]
    no_single = compute_kpis(pd.DataFrame(), walks, NO_ARRIVALS, "s", 1, pd.Series(dtype=float))
    assert "ped_wait_single_max" not in set(no_single["kpi"])


def test_read_statistics(write_file):
    stats = read_statistics(
        write_file(
            "statistics.xml",
            """<statistics>
  <vehicles loaded="690" inserted="688" running="0" waiting="2"/>
  <teleports total="1" jam="1" yield="0" wrongLane="0"/>
  <safety collisions="0" emergencyStops="3" emergencyBraking="4"/>
  <persons loaded="631" running="0" jammed="5"/>
  <personTeleports total="0" abortWait="0" wrongDest="0"/>
</statistics>""",
        )
    )
    assert stats == {"teleports": 1, "collisions": 0, "emergency_stops": 3,
                     "emergency_braking": 4, "vehicles_waiting_end": 2, "persons_jammed": 5,
                     "person_teleports": 0}  # fmt: skip
    assert set(read_statistics(write_file("x/none.xml", "<s/>").parent / "no.xml").values()) == {
        None
    }
