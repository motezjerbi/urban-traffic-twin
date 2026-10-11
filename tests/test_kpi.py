import pandas as pd
import pytest

from traffic_twin.sim.kpi import (
    KPI_COLUMNS,
    compute_kpis,
    read_bus_stop_arrivals,
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
