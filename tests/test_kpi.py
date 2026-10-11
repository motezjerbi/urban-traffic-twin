import pandas as pd
import pytest

from traffic_twin.sim.kpi import KPI_COLUMNS, compute_kpis, read_bus_stop_arrivals, read_tripinfo

TRIPINFO = """<tripinfos>
  <tripinfo id="car_W_straight.0" vType="car" depart="27000" arrival="27060" duration="60"
            timeLoss="20" waitingTime="10" waitingCount="1"/>
  <tripinfo id="car_W_straight.1" vType="car" depart="27010" arrival="27050" duration="40"
            timeLoss="0" waitingTime="0" waitingCount="0"/>
  <tripinfo id="bus_B1.0" vType="bus" depart="27000" arrival="27100" duration="100"
            timeLoss="30" waitingTime="15" waitingCount="1"/>
  <personinfo id="ped_w_in_e_out.0" depart="27000">
    <walk depart="27000" arrival="27300" duration="300" routeLength="400" timeLoss="45"/>
  </personinfo>
</tripinfos>"""

STOPS = """<stops>
  <stopinfo id="bus_B1.0" started="100"/><stopinfo id="bus_B1.1" started="500"/>
  <stopinfo id="bus_B1.2" started="940"/>
</stops>"""


def test_kpis_from_sumo_outputs(tmp_path):
    (tmp_path / "trip.xml").write_text(TRIPINFO)
    (tmp_path / "stops.xml").write_text(STOPS)
    trips, walks = read_tripinfo(tmp_path / "trip.xml")
    arrivals = read_bus_stop_arrivals(tmp_path / "stops.xml")
    kpis = compute_kpis(trips, walks, arrivals, "s", 1)
    assert list(kpis.columns) == KPI_COLUMNS
    get = kpis.set_index(["user_type", "kpi"])["value"]
    assert get["car", "time_loss_mean"] == 10
    assert get["car", "count"] == 2
    assert get["bus", "delay_mean"] == 30
    assert get["bus", "headway_std"] == pytest.approx(pd.Series([400, 440]).std())
    assert get["pedestrian", "ped_wait_mean"] == 45


def test_missing_stop_file_gives_no_headway(tmp_path):
    assert read_bus_stop_arrivals(tmp_path / "nope.xml").empty
