import pytest

from traffic_twin.data.intersection_score import score_candidates


def cfg(a_sensor: int = 5) -> dict:
    return {
        "weights": {"sensor": 5, "bus_line": 4},
        "eliminatory": ["sensor"],
        "candidates": {
            "a": {"scores": {"sensor": a_sensor, "bus_line": 5}},
            "b": {"scores": {"sensor": 3, "bus_line": 3}},
        },
    }


def test_ranking_and_percent():
    r = score_candidates(cfg())
    assert r.iloc[0]["candidate"] == "a"
    assert r.iloc[0]["score_pct"] == 100.0
    assert r.iloc[1]["score_pct"] == 60.0


def test_eliminatory_sensor_goes_last():
    r = score_candidates(cfg(a_sensor=0))
    assert r.iloc[0]["candidate"] == "b"
    assert bool(r.iloc[1]["eliminated"])


def test_rejects_out_of_range_and_missing():
    bad = cfg()
    bad["candidates"]["a"]["scores"]["bus_line"] = 7
    with pytest.raises(ValueError):
        score_candidates(bad)
    del bad["candidates"]["b"]["scores"]["bus_line"]
    with pytest.raises(ValueError):
        score_candidates({**bad, "candidates": {"b": bad["candidates"]["b"]}})


def test_repo_config_is_valid():
    from traffic_twin.config import load_config

    score_candidates(load_config("configs/intersections.yaml"))
