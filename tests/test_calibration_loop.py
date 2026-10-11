"""Calibration loop with a fake simulator: convergence, a capacity-limited arm frozen at
its last responsive scale, the iteration limit, a deterministic log, and the exit code of
the CLI (1 unless converged with the configured arm_scale)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from traffic_twin.validation import calibration_loop as cl
from traffic_twin.validation import load_validation
from traffic_twin.validation.calibration_loop import (
    calibrate,
    calibration_arms,
    window_slot,
    with_scale,
    write_log,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = {"N": 626.0, "S": 366.0, "W": 206.0}  # demand at scale 1 (veh/h)
REAL = {"N": 610.0, "S": 370.0, "W": 210.0}


def linear(k: dict[str, float]):
    """Loop flow = demand x k (an arm losing or gaining traffic before the loop)."""
    calls = []

    def simulate(scale):
        calls.append(dict(scale))
        return {a: BASE[a] * scale[a] * k[a] for a in BASE}

    return simulate, calls


def test_converges_within_two_updates():
    simulate, calls = linear({"N": 0.8, "S": 1.25, "W": 1.1})
    res = calibrate(
        simulate, REAL, BASE, {"N": 1.0, "S": 1.0, "W": 1.0}, hash_of=lambda s: f"h{s['N']:.3f}"
    )
    assert res.status == "converged" and res.iterations <= 2 and not res.frozen
    for a in BASE:
        assert BASE[a] * res.scale[a] * {"N": 0.8, "S": 1.25, "W": 1.1}[a] == pytest.approx(
            REAL[a], rel=0.05
        )
    assert calls[0] == {"N": 1.0, "S": 1.0, "W": 1.0}  # starts from the config's scale
    last = [r for r in res.rows if r["iteration"] == res.iterations]
    assert all(r["action"] == "within tolerance" for r in last)
    assert res.rows[0]["config_hash"] == "h1.000" and res.rows[0]["action"].startswith("scale x")


def test_already_calibrated_needs_no_update():
    simulate, _ = linear({"N": 1.0, "S": 1.0, "W": 1.0})
    res = calibrate(simulate, BASE, BASE, {})
    assert (res.status, res.iterations, res.scale) == (
        "converged",
        0,
        {"N": 1.0, "S": 1.0, "W": 1.0},
    )


def test_capacity_limited_arm_is_frozen_at_its_last_responsive_scale():
    cap_w = 160.0  # W cannot carry more than 160 veh/h, the sensor says 210

    def simulate(scale):
        return {
            "N": BASE["N"] * scale["N"],
            "S": BASE["S"] * scale["S"] * 0.9,
            "W": min(BASE["W"] * scale["W"], cap_w),
        }

    res = calibrate(simulate, REAL, BASE, {"N": 1.0, "S": 1.0, "W": 1.0})
    assert res.status == "stopped: capacity-limited"
    assert list(res.frozen) == ["W"] and "slope 0.00" in res.frozen["W"]
    assert res.scale["W"] == 1.0  # never inflated to hide the missing capacity
    w = [r for r in res.rows if r["arm"] == "W"]
    assert w[1]["slope"] == 0.0 and w[1]["action"].startswith("frozen, back to scale 1.000")
    assert abs(BASE["S"] * res.scale["S"] * 0.9 / REAL["S"] - 1) <= 0.05


def test_partial_response_then_plateau_reverts_to_the_responsive_scale():
    def simulate(scale):  # W carries 60 % of its demand up to 1.5x, then stalls
        return {"W": BASE["W"] * min(scale["W"], 1.5) * 0.6}

    res = calibrate(simulate, {"W": 210.0}, BASE, {"W": 1.0}, clip=(0.5, 1.3))
    w = res.rows
    assert [r["scale"] for r in w] == [1.0, 1.3, 1.69]
    assert w[1]["slope"] == pytest.approx(0.6, abs=0.001)  # 1.0 -> 1.3: responds
    assert w[2]["slope"] < 0.5 and w[2]["action"] == "frozen, back to scale 1.300"
    assert res.status == "stopped: capacity-limited" and res.scale == {"W": 1.3}


def test_iteration_limit():
    def simulate(scale):  # concave response: needs several updates
        return {"N": BASE["N"] * scale["N"] ** 0.8}

    res = calibrate(simulate, {"N": 2 * BASE["N"]}, BASE, {"N": 1.0}, max_iter=1)
    assert res.status == "stopped: max iterations" and res.iterations == 1
    assert res.rows[-1]["action"] == "max iterations reached"


def test_log_is_deterministic(tmp_path):
    simulate, _ = linear({"N": 0.8, "S": 1.25, "W": 1.1})
    res = calibrate(simulate, REAL, BASE, {"N": 1.0, "S": 1.0, "W": 1.0})
    header = {"scenario": "configs/pitie_fixed.yaml", "seeds": [101, 102, 103]}
    a = write_log(tmp_path / "a.yaml", header, res).read_bytes()
    b = write_log(tmp_path / "b.yaml", header, res).read_bytes()
    assert a == b
    doc = yaml.safe_load(a)["calibration"]
    assert doc["status"] == "converged" and doc["recommended_arm_scale"] == res.scale
    assert len(doc["rows"]) == 3 * (res.iterations + 1)
    assert "T" not in str(doc.get("date", ""))  # no wall-clock time anywhere


def test_repo_settings(pitie_cfg):
    vcfg = load_validation(REPO_ROOT / "configs/validation.yaml")
    assert calibration_arms(vcfg) == {"N": "N_in", "S": "S_in", "W": "W_in"}
    assert window_slot(vcfg, pitie_cfg)["name"] == "07:30-09:30"
    assert not set(vcfg["seeds"]["calibration"]) & set(vcfg["seeds"]["evaluation"])
    scaled = with_scale(pitie_cfg, {"W": 0.9})
    assert (
        scaled["demand"]["arm_scale"]["W"] == 0.9 and pitie_cfg["demand"]["arm_scale"]["W"] == 1.0
    )


def fake_measure(factor: dict[str, float], cap: dict[str, float] | None = None):
    """Stands in for the SUMO runs of `main`: loop flow = demand x factor, capped."""

    def measure(cfg, seeds, out_root, stations, slot):
        base = cl.base_demand(cfg, list(stations))
        scale = cfg["demand"]["arm_scale"]
        flows = {a: base[a] * scale[a] * factor.get(a, 1.0) for a in stations}
        return {a: min(v, (cap or {}).get(a, v)) for a, v in flows.items()}

    return measure


@pytest.mark.parametrize(
    ("factor", "cap", "message"),
    [
        ({"W": 1.0}, {"W": 140.0}, "capacity-limited (arm W): fix the network"),
        ({"N": 0.9}, None, "recommended arm_scale differs from demand.arm_scale (N 1.0 -> 1.111)"),
    ],
)
def test_main_exits_1_unless_calibrated(tmp_path, monkeypatch, capsys, factor, cap, message):
    """`tasks.py part3` stops on a non-zero exit: a capacity-limited arm or a scale still to
    copy into the config must stop it before compare and validate publish anything."""
    monkeypatch.setattr(cl, "measure", fake_measure(factor, cap))
    log = tmp_path / "log.yaml"
    with pytest.raises(SystemExit) as e:
        cl.main(["--log", str(log)])
    assert e.value.code == 1
    assert message in capsys.readouterr().err
    assert log.is_file()  # the log is still written, for the report


def test_main_exits_0_when_converged_with_the_configured_scale(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cl, "measure", fake_measure({}))
    cl.main(["--log", str(tmp_path / "log.yaml")])  # no SystemExit
    out = capsys.readouterr()
    assert "status: converged after 0 update(s)" in out.out and out.err == ""


def test_gate():
    done = cl.LoopResult("converged", {"N": 1.0, "S": 1.0})
    assert cl.gate(done, {"N": 1.0, "S": 1.0}) == []
    assert cl.gate(done, {"N": 1.0}) == []  # an arm absent from the config is at 1.0
    moved = cl.LoopResult("converged", {"N": 1.111, "S": 1.0})
    assert "N 1.0 -> 1.111" in cl.gate(moved, {"N": 1.0, "S": 1.0})[0]
    stuck = cl.LoopResult("stopped: max iterations", {"N": 1.0})
    assert cl.gate(stuck, {"N": 1.0})[0].startswith("stopped: max iterations")
