"""data.explore end to end on a synthetic raw export (clean -> profile -> figure): the
curves sit on the hour their values cover, the labels come from the profile's own day
set, and a profile written before the period-start fix is refused."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from traffic_twin.data import clean, explore  # noqa: E402
from traffic_twin.paths import HOURLY_PROFILE_FILE  # noqa: E402

BASE = {"1": 600.0, "2": 300.0}
DAYS = pd.date_range("2026-08-17", "2026-09-20", freq="D")  # summer holidays, then term
CALENDAR = {
    "weekdays": [1, 2, 3],
    "typical_period": {"from": "2026-09-01", "to": "2026-12-18"},
    "covers": {"from": "2026-04-01", "to": "2027-01-31"},
    "school_holidays": [{"from": "2026-07-04", "to": "2026-08-31"}],
    "public_holidays": ["2026-08-15"],
}


def flow(arc: str, start: pd.Timestamp) -> float:
    """Synthetic flow of the hour starting at `start`: arc level + 10 per hour of day,
    doubled in the summer holidays (a stale profile would show it)."""
    q = BASE[arc] + 10 * start.hour
    return 2 * q if start < pd.Timestamp("2026-09-01", tz="Europe/Paris") else q


def raw_frame() -> pd.DataFrame:
    rows = []
    for arc in BASE:
        for day in DAYS:
            for h in range(24):
                start = pd.Timestamp(f"{day.date()} {h:02d}:00", tz="Europe/Paris")
                rows.append(
                    {"iu_ac": arc, "libelle": f"Rue_{arc}", "t_1h": start + pd.Timedelta(hours=1),
                     "q": flow(arc, start), "k": 5.0, "etat_trafic": "Fluide",
                     "etat_barre": "Ouvert", "geo_point_2d": {"lat": 48.84, "lon": 2.36}}
                )  # fmt: skip
    raw = pd.DataFrame(rows)
    raw["t_1h"] = pd.to_datetime(raw["t_1h"]).dt.tz_convert("UTC")
    return raw


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A data config, data/raw with the synthetic zone and `data.clean` already run."""
    (tmp_path / "raw").mkdir()
    raw_frame().to_parquet(tmp_path / "raw" / "z.parquet")
    cfg = {
        "source": {"base_url": "-", "dataset": "-", "months": 1},
        "zones": {"z": {"label": "Test zone", "lat": 48.84, "lon": 2.36, "radius_m": 100}},
        "cleaning": {"weekdays": [1, 2, 3], "timezone": "Europe/Paris",
                     "max_flow_veh_h": 4000, "max_occupancy_pct": 100},
        "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                  "figures": str(tmp_path / "figures")},
        "calendar": CALENDAR,
    }  # fmt: skip
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    clean.main(["--config", str(tmp_path / "data.yaml")])
    return tmp_path


@pytest.fixture
def figures(monkeypatch) -> list:
    """Capture the figures explore saves (still open, so their artists can be read)."""
    saved = []

    def keep(fig, path):
        saved.append(fig)
        return Path(path)

    monkeypatch.setattr(explore.style, "savefig", keep)
    return saved


def test_profile_figure_labels_and_hour_alignment(workspace, figures):
    explore.main(["--config", str(workspace / "data.yaml")])
    (fig,) = figures
    ax_q, ax_k, ax_w = fig.axes[:3]
    assert ax_q.get_title(loc="left") == "Flow, school-term Tue–Thu mean"
    assert ax_q.get_xlim() == (0, 24)
    # each hourly mean is a step over the hour it covers: [h, h + 1)
    line = next(ln for ln in ax_q.get_lines() if ln.get_label().endswith("(1)"))
    assert line.get_drawstyle() == "steps-post"
    x, y = line.get_xdata(), line.get_ydata()
    assert list(x) == list(range(25))
    assert [y[h] for h in (7, 8, 9)] == [670.0, 680.0, 690.0]  # term values, not summer x2
    # the 07:30-09:30 band covers half of the 07h and 09h steps
    band = ax_q.patches[0].get_bbox()  # data x, axes-fraction y
    assert (band.x0, band.x1) == (7.5, 9.5)
    # weekday vs weekend: both from the term (summer weekends left out), counted in the legend
    legend = [t.get_text() for t in ax_w.get_legend().get_texts()]
    assert legend == ["school-term Tue–Thu (~9 days)", "school-term Sat–Sun (~6 days)"]


def test_labels_come_from_the_profile_not_the_config(workspace, figures):
    """A profile built without the calendar must not be called school-term even if the
    config has a calendar now."""
    path = workspace / "processed" / HOURLY_PROFILE_FILE
    sensors = pd.read_parquet(workspace / "processed" / clean.SENSOR_HOURLY_FILE)
    clean.hourly_profile(sensors, None, [1, 2, 3]).to_parquet(path, index=False)
    explore.main(["--config", str(workspace / "data.yaml")])
    title = figures[0].axes[0].get_title(loc="left")
    assert title == "Flow, Tue–Thu, all dates mean" and "school-term" not in title


def test_legacy_profile_is_refused(workspace, figures):
    """A profile written before the period-start fix has an end-of-period `hour` column
    and no day-set label: explore stops instead of mislabelling it."""
    path = workspace / "processed" / HOURLY_PROFILE_FILE
    legacy = pd.read_parquet(path).drop(columns="day_set")
    legacy.rename(columns={"hour_start": "hour"}).to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="predates the period-start fix"):
        explore.main(["--config", str(workspace / "data.yaml")])
    assert figures == []
    pd.read_parquet(path).rename(columns={"hour": "hour_start"}).to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="day_set"):  # period start but no day-set label
        explore.main(["--config", str(workspace / "data.yaml")])
    assert not (workspace / "figures").exists()
