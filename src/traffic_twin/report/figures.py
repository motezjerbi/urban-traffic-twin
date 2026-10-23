"""Report-only figures (docs/figures/report/), drawn from the facts.

- `pipeline.png`: the six layers of the twin (a diagram, no result);
- `person_delay.png`: the mean delay per person per controller on seeds 1-10. S0-S2 come
  from the evaluation; the S3 bar is drawn only from the Part 8 outputs (the final table),
  otherwise a PENDING note stands in its place.

PNG files are written without a timestamp, so a rebuild from the same facts gives the
same bytes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPORT_FIGURES = Path("docs/figures/report")
INK, MUTED, TEAL, TEAL_L, AMBER, GREY = (
    "#1F2A33",
    "#5B6770",
    "#0B6E69",
    "#5FB8B0",
    "#B57F00",
    "#A9B6BE",
)
DPI = 150
LAYERS = (
    ("Data", "Paris permanent\nsensors"),
    ("Simulation", "SUMO, calibrated\ndemand"),
    ("Control", "S0-S3 behind\none Safety Guard"),
    ("Prediction", "ETA with a\nconformal interval"),
    ("API", "FastAPI on\nprecomputed runs"),
    ("App", "five profiles,\nreplay, operator"),
)
PNG_META = {"Software": None}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata=PNG_META)
    plt.close(fig)
    return path


def pipeline(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (head, body) in enumerate(LAYERS):
        x = i + 0.06
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.12),
                0.8,
                0.76,
                boxstyle="round,pad=0.02",
                fc=INK if i % 2 == 0 else "#24333F",
                ec="none",
            )
        )
        ax.text(
            x + 0.4, 0.68, head, ha="center", va="center", color="white", fontsize=11, weight="bold"
        )
        ax.text(x + 0.4, 0.36, body, ha="center", va="center", color=TEAL_L, fontsize=8.5)
        if i < len(LAYERS) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.82, 0.5),
                    (x + 1.04, 0.5),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=AMBER,
                    lw=1.6,
                )
            )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return _save(fig, out / "pipeline.png")


def _value(facts: dict, key: str):
    f = facts.get(key)
    return None if f is None else f.get("value")


def person_delay(facts: dict, out: Path) -> Path:
    labels = ["S0", "S1", "S2", "S3"]
    names = {"S0": "S0 fixed", "S1": "S1 actuated", "S2": "S2 rules", "S3": "S3 RL"}
    vals = {}
    for lb in labels:
        v = _value(facts, f"final.all.person_delay_mean.{lb.lower()}")
        if v is None and lb != "S3":
            v = _value(facts, f"eval.person_delay.{lb.lower()}")
        if v is not None:
            vals[lb] = float(v)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = range(len(labels))
    for i, lb in zip(xs, labels, strict=True):
        if lb in vals:
            ax.bar(
                i,
                vals[lb],
                color=TEAL if lb == "S2" else (TEAL_L if lb == "S3" else GREY),
                width=0.62,
            )
            ax.text(
                i,
                vals[lb],
                f"{vals[lb]:.1f} s",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
                weight="bold",
            )
        else:
            ax.text(
                i,
                0,
                "PENDING\n(Part 8 run)",
                ha="center",
                va="bottom",
                fontsize=9,
                color=AMBER,
                weight="bold",
            )
    ax.set_xticks(list(xs), [names[lb] for lb in labels])
    ax.set_ylabel("Mean delay per person (s)")
    ax.spines[["top", "right"]].set_visible(False)
    top = max(vals.values()) if vals else 1.0
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis="y", color="#E1E6E9", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return _save(fig, out / "person_delay.png")


def make_figures(facts: dict, out: Path = REPORT_FIGURES) -> list[Path]:
    """The figures written."""
    out = Path(out)
    return [pipeline(out), person_delay(facts, out)]
