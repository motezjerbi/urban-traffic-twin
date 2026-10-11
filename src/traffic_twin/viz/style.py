"""One figure style for the whole project: colours per user type and per controller, ink
and grid tones, fonts, and a `savefig` that writes README-ready PNGs.

Colour rules (validated for colour-vision deficiency, all pairs, on a white surface):

- a user type keeps its colour in every figure: car blue, bus orange, ambulance violet,
  pedestrian aqua;
- a controller keeps its colour in every comparison figure: fixed violet, actuated
  yellow, rule-based magenta, RL green. When controllers are compared, user types go
  on the axes or in facets, so the two encodings never share one plot;
- text is never drawn in a series colour: labels use the ink tones, a coloured mark
  next to them carries the identity;
- `CRITICAL` is reserved for spec targets and limits (e.g. pedestrian max wait 90 s),
  always drawn with a text label.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to files, never shown
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure, SubFigure  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

# -- surfaces and ink ------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"  # titles, values
INK_2 = "#52514e"  # axis labels, secondary text
MUTED = "#898781"  # ticks, notes
GRID = "#e1e0d9"  # hairline gridlines
AXIS = "#c3c2b7"  # baseline / spines
CRITICAL = "#d03b3b"  # spec targets and limits (always with a label)

# -- categorical palettes (fixed order, never cycled) ----------------------------------
USER_TYPES = ("car", "bus", "ambulance", "pedestrian")
USER_LABELS = {
    "car": "Car / student",
    "bus": "Bus",
    "ambulance": "Ambulance",
    "pedestrian": "Pedestrian",
}
USER_COLORS = {
    "car": "#2a78d6",
    "bus": "#eb6834",
    "ambulance": "#4a3aa7",
    "pedestrian": "#1baf7a",
}

CONTROLLERS = ("fixed", "actuated", "rule_based", "rl")
CONTROLLER_LABELS = {
    "fixed": "Fixed-time",
    "actuated": "Actuated",
    "rule_based": "Rule-based",
    "rules": "Rule-based",
    "rl": "RL",
}
CONTROLLER_COLORS = {
    "fixed": "#4a3aa7",
    "actuated": "#eda100",
    "rule_based": "#e87ba4",
    "rules": "#e87ba4",
    "rl": "#008300",
}
SERIES = ("#4a3aa7", "#eda100", "#e87ba4", "#008300")  # fallback order for other series

# Light ramps for signal-stage shading: E-W stages in blue, N-S stages in orange,
# clearance (yellow, all-red, pedestrian clearance) in grey.
STAGE_FILL = {"EW": "#b7d3f6", "EW_LEFT": "#6da7ec", "NS": "#f8cdb8", "NS_LEFT": "#f0956b"}
STAGE_FILL_OTHER = ("#cde2fb", "#86b6ef", "#fbe0d2", "#f4ae8c")
CLEARANCE_FILL = "#ecebe6"

# -- sizes -----------------------------------------------------------------------------
DPI = 170
WIDTH_IN = 11.0  # full README width
FONT_SIZE = 10
LINE_W = 2.0

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": FONT_SIZE,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "lines.linewidth": LINE_W,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": FONT_SIZE - 1,
    "legend.labelcolor": INK_2,
    "legend.handlelength": 1.4,
    "figure.titlesize": FONT_SIZE + 3,
    "figure.titleweight": "semibold",
}


def apply_style() -> None:
    """Set the project rcParams (idempotent)."""
    plt.rcParams.update(RC)


def user_color(user: str) -> str:
    return USER_COLORS.get(user, INK_2)


def user_label(user: str) -> str:
    return USER_LABELS.get(user, user.replace("_", " ").capitalize())


def controller_label(controller: str) -> str:
    return CONTROLLER_LABELS.get(controller, controller.replace("_", " ").capitalize())


def series_colors(controllers: Iterable[str]) -> list[str]:
    """One colour per series: the controller's own colour, or the next free palette slot
    when a controller is unknown or appears twice (e.g. two RL variants)."""
    out: list[str] = []
    for c in controllers:
        color = CONTROLLER_COLORS.get(c)
        if color is None or color in out:
            color = next((s for s in SERIES if s not in out), MUTED)
        out.append(color)
    return out


def figure(
    width: float = WIDTH_IN, height: float = 3.6, header_in: float = 0.9, **subplots
) -> tuple[Figure, SubFigure, object]:
    """A styled figure: a header strip (title, subtitle, legend) above a body of
    subplots. Returns `(fig, header, axes)`; `subplots` go to `SubFigure.subplots`."""
    apply_style()
    fig = plt.figure(figsize=(width, height + header_in), layout="constrained")
    head, body = fig.subfigures(2, 1, height_ratios=[header_in, height], hspace=0.0)
    return fig, head, body.subplots(**subplots)


def header(
    head: SubFigure,
    text: str,
    subtitle: str | None = None,
    handles: list | None = None,
    ncols: int | None = None,
) -> None:
    """Left-aligned title, optional subtitle, and an optional one-row legend under them."""
    top = head.transSubfigure
    head.text(
        0.0, 1.0, text, transform=offset_copy(top, fig=head.figure, y=-2, units="points"),
        ha="left", va="top", color=INK, fontsize=FONT_SIZE + 3, fontweight="semibold",
    )  # fmt: skip
    if subtitle:
        head.text(
            0.0, 1.0, subtitle,
            transform=offset_copy(top, fig=head.figure, y=-(FONT_SIZE + 9), units="points"),
            ha="left", va="top", color=INK_2, fontsize=FONT_SIZE,
        )  # fmt: skip
    if handles:
        head.legend(
            handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.14), borderaxespad=0.0,
            ncols=ncols or len(handles), handlelength=1.6, columnspacing=1.8, borderpad=0.0,
        )  # fmt: skip


def note(fig: Figure, text: str) -> None:
    """Source / method note under the plots, in muted ink."""
    fig.text(
        0.0, 0.0, text, transform=offset_copy(fig.transFigure, fig=fig, y=-6, units="points"),
        ha="left", va="top", color=MUTED, fontsize=FONT_SIZE - 1.5,
    )  # fmt: skip


def target_line(ax: Axes, value: float, label: str, axis: str = "y") -> None:
    """A spec target drawn as a thin critical line, labelled just outside the plot
    (right of a horizontal line, top of a vertical one) so it never sits on the data."""
    if axis == "y":
        ax.axhline(value, color=CRITICAL, lw=1.2, zorder=4)
        ax.annotate(
            label, (1.0, value), xycoords=("axes fraction", "data"), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center", color=CRITICAL,
            fontsize=FONT_SIZE - 1,
        )  # fmt: skip
    else:
        ax.axvline(value, color=CRITICAL, lw=1.2, zorder=4)
        ax.annotate(
            label, (value, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
            textcoords="offset points", ha="left", va="top", color=CRITICAL,
            fontsize=FONT_SIZE - 1,
        )  # fmt: skip


def savefig(fig: Figure, path: Path | str, dpi: int = DPI) -> Path:
    """Write a PNG (tight, white background) and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15, facecolor=SURFACE)
    plt.close(fig)
    return path
