"""Printable field sheets for the site visit (docs/field_protocol.md), as one A4 PDF.

Every name on the sheets (signal heads, approaches, crossings, bus lines, stages) comes
from the scenario config, and the site map is drawn from the SUMO network, so the sheets
follow the model and the columns match `data/field/pitie_measurements.yaml`.

    python -m traffic_twin.validation.field_sheets --scenario configs/pitie_fixed.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from traffic_twin.config import load_scenario  # noqa: E402
from traffic_twin.paths import FIELD_FILE, PITIE_FIXED_CONFIG  # noqa: E402

log = logging.getLogger(__name__)

A4 = (8.27, 11.69)  # portrait, inches
A4_LANDSCAPE = (11.69, 8.27)
MARGIN = 0.06  # figure fraction
INK, GREY, LIGHT = "#111111", "#666666", "#e6e6e6"
DEFAULT_DATES = "Tue 3, Wed 4 or Thu 5 Nov 2026"
DEFAULT_OUT = Path("docs/field/pitie_field_sheets.pdf")
MOVEMENTS = ("N_S", "N_W", "S_N", "S_W", "W_N", "W_S")
BINS = tuple(f"{h:02d}:{m:02d}" for h in (7, 8, 9) for m in (0, 15, 30, 45))[2:-2]  # 07:30-09:15
CYCLES = 10
CYCLES_PER_PAGE = 3


def _page(pdf: PdfPages, title: str, subtitle: str, landscape: bool = False):
    fig = plt.figure(figsize=A4_LANDSCAPE if landscape else A4)
    fig.text(MARGIN, 1 - MARGIN * 0.75, title, fontsize=15, weight="bold", color=INK, va="top")
    wrapped = textwrap.fill(subtitle, 150 if landscape else 105)
    fig.text(MARGIN, 1 - MARGIN * 1.3, wrapped, fontsize=8.5, color=GREY, va="top")
    footer = "Pitié-Salpêtrière field visit · fill in data/field/pitie_measurements.yaml after"
    fig.text(MARGIN, MARGIN * 0.4, footer, fontsize=7, color=GREY)
    fig.text(1 - MARGIN, MARGIN * 0.4, f"page {pdf.get_pagecount() + 1}", fontsize=7,
             color=GREY, ha="right")  # fmt: skip
    return fig


def _table(
    fig,
    top: float,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    row_h: float = 0.022,
    font: float = 7.5,
    height_rows: int | None = None,
) -> float:
    """Ruled table in figure coordinates from `top` down; returns the bottom y."""
    x0, x1 = MARGIN, 1 - MARGIN
    total = sum(widths)
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + (x1 - x0) * w / total)
    n = max(len(rows), height_rows or 0)
    y = top
    fig.patches.append(plt.Rectangle((x0, y - row_h), x1 - x0, row_h, transform=fig.transFigure,
                                     color=LIGHT, zorder=0))  # fmt: skip
    for x, h in zip(xs, headers, strict=False):
        fig.text(x + 0.004, y - row_h / 2, h, fontsize=font, weight="bold", va="center")
    for i in range(n + 1):
        yy = y - row_h * (i + 1)
        fig.add_artist(plt.Line2D([x0, x1], [yy, yy], transform=fig.transFigure, lw=0.4,
                                  color=GREY))  # fmt: skip
        if i < n and i < len(rows):
            for x, cell in zip(xs, rows[i], strict=False):
                fig.text(x + 0.004, yy - row_h / 2, cell, fontsize=font, va="center")
    bottom = y - row_h * (n + 1)
    fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure, lw=0.6, color=INK))
    for x in xs:
        fig.add_artist(plt.Line2D([x, x], [y, bottom], transform=fig.transFigure, lw=0.4,
                                  color=GREY))  # fmt: skip
    return bottom


def _lines(fig, top: float, items: list[str], font: float = 8.5, step: float = 0.019) -> float:
    for i, s in enumerate(items):
        fig.text(MARGIN, top - i * step, s, fontsize=font, va="top", wrap=True)
    return top - len(items) * step


def page_numbers() -> dict[str, str]:
    """Page (or page range) of each sheet, in the order `build` writes them."""
    timing = math.ceil(CYCLES / CYCLES_PER_PAGE)
    order = [("cover", 1), ("map", 1), ("timing", timing), ("counts", 1), ("peds", 1),
             ("queues", 1), ("geometry", 1), ("buses", 1), ("transcription", 1)]  # fmt: skip
    pages, n = {}, 1
    for name, k in order:
        pages[name] = str(n) if k == 1 else f"{n}-{n + k - 1}"
        n += k
    return pages


def cover(pdf: PdfPages, cfg: dict, dates: str) -> None:
    pg = page_numbers()
    net = cfg["network"]
    fig = _page(pdf, "Field visit kit: Bd de l'Hôpital × Bd Saint-Marcel (Paris 13e)",
                f"Model TLS {net['tls_id']} · {dates}, 07:30-09:30 · print single-sided, "
                "bring 2 pens and a clipboard")  # fmt: skip
    y = _lines(fig, 0.9, [
        "WHY: the signal program, lane use and turn shares of the model are ASSUMED. This "
        "visit measures them and gives",
        "the independent checks of spec 13.4 (cycle, queues, turning counts). Never write an "
        "estimate: leave a box empty.",
    ])  # fmt: skip
    fig.text(MARGIN, y - 0.01, "Before you go", fontsize=11, weight="bold", va="top")
    y = _lines(fig, y - 0.035, [
        "[ ]  Check it is a school-term Tue/Wed/Thu, no strike or roadworks announced (Bd de "
        "l'Hôpital, RATP 24/57/91)",
        "[ ]  2 phones (charged, 30+ GB free) on small tripods or clamps, power bank, "
        "tally counter or counting app",
        "[ ]  This kit printed; a watch or phone showing seconds; a measuring wheel or 10 m "
        "tape for distances (optional)",
    ])  # fmt: skip
    fig.text(MARGIN, y - 0.01, "On site, in this order", fontsize=11, weight="bold", va="top")
    ref = net.get("reference_group", "N_straight")
    y = _lines(fig, y - 0.035, [
        f"07:10  Arrive. Set up camera A and B (site map, page {pg['map']}). Film your "
        "phone clock for 5 s at the start of each video:",
        "           every time on these sheets is PHONE time; write each camera's offset "
        "in seconds on the session box below.",
        f"07:20  Lane use and crossings questionnaire (page {pg['geometry']}), while "
        "traffic builds up.",
        f"07:30  Start the 15-min counts (pages {pg['counts']}, {pg['peds']}) and keep "
        "them going to 09:30 (or count later from the video).",
        f"08:00  Signal timing: at least 5 consecutive cycles, ideally 10 (pages "
        f"{pg['timing']}). Reference head: {ref}.",
        f"08:00  Bus arrivals and departures at Saint-Marcel - La Pitié (page "
        f"{pg['buses']}), until 09:00.",
        f"During  Queues: 2-3 times per approach N, S, W, at the END of red (page "
        f"{pg['queues']}). Not before 07:35.",
        "09:30  Stop. Photograph every sheet. Copy the videos to the computer the same day.",
    ])  # fmt: skip
    fig.text(MARGIN, y - 0.01, "Session", fontsize=11, weight="bold", va="top")
    _table(fig, y - 0.035, ["Field", "Value"], [
        ["date (YYYY-MM-DD)", ""], ["observer(s)", ""], ["weather", ""],
        ["camera A position / video file", ""], ["camera B position / video file", ""],
        ["camera offset s (camera - phone)", ""], ["incidents, roadworks, notes", ""],
    ], [1.2, 3], row_h=0.026, font=8)  # fmt: skip
    _lines(fig, 0.17, [
        "AFTER THE VISIT (docs/field_protocol.md): fill data/field/pitie_measurements.yaml, "
        "run `tasks.py field-check`",
        "until clean, re-download the sensors within a week (`tasks.py data calib-data`), "
        "then `tasks.py part3 part4`.",
        "Clock times are quoted strings in the YAML: \"08:15:20\".",
    ], font=8)  # fmt: skip
    pdf.savefig(fig)
    plt.close(fig)


def site_map(pdf: PdfPages, cfg: dict) -> None:
    import sumolib

    net_cfg = cfg["network"]
    net = sumolib.net.readNet(net_cfg["net_file"], withInternal=True,
                              withPedestrianConnections=True)  # fmt: skip
    fig = _page(pdf, "Site map (from the model network)",
                "Approaches N, S, W carry general traffic; NB and WB are bus-only roads. "
                "Crossings as in the model; camera spots are suggestions: check both "
                "on site.")  # fmt: skip
    ax = fig.add_axes([MARGIN, 0.12, 1 - 2 * MARGIN, 0.76])
    ax.set_aspect("equal")
    ax.axis("off")
    cx, cy = net.getNode(net_cfg["tls_id"]).getCoord()
    span = 120
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span * 1.05)
    bus_edges = {e for a in net_cfg["approaches"].values() if a.get("bus_only")
                 for e in (*a["in_edges"], *a["out_edges"])}  # fmt: skip
    for e in net.getEdges(withInternal=False):
        xs, ys = zip(*e.getShape(), strict=True)
        w = 3.2 * len(e.getLanes())
        ax.plot(xs, ys, color="#c9c9c9" if e.getID() not in bus_edges else "#f2d0b8",
                lw=w, solid_capstyle="butt", zorder=1)  # fmt: skip
        ax.plot(xs, ys, color=GREY, lw=0.4, zorder=2)
    jx, jy = zip(*net.getNode(net_cfg["tls_id"]).getShape(), strict=True)
    ax.fill(jx, jy, color="#c9c9c9", zorder=1)
    for name, c in net_cfg["crossings"].items():
        e = net.getEdge(c["edge"])
        xs, ys = zip(*e.getShape(), strict=True)
        ax.plot(xs, ys, color=INK, lw=4, ls=(0, (1, 1)), zorder=3)
        ax.annotate(f"crossing {name}", (sum(xs) / 2, sum(ys) / 2), xytext=(10, -12),
                    textcoords="offset points", fontsize=8, weight="bold",
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.5})  # fmt: skip
    for name, a in net_cfg["approaches"].items():
        e = net.getEdge(a["in_edges"][0])
        (x0, y0), (x1, y1) = e.getShape()[0], e.getShape()[-1]
        t = min(1.0, 70 / max(e.getLength(), 1))  # label ~70 m before the stop line
        lx, ly = x1 + (x0 - x1) * t, y1 + (y0 - y1) * t
        label = f"{name} in" + (" (bus only)" if a.get("bus_only") else "")
        ax.text(lx, ly, label, fontsize=9, weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round", "fc": "white", "ec": GREY, "lw": 0.5})  # fmt: skip
    for sid, s in net_cfg.get("bus_stops", {}).items():
        lane = net.getLane(s["lane"])
        x, y = sumolib.geomhelper.positionAtShapeOffset(
            lane.getShape(), (s["start"] + s["end"]) / 2
        )
        ax.plot(x, y, marker="s", ms=6, color="#c0392b", zorder=4)
        ax.annotate(sid, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7,
                    color="#c0392b")  # fmt: skip
    for tag, (dx, dy), what in (
        ("A", (-30, 34), "N approach stop line + heads N, W, ped_S"),
        ("B", (-6, -48), "S approach stop line + heads S, bus, ped_W"),
    ):
        ax.plot(cx + dx, cy + dy, marker="^", ms=12, color=INK, zorder=5)
        ax.text(cx + dx + 5, cy + dy + 3, f"camera {tag}: {what}", fontsize=8, weight="bold")
    ax.plot([cx - span + 10, cx - span + 60], [cy - span + 10] * 2, color=INK, lw=2)
    ax.text(cx - span + 35, cy - span + 15, "50 m", ha="center", fontsize=8)
    ax.annotate("N", (cx + span - 12, cy + span - 5), xytext=(cx + span - 12, cy + span - 30),
                ha="center", fontsize=11, weight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.5})  # fmt: skip
    stages = " → ".join(s["name"] for s in net_cfg["stages"])
    greens = ", ".join(f"{k} {v}" for k, v in cfg["fixed"]["green_s"].items())
    legend = (
        f"Model stage order (ASSUMED, check on site): {stages}; greens (s): {greens}, plus "
        f"clearances. Signal heads: {', '.join(net_cfg['signal_groups'])}."
    )
    fig.text(MARGIN, 0.105, textwrap.fill(legend, 120), fontsize=8, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def timing_sheets(pdf: PdfPages, cfg: dict) -> None:
    heads = list(cfg["network"]["signal_groups"])
    ref = cfg["network"].get("reference_group", heads[0])
    heads = [ref, *[h for h in heads if h != ref]]
    stages = [s["name"] for s in cfg["network"]["stages"]]
    for first in range(1, CYCLES + 1, CYCLES_PER_PAGE):
        last = min(CYCLES, first + CYCLES_PER_PAGE - 1)
        fig = _page(pdf, f"Signal timing · cycles {first}-{last}",
                    f"Times to the second (HH:MM:SS, phone time). A cycle starts at the green "
                    f"of {ref}. Pedestrian heads: 'yellow' = start of flashing. A head green "
                    "twice in a cycle: two lines.")  # fmt: skip
        y = 0.875
        for c in range(first, last + 1):
            fig.text(MARGIN, y, f"Cycle {c}    stage order seen: "
                     + "  ".join(f"{s} ___" for s in stages) + "    other: ______",
                     fontsize=8, weight="bold", va="bottom")  # fmt: skip
            rows = [[h, "", "", "", ""] for h in heads]
            y = _table(fig, y - 0.004, ["Head", "Green start", "Yellow / flash start",
                                        "Red start", "Note (2nd green, skipped...)"], rows,
                       [1.1, 1, 1.1, 1, 2], row_h=0.0165, font=7)  # fmt: skip
            y -= 0.03
        pdf.savefig(fig)
        plt.close(fig)


def count_sheets(pdf: PdfPages, cfg: dict) -> None:
    lines = [b["id"] for b in cfg["demand"]["bus"]["lines"]]
    bins = [f"{b}-{_add15(b)}" for b in BINS]
    fig = _page(pdf, "Turning counts, 15-min bins (vehicles, cars + vans + trucks)",
                "Count each vehicle once as it leaves the junction. Buses and two-wheelers in "
                "their own columns, not in the movements. Tally marks are fine.",
                landscape=True)  # fmt: skip
    headers = ["Bin", *[m.replace("_", "→") for m in MOVEMENTS], *lines, "2-wheel"]
    _table(fig, 0.88, headers, [[b] + [""] * (len(headers) - 1) for b in bins],
           [1.3] + [1.2] * len(MOVEMENTS) + [0.8] * len(lines) + [0.9], row_h=0.08,
           font=8)  # fmt: skip
    pdf.savefig(fig)
    plt.close(fig)

    crossings = list(cfg["network"]["crossings"])
    fig = _page(pdf, "Pedestrians and context flows, 15-min bins",
                "Pedestrians: people starting to cross, both directions together (SHOULD: at "
                "least 2 bins). Poliveau: vehicles entering from Rue Poliveau; Wallons: right "
                "turns into Rue des Wallons.", landscape=True)  # fmt: skip
    headers = ["Bin", *[f"ped {c}" for c in crossings], "Poliveau in", "Wallons right", "Note"]
    _table(fig, 0.88, headers, [[b] + [""] * (len(headers) - 1) for b in bins],
           [1.3] + [1] * len(crossings) + [1.1, 1.1, 2], row_h=0.08, font=8)  # fmt: skip
    pdf.savefig(fig)
    plt.close(fig)


def _add15(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    m += 15
    return f"{h + m // 60:02d}:{m % 60:02d}"


def queue_sheet(pdf: PdfPages, cfg: dict) -> None:
    approaches = [a for a, v in cfg["network"]["approaches"].items() if not v.get("bus_only")]
    fig = _page(pdf, "Queues at the end of red (cycle_max)",
                "2-3 instants per approach, 07:35-09:30. Count queued vehicles at the moment the "
                "light turns green; note how far you can see (m) and which lanes.")  # fmt: skip
    rows = [[a, "", "", "", "all", ""] for a in approaches for _ in range(3)]
    y = _table(fig, 0.9, ["Approach", "Time HH:MM:SS", "Vehicles", "Visible m", "Lanes",
                          "Note"], rows, [1, 1.3, 1, 1, 1, 3], row_h=0.03, font=8)  # fmt: skip
    fig.text(MARGIN, y - 0.03, "Saturation headways (NICE): N straight lane, 3-5 cycles, "
             "time each vehicle crossing the stop line from the 4th after green.", fontsize=8.5,
             weight="bold")  # fmt: skip
    _table(fig, y - 0.045, ["Approach / lane", "Cycle", "Crossing times from vehicle 4 (s)"],
           [["N / lane 2", str(i), ""] for i in range(1, 6)], [1.2, 0.6, 5], row_h=0.03,
           font=8)  # fmt: skip
    pdf.savefig(fig)
    plt.close(fig)


def geometry_sheet(pdf: PdfPages, cfg: dict) -> None:
    fig = _page(
        pdf,
        "Lane use, crossings, loops, ambulance",
        "Answers that change the network. Tick or write; leave blank if unsure.",
    )
    y = _table(fig, 0.9, ["Question", "Answer"], [
        ["Lanes at the N stop line, with arrows (e.g. 2: right | straight)", ""],
        ["Lanes at the S stop line, with arrows", ""],
        ["Lanes at the W (Saint-Marcel) stop line, with arrows", ""],
        ["S approach: length of the bus lane before the stop line (m)", ""],
        ["S approach: do cars use the right lane to go straight?   yes / no", ""],
        ["S → W left turn allowed?   yes / no      protected (own arrow)?   yes / no", ""],
        ["Crossing over Bd de l'Hôpital N at the junction itself?   yes / no", ""],
        ["Pedestrian-only stage (all cars red, peds green)?   yes / no", ""],
        ["Bus stages skipped when no bus?   yes / no / unclear", ""],
        ["Greens change from cycle to cycle (actuated)?   yes / no", ""],
    ], [5, 2.2], row_h=0.03, font=8)  # fmt: skip
    rows = [[c, "", "", "", ""] for c in cfg["network"]["crossings"]]
    y = _table(fig, y - 0.03, ["Crossing", "Signalised? (y/n)", "Distance to junction m",
                               "Length m", "Refuge island? (y/n)"], rows, [1, 1.3, 1.6, 1, 1.5],
               row_h=0.03, font=8)  # fmt: skip
    y = _table(fig, y - 0.03, ["Loop (saw cut) approach", "Lanes", "Distance to stop line m",
                               "Note"], [["", "", "", ""] for _ in range(4)], [1.6, 1, 1.6, 2.4],
               row_h=0.03, font=8)  # fmt: skip
    _table(fig, y - 0.03, ["Ambulance", "Answer"], [
        ["Route seen (from which arm to which arm)", ""],
        ["Hospital gate used (main gate 83 bd de l'Hôpital? other?)", ""],
    ], [5, 2.2], row_h=0.03, font=8)  # fmt: skip
    pdf.savefig(fig)
    plt.close(fig)


def bus_sheet(pdf: PdfPages, cfg: dict) -> None:
    lines = ", ".join(b["id"] for b in cfg["demand"]["bus"]["lines"])
    fig = _page(pdf, "Buses at Saint-Marcel - La Pitié (dwell), 08:00-09:00",
                f"One line per bus. Line ids as in the model: {lines}. Arrive = doors open, "
                "depart = bus moves off.")  # fmt: skip
    _table(fig, 0.9, ["Line", "Direction (E/W/N/S)", "Arrive HH:MM:SS", "Depart HH:MM:SS",
                      "Note"], [["", "", "", "", ""] for _ in range(28)],
           [0.8, 1.2, 1.4, 1.4, 3], row_h=0.028, font=8)  # fmt: skip
    pdf.savefig(fig)
    plt.close(fig)


def transcription_sheet(pdf: PdfPages, field_file: Path) -> None:
    pg = page_numbers()
    fig = _page(pdf, f"After the visit: sheet → {field_file.as_posix()}",
                "Keep the sheet photos with the videos. Run `tasks.py field-check` after each "
                "section you fill.")  # fmt: skip
    rows = [
        (
            "Session box",
            "cover",
            "sessions[0]",
            "date, observer, positions, video, camera_offset_s",
        ),
        ("Timing", "timing", "signal", "cycle_s = gaps between reference green starts"),
        ("", "", "", "per head: green_s = yellow - green; yellow_s = red - yellow"),
        ("", "", "", "all_red_after_s = red to the next vehicle green"),
        ("", "", "", "ped_NB green twice: green_s {HOP: [...], STM: [...]}"),
        ("", "", "", "stage_order, varies_between_cycles, bus_stages_on_demand, ped_only_stage"),
        ("Counts", "counts", "counts[].movements, buses", "one item per 15-min bin: begin, end"),
        ("Peds", "peds", "counts[].pedestrians", "S, W, WB, NB; poliveau_in, wallons_right"),
        ("Queues", "queues", "queues", "time, approach, vehicles, visible_m, lanes"),
        ("Headways", "queues", "saturation", "headways_s list"),
        ("Lane use", "geometry", "geometry, loops, ambulance", "lanes_at_stop_line, crossings.*"),
        ("Buses", "buses", "bus_stops", "line, direction, arrive, depart"),
    ]
    cells = [[f"{s} (p. {pg[k]})" if s else "", y, how] for s, k, y, how in rows]
    _table(fig, 0.9, ["Sheet", "YAML section", "How"], cells, [1.4, 1.8, 5], row_h=0.032,
           font=7.5)  # fmt: skip
    pdf.savefig(fig)
    plt.close(fig)


def build(cfg: dict, out: Path, dates: str = DEFAULT_DATES, field_file: Path = FIELD_FILE) -> int:
    """Write the sheets to `out`; returns the page count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "Pitié-Salpêtrière field sheets", "Creator": "traffic_twin"}
    with PdfPages(out, metadata={**metadata, "CreationDate": None}) as pdf:
        cover(pdf, cfg, dates)
        site_map(pdf, cfg)
        timing_sheets(pdf, cfg)
        count_sheets(pdf, cfg)
        queue_sheet(pdf, cfg)
        geometry_sheet(pdf, cfg)
        bus_sheet(pdf, cfg)
        transcription_sheet(pdf, field_file)
        pages = pdf.get_pagecount()
    log.info("field sheets: %s (%d pages)", out, pages)
    return pages


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", type=Path, default=PITIE_FIXED_CONFIG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dates", default=DEFAULT_DATES, help="visit dates printed on the cover")
    a = p.parse_args(argv)
    build(load_scenario(a.scenario), a.out, a.dates)


if __name__ == "__main__":
    main()
