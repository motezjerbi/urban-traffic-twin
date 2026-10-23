"""The printable user-test kit, one A4 PDF (`docs/usertests/usertest_kit.pdf`, D111).

Built like the field sheets (`validation.field_sheets`, whose `_table` and `_lines`
helpers are reused read-only; the kit has its own `_page` and footer): matplotlib
PdfPages with `CreationDate` None, so the same inputs give the same bytes.

The kit freezes the campaign. It records the run (resolved by scenario and seed), its
run_id and the sha256 of its `signal.parquet`, on the cover and in the PDF `Keywords`,
and `usertest-serve` refuses to start when they no longer match. It refuses to build when
the run is missing, when an expected answer no longer holds at its instant, or when a
label T-OP needs (S3) is missing from `/kpi`; `allow_pending_s3` builds a draft whose
cover says it is not for use with participants (Phase 2 gate only).

Pages: cover, facilitator script, consent and privacy, task order and start links, task
cards, answer key, one observation sheet per participant, questionnaire, issue log.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from traffic_twin.usertests.answers import answer_text, expected_answers  # noqa: E402
from traffic_twin.usertests.protocol import (  # noqa: E402
    clock,
    deep_link,
    task_order,
    tasks_by_id,
)
from traffic_twin.validation.field_sheets import (  # noqa: E402  (read-only helpers)
    A4,
    A4_LANDSCAPE,
    GREY,
    INK,
    MARGIN,
    _lines,
    _table,
)

RED = "#b03a2e"
CARDS_PER_PAGE = 4
SUS_FR = (  # our own French wording of the 10 SUS items: indicative, not a validated F-SUS
    "Je pense que j'aimerais utiliser cette application régulièrement.",
    "J'ai trouvé l'application inutilement compliquée.",
    "J'ai trouvé l'application facile à utiliser.",
    "Je pense que j'aurais besoin d'aide technique pour pouvoir utiliser l'application.",
    "J'ai trouvé que les écrans de l'application allaient bien ensemble.",
    "J'ai trouvé l'application trop incohérente.",
    "J'imagine que la plupart des gens apprendraient très vite à utiliser l'application.",
    "J'ai trouvé l'application très lourde à utiliser.",
    "Je me suis senti(e) en confiance en utilisant l'application.",
    "J'ai dû apprendre beaucoup de choses avant de pouvoir utiliser l'application.",
)
KEYWORDS_RE = re.compile(rb"/Keywords \(([^)]*)\)")


class KitError(RuntimeError):
    """The kit cannot be built for a real campaign (missing run, a failing answer, S3)."""


def _page(pdf: PdfPages, title: str, subtitle: str, landscape: bool = False):
    fig = plt.figure(figsize=A4_LANDSCAPE if landscape else A4)
    fig.text(MARGIN, 1 - MARGIN * 0.75, title, fontsize=15, weight="bold", color=INK, va="top")
    wrapped = textwrap.fill(subtitle, 150 if landscape else 105)
    fig.text(MARGIN, 1 - MARGIN * 1.3, wrapped, fontsize=8.5, color=GREY, va="top")
    footer = ("User tests (spec 12.2) · données simulées · copy the sheets into "
              "docs/usertests/results.yaml the same day")  # fmt: skip
    fig.text(MARGIN, MARGIN * 0.4, footer, fontsize=7, color=GREY)
    fig.text(1 - MARGIN, MARGIN * 0.4, f"page {pdf.get_pagecount() + 1}", fontsize=7,
             color=GREY, ha="right")  # fmt: skip
    return fig


def _save(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def _wrap(text: str, width: int) -> str:
    return textwrap.fill(" ".join(text.split()), width)


def _para(
    fig, top: float, text: str, width: int = 95, font: float = 9, step: float = 0.019
) -> float:
    """A wrapped paragraph from `top` down (no matplotlib auto-wrap); returns the bottom."""
    return _lines(fig, top, _wrap(text, width).splitlines(), font=font, step=step)


def page_numbers(protocol: dict) -> dict[str, str]:
    """Page (or page range) of each part of the kit, in the order `build_kit` writes them."""
    n_cards = -(-len(protocol["tasks"]) // CARDS_PER_PAGE)
    order = [("cover", 1), ("script", 1), ("consent", 1), ("links", 1), ("cards", n_cards),
             ("key", 1), ("sheets", protocol["participants"]["max"]), ("questions", 1),
             ("issues", 1)]  # fmt: skip
    pages, n = {}, 1
    for name, k in order:
        pages[name] = str(n) if k == 1 else f"{n}-{n + k - 1}"
        n += k
    return pages


def _instant(task: dict) -> str:
    t = (task.get("start") or {}).get("t")
    return "none" if t is None else f"{clock(t)} (t = {t} s)"


# -- pages -----------------------------------------------------------------------------
def cover(pdf: PdfPages, protocol: dict, answers: dict, app_version: str, draft: bool) -> None:
    run, api = answers["run"], answers["api"]
    pg = page_numbers(protocol)
    fig = _page(pdf, "User-test kit: Urban Traffic Twin app (spec 12.2, A-10)",
                f"Protocol configs/usertests.yaml, registered {protocol['preregistered']}; "
                "narrative docs/usertest_protocol.md. Print single-sided. Participant texts "
                "are French, facilitator notes English.")  # fmt: skip
    y = 0.885
    if draft:
        fig.patches.append(plt.Rectangle((MARGIN, y - 0.06), 1 - 2 * MARGIN, 0.06,
                                         transform=fig.transFigure, fill=False, ec=RED,
                                         lw=1.5))  # fmt: skip
        pending = ", ".join(answers["operator"]["pending"])
        fig.text(MARGIN + 0.01, y - 0.012, f"DRAFT: {pending} not evaluated yet (Part 8).",
                 fontsize=12, weight="bold", color=RED, va="top")  # fmt: skip
        fig.text(MARGIN + 0.01, y - 0.035, "Not for use with participants: T-OP has no answer "
                 "key. Rebuild with `tasks.py usertest-kit` after Part 8.", fontsize=9,
                 color=RED, va="top")  # fmt: skip
        y -= 0.08
    fig.text(MARGIN, y, "Frozen for the whole campaign", fontsize=11, weight="bold", va="top")
    sha = run["signal_sha256"] or "absent"
    rows = [
        ["run (scenario, seed)", f"{run['scenario']} ({run.get('label') or '?'}), seed "
         f"{run['seed']}"],
        ["run_id", run["run_id"]],
        ["signal.parquet sha256", sha[:32]],
        ["", sha[32:]],
        ["run code version", str(run.get("code_version"))],
        ["app version (kit built at)", app_version],
        ["data", f"{api.get('data_source')}, calibrated on {api.get('calibrated_on')}, "
         f"signal program {'provisional (ASSUMED)' if api.get('provisional') else 'measured'}"],
        ["app URL", protocol["app_url"]],
    ]  # fmt: skip
    y = _table(fig, y - 0.022, ["Item", "Value"], rows, [1.3, 3.2], row_h=0.024, font=8)
    y = _lines(fig, y - 0.012, [
        "`tasks.py usertest-serve` refuses to start when the run_id or the signal sha256 "
        "differ from this page, or on a",
        "modified tree: the app and the run do not change between participants. Write the "
        "app version it prints on each sheet.",
    ], font=8)  # fmt: skip
    fig.text(MARGIN, y - 0.012, "Session plan (about "
             f"{protocol.get('session_minutes', 27)} min; book 30-min slots)", fontsize=11,
             weight="bold", va="top")  # fmt: skip
    n = len([t for t in protocol["tasks"] if not t.get("optional")])
    plan = [
        [f"Consent (page {pg['consent']})", "3 min", "read the text, give the P-id slip, "
         "tick oral consent"],
        [f"Introduction (page {pg['script']})", "2 min", "the FR lines word for word"],
        [f"{n} tasks (+ optional replay)", "15 min", "card, 'Allez-y', stopwatch, SEQ, "
         "20 s debrief"],
        ["Transparency question", "1 min", "real, simulated or don't know (ENF-05)"],
        ["SUS-10 (optional)", "3 min", f"indicative only (page {pg['questions']})"],
        ["Debrief", "3 min", "what helped, what got in the way"],
    ]  # fmt: skip
    y = _table(fig, y - 0.035, ["Step", "Time", "What"], plan, [1.4, 0.5, 3], row_h=0.022,
               font=8)  # fmt: skip
    fig.text(MARGIN, y - 0.012, "Before each session", fontsize=11, weight="bold", va="top")
    y = _lines(fig, y - 0.035, [
        "[ ]  `tasks.py usertest-serve`; Chrome full screen, zoom 100 %, other tabs closed",
        f"[ ]  Observation sheet of this participant (pages {pg['sheets']}), cards in the order "
        "printed on it, a phone stopwatch (0.1 s)",
        "[ ]  A second observer for the timing, if possible; no audio or video recording",
    ], font=8.5)  # fmt: skip
    fig.text(MARGIN, y - 0.012, "After each session (same day)", fontsize=11, weight="bold",
             va="top")  # fmt: skip
    _lines(fig, y - 0.035, [
        "Copy the sheet into docs/usertests/results.yaml (P-id only, what happened on the "
        "screen, never who the person is),",
        "run `tasks.py usertest-check` until clean, log the problems as issues U01..., then "
        "`tasks.py usertest-docs`.",
    ], font=8.5)  # fmt: skip
    _save(pdf, fig)


def script(pdf: PdfPages, protocol: dict) -> None:
    fig = _page(pdf, "Facilitator script",
                "English instructions; the French lines in quotes are read word for word. One "
                "scripted hint is allowed per task: it caps the outcome at partial.")  # fmt: skip
    tg, pg = protocol["targets"], page_numbers(protocol)
    items = [
        ("Setup", [
            "Start the app with `tasks.py usertest-serve`. Fill in the P-id, profile, real / "
            "proxy and the app version it prints on the participant's sheet.",
        ]),
        ("Introduction", [
            "« On teste l'application, pas vous. Les données affichées sont simulées. »",
            "« Répondez à voix haute dès que vous savez. Je ne peux pas vous aider pendant la "
            "tâche. »",
        ]),
        ("Each task, in the order of the sheet", [
            f"1. Open the task's start link (page {pg['links']}) with the screen turned away "
            "from the participant.",
            "2. The participant reads the card and says « prêt ». Turn the screen, say "
            "« Allez-y » and start the stopwatch.",
            "3. Count the clicks (a time entry counts as 1). Stop the stopwatch at the first "
            "complete spoken answer and write the answer word for word.",
            f"4. T-AMB: show the screen for {tg['ambulance_glance_s']:g} s, cover it, then ask "
            "the question. No click.",
            "5. Outcome: success = correct, no hint, within the time limit and at most "
            f"{tg['max_clicks']} clicks; partial = correct after the hint or over the click "
            "limit; fail = wrong answer or timeout; skipped = not run.",
            "6. At the limit: « Merci, on passe à la suite. » (fail, time = the limit).",
            f"7. Ask the SEQ (page {pg['questions']}), then 20 s: « Qu'est-ce qui vous a aidé, "
            "ou gêné ? »",
        ]),
        ("Hint (once, only after 10 s of silence)", [
            "« Regardez la carte de votre profil sur l'écran d'accueil. »",
        ]),
        ("Closing", [
            f"« {protocol['after_tasks']['transparency_fr']} » Record simulated_understood.",
            f"Optional SUS-10 (page {pg['questions']}). Thank the participant and remind them "
            "that their P-id lets them withdraw their answers.",
        ]),
        ("Error codes for the sheet", [", ".join(protocol["error_codes"])]),
    ]  # fmt: skip
    y = 0.885
    for head, paragraphs in items:
        fig.text(MARGIN, y, head, fontsize=11, weight="bold", va="top")
        y -= 0.026
        for text in paragraphs:
            y = _para(fig, y, text, width=100, font=9, step=0.019) - 0.004
        y -= 0.014
    _save(pdf, fig)


def consent(pdf: PdfPages, protocol: dict) -> None:
    fig = _page(pdf, "Consentement (lu à voix haute) · consent and privacy",
                "The consent text is read to the participant. If the school requires "
                "written consent, the paper stays with the facilitator and is never scanned "
                "into the repository.")  # fmt: skip
    hi = protocol["participants"]["max"]
    y = _para(fig, 0.885,
              "Vous testez une application de démonstration qui affiche des données SIMULÉES : "
              "ce ne sont pas les feux réels. Nous notons seulement un identifiant anonyme "
              f"(P1 à P{hi}), le temps, le nombre de clics, vos réponses et vos remarques sur "
              "l'écran. Pas de nom, pas de coordonnées, pas d'enregistrement audio ni vidéo, "
              "aucune information sur votre santé. Vous pouvez arrêter à tout moment, sans vous "
              "justifier. Gardez votre identifiant : il permet de retirer vos réponses avant la "
              "remise du rapport.", width=80, font=11, step=0.025)  # fmt: skip
    y = _lines(fig, y - 0.015, ["Êtes-vous d'accord pour participer ?        [ ] oui, accord "
                                "oral"], font=11)  # fmt: skip
    fig.text(MARGIN, y - 0.025, "Privacy rules (ENF-07) for the facilitator", fontsize=11,
             weight="bold", va="top")  # fmt: skip
    y -= 0.05
    for text in (
        f"- Participants are P1..P{hi} only. Never write a name, contact, employer, age or "
        "health information anywhere.",
        "- Reduced mobility is health data: never record it. Record only the role played, and "
        "real (holds the role) or proxy (plays it).",
        "- Notes describe what happened ON THE SCREEN. `usertest-check` refuses unknown keys, "
        "e-mail addresses and phone numbers, but it cannot catch a description that identifies "
        "someone: re-read the notes before saving.",
    ):
        y = _para(fig, y, text, width=110, font=8.5, step=0.018) - 0.004
    fig.text(MARGIN, y - 0.02, "P-id slips (cut and hand one to each participant)",
             fontsize=11, weight="bold", va="top")  # fmt: skip
    top = y - 0.05
    w = (1 - 2 * MARGIN) / hi
    for k in range(hi):
        x = MARGIN + k * w
        fig.patches.append(plt.Rectangle((x + 0.005, top - 0.09), w - 0.01, 0.09,
                                         transform=fig.transFigure, fill=False, ls="--",
                                         lw=0.6, ec=GREY))  # fmt: skip
        fig.text(x + w / 2, top - 0.03, f"P{k + 1}", fontsize=20, weight="bold", ha="center",
                 va="center")  # fmt: skip
        fig.text(x + w / 2, top - 0.07, "Urban Traffic Twin · test", fontsize=7, ha="center",
                 color=GREY)  # fmt: skip
    _save(pdf, fig)


def order_and_links(pdf: PdfPages, protocol: dict) -> None:
    tasks = tasks_by_id(protocol)
    fig = _page(pdf, "Task order per participant and start links",
                f"{protocol['order']['first']} comes first for everyone (first-use time, "
                "ENF-06); the other tasks rotate by participant number; the optional task "
                "comes last. Start links set the run and the instant, never the view "
                "(except the ambulance screen).")  # fmt: skip
    hi = protocol["participants"]["max"]
    n = len(task_order(protocol, 1))
    rows = [[f"P{k}", *task_order(protocol, k)] for k in range(1, hi + 1)]
    y = _table(fig, 0.88, ["Participant", *[str(i) for i in range(1, n + 1)]], rows,
               [1.2] + [1] * n, row_h=0.026, font=8.5)  # fmt: skip
    fig.text(MARGIN, y - 0.025, "Start links (open with the screen turned away)", fontsize=11,
             weight="bold", va="top")  # fmt: skip
    rows = []
    for tid, t in tasks.items():
        rows.append([tid, _instant(t), deep_link(protocol, t, "start")])
    y = _table(fig, y - 0.05, ["Task", "Instant", "Start link"], rows, [0.7, 1.3, 5],
               row_h=0.024, font=7)  # fmt: skip
    _lines(fig, y - 0.02, [
        "`python -m traffic_twin.usertests links` prints the same links, and the answer-view "
        "links of the automated checks.",
        "The ambulance link opens the screen directly in kiosk mode: no interaction by "
        "design (spec 12).",
    ], font=8)  # fmt: skip
    _save(pdf, fig)


def cards(pdf: PdfPages, protocol: dict) -> None:
    tasks = list(protocol["tasks"])
    for first in range(0, len(tasks), CARDS_PER_PAGE):
        fig = _page(pdf, "Task cards · cartes de tâche (cut along the dashed lines)",
                    "The participant reads the card, then says « prêt ». The small English "
                    "line is for the facilitator.")  # fmt: skip
        for k, t in enumerate(tasks[first : first + CARDS_PER_PAGE]):
            y = 0.875 - k * 0.205
            fig.patches.append(plt.Rectangle((MARGIN, y - 0.185), 1 - 2 * MARGIN, 0.19,
                                             transform=fig.transFigure, fill=False, ls="--",
                                             lw=0.6, ec=GREY))  # fmt: skip
            tag = t["id"] + ("  (optionnelle)" if t.get("optional") else "")
            fig.text(MARGIN + 0.015, y - 0.01, tag, fontsize=9, color=GREY, va="top")
            fig.text(MARGIN + 0.015, y - 0.035, _wrap(t["card_fr"], 46), fontsize=17,
                     weight="bold", va="top", linespacing=1.25)  # fmt: skip
            fig.text(MARGIN + 0.015, y - 0.172, _wrap(t["card_en"], 120), fontsize=7,
                     color=GREY, va="bottom")  # fmt: skip
        _save(pdf, fig)


def answer_key(pdf: PdfPages, protocol: dict, answers: dict) -> None:
    fig = _page(
        pdf,
        "Answer key (facilitator only: never shown to the participant)",
        "Every value was read from the API on the frozen run at the task's instant "
        "when this kit was built (traffic_twin.usertests.answers).",
        landscape=True,
    )
    rows = []
    for tid, t in tasks_by_id(protocol).items():
        a = answers["tasks"][tid]
        key = answer_text(t, a)
        if tid == "T-OP" or any(c["expected"] == "argmin" for c in a["checks"]):
            op = answers["operator"]
            vals = ", ".join(f"{k} {'PENDING' if v is None else round(v, 1)}"
                             for k, v in op["values"].items())  # fmt: skip
            key = f"{op['argmin'] or 'PENDING'} (lowest {op['kpi']}: {vals})"
        rows.append([tid, t["view"], _instant(t), _wrap(t["answer_fr"], 46), _wrap(key, 58),
                     f"{t['time_limit_s']:g} s"])  # fmt: skip
    y = _table(fig, 0.87, ["Task", "View", "Instant", "Expected answer (FR)", "API value",
                           "Limit"], rows, [0.6, 0.8, 1.2, 3, 3.6, 0.5], row_h=0.075,
               font=7.5)  # fmt: skip
    tg = protocol["targets"]
    _lines(fig, y - 0.02, [
        f"Targets: T-PED under {tg['pedestrian_time_s']:g} s on first use; at most "
        f"{tg['max_clicks']} clicks on every task; T-AMB read in a "
        f"{tg['ambulance_glance_s']:g} s glance (ENF-06).",
        "Write the answer word for word; score it against this key afterwards if unsure. "
        "A number within the stated tolerance is correct.",
    ], font=8.5)  # fmt: skip
    _save(pdf, fig)


def observation_sheet(pdf: PdfPages, protocol: dict, participant: int) -> None:
    tasks = tasks_by_id(protocol)
    fig = _page(pdf, f"Observation sheet · participant P{participant}",
                "Profile ______________   real / proxy   ·   date ____-__-__   ·   round 1 / 2   "
                "·   app version ________   ·   device laptop / phone / tablet   ·   consent "
                "oral / written", landscape=True)  # fmt: skip
    rows = []
    for k, tid in enumerate(task_order(protocol, participant), start=1):
        rows.append([str(k), tid + (" (opt.)" if tasks[tid].get("optional") else ""),
                     f"{tasks[tid]['time_limit_s']:g} s", "", "", "", "", "S / P / F / skip",
                     "", "", ""])  # fmt: skip
    y = _table(fig, 0.86, ["#", "Task", "Limit", "Time s", "Clicks", "Hint", "Answer (verbatim)",
                           "Outcome", "Error codes", "SEQ 1-7", "Notes (on screen only)"], rows,
               [0.3, 1.15, 0.5, 0.6, 0.5, 0.4, 2.5, 1.1, 1.2, 0.65, 2.3], row_h=0.075,
               font=8)  # fmt: skip
    _lines(fig, y - 0.02, [
        f"Transparency: « {protocol['after_tasks']['transparency_fr']} »   réelles / simulées "
        "/ ne sait pas",
        "SUS-10 done:  yes / no      Remarks about the screens (participant's words): "
        "_____________________________________________",
    ], font=8.5, step=0.03)  # fmt: skip
    _save(pdf, fig)


def questionnaire(pdf: PdfPages, protocol: dict) -> None:
    fig = _page(
        pdf,
        "Questionnaire (FR): SEQ and SUS-10",
        "SEQ is asked after every task. The SUS-10 items below are our own French "
        "wording: indicative only with 3-5 people, not the validated F-SUS.",
    )
    fig.text(MARGIN, 0.88, "Après chaque tâche (SEQ)", fontsize=11, weight="bold", va="top")
    _lines(fig, 0.855, [
        "« Dans l'ensemble, cette tâche était-elle difficile ou facile ? »",
        "1 = très difficile   2   3   4   5   6   7 = très facile",
    ], font=10, step=0.024)  # fmt: skip
    fig.text(MARGIN, 0.79, "À la fin (SUS-10, facultatif) : 1 = pas du tout d'accord, 5 = tout "
             "à fait d'accord", fontsize=11, weight="bold", va="top")  # fmt: skip
    rows = [[str(i), item, "1   2   3   4   5"] for i, item in enumerate(SUS_FR, start=1)]
    _table(fig, 0.765, ["#", "Affirmation", "Réponse"], rows, [0.3, 5.2, 1.2], row_h=0.04,
           font=8.5)  # fmt: skip
    _save(pdf, fig)


def issue_log(pdf: PdfPages, protocol: dict) -> None:
    fig = _page(
        pdf,
        "Issue log (problems seen, triaged after each round)",
        "One line per problem. Severity: 4 blocks the task, 3 major, 2 minor, 1 "
        "cosmetic. Copy into results.yaml as U01, U02, ...; a fix names its commit "
        "and is verified by a later session or an automated test.",
        landscape=True,
    )
    rows = [["U__", "", "", "", "", "", ""] for _ in range(12)]
    _table(fig, 0.87, ["Id", "Tasks", "Sessions", "Sev. 0-4", "Kind", "Problem (on screen)",
                       "Proposed fix"], rows, [0.5, 0.8, 0.8, 0.7, 0.9, 4, 3], row_h=0.055,
           font=8)  # fmt: skip
    fig.text(MARGIN, 0.12, "Kinds: " + ", ".join(protocol["issue_kinds"]), fontsize=8)
    _save(pdf, fig)


# -- build -----------------------------------------------------------------------------
def kit_keywords(answers: dict, protocol: dict) -> str:
    run = answers["run"]
    return (f"run_id={run['run_id']}; signal_sha256={run['signal_sha256']}; "
            f"scenario={run['scenario']}; seed={run['seed']}; "
            f"preregistered={protocol['preregistered']}; "
            f"draft={'yes' if answers['operator']['pending'] else 'no'}")  # fmt: skip


def kit_identity(path: Path) -> dict | None:
    """The run identity a kit recorded (from its PDF Keywords), or None."""
    path = Path(path)
    if not path.is_file():
        return None
    m = KEYWORDS_RE.search(path.read_bytes())
    if not m:
        return None
    pairs = (p.split("=", 1) for p in m.group(1).decode("latin-1").split("; ") if "=" in p)
    return {k.strip(): v.strip() for k, v in pairs}


def page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", Path(path).read_bytes()))


def build_kit(
    protocol: dict,
    out: Path,
    *,
    allow_pending_s3: bool = False,
    answers: dict | None = None,
    app_version: str | None = None,
) -> Path:
    """Write the kit PDF. KitError when the run is missing, when an expected answer no
    longer holds, or when a label T-OP needs (S3) is missing from /kpi, unless
    `allow_pending_s3` (a DRAFT kit, for the Phase 2 gate only)."""
    if answers is None:
        answers = expected_answers(protocol)
    if answers["problems"]:
        listing = "\n  - ".join(answers["problems"])
        raise KitError(f"expected answers no longer hold on the frozen run:\n  - {listing}")
    pending = answers["operator"]["pending"]
    if pending and not allow_pending_s3:
        raise KitError(f"{', '.join(pending)} missing from /kpi: T-OP has no answer yet. Run "
                       "Part 8 first (or --allow-pending-s3 for a draft kit)")  # fmt: skip
    if not pending and answers["operator"]["argmin"] is None:
        raise KitError(f"/kpi misses {answers['operator']['missing']}: T-OP has no answer")
    if app_version is None:
        from traffic_twin.sim.results import code_version

        app_version = code_version()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "User-test kit (spec 12.2, A-10)",
        "Subject": "Urban Traffic Twin user tests" + (" (DRAFT: S3 pending)" if pending else ""),
        "Keywords": kit_keywords(answers, protocol),
        "CreationDate": None,
    }
    with PdfPages(out, metadata=metadata) as pdf:
        cover(pdf, protocol, answers, app_version, draft=bool(pending))
        script(pdf, protocol)
        consent(pdf, protocol)
        order_and_links(pdf, protocol)
        cards(pdf, protocol)
        answer_key(pdf, protocol, answers)
        for k in range(1, protocol["participants"]["max"] + 1):
            observation_sheet(pdf, protocol, k)
        questionnaire(pdf, protocol)
        issue_log(pdf, protocol)
    return out
