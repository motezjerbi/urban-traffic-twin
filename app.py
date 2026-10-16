"""Urban Traffic Twin : dashboard Streamlit des 4 scénarios.

Source unique : docs/evaluation.md (page générée par `tasks.py evaluate`).
Aucun chiffre n'est écrit dans ce fichier : tout est relu dans les tableaux de la page.

Lancer, depuis la racine du dépôt, venv activé :
    python -m pip install streamlit
    streamlit run app.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- chemins
_HERE = Path(__file__).resolve().parent
EVAL_MD = next((p for p in (_HERE / "docs" / "evaluation.md", Path("docs/evaluation.md")) if p.exists()),
               _HERE / "docs" / "evaluation.md")
EXPORT_DIR = _HERE / "data" / "exports"

# --------------------------------------------------------------------------- lecture de evaluation.md
NUM = r"[-+]?\d+(?:\.\d+)?"
MEAN_SD = re.compile(rf"({NUM})\s*±\s*({NUM})")
PAIR_RE = re.compile(r"\*\*(.+?)\*\* \(`(.+?)`\) vs \*\*(.+?)\*\* \(`(.+?)`\), family `(.+?)`, role \*(.+?)\*")
FAMILY_RE = re.compile(r"Family `(.+?)`")
A04_RE = re.compile(r"\*\*A-04 (PASS|FAIL)\*\* over (\d+) runs")
A05_RE = re.compile(
    r"Primary \(acceptance\):.*?\*\*(PASS|FAIL|INVALID)\*\*\. n = (\d+), mean ([\d.]+) → ([\d.]+) s, "
    r"HL ([-\d.]+) s \[([-\d.]+), ([-\d.]+)\], exact one-sided p ([\d.]+)")
A03_RE = re.compile(r"\*\*Verdicts:\*\* (.+)")


def _clean(cell: str) -> str:
    return cell.strip().replace("`", "").replace("**", "").strip()


def _split_row(line: str) -> list[str]:
    return [_clean(c) for c in line.strip().strip("|").split("|")]


def _to_float(text: str):
    m = re.search(NUM, text.replace("−", "-"))
    return float(m.group()) if m else None


def parse_evaluation(md_text: str) -> dict:
    lines = md_text.splitlines()
    out = {k: [] for k in ("kpis", "pairs", "equity", "verdict", "runs", "a04", "guard", "a05_seeds", "defs")}
    context = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("|"):
            if line.strip():
                context = line
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].startswith("|"):
            block.append(lines[i])
            i += 1
        if len(block) < 3:
            continue
        header = _split_row(block[0])
        rows = [_split_row(r) for r in block[2:]]
        fam = FAMILY_RE.search(context)
        family = fam.group(1) if fam else "?"

        if header[:3] == ["User", "KPI", "Unit"]:
            for r in rows:
                for name, cell in zip(header[3:], r[3:]):
                    m = MEAN_SD.search(cell)
                    out["kpis"].append({"family": family, "user": r[0], "kpi": r[1], "unit": r[2],
                                        "scenario": name,
                                        "mean": float(m.group(1)) if m else _to_float(cell),
                                        "sd": float(m.group(2)) if m else None})
        elif header[:2] == ["User", "KPI"] and "Change" in header and "p (Holm)" in header:
            m = PAIR_RE.search(context)
            if not m:
                continue
            base, _, scen, _, fam_, role = m.groups()
            for r in rows:
                if len(r) < 10:
                    continue
                out["pairs"].append({"family": fam_, "role": role, "baseline": base, "scenario": scen,
                                     "pair": f"{base} vs {scen}", "user": r[0], "kpi": r[1],
                                     "base_mean": _to_float(r[2]), "scen_mean": _to_float(r[3]),
                                     "diff": r[4], "change_pct": _to_float(r[5]),
                                     "p": _to_float(r[6]), "p_holm": _to_float(r[7]),
                                     "effect": r[8], "equity_flag": r[9].lower() == "yes"})
        elif header[:2] == ["Family", "Scenario"] and "Baseline" in header:
            for r in rows:
                out["equity"].append({"family": r[0], "scenario": r[1], "user": r[2], "kpi": r[3],
                                      "baseline": _to_float(r[4]), "scenario_mean": _to_float(r[5]),
                                      "change_pct": _to_float(r[6]), "p": _to_float(r[7])})
        elif header[:2] == ["Family", "Scenario vs baseline"]:
            out["verdict"] += [dict(zip(header, r)) for r in rows]
        elif header[:3] == ["Family", "Label", "Scenario"]:
            out["runs"] += [dict(zip(header, r)) for r in rows]
        elif header[:3] == ["Family", "Scenario", "Controller"]:
            out["a04"] += [dict(zip(header, r)) for r in rows]
        elif header[0] == "Counter":
            for r in rows:
                for name, cell in zip(header[1:], r[1:]):
                    out["guard"].append({"family": family, "counter": r[0], "scenario": name,
                                         "value": _to_float(cell)})
        elif header[0] == "Seed" and len(header) >= 4:
            for r in rows:
                out["a05_seeds"].append({"seed": int(float(r[0])), "S0": _to_float(r[1]),
                                         "S2": _to_float(r[2]), "diff": _to_float(r[3])})
        elif header[:2] == ["KPI", "Users"]:
            for r in rows:
                out["defs"].append({"kpi": r[0], "formula": r[3], "unit": r[4], "better_if": r[5]})

    data = {k: pd.DataFrame(v) for k, v in out.items()}
    a04, a05, a03 = A04_RE.search(md_text), A05_RE.search(md_text), A03_RE.search(md_text)
    data["a04_verdict"] = {"verdict": a04.group(1), "runs": int(a04.group(2))} if a04 else None
    data["a05_verdict"] = ({"verdict": a05.group(1), "n": int(a05.group(2)), "s0": float(a05.group(3)),
                            "s2": float(a05.group(4)), "hl": float(a05.group(5)),
                            "lo": float(a05.group(6)), "hi": float(a05.group(7)),
                            "p": float(a05.group(8))} if a05 else None)
    data["verdicts_line"] = a03.group(1).strip() if a03 else ""
    return data


def add_gaps(kpis: pd.DataFrame, defs: pd.DataFrame, baseline: str = "S0") -> pd.DataFrame:
    base = (kpis[kpis.scenario == baseline][["family", "user", "kpi", "mean"]]
            .rename(columns={"mean": "baseline_mean"}))
    out = kpis.merge(base, on=["family", "user", "kpi"], how="left")
    out["gap_abs"] = out["mean"] - out["baseline_mean"]
    out["gap_pct"] = out["gap_abs"] / out["baseline_mean"].where(out["baseline_mean"] != 0) * 100
    out["base_kpi"] = out["kpi"].str.split("__").str[0]
    if not defs.empty:
        out = out.merge(defs.rename(columns={"kpi": "base_kpi"})[["base_kpi", "formula", "better_if"]]
                        .drop_duplicates("base_kpi"), on="base_kpi", how="left")
    else:
        out["formula"], out["better_if"] = None, None

    def verdict(r):
        if r.scenario == baseline or pd.isna(r.gap_pct) or abs(r.gap_pct) < 0.5:
            return "" if r.scenario == baseline else "≈"
        if r.better_if == "lower":
            return "mieux" if r.gap_pct < 0 else "pire"
        if r.better_if == "higher":
            return "mieux" if r.gap_pct > 0 else "pire"
        return ""
    out["verdict"] = out.apply(verdict, axis=1)
    return out


def export_all(md_path: Path = EVAL_MD, out_dir: Path = EXPORT_DIR) -> list[Path]:
    d = parse_evaluation(md_path.read_text(encoding="utf-8"))
    d["kpis"] = add_gaps(d["kpis"], d["defs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("kpis", "pairs", "equity", "verdict", "a04", "guard", "a05_seeds"):
        if not d[name].empty:
            p = out_dir / f"{name}.csv"
            d[name].to_csv(p, index=False, encoding="utf-8-sig")
            written.append(p)
    return written


# --------------------------------------------------------------------------- identité visuelle
ORDER = ["S0", "S1", "S2_noP", "S2"]                   # du moins bon au meilleur
COLORS = {"S0": "#E5484D", "S1": "#F5A524", "S2_noP": "#A6CF45", "S2": "#2DD4A0"}
LABELS = {"S0": "Temps fixe", "S1": "Actuated", "S2_noP": "Règles sans préemption", "S2": "Règles complètes"}
BG, SURFACE, BORDER, TEXT, MUTED = "#0F1419", "#171E26", "#2A3441", "#E6EDF3", "#8B98A8"

HEADLINE = [("car", "time_loss_mean", "Voiture · perte de temps"), ("car", "wait_p95", "Voiture · attente p95"),
            ("bus", "delay_mean", "Bus · retard"), ("bus", "headway_std", "Bus · régularité"),
            ("pedestrian", "ped_wait_mean", "Piéton · attente moyenne"),
            ("pedestrian", "ped_curb_wait_max", "Piéton · attente max trottoir"),
            ("ambulance", "travel_time_total", "Ambulance · trajet total"),
            ("all", "person_delay_mean", "Retard pondéré par personne")]


SIDEBAR_CSS = """
[data-testid="stSidebar"] { background: linear-gradient(180deg, #131A22 0%, #0E1318 100%); border-right: 1px solid #232D38; }
[data-testid="stSidebar"] > div:first-child { padding-top: .6rem; }
section[data-testid="stSidebar"] { min-width: 335px !important; max-width: 335px !important; }
.brand { display: flex; gap: .9rem; align-items: center; margin: .3rem 0 .4rem; }
.signal { background: #090D11; border: 1px solid #2A3441; border-radius: 12px; padding: 8px 7px;
          display: flex; flex-direction: column; gap: 6px; }
.signal i { display: block; width: 13px; height: 13px; border-radius: 50%; }
.signal i:nth-child(1) { background: #E5484D; box-shadow: 0 0 9px #E5484D99; }
.signal i:nth-child(2) { background: #F5A524; box-shadow: 0 0 9px #F5A52499; }
.signal i:nth-child(3) { background: #2DD4A0; box-shadow: 0 0 9px #2DD4A099; }
.brand b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.28rem; line-height: 1.05; display: block; }
.brand span { color: #8B98A8; font-size: .8rem; }
.sb-h { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1rem; margin: 1.5rem 0 .55rem; color: #E6EDF3; }
.sb-h small { color: #8B98A8; font-weight: 500; font-size: .8rem; margin-left: .4rem; font-family: 'IBM Plex Sans', sans-serif; }
.sb-note { color: #8B98A8; font-size: .78rem; margin: .5rem 0 0; line-height: 1.35; }
.sb-desc { color: #8B98A8; font-size: .76rem; margin: -.25rem 0 .1rem 2.7rem; line-height: 1.3; }
[data-testid="stSidebar"] .stButton button { width: 100%; background: #171E26; border: 1px solid #2A3441; color: #E6EDF3;
      border-radius: 9px; font-size: .84rem; padding: .3rem .5rem; }
[data-testid="stSidebar"] .stButton button:hover { border-color: #2DD4A0; color: #2DD4A0; }
[data-testid="stSidebar"] [data-baseweb="button-group"] button, [data-testid="stSidebar"] [data-testid="stPills"] button {
      border-radius: 999px; }
"""
SCN_CARD_CSS = "".join(
    f".st-key-scn_card_{s} {{ border: 1px solid {BORDER}; border-left: 5px solid {c}; border-radius: 10px; "
    f"background: #141B23; padding: .55rem .8rem .5rem; margin-bottom: .5rem; }} "
    f".st-key-scn_card_{s}:has(input:not(:checked)) {{ opacity: .5; }} "
    for s, c in COLORS.items())

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background: #12181F; border-right: 1px solid #232D38; }}
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; color: {TEXT}; }}
.title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.7rem; line-height: 1.05; margin: 0; }}
.subtitle {{ color: {MUTED}; font-size: 1.02rem; margin-top: .45rem; }}
.lane {{ height: 4px; margin: 1.1rem 0 1.5rem;
        background: repeating-linear-gradient(90deg, #F2C94C 0 30px, transparent 30px 52px); opacity: .9; }}
.notice {{ background: #2A2210; border: 1px solid #5A4513; border-left: 5px solid #F2C94C; border-radius: 10px;
          padding: .85rem 1.1rem; font-size: .93rem; color: #F3E7C0; margin-bottom: 1.4rem; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-top: 5px solid var(--c); border-radius: 12px;
        padding: 1.05rem 1.15rem 1rem; height: 100%; }}
.card.off {{ opacity: .38; }}
.rank {{ font-size: .8rem; color: var(--c); font-weight: 600; }}
.name {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 800; line-height: 1.1; margin-top: .15rem; }}
.sub {{ color: {MUTED}; font-size: .86rem; margin-bottom: .75rem; }}
.big {{ font-family: 'IBM Plex Mono', monospace; font-size: 2.15rem; font-weight: 500; line-height: 1; }}
.big small {{ font-size: .95rem; color: {MUTED}; margin-left: .2rem; }}
.lab {{ color: {MUTED}; font-size: .8rem; margin: .25rem 0 .55rem; }}
.delta {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: .85rem; padding: .12rem .5rem;
         border-radius: 6px; background: #1F2933; }}
.good {{ color: #2DD4A0; }} .bad {{ color: #FF7A7F; }} .flat {{ color: {MUTED}; }}
.mini {{ display: flex; gap: 1rem; margin-top: .8rem; padding-top: .7rem; border-top: 1px solid {BORDER}; }}
.mini div {{ flex: 1; }}
.mini b {{ font-family: 'IBM Plex Mono', monospace; font-weight: 500; font-size: 1.02rem; display: block; }}
.mini span {{ color: {MUTED}; font-size: .74rem; }}
button[data-baseweb="tab"] {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.02rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom: 1px solid {BORDER}; }}
.section-note {{ color: {MUTED}; font-size: .9rem; margin: .1rem 0 .9rem; }}
.pill {{ display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600;
        border:1px solid var(--c); color: var(--c); margin-right:.3rem; }}
</style>
"""


def main() -> None:
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Urban Traffic Twin", page_icon="🚦", layout="wide")
    st.markdown(CSS + "<style>" + SIDEBAR_CSS + SCN_CARD_CSS + "</style>", unsafe_allow_html=True)

    def show_df(df, **kw):
        try:
            st.dataframe(df, width="stretch", hide_index=True, **kw)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True, **kw)

    def show_chart(ch):
        ch = (ch.configure_view(strokeWidth=0).configure(background="transparent")
              .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#222C37", domainColor=BORDER, tickColor=BORDER)
              .configure_legend(labelColor=TEXT, titleColor=MUTED)
              .configure_header(labelColor=TEXT, titleColor=TEXT))
        try:
            st.altair_chart(ch, width="stretch")
        except Exception:
            st.altair_chart(ch, use_container_width=True)

    @st.cache_data(show_spinner=False)
    def load(path: str, mtime: float):
        d = parse_evaluation(Path(path).read_text(encoding="utf-8"))
        d["kpis"] = add_gaps(d["kpis"], d["defs"])
        return d

    if not EVAL_MD.exists():
        st.error(f"Fichier introuvable : {EVAL_MD}. Génère-le avec `python tasks.py evaluate`, "
                 "puis relance l'application depuis la racine du dépôt.")
        st.stop()
    data = load(str(EVAL_MD), EVAL_MD.stat().st_mtime)
    kpis, pairs, equity = data["kpis"], data["pairs"], data["equity"]
    if kpis.empty:
        st.error("Aucun tableau de KPI lu dans evaluation.md. Le format de la page a peut-être changé.")
        st.stop()

    # ------------------------------------------------------------------ panneau de contrôle (barre latérale)
    sb = st.sidebar
    DESC = {"S0": "Cycle fixe, ne regarde pas le trafic", "S1": "Prolonge le vert si un véhicule arrive",
            "S2_noP": "Règles bus et piétons, sans préemption", "S2": "Règles, priorité bus, préemption ambulance"}
    USER_LABELS = {"car": "🚗 Voiture", "bus": "🚌 Bus", "pedestrian": "🚶 Piéton",
                   "ambulance": "🚑 Ambulance", "all": "👥 Tous", "junction": "🚦 Carrefour"}
    FAM_LABELS = {"main": "Principale", "ambulance_probe": "Sonde"}
    FAM_NOTES = {"main": "1 ambulance par simulation", "ambulance_probe": "7 ambulances par simulation"}

    sb.markdown('<div class="brand"><div class="signal"><i></i><i></i><i></i></div>'
                '<div><b>Panneau de contrôle</b><span>Bd de l\'Hôpital × Bd Saint-Marcel</span></div></div>',
                unsafe_allow_html=True)

    families = sorted(kpis.family.unique(), key=lambda f: f != "main")
    sb.markdown('<div class="sb-h">Famille de runs</div>', unsafe_allow_html=True)
    if hasattr(sb, "segmented_control"):
        family = sb.segmented_control("Famille", families, default=families[0], key="family",
                                      format_func=lambda f: FAM_LABELS.get(f, f), label_visibility="collapsed")
    else:
        family = sb.radio("Famille", families, horizontal=True, format_func=lambda f: FAM_LABELS.get(f, f),
                          label_visibility="collapsed")
    family = family or families[0]
    sb.markdown(f'<p class="sb-note">{FAM_NOTES.get(family, "")}</p>', unsafe_allow_html=True)

    fk = kpis[kpis.family == family]
    available = [x for x in ORDER if x in fk.scenario.unique()]

    def preset(names):
        for x in ORDER:
            st.session_state[f"scn_{x}"] = x in names

    sb.markdown('<div class="sb-h">Scénarios<small>du moins bon au meilleur</small></div>', unsafe_allow_html=True)
    b1, b2 = sb.columns(2)
    b1.button("Tous", key="p_all", on_click=preset, args=(ORDER,))
    b2.button("Duel S0 vs S2", key="p_duel", on_click=preset, args=(["S0", "S2"],))
    for x in ORDER:
        st.session_state.setdefault(f"scn_{x}", True)
    chosen = []
    for x in available:
        with sb.container(key=f"scn_card_{x}"):
            if st.toggle(f"**{x}**  {LABELS.get(x, '')}", key=f"scn_{x}"):
                chosen.append(x)
            st.markdown(f'<p class="sb-desc">{DESC.get(x, "")}</p>', unsafe_allow_html=True)
    sb.markdown('<p class="sb-note">S0 reste la référence des écarts, même désactivé.</p>', unsafe_allow_html=True)

    users = sorted(fk.user.unique())
    sb.markdown('<div class="sb-h">Usagers</div>', unsafe_allow_html=True)
    if hasattr(sb, "pills"):
        sel_users = sb.pills("Usagers", users, selection_mode="multi", default=users, key=f"users_{family}",
                             format_func=lambda u: USER_LABELS.get(u, u), label_visibility="collapsed")
    else:
        sel_users = sb.multiselect("Usagers", users, default=users, label_visibility="collapsed")
    sel_users = list(sel_users) or users

    sb.markdown('<div class="sb-h">Affichage</div>', unsafe_allow_html=True)
    only_head = sb.toggle("KPI phares seulement", value=True)
    sb.markdown('<p class="sb-note">Désactive pour voir les dizaines de KPI détaillés (par branche, ligne de bus, '
                'passage piéton).</p>', unsafe_allow_html=True)

    sb.markdown('<div class="sb-h">Export</div>', unsafe_allow_html=True)
    if sb.button("Exporter tout en CSV", key="p_export"):
        files = export_all()
        sb.success(f"{len(files)} fichiers écrits dans data/exports/")
    sb.markdown(f'<p class="sb-note">Source : {EVAL_MD.name}, modifié le '
                f'{pd.Timestamp(EVAL_MD.stat().st_mtime, unit="s"):%d/%m/%Y %H:%M}</p>', unsafe_allow_html=True)

    if not chosen:
        st.info("Coche au moins un scénario dans la barre latérale.")
        st.stop()

    # ------------------------------------------------------------------ en-tête
    st.markdown('<p class="title">Urban Traffic Twin</p>'
                '<p class="subtitle">Carrefour Bd de l\'Hôpital × Bd Saint-Marcel, Paris 13e. '
                'Quatre façons de piloter les mêmes feux, sur les mêmes 10 simulations.</p>'
                '<div class="lane"></div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>Résultats provisoires.</b> Le programme de feux (cycle 123 s) est supposé, '
                'pas mesuré : seuls les débits entrants sont validés. La visite terrain doit encore mesurer '
                'le phasage réel et les files d\'attente.</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------ rang et cartes
    def val(scn, user, kpi):
        r = fk[(fk.scenario == scn) & (fk.user == user) & (fk.kpi == kpi)]
        return float(r["mean"].iloc[0]) if len(r) and pd.notna(r["mean"].iloc[0]) else None

    def gap(scn, user, kpi):
        r = fk[(fk.scenario == scn) & (fk.user == user) & (fk.kpi == kpi)]
        return float(r["gap_pct"].iloc[0]) if len(r) and pd.notna(r["gap_pct"].iloc[0]) else None

    ranked = sorted(available, key=lambda s: (-(round(val(s, "all", "person_delay_mean") or 0, 1)),
                                              -(val(s, "ambulance", "travel_time_total") or 0)))
    n = len(ranked)

    def delta_html(g, lower_better=True):
        if g is None:
            return '<span class="delta flat">référence</span>'
        if abs(g) < 0.05:
            return '<span class="delta flat">= S0</span>'
        cls = "good" if (g < 0) == lower_better else "bad"
        return f'<span class="delta {cls}">{g:+.1f} % vs S0</span>'

    cols = st.columns(len(available))
    for col, scn in zip(cols, ranked):
        rk = ranked.index(scn) + 1
        tag = " · le moins bon" if rk == 1 else (" · le meilleur" if rk == n else "")
        pw, amb, car = val(scn, "all", "person_delay_mean"), val(scn, "ambulance", "travel_time_total"), val(scn, "car", "time_loss_mean")
        fmt = lambda v: "n.d." if v is None else f"{v:.1f}"
        off = "" if scn in chosen else " off"
        html = (f'<div class="card{off}" style="--c:{COLORS[scn]}"><div class="rank">Rang {rk}/{n}{tag}</div>'
                f'<div class="name">{scn}</div><div class="sub">{LABELS.get(scn, "")}</div>'
                f'<div class="big">{fmt(pw)}<small>s</small></div><div class="lab">Retard moyen pondéré par personne</div>'
                f'{delta_html(gap(scn, "all", "person_delay_mean") if scn != "S0" else None)}'
                f'<div class="mini"><div><b>{fmt(car)} s</b><span>Voiture, perte de temps</span></div>'
                f'<div><b>{fmt(amb)} s</b><span>Ambulance, trajet total</span></div></div></div>')
        col.markdown(html, unsafe_allow_html=True)
    st.caption("Rangs calculés sur le retard pondéré (arrondi à 0,1 s), puis départagés par le trajet de l'ambulance. "
               "Aucun scénario n'est meilleur pour tous les usagers : voir l'onglet Équité.")
    st.write("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Vue d'ensemble", "Tous les KPI", "Écarts et tests",
                                            "Équité", "Sécurité et acceptation"])

    # ------------------------------------------------------------------ 1. vue d'ensemble
    with tab1:
        head = pd.DataFrame(HEADLINE, columns=["user", "kpi", "label"])
        v = fk.merge(head, on=["user", "kpi"])
        v = v[v.user.isin(sel_users) & v.scenario.isin(chosen)]
        if v.empty:
            st.info("Aucun KPI phare pour ces usagers.")
        else:
            st.markdown("#### Valeurs moyennes par KPI")
            st.markdown('<p class="section-note">Moyenne sur les seeds 1 à 10, barre d\'erreur = un écart-type. '
                        'Pour tous ces KPI, plus bas = mieux.</p>', unsafe_allow_html=True)
            v = v.assign(lo=v["mean"] - v["sd"].fillna(0), hi=v["mean"] + v["sd"].fillna(0))
            dom = [s for s in ORDER if s in chosen]
            color = alt.Color("scenario:N", scale=alt.Scale(domain=dom, range=[COLORS[s] for s in dom]),
                              legend=alt.Legend(title=None, orient="top"), sort=dom)
            base = alt.Chart(v).encode(x=alt.X("scenario:N", sort=dom, title=None, axis=alt.Axis(labelAngle=0)))
            bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                y=alt.Y("mean:Q", title=None), color=color,
                tooltip=["label", "scenario", alt.Tooltip("mean:Q", format=".1f"), alt.Tooltip("sd:Q", format=".1f")])
            err = base.mark_errorbar(color="#C9D3DD").encode(y=alt.Y("lo:Q", title=None), y2="hi:Q")
            ch = (bars + err).properties(width=150, height=190).facet(
                facet=alt.Facet("label:N", title=None, sort=[h[2] for h in HEADLINE]), columns=4
            ).resolve_scale(y="independent")
            show_chart(ch)

            st.markdown("#### Écart par rapport à S0 (%)")
            st.markdown('<p class="section-note">Vers la gauche = valeur plus basse que S0, donc mieux '
                        'pour ces KPI.</p>', unsafe_allow_html=True)
            g = v[(v.scenario != "S0")].dropna(subset=["gap_pct"])
            if g.empty:
                st.info("Choisis un scénario autre que S0 pour voir les écarts.")
            else:
                dom2 = [s for s in ORDER if s in g.scenario.unique()]
                gc = (alt.Chart(g).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("label:N", title=None, sort=[h[2] for h in HEADLINE], axis=alt.Axis(labelLimit=260)),
                    x=alt.X("gap_pct:Q", title="Écart vs S0 (%)"),
                    yOffset=alt.YOffset("scenario:N", sort=dom2),
                    color=alt.Color("scenario:N", scale=alt.Scale(domain=dom2, range=[COLORS[s] for s in dom2]),
                                    legend=alt.Legend(title=None, orient="top")),
                    tooltip=["label", "scenario", alt.Tooltip("mean:Q", format=".1f"),
                             alt.Tooltip("baseline_mean:Q", title="S0", format=".1f"),
                             alt.Tooltip("gap_pct:Q", format="+.1f")])
                    .properties(height=max(280, 46 * g.label.nunique())))
                show_chart(gc)

    # ------------------------------------------------------------------ 2. tous les KPI
    with tab2:
        q = st.text_input("Rechercher un KPI", placeholder="par exemple wait, crossing, arm_S, delay…")
        v = fk[fk.user.isin(sel_users) & fk.scenario.isin(chosen)]
        if only_head:
            v = v.merge(pd.DataFrame(HEADLINE, columns=["user", "kpi", "label"])[["user", "kpi"]], on=["user", "kpi"])
        if q:
            v = v[v.kpi.str.contains(q, case=False, regex=False) | v.user.str.contains(q, case=False, regex=False)]
        if v.empty:
            st.info("Aucun KPI ne correspond à ces filtres.")
        else:
            v = v.assign(txt=v.apply(lambda r: "n.d." if pd.isna(r["mean"]) else
                                     (f"{r['mean']:.4g}" + (f" ± {r['sd']:.3g}" if pd.notna(r["sd"]) else "")), axis=1))
            idx = ["user", "kpi", "unit", "better_if", "formula"]
            wide = v.pivot_table(index=idx, columns="scenario", values="txt", aggfunc="first", sort=False)
            wide = wide[[s for s in ORDER if s in wide.columns]].reset_index()
            for s in [s for s in chosen if s != "S0"]:
                gp = v[v.scenario == s].set_index(["user", "kpi"])["gap_pct"]
                vd = v[v.scenario == s].set_index(["user", "kpi"])["verdict"]
                keys = list(zip(wide.user, wide.kpi))
                wide[f"Δ {s} %"] = [round(gp.get(k), 1) if k in gp.index and pd.notna(gp.get(k)) else None for k in keys]
                wide[f"{s}"] = wide[s] + [("  " + vd.get(k, "")) if vd.get(k, "") else "" for k in keys]
            wide = wide.rename(columns={"better_if": "mieux si", "formula": "définition", "unit": "unité"})
            st.markdown(f'<p class="section-note">{len(wide)} KPI. Moyenne ± écart-type sur les seeds 1 à 10. '
                        '« mieux » ou « pire » compare à S0 avec un seuil de 0,5 %.</p>', unsafe_allow_html=True)
            show_df(wide)
            st.download_button("Télécharger ce tableau (CSV)", wide.to_csv(index=False).encode("utf-8-sig"),
                               f"kpis_{family}.csv", "text/csv")

    # ------------------------------------------------------------------ 3. écarts et tests
    with tab3:
        p = pairs[(pairs.family == family) & pairs.baseline.isin(chosen + ["S0"]) & pairs.scenario.isin(chosen + ["S0"])]
        if p.empty:
            st.info("Aucune paire pour ces scénarios. Coche S0 et au moins un autre scénario.")
        else:
            pair = st.selectbox("Paire comparée", sorted(p.pair.unique()))
            pp = p[(p.pair == pair) & p.user.isin(sel_users)].copy()
            role = pp.role.iloc[0] if len(pp) else ""
            st.markdown(f'<p class="section-note">Rôle de la paire : <b>{role}</b>. Test de Wilcoxon exact bilatéral '
                        'apparié sur 10 seeds (plus petite p-value possible : 0,0020). <b>Holm ne porte que sur les '
                        '7 KPI phares</b> : pour les autres lignes, seul le p brut a un sens.</p>', unsafe_allow_html=True)
            shown = pp.rename(columns={"base_mean": f"moy. {pp.baseline.iloc[0] if len(pp) else ''}",
                                       "scen_mean": f"moy. {pp.scenario.iloc[0] if len(pp) else ''}",
                                       "diff": "différence [IC 95 %]", "change_pct": "changement %",
                                       "p_holm": "p (Holm)", "effect": "effet", "equity_flag": "alerte équité"})
            show_df(shown.drop(columns=["family", "baseline", "scenario", "pair", "role"]))
            pp["label"] = pp.user + " · " + pp.kpi
            chart = (alt.Chart(pp.dropna(subset=["change_pct"])).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
                x=alt.X("change_pct:Q", title="Changement (%)"),
                color=alt.Color("effect:N", title=None, scale=alt.Scale(
                    domain=["better", "worse", "same"], range=["#2DD4A0", "#FF7A7F", "#8B98A8"])),
                tooltip=["label", "change_pct", "p", "p_holm"]).properties(height=max(200, 30 * len(pp))))
            show_chart(chart)
            st.download_button("Télécharger les écarts (CSV)", pp.drop(columns="label").to_csv(index=False).encode("utf-8-sig"),
                               f"ecarts_{family}.csv", "text/csv")

    # ------------------------------------------------------------------ 4. équité
    with tab4:
        e = equity[(equity.family == family) & equity.scenario.isin(chosen)]
        st.markdown('<p class="section-note">Chaque KPI plus de 5 % pire que S0 est signalé. Le retard pondéré peut '
                    's\'améliorer alors que des sous-groupes (une branche, une ligne de bus, un passage piéton) se '
                    'dégradent : c\'est la « dégradation masquée ». Le p est brut, non ajusté.</p>', unsafe_allow_html=True)
        vd = data["verdict"]
        if not vd.empty:
            vv = vd[(vd.Family == family) & vd["Scenario vs baseline"].isin(chosen)]
            if not vv.empty:
                show_df(vv.drop(columns=["Family"]))
        if e.empty:
            st.info("Aucun KPI signalé pour ces scénarios.")
        else:
            counts = e.groupby("scenario").size().reindex([s for s in ORDER if s in chosen]).dropna().astype(int)
            st.markdown("".join(f'<span class="pill" style="--c:{COLORS[s]}">{s} : {c} KPI signalés</span>'
                                for s, c in counts.items()), unsafe_allow_html=True)
            st.write("")
            only_sub = st.toggle("Sous-groupes principaux seulement (branches, lignes de bus, passages)", value=True)
            ee = e.copy()
            if only_sub:
                ee = ee[ee.kpi.str.contains("__")]
            show_df(ee.sort_values("change_pct", ascending=False).drop(columns=["family"]))
            st.download_button("Télécharger l'équité (CSV)", ee.to_csv(index=False).encode("utf-8-sig"),
                               f"equite_{family}.csv", "text/csv")

    # ------------------------------------------------------------------ 5. sécurité et acceptation
    with tab5:
        c1, c2, c3 = st.columns(3)
        a04, a05 = data["a04_verdict"], data["a05_verdict"]
        c1.metric("A-04 sécurité", a04["verdict"] if a04 else "n.d.", f"{a04['runs']} runs" if a04 else None, delta_color="off")
        c2.metric("A-05 ambulance", a05["verdict"] if a05 else "n.d.",
                  f"{a05['s0']:.1f} → {a05['s2']:.1f} s, p = {a05['p']:.4f}" if a05 else None, delta_color="off")
        m03 = re.search(r"A-03 (\d+/\d+)", data["verdicts_line"])
        c3.metric("A-03 contrôleurs", m03.group(1) if m03 else "n.d.", "le RL arrive aux parties 6-7", delta_color="off")
        if a05:
            st.caption(f"A-05 : estimateur de Hodges-Lehmann {a05['hl']:+.1f} s, intervalle [{a05['lo']:+.1f} ; {a05['hi']:+.1f}], "
                       f"test de Wilcoxon exact unilatéral, n = {a05['n']}. Le p unilatéral minimal est 0,0010.")
        runs = data["runs"]
        lab = (runs[runs.Family == family].set_index("Scenario")["Label"].to_dict() if not runs.empty else {})
        st.markdown("#### A-04 : sécurité de chaque scénario")
        a = data["a04"]
        if not a.empty:
            aa = a[a.Family == family].copy()
            aa.insert(1, "Label", aa["Scenario"].map(lab))
            aa = aa[aa["Label"].isin(chosen)]
            show_df(aa.drop(columns=["Family"]))
        st.markdown("#### A-05 : ambulance, S2 contre S0, seed par seed")
        s5 = data["a05_seeds"]
        if family == "main" and not s5.empty:
            long = s5.melt(id_vars="seed", value_vars=["S0", "S2"], var_name="scénario", value_name="secondes")
            lc = (alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("seed:O", title="Seed"), y=alt.Y("secondes:Q", title="Trajet depuis le départ prévu (s)"),
                xOffset="scénario:N", color=alt.Color("scénario:N", scale=alt.Scale(
                    domain=["S0", "S2"], range=[COLORS["S0"], COLORS["S2"]]), legend=alt.Legend(title=None, orient="top")),
                tooltip=["seed", "scénario", "secondes"]).properties(height=260))
            show_chart(lc)
            st.caption("Seed 5 est la seule où S2 est un peu plus lent que S0.")
        else:
            st.info("Le détail seed par seed n'existe que pour la famille principale.")
        st.markdown("#### Activité du Safety Guard et du contrôleur")
        gd = data["guard"]
        if not gd.empty:
            gg = gd[(gd.family == family) & gd.scenario.isin(chosen + ["runs"])]
            wide = gg.pivot_table(index="counter", columns="scenario", values="value", sort=False)
            wide = wide[[s for s in ORDER if s in wide.columns]].reset_index()
            show_df(wide)
            st.caption("Une intervention du guard annule une demande d'un contrôleur. « ped_yield » compte les sauts de "
                       "préemption que le bouclier piéton a remplacés par la phase suivante.")


if __name__ == "__main__":
    main()