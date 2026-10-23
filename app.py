"""Urban Traffic Twin : panneau Opérateur Streamlit (spec 12 « Opérateur », D102, D108).

Source unique : l'API du jumeau, lue par `traffic_twin.app.client` (httpx, délai 3 s).
Aucun chiffre n'est écrit dans ce fichier, et l'application ne lit aucun fichier du
dépôt : libellés, couleurs, ordre, KPI, paires, A-03 à A-06, courbes et sensibilité
viennent des réponses de l'API (`/scenarios`, `/evaluation`, `/kpi`, `/compare`,
`/series`, `/sensitivity`). Les exports sont des téléchargements (rien n'est écrit sur
le disque).

Lancer, depuis la racine du dépôt, venv activé, l'API démarrée :
    python tasks.py demo            (API + Opérateur)
    python tasks.py operator        (Opérateur seul ; streamlit run app.py)
`TT_API_URL` change l'adresse de l'API (défaut http://127.0.0.1:8000).
"""

from __future__ import annotations

import time

import pandas as pd

from traffic_twin.app import client

# --------------------------------------------------------------------------- identité visuelle
BASELINE = "S0"  # la référence des écarts de l'API (`rel_change_vs_s0_pct`)
RL_GREEN = "#008300"  # = viz.style.CONTROLLER_COLORS["rl"] (S3, D108)
S4_GREY_GREEN = "#6F8F76"  # S3 sans préemption (D108)
COLORS = {
    "S0": "#E5484D",
    "S1": "#F5A524",
    "S2_noP": "#A6CF45",
    "S2": "#2DD4A0",
    "S3": RL_GREEN,
    "S4": S4_GREY_GREEN,
}
CONTROLLER_COLORS = {
    "fixed": "#E5484D",
    "actuated": "#F5A524",
    "rules": "#2DD4A0",
    "rule_based": "#2DD4A0",
    "rl": RL_GREEN,
}
OTHER_COLORS = ("#7AA2F7", "#C792EA", "#FF9E64", "#73DACA")  # labels inconnus, dans l'ordre
LABELS = {
    "S0": "Temps fixe",
    "S1": "Actuated",
    "S2_noP": "Règles sans préemption",
    "S2": "Règles complètes",
    "S3": "RL (politique S3)",
    "S4": "RL sans préemption",
}
CONTROLLER_NAMES = {
    "fixed": "Temps fixe",
    "actuated": "Actuated",
    "rules": "Règles",
    "rule_based": "Règles",
    "rl": "RL",
}
DESC = {
    "S0": "Cycle fixe, ne regarde pas le trafic",
    "S1": "Prolonge le vert si un véhicule arrive",
    "S2_noP": "Règles bus et piétons, sans préemption",
    "S2": "Règles, priorité bus, préemption ambulance",
    "S3": "Agent DQN de la partie 7, sous le Safety Guard",
    "S4": "S3 sans préemption : hors distribution, exploratoire",
}
ROLE_NAMES = {"ablation": "ablation", "exploratory": "exploratoire"}
BG, SURFACE, BORDER, TEXT, MUTED = "#0F1419", "#171E26", "#2A3441", "#E6EDF3", "#8B98A8"

HEADLINE = [
    ("car", "time_loss_mean", "Voiture · perte de temps"),
    ("car", "wait_p95", "Voiture · attente p95"),
    ("bus", "delay_mean", "Bus · retard"),
    ("bus", "headway_std", "Bus · régularité"),
    ("pedestrian", "ped_wait_mean", "Piéton · attente moyenne"),
    ("pedestrian", "ped_curb_wait_max", "Piéton · attente max trottoir"),
    ("ambulance", "travel_time_total", "Ambulance · trajet total"),
    ("all", "person_delay_mean", "Retard pondéré par personne"),
]
KPI_NAMES = {
    **{(u, k): name for u, k, name in HEADLINE},
    ("car", "wait_mean"): "Voiture · attente moyenne",
    ("car", "throughput_vph"): "Voiture · débit",
    ("bus", "dwell_mean"): "Bus · arrêt en station",
    ("pedestrian", "ped_curb_wait_mean"): "Piéton · attente trottoir",
    ("pedestrian", "ped_curb_wait_p95"): "Piéton · attente trottoir p95",
    ("pedestrian", "crossing_complete_rate"): "Piéton · traversées complètes",
    ("ambulance", "stops"): "Ambulance · arrêts",
    ("ambulance", "stop_duration_mean"): "Ambulance · durée des arrêts",
}
USER_LABELS = {
    "car": "🚗 Voiture",
    "bus": "🚌 Bus",
    "pedestrian": "🚶 Piéton",
    "ambulance": "🚑 Ambulance",
    "all": "👥 Tous",
    "junction": "🚦 Carrefour",
}
FAM_LABELS = {"main": "Principale", "ambulance_probe": "Sonde"}
FAM_NOTES = {
    "main": "1 ambulance par simulation",
    "ambulance_probe": "7 ambulances par simulation",
}
APPROACHES = {
    "all": "Toutes les branches",
    "N": "Branche Nord",
    "S": "Branche Sud",
    "W": "Branche Ouest",
}
EXTRA_SECTIONS = ("safety", "activity", "equity", "occupancy")  # lignes hors KPI du tableau final
A05_COLUMNS = ("family", "a", "b", "user", "kpi", "alternative", "n", "hl", "hl_ci", "p_one_sided", "status")  # fmt: skip
CACHE_TTL_S = 15.0  # réponses de l'API gardées le temps d'une interaction
PAIRS_TTL_S = 60.0


# --------------------------------------------------------------------------- lecture des réponses
def kpi_name(user: str, kpi: str) -> str:
    return KPI_NAMES.get((user, kpi), f"{user} · {kpi}")


def label_info(evaluation: dict | None, scenarios: dict | None, family: str) -> list[dict]:
    """Labels of `family`, in the API's order: `/evaluation.labels` (every registered
    label), else the `/scenarios` groups of the family (labels with runs)."""
    rows = list((evaluation or {}).get("labels") or [])
    if not rows:
        for g in (scenarios or {}).get("scenarios") or []:
            if g.get("label") and g.get("family") == family:
                rows.append(
                    {
                        "label": g["label"],
                        "scenario": g.get("scenario"),
                        "controller": g.get("controller"),
                        "role": g.get("role"),
                    }
                )
    out, seen = [], set()
    for r in rows:
        lab = r.get("label")
        if lab and lab not in seen:
            seen.add(lab)
            out.append({**r, "role": r.get("role") or "main"})
    return out


def families_of(scenarios: dict | None) -> list[str]:
    fams = {g.get("family") for g in (scenarios or {}).get("scenarios") or [] if g.get("family")}
    return sorted(fams, key=lambda f: (f != "main", f)) or ["main"]


def color_map(labels: list[dict]) -> dict[str, str]:
    colors, extra = {}, iter(OTHER_COLORS * 4)
    for r in labels:
        lab = r["label"]
        colors[lab] = (
            COLORS.get(lab) or CONTROLLER_COLORS.get(r.get("controller") or "") or next(extra)
        )
    return colors


def label_name(r: dict) -> str:
    return LABELS.get(r["label"]) or CONTROLLER_NAMES.get(r.get("controller") or "", "")


def kpi_frame(replies: dict[str, client.Reply]) -> pd.DataFrame:
    """One row per (label, user, kpi) of the `/kpi` answers."""
    recs = []
    for lab, rep in replies.items():
        for r in (rep.payload or {}).get("rows") or []:
            recs.append(
                {
                    "label": lab,
                    "user": r.get("user"),
                    "kpi": r.get("kpi"),
                    "mean": r.get("mean"),
                    "sd": r.get("std"),
                    "n": r.get("n"),
                    "rel_pct": r.get("rel_change_vs_s0_pct"),
                    "p": r.get("p_value"),
                    "worse_flag": r.get("worse_flag"),
                }
            )
    cols = ["label", "user", "kpi", "mean", "sd", "n", "rel_pct", "p", "worse_flag"]
    return pd.DataFrame(recs, columns=cols)


def final_frames(rows: list[dict], family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(KPI rows, extra rows) of `/evaluation.final_table` for `family`."""
    df = pd.DataFrame(rows)
    if df.empty or not {"section", "user", "kpi", "label"} <= set(df.columns):
        return pd.DataFrame(), pd.DataFrame()
    if "family" in df.columns:
        df = df[df["family"].isna() | (df["family"] == family)]
    extra = df[df["section"].isin(EXTRA_SECTIONS)]
    kpis = df[~df["section"].isin(EXTRA_SECTIONS)]
    probe = kpis["section"].astype(str).str.contains("probe")
    kpis = kpis[probe] if family == "ambulance_probe" else kpis[~probe]
    kpis = kpis.rename(columns={"rel_vs_s0_pct": "rel_pct", "scenario": "scenario_name"})
    return kpis.reset_index(drop=True), extra.reset_index(drop=True)


def pair_frame(rows: list[dict], a: str, b: str) -> pd.DataFrame:
    """The `/compare` rows (columns of pairs/<a>__vs__<b>.csv), for display."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    al, bl = a.lower(), b.lower()

    def col(name: str):
        return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index)

    def ci(lo, hi, mid):
        if pd.isna(mid):
            return ""
        if pd.isna(lo) or pd.isna(hi):
            return f"{mid:+.2f}"
        return f"{mid:+.2f} [{lo:+.2f} ; {hi:+.2f}]"

    out = pd.DataFrame(
        {
            "user": col("user_type") if "user_type" in df.columns else col("user"),
            "kpi": col("kpi"),
            "phare": col("headline"),
            "mieux si": col("better_if"),
            "n": col("n_pairs"),
            f"moy. {a}": col(f"{al}_mean"),
            f"moy. {b}": col(f"{bl}_mean"),
            "différence HL [IC 95 %]": [
                ci(lo, hi, m)
                for lo, hi, m in zip(
                    col("hl_ci_low"), col("hl_ci_high"), col("hl_diff"), strict=True
                )
            ],
            "changement %": col("rel_change_pct"),
            "p": col("p_value"),
            "p (Holm)": col("p_holm"),
            "effet": col("effect"),
            "alerte équité": col("equity_flag"),
        }
    )
    return out


def said(prefix: str, message: str | None, default: str) -> str:
    """ "<prefix> : <message>." without repeating a prefix the API notice already has."""
    msg = (message or default).strip().rstrip(".")
    if msg.lower().startswith(prefix.lower()):
        return msg[0].upper() + msg[1:] + "."
    return f"{prefix} : {msg}."


def strip_label(message: str, label: str) -> str:
    """An API notice without its leading "<label> : " (several labels in one banner)."""
    for sep in (" : ", ": "):
        if message.startswith(label + sep):
            return message[len(label) + len(sep) :]
    return message


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def fmt_age(seconds) -> str:
    if seconds is None or pd.isna(seconds):
        return "n.d."
    s = float(seconds)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.0f} h"
    return f"{s / 86400:.0f} j"


def fmt_ts(value) -> str:
    try:
        return f"{pd.Timestamp(value):%d/%m/%Y %H:%M:%S}"
    except (TypeError, ValueError):
        return "n.d."


def footer_text(payload: dict | None) -> str:
    """ENF-05 / ENF-04 footer from an envelope: source, calibration date, provisional
    program, timestamp and age."""
    p = payload or {}
    parts = [f"Données simulées (source : {p.get('data_source') or 'simulation'})"]
    parts.append(f"capteurs réels jusqu'au {p.get('calibrated_on') or 'n.d.'}")
    if p.get("provisional", True):
        parts.append("programme de feux provisoire")
    parts.append(f"données du {fmt_ts(p.get('timestamp'))} (âge {fmt_age(p.get('data_age_s'))})")
    parts.append(client.describe())
    return " · ".join(parts)


def seeds_text(seeds: list) -> str:
    seeds = sorted(int(s) for s in seeds or [])
    if not seeds:
        return "les graines d'évaluation"
    if seeds == list(range(seeds[0], seeds[-1] + 1)) and len(seeds) > 2:
        return f"{len(seeds)} graines d'évaluation ({seeds[0]} à {seeds[-1]})"
    return f"{len(seeds)} graines d'évaluation ({', '.join(map(str, seeds))})"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- styles (Motez)
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
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p.sb-desc { font-size: .76rem; margin: -.25rem 0 .1rem 2.7rem;
      line-height: 1.3; }
[data-testid="stSidebar"] [class*="st-key-scn_card_"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[data-testid="stSidebar"] .stButton button { width: 100%; background: #171E26; border: 1px solid #2A3441; color: #E6EDF3;
      border-radius: 9px; font-size: .84rem; padding: .3rem .5rem; }
[data-testid="stSidebar"] .stButton button:hover { border-color: #2DD4A0; color: #2DD4A0; }
[data-testid="stSidebar"] [data-baseweb="button-group"] button, [data-testid="stSidebar"] [data-testid="stPills"] button {
      border-radius: 999px; }
"""


def scn_card_css(colors: dict[str, str]) -> str:
    return "".join(
        f".st-key-scn_card_{s} {{ border: 1px solid {BORDER}; border-left: 5px solid {c}; border-radius: 10px; "
        f"background: #141B23; padding: .55rem .8rem .5rem; margin-bottom: .5rem; }} "
        f".st-key-scn_card_{s}:has(input:not(:checked)) {{ opacity: .5; }} "
        for s, c in colors.items()
    )


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
.card.abl {{ opacity: .72; }}
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

    def show_df(df, **kw):
        try:
            st.dataframe(df, width="stretch", hide_index=True, **kw)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True, **kw)

    def show_chart(ch):
        ch = (
            ch.configure_view(strokeWidth=0)
            .configure(background="transparent")
            .configure_axis(
                labelColor=MUTED,
                titleColor=MUTED,
                gridColor="#222C37",
                domainColor=BORDER,
                tickColor=BORDER,
            )
            .configure_legend(labelColor=TEXT, titleColor=MUTED)
            .configure_header(labelColor=TEXT, titleColor=TEXT)
        )
        try:
            st.altair_chart(ch, width="stretch")
        except Exception:
            st.altair_chart(ch, use_container_width=True)

    # ------------------------------------------------------------------ lecture de l'API (12.1)
    cache = st.session_state.setdefault("_api_cache", {})

    def api(path: str, **params) -> client.Reply:
        """`client.fetch`, kept CACHE_TTL_S per session; errors are never kept."""
        key = (client.identity(), path, tuple(sorted(params.items())))
        hit = cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
            return hit[1]
        rep = client.fetch(path, **params)
        if rep.state != "error":
            cache[key] = (time.monotonic(), rep)
        return rep

    def header(subtitle: str) -> None:
        st.markdown(
            '<p class="title">Urban Traffic Twin</p>'
            f'<p class="subtitle">{subtitle}</p><div class="lane"></div>',
            unsafe_allow_html=True,
        )

    st.markdown(CSS, unsafe_allow_html=True)
    with st.spinner("Chargement des données de l'API…"):
        sc = api("/scenarios")
    if sc.state == "error":
        header("Carrefour Bd de l'Hôpital × Bd Saint-Marcel, Paris 13e.")
        st.error(
            f"Données temporairement indisponibles : {sc.message}. Démarrez l'API "
            "(`python tasks.py demo`), puis rechargez la page."
        )
        st.caption(f"Données simulées · source attendue : {client.describe()}")
        st.stop()

    # ------------------------------------------------------------------ panneau de contrôle (barre latérale)
    sb = st.sidebar
    sb.markdown(
        '<div class="brand"><div class="signal"><i></i><i></i><i></i></div>'
        "<div><b>Panneau de contrôle</b><span>Bd de l'Hôpital × Bd Saint-Marcel</span></div></div>",
        unsafe_allow_html=True,
    )

    families = families_of(sc.payload)
    sb.markdown('<div class="sb-h">Famille de runs</div>', unsafe_allow_html=True)
    if hasattr(sb, "segmented_control"):
        family = sb.segmented_control(
            "Famille",
            families,
            default=families[0],
            key="family",
            format_func=lambda f: FAM_LABELS.get(f, f),
            label_visibility="collapsed",
        )
    else:
        family = sb.radio(
            "Famille",
            families,
            horizontal=True,
            format_func=lambda f: FAM_LABELS.get(f, f),
            label_visibility="collapsed",
        )
    family = family or families[0]
    sb.markdown(f'<p class="sb-note">{FAM_NOTES.get(family, "")}</p>', unsafe_allow_html=True)

    with st.spinner("Chargement des résultats de l'évaluation…"):
        ev = api("/evaluation", family=family)
        evp = ev.payload or {}
        labels = label_info(evp, sc.payload, family)
        kpi_replies = {r["label"]: api("/kpi", scenario=r["label"], family=family) for r in labels}
    kpis = kpi_frame(kpi_replies)
    info = {r["label"]: r for r in labels}
    order = [r["label"] for r in labels]
    colors = color_map(labels)
    st.markdown("<style>" + SIDEBAR_CSS + scn_card_css(colors) + "</style>", unsafe_allow_html=True)

    def preset(names):
        for x in order:
            st.session_state[f"scn_{x}"] = x in names

    sb.markdown(
        '<div class="sb-h">Scénarios<small>ordre du registre d\'évaluation</small></div>',
        unsafe_allow_html=True,
    )
    b1, b2 = sb.columns(2)
    b1.button("Tous", key="p_all", on_click=preset, args=(order,))
    b2.button("Duel S0 vs S2", key="p_duel", on_click=preset, args=(["S0", "S2"],))
    for x in order:
        st.session_state.setdefault(f"scn_{x}", True)
    chosen = []
    for x in order:
        role = ROLE_NAMES.get(info[x].get("role") or "")
        with sb.container(key=f"scn_card_{x}"):
            if st.toggle(
                f"**{x}**  {label_name(info[x])}" + (f" · {role}" if role else ""), key=f"scn_{x}"
            ):
                chosen.append(x)
            st.markdown(f'<p class="sb-desc">{DESC.get(x, "")}</p>', unsafe_allow_html=True)
    sb.markdown(
        '<p class="sb-note">S0 reste la référence des écarts, même désactivé. Les ablations et '
        "les variantes exploratoires sont atténuées.</p>",
        unsafe_allow_html=True,
    )

    users = sorted(u for u in kpis.user.dropna().unique())
    sb.markdown('<div class="sb-h">Usagers</div>', unsafe_allow_html=True)
    if users and hasattr(sb, "pills"):
        sel_users = sb.pills(
            "Usagers",
            users,
            selection_mode="multi",
            default=users,
            key=f"users_{family}",
            format_func=lambda u: USER_LABELS.get(u, u),
            label_visibility="collapsed",
        )
    else:
        sel_users = sb.multiselect("Usagers", users, default=users, label_visibility="collapsed")
    sel_users = list(sel_users or []) or users

    sb.markdown('<div class="sb-h">Affichage</div>', unsafe_allow_html=True)
    only_head = sb.toggle("KPI phares seulement", value=True)
    sb.markdown(
        '<p class="sb-note">Désactive pour voir les dizaines de KPI détaillés (par branche, ligne de bus, '
        "passage piéton).</p>",
        unsafe_allow_html=True,
    )

    sb.markdown('<div class="sb-h">Export</div>', unsafe_allow_html=True)
    sb.download_button(
        "Exporter tous les KPI en CSV",
        csv_bytes(kpis.assign(family=family)),
        f"kpis_{family}_tous.csv",
        "text/csv",
        key="dl_all_kpis",
    )
    if sb.button("Rafraîchir les données", key="p_refresh"):
        cache.clear()
        st.session_state.pop("_pairs", None)
        st.rerun()
    sb.markdown(
        f'<p class="sb-note">Source : {client.describe()}, évaluation du {fmt_ts(evp.get("timestamp"))}. '
        "Les autres exports sont dans l'onglet Exports.</p>",
        unsafe_allow_html=True,
    )

    if not labels:
        header("Carrefour Bd de l'Hôpital × Bd Saint-Marcel, Paris 13e.")
        st.warning(
            f"Aucun scénario évalué pour la famille {FAM_LABELS.get(family, family)} : "
            f"{ev.message or sc.message or 'aucun résultat de simulation'}."
        )
        st.caption(footer_text(evp or sc.payload))
        st.stop()
    if not chosen:
        st.info("Coche au moins un scénario dans la barre latérale.")
        st.stop()

    # ------------------------------------------------------------------ en-tête et bandeaux (12.1)
    header(
        "Carrefour Bd de l'Hôpital × Bd Saint-Marcel, Paris 13e. "
        f"{len(order)} façons de piloter les mêmes feux, sur {seeds_text(evp.get('seeds'))}."
    )
    env = evp or next((r.payload for r in kpi_replies.values() if r.payload), sc.payload) or {}
    prov = (
        " Le programme de feux est supposé, pas mesuré (programme de feux provisoire) : seuls "
        "les débits entrants sont validés. La visite terrain doit encore mesurer le phasage réel "
        "et les files d'attente."
        if env.get("provisional", True)
        else ""
    )
    st.markdown(
        f'<div class="notice"><b>Données simulées.</b> Demande calibrée sur des capteurs réels '
        f"jusqu'au {env.get('calibrated_on') or 'n.d.'}.{prov}</div>",
        unsafe_allow_html=True,
    )
    if ev.state == "error":
        st.error(f"Évaluation : données temporairement indisponibles ({ev.message}).")
    elif ev.state == "unavailable":
        st.warning(said("Évaluation", ev.message, "pas encore évaluée"))
    elif ev.state == "degraded":
        st.warning(f"Mode dégradé (évaluation) : {ev.message or 'résultats partiels'}.")
    degraded = {lab: r for lab, r in kpi_replies.items() if r.state == "degraded"}
    if degraded:
        notices = sorted({r.message or "résultats partiels" for r in degraded.values()})
        st.warning(f"Mode dégradé ({', '.join(degraded)}) : {' ; '.join(notices)}.")
    missing = {lab: r for lab, r in kpi_replies.items() if r.state in ("unavailable", "error")}
    if missing:
        notices = sorted(
            {strip_label(r.message or "pas encore évalué", lab) for lab, r in missing.items()}
        )
        st.info(f"Sans résultat pour l'instant : {', '.join(missing)} ({' ; '.join(notices)}).")
    stale = [r for r in [ev, *kpi_replies.values()] if r.payload and r.payload.get("stale")]
    if stale:
        age = max((r.payload.get("data_age_s") or 0) for r in stale)
        st.warning(f"Dernière valeur connue : données anciennes (âge {fmt_age(age)}).")

    # ------------------------------------------------------------------ rang et cartes
    def row(lab, user, kpi):
        r = kpis[(kpis.label == lab) & (kpis.user == user) & (kpis.kpi == kpi)]
        return r.iloc[0] if len(r) else None

    def val(lab, user, kpi):
        r = row(lab, user, kpi)
        return float(r["mean"]) if r is not None and pd.notna(r["mean"]) else None

    def gap(lab, user, kpi):
        r = row(lab, user, kpi)
        return float(r["rel_pct"]) if r is not None and pd.notna(r["rel_pct"]) else None

    rankable = [x for x in order if val(x, "all", "person_delay_mean") is not None]
    ranked = sorted(
        rankable,
        key=lambda s: (
            -(round(val(s, "all", "person_delay_mean") or 0, 1)),
            -(val(s, "ambulance", "travel_time_total") or 0),
        ),
    )
    n = len(ranked)
    main_ranked = [x for x in ranked if (info[x].get("role") or "main") == "main"]

    def delta_html(g, lower_better=True):
        if g is None:
            return '<span class="delta flat">référence</span>'
        if abs(g) < 0.05:
            return '<span class="delta flat">= S0</span>'
        cls = "good" if (g < 0) == lower_better else "bad"
        return f'<span class="delta {cls}">{g:+.1f} % vs S0</span>'

    def fmt(v):
        return "n.d." if v is None else f"{v:.1f}"

    cards = ranked + [x for x in order if x not in ranked]
    per_row = len(cards) if len(cards) <= 4 else 3  # Motez's 4 cards per row, 3 + 3 for 6
    cols = [c for i in range(0, len(cards), per_row) for c in st.columns(per_row)]
    for col, scn in zip(cols, cards, strict=False):
        role = info[scn].get("role") or "main"
        off = "" if scn in chosen else " off"
        abl = " abl" if role != "main" and not off else ""
        role_tag = f" · {ROLE_NAMES[role]}" if role in ROLE_NAMES else ""
        if scn in ranked:
            rk = ranked.index(scn) + 1
            # "le meilleur" / "le moins bon" among the main scenarios only (ablations and
            # exploratory runs are ranked but never named best, as in the /mock Operator)
            tag = (
                (
                    " · le moins bon"
                    if scn == main_ranked[0] and len(main_ranked) > 1
                    else (" · le meilleur (hors ablations)" if scn == main_ranked[-1] else "")
                )
                if scn in main_ranked
                else ""
            )
            pw, amb, car = (
                val(scn, "all", "person_delay_mean"),
                val(scn, "ambulance", "travel_time_total"),
                val(scn, "car", "time_loss_mean"),
            )
            html = (
                f'<div class="card{off}{abl}" style="--c:{colors[scn]}"><div class="rank">Rang {rk}/{n}{tag}{role_tag}</div>'
                f'<div class="name">{scn}</div><div class="sub">{label_name(info[scn])}</div>'
                f'<div class="big">{fmt(pw)}<small>s</small></div><div class="lab">Retard moyen pondéré par personne</div>'
                f"{delta_html(gap(scn, 'all', 'person_delay_mean') if scn != BASELINE else None)}"
                f'<div class="mini"><div><b>{fmt(car)} s</b><span>Voiture, perte de temps</span></div>'
                f"<div><b>{fmt(amb)} s</b><span>Ambulance, trajet total</span></div></div></div>"
            )
        else:
            why = kpi_replies[scn].message or "pas encore évalué"
            html = (
                f'<div class="card off" style="--c:{colors[scn]}"><div class="rank">Non classé{role_tag}</div>'
                f'<div class="name">{scn}</div><div class="sub">{label_name(info[scn])}</div>'
                f'<div class="big">n.d.</div><div class="lab">{why}</div></div>'
            )
        col.markdown(html, unsafe_allow_html=True)
    st.caption(
        "Rangs calculés sur le retard pondéré (arrondi à 0,1 s), puis départagés par le trajet de l'ambulance. "
        "Aucun scénario n'est meilleur pour tous les usagers : voir l'onglet Équité."
    )
    st.write("")

    # paires pré-enregistrées : `/evaluation.pairs` si l'API les donne, sinon sondées par /compare
    def discover_pairs() -> list[tuple[str, str, str | None, client.Reply]]:
        key = (client.identity(), family, tuple(order))
        hit = st.session_state.get("_pairs")
        if hit and hit[0] == key and time.monotonic() - hit[1] < PAIRS_TTL_S:
            return hit[2]
        listed = [(p.get("a"), p.get("b")) for p in evp.get("pairs") or [] if isinstance(p, dict)]
        cands = listed or [(a, b) for a in order for b in order if a != b]
        found = []
        for a, b in cands:
            rep = api("/compare", a=a, b=b, family=family)
            p = rep.payload or {}
            if rep.state != "error" and (p.get("rows") or p.get("role")):
                found.append((a, b, p.get("role"), rep))
        st.session_state["_pairs"] = (key, time.monotonic(), found)
        return found

    with st.spinner("Chargement des paires comparées…"):
        pairs = discover_pairs()

    final_kpis, final_extra = final_frames(evp.get("final_table") or [], family)
    better_if = {}
    for _, _, _, rep in pairs:
        for r in (rep.payload or {}).get("rows") or []:
            if r.get("better_if"):
                better_if.setdefault(
                    (r.get("user_type") or r.get("user"), r.get("kpi")), r["better_if"]
                )
    if not final_kpis.empty and "better_if" in final_kpis.columns:
        for r in final_kpis.itertuples():
            if isinstance(r.better_if, str):
                better_if.setdefault((r.user, r.kpi), r.better_if)
    headline = (
        set(zip(final_kpis.user, final_kpis.kpi, strict=True))
        if not final_kpis.empty
        else {(u, k) for u, k, _ in HEADLINE}
    )

    tabs = st.tabs(
        [
            "Comparaison",
            "Tous les KPI",
            "Tests et paires",
            "Équité",
            "Sécurité et ambulance",
            "Courbes",
            "Sensibilité des poids",
            "Exports",
        ]
    )
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs

    # ------------------------------------------------------------------ 1. comparaison (tableau final)
    with tab1:
        if not final_kpis.empty:
            st.markdown("#### Tableau comparatif final (Partie 8)")
            comp = final_kpis.copy()
            source_note = "Tableau final de l'évaluation (<code>/evaluation</code>)"
        else:
            st.info(
                "Tableau comparatif final : pas encore évalué (Partie 8). KPI phares lus dans `/kpi`."
            )
            head = pd.DataFrame(HEADLINE, columns=["user", "kpi", "name"])
            comp = kpis.merge(head[["user", "kpi"]], on=["user", "kpi"])
            source_note = "KPI phares de <code>/kpi</code>"
        comp = comp[comp.user.isin(sel_users) & comp.label.isin(chosen)].copy()
        if comp.empty:
            st.info("Aucun KPI phare pour ces usagers.")
        else:
            comp["name"] = [kpi_name(u, k) for u, k in zip(comp.user, comp.kpi, strict=True)]
            comp["mean"] = pd.to_numeric(comp["mean"], errors="coerce")
            comp["sd"] = pd.to_numeric(comp["sd"], errors="coerce")
            comp["rel_pct"] = pd.to_numeric(comp["rel_pct"], errors="coerce")
            row_order = (
                list(zip(final_kpis.user, final_kpis.kpi, strict=True))
                if not final_kpis.empty
                else [(u, k) for u, k, _ in HEADLINE]
            )
            rank = {key: i for i, key in enumerate(dict.fromkeys(row_order))}
            comp = comp.assign(
                _o=[rank.get((u, k), len(rank)) for u, k in zip(comp.user, comp.kpi, strict=True)]
            ).sort_values("_o", kind="stable")
            names = list(dict.fromkeys(comp["name"]))
            ns = sorted({int(x) for x in pd.to_numeric(comp["n"], errors="coerce").dropna()})
            n_txt = f"n = {', '.join(map(str, ns))} graines" if ns else "les graines d'évaluation"
            comp["txt"] = [
                "n.d." if pd.isna(m) else f"{m:.4g}" + (f" ± {s:.3g}" if pd.notna(s) else "")
                for m, s in zip(comp["mean"], comp["sd"], strict=True)
            ]
            wide = comp.pivot_table(
                index=["user", "kpi", "name"],
                columns="label",
                values="txt",
                aggfunc="first",
                sort=False,
            )
            wide = wide[[s for s in order if s in wide.columns]].reset_index()
            for s in [s for s in order if s in chosen and s != BASELINE]:
                gp = comp[comp.label == s].set_index(["user", "kpi"])["rel_pct"]
                wide[f"Δ {s} % vs S0"] = [
                    round(gp.get((u, k)), 1)
                    if (u, k) in gp.index and pd.notna(gp.get((u, k)))
                    else None
                    for u, k in zip(wide.user, wide.kpi, strict=True)
                ]
            wide["mieux si"] = [
                better_if.get((u, k), "") for u, k in zip(wide.user, wide.kpi, strict=True)
            ]
            st.markdown(
                f'<p class="section-note">{source_note}. Moyenne ± écart-type, {n_txt} ; Δ = changement '
                "relatif à S0 calculé par l'évaluation. Ablations et variantes exploratoires : "
                f"{', '.join(x for x in order if (info[x].get('role') or 'main') != 'main') or 'aucune'}.</p>",
                unsafe_allow_html=True,
            )
            show_df(wide.drop(columns=["name"]).rename(columns={"user": "usager", "kpi": "KPI"}))

            st.markdown("#### Valeurs moyennes par KPI")
            st.markdown(
                '<p class="section-note">Barre d\'erreur = un écart-type. Le sens « mieux » de chaque KPI '
                "est donné dans le tableau ci-dessus.</p>",
                unsafe_allow_html=True,
            )
            v = comp.rename(columns={"label": "scenario", "name": "label"})
            v = v.assign(lo=v["mean"] - v["sd"].fillna(0), hi=v["mean"] + v["sd"].fillna(0))
            v["role"] = [info[s].get("role") or "main" for s in v.scenario]
            dom = [s for s in order if s in chosen]
            color = alt.Color(
                "scenario:N",
                scale=alt.Scale(domain=dom, range=[colors[s] for s in dom]),
                legend=alt.Legend(title=None, orient="top"),
                sort=dom,
            )
            opacity = alt.condition(alt.datum.role == "main", alt.value(1.0), alt.value(0.55))
            base = alt.Chart(v).encode(
                x=alt.X("scenario:N", sort=dom, title=None, axis=alt.Axis(labelAngle=0))
            )
            bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                y=alt.Y("mean:Q", title=None),
                color=color,
                opacity=opacity,
                tooltip=[
                    "label",
                    "scenario",
                    alt.Tooltip("mean:Q", format=".1f"),
                    alt.Tooltip("sd:Q", format=".1f"),
                ],
            )
            err = base.mark_errorbar(color="#C9D3DD").encode(y=alt.Y("lo:Q", title=None), y2="hi:Q")
            ch = (
                (bars + err)
                .properties(width=150, height=190)
                .facet(facet=alt.Facet("label:N", title=None, sort=names), columns=4)
                .resolve_scale(y="independent")
            )
            show_chart(ch)

            st.markdown("#### Écart par rapport à S0 (%)")
            st.markdown(
                '<p class="section-note">Vers la gauche = valeur plus basse que S0, donc mieux '
                "pour les KPI « plus bas = mieux ».</p>",
                unsafe_allow_html=True,
            )
            g = v[(v.scenario != BASELINE)].dropna(subset=["rel_pct"])
            if g.empty:
                st.info("Choisis un scénario autre que S0 pour voir les écarts.")
            else:
                dom2 = [s for s in order if s in g.scenario.unique()]
                gc = (
                    alt.Chart(g)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        y=alt.Y("label:N", title=None, sort=names, axis=alt.Axis(labelLimit=260)),
                        x=alt.X("rel_pct:Q", title="Écart vs S0 (%)"),
                        yOffset=alt.YOffset("scenario:N", sort=dom2),
                        color=alt.Color(
                            "scenario:N",
                            scale=alt.Scale(domain=dom2, range=[colors[s] for s in dom2]),
                            legend=alt.Legend(title=None, orient="top"),
                        ),
                        opacity=opacity,
                        tooltip=[
                            "label",
                            "scenario",
                            alt.Tooltip("mean:Q", format=".1f"),
                            alt.Tooltip("rel_pct:Q", title="écart vs S0 %", format="+.1f"),
                        ],
                    )
                    .properties(height=max(280, 46 * g.label.nunique()))
                )
                show_chart(gc)

    # ------------------------------------------------------------------ 2. tous les KPI
    with tab2:
        q = st.text_input(
            "Rechercher un KPI", placeholder="par exemple wait, crossing, arm_S, delay…"
        )
        v = kpis[kpis.user.isin(sel_users) & kpis.label.isin(chosen)]
        if only_head:
            v = v[[(u, k) in headline for u, k in zip(v.user, v.kpi, strict=True)]]
        if q:
            v = v[
                v.kpi.str.contains(q, case=False, regex=False)
                | v.user.str.contains(q, case=False, regex=False)
            ]
        if v.empty:
            st.info("Aucun KPI ne correspond à ces filtres.")
        else:
            v = v.assign(
                txt=[
                    "n.d." if pd.isna(m) else f"{m:.4g}" + (f" ± {s:.3g}" if pd.notna(s) else "")
                    for m, s in zip(v["mean"], v["sd"], strict=True)
                ]
            )
            wide = v.pivot_table(
                index=["user", "kpi"], columns="label", values="txt", aggfunc="first", sort=False
            )
            wide = wide[[s for s in order if s in wide.columns]].reset_index()
            keys = list(zip(wide.user, wide.kpi, strict=True))
            wide.insert(2, "mieux si", [better_if.get(k, "") for k in keys])
            for s in [s for s in order if s in chosen and s != BASELINE]:
                sub = v[v.label == s].set_index(["user", "kpi"])
                gp, wf = sub["rel_pct"], sub["worse_flag"]
                wide[f"Δ {s} %"] = [
                    round(gp.get(k), 1) if k in gp.index and pd.notna(gp.get(k)) else None
                    for k in keys
                ]
                if s in wide.columns:
                    wide[s] = [
                        t + ("  ⚠" if k in wf.index and wf.get(k) is True else "")
                        for t, k in zip(wide[s].fillna(""), keys, strict=True)
                    ]
            wide = wide.rename(columns={"unit": "unité", "user": "usager", "kpi": "KPI"})
            ns = sorted({int(x) for x in pd.to_numeric(v["n"], errors="coerce").dropna()})
            st.markdown(
                f'<p class="section-note">{len(wide)} KPI. Moyenne ± écart-type (n = '
                f"{', '.join(map(str, ns)) or 'n.d.'}). Δ = changement relatif à S0 donné par l'API ; "
                "⚠ = plus de 5 % pire que S0 (règle d'équité pré-enregistrée).</p>",
                unsafe_allow_html=True,
            )
            show_df(wide)
            st.download_button(
                "Télécharger ce tableau (CSV)",
                csv_bytes(wide),
                f"kpis_{family}.csv",
                "text/csv",
                key="dl_kpis",
            )

    # ------------------------------------------------------------------ 3. tests et paires
    with tab3:
        a06 = evp.get("a06")
        if family == "main" or (a06 and a06.get("family") == family):
            st.markdown("#### A-06 : S3 contre les références (Partie 8)")
            if not a06:
                st.info("A-06 : pas encore évalué (Partie 8).")
            else:
                comps = pd.DataFrame(a06.get("comparisons") or [])
                prim = a06.get("primary")
                sentence = next(
                    (
                        c.get("sentence")
                        for c in a06.get("comparisons") or []
                        if c.get("reference") == prim
                    ),
                    None,
                )
                st.markdown(
                    f"**A-06 {a06.get('status', 'n.d.')}** (statut procédural : les trois comparaisons "
                    "sont valides et montrées, quel que soit leur sens)."
                    + (f" Comparaison principale vs {prim} : {sentence}" if sentence else "")
                )
                for prob in a06.get("problems") or []:
                    st.warning(f"A-06 : {prob}")
                if not comps.empty:
                    show_df(comps)
                rep = a06.get("replication_10_6") or evp.get("replication_10_6")
                if rep:
                    st.markdown("#### Vérification 10.6 sur les graines d'évaluation (Partie 8)")
                    checks = []
                    for c in ("c1", "c2", "c3", "c4"):
                        d = rep.get(c) or {}
                        ok = d.get("pass") if "pass" in d else d.get("cited")
                        checks.append({"critère": c.upper(), "résultat": ok})
                    show_df(pd.DataFrame(checks))
                    if rep.get("text"):
                        st.caption(rep["text"])
                    if rep.get("main_solution"):
                        st.caption(
                            f"Solution principale : {rep['main_solution']} (décision de la partie 7, "
                            "A-06 ne la rouvre pas)."
                        )
        st.markdown("#### Paires comparées (Holm)")
        shown_pairs = [
            p for p in pairs if p[0] in chosen + [BASELINE] and p[1] in chosen + [BASELINE]
        ]
        if not shown_pairs:
            st.info(
                "Aucune paire évaluée pour ces scénarios. Coche S0 et au moins un autre scénario."
                if pairs
                else "Paires : pas encore évaluées."
            )
        else:
            names = [
                f"{a} vs {b}" + (f" · {role}" if role else "") for a, b, role, _ in shown_pairs
            ]
            pick = st.selectbox("Paire comparée", names)
            a, b, role, rep = shown_pairs[names.index(pick)]
            rows = (rep.payload or {}).get("rows") or []
            pp = pair_frame(rows, a, b)
            if rep.state != "ok" and rows:
                st.warning(f"Paire {a} vs {b} : {rep.message or rep.state}.")
            if pp.empty:
                st.info(said(f"Paire {a} vs {b}", rep.message, "pas encore évaluée"))
            else:
                k_head = int(pp["phare"].eq(True).sum())
                pp = pp[pp.user.isin(sel_users)]
                if only_head and pp["phare"].notna().any():
                    pp = pp[pp["phare"].eq(True)]
                nn = sorted({int(x) for x in pd.to_numeric(pp["n"], errors="coerce").dropna()})
                st.markdown(
                    f'<p class="section-note">Rôle de la paire : <b>{role or "n.d."}</b>. Test de Wilcoxon '
                    f"exact bilatéral apparié (n = {', '.join(map(str, nn)) or 'n.d.'} graines). "
                    f"<b>Holm ne porte que sur les {k_head} KPI phares</b> (colonne « phare ») : pour les "
                    "autres lignes, seul le p brut a un sens.</p>",
                    unsafe_allow_html=True,
                )
                show_df(pp)
                pp = pp.assign(label=pp.user + " · " + pp.kpi)
                chart = (
                    alt.Chart(pp.dropna(subset=["changement %"]))
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
                        x=alt.X("changement %:Q", title="Changement (%)"),
                        color=alt.Color(
                            "effet:N",
                            title=None,
                            scale=alt.Scale(
                                domain=["better", "worse", "same"],
                                range=["#2DD4A0", "#FF7A7F", "#8B98A8"],
                            ),
                        ),
                        tooltip=["label", "changement %", "p", "p (Holm)"],
                    )
                    .properties(height=max(200, 30 * len(pp)))
                )
                show_chart(chart)
                st.download_button(
                    "Télécharger les écarts (CSV)",
                    csv_bytes(pd.DataFrame(rows)),
                    f"ecarts_{family}_{a}__vs__{b}.csv",
                    "text/csv",
                    key="dl_pair",
                )

    # ------------------------------------------------------------------ 4. équité
    with tab4:
        refs = [r for r in (BASELINE, "S1", "S2") if r in order]
        ref = st.radio("Référence", refs, horizontal=True, key="eq_ref") if refs else BASELINE
        if ref == BASELINE:
            st.markdown(
                '<p class="section-note">Chaque KPI plus de 5 % pire que S0 est signalé. Le retard pondéré peut '
                "s'améliorer alors que des sous-groupes (une branche, une ligne de bus, un passage piéton) se "
                "dégradent : c'est la « dégradation masquée ». Le p est brut, non ajusté.</p>",
                unsafe_allow_html=True,
            )
            hd = pd.DataFrame(evp.get("hidden_degradation") or [])
            if not hd.empty:
                if "family" in hd.columns:
                    hd = hd[hd["family"] == family]
                if "scenario" in hd.columns:
                    hd = hd[hd["scenario"].isin(chosen)]
                if "flagged" in hd.columns:
                    hd = hd.assign(
                        flagged=[len(f) if isinstance(f, list) else f for f in hd.flagged]
                    )
                show_df(hd.drop(columns=[c for c in ("family",) if c in hd.columns]))
            base_mean = kpis[kpis.label == BASELINE].set_index(["user", "kpi"])["mean"]
            e = kpis[(kpis.worse_flag == True) & kpis.label.isin(chosen)].copy()  # noqa: E712
            if e.empty:
                st.info("Aucun KPI signalé pour ces scénarios.")
            else:
                e["S0"] = [base_mean.get((u, k)) for u, k in zip(e.user, e.kpi, strict=True)]
                counts = e.groupby("label").size()
                st.markdown(
                    "".join(
                        f'<span class="pill" style="--c:{colors[s]}">{s} : {int(counts[s])} KPI signalés</span>'
                        for s in order
                        if s in counts.index
                    ),
                    unsafe_allow_html=True,
                )
                st.write("")
                only_sub = st.toggle(
                    "Sous-groupes principaux seulement (branches, lignes de bus, passages)",
                    value=True,
                )
                ee = e[e.kpi.str.contains("__")] if only_sub else e
                ee = ee.rename(
                    columns={"label": "scénario", "rel_pct": "changement %", "mean": "moyenne"}
                )[["scénario", "user", "kpi", "S0", "moyenne", "changement %", "p"]]
                show_df(ee.sort_values("changement %", ascending=False))
                st.download_button(
                    "Télécharger l'équité (CSV)",
                    csv_bytes(ee),
                    f"equite_{family}_vs_S0.csv",
                    "text/csv",
                    key="dl_eq",
                )
        else:
            st.markdown(
                f'<p class="section-note">KPI plus de 5 % pires que {ref} (paires pré-enregistrées '
                f"contre {ref} ; p brut, non ajusté).</p>",
                unsafe_allow_html=True,
            )
            eqv = ((evp.get("a06") or {}).get("equity_vs") or {}).get(ref)
            if eqv:
                show_df(pd.DataFrame(eqv))
            recs = []
            for a, b, _, rep in pairs:
                if a != ref or b not in chosen:
                    continue
                for r in (rep.payload or {}).get("rows") or []:
                    if r.get("equity_flag") is True:
                        recs.append(
                            {
                                "scénario": b,
                                **{
                                    k: r.get(k)
                                    for k in ("user_type", "kpi", "rel_change_pct", "p_value")
                                },
                            }
                        )
            if recs:
                show_df(pd.DataFrame(recs))
            elif not eqv:
                st.info(
                    f"Aucun KPI signalé contre {ref} pour ces scénarios (ou paire pas encore évaluée)."
                )

    # ------------------------------------------------------------------ 5. sécurité et ambulance
    with tab5:
        a03, a04, a05 = evp.get("a03") or {}, evp.get("a04") or {}, evp.get("a05") or {}
        a06 = evp.get("a06") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "A-03 contrôleurs",
            a03.get("text") or "n.d.",
            ", ".join(a03.get("controllers") or []) or None,
            delta_color="off",
        )
        c2.metric(
            "A-04 sécurité",
            a04.get("verdict") or "n.d.",
            f"{a04['runs']} runs" if a04.get("runs") is not None else None,
            delta_color="off",
        )
        prim5 = a05.get("primary") or {}
        p5 = prim5.get("p_one_sided", prim5.get("p"))
        c3.metric(
            "A-05 ambulance",
            a05.get("verdict") or "n.d.",
            f"HL {prim5['hl']:+.1f} s, p = {p5:.4f}"
            if prim5.get("hl") is not None and p5 is not None
            else None,
            delta_color="off",
        )
        c4.metric(
            "A-06 RL (S3)",
            a06.get("status") or "n.d.",
            f"principale vs {a06['primary']}" if a06.get("primary") else "pas encore évalué",
            delta_color="off",
        )
        if prim5.get("hl") is not None:
            ci5 = prim5.get("hl_ci") or [None, None]
            ci_txt = f", intervalle [{ci5[0]:+.1f} ; {ci5[1]:+.1f}]" if None not in ci5[:2] else ""
            st.caption(
                f"A-05 : {prim5.get('b', 'S2')} contre {prim5.get('a', 'S0')}, estimateur de "
                f"Hodges-Lehmann {prim5['hl']:+.1f} s{ci_txt}, test de Wilcoxon exact unilatéral, "
                f"n = {prim5.get('n', 'n.d.')}. Une seule ambulance par simulation : puissance "
                "limitée, la famille Sonde est la preuve robuste."
            )
        st.markdown("#### A-04 : sécurité de chaque scénario")
        fails = a04.get("failures") or []
        safety = (
            final_extra[final_extra.section == "safety"] if not final_extra.empty else final_extra
        )
        if not safety.empty:
            show_df(safety[safety.label.isin(chosen)].dropna(axis=1, how="all"))
        elif a04:
            st.markdown(
                f"A-04 **{a04.get('verdict', 'n.d.')}** sur {a04.get('runs', 'n.d.')} runs"
                + (f", {len(fails)} défaut(s)." if fails else ", aucun défaut.")
            )
        else:
            st.info("A-04 : pas encore évalué.")
        if fails:
            show_df(
                pd.DataFrame(fails)
                if isinstance(fails[0], dict)
                else pd.DataFrame({"défaut": fails})
            )

        st.markdown("#### A-05 : ambulance, graine par graine")
        seeds5 = prim5.get("seeds") or []
        av, bv = prim5.get("a_values") or [], prim5.get("b_values") or []
        if family == prim5.get("family", "main") and seeds5 and len(av) == len(bv) == len(seeds5):
            la, lb = prim5.get("a", "S0"), prim5.get("b", "S2")
            long = pd.DataFrame(
                {
                    "graine": seeds5 * 2,
                    "scénario": [la] * len(seeds5) + [lb] * len(seeds5),
                    "secondes": list(av) + list(bv),
                }
            )
            lc = (
                alt.Chart(long)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("graine:O", title="Graine"),
                    y=alt.Y("secondes:Q", title="Trajet depuis le départ prévu (s)"),
                    xOffset="scénario:N",
                    color=alt.Color(
                        "scénario:N",
                        scale=alt.Scale(
                            domain=[la, lb],
                            range=[colors.get(la, COLORS["S0"]), colors.get(lb, COLORS["S2"])],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=["graine", "scénario", "secondes"],
                )
                .properties(height=260)
            )
            show_chart(lc)
            slower = [s for s, x, y in zip(seeds5, av, bv, strict=True) if y > x]
            st.caption(
                f"Graines où {lb} est plus lent que {la} : "
                + (", ".join(map(str, slower)) if slower else "aucune")
                + "."
            )
        elif family != "main":
            st.info("Le détail graine par graine n'existe que pour la famille principale.")
        else:
            st.info("A-05 : détail graine par graine pas encore fourni par l'API.")
        sec = a05.get("secondary") or []
        if sec:
            st.markdown("#### A-05 : tests secondaires")
            sdf = pd.DataFrame(sec)
            keep = [c for c in A05_COLUMNS if c in sdf.columns]
            show_df(sdf[keep] if keep else sdf)

        st.markdown("#### Activité du Safety Guard et du contrôleur")
        act = (
            final_extra[final_extra.section == "activity"] if not final_extra.empty else final_extra
        )
        if act.empty:
            st.info("Activité du guard et du contrôleur : pas encore évaluée (Partie 8).")
        else:
            act = act[act.label.isin(chosen)]
            wide = act.pivot_table(
                index="kpi", columns="label", values="mean", aggfunc="first", sort=False
            )
            wide = wide[[s for s in order if s in wide.columns]].reset_index()
            show_df(wide.rename(columns={"kpi": "compteur"}))
            st.caption(
                "Une intervention du guard annule une demande d'un contrôleur. « ped_yield » compte les sauts de "
                "préemption que le bouclier piéton a remplacés par la phase suivante."
            )
        occ = (
            final_extra[final_extra.section == "occupancy"]
            if not final_extra.empty
            else final_extra
        )
        if not occ.empty:
            st.markdown("#### Retard par personne selon l'occupation des bus")
            occ = occ[occ.label.isin(chosen)]
            wide = occ.pivot_table(
                index="kpi", columns="label", values="mean", aggfunc="first", sort=False
            )
            show_df(wide[[s for s in order if s in wide.columns]].reset_index())

    # ------------------------------------------------------------------ 6. courbes (Plotly)
    with tab6:
        import plotly.graph_objects as go

        approach = st.selectbox(
            "Branche", list(APPROACHES), format_func=lambda k: APPROACHES[k], key="series_approach"
        )
        with st.spinner("Chargement des courbes…"):
            srs = api(
                "/series",
                family=family,
                metric="halting_veh",
                approach=approach,
                labels=",".join(chosen),
            )
        pts = pd.DataFrame((srs.payload or {}).get("points") or [])
        if srs.state == "error":
            st.error(f"Courbes : données temporairement indisponibles ({srs.message}).")
        elif pts.empty or not {"label", "t_s", "mean"} <= set(pts.columns):
            st.info(said("Courbes", srs.message, "pas encore évaluées"))
        else:
            if srs.state in ("degraded", "stale"):
                st.warning(said("Courbes", srs.message, srs.state))
            fig = go.Figure()
            for lab in [s for s in order if s in set(pts.label)]:
                d = pts[pts.label == lab].sort_values("t_s")
                x = d["t_s"] / 60.0
                c = colors.get(lab, MUTED)
                if {"p10", "p90"} <= set(d.columns):
                    fig.add_trace(
                        go.Scatter(
                            x=list(x) + list(x[::-1]),
                            y=list(d["p90"]) + list(d["p10"][::-1]),
                            fill="toself",
                            fillcolor=rgba(c, 0.15),
                            line={"width": 0},
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{lab} p10–p90",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=x, y=d["mean"], mode="lines", name=lab, line={"color": c, "width": 2}
                    )
                )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": TEXT},
                legend={"orientation": "h", "y": 1.08},
                margin={"l": 10, "r": 10, "t": 30, "b": 10},
                xaxis={
                    "title": "temps depuis le début de la simulation (min)",
                    "gridcolor": "#222C37",
                },
                yaxis={"title": "véhicules à l'arrêt", "gridcolor": "#222C37"},
                height=420,
            )
            step = (srs.payload or {}).get("step_s")
            st.markdown(
                f'<p class="section-note">{APPROACHES[approach]} : moyenne sur les graines d\'évaluation, '
                f"bande p10–p90" + (f", pas de {step} s" if step else "") + ".</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, key="series_chart")

    # ------------------------------------------------------------------ 7. sensibilité des poids
    with tab7:
        st.markdown("#### Sensibilité des poids (étude RL Partie 7)")
        with st.spinner("Chargement de l'étude de sensibilité…"):
            sens = api("/sensitivity")
        srows = pd.DataFrame((sens.payload or {}).get("rows") or [])
        if sens.state == "error":
            st.error(f"Sensibilité : données temporairement indisponibles ({sens.message}).")
        elif srows.empty:
            st.info(said("Sensibilité", sens.message, "non disponible"))
        else:
            st.markdown(
                '<p class="section-note">Étude RL Partie 7 : chaque réglage des poids de la récompense, '
                "ses KPI et leur écart au réglage de base. Ce n'est pas un résultat de l'évaluation sur "
                "les graines d'évaluation.</p>",
                unsafe_allow_html=True,
            )
            show_df(srows)
            if (sens.payload or {}).get("source"):
                st.caption(f"Source : {sens.payload['source']}")
            st.download_button(
                "Télécharger la sensibilité (CSV)",
                csv_bytes(srows),
                "sensibilite_poids_rl7.csv",
                "text/csv",
                key="dl_sens",
            )

    # ------------------------------------------------------------------ 8. exports (téléchargements)
    with tab8:
        st.markdown(
            '<p class="section-note">Fichiers CSV construits en mémoire à partir des réponses de l\'API '
            "(rien n'est écrit sur le disque).</p>",
            unsafe_allow_html=True,
        )
        ft = pd.DataFrame(evp.get("final_table") or [])
        if ft.empty:
            st.info("Tableau comparatif final : pas encore évalué (Partie 8).")
        else:
            st.download_button(
                "Tableau comparatif final (CSV)",
                csv_bytes(ft),
                "final_table.csv",
                "text/csv",
                key="dl_final",
            )
        st.download_button(
            f"KPI de la famille {FAM_LABELS.get(family, family)} (CSV)",
            csv_bytes(kpis.assign(family=family)),
            f"kpis_{family}_tous.csv",
            "text/csv",
            key="dl_kpis_all",
        )
        for a, b, role, rep in pairs:
            rows = (rep.payload or {}).get("rows") or []
            if rows:
                st.download_button(
                    f"Paire {a} vs {b}" + (f" · {role}" if role else "") + " (CSV)",
                    csv_bytes(pd.DataFrame(rows)),
                    f"{family}_{a}__vs__{b}.csv",
                    "text/csv",
                    key=f"dl_pair_{a}__{b}",
                )

    # ------------------------------------------------------------------ pied de page (ENF-04, ENF-05)
    st.divider()
    st.caption(footer_text(env))


if __name__ == "__main__":
    main()
