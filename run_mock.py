#!/usr/bin/env python3
"""Serveur Mock Cockpit Autonome & Dynamique (Pitié-Salpêtrière).

Gère l'interface mock.html et produit des réponses API réactives et cohérentes :
- Réagit au temps de simulation `t` (07:30 -> 09:30, horloge et bouton play)
- Réagit au changement de ligne de bus (91, 24, 57) et décompte le temps jusqu'au passage
- Réagit aux passages piétons (S, W, WB, NB) et vitesses (normale, lente, réduite)
- Simule le déplacement de l'ambulance SAMU et l'onde verte V2I en temps réel
- Calcule l'ETA étudiant, l'intervalle CQR et la marge selon le trajet et l'heure d'examen
"""

from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "src" / "traffic_twin" / "app"
MOCK_HTML = APP_DIR / "mock.html"
THREE_JS = APP_DIR / "three.min.js"

HOST = "127.0.0.1"
PORT = 8000

# Géométrie des passages piétons (longueur en mètres et fenêtres de vert sur cycle 90s)
CROSSING_SPECS = {
    "S": {
        "name": "Passage Sud (Bd de l'Hôpital)",
        "length_m": 18.4,
        "green_start": 0,
        "green_end": 28,
        "clearance_end": 33,
    },
    "W": {
        "name": "Passage Ouest (Bd Saint-Marcel)",
        "length_m": 14.8,
        "green_start": 42,
        "green_end": 70,
        "clearance_end": 75,
    },
    "WB": {
        "name": "Passage Ouest Couloir Bus",
        "length_m": 8.5,
        "green_start": 42,
        "green_end": 74,
        "clearance_end": 78,
    },
    "NB": {
        "name": "Passage Nord Couloir Bus",
        "length_m": 7.2,
        "green_start": 0,
        "green_end": 32,
        "clearance_end": 36,
    },
}

SPEED_VALUES = {"normal": 1.2, "slow": 1.0, "reduced": 0.8}

BUS_SPECS = {
    "91": {
        "name": "Ligne 91 (Montparnasse <-> Bastille / Gare de Lyon)",
        "flow": "Sens Est (Gare de Lyon -> Montparnasse)",
        "headway_s": 360.0,  # 6 min
        "base_delay": 12.0,
        "rush_multiplier": 22.0,
    },
    "24": {
        "name": "Ligne 24 (Panthéon <-> Gare d'Austerlitz)",
        "flow": "Sens Ouest (Panthéon -> Austerlitz)",
        "headway_s": 480.0,  # 8 min
        "base_delay": 16.0,
        "rush_multiplier": 30.0,
    },
    "57": {
        "name": "Ligne 57 (Arcueil Laplace <-> Gare d'Austerlitz)",
        "flow": "Sens Nord (Arcueil -> Austerlitz)",
        "headway_s": 600.0,  # 10 min
        "base_delay": 22.0,
        "rush_multiplier": 40.0,
    },
}

KPI_DATA = {
    "S0": {
        "scenario": "S0",
        "rows": [
            {
                "user": "car",
                "kpi": "time_loss_mean",
                "mean": 75.9,
                "rel_change_vs_s0_pct": 0.0,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "bus",
                "kpi": "delay_mean",
                "mean": 61.4,
                "rel_change_vs_s0_pct": 0.0,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "pedestrian",
                "kpi": "ped_curb_wait_max",
                "mean": 86.0,
                "rel_change_vs_s0_pct": 0.0,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "ambulance",
                "kpi": "travel_time_total",
                "mean": 141.5,
                "rel_change_vs_s0_pct": 0.0,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "all",
                "kpi": "person_delay_mean",
                "mean": 62.7,
                "rel_change_vs_s0_pct": 0.0,
                "worse_flag": False,
                "n": 10,
            },
        ],
    },
    "S1": {
        "scenario": "S1",
        "rows": [
            {
                "user": "car",
                "kpi": "time_loss_mean",
                "mean": 68.2,
                "rel_change_vs_s0_pct": -10.1,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "bus",
                "kpi": "delay_mean",
                "mean": 55.0,
                "rel_change_vs_s0_pct": -10.4,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "pedestrian",
                "kpi": "ped_curb_wait_max",
                "mean": 89.5,
                "rel_change_vs_s0_pct": 4.1,
                "worse_flag": True,
                "n": 10,
            },
            {
                "user": "ambulance",
                "kpi": "travel_time_total",
                "mean": 128.0,
                "rel_change_vs_s0_pct": -9.5,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "all",
                "kpi": "person_delay_mean",
                "mean": 56.1,
                "rel_change_vs_s0_pct": -10.5,
                "worse_flag": False,
                "n": 10,
            },
        ],
    },
    "S2": {
        "scenario": "S2",
        "rows": [
            {
                "user": "car",
                "kpi": "time_loss_mean",
                "mean": 61.8,
                "rel_change_vs_s0_pct": -18.6,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "bus",
                "kpi": "delay_mean",
                "mean": 51.2,
                "rel_change_vs_s0_pct": -16.6,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "pedestrian",
                "kpi": "ped_curb_wait_max",
                "mean": 87.2,
                "rel_change_vs_s0_pct": 1.4,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "ambulance",
                "kpi": "travel_time_total",
                "mean": 105.4,
                "rel_change_vs_s0_pct": -25.5,
                "worse_flag": False,
                "n": 10,
            },
            {
                "user": "all",
                "kpi": "person_delay_mean",
                "mean": 50.8,
                "rel_change_vs_s0_pct": -19.0,
                "worse_flag": False,
                "n": 10,
            },
        ],
    },
}


def compute_rush_factor(sim_t: float) -> float:
    """Courbe de pointe du trafic : pic centré vers 08:15 (t=2700s)."""
    return math.exp(-0.5 * ((sim_t - 2700.0) / 1400.0) ** 2)


class MockAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass  # Mode silencieux pour garder la console claire

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # 1. Pages statiques
        if path in ("/", "/mock"):
            self.send_file(MOCK_HTML, "text/html; charset=utf-8")
            return
        if path == "/three.min.js":
            self.send_file(THREE_JS, "application/javascript")
            return

        # 2. Temps de simulation (par défaut 08:15 = 2700s)
        sim_t = float(qs.get("t", [2700])[0]) if "t" in qs else 2700.0
        rush = compute_rush_factor(sim_t)

        # Horodatage dynamique calculé (base 07:30:00 = 27000s)
        total_clock_s = int(27000 + sim_t)
        hh = (total_clock_s // 3600) % 24
        mm = (total_clock_s % 3600) // 60
        ss = total_clock_s % 60
        current_iso = f"2026-10-22T{hh:02d}:{mm:02d}:{ss:02d}+02:00"

        if path == "/health":
            self.send_json(
                {
                    "status": "ok",
                    "data_source": "simulation Pitié-Salpêtrière (S0/S1/S2/S3)",
                    "calibrated_on": "2026-10-14",
                    "provisional": False,
                    "timestamp": current_iso,
                    "sim_t_s": sim_t,
                    "data_age_s": 0.5,
                    "default_run": {
                        "run_id": "pitie_rules_seed4000",
                        "scenario": "pitie_rules",
                        "seed": 4000,
                        "has_eta": True,
                    },
                }
            )
            return

        if path == "/scenarios":
            self.send_json(
                {
                    "scenarios": [
                        {
                            "scenario": "pitie_fixed",
                            "label": "S0 (Feux Fixes)",
                            "runs": [
                                {"run_id": "pitie_fixed_seed1", "seed": 1, "has_eta": False},
                                {"run_id": "pitie_fixed_seed4000", "seed": 4000, "has_eta": False},
                            ],
                        },
                        {
                            "scenario": "pitie_actuated",
                            "label": "S1 (Feux Actués)",
                            "runs": [
                                {"run_id": "pitie_actuated_seed1", "seed": 1, "has_eta": False}
                            ],
                        },
                        {
                            "scenario": "pitie_rules",
                            "label": "S2 (Règles IA)",
                            "runs": [
                                {"run_id": "pitie_rules_seed4000", "seed": 4000, "has_eta": True}
                            ],
                        },
                    ]
                }
            )
            return

        if path == "/replay/default":
            self.send_json(
                {
                    "status": "ok",
                    "data_source": "simulation Pitié-Salpêtrière (relecture)",
                    "calibrated_on": "2026-10-14",
                    "provisional": False,
                    "timestamp": current_iso,
                    "sim_t_s": sim_t,
                    "data_age_s": 1.0,
                    "run": {"scenario": "pitie_rules", "seed": 4000},
                }
            )
            return

        # -------------------------------------------------------------
        # 🚶 PIÉTON : passage et vitesse dynamiques + cycle de feu
        # -------------------------------------------------------------
        if path.startswith("/crossing/"):
            crossing_id = path.split("/")[-1].upper()
            spec = CROSSING_SPECS.get(crossing_id, CROSSING_SPECS["S"])
            speed_key = qs.get("speed", ["normal"])[0]
            speed_mps = SPEED_VALUES.get(speed_key, 1.2)

            length_m = spec["length_m"]
            walk_time_s = round(length_m / speed_mps, 1)

            # Cycle de feux de 90 secondes
            cycle = sim_t % 90.0
            g_start = spec["green_start"]
            g_end = spec["green_end"]
            c_end = spec["clearance_end"]

            if g_start <= cycle < g_end:
                signal = "green"
                green_left = round(g_end - cycle, 1)
                next_green = 0.0
                can_cross = green_left >= walk_time_s
            elif g_end <= cycle < c_end:
                signal = "clearance"
                green_left = 0.0
                next_green = round((90.0 - cycle + g_start) % 90.0, 1)
                can_cross = False
            else:
                signal = "red"
                green_left = 0.0
                next_green = round((g_start - cycle + 90.0) % 90.0, 1)
                can_cross = False

            self.send_json(
                {
                    "status": "ok",
                    "crossing": crossing_id,
                    "signal": signal,
                    "can_cross": can_cross,
                    "green_remaining_s": green_left,
                    "next_green_in_s": next_green,
                    "length_m": length_m,
                    "walk_time_s": walk_time_s,
                    "calibrated_on": "2026-10-14",
                    "provisional": False,
                    "timestamp": current_iso,
                    "sim_t_s": sim_t,
                    "data_age_s": 0.8,
                }
            )
            return

        # -------------------------------------------------------------
        # 🚌 BUS : lignes 91, 24, 57 avec intervalles et retards réactifs
        # -------------------------------------------------------------
        if path.startswith("/bus/"):
            line = path.split("/")[2] if len(path.split("/")) > 2 else "91"
            bspec = BUS_SPECS.get(line, BUS_SPECS["91"])
            headway_s = bspec["headway_s"]

            # Temps avant prochain passage (décompte dynamique selon sim_t)
            time_to_next = headway_s - (sim_t % headway_s)
            next_arrival_s = round(time_to_next, 1)
            next_min = math.ceil(next_arrival_s / 60.0)

            if next_arrival_s < 60:
                next_arrival_text = f"{int(next_arrival_s)} SEC"
            else:
                next_arrival_text = f"~{next_min} MIN"

            # Retard calculé selon l'heure de pointe et la ligne
            delay_s = round(
                bspec["base_delay"]
                + bspec["rush_multiplier"] * rush
                + 4.0 * math.sin(sim_t / 50.0),
                1,
            )
            bus_idx = int(sim_t // headway_s) % 8 + 1

            self.send_json(
                {
                    "status": "ok",
                    "line": line,
                    "vehicle_id": f"bus_{line}_{bus_idx}",
                    "flow": bspec["flow"],
                    "delay_s": delay_s,
                    "headway_s": headway_s,
                    "scheduled_headway_s": headway_s,
                    "next_arrival_s": next_arrival_s,
                    "next_arrival_text": next_arrival_text,
                    "priority_active": True,
                    "message": f"Télématique RATP · Priorité feux V2I active pour la ligne {line}.",
                    "calibrated_on": "2026-10-14",
                    "provisional": False,
                    "timestamp": current_iso,
                    "sim_t_s": sim_t,
                    "data_age_s": 1.2,
                }
            )
            return

        # -------------------------------------------------------------
        # 🚑 AMBULANCE : progression de mission, distance et compte à rebours
        # -------------------------------------------------------------
        if path == "/ambulance/next-intersections":
            # Cycle d'intervention récurrent de 10 minutes (600s)
            cycle_amb = sim_t % 600.0

            if cycle_amb < 200.0:
                # Phase Standby
                countdown_text = "STANDBY"
                unit_text = "En attente d'appel d'urgence (SAMU Pitié)"
                vehicle_id = "ambulance_SAMU_0"
                intersections = [
                    {
                        "name": "Bd de l'Hôpital x Bd Saint-Marcel",
                        "modelled": True,
                        "passed": False,
                        "status": "clear",
                        "distance_m": None,
                        "reason": "Couloir surveillé",
                    },
                    {
                        "name": "Carrefour amont Hôpital (U1)",
                        "modelled": False,
                        "passed": False,
                        "status": "clear",
                        "distance_m": None,
                        "reason": "Couloir dégagé",
                    },
                    {
                        "name": "Porte Principale Urgences Pitié (U2)",
                        "modelled": False,
                        "passed": False,
                        "status": "clear",
                        "distance_m": None,
                        "reason": "Accès réservé SAMU",
                    },
                ]
            elif cycle_amb < 340.0:
                # Phase Approche rapide (distance 560m -> 10m)
                dist = max(5.0, round((340.0 - cycle_amb) * 4.0, 1))
                t_left = max(1, int(350.0 - cycle_amb))
                countdown_text = (
                    f"00:{t_left:02d}" if t_left < 60 else f"{t_left // 60:02d}:{t_left % 60:02d}"
                )
                unit_text = "Secondes avant franchissement carrefour"
                vehicle_id = "ambulance_SAMU_0"
                intersections = [
                    {
                        "name": "Bd de l'Hôpital x Bd Saint-Marcel",
                        "modelled": True,
                        "passed": False,
                        "status": "green_wave",
                        "distance_m": dist,
                        "reason": "⚡ Onde verte V2I active",
                    },
                    {
                        "name": "Carrefour amont Hôpital (U1)",
                        "modelled": False,
                        "passed": False,
                        "status": "green_wave",
                        "distance_m": dist + 210.0,
                        "reason": "Pré-dégagement en cours",
                    },
                    {
                        "name": "Porte Principale Urgences Pitié (U2)",
                        "modelled": False,
                        "passed": False,
                        "status": "clear",
                        "distance_m": dist + 430.0,
                        "reason": "Accès prêt",
                    },
                ]
            elif cycle_amb < 380.0:
                # Phase Franchissement sous feu vert prioritaire
                countdown_text = "PRIORITÉ" if cycle_amb < 365.0 else "00:04"
                unit_text = "Franchissement sous feu vert exclusif SAMU"
                vehicle_id = "ambulance_SAMU_0"
                intersections = [
                    {
                        "name": "Bd de l'Hôpital x Bd Saint-Marcel",
                        "modelled": True,
                        "passed": False,
                        "status": "green_wave",
                        "distance_m": 0.0,
                        "reason": "Véhicule engagé dans l'intersection",
                    },
                    {
                        "name": "Carrefour amont Hôpital (U1)",
                        "modelled": False,
                        "passed": False,
                        "status": "clear",
                        "distance_m": 180.0,
                        "reason": "Voie dégagée",
                    },
                    {
                        "name": "Porte Principale Urgences Pitié (U2)",
                        "modelled": False,
                        "passed": False,
                        "status": "clear",
                        "distance_m": 390.0,
                        "reason": "Accès réservé",
                    },
                ]
            elif cycle_amb < 480.0:
                # Phase Carrefour franchi, arrivée Porte Urgences
                countdown_text = "FRANCHI"
                unit_text = "Carrefour franchi sans arrêt (-29.5% temps)"
                vehicle_id = "ambulance_SAMU_0"
                rem_dist = max(0.0, round(390.0 - (cycle_amb - 380.0) * 4.0, 1))
                intersections = [
                    {
                        "name": "Bd de l'Hôpital x Bd Saint-Marcel",
                        "modelled": True,
                        "passed": True,
                        "status": "clear",
                        "distance_m": None,
                        "reason": "Carrefour franchi avec succès",
                    },
                    {
                        "name": "Carrefour amont Hôpital (U1)",
                        "modelled": False,
                        "passed": cycle_amb > 420.0,
                        "distance_m": max(0.0, rem_dist - 200.0) if cycle_amb <= 420.0 else None,
                        "reason": "En transit",
                    },
                    {
                        "name": "Porte Principale Urgences Pitié (U2)",
                        "modelled": False,
                        "passed": False,
                        "status": "clear",
                        "distance_m": rem_dist,
                        "reason": "En approche Porte Urgences",
                    },
                ]
            else:
                # Phase Mission terminée
                countdown_text = "ARRIVÉE"
                unit_text = "Patient pris en charge aux Urgences Pitié"
                vehicle_id = "ambulance_SAMU_0"
                intersections = [
                    {
                        "name": "Bd de l'Hôpital x Bd Saint-Marcel",
                        "modelled": True,
                        "passed": True,
                        "status": "clear",
                        "distance_m": None,
                        "reason": "Mission terminée",
                    },
                    {
                        "name": "Carrefour amont Hôpital (U1)",
                        "modelled": False,
                        "passed": True,
                        "status": "clear",
                        "distance_m": None,
                        "reason": "Mission terminée",
                    },
                    {
                        "name": "Porte Principale Urgences Pitié (U2)",
                        "modelled": False,
                        "passed": True,
                        "status": "clear",
                        "distance_m": 0.0,
                        "reason": "Arrivée aux urgences",
                    },
                ]

            self.send_json(
                {
                    "status": "ok",
                    "vehicle_id": vehicle_id,
                    "countdown_text": countdown_text,
                    "unit_text": unit_text,
                    "intersections": intersections,
                    "calibrated_on": "2026-10-14",
                    "provisional": False,
                    "timestamp": current_iso,
                    "sim_t_s": sim_t,
                    "data_age_s": 0.5,
                }
            )
            return

        # -------------------------------------------------------------
        # 🎓 ÉTUDIANT : ETA, quantiles CQR et marge d'examen réactifs
        # -------------------------------------------------------------
        if path == "/eta":
            trajet = qs.get("trajet", ["N_S"])[0]
            deadline_str = qs.get("deadline", ["09:00"])[0] or "09:00"

            # Différenciation nette selon le trajet
            if trajet == "N_S":
                # Austerlitz -> Pitié : trajet plus court
                base_crossing = 130.0 + 85.0 * rush + 6.0 * math.sin(sim_t / 45.0)
                rest_of_trip_s = 120.0  # 2 min marche
                spread_low = 0.82
                spread_high = 1.25
            else:
                # Bd Saint-Marcel -> Pitié : trajet plus long
                base_crossing = 180.0 + 115.0 * rush + 8.0 * math.cos(sim_t / 55.0)
                rest_of_trip_s = 150.0  # 2.5 min marche
                spread_low = 0.85
                spread_high = 1.30

            eta_median_s = round(base_crossing, 1)
            eta_low_s = round(eta_median_s * spread_low, 1)
            eta_high_s = round(eta_median_s * spread_high, 1)

            # Calcul de la marge par rapport à l'heure d'examen choisie
            try:
                d_parts = [int(p) for p in deadline_str.split(":")]
                deadline_s = d_parts[0] * 3600 + d_parts[1] * 60
            except Exception:
                deadline_s = 9 * 3600  # 09:00

            expected_arrival_s = total_clock_s + eta_median_s + rest_of_trip_s
            margin_s = round(deadline_s - expected_arrival_s)

            if margin_s >= 900:
                advice = {
                    "code": "comfortable",
                    "message": "Partez maintenant, marge de sécurité confortable pour l'examen",
                    "margin_s": margin_s,
                }
            elif margin_s >= 300:
                advice = {
                    "code": "small_margin",
                    "message": "Départ recommandé sans attendre, marge réduite",
                    "margin_s": margin_s,
                }
            elif margin_s >= 0:
                advice = {
                    "code": "risk_of_delay",
                    "message": "Attention : risque élevé de retard à l'examen !",
                    "margin_s": margin_s,
                }
            else:
                advice = {
                    "code": "risk_of_delay",
                    "message": "Retard estimé à l'examen ! Empruntez un itinéraire accéléré",
                    "margin_s": margin_s,
                }

            self.send_json(
                {
                    "status": "ok",
                    "trajet": trajet,
                    "eta_median_s": eta_median_s,
                    "eta_low_s": eta_low_s,
                    "eta_high_s": eta_high_s,
                    "rest_of_trip_s": rest_of_trip_s,
                    "model_version": "XGBoost CQR v1 (Quantiles p10/p50/p90)",
                    "advice": advice,
                    "calibrated_on": "2026-10-14",
                    "provisional": False,
                    "timestamp": current_iso,
                    "sim_t_s": sim_t,
                    "data_age_s": 0.9,
                }
            )
            return

        # -------------------------------------------------------------
        # 📊 OPÉRATEUR : scoreboard comparatif
        # -------------------------------------------------------------
        if path == "/kpi":
            scenario = qs.get("scenario", ["S0"])[0]
            data = KPI_DATA.get(scenario, KPI_DATA["S0"])
            self.send_json(
                {
                    "status": "ok",
                    "scenario": scenario,
                    "rows": data["rows"],
                    "calibrated_on": "2026-10-14",
                    "provisional": False,
                    "timestamp": current_iso,
                    "sim_t_s": sim_t,
                    "data_age_s": 2.0,
                }
            )
            return

        self.send_json({"error": "endpoint not found"}, status=404)


def main():
    print("============================================================")
    print("🚀 Serveur Cockpit Mock Dynamique (Pitié-Salpêtrière)")
    print(f"📍 URL de l'interface : http://{HOST}:{PORT}/mock")
    print("⚡ Simulation interactive : Bus, Piétons, SAMU & Étudiant")
    print("============================================================")
    server = HTTPServer((HOST, PORT), MockAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
        server.server_close()


if __name__ == "__main__":
    main()
