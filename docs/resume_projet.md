# Résumé Complet du Projet : Urban Traffic Twin
**Jumeau numérique d'un carrefour urbain à feux dans SUMO calibré sur données réelles**  
*Document de synthèse de l'état d'avancement, de l'architecture et de la feuille de route restante*

---

## Sommaire
1. [Où trouver les informations dans le dépôt ?](#1-où-trouver-les-informations-dans-le-dépôt-)
2. [Travail Achevé : Détails Complets (Parties 1 à 4)](#2-travail-achevé--détails-complets-parties-1-à-4)
   - [Partie 1 : Données Capteurs & Choix du Carrefour](#partie-1--données-capteurs--choix-du-carrefour)
   - [Partie 2 : Modélisation SUMO & Baseline à Feux Fixes (S0)](#partie-2--modélisation-sumo--baseline-à-feux-fixes-s0)
   - [Partie 3 : Réseau Réel Pitié-Salpêtrière, Calibration & Contrôleur Actuated (S1)](#partie-3--réseau-réel-pitié-salpêtrière-calibration--contrôleur-actuated-s1)
   - [Partie 4 : Contrôleur par Règles (S2), Safety Guard & Évaluation Statistique](#partie-4--contrôleur-par-règles-s2-safety-guard--évaluation-statistique)
3. [Détails de la Base de Données (Supabase & Storage)](#3-détails-de-la-base-de-données-supabase--storage)
4. [Ce qui manque pour achever le travail (Parties 5 à 10 & Relevé Terrain)](#4-ce-qui-manque-pour-achever-le-travail-parties-5-à-10--relevé-terrain)

---

## 1. Où trouver les informations dans le dépôt ?

L'ensemble de la documentation et du code est structuré comme suit :

| Document | Emplacement | Contenu |
| :--- | :--- | :--- |
| **Cahier des charges complet** | [`docs/cahier_des_charges.docx`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/cahier_des_charges.docx) | Document officiel de cadrage (cahier des charges, personas, méthodologie, calendrier S1-S10, critères d'acceptation). |
| **Guide d'architecture technique** | [`docs/architecture.md`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/architecture.md) | Cartographie des modules, flux de données, décisions D1 à D36, invariants de sécurité. |
| **Rapport d'évaluation multi-contrôleurs** | [`docs/evaluation.md`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/evaluation.md) | Résultats des 70 runs comparant S0, S1 et S2, tests de Wilcoxon, audits A-04 et A-05, analyse d'équité. |
| **Rapport de validation sim vs réalité** | [`docs/validation.md`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/validation.md) | Confrontation des flux simulés aux boucles de comptage réelles (statistique GEH). |
| **Protocole de visite terrain** | [`docs/field_protocol.md`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/field_protocol.md) | Guide pour relever les temps de feux et les files sur place pour lever les hypothèses provisoires. |
| **Vue d'ensemble et commandes** | [`README.md`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/README.md) | Tableau de bord de la roadmap, guide d'installation et commandes d'exécution. |
| **Schéma SQL Supabase** | [`supabase/migrations/20261011000000_init_schema.sql`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/supabase/migrations/20261011000000_init_schema.sql) | DDL complet des 7 tables PostgreSQL et du bucket S3 `sim-logs`. |

---

## 2. Travail Achevé : Détails Complets (Parties 1 à 4)

Le projet est conçu de façon modulaire en 10 parties. **Les 4 premières parties sont entièrement développées, fonctionnelles et testées (831 tests passants).**

### Partie 1 : Données Capteurs & Choix du Carrefour
* **Source de données** : API ouverte Open Data de la Ville de Paris (données horaires de débit et de taux d'occupation issues des boucles électromagnétiques permanentes).
* **Pipeline de nettoyage** ([`clean.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/data/clean.py)) : Détection des anomalies, gestion des décalages d'horodatage (stamps fin d'heure), distinction trafic fluide vs saturé.
* **Sélection multicritère** ([`intersection_score.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/data/intersection_score.py)) : Comparaison de 3 carrefours parisiens. Le carrefour **Bd de l'Hôpital × Bd Saint-Marcel (Pitié-Salpêtrière, 13ᵉ)** a été sélectionné avec un score de **79.3 %** (présence de capteurs fiables sur chaque approche, lignes de bus majeures, couloir d'urgence hôpital, passages piétons).

### Partie 2 : Modélisation SUMO & Baseline à Feux Fixes (S0)
* **Carrefour témoin (Grille)** : Modélisation dans SUMO d'un carrefour avec 4 branches, tourne-à-gauche protégés, traversées piétonnes et ligne de bus avec arrêt.
* **Contrôleur S0 (Feux fixes)** ([`fixed.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/control/fixed.py)) : Cycle de 100 s avec temps de vert fixes (35 s EW, 10 s EW_LEFT, 22 s NS, 7 s NS_LEFT) et phases de dégagement intervert.
* **Extraction des KPIs par profil** ([`kpi.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/sim/kpi.py)) :
  * *Voitures* : Retard total moyen, temps perdu, temps d'attente p95, débit horaire.
  * *Bus* : Retard moyen, écart-type d'intervalle (régularité), arrêts.
  * *Ambulance* : Temps de trajet total, durée d'arrêt, retards au départ.
  * *Piétons* : Temps d'attente moyen et maximal au trottoir.

### Partie 3 : Réseau Réel Pitié-Salpêtrière, Calibration & Contrôleur Actuated (S1)
* **Réseau réel depuis OpenStreetMap** ([`osm_import.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/sim/osm_import.py)) : Extraction et curation du carrefour réel de la Pitié-Salpêtrière (`pitie.net.xml`), 17 phases de feux, 16 mouvements véhiculaires et 4 passages piétons.
* **Calibration de la demande** ([`calibration_loop.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/validation/calibration_loop.py)) : Filtrage robuste des jours ouvrés types (mardi-mercredi-jeudi hors vacances/fériés) sur le créneau 07h30-09h30. Les flux simulés reproduisent fidèlement la réalité avec une statistique GEH $< 5$.
* **Contrôleur S1 Actuated** ([`actuated.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/control/actuated.py)) : Régulation adaptative classique par détection de lacunes (gap logic) basée sur des boucles amont.

### Partie 4 : Contrôleur par Règles (S2), Safety Guard & Évaluation Statistique
* **Contrôleur par règles S2** ([`rules.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/control/rules.py)) :
  1. *Préemption Ambulance* : Détection amont du véhicule de secours ([`emergency.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/sim/emergency.py)), bascule immédiate vers la phase verte requise par le chemin le plus court et sûr.
  2. *Priorité Bus (TSP)* : Prolongation de vert ou coupure anticipée pour laisser passer les bus en approche selon leur avance/retard ([`transit.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/sim/transit.py)).
  3. *Bouclier Piéton (`PedestrianShield`)* : Force le service des piétons pour que le temps d'attente au trottoir ne dépasse jamais le seuil de 90 s.
  4. *Anti-blocage de carrefour (`JunctionMonitor`)* : Retient les véhicules en amont si la voie de sortie est encombrée pour éviter de paralyser le carrefour.
* **SafetyGuard Formel** ([`safety.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/control/safety.py)) : Filtre de sécurité matériel placé entre le contrôleur et SUMO. Il bloque toute décision dangereuse (interdiction des verts antagonistes, respect des interverts minimaux).
* **Protocole d'Évaluation Pré-enregistré** ([`evaluate.py`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/evaluation/evaluate.py)) :
  * 70 simulations menées sur 10 graines aléatoires.
  * **Audit A-04 (Sécurité)** : **PASS** (0 collision, 0 intervert violé).
  * **Audit A-05 (Ambulance)** : **PASS** (Gain médian de temps de parcours de $-36.5\text{ s}$, $p = 0.0020$).
  * Rapport détaillé produit dans [`docs/evaluation.md`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/evaluation.md).

---

## 3. Détails de la Base de Données (Supabase & Storage)

L'architecture de stockage ([`supabase/migrations/20261011000000_init_schema.sql`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/supabase/migrations/20261011000000_init_schema.sql)) combine deux technologies :

```
                        ARCHITECTURE SUPABASE
+--------------------------------------------------------------------+
|  PostgreSQL Relationnel                     Supabase Storage S3    |
|  ----------------------                     -------------------    |
|  * sensor_hourly    (capteurs Paris)        * Bucket "sim-logs"    |
|  * demand_profile   (profils par branche)   * <scenario>/<run_id>/ |
|  * run              (métadonnées & JSONB)     state_log.parquet    |
|  * trip             (trajets individuels)                          |
|  * kpi_run          (indicateurs usagers)                          |
|  * state_log        (échantillons replay)                          |
|  * eta_prediction   (prédictions ML)                               |
+--------------------------------------------------------------------+
```

1. **`sensor_hourly`** : Stocke l'historique des capteurs réels de Paris (`iu_ac`, `ts`, `flow_veh_h`, `occupancy_pct`, `traffic_state`, coordonnées GPS). Clé primaire : `(iu_ac, ts)`.
2. **`demand_profile`** : Profils de trafic par branche, heure et type de jour (`approach`, `hour`, `day_type`, `flow_veh_h`).
3. **`run`** : Métadonnées de chaque simulation (`run_id` UUID, `scenario`, `seed`, `code_version` SHA git, configuration complète en `JSONB`, chemin vers le fichier Parquet dans le Storage).
4. **`trip`** : Trajet de chaque véhicule ou piéton simulé (`depart_s`, `arrival_s`, `duration_s`, `time_loss_s`, `waiting_s`, `stops`).
5. **`kpi_run`** : Table pivot normalisée des résultats (`scenario`, `seed`, `user_type`, `kpi`, `value`).
6. **`state_log`** : Positions spatiales $(x, y)$, vitesses et états des feux au cours du temps pour alimenter le Replay 2D de l'application.
7. **`eta_prediction`** : Table réservée aux prédictions ML avec incertitude (`eta_median_s`, `eta_low_s`, `eta_high_s`).
8. **Bucket `sim-logs`** : Stockage privé Parquet conservant l'état brut pas-à-pas de toutes les simulations.
9. **Sécurité & Intégrité** : **Row Level Security (RLS)** activé sur l'ensemble des tables ; seuls les scripts backends disposant de la clé de rôle de service (`SUPABASE_SERVICE_ROLE_KEY`) peuvent lire et écrire.

---

## 4. Ce qui manque pour achever le travail (Parties 5 à 10 & Relevé Terrain)

Voici les travaux restants d'après le cahier des charges ([`docs/cahier_des_charges.docx`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/cahier_des_charges.docx), chapitre 15) :

### A. Visite de Terrain (Mesures in-situ)
* **Action** : Réaliser le relevé physique sur le carrefour Pitié-Salpêtrière selon le guide [`docs/field_protocol.md`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/docs/field_protocol.md).
* **Objectif** : Chronométrer les temps de cycle réels et mesurer les longueurs maximales de file d'attente pour faire passer le programme de feux de `ASSUMED` à `MEASURED`.

### B. Partie 5 : Modèle d'ETA (Machine Learning) & API REST v1
* **Dataset d'entraînement** : Extraction des temps de traversée sur plusieurs centaines de simulations.
* **Modèle ML (XGBoost Quantiles)** : Modèle prédisant la durée de trajet avec encadrement de l'incertitude (p10, p50 médiane, p90).
* **API REST v1** ([`src/traffic_twin/api/`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/api)) : Endpoints FastAPI exposant l'état du carrefour, les prédictions d'ETA et les KPIs.

### C. Parties 6 & 7 : Apprentissage par Renforcement (RL - Contrôleur S3)
* **Environnement Gymnasium** ([`src/traffic_twin/rl/`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/rl)) : Wrapper Gymnasium connectant SUMO à pas de décision de 5 secondes.
* **Récompense multi-objectifs** : Pénalisation des files d'attente, retards bus, attentes piétonnes et arrêts d'urgence.
* **Entraînement** : Agent DQN ou PPO sous la surveillance obligatoire du `SafetyGuard`.

### D. Partie 8 : Évaluation Comparative Finale à 10 Graines
* Comparaison rigoureuse à 10 graines aléatoires des 4 contrôleurs : **S0 (Fixe) vs S1 (Actuated) vs S2 (Règles) vs S3 (RL)**.
* Analyse statistique complète (Wilcoxon, estimateur de Hodges-Lehmann, correction de Holm).

### E. Partie 9 : Application Web Multi-Profils & Mode Replay
* **Interface utilisateur** ([`src/traffic_twin/app/`](file:///c:/Users/user/Documents/GitHub/urban-traffic-twin/src/traffic_twin/app)) : Application sans authentification proposant 5 vues spécialisées :
  1. *Étudiant* : Grand affichage de l'ETA médian $\pm$ intervalle d'incertitude et conseil d'heure de départ.
  2. *Ambulance* : État visuel contrasté du carrefour (dégagé / bloqué / onde verte) sans interaction requise.
  3. *Bus* : Avance/retard par rapport à la grille horaire, écart avec le bus amont, statut de la priorité aux feux.
  4. *Piétonne* : Temps d'attente estimé avant le prochain vert piéton.
  5. *Gestionnaire* : Replay cartographique interactif 2D rejouant les flux de véhicules et piétons.

### F. Partie 10 : Rapport Final, Support de Présentation & Démonstration
* Rédaction du mémoire final, préparation du diaporama de soutenance et répétition de la démo en direct.
