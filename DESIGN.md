# DESIGN.md · « Carte vivante »

Direction de design des deux interfaces du jumeau numérique Pitié-Salpêtrière
(carrefour Bd de l'Hôpital × Bd Saint-Marcel, Paris 13e) :

- l'application usagers `src/traffic_twin/app/mock.html`, servie à `/mock` ;
- l'Opérateur Streamlit `app.py`, avec `.streamlit/config.toml`.

Ce document remplace l'ancienne analyse « Apple ». Les deux interfaces sont en français,
en mode clair. Elles sont pensées d'abord pour un vidéoprojecteur (soutenance) et pour
des testeurs non experts.

## 1. Principes

1. **La carte d'abord.** Dans l'app usagers, le plan 2D du carrefour reste toujours
   visible. Sur ordinateur il est en fond, et la réponse est un panneau posé dessus (à
   gauche). Sur mobile il occupe le haut de l'écran (40 vh), et la réponse est une
   bottom sheet avec une poignée. Opérateur et Admin sont des documents (grands tableaux)
   et s'affichent sans carte.
2. **Signalétique, pas décoration.** Les panneaux sont des cartouches : aplat blanc,
   bord gauche épais de 4 px dans la couleur de sens, typographie grande.
3. **Le trafic se lit par niveaux** (fluide, dense, saturé, bloqué) avant les chiffres.
   Chaque niveau a une couleur et un motif.
4. **La simulation se voit sans gêner.** Les hachures diagonales (`--sim-hatch`) sont
   la signature du « simulé » partout : bandeau, pastille « Provisoire », pied de
   panneau, pied de page, fond de la mini-courbe, notes du cockpit, bandeau de
   l'Opérateur.
5. **Une vue = une réponse.** Chaque vue a un élément principal (un verbe d'action et
   un chiffre) et au plus deux éléments secondaires. Pas de tableau de bord à douze
   cartes égales.

## 2. Tokens (`:root` de mock.html)

### Plan de ville

| Token | Valeur | Usage |
|---|---|---|
| `--bg-map` | `#E9ECE6` | îlots, fond de carte, fond des pages |
| `--bg-building` | `#DCDFD8` | bâtiments |
| `--bg-hospital` | `#D3D7CE` | emprise de la Pitié-Salpêtrière (nommée) |
| `--road` | `#FFFFFF` | chaussée |
| `--road-edge` | `#C9CDC4` | bord de chaussée, marquages légers |
| `--crosswalk-band` | `#B8BDB3` | bandes des passages piétons |

### Surfaces et texte

| Token | Valeur | Usage |
|---|---|---|
| `--surface` | `#FFFFFF` | panneaux |
| `--surface-2` | `#F5F6F3` | fonds secondaires (faits, en-têtes de tableau) |
| `--ink` | `#121A21` | texte principal |
| `--ink-2` | `#4A545E` | texte secondaire (aucun gris plus clair pour du texte) |
| `--line` | `#D4D8CF` | séparateurs décoratifs uniquement |
| `--line-strong` | `#7A848D` | bordures des contrôles (champs, boutons, segments) |
| `--accent` | `#14365C` | bleu marine signalétique : actions, sélection, focus |
| `--accent-soft` | `#E3EAF2` | fond de sélection, meilleure valeur du cockpit |

### Niveaux de trafic

Le niveau vient des véhicules à l'arrêt par branche, donnés par `/replay` (`queues`).
Les seuils servent seulement à l'affichage et sont écrits dans la légende :

| Niveau | Véhicules à l'arrêt | Couleur | Motif sur la carte |
|---|---|---|---|
| Fluide | 0 à 4 | `#17773F` | trait plein |
| Dense | 5 à 9 | `#8A5A00`, remplissage `#E3A21A` | tirets longs bordés de `#8A5A00` |
| Saturé | 10 à 15 | `#C62828` | trait double |
| Bloqué | 16 et plus | `#6E1414` | trait plein épais et chevrons blancs |

La bande couvre la longueur de la file derrière la ligne d'arrêt (environ 7,5 m par
véhicule, au moins 40 unités de carte).

### Sens et statut

Ces couleurs ont un sens fixe. Elles ne servent jamais à décorer :

| Token | Valeur | Sens |
|---|---|---|
| `--go` / `--go-soft` | `#17773F` / `#E4F1E8` | passer, gain |
| `--caution` / `--caution-soft` | `#8A5A00` / `#FBF1DC` | prudence, données partielles ou anciennes |
| `--stop` / `--stop-soft` | `#C62828` / `#FBE9E7` | arrêt, perte, erreur, ambulance |
| `--ai` / `--ai-soft` | `#5E3BA3` / `#EEE9F7` | prédiction (ETA, intervalle conforme) |
| `--sim-hatch` | `repeating-linear-gradient(135deg, #121A21 à 8 % 0 1px, transparent 1px 6px)` | « simulé » |

### Scénarios (couleur, forme et trait, jamais la couleur seule)

La même table est utilisée par l'app usagers (`SCN`) et par l'Opérateur (`COLORS`,
`SHAPES`, `PLOTLY_SYMBOLS`, `DASHES`, `WIDTHS`). Un test vérifie qu'elles restent
identiques.

| Scénario | Couleur | Marqueur | Trait |
|---|---|---|---|
| S0 (référence, temps fixe) | `#6B7280` | ○ cercle vide | plein fin (1,5) |
| S1 (actuated) | `#2C7DA0` | ■ carré | plein (2) |
| S2 (règles, **retenue**) | `#14365C` | ◆ losange | plein épais (3,5) et badge « Retenue » |
| S3 (RL) | `#5E3BA3` | ▲ triangle | plein (2) |
| S4 (S3 sans préemption) | `#A07FD6` | △ triangle vide | pointillés |
| S2_noP (S2 sans préemption) | `#5D7A99` | ◇ losange vide | pointillés |

Le badge « Retenue » suit `replication_10_6.main_solution` de `/evaluation`. Il n'est
pas écrit en dur.

### Typographie

- Texte : `"Atkinson Hyperlegible Next", "Atkinson Hyperlegible", system-ui, sans-serif`.
- Chiffres et horloge : `"Atkinson Hyperlegible Mono", "IBM Plex Mono", ui-monospace,
  monospace`, avec `font-variant-numeric: tabular-nums`.
- Échelle (rem) : 0,8125 / 0,875 / 1 / 1,125 / 1,375 / 1,75 / 2,5 / 3,5.
- Réponse principale : 2,5 rem, graisse 700. Elle passe à 3,5 rem au-delà de 1280 px
  quand elle est courte (retard du bus). Les messages longs du conseil de départ
  descendent à 1,75 rem. Interligne : 1,45 pour le texte, 1,1 pour les grands chiffres.

### Espacements, rayons, ombre

- Espacements : 4 / 8 / 12 / 16 / 24 / 32 / 48 px.
- Rayons : 4 px (badges, pastilles), 8 px (panneaux, contrôles), 12 px (bottom sheet).
- Une seule ombre, `--shadow-panel`, pour détacher les panneaux et les contrôles posés
  sur la carte.
- Bord de sens : 4 px à gauche des panneaux de réponse (`.answer.go|caution|stop|ai`).

## 3. Contrastes vérifiés (WCAG 2.2)

Ratios calculés avec la formule WCAG de luminance relative.

| Premier plan | Fond | Ratio | Exigence |
|---|---|---|---|
| `--ink` | `--surface` | 17,57 | texte 4,5:1 ✓ |
| `--ink` | `--bg-map` | 14,73 | texte ✓ |
| `--ink-2` | `--surface` | 7,72 | texte ✓ |
| `--ink-2` | `--surface-2` | 7,11 | texte ✓ |
| `--ink-2` | `--bg-map` | 6,47 | texte ✓ |
| blanc | `--accent` | 12,27 | texte des boutons et pilules actifs ✓ |
| `--accent` | `--accent-soft` | 10,12 | texte ✓ |
| `--go` | `--go-soft` | 4,82 | texte des pastilles ✓ |
| `--caution` | `--caution-soft` | 5,28 | texte ✓ |
| `--stop` | `--stop-soft` | 4,80 | texte ✓ |
| `--ai` | `--ai-soft` | 6,70 | texte ✓ |
| `--go` / `--stop` / `--caution` / `--ai` | `--surface` | 5,60 / 5,62 / 5,93 / 7,98 | texte ✓ |
| blanc | `--go` / `--stop` | 5,60 / 5,62 | texte ✓ |
| `--line-strong` | `--surface` | 3,81 | bordure de contrôle 3:1 ✓ |
| `--line-strong` | `--surface-2` | 3,51 | bordure ✓ |
| voiture `#56606B` | chaussée blanche | 6,40 | objet graphique 3:1 ✓ |
| `--ink` | remplissage dense `#E3A21A` | 7,90 | ✓ |
| remplissage dense `#E3A21A` | chaussée | 2,22 | **insuffisant seul** : bordé de `#8A5A00` (5,93) et doublé d'un motif |
| S0 / S1 / S2 / S3 | `--surface` | 4,83 / 4,62 / 12,27 / 7,98 | graphique ✓ |
| S2_noP / S4 | `--surface` | 4,46 / 3,23 | graphique ✓, **jamais pour du texte** (le libellé reste en `--ink`) |
| chaussée, bord de chaussée | `--bg-map` | 1,2 / 1,35 | décor du plan, pas porteur d'information |

## 4. Composants

### App usagers

- **Barre supérieure** (56 px) : logo, titre, pastille d'état de la vue, pastille
  hachurée « Provisoire », sélecteur « Scénario · graine ».
- **Bandeau simulation** (28 px, hachuré) : « SIMULATION · programme de feux provisoire
  (visite terrain à venir) · calibré le JJ/MM/AAAA ». On peut le replier en une pastille
  « Simulé » qui reste visible. Le choix est gardé dans `localStorage` (dans un
  try/catch).
- **Pilules de profil** : 44 px de haut, marine quand le profil est actif,
  `aria-current="page"`, défilement horizontal sur mobile.
- **Carte** (`#trafficCanvas`) :
  - plan 2D par défaut ; la vue 3D (three.js r128 servi en local) est proposée dans
    Relecture seulement ;
  - zoom de 0,5 à 6 à la molette de la souris (environ ×1,5 par cran, animé en douceur
    sur environ 0,2 s), centré sur le point visé (le point sous la souris reste en
    place) ; glisser déplace le plan (borné), un clic sans glisser
    sélectionne toujours un véhicule ; boutons +, − et recentrer (aussi au clavier, sur
    la carte : +, −, 0) ;
  - en 3D, rotation, inclinaison (environ 26° à 88°), distance et déplacement sont
    bornés ;
  - feux affichés avec leur couleur et leur lettre (V, O, R ou ? si l'état n'est pas
    fourni) ;
  - véhicules vus de dessus : voiture ardoise, bus encre avec « BUS ligne », ambulance
    blanche bordée de rouge avec une croix et un halo pulsé ;
  - piétons en points ; traînée d'une seconde à 30 % ;
  - un clic ouvre une fiche latérale (type, vitesse, état, identifiant) ; le temps
    d'attente n'est pas fourni par `/replay` et la fiche le dit ;
  - légende repliable ; liste des objets au clavier ; résumé `aria-live` limité à un
    toutes les 5 s.
- **Horloge** :
  - play/pause (44 px) ; horloge mono de 1,75 rem ; vitesses ×1, ×5, ×20 en contrôle
    segmenté avec `aria-pressed` ;
  - curseur avec une graduation tous les quarts d'heure ;
  - marqueurs d'événements du run lus dans `/replay/{run}/events` : croix pour la
    demande de préemption de l'ambulance, carré pour la priorité bus, triangle pour le
    pic de file. Un clic saute à l'instant ;
  - mini-courbe de la file totale du run (un `/replay` toutes les 2 min) sur fond
    hachuré ;
  - clavier : Espace pour lecture/pause ; ←/→ pour ±1 s et Maj pour ±10 s ; Début/Fin
    sur le curseur. `aria-valuetext` vaut par exemple « 8 heures 15 ».
- **Pastilles d'état** : icône, texte et couleur ; 28 px ; rayon 4 px. Une infobulle
  (survol et focus) donne la source, la date de calibration et l'âge des données. Les
  états : À jour, Partiel, Données de il y a X min, Indisponible, Erreur, Chargement.
- **Panneau de réponse** : en-tête, contrôles, bannières d'état, réponse principale
  (bord de sens), au plus deux sections secondaires, puis le pied hachuré « Simulé ·
  source · scénario graine · âge ».
- **Cartes KPI** (Opérateur et cockpit) : valeur mono, flèche ▼/▲, signe explicite et
  « vs S0 ». Vert = gain et rouge = perte, selon le sens de l'indicateur.

### Cockpit admin

- **Tableau** :
  - en-tête et première colonne collants, défilement interne ;
  - cellule : valeur mono alignée à droite, écart vs S0 en dessous ;
  - meilleure valeur : fond `--accent-soft`, étoile et texte masqué « meilleure valeur » ;
  - drapeau d'équité : icône balance et infobulle ; test de Holm suivi de « * » sous
    α = 0,05, avec légende ;
  - colonne de la solution retenue entourée d'un filet marine ;
  - densité confortable (44 px par ligne), ou compacte (36 px) sur demande.
- **Comparaison synchronisée** :
  - élément principal : quatre mini-cartes en 2 × 2, titrées par la forme et la couleur
    du scénario ;
  - secondaires : frises de phases empilées (V, O, R et hachures de dégagement) et
    phrases générées à partir des événements ;
  - les fiches d'identité et le radar passent dans un tiroir latéral ouvert par
    « Fiche » ; le radar a toujours un tableau équivalent.

### Opérateur Streamlit

- `.streamlit/config.toml` : `primaryColor #14365C`, fond `#E9ECE6`, surfaces blanches,
  texte `#121A21`.
- Un seul bloc CSS injecté : polices Atkinson, bandeau « SIMULATION » hachuré et collant,
  cartes scénario à bord gauche de 4 px avec la forme du scénario et « Retenue »,
  métriques en mono, onglets soulignés en marine (3 px, 44 px de haut).
- Graphiques :
  - thème Altair partagé (`show_chart`) : police, axes, mention « Simulé » dans le coin ;
  - barres d'écart doublées d'un marqueur à la forme du scénario ;
  - courbes Plotly avec couleur, symbole, pointillés et épaisseur par scénario ; bande
    p10–p90 à 15 % d'opacité ; annotation « Simulé ».
- Chaque onglet se termine par la légende « Simulé · source · calibration · âge ».

## 5. Les cinq états de chaque vue

| État | Rendu |
|---|---|
| Chargement | squelette aux dimensions finales et « Chargement de la simulation… » ; carte grise sans véhicule, feux « ? » |
| Dégradé | la vue s'affiche quand même ; bannière ambre « Partiel : … » ; l'élément manquant devient un cadre en pointillés qui explique pourquoi (par exemple : intervalle ETA absent pour les runs RL) |
| Périmé | bannière « Dernière valeur connue (âge …) » avec « Actualiser » ; pastille horloge « Données de il y a X min » |
| Indisponible | cercle barré, phrase claire, bouton « Choisir un autre scénario » |
| Erreur | message humain, « Détail technique » repliable (code, HTTP), bouton « Réessayer » ; jamais de trace d'appel |
| Vide | phrase explicite (« Aucun bus de la ligne 91 au carrefour à 08:15:00 dans cette simulation ») et saut au prochain événement du run quand il existe |

Le paramètre `?etat=loading|degraded|stale|unavailable|error` montre un état sur
n'importe quelle vue. C'est une démonstration : la réponse réelle de l'API est seulement
présentée comme dans cet état.

## 6. Accessibilité

- Cibles de 44 × 44 px au moins. Les marqueurs de la frise font 24 px, avec une zone de
  clic invisible élargie.
- Aucune information portée par la couleur seule : lettre sur les feux, motif sur les
  niveaux, forme sur les scénarios, icône et texte sur les pastilles, flèche et signe sur
  les écarts.
- Focus visible partout (contour de 3 px `--accent`, décalage de 2 px), carte comprise.
  Un lien d'évitement mène à la réponse.
- `prefers-reduced-motion` : pas de halo pulsé ni de traînée ; véhicules par pas de
  1 s ; animations décoratives de la 3D coupées.
- La carte a un équivalent textuel : résumé `aria-live` et liste des objets. La réponse
  principale est annoncée au plus une fois toutes les 5 s.
- `lang="fr"`, nombres au format français (virgule, moins typographique), heures sur 24 h.
- Vidéoprojecteur : texte de 16 px au moins dans les panneaux, rien de plus clair que
  `--ink-2`, traits de 2 px au moins pour les courbes.

## 7. À ne pas faire

- Pas de thème sombre néon, de teal, de lueur, de dégradé décoratif ni de glassmorphism.
- Pas d'emoji comme icône : seulement des SVG au trait, inline (`ICONS`).
- Le vert, l'ambre, le rouge et le violet ne décorent jamais.
- Pas de valeur simulée sans marque « Simulé » dans la même vue.
- Ne jamais présenter le programme de feux comme définitif.
- Pas de 3D par défaut, pas de caméra libre, pas d'animation qui bloque l'interaction.
- Pas de dépendance ajoutée côté usagers (three.js seulement) ; pas de composant
  Streamlit tiers.
- Pas de texte anglais visible.
- Ne pas toucher à l'API, au chemin `/mock`, aux points d'accroche de test
  (`data-field`, `data-view-state`, `<main id="view">`, `<footer>`) ni au temps de
  réponse (< 500 ms).

## 8. Écarts assumés

- **Carte 2D en canvas, pas en three.js orthographique.** Le plan 2D reste le canvas
  existant (testé, plus léger) ; three.js sert la vue 3D. Le rendu attendu est obtenu :
  vue de dessus, zoom borné.
- **Pas de mode sombre.** Il était permis, pas demandé. Le plan canvas est dessiné en
  couleurs claires fixes.
- **Opérateur et Admin sans carte en fond.** Leurs tableaux larges ne tiennent pas dans
  un panneau de 380 px.
- **Pas d'« attente » dans la fiche objet** ni de « temps gagné vs sans préemption »
  dans la vue Ambulance : l'API v1 ne les fournit pas (`/replay` positions,
  `/ambulance`), et la page n'invente aucune valeur.
- **« Partez à 08:22 ».** La vue Étudiant affiche le conseil de l'API (« Partez
  maintenant, marge confortable ») et l'heure d'arrivée prévue, pas une heure de départ
  recalculée dans la page.
- **Heure d'examen.** Le champ `type="time"` suit la langue du navigateur ; un
  navigateur en anglais l'affiche en AM/PM.
