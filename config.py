"""
🍄 CARTOMORILLES — Configuration du modèle (v2.2.0)
   Zone : Grenoble / Bois des Vouillants / Mont Néron (Isère, 38)

   Corrections v2.2.0 :
     - BUG FIX : SLOPE_STEEP 4.0 → 35.0 (écrasait tous les scores >4°)
     - SLOPE_OPTIMAL élargi (0,10), SLOPE_MODERATE 20, SLOPE_MAX 45
     - ALTITUDE_OPTIMAL élargi (200, 600) — capture Vouillants 300-630m
     - Poids rééquilibrés : geology↑0.22, canopy↑0.15, tree↓0.15
     - unknown tree score 0.15 → 0.25 (88% forêts sans essence connue)
     - Alluvial bonus centré sur 350m (Vouillants) au lieu de 250m
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Final

from pyproj import Transformer

# ═══════════════════════════════════════════════════════════════
#  MÉTADONNÉES
# ═══════════════════════════════════════════════════════════════

CONFIG_VERSION = "2.2.0"

logger = logging.getLogger("cartomorilles.config")

__all__ = [
    "CONFIG_VERSION",
    "BBOX_WGS84", "BBOX", "CELL_SIZE", "MAP_CENTER", "DATA_BUFFER",
    "WEIGHTS",
    "TREE_SCORES", "TREE_ALIASES",
    "resolve_tree_name", "get_tree_score",
    "GEOLOGY_SCORES", "GEOLOGY_BRGM_MAP",
    "resolve_geology", "get_geology_score",
    "ALTITUDE_OPTIMAL", "ALTITUDE_RANGE", "ALTITUDE_ALLUVIAL_CENTER",
    "SLOPE_OPTIMAL", "SLOPE_MODERATE", "SLOPE_STEEP", "SLOPE_MAX",
    "ROUGHNESS_WINDOW", "ROUGHNESS_OPTIMAL", "ROUGHNESS_MAX",
    "DIST_WATER_OPTIMAL", "DIST_WATER_MAX", "WATER_TYPE_BONUS",
    "ASPECT_SCORES",
    "CANOPY_OPTIMAL_OPENNESS", "CANOPY_MIN_OPENNESS", "CANOPY_MAX_OPENNESS",
    "GROUND_COVER_PREFERENCES", "DISTURBANCE_SCORES",
    "PHENOLOGY_ENABLED", "PHENOLOGY_GRADIENT",
    "PHENOLOGY_BASE_MONTH", "PHENOLOGY_BASE_ALT", "ELIMINATORY_GEOLOGY",
    "PROBABILITY_THRESHOLDS", "PROBABILITY_LABELS",
    "BASEMAPS", "DEFAULT_BASEMAP",
    "validate_config",
]


# ═══════════════════════════════════════════════════════════════
#  EMPRISE — Grenoble rayon 10 km
# ═══════════════════════════════════════════════════════════════

_CENTER_X_L93: float = 913_100.0
_CENTER_Y_L93: float = 6_458_800.0
_RADIUS_M: float = 10_000.0

BBOX: MappingProxyType = MappingProxyType({
    "xmin": _CENTER_X_L93 - _RADIUS_M,
    "ymin": _CENTER_Y_L93 - _RADIUS_M,
    "xmax": _CENTER_X_L93 + _RADIUS_M,
    "ymax": _CENTER_Y_L93 + _RADIUS_M,
})

BBOX_WGS84: MappingProxyType = MappingProxyType({
    "west":  5.586530,
    "south": 45.098498,
    "east":  5.862470,
    "north": 45.278502,
})

MAP_CENTER: MappingProxyType = MappingProxyType({
    "lat": 45.1885,
    "lon": 5.7245,
})

CELL_SIZE: float = 5.0
DATA_BUFFER: float = 500.0


# ═══════════════════════════════════════════════════════════════
#  PONDÉRATIONS DU MODÈLE MULTICRITÈRE  (v2.2.0)
# ═══════════════════════════════════════════════════════════════
#
#  Rééquilibrage v2.2.0 — motivations :
#
#  geology        ↑ 0.22  sol calcaire = quasi-obligatoire pour morilles
#  canopy_openness↑ 0.15  discriminant forêt vs terrain ouvert (proxy)
#  tree_species   ↓ 0.15  88% unknown dilue le signal ; reste utile si data
#  altitude       ↑ 0.10  300-600m (Vouillants) vs plaine 210m
#  slope          ↓ 0.10  moins critique après fix bug SLOPE_STEEP
#  dist_water     = 0.10  inchangé
#  terrain_rough. ↓ 0.06
#  aspect         = 0.04
#  ground_cover   = 0.04
#  disturbance    = 0.04

_WEIGHTS_DICT: dict[str, float] = {
    "geology":            0.18,
    "tree_species":       0.14,
    "canopy_openness":    0.09,   # réduit : dégradé sans landcover (→ forest_edge)
    "twi":                0.11,
    "urban_proximity":    0.10,
    "altitude":           0.08,
    "forest_edge":        0.04,   # lisière forestière (BD Forêt + EDT)
    "slope":              0.06,
    "dist_water":         0.05,
    "terrain_roughness":  0.05,
    "ground_cover":       0.04,
    "disturbance":        0.03,
    "aspect":             0.03,
}

_w_total = sum(_WEIGHTS_DICT.values())
if abs(_w_total - 1.0) > 0.001:
    raise ValueError(
        f"config.py: Somme des poids = {_w_total:.4f}, attendu 1.0000."
    )

WEIGHTS: MappingProxyType = MappingProxyType(_WEIGHTS_DICT)

# ── Proximité urbaine ─────────────────────────────────────────────
# TODO: grid_builder.py — implémenter score_urban_proximity()
#       EDT sur urban_mask inversé → distance en mètres × CELL_SIZE
#       < URBAN_DIST_ELIMINATORY  → 0.0
#       ELIMINATORY..PENALTY      → rampe linéaire [FLOOR..0.6]
#       PENALTY..FULL             → rampe linéaire [0.6..1.0]
#       > URBAN_DIST_FULL         → 1.0
# TODO: scoring.py — ajouter urban_proximity < URBAN_DIST_ELIMINATORY
#       comme facteur éliminatoire (score → 0)

URBAN_DIST_ELIMINATORY: float = 15.0    # m — score dur → 0
URBAN_DIST_PENALTY: float = 100.0       # m — fin de pénalité forte
URBAN_DIST_FULL: float = 250.0          # m — plus aucune pénalité
URBAN_PROXIMITY_FLOOR: float = 0.05     # plancher zone tampon

# ═══════════════════════════════════════════════════════════════
#  ESSENCES FORESTIÈRES
# ═══════════════════════════════════════════════════════════════

_TREE_SCORES_DICT: dict[str, float] = {
    # ── Feuillus plaine alluviale (M. esculenta / M. vulgaris) ──
    "frene":              1.00,
    "orme":               0.95,
    "pommier":            0.90,
    "poirier":            0.85,
    "peuplier":           0.80,
    "chataignier":        0.80,
    "aulne":              0.55,
    "saule":              0.45,
    "noisetier":          0.40,
    "charme":             0.35,
    "tilleul":            0.35,
    "erable":             0.30,
    "hetre":              0.30,
    # ── Conifères montagnards (M. elata) ──
    "epicea":             0.35,
    "douglas":            0.30,
    "sapin":              0.25,
    "pin_sylvestre":      0.15,
    # ── Essences peu/pas favorables ──
    "robinier":           0.20,
    "platane":            0.20,
    "chene":              0.15,
    "chene_pubescent":    0.10,
    "bouleau":            0.05,
    "buis":               0.05,
    # ── Défaut — v2.2.0 : 0.15 → 0.25 (forêt sans essence ≠ pas de forêt)
    "unknown":            0.25,
}

# ── config.py — ajouter après FILL_NO_GEOLOGY ou dans le bloc constantes ──

# Landcover : plancher green_score pour cellules en forêt connue (BD Forêt v2)
# Empêche les couleurs sombres OSM Carto d'écraser le signal forestier.
# Valeur 0.80 = 80% du score végétation préservé sous couvert forestier.
LANDCOVER_FOREST_FLOOR: Final[float] = 0.80


_TREE_ALIASES_DICT: dict[str, str] = {
    "frêne": "frene", "fraxinus": "frene",
    "fraxinus_excelsior": "frene", "frene_commun": "frene",
    "frêne_commun": "frene",
    "ulmus": "orme", "orme_champetre": "orme",
    "malus": "pommier", "pyrus": "poirier",
    "populus": "peuplier", "peuplier_noir": "peuplier",
    "peuplier_blanc": "peuplier", "populus_nigra": "peuplier",
    "alnus": "aulne", "aulne_glutineux": "aulne",
    "alnus_glutinosa": "aulne", "vergne": "aulne",
    "salix": "saule", "saule_blanc": "saule",
    "salix_alba": "saule", "osier": "saule",
    "corylus": "noisetier", "corylus_avellana": "noisetier",
    "carpinus": "charme", "carpinus_betulus": "charme",
    "tilia": "tilleul", "tilleul_a_grandes_feuilles": "tilleul",
    "acer": "erable", "érable": "erable",
    "erable_sycomore": "erable", "erable_champetre": "erable",
    "hêtre": "hetre", "fagus": "hetre", "fagus_sylvatica": "hetre",
    "picea": "epicea", "épicéa": "epicea", "picea_abies": "epicea",
    "pseudotsuga": "douglas", "pseudotsuga_menziesii": "douglas",
    "abies": "sapin", "abies_alba": "sapin", "sapin_blanc": "sapin",
    "pinus_sylvestris": "pin_sylvestre", "pin": "pin_sylvestre",
    "robinia": "robinier", "robinia_pseudoacacia": "robinier",
    "faux_acacia": "robinier", "acacia": "robinier",
    "platanus": "platane",
    "quercus": "chene", "chêne": "chene",
    "quercus_pubescens": "chene_pubescent",
    "chêne_pubescent": "chene_pubescent",
    "quercus_petraea": "chene", "quercus_robur": "chene",
    "betula": "bouleau", "betula_pendula": "bouleau",
    "buxus": "buis",
    "castanea": "chataignier", "châtaignier": "chataignier",
    "castanea_sativa": "chataignier",
}

TREE_SCORES: MappingProxyType = MappingProxyType(_TREE_SCORES_DICT)
TREE_ALIASES: MappingProxyType = MappingProxyType(_TREE_ALIASES_DICT)

def resolve_tree_name(raw_name: str | None) -> str:
    """Normalise un nom d'essence vers la forme canonique."""
    if raw_name is None:
        return "unknown"
    key = (
        raw_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("û", "u")
    )
    if key in TREE_SCORES:
        return key
    if key in TREE_ALIASES:
        return TREE_ALIASES[key]
    for canonical in TREE_SCORES:
        if canonical != "unknown" and canonical in key:
            return canonical
    for alias, canonical in TREE_ALIASES.items():
        if alias in key:
            return canonical
    logger.debug("Essence non reconnue : '%s' → 'unknown'", raw_name)
    return "unknown"


def get_tree_score(raw_name: str | None) -> float:
    """Retourne le score d'une essence."""
    canonical = resolve_tree_name(raw_name)
    return float(TREE_SCORES.get(canonical, TREE_SCORES["unknown"]))


# ═══════════════════════════════════════════════════════════════
#  GÉOLOGIE
# ═══════════════════════════════════════════════════════════════

# ── Keywords géologie → catégorie (P10 : tuple immutable) ──────────
# Multi-mots EN TÊTE (prioritaires), puis substrats, puis dépôts
_GEOLOGY_KEYWORD_MAP: tuple[tuple[str, str], ...] = (
    # ── Multi-mots (prioritaires) ──
    ("terres noires",    "marne"),
    ("couches rousses",  "calcaire"),
    # ── Substrats primaires ──
    ("alluvion",         "alluvions"),
    ("calcaire",         "calcaire"),
    ("marne",            "marne"),
    ("moraine",          "moraine"),
    ("colluvion",        "colluvions"),
    ("eboulis",          "eboulis_calcaire"),
    ("éboulis",          "eboulis_calcaire"),
    ("granite",          "granite"),
    ("gneiss",           "gneiss"),
    ("schiste",          "schiste"),
    ("grès",             "gres"),
    ("gres",             "gres"),
    ("molasse",          "molasse"),
    ("flysch",           "flysch"),
    ("silice",           "siliceux"),
    ("sidérose",            "siliceux"),          # Filons Fe — substrat acide
    ("sidérite",            "siliceux"),          # Variante orthographe
    # ── BDCharm-50 Alpes — formations calcaires ──
    ("lauze",            "calcaire"),
    ("biodétritique",    "calcaire"),
    ("travertin",        "calcaire"),
    ("tuf",              "calcaire"),
    ("barrémien",        "calcaire"),
    ("sénonien",         "calcaire"),
    ("kimméridgien",     "calcaire"),
    ("campanien",        "calcaire"),
    ("urgonien",         "calcaire"),
    # ── Dépôts superficiels ──
    ("cône",             "alluvions"),
    ("déjection",        "alluvions"),
    ("lacustre",         "alluvions"),
    ("argile",           "alluvions"),
    ("marais",           "alluvions"),
    ("glaciaire",        "moraine"),
    ("solifluxion",      "moraine"),
    ("coulée",           "moraine"),
)

_GEOLOGY_SCORES_DICT: dict[str, float] = {
    # ── Substrats calcaires = optimaux morilles ──────────────────
    "calcaire":              0.95,    # was 0.90 — calcaire massif, pH idéal
    "calcaire_marneux":      0.90,    # was 0.90 — inchangé
    "marne":                 0.85,    # was 0.85 — inchangé (Terres Noires OK)
    "alluvions_calcaires":   0.85,    # was 0.95 — bon pH mais variable
    "calcaire_lacustre":     0.80,    # was 0.60 — calcaire = bon pH, rehaussé
    "eboulis_calcaire":      0.75,    # was 0.45 — éboulis de calcaire, bon pH
    "dolomie":               0.75,    # NOUVEAU si présent

    # ── Substrats alluviaux = variables ──────────────────────────
    "alluvions":             0.70,    # was 0.80 — hétérogène
    "alluvions_recentes":    0.65,    # was 1.00 — sables/graviers souvent lessivés
    "colluvions":            0.70,    # was 0.65 — dépend substrat parent, rehaussé

    # ── Substrats intermédiaires ─────────────────────────────────
    "moraine":               0.55,    # was 0.50 — variable selon substrat
    "flysch":                0.50,    # was 0.40 — mélange calcaire/argile
    "molasse":               0.35,    # was 0.25 — grès/conglomérat, rehaussé
    "gres":                  0.25,    # was 0.25 — inchangé, souvent acide

    # ── Substrats défavorables / éliminatoires ───────────────────
    "schiste":               0.15,    # was 0.15 — inchangé
    "granite":               0.00,    # was 0.05 — éliminatoire → 0.00
    "gneiss":                0.00,    # was 0.05 — éliminatoire → 0.00
    "siliceux":              0.00,    # was 0.05 — éliminatoire → 0.00

    # ── Par défaut ───────────────────────────────────────────────
    "unknown":               0.30,    # inchangé
}


_GEOLOGY_BRGM_MAP_DICT: dict[str, str] = {
    "Fz":   "alluvions_recentes",
    "Fy":   "alluvions",
    "Fx":   "alluvions",
    "Fz-y": "alluvions_recentes",
    "CFp":  "colluvions",
    "Gx":   "moraine",
    "Gy":   "moraine",
    "E":    "eboulis_calcaire",
    "j6":   "calcaire",
    "j5":   "calcaire",
    "j4":   "calcaire_marneux",
    "j3-6": "calcaire",
    "j3":   "calcaire_marneux",
    "j1-2": "marne",
    "c1":   "calcaire_marneux",
    "c2":   "calcaire",
    "n":    "marne",
    "m":    "molasse",
    "e":    "calcaire_lacustre",
    "γ":    "granite",
    "ξ":    "gneiss",
}

GEOLOGY_SCORES: MappingProxyType = MappingProxyType(_GEOLOGY_SCORES_DICT)
GEOLOGY_BRGM_MAP: MappingProxyType = MappingProxyType(_GEOLOGY_BRGM_MAP_DICT)

ELIMINATORY_GEOLOGY: frozenset[str] = frozenset({
    "granite", "gneiss", "siliceux",
})


def resolve_geology(raw_code: str | None) -> str:
    """
    Résout un code géologique BRGM vers la catégorie simplifiée.

    Cascade :
      1. GEOLOGY_BRGM_MAP (codes directs)
      2. GEOLOGY_SCORES (clés normalisées)
      3. _GEOLOGY_KEYWORD_MAP (keywords dans DESCR, BDCharm-50 compatible)
    """
    if raw_code is None:
        return "unknown"
    code = raw_code.strip()
    if code in GEOLOGY_BRGM_MAP:
        return GEOLOGY_BRGM_MAP[code]
    key = code.lower().replace(" ", "_").replace("-", "_")
    if key in GEOLOGY_SCORES:
        return key
    cl = code.lower()
    for keyword, category in _GEOLOGY_KEYWORD_MAP:
        if keyword in cl:
            return category
    logger.debug("Géologie non reconnue : '%s' → 'unknown'", raw_code)
    return "unknown"


def get_geology_score(raw_code: str | None) -> float:
    """Retourne le score d'un type géologique."""
    category = resolve_geology(raw_code)
    return float(GEOLOGY_SCORES.get(category, GEOLOGY_SCORES["unknown"]))


# ═══════════════════════════════════════════════════════════════
#  ALTITUDE  (v2.2.0 : optimal élargi 200-600m)
# ═══════════════════════════════════════════════════════════════
#
#  M. esculenta — optimal 200-600m (plaine + Vouillants)
#  M. elata     — peut monter à 800-1200m (conifères)

ALTITUDE_OPTIMAL: tuple[float, float] = (200.0, 600.0)
ALTITUDE_RANGE: tuple[float, float] = (150.0, 900.0)

# Centre du bonus micro-nappe / humidité résiduelle
ALTITUDE_ALLUVIAL_CENTER: float = 350.0


# ═══════════════════════════════════════════════════════════════
#  PENTE  (v2.2.0 — BUG FIX CRITIQUE)
# ═══════════════════════════════════════════════════════════════
#
#  BUG v2.1.1 : SLOPE_STEEP = 4.0 → le mask « >steep & ≤max »
#  (4-60°) ÉCRASAIT le mask « >optimal & ≤moderate » (8-25°).
#  Résultat : tout slope > 4° recevait score ≈ 0.08.
#
#  Fix v2.2.0 : seuils non chevauchants, décroissance propre.
#
#   0-10°  : optimal (score 1.0)
#  10-20°  : modéré  (1.0 → 0.6)
#  20-35°  : raide   (0.6 → 0.1)
#  35-45°  : très raide (0.1 → 0.0)
#    >45°  : éliminatoire

SLOPE_OPTIMAL: Final[tuple[float, float]] = (0.0, 15.0)   # was (0, 10)
SLOPE_MODERATE: Final[float] = 30.0                         # was 20
SLOPE_STEEP: Final[float] = 40.0                             # was 35
SLOPE_MAX: Final[float] = 50.0                               # was 45


# ═══════════════════════════════════════════════════════════════
#  RUGOSITÉ TERRAIN
# ═══════════════════════════════════════════════════════════════

ROUGHNESS_WINDOW: int = 7
ROUGHNESS_OPTIMAL: float = 3.0
ROUGHNESS_MAX: float = 12.0


# ═══════════════════════════════════════════════════════════════
#  DISTANCE AUX COURS D'EAU
# ═══════════════════════════════════════════════════════════════
# ── Distance eau — v2.3.4 fix #39 ───────────────────────────────
# Calibré Alpes : l'humidité du sol dépend autant du couvert forestier
# et de l'exposition que de la distance brute au cours d'eau.
# Ancien : score ~0 au-delà de 50m. Nouveau : décroissance plus lente,
# score plancher forêt intégré dans grid_builder.
DIST_WATER_OPTIMAL: Final[tuple[float, float]] = (5.0, 80.0)    # was (5, 15)
DIST_WATER_GOOD: Final[float] = 100.0                            # was 50
DIST_WATER_MODERATE: Final[float] = 500.0                         # was 300
DIST_WATER_MAX: Final[float] = 1000.0                             # was 300
# Plancher pour cellules forestières (sol retient l'humidité)
DIST_WATER_FOREST_FLOOR: Final[float] = 0.20
_WATER_TYPE_BONUS_DICT: dict[str, float] = {
    "bras_mort":        1.30,
    "plan_eau":         1.20,
    "canal":            1.10,
    "riviere":          1.00,
    "ruisseau":         0.90,
    "torrent":          0.70,
    "unknown":          0.90,
}

WATER_TYPE_BONUS: MappingProxyType = MappingProxyType(_WATER_TYPE_BONUS_DICT)

# ═══════════════════════════════════════════════════════════════
# TWI — Topographic Wetness Index (fix #46 v2.3.5)
# ═══════════════════════════════════════════════════════════════
# TWI = ln(a / tan(β))  où a = aire drainée spécifique, β = pente locale
# Valeurs typiques : 2 (crête) → 5 (versant) → 12 (fond de vallée) → 20+ (zone humide)
#
# Morilles : sol bien drainé mais humide → TWI optimal entre 6 et 10
# Trop bas (crête sèche) = pas assez d'eau → pénalisé
# Trop haut (zone engorgée) = asphyxie racinaire → fortement pénalisé

TWI_OPTIMAL: tuple[float, float] = (6.0, 10.0)     # zone idéale
TWI_DRY_LIMIT: float = 3.0                           # en dessous = crête très sèche → 0.10
TWI_WET_LIMIT: float = 14.0                          # au-dessus = zone engorgée → 0.10
TWI_WATERLOG: float = 18.0                            # au-dessus = marécage → 0.0
TWI_DRY_FLOOR: float = 0.10                           # score min crête
TWI_WET_FLOOR: float = 0.10                           # score min zone humide


# ═══════════════════════════════════════════════════════════════
#  EXPOSITION (ASPECT)
# ═══════════════════════════════════════════════════════════════

_ASPECT_SCORES_DICT: dict[str, float] = {
    "S":    1.00,
    "SE":   0.90,
    "flat": 0.85,
    "SW":   0.80,
    "E":    0.70,
    "W":    0.60,
    "NE":   0.50,
    "NW":   0.40,
    "N":    0.30,
}

ASPECT_SCORES: MappingProxyType = MappingProxyType(_ASPECT_SCORES_DICT)


# ═══════════════════════════════════════════════════════════════
#  MICRO-HABITAT
# ═══════════════════════════════════════════════════════════════

CANOPY_OPTIMAL_OPENNESS: float = 0.4
CANOPY_MIN_OPENNESS: float = 0.1
CANOPY_MAX_OPENNESS: float = 0.9

_GROUND_COVER_DICT: dict[str, float] = {
    "litiere_seche":     1.00,
    "litiere_humide":    0.60,
    "herbe_rase":        0.50,
    "mousse_legere":     0.40,
    "mousse_epaisse":    0.10,
    "lierre":            0.05,
    "sol_nu":            0.70,
}

GROUND_COVER_PREFERENCES: MappingProxyType = MappingProxyType(
    _GROUND_COVER_DICT
)

_DISTURBANCE_DICT: dict[str, float] = {
    "coupe_recente_1_3ans": 0.90,
    "chemin_forestier":     0.70,
    "chablis":              0.60,
    "brulis":               0.95,
    "paillage_brf":         0.80,
    "labour_recent":        0.30,
    "aucune":               0.50,
}

DISTURBANCE_SCORES: MappingProxyType = MappingProxyType(_DISTURBANCE_DICT)


# ═══════════════════════════════════════════════════════════════
#  PHÉNOLOGIE / SAISONNALITÉ (optionnel)
# ═══════════════════════════════════════════════════════════════

PHENOLOGY_ENABLED: bool = False
PHENOLOGY_GRADIENT: float = 300.0
PHENOLOGY_BASE_MONTH: int = 3
PHENOLOGY_BASE_ALT: float = 200.0


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION DE PROBABILITÉ
# ═══════════════════════════════════════════════════════════════

PROBABILITY_THRESHOLDS: tuple[float, ...] = (0.15, 0.30, 0.45, 0.60, 0.75)

PROBABILITY_LABELS: tuple[str, ...] = (
    "Nul",
    "Très faible",
    "Faible",
    "Moyen",
    "Bon",
    "Excellent",
)


# ═══════════════════════════════════════════════════════════════
#  FONDS DE CARTE FOLIUM
# ═══════════════════════════════════════════════════════════════

_GEOPF_BASE = (
    "https://data.geopf.fr/wmts?"
    "SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
    "&STYLE=normal&TILEMATRIXSET=PM"
    "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
)

_BASEMAPS_DICT: dict[str, dict[str, Any]] = {
    "IGN Plan v2": {
        "tiles": f"{_GEOPF_BASE}&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2"
                 "&FORMAT=image/png",
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Plan v2",
        "max_zoom": 18,
    },
    "IGN Scan25": {
        "tiles": f"{_GEOPF_BASE}&LAYER=GEOGRAPHICALGRIDSYSTEMS.MAPS"
                 "&FORMAT=image/jpeg",
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Scan25 (topo)",
        "max_zoom": 16,
    },
    "IGN Photographies aériennes": {
        "tiles": f"{_GEOPF_BASE}&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
                 "&FORMAT=image/jpeg",
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Photo aérienne",
        "max_zoom": 19,
    },
    "IGN Occupation du sol": {
        "tiles": f"{_GEOPF_BASE}&LAYER=LANDUSE.AGRICULTURE2023"
                 "&FORMAT=image/png",
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Occupation du sol",
        "max_zoom": 18,
    },
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": "© OpenStreetMap contributors",
        "name": "OpenStreetMap",
        "max_zoom": 19,
    },
    "Esri Satellite": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": "© Esri, Maxar, Earthstar Geographics",
        "name": "Esri Satellite",
        "max_zoom": 18,
    },
    "OpenTopoMap": {
        "tiles": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "© OpenTopoMap contributors",
        "name": "OpenTopoMap",
        "max_zoom": 17,
    },
}

BASEMAPS: MappingProxyType = MappingProxyType(_BASEMAPS_DICT)
DEFAULT_BASEMAP: str = "IGN Plan v2"


# ═══════════════════════════════════════════════════════════════
#  VALIDATION GLOBALE
# ═══════════════════════════════════════════════════════════════

def validate_config() -> bool:
    """Vérifie la cohérence interne de la configuration."""
    ok = True

    w_sum = sum(WEIGHTS.values())
    if abs(w_sum - 1.0) > 0.01:
        logger.error("Somme des poids = %.4f ≠ 1.0", w_sum)
        ok = False

    for name, scores in [
        ("TREE_SCORES", TREE_SCORES),
        ("GEOLOGY_SCORES", GEOLOGY_SCORES),
        ("ASPECT_SCORES", ASPECT_SCORES),
    ]:
        for key, val in scores.items():
            if not (0.0 <= val <= 1.0):
                logger.error("%s['%s'] = %.2f hors [0, 1]", name, key, val)
                ok = False

    # Seuils de pente cohérents (v2.2.0 — non chevauchants)
    if SLOPE_OPTIMAL[1] >= SLOPE_MODERATE:
        logger.warning(
            "SLOPE_OPTIMAL[1]=%.0f ≥ SLOPE_MODERATE=%.0f",
            SLOPE_OPTIMAL[1], SLOPE_MODERATE,
        )
    if SLOPE_MODERATE >= SLOPE_STEEP:
        logger.warning(
            "SLOPE_MODERATE=%.0f ≥ SLOPE_STEEP=%.0f",
            SLOPE_MODERATE, SLOPE_STEEP,
        )
    if SLOPE_STEEP >= SLOPE_MAX:
        logger.warning(
            "SLOPE_STEEP=%.0f ≥ SLOPE_MAX=%.0f",
            SLOPE_STEEP, SLOPE_MAX,
        )

    if ALTITUDE_OPTIMAL[0] < ALTITUDE_RANGE[0]:
        logger.warning(
            "ALTITUDE_OPTIMAL[0]=%.0f < ALTITUDE_RANGE[0]=%.0f",
            ALTITUDE_OPTIMAL[0], ALTITUDE_RANGE[0],
        )

    if BBOX["xmin"] >= BBOX["xmax"] or BBOX["ymin"] >= BBOX["ymax"]:
        logger.error("BBOX dégénéré : %s", dict(BBOX))
        ok = False

    if CELL_SIZE < 1.0 or CELL_SIZE > 100.0:
        logger.warning("CELL_SIZE=%.1f inhabituel (attendu 1-100m)", CELL_SIZE)

    for alias, canonical in TREE_ALIASES.items():
        if canonical not in TREE_SCORES:
            logger.error(
                "TREE_ALIASES['%s'] → '%s' absent de TREE_SCORES",
                alias, canonical,
            )
            ok = False

    if len(PROBABILITY_LABELS) != len(PROBABILITY_THRESHOLDS) + 1:
        logger.error(
            "PROBABILITY_LABELS (%d) != PROBABILITY_THRESHOLDS (%d) + 1",
            len(PROBABILITY_LABELS), len(PROBABILITY_THRESHOLDS),
        )
        ok = False

    for i in range(1, len(PROBABILITY_THRESHOLDS)):
        if PROBABILITY_THRESHOLDS[i] <= PROBABILITY_THRESHOLDS[i - 1]:
            logger.error(
                "PROBABILITY_THRESHOLDS non croissant : [%d]=%.2f ≤ [%d]=%.2f",
                i, PROBABILITY_THRESHOLDS[i],
                i - 1, PROBABILITY_THRESHOLDS[i - 1],
            )
            ok = False

    for geo in ELIMINATORY_GEOLOGY:
        if geo not in GEOLOGY_SCORES:
            logger.error("ELIMINATORY_GEOLOGY '%s' absent de GEOLOGY_SCORES", geo)
            ok = False

    if DEFAULT_BASEMAP not in BASEMAPS:
        logger.error("DEFAULT_BASEMAP '%s' absent de BASEMAPS", DEFAULT_BASEMAP)
        ok = False

    if ok:
        logger.info("✅ Configuration validée (v%s)", CONFIG_VERSION)
    else:
        logger.error("❌ Configuration invalide — voir erreurs ci-dessus")

    return ok