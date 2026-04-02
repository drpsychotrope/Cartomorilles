"""
🍄 CARTOMORILLES — Configuration du modèle (v2.4.0)
   Zone : Isère (38) — 74.5×98.8 km, emprise limitée par données

   v2.4.0 :
     - Extension emprise : Grenoble 20×20km → Isère 74.5×98.8km
     - DEM Copernicus N45_E005 ∩ BD Forêt v2 ∩ Géologie BDCharm-50
     - CELL_SIZE 5m → 10m (73.6M cellules, ~4.4 GB RAM)
     - Chemins données : DEM_FILE, FOREST_SHAPEFILE, GEOLOGY_SHAPEFILE
     - DIST_WATER_FOREST_FLOOR 0.20 → 0.15 (cohérence D9)

   v2.3.6 :
     - Scores résineux montagnards rehaussés (sapin 0.25→0.55,
       épicéa 0.35→0.45, pin_sylvestre 0.15→0.45)
     - Poids rééquilibrés : geology 0.18→0.20, tree_species 0.14→0.16,
       canopy_openness 0.13→0.11, twi 0.11→0.07

   v2.3.5 :
     - BUG FIX : SLOPE_STEEP 4.0 → 35.0
     - ALTITUDE_OPTIMAL élargi (200, 600)
     - unknown tree score 0.15 → 0.25
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Final

# ═══════════════════════════════════════════════════════════════
#  MÉTADONNÉES
# ═══════════════════════════════════════════════════════════════

CONFIG_VERSION: Final[str] = "2.4.0"

logger = logging.getLogger("cartomorilles.config")

__all__ = [
    # Métadonnées
    "CONFIG_VERSION",
    # Emprise
    "BBOX_WGS84", "BBOX", "CELL_SIZE", "MAP_CENTER", "DATA_BUFFER",
    # Chemins données
    "DEM_FILE", "FOREST_SHAPEFILE",
    "DATA_SUBDIR_GEOLOGY", "GEOLOGY_SHAPEFILE",
    # Poids
    "WEIGHTS",
    # Essences
    "TREE_SCORES", "TREE_ALIASES",
    "resolve_tree_name", "get_tree_score",
    # Géologie
    "GEOLOGY_SCORES", "GEOLOGY_BRGM_MAP", "ELIMINATORY_GEOLOGY",
    "resolve_geology", "get_geology_score",
    # Altitude
    "ALTITUDE_OPTIMAL", "ALTITUDE_RANGE", "ALTITUDE_ALLUVIAL_CENTER",
    # Pente
    "SLOPE_OPTIMAL", "SLOPE_MODERATE", "SLOPE_STEEP", "SLOPE_MAX",
    # Rugosité
    "ROUGHNESS_WINDOW", "ROUGHNESS_OPTIMAL", "ROUGHNESS_MAX",
    # Hydrographie
    "DIST_WATER_OPTIMAL", "DIST_WATER_GOOD", "DIST_WATER_MODERATE",
    "DIST_WATER_MAX", "DIST_WATER_FOREST_FLOOR", "WATER_TYPE_BONUS",
    # TWI
    "TWI_OPTIMAL", "TWI_DRY_LIMIT", "TWI_WET_LIMIT", "TWI_WATERLOG",
    "TWI_DRY_FLOOR", "TWI_WET_FLOOR",
    # Exposition
    "ASPECT_SCORES",
    # Micro-habitat
    "CANOPY_OPTIMAL_OPENNESS", "CANOPY_MIN_OPENNESS", "CANOPY_MAX_OPENNESS",
    "GROUND_COVER_PREFERENCES", "DISTURBANCE_SCORES",
    # Landcover
    "LANDCOVER_FOREST_FLOOR",
    "LANDCOVER_AUTO_MAX_KM2",
    # Urbain
    "URBAN_DIST_ELIMINATORY", "URBAN_DIST_PENALTY",
    "URBAN_DIST_FULL", "URBAN_PROXIMITY_FLOOR",
    # Phénologie
    "PHENOLOGY_ENABLED", "PHENOLOGY_GRADIENT",
    "PHENOLOGY_BASE_MONTH", "PHENOLOGY_BASE_ALT",
    # Classification
    "PROBABILITY_THRESHOLDS", "PROBABILITY_LABELS",
    # Fonds de carte
    "BASEMAPS", "DEFAULT_BASEMAP",
    # Validation
    "validate_config",
]


# ══════════════════════════════════════════════════════════════════════
# Emprise — Isère 38 (limitée par données disponibles)
# ══════════════════════════════════════════════════════════════════════
# Intersection : DEM Copernicus N45_E005 ∩ BD Forêt ∩ Géologie BRGM 38
# X limité par DEM (lon 5°–6°) — Oisans (lon > 6°) non couvert
# Y limité par BD Forêt (6 534 209 m nord)

_BBOX_XMIN: Final[float] = 857_571.0
_BBOX_YMIN: Final[float] = 6_435_430.0
_BBOX_XMAX: Final[float] = 932_112.0
_BBOX_YMAX: Final[float] = 6_534_209.0

_CENTER_X_L93: Final[float] = (_BBOX_XMIN + _BBOX_XMAX) / 2  # ≈ 894 842
_CENTER_Y_L93: Final[float] = (_BBOX_YMIN + _BBOX_YMAX) / 2  # ≈ 6 484 820

BBOX: MappingProxyType[str, float] = MappingProxyType({
    "xmin": _BBOX_XMIN,
    "ymin": _BBOX_YMIN,
    "xmax": _BBOX_XMAX,
    "ymax": _BBOX_YMAX,
})

BBOX_WGS84: MappingProxyType[str, float] = MappingProxyType({
    "west": 5.0000,
    "south": 45.0001,
    "east": 5.9999,
    "north": 45.8564,
})

MAP_CENTER: MappingProxyType[str, float] = MappingProxyType({
    "lat": 45.428,
    "lon": 5.500,
})


# ══════════════════════════════════════════════════════════════════════
# Chemins de données (relatifs à data/)
# ══════════════════════════════════════════════════════════════════════

DEM_FILE: Final[str] = "Copernicus_DSM_COG_10_N45_00_E005_00_DEM.tif"
FOREST_SHAPEFILE: Final[str] = "FORMATION_VEGETALE.shp"
DATA_SUBDIR_GEOLOGY: Final[str] = "geologie_38"
GEOLOGY_SHAPEFILE: Final[str] = "GEO050K_HARM_038_S_FGEOL_2154.shp"


# ══════════════════════════════════════════════════════════════════════
# Maillage & buffers
# ══════════════════════════════════════════════════════════════════════

CELL_SIZE: float = 10.0
DATA_BUFFER: Final[float] = 500.0

# ── Auto-scaling résolution ──────────────────────────────────────
# (seuil_km², cell_m) — premier seuil ≥ emprise gagne.
CELL_SIZE_AUTO_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (500.0, 5.0),
    (2000.0, 10.0),
    (8000.0, 50.0),
)
CELL_SIZE_AUTO_FALLBACK: Final[float] = 100.0

# ═══════════════════════════════════════════════════════════════
#  PONDÉRATIONS DU MODÈLE MULTICRITÈRE  (v2.3.6)
# ═══════════════════════════════════════════════════════════════
#
#  Rééquilibrage v2.3.6 — corrige biais plaine vs montagne :
#
#  geology         0.20  meilleur discriminant réel (calcaire vs alluvions)
#  tree_species    0.16  amplifie correction résineux montagnards
#  canopy_openness 0.11  biais plat (open edges surnotées en plaine)
#  urban_proximity 0.10
#  altitude        0.08
#  twi             0.07  ~1.0 en plaine systématiquement, redondant dist_water
#  slope           0.06
#  aspect          0.05  exposition Sud discriminante en montagne
#  dist_water      0.05
#  terrain_rough.  0.05
#  ground_cover    0.04
#  disturbance     0.03

_WEIGHTS_DICT: dict[str, float] = {
    "geology":            0.20,
    "tree_species":       0.16,
    "canopy_openness":    0.07,   # réduit : dégradé sans landcover (→ forest_edge)
    "urban_proximity":    0.10,
    "altitude":           0.08,
    "twi":                0.07,
    "forest_edge":        0.04,   # lisière forestière (BD Forêt + EDT)
    "slope":              0.06,
    "aspect":             0.05,
    "dist_water":         0.05,
    "terrain_roughness":  0.04,   # -0.01 → geology_contact
    "ground_cover":       0.04,
    "disturbance":        0.02,   # -0.01 → geology_contact
    "geology_contact":    0.02,   # contacts géologiques (BDCharm-50 L_FGEOL)
}

_w_total = sum(_WEIGHTS_DICT.values())
if abs(_w_total - 1.0) > 0.001:
    raise ValueError(
        f"config.py: Somme des poids = {_w_total:.4f}, attendu 1.0000.",
    )

WEIGHTS: MappingProxyType[str, float] = MappingProxyType(_WEIGHTS_DICT)


# ═══════════════════════════════════════════════════════════════
#  PROXIMITÉ URBAINE
# ═══════════════════════════════════════════════════════════════
# TODO: grid_builder.py — implémenter score_urban_proximity()
#       EDT sur urban_mask inversé → distance en mètres × CELL_SIZE
#       < URBAN_DIST_ELIMINATORY  → 0.0
#       ELIMINATORY..PENALTY      → rampe linéaire [FLOOR..0.6]
#       PENALTY..FULL             → rampe linéaire [0.6..1.0]
#       > URBAN_DIST_FULL         → 1.0
# TODO: scoring.py — ajouter urban_proximity < URBAN_DIST_ELIMINATORY
#       comme facteur éliminatoire (score → 0)

URBAN_DIST_ELIMINATORY: Final[float] = 15.0    # m — score dur → 0
URBAN_DIST_PENALTY: Final[float] = 100.0       # m — fin de pénalité forte
URBAN_DIST_FULL: Final[float] = 250.0          # m — plus aucune pénalité
URBAN_PROXIMITY_FLOOR: Final[float] = 0.05     # plancher zone tampon
# Filtre densité urbaine (résolution-adaptatif)
# Calibré @10m : seuil=30 dans fenêtre 7×7=49 → fraction ≈ 0.61
URBAN_DENSITY_RADIUS_M: Final[float] = 30.0
URBAN_DENSITY_FRACTION: Final[float] = 30.0 / 49.0
URBAN_DENSITY_MIN_WINDOW: Final[int] = 5  # px — plancher pour grandes cellules


# ═══════════════════════════════════════════════════════════════
#  ESSENCES FORESTIÈRES  (v2.3.6 — rehaussement résineux montagnards)
# ═══════════════════════════════════════════════════════════════

_TREE_SCORES_DICT: dict[str, float] = {
    # ── Feuillus plaine alluviale (M. esculenta / M. vulgaris) ──
    "frene":              1.00,
    "orme":               0.95,
    "pommier":            0.90,
    "poirier":            0.85,
    "chataignier":        0.80,     # D3 : 0.80, PAS éliminatoire
    "peuplier":           0.70,
    "aulne":              0.55,
    "saule":              0.45,
    "noisetier":          0.40,
    "charme":             0.35,
    "tilleul":            0.35,
    "erable":             0.30,
    "hetre":              0.30,
    # ── Conifères montagnards (M. elata / M. conica) ─────────────
    # v2.3.6 : rehaussés — M. elata documentée en sapinière calcaire,
    # épicéa en forêts mixtes montagne, pin sylvestre sols calcaires secs
    "sapin":              0.55,
    "epicea":             0.45,
    "pin_sylvestre":      0.45,
    "douglas":            0.30,
    # ── Essences peu/pas favorables ──
    "platane":            0.20,
    "robinier":           0.15,
    "chene":              0.15,
    "chene_pubescent":    0.10,
    "bouleau":            0.05,
    "buis":               0.05,
    # ── Défaut — forêt sans essence connue ≠ pas de forêt ──
    "unknown":            0.25,
}

# ── Landcover : plancher green_score pour cellules BD Forêt ──
# D8 : Forest floor 0.80 pour cellules BD Forêt
LANDCOVER_FOREST_FLOOR: Final[float] = 0.80
# Surface maximale (km²) pour le landcover automatique.
# Au-delà, landcover est désactivé sauf --landcover explicite.
LANDCOVER_AUTO_MAX_KM2: Final[float] = 2000.0


_TREE_ALIASES_DICT: dict[str, str] = {
    # ── Frêne ──
    "frêne": "frene", "fraxinus": "frene",
    "fraxinus_excelsior": "frene", "frene_commun": "frene",
    "frêne_commun": "frene",
    # ── Orme ──
    "ulmus": "orme", "orme_champetre": "orme",
    # ── Fruitiers ──
    "malus": "pommier", "pyrus": "poirier",
    # ── Peuplier ──
    "populus": "peuplier", "peuplier_noir": "peuplier",
    "peuplier_blanc": "peuplier", "populus_nigra": "peuplier",
    # ── Aulne ──
    "alnus": "aulne", "aulne_glutineux": "aulne",
    "alnus_glutinosa": "aulne", "vergne": "aulne",
    # ── Saule ──
    "salix": "saule", "saule_blanc": "saule",
    "salix_alba": "saule", "osier": "saule",
    # ── Noisetier ──
    "corylus": "noisetier", "corylus_avellana": "noisetier",
    # ── Charme ──
    "carpinus": "charme", "carpinus_betulus": "charme",
    # ── Tilleul ──
    "tilia": "tilleul", "tilleul_a_grandes_feuilles": "tilleul",
    # ── Érable ──
    "acer": "erable", "érable": "erable",
    "erable_sycomore": "erable", "erable_champetre": "erable",
    # ── Hêtre ──
    "hêtre": "hetre", "fagus": "hetre", "fagus_sylvatica": "hetre",
    # ── Épicéa ──
    "picea": "epicea", "épicéa": "epicea", "picea_abies": "epicea",
    # ── Douglas ──
    "pseudotsuga": "douglas", "pseudotsuga_menziesii": "douglas",
    # ── Sapin ──
    "abies": "sapin", "abies_alba": "sapin", "sapin_blanc": "sapin",
    "sapin_pectine": "sapin",
    # ── Pin sylvestre ──
    "pinus_sylvestris": "pin_sylvestre", "pin": "pin_sylvestre",
    "pin_noir": "pin_sylvestre", "pinus": "pin_sylvestre",
    "pinus_nigra": "pin_sylvestre",
    # ── Robinier ──
    "robinia": "robinier", "robinia_pseudoacacia": "robinier",
    "faux_acacia": "robinier", "acacia": "robinier",
    # ── Platane ──
    "platanus": "platane",
    # ── Chêne ──
    "quercus": "chene", "chêne": "chene",
    "quercus_pubescens": "chene_pubescent",
    "chêne_pubescent": "chene_pubescent",
    "quercus_petraea": "chene", "quercus_robur": "chene",
    # ── Bouleau ──
    "betula": "bouleau", "betula_pendula": "bouleau",
    # ── Buis ──
    "buxus": "buis",
    # ── Châtaignier ──
    "castanea": "chataignier", "châtaignier": "chataignier",
    "castanea_sativa": "chataignier",
}

TREE_SCORES: MappingProxyType[str, float] = MappingProxyType(
    _TREE_SCORES_DICT,
)
TREE_ALIASES: MappingProxyType[str, str] = MappingProxyType(
    _TREE_ALIASES_DICT,
)


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
    ("terres noires",       "marne"),
    ("couches rousses",     "calcaire"),
    ("couches à myes",      "calcaire"),
    ("remblais artificiels","remblai"),
    ("dépôts et remblais",  "remblai"),
    ("réseau hydrographique","hydrographie"),
    ("formations résiduelles","colluvions"),
    ("formation de bonnevaux","alluvions"),
    ("sables fins bariolés","sable"),
    ("cailloutis polygéniques","alluvions"),
    # ── Substrats primaires ──
    ("alluvion",            "alluvions"),
    ("calcaire",            "calcaire"),
    ("marne",               "marne"),
    ("moraine",             "moraine"),
    ("colluvion",           "colluvions"),
    ("eboulis",             "eboulis_calcaire"),
    ("éboulis",             "eboulis_calcaire"),
    ("granite",             "granite"),
    ("gneiss",              "gneiss"),
    ("schiste",             "schiste"),
    ("grès",                "gres"),
    ("gres",                "gres"),
    ("molasse",             "molasse"),
    ("flysch",              "flysch"),
    ("silice",              "siliceux"),
    ("sidérose",            "siliceux"),
    ("sidérite",            "siliceux"),
    # ── BDCharm-50 Alpes — formations calcaires ──
    ("lauze",               "calcaire"),
    ("biodétritique",       "calcaire"),
    ("travertin",           "calcaire"),
    ("tuf",                 "calcaire"),
    ("barrémien",           "calcaire"),
    ("sénonien",            "calcaire"),
    ("kimméridgien",        "calcaire"),
    ("campanien",           "calcaire"),
    ("urgonien",            "calcaire"),
    ("hauterivien",         "calcaire"),
    ("valanginien",         "calcaire"),
    ("tithonien",           "calcaire"),
    ("berriasien",          "calcaire"),
    ("oxfordien",           "calcaire"),
    ("callovien",           "calcaire"),
    ("bajocien",            "calcaire"),
    ("albien",              "calcaire_marneux"),
    ("cénomanien",          "calcaire_marneux"),
    ("rhétien",             "calcaire"),
    ("lumachelle",          "calcaire"),
    # ── Dépôts fins / organiques ──
    ("loess",               "alluvions"),
    ("limon",               "alluvions"),
    ("tourbe",              "alluvions"),
    ("palust",              "alluvions"),
    # ── Cristallin Belledonne ──
    ("métagabbro",          "gneiss"),
    ("métapyroxénite",      "gneiss"),
    ("leptynite",           "gneiss"),
    ("métagranophyre",      "gneiss"),
    ("cristallophyllien",   "gneiss"),
    ("conglomérat métam",   "gneiss"),
    ("porphyroïde",         "gneiss"),
    ("carbonat",            "calcaire"),
    ("cristallin",          "gneiss"),
    # ── Glaciers / terrain instable ──
    ("glacier",             "moraine"),
    ("névé",                "moraine"),
    ("gliss",               "colluvions"),
    # ── Tertiaire ──
    ("tortonien",           "molasse"),
    ("miocène",             "molasse"),
    ("sannoisien",          "sable"),
    ("pliocène",            "alluvions"),
    ("anthropi",            "remblai"),
    # ── Dépôts superficiels ──
    ("cône",                "alluvions"),
    ("déjection",           "alluvions"),
    ("lacustre",            "alluvions"),
    ("argile",              "alluvions"),
    ("marais",              "alluvions"),
    ("sable",               "sable"),
    ("glaciaire",           "moraine"),
    ("solifluxion",         "moraine"),
    ("coulée",              "moraine"),
)

_GEOLOGY_SCORES_DICT: dict[str, float] = {
    # ── Substrats calcaires = optimaux morilles ──────────────────
    "calcaire":              0.95,
    "calcaire_marneux":      0.90,
    "marne":                 0.85,
    "alluvions_calcaires":   0.85,
    "calcaire_lacustre":     0.80,
    "eboulis_calcaire":      0.75,
    "dolomie":               0.75,
    # ── Substrats alluviaux = variables ──────────────────────────
    "colluvions":            0.65,
    "alluvions":             0.55,
    "alluvions_recentes":    0.50,
    # ── Substrats intermédiaires ─────────────────────────────────
    "moraine":               0.55,
    "flysch":                0.50,
    "sable":                 0.40,
    "molasse":               0.35,
    "gres":                  0.25,
    # ── Substrats défavorables / éliminatoires ───────────────────
    "schiste":               0.15,
    "remblai":               0.10,
    "hydrographie":          0.50,
    "granite":               0.00,
    "gneiss":                0.00,
    "siliceux":              0.00,
    # ── Par défaut ───────────────────────────────────────────────
    "unknown":               0.30,
}

_GEOLOGY_BRGM_MAP_DICT: dict[str, str] = {
    # ── Alluvions & dépôts récents ──
    "Fz":    "alluvions_recentes",
    "Fy":    "alluvions",
    "Fx":    "alluvions",
    "Fz-y":  "alluvions_recentes",
    "CFp":   "colluvions",
    "CF":    "colluvions",
    "C":     "colluvions",
    # ── Glaciaire ──
    "Gx":    "moraine",
    "Gy":    "moraine",
    "Gz":    "moraine",
    "G":     "moraine",
    "FG":    "moraine",
    # ── Éboulis ──
    "E":     "eboulis_calcaire",
    "Eb":    "eboulis_calcaire",
    # ── Jurassique ──
    "j6":    "calcaire",
    "j5":    "calcaire",
    "j4":    "calcaire_marneux",
    "j3-6":  "calcaire",
    "j3":    "calcaire_marneux",
    "j2":    "calcaire",
    "j1-2":  "marne",
    "j1":    "marne",
    # ── Crétacé ──
    "c1":    "calcaire_marneux",
    "c2":    "calcaire",
    "c3":    "calcaire",
    "n1":    "calcaire",
    "n2":    "calcaire_marneux",
    "n3":    "marne",
    "n4":    "calcaire",
    "n5":    "calcaire",
    # ── Tertiaire ──
    "n":     "marne",
    "m":     "molasse",
    "m1":    "molasse",
    "m2":    "molasse",
    "e":     "calcaire_lacustre",
    "e1":    "calcaire_lacustre",
    "e2":    "calcaire_lacustre",
    # ── Cristallin ──
    "γ":     "granite",
    "γ2":    "granite",
    "γ3":    "granite",
    "ξ":     "gneiss",
    "ξ2":    "gneiss",
    # ── Remblais ──
    "X":     "remblai",
}

GEOLOGY_SCORES: MappingProxyType[str, float] = MappingProxyType(
    _GEOLOGY_SCORES_DICT,
)
GEOLOGY_BRGM_MAP: MappingProxyType[str, str] = MappingProxyType(
    _GEOLOGY_BRGM_MAP_DICT,
)

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
#  M. esculenta — optimal 200-600m (plaine + collines)
#  M. elata     — peut monter à 800-1200m (conifères)

ALTITUDE_OPTIMAL: Final[tuple[float, float]] = (200.0, 600.0)
ALTITUDE_RANGE: Final[tuple[float, float]] = (150.0, 900.0)

# Centre du bonus micro-nappe / humidité résiduelle
ALTITUDE_ALLUVIAL_CENTER: Final[float] = 350.0


# ═══════════════════════════════════════════════════════════════
#  PENTE  (v2.2.0 — BUG FIX CRITIQUE)
# ═══════════════════════════════════════════════════════════════
#
#  BUG v2.1.1 : SLOPE_STEEP = 4.0 → écrasait tout slope > 4°
#  Fix v2.2.0 : seuils non chevauchants, décroissance propre.
#
#   0-15°  : optimal (score 1.0)
#  15-30°  : modéré  (1.0 → 0.6)
#  30-40°  : raide   (0.6 → 0.1)
#  40-50°  : très raide (0.1 → 0.0)
#    >50°  : éliminatoire

SLOPE_OPTIMAL: Final[tuple[float, float]] = (0.0, 15.0)
SLOPE_MODERATE: Final[float] = 30.0
SLOPE_STEEP: Final[float] = 40.0
SLOPE_MAX: Final[float] = 50.0


# ═══════════════════════════════════════════════════════════════
#  RUGOSITÉ TERRAIN
# ═══════════════════════════════════════════════════════════════

ROUGHNESS_WINDOW: Final[int] = 7
ROUGHNESS_OPTIMAL: Final[float] = 3.0
ROUGHNESS_MAX: Final[float] = 12.0


# ═══════════════════════════════════════════════════════════════
#  DISTANCE AUX COURS D'EAU
# ═══════════════════════════════════════════════════════════════
# Calibré Alpes : l'humidité du sol dépend autant du couvert forestier
# et de l'exposition que de la distance brute au cours d'eau.

DIST_WATER_OPTIMAL: Final[tuple[float, float]] = (5.0, 80.0)
DIST_WATER_GOOD: Final[float] = 100.0
DIST_WATER_MODERATE: Final[float] = 500.0
DIST_WATER_MAX: Final[float] = 1000.0

# D9 : plancher 0.15 pour cellules forestières (cours d'eau temporaires)
DIST_WATER_FOREST_FLOOR: Final[float] = 0.15

_WATER_TYPE_BONUS_DICT: dict[str, float] = {
    "bras_mort":        1.30,
    "plan_eau":         1.20,
    "canal":            1.10,
    "riviere":          1.00,
    "ruisseau":         0.90,
    "torrent":          0.70,
    "unknown":          0.90,
}

WATER_TYPE_BONUS: MappingProxyType[str, float] = MappingProxyType(
    _WATER_TYPE_BONUS_DICT,
)


# ═══════════════════════════════════════════════════════════════
# TWI — Topographic Wetness Index (fix #46 v2.3.5)
# ═══════════════════════════════════════════════════════════════
# TWI = ln(a / tan(β))  où a = aire drainée spécifique, β = pente locale
# Valeurs typiques : 2 (crête) → 5 (versant) → 12 (fond vallée) → 20+
#
# Morilles : sol bien drainé mais humide → TWI optimal entre 6 et 10

TWI_OPTIMAL: Final[tuple[float, float]] = (6.0, 10.0)
TWI_DRY_LIMIT: Final[float] = 3.0
TWI_WET_LIMIT: Final[float] = 14.0
TWI_WATERLOG: Final[float] = 18.0
TWI_DRY_FLOOR: Final[float] = 0.10
TWI_WET_FLOOR: Final[float] = 0.10


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

ASPECT_SCORES: MappingProxyType[str, float] = MappingProxyType(
    _ASPECT_SCORES_DICT,
)


# ═══════════════════════════════════════════════════════════════
#  MICRO-HABITAT
# ═══════════════════════════════════════════════════════════════

CANOPY_OPTIMAL_OPENNESS: Final[float] = 0.4
CANOPY_MIN_OPENNESS: Final[float] = 0.1
CANOPY_MAX_OPENNESS: Final[float] = 0.9

_GROUND_COVER_DICT: dict[str, float] = {
    "litiere_seche":     1.00,
    "litiere_humide":    0.60,
    "herbe_rase":        0.50,
    "mousse_legere":     0.40,
    "mousse_epaisse":    0.10,
    "lierre":            0.05,
    "sol_nu":            0.70,
}

GROUND_COVER_PREFERENCES: MappingProxyType[str, float] = MappingProxyType(
    _GROUND_COVER_DICT,
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

DISTURBANCE_SCORES: MappingProxyType[str, float] = MappingProxyType(
    _DISTURBANCE_DICT,
)


# ═══════════════════════════════════════════════════════════════
#  PHÉNOLOGIE / SAISONNALITÉ (optionnel)
# ═══════════════════════════════════════════════════════════════

PHENOLOGY_ENABLED: Final[bool] = False
PHENOLOGY_GRADIENT: Final[float] = 300.0
PHENOLOGY_BASE_MONTH: Final[int] = 3
PHENOLOGY_BASE_ALT: Final[float] = 200.0


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION DE PROBABILITÉ
# ═══════════════════════════════════════════════════════════════

PROBABILITY_THRESHOLDS: Final[tuple[float, ...]] = (
    0.15, 0.30, 0.45, 0.60, 0.75,
)

PROBABILITY_LABELS: Final[tuple[str, ...]] = (
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

_GEOPF_BASE: Final[str] = (
    "https://data.geopf.fr/wmts?"
    "SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
    "&STYLE=normal&TILEMATRIXSET=PM"
    "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
)

_BASEMAPS_DICT: dict[str, dict[str, Any]] = {
    "IGN Plan v2": {
        "tiles": (
            f"{_GEOPF_BASE}"
            "&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2"
            "&FORMAT=image/png"
        ),
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Plan v2",
        "max_zoom": 18,
    },
    "IGN Scan25": {
        "tiles": (
            f"{_GEOPF_BASE}"
            "&LAYER=GEOGRAPHICALGRIDSYSTEMS.MAPS"
            "&FORMAT=image/jpeg"
        ),
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Scan25 (topo)",
        "max_zoom": 16,
    },
    "IGN Photographies aériennes": {
        "tiles": (
            f"{_GEOPF_BASE}"
            "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
            "&FORMAT=image/jpeg"
        ),
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Photo aérienne",
        "max_zoom": 19,
    },
    "IGN Occupation du sol": {
        "tiles": (
            f"{_GEOPF_BASE}"
            "&LAYER=LANDUSE.AGRICULTURE2023"
            "&FORMAT=image/png"
        ),
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

BASEMAPS: MappingProxyType[str, dict[str, Any]] = MappingProxyType(
    _BASEMAPS_DICT,
)
DEFAULT_BASEMAP: Final[str] = "IGN Plan v2"


# ═══════════════════════════════════════════════════════════════
#  VALIDATION GLOBALE
# ═══════════════════════════════════════════════════════════════

def validate_config() -> bool:
    """Vérifie la cohérence interne de la configuration."""
    ok = True

    # ── Poids ──
    w_sum = sum(WEIGHTS.values())
    if abs(w_sum - 1.0) > 0.01:
        logger.error("Somme des poids = %.4f ≠ 1.0", w_sum)
        ok = False

    # ── Scores bornés [0, 1] ──
    for name, scores in [
        ("TREE_SCORES", TREE_SCORES),
        ("GEOLOGY_SCORES", GEOLOGY_SCORES),
        ("ASPECT_SCORES", ASPECT_SCORES),
    ]:
        for key, val in scores.items():
            if not (0.0 <= val <= 1.0):
                logger.error("%s['%s'] = %.2f hors [0, 1]", name, key, val)
                ok = False

    # ── Water type bonus ≥ 0 ──
    for key, val in WATER_TYPE_BONUS.items():
        if val < 0.0:
            logger.error("WATER_TYPE_BONUS['%s'] = %.2f < 0", key, val)
            ok = False

    # ── Seuils de pente cohérents (non chevauchants) ──
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

    # ── Altitude cohérente ──
    if ALTITUDE_OPTIMAL[0] < ALTITUDE_RANGE[0]:
        logger.warning(
            "ALTITUDE_OPTIMAL[0]=%.0f < ALTITUDE_RANGE[0]=%.0f",
            ALTITUDE_OPTIMAL[0], ALTITUDE_RANGE[0],
        )

    # ── TWI cohérent ──
    if TWI_DRY_LIMIT >= TWI_OPTIMAL[0]:
        logger.warning(
            "TWI_DRY_LIMIT=%.1f ≥ TWI_OPTIMAL[0]=%.1f",
            TWI_DRY_LIMIT, TWI_OPTIMAL[0],
        )
    if TWI_OPTIMAL[1] >= TWI_WET_LIMIT:
        logger.warning(
            "TWI_OPTIMAL[1]=%.1f ≥ TWI_WET_LIMIT=%.1f",
            TWI_OPTIMAL[1], TWI_WET_LIMIT,
        )
    if TWI_WET_LIMIT >= TWI_WATERLOG:
        logger.warning(
            "TWI_WET_LIMIT=%.1f ≥ TWI_WATERLOG=%.1f",
            TWI_WET_LIMIT, TWI_WATERLOG,
        )

    # ── BBOX valide ──
    if BBOX["xmin"] >= BBOX["xmax"] or BBOX["ymin"] >= BBOX["ymax"]:
        logger.error("BBOX dégénéré : %s", dict(BBOX))
        ok = False

    # ── Cell size raisonnable ──
    if CELL_SIZE < 1.0 or CELL_SIZE > 100.0:
        logger.warning(
            "CELL_SIZE=%.1f inhabituel (attendu 1-100m)", CELL_SIZE,
        )

    # ── Aliases résolvent vers des scores connus ──
    for alias, canonical in TREE_ALIASES.items():
        if canonical not in TREE_SCORES:
            logger.error(
                "TREE_ALIASES['%s'] → '%s' absent de TREE_SCORES",
                alias, canonical,
            )
            ok = False

    # ── Thresholds / labels cohérents ──
    if len(PROBABILITY_LABELS) != len(PROBABILITY_THRESHOLDS) + 1:
        logger.error(
            "PROBABILITY_LABELS (%d) != PROBABILITY_THRESHOLDS (%d) + 1",
            len(PROBABILITY_LABELS), len(PROBABILITY_THRESHOLDS),
        )
        ok = False

    for i in range(1, len(PROBABILITY_THRESHOLDS)):
        if PROBABILITY_THRESHOLDS[i] <= PROBABILITY_THRESHOLDS[i - 1]:
            logger.error(
                "PROBABILITY_THRESHOLDS non croissant : "
                "[%d]=%.2f ≤ [%d]=%.2f",
                i, PROBABILITY_THRESHOLDS[i],
                i - 1, PROBABILITY_THRESHOLDS[i - 1],
            )
            ok = False

    # ── Éliminatoires existent dans les scores ──
    for geo in ELIMINATORY_GEOLOGY:
        if geo not in GEOLOGY_SCORES:
            logger.error(
                "ELIMINATORY_GEOLOGY '%s' absent de GEOLOGY_SCORES", geo,
            )
            ok = False

    # ── Basemap par défaut existe ──
    if DEFAULT_BASEMAP not in BASEMAPS:
        logger.error(
            "DEFAULT_BASEMAP '%s' absent de BASEMAPS", DEFAULT_BASEMAP,
        )
        ok = False

    if ok:
        logger.info("✅ Configuration validée (v%s)", CONFIG_VERSION)
    else:
        logger.error("❌ Configuration invalide — voir erreurs ci-dessus")

    return ok