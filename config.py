"""
🍄 CARTOMORILLES — Configuration du modèle (v2.5.4)
   Zone : Isère (38) — 107.4×131.2 km, emprise dept élargie est+sud

   v2.5.4 — TENTATIVES TOUTES ABANDONNÉES (régressions confirmées) :
     - SLOPE_OPTIMAL (0,15)→(0,20) : régresse max 0.739→0.679
       contre-intuitif (slope widening augmente pourtant pré-pipeline scores).
       Cause exacte : interaction smoothing + redistribution scores neighbors.
     - ALTITUDE_OPTIMAL (200,600)→(200,700) : régresse de manière similaire
     - SOFT_ELIM_BUFFER 15m→5m : régresse max 0.739→0.680 — buffer 15m
       agit comme anti-noise indirect (réduit cellules low-score près d'élim,
       élève baricentre smoothing pour cellules forêt voisines).
     - scoring.apply_spatial_smoothing truncate=4.0→2.0 : régresse pour
       raison similaire — smoothing normalisé BOOST les borderlines via
       convolution avec voisins valides, pas réduction des pics.
     APIs ajoutées et CONSERVÉES (utiles indépendamment du score) :
     - _accel.gaussian_filter accepte truncate kwarg
     - SOFT_ELIM_BUFFER explicité dans config (avant : fallback hardcoded)
     CONFIG_VERSION conservé en 2.5.4 pour tracer les apprentissages.
     Décision : v2.5.3 retenu comme état production. D36 documente l'échec.

   v2.5.3 :
     - calc_dry threshold tree_species ≤ 0.50 → ≤ 0.30 (scoring.py)
       Exclut `unknown` (0.40) et `platane` (0.20) du malus.
       Cible restreinte aux vraies chênaies/buis xérothermophiles.
     - _CANOPY_BASE_INTERIOR 0.45 → 0.55 (grid_builder.py)
       Forêt intérieure mieux notée — cap 0.90 préservé.
     - forest_edge interior floor 0.5 → 0.65 + transition 20-50m 0.75→0.65
       (grid_builder.py) — réduit pénalité d'intérieur forêt profonde.
     - Décisions ajoutées : D35

   v2.5.2 :
     - Tree score `unknown` 0.25 → 0.40 (forêt sans essence ID = 64% du couvert)
     - Géologie calcaire saturée : calcaire 0.95 → 1.00, cascade marneux/marne/etc.
     - LANDCOVER_FOREST_FLOOR 0.80 → 0.85 (D8 floor relevé)
     - scoring.py : _COVERAGE_PENALTY_FLOOR 0.75 → 0.85
     - scoring.py : _CALCDRY_FLOOR 0.75 → 0.85, _CALCDRY_MAX_PENALTY 0.25 → 0.15
     - Cible : score_max 0.72 → 0.78, mean forêt 0.32 → 0.40
     - Décisions ajoutées : D32, D33, D34

   v2.5.0 :
     - Emprise élargie : +Oisans (est, xmax 932112→965000)
                        +Trièves (sud, ymin 6435430→6403000)
     - DEM multi-tuiles : Copernicus N44-45 × E004-006 (merge auto)
     - CELL_SIZE 10m → 20m (35.2M cellules, sous seuil warn 50M)

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
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

# ═══════════════════════════════════════════════════════════════
#  MÉTADONNÉES
# ═══════════════════════════════════════════════════════════════

CONFIG_VERSION: Final[str] = "2.5.4"

logger = logging.getLogger("cartomorilles.config")

__all__ = [
    # Métadonnées
    "CONFIG_VERSION",
    # Emprise
    "BBOX_WGS84", "BBOX", "CELL_SIZE", "MAP_CENTER", "DATA_BUFFER",
    "DEM_SOURCE", "DEM_RESOLUTION_M", "RGEALTI_WMS_URL", "RGEALTI_LAYER",
    # Chemins données
    "DEM_FILE", "FOREST_SHAPEFILE",
    "DATA_SUBDIR_GEOLOGY", "GEOLOGY_SHAPEFILE",
    # Poids
    "WEIGHTS",
    # Espèces (multi-species depuis v2.6.0)
    "SPECIES_PARAMS", "SUPPORTED_SPECIES", "DEFAULT_SPECIES",
    "get_species_params", "species_with_taxon",
    # Essences
    "TREE_SCORES", "TREE_ALIASES",
    "resolve_tree_name", "get_tree_score",
    # Géologie
    "GEOLOGY_SCORES", "GEOLOGY_BRGM_MAP", "ELIMINATORY_GEOLOGY",
    "resolve_geology", "get_geology_score",
    # Libellés d'affichage
    "TREE_LABELS", "GEOLOGY_LABELS", "tree_label", "geology_label",
    # Altitude
    "ALTITUDE_OPTIMAL", "ALTITUDE_RANGE", "ALTITUDE_ALLUVIAL_CENTER",
    # Pente
    "SLOPE_OPTIMAL", "SLOPE_MODERATE", "SLOPE_STEEP", "SLOPE_MAX", "slope_breaks",
    # Rugosité
    "ROUGHNESS_WINDOW", "ROUGHNESS_OPTIMAL", "ROUGHNESS_MAX",
    # Hydrographie
    "DIST_WATER_OPTIMAL", "DIST_WATER_GOOD", "DIST_WATER_MODERATE",
    "DIST_WATER_MAX", "DIST_WATER_FOREST_FLOOR", "WATER_TYPE_BONUS",
    "STREAM_HALF_WIDTH_M", "WATER_BANK_EXCLUSION_M",
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
    "URBAN_DIST_FULL", "URBAN_PROXIMITY_FLOOR", "URBAN_GAP_CLOSE_M",
    # Phénologie
    "PHENOLOGY_ENABLED", "PHENOLOGY_GRADIENT",
    "PHENOLOGY_BASE_MONTH", "PHENOLOGY_BASE_ALT",
    # Classification
    "PROBABILITY_THRESHOLDS", "PROBABILITY_LABELS", "CLASS_RANK_QUANTILES",
    # Fonds de carte
    "BASEMAPS", "DEFAULT_BASEMAP",
    # Validation
    "validate_config",
]


# ══════════════════════════════════════════════════════════════════════
# Emprise — Isère 38 (dept élargi : +Oisans est, +Trièves sud)
# ══════════════════════════════════════════════════════════════════════
# DEM Copernicus multi-tuiles (N44-45 × E004-006, merge auto)
# BD Forêt v2 (full dept, 127×131 km) ∩ Géologie BRGM dept 38
# Limite ouest conservée (Bas-Dauphiné). Rhône-Vienne hors scope.

_BBOX_XMIN: Final[float] = 857_571.0
_BBOX_YMIN: Final[float] = 6_403_000.0
_BBOX_XMAX: Final[float] = 965_000.0
_BBOX_YMAX: Final[float] = 6_534_209.0

_CENTER_X_L93: Final[float] = (_BBOX_XMIN + _BBOX_XMAX) / 2  # ≈ 911 286
_CENTER_Y_L93: Final[float] = (_BBOX_YMIN + _BBOX_YMAX) / 2  # ≈ 6 468 604

BBOX: MappingProxyType[str, float] = MappingProxyType({
    "xmin": _BBOX_XMIN,
    "ymin": _BBOX_YMIN,
    "xmax": _BBOX_XMAX,
    "ymax": _BBOX_YMAX,
})

# WGS84 (Overpass/OSM) — bornes convertie depuis L93 + buffer 0.001°
BBOX_WGS84: MappingProxyType[str, float] = MappingProxyType({
    "west": 4.9990,
    "south": 44.7180,
    "east": 6.4421,
    "north": 45.8574,
})

MAP_CENTER: MappingProxyType[str, float] = MappingProxyType({
    "lat": 45.288,
    "lon": 5.721,
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

CELL_SIZE: float = 20.0
DATA_BUFFER: Final[float] = 500.0

# ── MNT (D63) ────────────────────────────────────────────────────
# RGE ALTI® de l'IGN : modèle de terrain (le sol, sous les arbres et les
# bâtiments), pas de 1 à 5 m, Licence Ouverte Etalab 2.0, lu sur le WMS
# raster de la Géoplateforme à 5 m et moyenné à DEM_RESOLUTION_M. Repli :
# Copernicus GLO-30, modèle de surface (cime des arbres, toits). Pente,
# exposition, rugosité et TWI sont calculés à DEM_RESOLUTION_M quelle que
# soit la maille de la grille.
DEM_SOURCE: Final[str] = "rgealti"            # "rgealti" | "copernicus"
DEM_RESOLUTION_M: Final[float] = 10.0
RGEALTI_WMS_URL: Final[str] = "https://data.geopf.fr/wms-r/wms"
RGEALTI_LAYER: Final[str] = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"

# ── Auto-scaling résolution ──────────────────────────────────────
# (seuil_km², cell_m) — premier seuil ≥ emprise gagne.
CELL_SIZE_AUTO_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (500.0, 5.0),
    (2000.0, 10.0),
    (8000.0, 10.0),
)
CELL_SIZE_AUTO_FALLBACK: Final[float] = 50.0

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
    # ── v2.5.0 : redistribution centrée sur discriminants humidité/perturbation
    "geology":            0.20,   # top discriminant (calcaire/alluvions)
    "tree_species":       0.20,   # top discriminant (hêtre/frêne/pommier)
    "altitude":           0.08,   # +0.01 — thermique important
    "twi":                0.08,   # +0.02 — humidité sol, signal fort
    "disturbance":        0.08,   # +0.06 — vrai signal morilles (chablis, coupes)
    "dist_water":         0.07,   # +0.02 — corrélé humidité
    "slope":              0.05,   # -0.01
    "aspect":             0.05,   # +0.02 — exposition discriminante montagne
    "forest_edge":        0.05,   # -0.03 — morilles aussi en intérieur (floor 0.5)
    "ground_cover":       0.05,   # +0.01 — bon indicateur composite
    "canopy_openness":    0.04,   # -0.02 — dégradé sans landcover fin
    "terrain_roughness":  0.03,   # -0.01 — corrélé slope
    "urban_proximity":    0.01,   # -0.06 — éliminatoire gère déjà
    "geology_contact":    0.01,   # -0.01 — redondant avec geology
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

URBAN_DIST_ELIMINATORY: Final[float] = 15.0    # m — score dur → 0
URBAN_DIST_PENALTY: Final[float] = 100.0       # m — fin de pénalité forte
URBAN_DIST_FULL: Final[float] = 250.0          # m — plus aucune pénalité
URBAN_PROXIMITY_FLOOR: Final[float] = 0.05     # plancher zone tampon
# Filtre densité urbaine (résolution-adaptatif)
# Calibré @10m : seuil=30 dans fenêtre 7×7=49 → fraction ≈ 0.61
URBAN_DENSITY_RADIUS_M: Final[float] = 30.0
URBAN_DENSITY_FRACTION: Final[float] = 30.0 / 49.0
URBAN_DENSITY_MIN_WINDOW: Final[int] = 5  # px — plancher pour grandes cellules
# Tissu urbain en maille fine (D56) : la couche urbaine n'a que des
# bâtiments ; les vides de moins de 2 × URBAN_GAP_CLOSE_M entre eux (rues,
# places, cours) sont comblés par fermeture morphologique. Rayon en cellules
# arrondi vers le bas : rien à combler à 50 m, où une cellule touchée par
# un bâtiment couvre déjà la rue.
URBAN_GAP_CLOSE_M: Final[float] = 40.0


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
    # v2.5.1 : feuillus montagnards rehaussés (M. elata en hêtraies-
    # sapinières Vercors/Chartreuse/Belledonne, sol calcaire frais)
    "hetre":              0.75,   # était 0.30 — hêtraies Vouillants & Co
    "tilleul":            0.65,   # était 0.35 — sol calcaire frais
    "erable":             0.65,   # était 0.30 — sycomore en versants
    "charme":             0.55,   # était 0.35
    # ── Conifères montagnards (M. elata / M. conica) ─────────────
    # v2.3.6 : rehaussés — M. elata documentée en sapinière calcaire,
    # épicéa en forêts mixtes montagne, pin sylvestre sols calcaires secs
    "sapin":              0.55,
    "epicea":             0.45,
    "pin_sylvestre":      0.45,
    "douglas":            0.30,
    "meleze":             0.30,   # D49 : mélézins = morille noire, pas la grise
    # ── Essences peu/pas favorables ──
    "platane":            0.20,
    "robinier":           0.15,
    "chene":              0.15,
    "chene_pubescent":    0.10,
    "bouleau":            0.05,
    "buis":               0.05,
    # ── Défaut — forêt sans essence connue ≠ pas de forêt ──
    # v2.5.2 : 0.25 → 0.40 — forêt taguée sans essence ID = 64% du couvert
    # D46 : NaN — information manquante, pas une essence médiocre : la
    # cellule est notée sur ses autres critères (poids renormalisés, légère
    # pénalité de couverture D10). Dans l'enrichissement IFN, une essence
    # notée garde son score ; « feuillus » / « conifères » sans essence → NaN.
    "unknown":            float("nan"),
}

# ── Landcover : plancher green_score pour cellules BD Forêt ──
# D8 : Forest floor pour cellules BD Forêt
# v2.5.2 : 0.80 → 0.85 — relève plancher micro-habitats forêt
LANDCOVER_FOREST_FLOOR: Final[float] = 0.85
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
    # ── Mélèze ──
    "larix": "meleze", "larix_decidua": "meleze",
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


def get_tree_score(raw_name: str | None, species: str | None = None) -> float:
    """Retourne le score d'une essence pour l'espèce modélisée (`species` :
    None = DEFAULT_SPECIES, la morille ; « edulis » = table du cèpe — D47).
    NaN si l'essence est inconnue : aucune information, à exclure des
    moyennes (D46)."""
    table: MappingProxyType[str, float] = (
        TREE_SCORES if species is None
        else get_species_params(species)["tree_scores"]
    )
    canonical = resolve_tree_name(raw_name)
    return float(table.get(canonical, table["unknown"]))


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
    # v2.5.2 : saturation optimum biblio (D33) — calcaire = pic absolu
    "calcaire":              1.00,   # était 0.95
    "calcaire_marneux":      0.95,   # était 0.90
    "marne":                 0.90,   # était 0.85
    "alluvions_calcaires":   0.90,   # était 0.85
    "calcaire_lacustre":     0.85,   # était 0.80
    "eboulis_calcaire":      0.80,   # était 0.75
    "dolomie":               0.80,   # était 0.75
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
#  LIBELLÉS D'AFFICHAGE (carte + console)
# ═══════════════════════════════════════════════════════════════
# Clés canoniques (TREE_SCORES / GEOLOGY_SCORES) → libellés lisibles.
# « unknown » se lit après un intitulé (« Essence dominante : non précisée »).

TREE_LABELS: MappingProxyType[str, str] = MappingProxyType({
    "frene":            "Frêne",
    "orme":             "Orme",
    "pommier":          "Pommier",
    "poirier":          "Poirier",
    "chataignier":      "Châtaignier",
    "peuplier":         "Peuplier",
    "aulne":            "Aulne",
    "saule":            "Saule",
    "noisetier":        "Noisetier",
    "hetre":            "Hêtre",
    "tilleul":          "Tilleul",
    "erable":           "Érable",
    "charme":           "Charme",
    "sapin":            "Sapin",
    "epicea":           "Épicéa",
    "pin_sylvestre":    "Pin sylvestre",
    "douglas":          "Douglas",
    "meleze":           "Mélèze",
    "platane":          "Platane",
    "robinier":         "Robinier",
    "chene":            "Chêne",
    "chene_pubescent":  "Chêne pubescent",
    "bouleau":          "Bouleau",
    "buis":             "Buis",
    "unknown":          "non précisée",
})

GEOLOGY_LABELS: MappingProxyType[str, str] = MappingProxyType({
    "calcaire":             "Calcaire",
    "calcaire_marneux":     "Calcaire marneux",
    "marne":                "Marne",
    "alluvions_calcaires":  "Alluvions calcaires",
    "calcaire_lacustre":    "Calcaire lacustre",
    "eboulis_calcaire":     "Éboulis calcaire",
    "dolomie":              "Dolomie",
    "colluvions":           "Colluvions",
    "alluvions":            "Alluvions",
    "alluvions_recentes":   "Alluvions récentes",
    "moraine":              "Moraine",
    "flysch":               "Flysch",
    "sable":                "Sable",
    "molasse":              "Molasse",
    "gres":                 "Grès",
    "schiste":              "Schiste",
    "remblai":              "Remblai",
    "hydrographie":         "Réseau hydrographique",
    "granite":              "Granite",
    "gneiss":               "Gneiss",
    "siliceux":             "Roche siliceuse",
    "unknown":              "non classée",
})


def _display_label(key: object, labels: MappingProxyType[str, str]) -> str | None:
    if key is None:
        return None
    text = str(key).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    if text in labels:
        return labels[text]
    # Clé absente de la table : jamais de « _ » brut à l'écran
    text = text.replace("_", " ")
    return text[:1].upper() + text[1:]


def tree_label(key: object) -> str | None:
    """Libellé affichable d'une essence canonique (None si vide)."""
    return _display_label(key, TREE_LABELS)


def geology_label(key: object) -> str | None:
    """Libellé affichable d'une catégorie géologique (None si vide)."""
    return _display_label(key, GEOLOGY_LABELS)


# ═══════════════════════════════════════════════════════════════
#  ALTITUDE  (v2.2.0 : optimal élargi 200-600m)
# ═══════════════════════════════════════════════════════════════
#
#  M. esculenta — optimal 200-600m (plaine + collines)
#  M. elata     — peut monter à 800-1200m (conifères)

ALTITUDE_OPTIMAL: Final[tuple[float, float]] = (200.0, 600.0)   # v2.5.4 (200,700) testé puis reverté
ALTITUDE_RANGE: Final[tuple[float, float]] = (150.0, 900.0)

# Centre du bonus micro-nappe / humidité résiduelle — sans effet : le bonus
# s'ajoutait à des cellules déjà à 1,0 puis était plafonné à 1,0 (retiré du
# score d'altitude, D60 ; constante gardée pour compatibilité)
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
#  40-45°  : très raide (0.1 → 0.0)
#    >45°  : éliminatoire

SLOPE_OPTIMAL: Final[tuple[float, float]] = (0.0, 15.0)   # v2.5.4 (0,20) testé puis reverté
SLOPE_MODERATE: Final[float] = 30.0
SLOPE_STEEP: Final[float] = 40.0
SLOPE_MAX: Final[float] = 45.0

# v2.5.4 : SOFT_ELIM_BUFFER explicité dans config (auparavant fallback 15.0
# dans scoring.py via getattr). Tests 5m régressaient le max — 15m conservé.
# D60 : 0 = pas de transition. À 15 m, l'arrondi au pixel la rendait inactive
# dès 11 m de maille (20 et 50 m : toutes les mesures D45–D59) et divisait
# par deux, à 10 m seulement, la cellule voisine de toute zone éliminée —
# dont la berge des cours d'eau, habitat favorable. Mesurée en mètres
# depuis le bord de la zone éliminée si on la réactive.
SOFT_ELIM_BUFFER: Final[float] = 0.0


# ═══════════════════════════════════════════════════════════════
#  RUGOSITÉ TERRAIN
# ═══════════════════════════════════════════════════════════════

# Fenêtre (pixels du MNT natif, 20 m : 100 m) de l'écart-type local de la
# pente. 7 n'était pas lu : grid_builder avait sa propre copie à 5 (D60)
ROUGHNESS_WINDOW: Final[int] = 5
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

# Bords de l'eau (D55) : la berge est exclue sur WATER_BANK_EXCLUSION_M
# depuis la rive (choix de l'utilisateur : 5 m, pas la maille entière).
# Une cellule est éliminée si son centre tombe dans l'eau ou dans cette
# bande. Rive = contour des surfaces en eau BD TOPO (lacs, lits des
# rivières de plus de 7,5 m) ; un cours d'eau sans surface est une ligne,
# de demi-largeur supposée STREAM_HALF_WIDTH_M.
STREAM_HALF_WIDTH_M: Final[float] = 2.0
WATER_BANK_EXCLUSION_M: Final[float] = 5.0

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

# Classes de la carte par rang (D52). Le score se tasse entre 0,45 et 0,85
# pour toute forêt non éliminée : des seuils absolus classaient un tiers à
# la moitié de l'Isère en Bon ou Excellent. Chaque espèce est donc classée
# sur ses propres cellules notées (score ≥ PROBABILITY_THRESHOLDS[0]) de la
# zone étudiée : bornes cumulées des quantiles Très faible | Faible | Moyen
# | Bon | Excellent → 40 % / 30 % / 20 % / 8 % / 2 %. Sous le premier seuil
# absolu : Nul (cellule éliminée). Les seuils absolus restent l'échelle de
# la météo (D44) et le repli quand la zone n'a pas de cellule notée.
CLASS_RANK_QUANTILES: Final[tuple[float, ...]] = (0.40, 0.70, 0.90, 0.98)

# Anciens seuils binaires (n_high / n_medium) : gardés seulement dans le
# rapport JSON pour comparer avec les calibrations passées (D35). La carte,
# la console et la météo utilisent les 6 classes ci-dessus.
SCORE_THRESHOLD_HIGH: Final[float] = 0.7
SCORE_THRESHOLD_MEDIUM: Final[float] = 0.4

# Nombre de grilles float32 gardées en mémoire simultanément (estim. RAM)
N_SCORING_GRIDS: Final[int] = 15


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
    # Remplace « IGN Scan25 » (GEOGRAPHICALGRIDSYSTEMS.MAPS : HTTP 400 sans
    # clé d'API). Ombrage BD ALTI 25 m sans clé : noir semi-transparent, donc
    # relief gris sur le fond neutre de Leaflet. Style imposé par le service.
    "IGN Estompage": {
        "tiles": (
            "https://data.geopf.fr/wmts?"
            "SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            "&STYLE=estompage_grayscale&TILEMATRIXSET=PM"
            "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
            "&LAYER=ELEVATION.ELEVATIONGRIDCOVERAGE.SHADOW"
            "&FORMAT=image/png"
        ),
        "attr": "© IGN - Géoplateforme",
        "name": "IGN Estompage (relief)",
        "max_zoom": 18,
        "max_native_zoom": 15,   # pas de tuiles au-delà : agrandies
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
#  SPECIES_PARAMS — paramètres écologiques par espèce (v2.6.0)
# ═══════════════════════════════════════════════════════════════
#
#  Ajoute Boletus edulis (cèpe) à côté de Morchella esculenta.
#  Les constantes morille existantes (TREE_SCORES, GEOLOGY_SCORES,
#  ALTITUDE_OPTIMAL, etc.) sont préservées comme défauts ; elles
#  alimentent le sous-dict "esculenta". Le sous-dict "edulis"
#  redéfinit chaque paramètre selon l'écologie de B. edulis :
#  - Montagnard (alt 600-1400m vs 200-600m morille)
#  - Mycorhize Pinaceae/Fagaceae (sapin/épicéa/hêtre/chêne ≥ 0.85)
#  - Pas de thermophilie : aspect N/NE favorisés
#  - Ubiquiste géologie : ELIMINATORY_GEOLOGY = ∅ (granite OK)
#  - Pénalité calcdry désactivée (pas d'anti-habitat xérique chêne-buis)
#
#  Utilisation :
#    params = config.SPECIES_PARAMS["edulis"]
#    optimal = params["altitude_optimal"]
#

# ── Cèpe : tree_scores (mêmes clés canoniques que TREE_SCORES morille) ──
_TREE_SCORES_EDULIS_DICT: dict[str, float] = {
    # ── Feuillus de plaine ──
    "frene":              0.40,   # médiocre (sols saturés/calcaires fertiles)
    "orme":               0.35,
    "pommier":            0.50,
    "poirier":            0.50,
    "chataignier":        0.85,   # excellent — castaneraies acides
    "peuplier":           0.20,   # mauvais — sols saturés
    "aulne":              0.25,
    "saule":              0.15,
    "noisetier":          0.55,
    # ── Feuillus mycorhiziques montagne (excellents partenaires) ──
    "hetre":              0.95,   # hêtraies Vercors/Chartreuse — top
    "chene":              0.90,   # chênaies sessiles/pédonculées
    "chene_pubescent":    0.60,   # xérique — moins favorable
    "tilleul":            0.40,
    "erable":             0.45,
    "charme":             0.65,
    "bouleau":            0.75,   # bouleaux acides — bons
    # ── Conifères : optimums absolus B. edulis ──
    "sapin":              1.00,   # sapinière calcaire/acide — optimal
    "epicea":             1.00,   # pessière montagnarde — optimal
    "pin_sylvestre":      0.85,
    "douglas":            0.70,
    "meleze":             0.70,   # D49 : cèpes des mélézins
    # ── Essences peu/pas favorables ──
    "platane":            0.10,
    "robinier":           0.30,   # allélopathique
    "buis":               0.10,
    # ── Défaut — forêt sans essence connue : exclue de la moyenne (D46) ──
    "unknown":            float("nan"),
}

# ── Cèpe : geology_scores (mêmes clés canoniques que GEOLOGY_SCORES) ──
# Cèpe ubiquiste : tolère acide ET calcaire. Préfère substrats drainants
# (granite/gneiss/schiste/grès) ; calcaire correct mais pas optimum (vs
# morille saturée à calcaire). Alluvions saturées défavorables.
_GEOLOGY_SCORES_EDULIS_DICT: dict[str, float] = {
    # ── Calcaires : bons mais pas optimaux pour cèpe ──
    "calcaire":              0.70,
    "calcaire_marneux":      0.65,
    "marne":                 0.55,
    "alluvions_calcaires":   0.55,
    "calcaire_lacustre":     0.60,
    "eboulis_calcaire":      0.65,
    "dolomie":               0.65,
    # ── Substrats alluviaux : faibles (saturation, peuplements monocultures) ──
    "colluvions":            0.65,
    "alluvions":             0.40,
    "alluvions_recentes":    0.30,
    # ── Substrats intermédiaires ──
    "moraine":               0.85,   # souvent matériel cristallin granitique
    "flysch":                0.55,
    "sable":                 0.55,
    "molasse":               0.65,
    # ── Substrats acides drainants : OPTIMAUX cèpe ──
    "gres":                  0.85,
    "schiste":               0.85,
    "granite":               0.95,
    "gneiss":                0.90,
    "siliceux":              0.95,
    # ── Substrats non favorables ──
    "remblai":               0.10,
    "hydrographie":          0.30,
    # ── Par défaut ──
    "unknown":               0.45,
}

# ── Cèpe : aspect_scores (mêmes clés canoniques que ASPECT_SCORES) ──
# Cèpe non-thermophile : préfère versants frais N/NE (humidité retenue).
# Sud sec moins favorable.
_ASPECT_SCORES_EDULIS_DICT: dict[str, float] = {
    "S":    0.65,
    "SE":   0.70,
    "flat": 0.85,
    "SW":   0.70,
    "E":    0.80,
    "W":    0.80,
    "NE":   1.00,
    "NW":   0.90,
    "N":    1.00,
}

# ── Frozen wrappers pour les sous-dicts cèpe ──
_TREE_SCORES_EDULIS: MappingProxyType[str, float] = MappingProxyType(
    _TREE_SCORES_EDULIS_DICT,
)
_GEOLOGY_SCORES_EDULIS: MappingProxyType[str, float] = MappingProxyType(
    _GEOLOGY_SCORES_EDULIS_DICT,
)
_ASPECT_SCORES_EDULIS: MappingProxyType[str, float] = MappingProxyType(
    _ASPECT_SCORES_EDULIS_DICT,
)

# ── Morille noire (groupe Morchella elata : elata, conica, deliciosa,
#    purpurascens) — D49. Conifères montagnards optimaux, feuillus de plaine
#    médiocres ; sols acides notés bas mais pas éliminés.
_TREE_SCORES_ELATA_DICT: dict[str, float] = {
    # ── Conifères montagnards : optimum ──
    "sapin":              0.95,
    "epicea":             0.95,   # BD Forêt confond « Sapin, épicéa »
    "meleze":             0.85,   # mélézins (Oisans, Belledonne)
    "pin_sylvestre":      0.60,   # pins à crochets, cembro, noir y sont résolus
    "douglas":            0.40,
    # ── Feuillus de montagne (hêtraies-sapinières) ──
    "hetre":              0.60,
    "erable":             0.50,
    "bouleau":            0.40,   # pionnier après coupe ou chablis
    "noisetier":          0.40,
    "tilleul":            0.35,
    # ── Feuillus de plaine : habitat de la morille grise, pas de la noire ──
    "frene":              0.30,
    "orme":               0.30,
    "pommier":            0.30,
    "poirier":            0.30,
    "chataignier":        0.30,
    "aulne":              0.30,
    "charme":             0.30,
    "peuplier":           0.25,
    "saule":              0.20,
    "chene":              0.20,
    "chene_pubescent":    0.10,
    "robinier":           0.10,
    "platane":            0.05,
    "buis":               0.05,
    # ── Forêt sans essence connue : exclue de la moyenne (D46) ──
    "unknown":            float("nan"),
}

_GEOLOGY_SCORES_ELATA_DICT: dict[str, float] = {
    # ── Calcaires : optimum, comme la grise ──
    "calcaire":              1.00,
    "calcaire_marneux":      0.90,
    "dolomie":               0.85,
    "eboulis_calcaire":      0.70,
    "marne":                 0.70,
    "calcaire_lacustre":     0.70,
    "alluvions_calcaires":   0.60,
    # ── Dépôts de montagne : fréquents sous les pessières ──
    "moraine":               0.80,
    "colluvions":            0.65,
    "flysch":                0.60,
    # ── Alluviaux et intermédiaires ──
    "alluvions":             0.45,
    "molasse":               0.45,
    "schiste":               0.40,
    "sable":                 0.40,
    "alluvions_recentes":    0.35,
    "gres":                  0.35,
    "hydrographie":          0.30,
    # ── Sols acides : notés bas, pas éliminatoires (≠ grise) ──
    "granite":               0.25,
    "gneiss":                0.25,
    "siliceux":              0.25,
    "remblai":               0.10,
    # ── Par défaut ──
    "unknown":               0.35,
}

_TREE_SCORES_ELATA: MappingProxyType[str, float] = MappingProxyType(
    _TREE_SCORES_ELATA_DICT,
)
_GEOLOGY_SCORES_ELATA: MappingProxyType[str, float] = MappingProxyType(
    _GEOLOGY_SCORES_ELATA_DICT,
)

# ── Liste des espèces supportées + défaut ──
SUPPORTED_SPECIES: Final[tuple[str, ...]] = ("esculenta", "elata", "edulis")
DEFAULT_SPECIES: Final[str] = "esculenta"

# ── SPECIES_PARAMS : dispatch par espèce ──
# Esculenta = morille grise (référence existante, ALT/SLOPE/ASPECT/TWI inchangés).
# Elata     = morille noire (conifères montagnards, D49).
# Edulis    = cèpe   (paramètres distincts, validés brainstorming).
# "taxon" = (nom, rang) GBIF des occurrences naturalistes de l'espèce ; les
# deux morilles partagent le genre Morchella (observateurs peu fiables entre
# grise et noire).
_SPECIES_PARAMS_DICT: dict[str, MappingProxyType[str, Any]] = {
    "esculenta": MappingProxyType({
        "common_name":           "Morille grise",
        "common_name_plural":    "morilles grises",
        "scientific_name":       "Morchella esculenta",
        "tree_scores":           TREE_SCORES,
        "geology_scores":        GEOLOGY_SCORES,
        "eliminatory_geology":   ELIMINATORY_GEOLOGY,
        "altitude_optimal":      ALTITUDE_OPTIMAL,        # (200, 600)
        "altitude_range":        ALTITUDE_RANGE,          # (150, 900)
        "alt_eliminatory_min":   100.0,
        "alt_eliminatory_max":   1400.0,
        "slope_optimal":         SLOPE_OPTIMAL,           # (0, 15)
        "slope_max":             SLOPE_MAX,               # 45
        "aspect_scores":         ASPECT_SCORES,
        "aspect_ref_deg":        180.0,                   # sud
        "twi_optimal":           TWI_OPTIMAL,             # (6, 10)
        "twi_dry_limit":         TWI_DRY_LIMIT,
        "twi_wet_limit":         TWI_WET_LIMIT,
        "twi_waterlog":          TWI_WATERLOG,            # 18
        "apply_calcdry":         True,
        "apply_monotony":        True,
        "taxon":                 ("Morchella", "GENUS"),
    }),
    "elata": MappingProxyType({
        "common_name":           "Morille noire",
        "common_name_plural":    "morilles noires",
        "scientific_name":       "Morchella elata (groupe)",
        "tree_scores":           _TREE_SCORES_ELATA,
        "geology_scores":        _GEOLOGY_SCORES_ELATA,
        "eliminatory_geology":   frozenset(),             # ∅ — sols acides notés bas
        "altitude_optimal":      (800.0, 1500.0),
        "altitude_range":        (500.0, 1900.0),
        "alt_eliminatory_min":   300.0,
        "alt_eliminatory_max":   2200.0,                  # limite des arbres, mélézins
        "slope_optimal":         (0.0, 30.0),
        "slope_max":             45.0,
        "aspect_scores":         ASPECT_SCORES,           # comme la grise
        "aspect_ref_deg":        180.0,                   # sud
        "twi_optimal":           TWI_OPTIMAL,             # comme la grise
        "twi_dry_limit":         TWI_DRY_LIMIT,
        "twi_wet_limit":         TWI_WET_LIMIT,
        "twi_waterlog":          TWI_WATERLOG,
        "apply_calcdry":         False,                   # anti-habitat xérique de plaine
        "apply_monotony":        True,
        "taxon":                 ("Morchella", "GENUS"),  # occurrences partagées avec la grise
    }),
    "edulis": MappingProxyType({
        "common_name":           "Cèpe",
        "common_name_plural":    "cèpes",
        "scientific_name":       "Boletus edulis",
        "tree_scores":           _TREE_SCORES_EDULIS,
        "geology_scores":        _GEOLOGY_SCORES_EDULIS,
        "eliminatory_geology":   frozenset(),             # ∅ — cèpe ubiquiste
        "altitude_optimal":      (600.0, 1400.0),
        "altitude_range":        (300.0, 1800.0),
        "alt_eliminatory_min":   200.0,
        "alt_eliminatory_max":   2200.0,
        "slope_optimal":         (0.0, 25.0),
        "slope_max":             45.0,
        "aspect_scores":         _ASPECT_SCORES_EDULIS,
        "aspect_ref_deg":        0.0,                     # nord
        "twi_optimal":           (5.0, 9.0),
        "twi_dry_limit":         3.0,
        "twi_wet_limit":         12.0,
        "twi_waterlog":          16.0,
        # Mycorhizien strict : pas de cèpe sans arbre hôte — cellule à plus
        # de 30 m d'un peuplement BD Forêt (landes, pelouses et forêts sans
        # couvert exclues) éliminée (D57). Absent = pas de règle (morilles,
        # qui sortent aussi des jardins, vergers et lisières).
        "host_tree_max_dist_m":  30.0,
        "apply_calcdry":         False,                   # OFF — cèpe pas anti-xérique
        "apply_monotony":        True,
        "taxon":                 ("Boletus edulis", "SPECIES"),
    }),
}

SPECIES_PARAMS: MappingProxyType[str, MappingProxyType[str, Any]] = MappingProxyType(
    _SPECIES_PARAMS_DICT,
)


def get_species_params(species: str) -> MappingProxyType[str, Any]:
    """Retourne les paramètres d'une espèce (fallback default si inconnue)."""
    return SPECIES_PARAMS.get(species, SPECIES_PARAMS[DEFAULT_SPECIES])


def species_with_taxon(taxon_name: str) -> tuple[str, ...]:
    """Espèces modélisées (ordre de SUPPORTED_SPECIES) dont le taxon
    naturaliste porte ce nom : « Morchella » → les deux morilles (D49)."""
    return tuple(
        sp for sp in SUPPORTED_SPECIES
        if SPECIES_PARAMS[sp]["taxon"][0] == taxon_name
    )


# ═══════════════════════════════════════════════════════════════
#  VALIDATION GLOBALE
# ═══════════════════════════════════════════════════════════════

def slope_breaks(params: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Bornes du score de pente d'une espèce : (fin de l'optimum, fin du
    modéré, fin du raide, éliminatoire). Chaque segment garde au moins 5°
    pour que le score reste continu quand l'optimum de l'espèce atteint
    SLOPE_MODERATE (morille noire : optimum jusqu'à 30°) — D60."""
    opt_max = float(params["slope_optimal"][1])
    moderate = max(SLOPE_MODERATE, opt_max + 5.0)
    steep = max(SLOPE_STEEP, moderate + 5.0)
    return opt_max, moderate, steep, float(params["slope_max"])


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
            if name == "TREE_SCORES" and key == "unknown" and math.isnan(val):
                continue   # essence inconnue = exclue de la moyenne (D46)
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

    # ── Tissu urbain (D56) ──
    if URBAN_GAP_CLOSE_M < 0:
        logger.error("URBAN_GAP_CLOSE_M=%.1f négatif", URBAN_GAP_CLOSE_M)
        ok = False

    # ── Bords de l'eau (D55) ──
    if STREAM_HALF_WIDTH_M <= 0 or WATER_BANK_EXCLUSION_M < 0:
        logger.error(
            "STREAM_HALF_WIDTH_M=%.1f / WATER_BANK_EXCLUSION_M=%.1f invalides",
            STREAM_HALF_WIDTH_M, WATER_BANK_EXCLUSION_M,
        )
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

    # ── Classes par rang (D52) : une borne de moins que les seuils absolus
    #    (Nul reste absolu), quantiles croissants dans ]0, 1[ ──
    q = CLASS_RANK_QUANTILES
    if (
        len(q) != len(PROBABILITY_THRESHOLDS) - 1
        or any(not 0.0 < v < 1.0 for v in q)
        or any(b <= a for a, b in zip(q, q[1:]))
    ):
        logger.error("CLASS_RANK_QUANTILES invalide : %s", q)
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

    # ── SPECIES_PARAMS : cohérence par espèce ──
    if DEFAULT_SPECIES not in SPECIES_PARAMS:
        logger.error(
            "DEFAULT_SPECIES '%s' absent de SPECIES_PARAMS", DEFAULT_SPECIES,
        )
        ok = False
    for sp_name in SUPPORTED_SPECIES:
        if sp_name not in SPECIES_PARAMS:
            logger.error("SUPPORTED_SPECIES '%s' absent de SPECIES_PARAMS", sp_name)
            ok = False
            continue
        params = SPECIES_PARAMS[sp_name]
        taxon = params.get("taxon")
        if (
            not isinstance(taxon, tuple) or len(taxon) != 2
            or not all(isinstance(t, str) and t for t in taxon)
        ):
            logger.error(
                "SPECIES_PARAMS['%s'].taxon doit être (nom, rang) : %r",
                sp_name, taxon,
            )
            ok = False
        # Scores [0,1] ; « unknown » = NaN (exclu de la moyenne, D46)
        for tree_key, tree_val in params["tree_scores"].items():
            if tree_key == "unknown" and math.isnan(tree_val):
                continue
            if not (0.0 <= tree_val <= 1.0):
                logger.error(
                    "SPECIES_PARAMS['%s'].tree_scores['%s']=%.2f hors [0,1]",
                    sp_name, tree_key, tree_val,
                )
                ok = False
        for geo_key, geo_val in params["geology_scores"].items():
            if not (0.0 <= geo_val <= 1.0):
                logger.error(
                    "SPECIES_PARAMS['%s'].geology_scores['%s']=%.2f hors [0,1]",
                    sp_name, geo_key, geo_val,
                )
                ok = False
        for asp_key, asp_val in params["aspect_scores"].items():
            if not (0.0 <= asp_val <= 1.0):
                logger.error(
                    "SPECIES_PARAMS['%s'].aspect_scores['%s']=%.2f hors [0,1]",
                    sp_name, asp_key, asp_val,
                )
                ok = False
        # Ranges cohérents
        alt_opt = params["altitude_optimal"]
        alt_rng = params["altitude_range"]
        if alt_opt[0] >= alt_opt[1]:
            logger.error("SPECIES_PARAMS['%s'].altitude_optimal dégénéré: %s",
                         sp_name, alt_opt)
            ok = False
        if alt_rng[0] >= alt_rng[1]:
            logger.error("SPECIES_PARAMS['%s'].altitude_range dégénéré: %s",
                         sp_name, alt_rng)
            ok = False
        if (params["alt_eliminatory_min"] >= alt_opt[0]
                or params["alt_eliminatory_max"] <= alt_opt[1]):
            logger.warning(
                "SPECIES_PARAMS['%s'] altitude_eliminatory ne contient pas "
                "altitude_optimal", sp_name,
            )
        slo_opt = params["slope_optimal"]
        _o, _m, steep_end, slope_max = slope_breaks(params)
        if steep_end >= slope_max:
            logger.error(
                "SPECIES_PARAMS['%s'] : pente raide jusqu'à %.0f° ≥ "
                "slope_max=%.0f°", sp_name, steep_end, slope_max,
            )
            ok = False
        if slo_opt[0] >= slo_opt[1] or slo_opt[1] > params["slope_max"]:
            logger.error(
                "SPECIES_PARAMS['%s'].slope_optimal/slope_max incohérent: "
                "%s, max=%.1f", sp_name, slo_opt, params["slope_max"],
            )
            ok = False
        twi_opt = params["twi_optimal"]
        if twi_opt[0] >= twi_opt[1]:
            logger.error("SPECIES_PARAMS['%s'].twi_optimal dégénéré: %s",
                         sp_name, twi_opt)
            ok = False
        host_max = params.get("host_tree_max_dist_m")
        if host_max is not None and float(host_max) < 0:
            logger.error(
                "SPECIES_PARAMS['%s'].host_tree_max_dist_m négatif : %s",
                sp_name, host_max,
            )
            ok = False
        # Éliminatoires existent dans geology_scores
        for geo in params["eliminatory_geology"]:
            if geo not in params["geology_scores"]:
                logger.error(
                    "SPECIES_PARAMS['%s'].eliminatory_geology '%s' absent "
                    "de geology_scores", sp_name, geo,
                )
                ok = False
        # Libellés d'affichage : non bloquant (repli « clé sans _ »)
        for kind, keys, labels in (
            ("TREE_LABELS", params["tree_scores"], TREE_LABELS),
            ("GEOLOGY_LABELS", params["geology_scores"], GEOLOGY_LABELS),
        ):
            missing = sorted(k for k in keys if k not in labels)
            if missing:
                logger.warning(
                    "%s : clés sans libellé (%s) : %s",
                    kind, sp_name, ", ".join(missing),
                )

    if ok:
        logger.info("✅ Configuration validée (v%s)", CONFIG_VERSION)
    else:
        logger.error("❌ Configuration invalide — voir erreurs ci-dessus")

    return ok