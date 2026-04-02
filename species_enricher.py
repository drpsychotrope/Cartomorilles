"""
species_enricher.py — Enrichissement essences forestières inconnues.

Cascade à 4 niveaux :
  A. BD Forêt v2 ESSENCE/CODE_TFV/TFV texte → espèce directe
  B. Statistiques régionales IFN 1997 × type forêt × altitude × substrat
  C. Observations terrain utilisateur (JSON)
  D. Modèle altitude-only (fallback ultime)

Calibré sur :
  - BD Forêt v2 Isère (FORMATION_VEGETALE.shp, 20 191 polygones)
  - Régions forestières IFN (rfifn250_l93.shp, DEP=38, 11 régions)
  - IFN Isère 3ème inventaire 1997

v2.4.1 — Rasterisation parallèle + cache (forêt type, substrat depuis
          geology int raster), enrichissement vectorisé par combo keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

import _accel
import config

try:
    import geopandas as gpd
    from shapely.geometry import box

    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False

logger = logging.getLogger("cartomorilles.species_enricher")

__all__ = ["SpeciesEnricher"]

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES — MAPPING BD FORÊT v2 (calibré sur données réelles)
# ═══════════════════════════════════════════════════════════════════

# CODE_TFV espèce (2 chiffres) → canonical
# Réf : BD Forêt v2 documentation IGN — table des codes espèces
_TFV_SPECIES: dict[str, str] = {
    # ── Indéterminés ──
    "00": "unknown",
    "49": "unknown",
    "90": "unknown",
    "91": "unknown",
    # ── Chênes (groupe G01) ──
    "01": "chêne_sessile",
    "02": "chêne_sessile",
    "03": "chêne_sessile",
    "04": "chêne_pubescent",
    "05": "chêne_pubescent",
    "06": "chêne_sessile",
    # ── Feuillus majeurs ──
    "07": "châtaignier",
    "08": "charme",
    "09": "hêtre",
    "10": "châtaignier",
    "11": "bouleau",
    "12": "frêne",
    "13": "érable_champêtre",
    "14": "robinier",
    "15": "aulne",
    "16": "aulne",
    "17": "tilleul",
    "18": "peuplier",
    "19": "peuplier",
    "20": "orme",
    "21": "merisier",
    "22": "noisetier",
    "23": "noyer",
    "24": "saule",
    "25": "tremble",
    # ── Conifères ──
    "50": "pin_sylvestre",
    "51": "pin_sylvestre",
    "52": "pin_sylvestre",
    "53": "pin_noir",
    "54": "pin_noir",
    "55": "pin_noir",
    "56": "pin_sylvestre",
    "58": "pin_à_crochets",
    "61": "sapin",
    "62": "sapin",
    "63": "mélèze",
    "64": "douglas",
    "65": "épicéa",
    "80": "pin_sylvestre",
    "81": "pin_sylvestre",
}

# ESSENCE (texte exact BD Forêt v2 Isère) → canonical
_ESSENCE_MAP: dict[str, str] = {
    "Châtaignier":                "châtaignier",
    "Chênes décidus":             "chêne_sessile",
    "Chêne pédonculé":            "chêne_sessile",
    "Chêne sessile":              "chêne_sessile",
    "Chêne pubescent":            "chêne_pubescent",
    "Chêne tauzin":               "chêne_pubescent",
    "Chêne vert":                 "chêne_sessile",
    "Hêtre":                      "hêtre",
    "Robinier":                   "robinier",
    "Frêne":                      "frêne",
    "Charme":                     "charme",
    "Érable":                     "érable_champêtre",
    "Érables":                    "érable_champêtre",
    "Orme":                       "orme",
    "Tilleul":                    "tilleul",
    "Bouleau":                    "bouleau",
    "Aulne":                      "aulne",
    "Aulne glutineux":            "aulne",
    "Merisier":                   "merisier",
    "Noisetier":                  "noisetier",
    "Noyer":                      "noyer",
    "Saule":                      "saule",
    "Tremble":                    "tremble",
    "Peuplier":                   "peuplier",
    "Sapin, épicéa":              "sapin",
    "Sapin pectiné":              "sapin",
    "Épicéa commun":              "épicéa",
    "Pin sylvestre":              "pin_sylvestre",
    "Pin laricio, pin noir":      "pin_noir",
    "Pin noir":                   "pin_noir",
    "Pin d'Alep":                 "pin_noir",
    "Pin à crochets, pin cembro": "pin_à_crochets",
    "Douglas":                    "douglas",
    "Mélèze":                     "mélèze",
    "Pins mélangés":              "pin_sylvestre",
    "Pin autre":                  "pin_sylvestre",
    "Feuillus":                   "unknown",
    "Conifères":                  "unknown",
    "Mixte":                      "unknown",
    "NC":                         "unknown",
    "NR":                         "unknown",
}

# ═══════════════════════════════════════════════════════════════════
# PARSING TEXTE TFV — extraction espèces par mots-clés
# ═══════════════════════════════════════════════════════════════════

_TFV_TEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("buis", "buis"),
    ("chêne pubescent", "chêne_pubescent"),
    ("chene pubescent", "chêne_pubescent"),
    ("chêne tauzin", "chêne_pubescent"),
    ("chêne vert", "chêne_sessile"),
    ("chêne pédonculé", "chêne_sessile"),
    ("chênes décidus", "chêne_sessile"),
    ("chêne", "chêne_sessile"),
    ("chene", "chêne_sessile"),
    ("frêne", "frêne"),
    ("frene", "frêne"),
    ("orme", "orme"),
    ("noisetier", "noisetier"),
    ("noyer", "noyer"),
    ("merisier", "merisier"),
    ("tilleul", "tilleul"),
    ("hêtre", "hêtre"),
    ("hetre", "hêtre"),
    ("châtaignier", "châtaignier"),
    ("chataignier", "châtaignier"),
    ("charme", "charme"),
    ("érable", "érable_champêtre"),
    ("erable", "érable_champêtre"),
    ("aulne", "aulne"),
    ("bouleau", "bouleau"),
    ("tremble", "tremble"),
    ("robinier", "robinier"),
    ("acacia", "robinier"),
    ("peuplier", "peuplier"),
    ("peupleraie", "peuplier"),
    ("saule", "saule"),
    ("douglas", "douglas"),
    ("mélèze", "mélèze"),
    ("meleze", "mélèze"),
    ("pin sylvestre", "pin_sylvestre"),
    ("pin noir", "pin_noir"),
    ("pin laricio", "pin_noir"),
    ("pin à crochets", "pin_à_crochets"),
    ("pin cembro", "pin_à_crochets"),
    ("sapin pectiné", "sapin"),
    ("épicéa", "épicéa"),
    ("epicea", "épicéa"),
    ("sapin", "sapin"),
    ("pin", "pin_sylvestre"),
)

# Colonnes texte à scanner pour extraction espèces
_TEXT_COLUMNS: tuple[str, ...] = ("TFV", "TFVG11", "TFV_G11", "LIBELLE")

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES — TYPE FORESTIER (codes int8)
# ═══════════════════════════════════════════════════════════════════

_FT_UNKNOWN: int = 0
_FT_FEUILLUS: int = 1
_FT_CONIFERES: int = 2
_FT_MIXTE: int = 3
_FT_LANDE: int = 4

_FT_NAMES: tuple[str, ...] = (
    "unknown", "feuillus", "conifères", "mixte", "lande",
)

_LANDE_DEFAULT_SCORE: float = 0.10

# TFV_G11 libellé → code int
_FOREST_TYPE_MAP: dict[str, int] = {
    "Forêt fermée feuillus":            _FT_FEUILLUS,
    "Forêt ouverte feuillus":           _FT_FEUILLUS,
    "Peupleraie":                       _FT_FEUILLUS,
    "Forêt fermée conifères":           _FT_CONIFERES,
    "Forêt ouverte conifères":          _FT_CONIFERES,
    "Forêt fermée mixte":               _FT_MIXTE,
    "Forêt ouverte mixte":              _FT_MIXTE,
    "Forêt fermée sans couvert arboré": _FT_UNKNOWN,
    "Lande":                            _FT_LANDE,
    "Lande ligneuse":                   _FT_LANDE,
    "Formation herbacée":               _FT_UNKNOWN,
}

_DECIDUOUS: frozenset[str] = frozenset({
    "chêne_sessile", "chêne_pédonculé", "chêne_pubescent",
    "hêtre", "châtaignier", "frêne", "charme", "tilleul",
    "érable_champêtre", "érable_sycomore", "orme", "aulne",
    "bouleau", "merisier", "noisetier", "noyer", "robinier",
    "peuplier", "tremble", "saule",
})
_CONIFEROUS: frozenset[str] = frozenset({
    "sapin", "épicéa", "pin_sylvestre", "pin_noir",
    "pin_à_crochets", "pin_cembro", "mélèze", "douglas",
    "cèdre", "if",
})

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES — SUBSTRAT (codes int8 + mapping depuis géologie)
# ═══════════════════════════════════════════════════════════════════

_SUB_UNKNOWN: int = 0
_SUB_CALC_DRY: int = 1
_SUB_ALLUVIAL: int = 2
_SUB_MARLY: int = 3
_SUB_SILICEOUS: int = 4

_SUB_NAMES: tuple[str, ...] = (
    "unknown", "calc_dry", "alluvial", "marly", "siliceous",
)

# Mapping catégorie géologique canonique → code substrat
_GEOLOGY_TO_SUB: dict[str, int] = {
    "calcaire": _SUB_CALC_DRY,
    "dolomie": _SUB_CALC_DRY,
    "calcaire_lacustre": _SUB_CALC_DRY,
    "calcaire_recifal": _SUB_CALC_DRY,
    "calcaire_oolithique": _SUB_CALC_DRY,
    "calcaire_marneux": _SUB_MARLY,
    "marne": _SUB_MARLY,
    "molasse": _SUB_MARLY,
    "flysch": _SUB_MARLY,
    "alluvions": _SUB_ALLUVIAL,
    "alluvions_recentes": _SUB_ALLUVIAL,
    "alluvions_anciennes": _SUB_ALLUVIAL,
    "colluvions": _SUB_ALLUVIAL,
    "moraine": _SUB_ALLUVIAL,
    "glaciaire": _SUB_ALLUVIAL,
    "fluvioglaciaire": _SUB_ALLUVIAL,
    "eboulis": _SUB_ALLUVIAL,
    "tourbe": _SUB_ALLUVIAL,
    "granite": _SUB_SILICEOUS,
    "gneiss": _SUB_SILICEOUS,
    "schiste": _SUB_SILICEOUS,
    "siliceux": _SUB_SILICEOUS,
    "gres": _SUB_SILICEOUS,
    "micaschiste": _SUB_SILICEOUS,
    "quartzite": _SUB_SILICEOUS,
    "amphibolite": _SUB_SILICEOUS,
    "serpentinite": _SUB_SILICEOUS,
}

# Seuil de pente (°) pour reclassification calcaire sec → marneux
_CALC_DRY_SLOPE_THRESHOLD: float = 10.0

# Multiplicateurs espèce × substrat (feuillus collinéens).
# >1 = espèce favorisée, <1 = défavorisée. Absent = 1.0.
_SUBSTRATE_MODIFIERS: dict[str, dict[str, float]] = {
    "calc_dry": {
        "chêne_pubescent": 3.0, "buis": 2.5, "chêne_sessile": 1.5,
        "charme": 0.8, "frêne": 0.3, "orme": 0.2, "peuplier": 0.1,
        "aulne": 0.1, "robinier": 0.5, "noisetier": 1.2,
        "érable_champêtre": 1.3,
    },
    "alluvial": {
        "frêne": 3.0, "orme": 2.5, "peuplier": 2.0, "aulne": 2.0,
        "saule": 1.8, "noyer": 1.5, "chêne_sessile": 0.8,
        "chêne_pubescent": 0.2, "buis": 0.1, "robinier": 1.2,
    },
    "marly": {
        "frêne": 1.3, "chêne_sessile": 1.1, "hêtre": 1.2, "charme": 1.1,
    },
    "siliceous": {
        "châtaignier": 2.5, "bouleau": 2.0, "chêne_sessile": 1.2,
        "chêne_pubescent": 0.3, "frêne": 0.4, "buis": 0.1, "hêtre": 0.8,
    },
}

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES — RÉGIONS FORESTIÈRES IFN (rfifn250_l93.shp DEP=38)
# ═══════════════════════════════════════════════════════════════════

_REGD_TO_KEY: dict[str, str] = {
    "0": "oisans",
    "1": "bas_dauphine",
    "2": "chambaran",
    "3": "ile_cremieu",
    "4": "gresivaudan",
    "5": "bas_drac",
    "6": "trieves",
    "7": "belledonne",
    "8": "vercors",
    "9": "chartreuse",
    "A": "haut_diois",
}

_REGIONAL_PROPORTIONS: dict[str, dict[str, float]] = {
    "gresivaudan": {
        "chêne_sessile": 0.11, "châtaignier": 0.07, "hêtre": 0.09,
        "frêne": 0.04, "robinier": 0.17, "charme": 0.04,
        "érable_champêtre": 0.03, "merisier": 0.02,
        "épicéa": 0.25, "sapin": 0.03, "pin_sylvestre": 0.07,
        "douglas": 0.01,
    },
    "chartreuse": {
        "chêne_sessile": 0.04, "hêtre": 0.18, "frêne": 0.03,
        "charme": 0.02, "érable_champêtre": 0.02,
        "épicéa": 0.28, "sapin": 0.44,
    },
    "belledonne": {
        "chêne_sessile": 0.02, "châtaignier": 0.02, "hêtre": 0.09,
        "frêne": 0.01, "charme": 0.01,
        "épicéa": 0.58, "sapin": 0.20, "mélèze": 0.02,
    },
    "vercors": {
        "chêne_sessile": 0.04, "hêtre": 0.20, "frêne": 0.02,
        "charme": 0.02, "érable_champêtre": 0.02,
        "épicéa": 0.15, "sapin": 0.38, "pin_sylvestre": 0.06,
        "douglas": 0.01,
    },
    "bas_drac": {
        "chêne_sessile": 0.03, "châtaignier": 0.04, "hêtre": 0.12,
        "frêne": 0.02, "charme": 0.01,
        "épicéa": 0.16, "sapin": 0.04, "pin_sylvestre": 0.24,
        "mélèze": 0.03,
    },
    "oisans": {
        "chêne_sessile": 0.03, "hêtre": 0.02,
        "épicéa": 0.14, "pin_sylvestre": 0.06,
        "mélèze": 0.10, "sapin": 0.05,
    },
    "bas_dauphine": {
        "chêne_sessile": 0.27, "châtaignier": 0.11, "hêtre": 0.08,
        "frêne": 0.04, "robinier": 0.43, "charme": 0.05,
        "noyer": 0.01,
    },
    "chambaran": {
        "chêne_sessile": 0.34, "châtaignier": 0.16, "hêtre": 0.05,
        "frêne": 0.05, "robinier": 0.08, "charme": 0.06,
        "épicéa": 0.10, "sapin": 0.01, "pin_sylvestre": 0.04,
        "douglas": 0.05,
    },
    "ile_cremieu": {
        "chêne_sessile": 0.62, "châtaignier": 0.01, "hêtre": 0.02,
        "frêne": 0.06, "robinier": 0.17, "charme": 0.08,
        "épicéa": 0.01,
    },
    "trieves": {
        "chêne_sessile": 0.06, "hêtre": 0.09, "frêne": 0.03,
        "épicéa": 0.08, "sapin": 0.15, "pin_sylvestre": 0.45,
        "pin_noir": 0.05, "mélèze": 0.02,
    },
    "haut_diois": {
        "hêtre": 0.06, "chêne_sessile": 0.04,
        "épicéa": 0.07, "sapin": 0.50, "pin_sylvestre": 0.16,
        "pin_noir": 0.07, "mélèze": 0.03,
    },
    "_default": {
        "chêne_sessile": 0.15, "châtaignier": 0.12, "hêtre": 0.10,
        "frêne": 0.05, "robinier": 0.03, "charme": 0.04,
        "érable_champêtre": 0.02, "tilleul": 0.01,
        "épicéa": 0.19, "sapin": 0.15, "pin_sylvestre": 0.07,
        "mélèze": 0.01, "douglas": 0.01,
    },
}

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES — ALTITUDE
# ═══════════════════════════════════════════════════════════════════

_ALT_BANDS: tuple[tuple[str, int, int], ...] = (
    ("collineen",      0,   600),
    ("montagnard_inf", 600, 1000),
    ("montagnard_sup", 1000, 1500),
    ("subalpin",       1500, 1900),
    ("alpin",          1900, 9999),
)

_ALT_AFFINITY: dict[str, tuple[float, ...]] = {
    "chêne_sessile":       (1.8, 0.8, 0.1, 0.0, 0.0),
    "chêne_pubescent":     (1.8, 0.6, 0.0, 0.0, 0.0),
    "châtaignier":         (1.6, 1.0, 0.1, 0.0, 0.0),
    "hêtre":               (0.6, 1.5, 1.2, 0.3, 0.0),
    "frêne":               (1.8, 0.8, 0.1, 0.0, 0.0),
    "charme":              (1.6, 0.5, 0.0, 0.0, 0.0),
    "érable_champêtre":    (1.4, 0.8, 0.2, 0.0, 0.0),
    "tilleul":             (1.5, 0.5, 0.0, 0.0, 0.0),
    "robinier":            (1.5, 0.3, 0.0, 0.0, 0.0),
    "orme":                (1.8, 0.3, 0.0, 0.0, 0.0),
    "aulne":               (1.5, 0.8, 0.2, 0.0, 0.0),
    "bouleau":             (1.0, 1.0, 0.6, 0.2, 0.0),
    "merisier":            (1.4, 0.8, 0.1, 0.0, 0.0),
    "noisetier":           (1.4, 0.8, 0.2, 0.0, 0.0),
    "noyer":               (1.6, 0.4, 0.0, 0.0, 0.0),
    "peuplier":            (1.8, 0.2, 0.0, 0.0, 0.0),
    "sapin":               (0.0, 0.8, 1.6, 0.8, 0.0),
    "épicéa":              (0.1, 0.8, 1.5, 1.4, 0.2),
    "pin_sylvestre":       (1.0, 1.2, 0.8, 0.2, 0.0),
    "pin_noir":            (1.0, 1.0, 0.5, 0.0, 0.0),
    "pin_à_crochets":      (0.0, 0.0, 0.5, 1.5, 0.5),
    "mélèze":              (0.0, 0.3, 1.0, 1.5, 0.3),
    "douglas":             (0.5, 1.2, 0.8, 0.0, 0.0),
    "buis":                (1.5, 0.8, 0.1, 0.0, 0.0),
    "saule":               (1.4, 0.6, 0.1, 0.0, 0.0),
    "tremble":             (0.8, 1.0, 0.5, 0.1, 0.0),
}

# Centre heuristique Grenoble (L93)
_CX: float = 913_100.0
_CY: float = 6_458_800.0


# ═══════════════════════════════════════════════════════════════════
class SpeciesEnricher:
    """
    Enrichit les essences forestières via cascade A→B→C→D.

    Usage::

        enricher = SpeciesEnricher(
            bd_foret_path="data/FORMATION_VEGETALE.shp",
            regions_shp_path="data/rfifn250_l93.shp",
        )
        forest_gdf = enricher.load_bd_foret()
        grid.score_tree_species(forest_gdf)
        enricher.enrich_grid_scores(grid, forest_gdf=forest_gdf,
                                    geology_gdf=geology_gdf)
    """

    def __init__(
        self,
        bd_foret_path: str | Path | None = None,
        regions_shp_path: str | Path | None = None,
        observations_path: str | Path | None = None,
    ) -> None:
        self._bd_foret_path = Path(bd_foret_path) if bd_foret_path else None
        self._regions_shp_path = (
            Path(regions_shp_path) if regions_shp_path else None
        )
        self._observations: list[dict[str, Any]] = []
        if observations_path:
            self._load_observations(Path(observations_path))

        self._forest_type_grid: np.ndarray | None = None
        self._region_grid: np.ndarray | None = None
        self._region_key_map: dict[int, str] = {}
        self._substrate_grid: np.ndarray | None = None

    # ═══════════════════════════════════════════════════════════════
    # NIVEAU A — BD Forêt v2 comme source forêt
    # ═══════════════════════════════════════════════════════════════

    def load_bd_foret(self) -> gpd.GeoDataFrame | None:  # type: ignore[name-defined]
        """
        Charge BD Forêt v2, extrait essence_canonical + forest_type.

        Retourne un GeoDataFrame prêt pour grid.score_tree_species(),
        ou None si indisponible.
        """
        if not _HAS_GEO:
            logger.warning("geopandas indisponible — BD Forêt ignorée")
            return None

        path = self._bd_foret_path
        if path is None or not path.exists():
            logger.info("BD Forêt v2 non trouvée (%s)", path)
            return None

        import os

        os.environ["SHAPE_RESTORE_SHX"] = "YES"

        logger.info("Chargement BD Forêt v2 : %s", path.name)

        bbox_l93 = dict(config.BBOX)
        clip = box(
            bbox_l93["xmin"], bbox_l93["ymin"],
            bbox_l93["xmax"], bbox_l93["ymax"],
        )

        try:
            gdf = gpd.read_file(path, bbox=clip)
        except Exception as exc:
            logger.error("Erreur lecture BD Forêt : %s", exc)
            return None

        if gdf.empty:
            logger.warning("BD Forêt vide après clip bbox")
            return None

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=2154)
        elif gdf.crs.to_epsg() != 2154:
            gdf = gdf.to_crs(epsg=2154)

        logger.info("  %d polygones dans le bbox", len(gdf))

        # ── Parser essence ────────────────────────────────────
        gdf["essence_canonical"] = "unknown"
        gdf["forest_type"] = 0
        gdf["source"] = "bd_foret_v2"

        n_total = len(gdf)

        # ── Étape A1 : ESSENCE directe ──
        if "ESSENCE" in gdf.columns:
            gdf["essence_canonical"] = (
                gdf["ESSENCE"].map(_ESSENCE_MAP).fillna("unknown")
            )

        n_a1 = int((gdf["essence_canonical"] != "unknown").sum())
        logger.info(
            "  A1 ESSENCE directe : %d/%d (%.1f%%)",
            n_a1, n_total, n_a1 / max(n_total, 1) * 100,
        )

        # ── Étape A2 : CODE_TFV fallback ──
        if "CODE_TFV" in gdf.columns:
            mask_unk = gdf["essence_canonical"] == "unknown"
            if mask_unk.any():
                parsed = gdf.loc[mask_unk, "CODE_TFV"].apply(self._parse_tfv)
                gdf.loc[mask_unk, "essence_canonical"] = parsed.apply(
                    lambda x: x[0],
                )
                gdf.loc[mask_unk, "forest_type"] = parsed.apply(
                    lambda x: x[1],
                )

        n_a2 = int((gdf["essence_canonical"] != "unknown").sum())
        logger.info(
            "  A2 CODE_TFV        : +%d → %d/%d (%.1f%%)",
            n_a2 - n_a1, n_a2, n_total, n_a2 / max(n_total, 1) * 100,
        )

        # ── Étape A3 : parsing texte TFV ──
        n_before_a3 = n_a2
        for col_name in _TEXT_COLUMNS:
            if col_name not in gdf.columns:
                continue
            mask_unk = gdf["essence_canonical"] == "unknown"
            if not mask_unk.any():
                break
            extracted = gdf.loc[mask_unk, col_name].apply(
                self._extract_species_from_text,
            )
            resolved = extracted != "unknown"
            if resolved.any():
                resolved_idx = resolved[resolved].index
                gdf.loc[resolved_idx, "essence_canonical"] = (
                    extracted[resolved_idx]
                )
                n_new = int(resolved.sum())
                logger.info(
                    "  A3 texte %-10s: +%d espèces", col_name, n_new,
                )

        n_a3 = int((gdf["essence_canonical"] != "unknown").sum())
        if n_a3 > n_before_a3:
            logger.info(
                "  A3 total texte     : +%d → %d/%d (%.1f%%)",
                n_a3 - n_before_a3, n_a3, n_total,
                n_a3 / max(n_total, 1) * 100,
            )

        # ── TFV_G11 → forest_type (ne pas écraser si déjà > 0) ──
        if "TFV_G11" in gdf.columns:
            ft = gdf["TFV_G11"].map(_FOREST_TYPE_MAP).fillna(0).astype(int)
            gdf["forest_type"] = np.where(
                gdf["forest_type"] > 0, gdf["forest_type"], ft,
            )

        # ── Stats finales ──
        n_known = int((gdf["essence_canonical"] != "unknown").sum())
        n_lande = int((gdf["forest_type"] == _FT_LANDE).sum())
        logger.info(
            "  Essences résolues : %d/%d (%.1f%%)",
            n_known, n_total, n_known / max(n_total, 1) * 100,
        )
        if n_lande > 0:
            logger.info("  Lande/matorral    : %d polygones", n_lande)

        for ess, cnt in (
            gdf["essence_canonical"].value_counts().head(15).items()
        ):
            logger.info(
                "    %-25s : %5d (%5.1f%%)",
                ess, cnt, cnt / n_total * 100,
            )

        return gdf

    # ═══════════════════════════════════════════════════════════════
    # NIVEAUX B+C+D — Enrichissement grille (vectorisé v2.4.1)
    # ═══════════════════════════════════════════════════════════════

    def enrich_grid_scores(
        self,
        grid: Any,
        forest_gdf: Any = None,
        geology_gdf: Any = None,
    ) -> None:
        """
        Enrichit grid.scores["tree_species"] pour les cellules inconnues.

        Identifie les cellules unknown via le tree species int raster (v2.4),
        avec fallback sur comparaison float pour compatibilité.

        Appelé APRÈS score_tree_species(), AVANT apply_landcover_mask().
        """
        t0 = _time.perf_counter()

        ts = grid.scores.get("tree_species")
        if ts is None or not isinstance(ts, np.ndarray):
            logger.warning("Score tree_species absent — enrichissement ignoré")
            return

        tree_scores: np.ndarray = ts
        ny, nx = tree_scores.shape

        # ── Identifier les cellules unknown ────────────────────────
        int_raster = getattr(grid, "_tree_species_int_raster", None)
        code_to_name = getattr(grid, "_tree_code_to_name", None)

        enrichable: np.ndarray
        if int_raster is not None and code_to_name is not None:
            # v2.4 : identification via int raster
            int_raster = np.asarray(int_raster, dtype=np.int16)
            unknown_code: int | None = None
            for code, name in code_to_name.items():
                if name == "unknown":
                    unknown_code = code
                    break
            if unknown_code is None:
                logger.info(
                    "   Enrichissement : aucun code unknown → skip",
                )
                return
            # Unknown ET couvert forêt (code > 0)
            enrichable = (int_raster == unknown_code)
        else:
            # Fallback float (compatibilité pré-v2.4)
            fill_unknown = 0.25
            fill_no_forest = getattr(config, "FILL_NO_FOREST", 0.05)
            is_unknown = np.isclose(tree_scores, fill_unknown, atol=0.02)
            is_no_forest = np.isclose(
                tree_scores, fill_no_forest, atol=0.02,
            )
            enrichable = is_unknown & ~is_no_forest

        # ── Forest type grid → lande ──────────────────────────────
        ft_grid = self._build_forest_type_grid(grid, forest_gdf)
        grid._forest_type_grid = ft_grid

        is_lande = ft_grid == _FT_LANDE
        lande_enrichable = enrichable & is_lande
        n_lande = int(np.count_nonzero(lande_enrichable))
        if n_lande > 0:
            tree_scores[lande_enrichable] = np.float32(_LANDE_DEFAULT_SCORE)
            enrichable = enrichable & ~lande_enrichable
            logger.info(
                "  Lande : %d cellules → score %.2f (pas d'enrichissement)",
                n_lande, _LANDE_DEFAULT_SCORE,
            )

        n_enrich = int(np.count_nonzero(enrichable))
        if n_enrich == 0:
            logger.info("Enrichissement : aucune cellule unknown restante")
            return

        logger.info(
            "Enrichissement : %d cellules (%.1f%%)",
            n_enrich, n_enrich / max(tree_scores.size, 1) * 100,
        )

        altitude = getattr(grid, "altitude", None)
        if altitude is None or not isinstance(altitude, np.ndarray):
            logger.warning("  Altitude indisponible — enrichissement annulé")
            return
        _alt: np.ndarray = np.asarray(altitude, dtype=np.float32)

        x_coords = getattr(grid, "x_coords", None)
        y_coords = getattr(grid, "y_coords", None)

        # ── Grilles auxiliaires ────────────────────────────────────
        rg_grid = self._build_region_grid(
            x_coords, y_coords, _alt, ny, nx,
        )
        sub_grid = self._build_substrate_grid(grid)

        if sub_grid is not None:
            grid.substrate_grid = sub_grid
            n_sub = int(np.count_nonzero(sub_grid > 0))
            if n_sub > 0:
                parts = []
                for code in range(1, len(_SUB_NAMES)):
                    n = int(np.count_nonzero(sub_grid == code))
                    if n > 0:
                        parts.append(f"{_SUB_NAMES[code]}={n}")
                logger.info(
                    "  Substrat : %d cellules classées — %s",
                    n_sub, " ".join(parts),
                )

        # ── Niveau C : observations ────────────────────────────────
        n_obs = self._apply_observations(
            tree_scores, enrichable, x_coords, y_coords,
        )
        if n_obs > 0:
            logger.info("  Niveau C : %d cellules", n_obs)

        # ── Niveaux B+D — vectorisé par combo keys ────────────────
        enriched = self._compute_regional_scores(
            _alt, rg_grid, ft_grid, enrichable,
            substrate_grid=sub_grid,
        )

        n_final = int(np.count_nonzero(enrichable))
        tree_scores[enrichable] = enriched[enrichable]

        # Propager dans _raw_tree_species si présent
        raw = getattr(grid, "_raw_tree_species", None)
        if raw is not None and isinstance(raw, np.ndarray):
            raw[enrichable] = enriched[enrichable]

        mean_s = (
            float(enriched[enrichable].mean()) if n_final > 0 else 0.0
        )
        logger.info(
            "  Niveaux B+D : %d cellules, score moyen=%.3f",
            n_final, mean_s,
        )

        # ── Stats finales ──────────────────────────────────────────
        if int_raster is not None and code_to_name is not None:
            assert unknown_code is not None
            n_still_unknown = int(
                np.count_nonzero(
                    (int_raster == unknown_code) & enrichable,
                ),
            )
            n_total_forest = int(np.count_nonzero(int_raster > 0))
            n_known_final = n_total_forest - n_still_unknown
        else:
            fill_unknown = 0.25
            fill_no_forest = getattr(config, "FILL_NO_FOREST", 0.05)
            n_still_unknown = int(
                np.count_nonzero(
                    np.isclose(tree_scores, fill_unknown, atol=0.02),
                ),
            )
            n_no_forest = int(
                np.count_nonzero(
                    np.isclose(tree_scores, fill_no_forest, atol=0.02),
                ),
            )
            n_known_final = tree_scores.size - n_still_unknown - n_no_forest

        n_report_total = n_known_final + n_still_unknown
        logger.info(
            "   ▸ Essences : %d/%d connues (%.1f%%), "
            "%d inconnues restantes",
            n_known_final,
            n_report_total,
            n_known_final / max(n_report_total, 1) * 100,
            n_still_unknown,
        )

        dt = _time.perf_counter() - t0
        logger.info("   ⏱️  Enrichissement : %.1fs", dt)

    # ═══════════════════════════════════════════════════════════════
    # SCORES RÉGIONAUX — vectorisé par combo keys (v2.4.1)
    # ═══════════════════════════════════════════════════════════════

    def _compute_regional_scores(
        self,
        altitude: np.ndarray,
        region_grid: np.ndarray,
        forest_type_grid: np.ndarray,
        mask: np.ndarray,
        substrate_grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calcule les scores enrichis — vectorisé par combo keys.

        Encode (region, forest_type, altitude_band, substrate) en clé int32,
        itère uniquement sur les combinaisons présentes dans le masque.
        """
        result = np.full_like(altitude, 0.25, dtype=np.float32)

        # ── Bande altitudinale ─────────────────────────────────────
        alt_idx = np.zeros_like(altitude, dtype=np.int8)
        for i, (_name, a_min, a_max) in enumerate(_ALT_BANDS):
            alt_idx[(altitude >= a_min) & (altitude < a_max)] = i

        has_substrate = (
            substrate_grid is not None
            and np.any(substrate_grid > 0)
        )

        # ── Pré-calcul lookup (region_key, ft, band, sub) → score ─
        lookup: dict[tuple[str, int, int, int], float] = {}
        all_keys = set(self._region_key_map.values()) | {"_default"}

        for rk in all_keys:
            props = _REGIONAL_PROPORTIONS.get(
                rk, _REGIONAL_PROPORTIONS["_default"],
            )
            for ft in range(_FT_LANDE):  # 0..3
                filtered = self._filter_by_forest_type(props, ft)
                for bi in range(len(_ALT_BANDS)):
                    lookup[(rk, ft, bi, 0)] = self._weighted_morel_score(
                        filtered, bi,
                    )
                    if has_substrate:
                        for sub_code in range(1, len(_SUB_NAMES)):
                            sub_key = _SUB_NAMES[sub_code]
                            modulated = self._apply_substrate_modifiers(
                                filtered, sub_key,
                            )
                            lookup[(rk, ft, bi, sub_code)] = (
                                self._weighted_morel_score(modulated, bi)
                            )

        # ── Combo key int32 : region*1000 + ft*100 + band*10 + sub ─
        region_i32 = np.asarray(region_grid, dtype=np.int32)
        ft_i32 = np.asarray(forest_type_grid, dtype=np.int32)
        alt_i32 = np.asarray(alt_idx, dtype=np.int32)
        sub_i32 = (
            np.asarray(substrate_grid, dtype=np.int32)
            if has_substrate
            else np.zeros_like(region_i32)
        )

        combo = region_i32 * 1000 + ft_i32 * 100 + alt_i32 * 10 + sub_i32

        # ── Unique keys sur le masque uniquement ───────────────────
        keys_in_mask = combo[mask]
        unique_keys = np.unique(keys_in_mask)

        for key in unique_keys:
            r_int = int(key // 1000)
            ft = int((key % 1000) // 100)
            bi = int((key % 100) // 10)
            sv = int(key % 10)

            rk = self._region_key_map.get(r_int, "_default")

            score = lookup.get(
                (rk, ft, bi, sv),
                lookup.get(
                    (rk, ft, bi, 0),
                    lookup.get(("_default", ft, bi, 0), 0.25),
                ),
            )

            group_mask = mask & (combo == key)
            result[group_mask] = np.float32(score)

        # ── Fallback : cellules encore à 0.25 ─────────────────────
        still = mask & np.isclose(result, 0.25, atol=0.01)
        if np.any(still):
            for key in np.unique(combo[still]):
                ft = int((key % 1000) // 100)
                bi = int((key % 100) // 10)
                s = lookup.get(("_default", ft, bi, 0), 0.25)
                group_mask = still & (combo == key)
                result[group_mask] = np.float32(s)

        return result

    @staticmethod
    def _apply_substrate_modifiers(
        proportions: dict[str, float],
        substrate_key: str | None,
    ) -> dict[str, float]:
        """Applique les multiplicateurs substrat aux proportions espèces."""
        if substrate_key is None:
            return proportions

        modifiers = _SUBSTRATE_MODIFIERS.get(substrate_key)
        if modifiers is None:
            return proportions

        modulated: dict[str, float] = {}
        total = 0.0
        for sp, p in proportions.items():
            m = modifiers.get(sp, 1.0)
            new_p = p * m
            modulated[sp] = new_p
            total += new_p

        if total < 1e-10:
            return proportions

        return {sp: p / total for sp, p in modulated.items()}

    @staticmethod
    def _filter_by_forest_type(
        proportions: dict[str, float],
        forest_type: int,
    ) -> dict[str, float]:
        """Filtre par type (1=feuillus, 2=conifères, 3/0=tout)."""
        if forest_type in (_FT_UNKNOWN, _FT_MIXTE):
            return proportions

        target = _DECIDUOUS if forest_type == _FT_FEUILLUS else _CONIFEROUS

        filtered: dict[str, float] = {}
        total = 0.0
        for sp, p in proportions.items():
            if sp in target:
                filtered[sp] = p
                total += p

        if total < 1e-10:
            return proportions

        return {sp: p / total for sp, p in filtered.items()}

    @staticmethod
    def _weighted_morel_score(
        proportions: dict[str, float],
        band_idx: int,
    ) -> float:
        """Σ(proportion × affinité × score_morille) / Σ(proportion × affinité)."""
        w_sum = 0.0
        w_total = 0.0

        for species, proportion in proportions.items():
            if proportion <= 0:
                continue

            score = config.get_tree_score(species)
            if score < 0:
                score = 0.25

            aff = _ALT_AFFINITY.get(species)
            affinity = aff[band_idx] if aff and band_idx < len(aff) else 1.0

            w = proportion * affinity
            w_sum += w * score
            w_total += w

        if w_total < 1e-10:
            return 0.25

        return float(np.clip(w_sum / w_total, 0.05, 1.0))

    # ═══════════════════════════════════════════════════════════════
    # GRILLE SUBSTRAT — depuis geology int raster (zéro rasterisation)
    # ═══════════════════════════════════════════════════════════════

    def _build_substrate_grid(self, grid: Any) -> np.ndarray | None:
        """
        Construit la grille substrat depuis le geology int raster.

        Dérive de grid._geology_int_raster + _geology_code_to_name
        via np.take (zéro rasterisation). Applique correction pente :
        calcaire plat (<10°) → marneux.

        Returns
        -------
        np.ndarray int8, shape (ny, nx), codes _SUB_*
        """
        if self._substrate_grid is not None:
            return self._substrate_grid

        int_raster = getattr(grid, "_geology_int_raster", None)
        code_to_name = getattr(grid, "_geology_code_to_name", None)

        if int_raster is None or code_to_name is None:
            logger.info("  Substrat : pas de geology int raster → skip")
            return None

        int_raster = np.asarray(int_raster, dtype=np.int16)

        # ── Lookup table : geology code → substrate code ───────────
        max_code = int(int_raster.max()) + 1
        geo_to_sub = np.full(max_code, _SUB_UNKNOWN, dtype=np.int8)

        for code, geo_name in code_to_name.items():
            if 0 < code < max_code:
                geo_to_sub[code] = _GEOLOGY_TO_SUB.get(
                    geo_name, _SUB_UNKNOWN,
                )

        # ── Vectorized lookup ──────────────────────────────────────
        safe_codes = np.clip(int_raster, 0, max_code - 1)
        substrate = np.take(geo_to_sub, safe_codes).astype(np.int8)

        # ── Correction pente : calcaire plat → marneux ─────────────
        slope = getattr(grid, "slope", None)
        if slope is not None:
            slope = np.asarray(slope, dtype=np.float32)
            flat_calc = (
                (substrate == _SUB_CALC_DRY)
                & (slope < _CALC_DRY_SLOPE_THRESHOLD)
            )
            n_reclass = int(np.count_nonzero(flat_calc))
            if n_reclass > 0:
                substrate[flat_calc] = _SUB_MARLY
                logger.info(
                    "  Substrat : %d cellules calcaire plat → marly "
                    "(pente<%.0f°)",
                    n_reclass, _CALC_DRY_SLOPE_THRESHOLD,
                )

        self._substrate_grid = substrate
        return substrate

    # ═══════════════════════════════════════════════════════════════
    # GRILLE TYPE FORÊT — parallel_rasterize_categorical + cache
    # ═══════════════════════════════════════════════════════════════

    def _build_forest_type_grid(
        self,
        grid: Any,
        forest_gdf: Any,
    ) -> np.ndarray:
        """
        Construit la grille type forestier via parallel_rasterize_categorical.

        Rasterise la colonne forest_type (déjà remplie par load_bd_foret)
        avec cache disque.
        """
        if self._forest_type_grid is not None:
            return self._forest_type_grid

        ny: int = grid.ny
        nx: int = grid.nx
        out_shape = (ny, nx)
        ft_grid = np.zeros(out_shape, dtype=np.int8)

        if (
            forest_gdf is None
            or not _HAS_GEO
            or "forest_type" not in getattr(forest_gdf, "columns", [])
        ):
            self._forest_type_grid = ft_grid
            return ft_grid

        try:
            # ── Filtrage ───────────────────────────────────────────
            valid_mask = (
                forest_gdf.geometry.notna()
                & forest_gdf.geometry.is_valid
                & (~forest_gdf.geometry.is_empty)
                & (forest_gdf["forest_type"] != 0)
            )
            gdf = forest_gdf.loc[valid_mask].copy()

            if gdf.empty:
                self._forest_type_grid = ft_grid
                return ft_grid

            if gdf.crs is not None and gdf.crs.to_epsg() != 2154:
                gdf = gdf.to_crs(epsg=2154)

            codes = np.asarray(
                gdf["forest_type"].values, dtype=np.int16,
            )
            geometries: list[Any] = gdf.geometry.tolist()

            # ── Cache ──────────────────────────────────────────────
            codes_hash = hashlib.md5(
                codes.tobytes(), usedforsecurity=False,
            ).hexdigest()[:8]
            cache_path = _accel.raster_cache_path(
                "forest_type",
                f"ft_{codes_hash}",
                len(geometries),
                grid.cell_size,
                out_shape,
            )
            cached = _accel.raster_cache_load(cache_path)

            if cached is not None:
                ft_grid = np.asarray(cached, dtype=np.int8)
                logger.info(
                    "✅ Forest type : cache disque (%s)",
                    cache_path.name,
                )
            else:
                int_raster = _accel.parallel_rasterize_categorical(
                    geometries=geometries,
                    category_codes=codes,
                    out_shape=out_shape,
                    transform=grid.transform,
                    all_touched=True,
                    nodata=0,
                )
                ft_grid = np.asarray(int_raster, dtype=np.int8)
                _accel.raster_cache_save(cache_path, ft_grid)

            parts = []
            for code, name in enumerate(_FT_NAMES):
                if code == 0:
                    continue
                n = int(np.count_nonzero(ft_grid == code))
                if n > 0:
                    parts.append(f"{name}={n}")
            logger.info("  Forest type : %s", ", ".join(parts))

        except Exception as exc:
            logger.warning("  Rasterisation forest_type : %s", exc)

        self._forest_type_grid = ft_grid
        return ft_grid

    # ═══════════════════════════════════════════════════════════════
    # GRILLE RÉGIONS FORESTIÈRES
    # ═══════════════════════════════════════════════════════════════

    def _build_region_grid(
        self,
        x_coords: Any,
        y_coords: Any,
        altitude: np.ndarray,
        ny: int,
        nx: int,
    ) -> np.ndarray:
        """Construit la grille régions IFN (shapefile ou heuristique)."""
        if self._region_grid is not None:
            return self._region_grid

        rg = self._rasterize_regions(ny, nx)
        if rg is not None:
            self._region_grid = rg
            return rg

        logger.info("  Régions : heuristique Grenoble")
        rg = self._heuristic_regions(x_coords, y_coords, altitude, ny, nx)
        self._region_grid = rg
        return rg

    def _rasterize_regions(self, ny: int, nx: int) -> np.ndarray | None:
        """Rasterise les régions IFN depuis le shapefile rfifn250_l93."""
        if not _HAS_GEO or self._regions_shp_path is None:
            return None
        if not self._regions_shp_path.exists():
            return None

        import os

        os.environ["SHAPE_RESTORE_SHX"] = "YES"

        try:
            from rasterio.features import rasterize  # type: ignore[import-untyped]
            from rasterio.transform import from_bounds  # type: ignore[import-untyped]

            rf = gpd.read_file(self._regions_shp_path)

            rf["DEP"] = (
                rf["DEP"].astype(str).str.strip().str.split(".").str[0]
            )
            rf38 = rf[rf["DEP"] == "38"].copy()

            if rf38.empty:
                logger.warning("  Aucune région DEP=38")
                return None

            if rf38.crs is None:
                rf38 = rf38.set_crs(epsg=2154)
            elif rf38.crs.to_epsg() != 2154:
                rf38 = rf38.to_crs(epsg=2154)

            bbox = dict(config.BBOX)
            clip = box(
                bbox["xmin"], bbox["ymin"],
                bbox["xmax"], bbox["ymax"],
            )
            rf38 = rf38[rf38.intersects(clip)]

            if rf38.empty:
                logger.warning("  Aucune région intersecte bbox")
                return None

            next_id = 1
            regd_to_int: dict[str, int] = {}

            for regd_val in rf38["REGD"].unique():
                regd_str = str(regd_val).strip()
                reg_key = _REGD_TO_KEY.get(regd_str, "_default")
                regd_to_int[regd_str] = next_id
                self._region_key_map[next_id] = reg_key
                _sub_df = rf38[rf38["REGD"] == regd_val]
                regiond = (
                    str(_sub_df["REGIOND"].iloc[0])
                    if len(_sub_df) > 0
                    else "?"
                )
                logger.info(
                    "    REGD=%s → %s (id=%d)  '%s'",
                    regd_str, reg_key, next_id, regiond,
                )
                next_id += 1

            self._region_key_map[0] = "_default"

            shapes = []
            for _, row in rf38.iterrows():
                rid = regd_to_int.get(str(row["REGD"]).strip(), 0)
                if rid > 0:
                    shapes.append((row.geometry, rid))

            transform = from_bounds(
                bbox["xmin"], bbox["ymin"],
                bbox["xmax"], bbox["ymax"],
                nx, ny,
            )

            result = np.asarray(
                rasterize(
                    shapes,
                    out_shape=(ny, nx),
                    transform=transform,
                    fill=0,
                    dtype="int16",
                ),
            )

            n_mapped = int(np.count_nonzero(result > 0))
            logger.info(
                "  Régions rasterisées : %d/%d cellules (%.1f%%)",
                n_mapped, result.size,
                n_mapped / max(result.size, 1) * 100,
            )

            return result

        except Exception as exc:
            logger.warning("  Rasterisation régions : %s", exc)
            return None

    def _heuristic_regions(
        self,
        x_coords: Any,
        y_coords: Any,
        altitude: np.ndarray,
        ny: int,
        nx: int,
    ) -> np.ndarray:
        """Fallback : régions heuristiques centrées sur Grenoble."""
        self._region_key_map.update({
            10: "gresivaudan",
            11: "chartreuse",
            12: "belledonne",
            13: "vercors",
            14: "bas_drac",
            0: "_default",
        })

        region = np.full((ny, nx), 10, dtype=np.int16)

        if x_coords is None or y_coords is None:
            return region

        if isinstance(x_coords, np.ndarray) and x_coords.ndim == 1:
            xx, yy = np.meshgrid(x_coords, y_coords)
        else:
            xx = np.broadcast_to(np.asarray(x_coords), (ny, nx))
            yy = np.broadcast_to(np.asarray(y_coords), (ny, nx))

        dx, dy = xx - _CX, yy - _CY
        alt_ok = np.isfinite(altitude)

        region[(dy > 2000) & alt_ok & (altitude > 400)] = 11
        region[(dx > 3000) & alt_ok & (altitude > 600)] = 12
        region[(dx < -3000) & alt_ok & (altitude > 500)] = 13
        region[(dy < -3000)] = 14

        for rid, rk in self._region_key_map.items():
            if rid >= 10:
                cnt = int(np.count_nonzero(region == rid))
                if cnt > 0:
                    logger.info("    %s : %d cellules", rk, cnt)

        return region

    # ═══════════════════════════════════════════════════════════════
    # OBSERVATIONS TERRAIN (NIVEAU C)
    # ═══════════════════════════════════════════════════════════════

    def _load_observations(self, path: Path) -> None:
        """Charge les observations terrain depuis un fichier JSON."""
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._observations = data
                logger.info("  %d observations terrain chargées", len(data))
        except Exception as exc:
            logger.warning("Erreur observations : %s", exc)

    def _apply_observations(
        self,
        tree_scores: np.ndarray,
        enrichable: np.ndarray,
        x_coords: Any,
        y_coords: Any,
    ) -> int:
        """Applique les overrides d'observations terrain sur les scores."""
        if not self._observations or x_coords is None or y_coords is None:
            return 0

        ny, nx = tree_scores.shape
        if isinstance(x_coords, np.ndarray) and x_coords.ndim == 1:
            xx, yy = np.meshgrid(x_coords, y_coords)
        else:
            xx, yy = np.asarray(x_coords), np.asarray(y_coords)

        count = 0
        for obs in self._observations:
            zone = (
                enrichable
                & (xx >= obs.get("xmin", 0))
                & (xx <= obs.get("xmax", 0))
                & (yy >= obs.get("ymin", 0))
                & (yy <= obs.get("ymax", 0))
            )
            if not np.any(zone):
                continue

            override = obs.get("score_override")
            if override is not None:
                score = float(override)
            else:
                essences = obs.get("essences", {})
                if not essences:
                    continue
                score = sum(
                    p * max(config.get_tree_score(s), 0.0)
                    for s, p in essences.items()
                )

            tree_scores[zone] = np.float32(score)
            enrichable[zone] = False
            n = int(np.count_nonzero(zone))
            count += n
            logger.info(
                "    Obs '%s' : %d cells → %.3f",
                obs.get("nom", "?"), n, score,
            )

        return count

    # ═══════════════════════════════════════════════════════════════
    # PARSEUR TFV — code structuré + texte libre
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_tfv(tfv_code: Any) -> tuple[str, int]:
        """
        Parse CODE_TFV → (essence, forest_type).

        Gère FF/FO (forêts), FP (peupleraie), LA (lande), FH (herbacée).

        Exemples réels Isère :
          FF1-00     → (unknown, 1)      FF1-09-09  → (hêtre, 1)
          FF2G61-61  → (sapin, 2)        FF31       → (unknown, 3)
          LA4        → (unknown, 4)      LA6-04     → (chêne_pubescent, 4)
          FP         → (peuplier, 1)     FH         → (unknown, 0)
        """
        if not isinstance(tfv_code, str) or len(tfv_code) < 2:
            return ("unknown", _FT_UNKNOWN)

        code = tfv_code.strip().upper()

        # Formations herbacées
        if code.startswith("FH"):
            return ("unknown", _FT_UNKNOWN)

        # Peupleraie
        if code.startswith("FP"):
            return ("peuplier", _FT_FEUILLUS)

        # Lande / matorral — tenter extraction espèce
        is_lande = code.startswith("LA")

        # Type forêt depuis préfixe
        forest_type = _FT_LANDE if is_lande else _FT_UNKNOWN
        if not is_lande and code.startswith(("FF", "FO")) and len(code) > 2:
            c = code[2]
            if c in ("1", "0"):
                forest_type = _FT_FEUILLUS
            elif c == "2":
                forest_type = _FT_CONIFERES
            elif c == "3":
                forest_type = _FT_MIXTE

        # Extraire codes espèces après le préfixe
        clean = code.replace("G", "-")
        parts = clean.split("-")

        for part in parts[1:] if not is_lande else parts:
            digits = "".join(c for c in part if c.isdigit())
            if len(digits) >= 2:
                sp_code = digits[:2]
                if sp_code != "00":
                    species = _TFV_SPECIES.get(sp_code, "unknown")
                    if species != "unknown":
                        return (species, forest_type)

        return ("unknown", forest_type)

    @staticmethod
    def _extract_species_from_text(text: Any) -> str:
        """
        Extrait une essence canonique depuis un texte libre TFV.

        Scan séquentiel de _TFV_TEXT_PATTERNS (ordonné spécifique→générique).
        Retourne "unknown" si aucun pattern ne matche.
        """
        if not isinstance(text, str) or len(text) < 3:
            return "unknown"

        lower = text.lower()

        for pattern, species in _TFV_TEXT_PATTERNS:
            if pattern in lower:
                return species

        return "unknown"

    # ═══════════════════════════════════════════════════════════════
    # STATISTIQUES
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self, grid: Any) -> dict[str, Any]:
        """Retourne les statistiques d'enrichissement."""
        ts = grid.scores.get("tree_species")
        if ts is None or not isinstance(ts, np.ndarray):
            return {"status": "no_tree_scores"}

        total = int(ts.size)

        # ── Détection via int raster si disponible ─────────────────
        int_raster = getattr(grid, "_tree_species_int_raster", None)
        code_to_name = getattr(grid, "_tree_code_to_name", None)

        if int_raster is not None and code_to_name is not None:
            int_raster = np.asarray(int_raster, dtype=np.int16)
            n_forest = int(np.count_nonzero(int_raster > 0))
            unknown_code: int | None = None
            for code, name in code_to_name.items():
                if name == "unknown":
                    unknown_code = code
                    break
            n_unk = (
                int(np.count_nonzero(int_raster == unknown_code))
                if unknown_code is not None
                else 0
            )
            n_nf = total - n_forest
            n_known = n_forest - n_unk
        else:
            n_nf = int(np.count_nonzero(np.isclose(ts, 0.05, atol=0.02)))
            n_unk = int(np.count_nonzero(np.isclose(ts, 0.25, atol=0.02)))
            n_known = total - n_nf - n_unk

        ft = self._forest_type_grid
        n_lande = (
            int(np.count_nonzero(ft == _FT_LANDE))
            if ft is not None
            else 0
        )

        sub = self._substrate_grid
        sub_stats: dict[str, int] = {}
        if sub is not None:
            for code in range(1, len(_SUB_NAMES)):
                n = int(np.count_nonzero(sub == code))
                if n > 0:
                    sub_stats[_SUB_NAMES[code]] = n

        return {
            "total_cells": total,
            "forest_cells": total - n_nf,
            "species_known": n_known,
            "species_unknown": n_unk,
            "no_forest": n_nf,
            "lande_cells": n_lande,
            "substrate": sub_stats,
            "pct_known": round(
                n_known / max(total - n_nf, 1) * 100, 1,
            ),
            "observations": len(self._observations),
        }