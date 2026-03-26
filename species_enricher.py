"""
species_enricher.py — Enrichissement essences forestières inconnues.

Cascade à 4 niveaux :
  A. BD Forêt v2 ESSENCE/CODE_TFV → espèce directe (~23%)
  B. Statistiques régionales IFN 1997 × type forêt × altitude (~77%)
  C. Observations terrain utilisateur (JSON)
  D. Modèle altitude-only (fallback ultime)

Calibré sur :
  - BD Forêt v2 Isère (FORMATION_VEGETALE.shp, 20 191 polygones)
  - Régions forestières IFN (rfifn250_l93.shp, DEP=38, 11 régions)
  - IFN Isère 3ème inventaire 1997

v2.3.1 — Création initiale, mapping REGD calibré
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

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
# Vérifié contre la distribution réelle du jeu Isère
_TFV_SPECIES: dict[str, str] = {
    "00": "unknown",
    "01": "chêne_sessile",       # FF1G01-01 → "Chênes décidus"
    "09": "hêtre",               # FF1-09-09 → "Hêtre"
    "10": "châtaignier",         # FF1-10-10 → "Châtaignier"
    "14": "robinier",            # FF1-14-14 → "Robinier"
    "49": "unknown",             # FF1-49-49 → "Feuillus" (indéterminé)
    "52": "pin_sylvestre",       # FF2-52-52 → "Pin sylvestre"
    "53": "pin_noir",            # FF2G53-53 → "Pin laricio, pin noir"
    "58": "pin_à_crochets",      # FF2G58-58 → "Pin à crochets, pin cembro"
    "61": "sapin",               # FF2G61-61 → "Sapin, épicéa"
    "63": "mélèze",              # FF2-63-63 → "Mélèze"
    "64": "douglas",             # FF2-64-64 → "Douglas"
    "80": "pin_sylvestre",       # FF2-80-80 → "Pins mélangés"
    "81": "pin_sylvestre",       # FF2-81-81 → "Pin autre"
    "90": "unknown",             # FF2-90-90 → autres feuillus
    "91": "unknown",             # FF2-91-91
}

# ESSENCE (texte exact BD Forêt v2 Isère) → canonical
_ESSENCE_MAP: dict[str, str] = {
    "Châtaignier":                "châtaignier",
    "Chênes décidus":             "chêne_sessile",
    "Hêtre":                      "hêtre",
    "Robinier":                   "robinier",
    "Sapin, épicéa":              "sapin",
    "Pin sylvestre":              "pin_sylvestre",
    "Pin laricio, pin noir":      "pin_noir",
    "Pin à crochets, pin cembro": "pin_à_crochets",
    "Douglas":                    "douglas",
    "Mélèze":                     "mélèze",
    "Peuplier":                   "peuplier",
    "Pins mélangés":              "pin_sylvestre",
    "Pin autre":                  "pin_sylvestre",
    # Catégories génériques → unknown → enrichissement B-D
    "Feuillus":                   "unknown",
    "Conifères":                  "unknown",
    "Mixte":                      "unknown",
    "NC":                         "unknown",
    "NR":                         "unknown",
}

# TFV_G11 → type de forêt (1=feuillus, 2=conifères, 3=mixte, 0=non-forêt)
_FOREST_TYPE_MAP: dict[str, int] = {
    "Forêt fermée feuillus":               1,
    "Forêt ouverte feuillus":              1,
    "Peupleraie":                          1,
    "Forêt fermée conifères":              2,
    "Forêt ouverte conifères":             2,
    "Forêt fermée mixte":                  3,
    "Forêt ouverte mixte":                 3,
    "Forêt fermée sans couvert arboré":    0,
    "Lande":                               0,
    "Formation herbacée":                  0,
}

# Classification feuillus / conifères
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
# CONSTANTES — RÉGIONS FORESTIÈRES IFN (rfifn250_l93.shp DEP=38)
# ═══════════════════════════════════════════════════════════════════

# Mapping exact REGD → clé interne
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

# Proportions surfaciques IFN 1997 Isère (ha → normalisées)
# Source : IFN Isère, 3ème inventaire 1997, tableau 5.1
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
    # Fallback département
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

#                           collin  mont_inf  mont_sup  subalp  alpin
_ALT_AFFINITY: dict[str, tuple[float, ...]] = {
    "chêne_sessile":       (1.8,    0.8,      0.1,      0.0,    0.0),
    "chêne_pubescent":     (1.8,    0.6,      0.0,      0.0,    0.0),
    "châtaignier":         (1.6,    1.0,      0.1,      0.0,    0.0),
    "hêtre":               (0.6,    1.5,      1.2,      0.3,    0.0),
    "frêne":               (1.8,    0.8,      0.1,      0.0,    0.0),
    "charme":              (1.6,    0.5,      0.0,      0.0,    0.0),
    "érable_champêtre":    (1.4,    0.8,      0.2,      0.0,    0.0),
    "tilleul":             (1.5,    0.5,      0.0,      0.0,    0.0),
    "robinier":            (1.5,    0.3,      0.0,      0.0,    0.0),
    "orme":                (1.8,    0.3,      0.0,      0.0,    0.0),
    "aulne":               (1.5,    0.8,      0.2,      0.0,    0.0),
    "bouleau":             (1.0,    1.0,      0.6,      0.2,    0.0),
    "merisier":            (1.4,    0.8,      0.1,      0.0,    0.0),
    "noisetier":           (1.4,    0.8,      0.2,      0.0,    0.0),
    "noyer":               (1.6,    0.4,      0.0,      0.0,    0.0),
    "peuplier":            (1.8,    0.2,      0.0,      0.0,    0.0),
    "sapin":               (0.0,    0.8,      1.6,      0.8,    0.0),
    "épicéa":              (0.1,    0.8,      1.5,      1.4,    0.2),
    "pin_sylvestre":       (1.0,    1.2,      0.8,      0.2,    0.0),
    "pin_noir":            (1.0,    1.0,      0.5,      0.0,    0.0),
    "pin_à_crochets":      (0.0,    0.0,      0.5,      1.5,    0.5),
    "mélèze":              (0.0,    0.3,      1.0,      1.5,    0.3),
    "douglas":             (0.5,    1.2,      0.8,      0.0,    0.0),
}

# Coordonnées centre Grenoble L93
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
        enricher.enrich_grid_scores(grid, forest_gdf=forest_gdf)
    """

    def __init__(
        self,
        bd_foret_path: str | Path | None = None,
        regions_shp_path: str | Path | None = None,
        observations_path: str | Path | None = None,
    ) -> None:
        self._bd_foret_path = Path(bd_foret_path) if bd_foret_path else None
        self._regions_shp_path = Path(regions_shp_path) if regions_shp_path else None
        self._observations: list[dict[str, Any]] = []
        if observations_path:
            self._load_observations(Path(observations_path))

        self._forest_type_grid: np.ndarray | None = None
        self._region_grid: np.ndarray | None = None
        self._region_key_map: dict[int, str] = {}

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

        # 1) ESSENCE directe
        if "ESSENCE" in gdf.columns:
            gdf["essence_canonical"] = (
                gdf["ESSENCE"].map(_ESSENCE_MAP).fillna("unknown")
            )

        # 2) CODE_TFV fallback pour les unknown
        if "CODE_TFV" in gdf.columns:
            mask_unk = gdf["essence_canonical"] == "unknown"
            if mask_unk.any():
                parsed = gdf.loc[mask_unk, "CODE_TFV"].apply(self._parse_tfv)
                gdf.loc[mask_unk, "essence_canonical"] = parsed.apply(
                    lambda x: x[0]
                )
                # forest_type depuis CODE_TFV
                gdf.loc[mask_unk, "forest_type"] = parsed.apply(
                    lambda x: x[1]
                )

        # 3) TFV_G11 → forest_type (ne pas écraser si déjà > 0)
        if "TFV_G11" in gdf.columns:
            ft = gdf["TFV_G11"].map(_FOREST_TYPE_MAP).fillna(0).astype(int)
            gdf["forest_type"] = np.where(
                gdf["forest_type"] > 0, gdf["forest_type"], ft,
            )

        # Stats
        n_known = int((gdf["essence_canonical"] != "unknown").sum())
        n_total = len(gdf)
        logger.info(
            "  Essences résolues : %d/%d (%.1f%%)",
            n_known, n_total, n_known / max(n_total, 1) * 100,
        )
        for ess, cnt in gdf["essence_canonical"].value_counts().head(15).items():
            logger.info(
                "    %-25s : %5d (%5.1f%%)", ess, cnt, cnt / n_total * 100,
            )

        return gdf

    # ═══════════════════════════════════════════════════════════════
    # NIVEAUX B+C+D — Enrichissement grille
    # ═══════════════════════════════════════════════════════════════

    def enrich_grid_scores(
        self,
        grid: Any,
        forest_gdf: Any = None,
    ) -> None:
        """
        Enrichit grid.scores["tree_species"] pour les cellules inconnues.

        Appelé APRÈS score_tree_species(), AVANT apply_landcover_mask().
        """
        ts = grid.scores.get("tree_species")
        if ts is None or not isinstance(ts, np.ndarray):
            logger.warning("Score tree_species absent — enrichissement ignoré")
            return

        tree_scores: np.ndarray = ts
        fill_unknown = 0.25
        fill_no_forest = getattr(config, "FILL_NO_FOREST", 0.05)

        is_unknown = np.isclose(tree_scores, fill_unknown, atol=0.02)
        is_no_forest = np.isclose(tree_scores, fill_no_forest, atol=0.02)
        enrichable = is_unknown & ~is_no_forest

        n_enrich = int(enrichable.sum())
        if n_enrich == 0:
            logger.info("Enrichissement : aucune cellule unknown")
            return

        logger.info(
            "Enrichissement : %d cellules (%.1f%%)",
            n_enrich, n_enrich / max(tree_scores.size, 1) * 100,
        )

        altitude = getattr(grid, "altitude", None)
        x_coords = getattr(grid, "x_coords", None)
        y_coords = getattr(grid, "y_coords", None)

        if altitude is None or not isinstance(altitude, np.ndarray):
            logger.warning("  Altitude indisponible — enrichissement annulé")
            return
        _alt: np.ndarray = altitude
        ny, nx = _alt.shape

        # Grilles auxiliaires
        ft_grid = self._build_forest_type_grid(forest_gdf, ny, nx)
        rg_grid = self._build_region_grid(x_coords, y_coords, _alt, ny, nx)

        # Niveau C : observations
        n_obs = self._apply_observations(
            tree_scores, enrichable, x_coords, y_coords,
        )
        if n_obs > 0:
            logger.info("  Niveau C : %d cellules", n_obs)
            enrichable = enrichable & np.isclose(
                tree_scores, fill_unknown, atol=0.02,
            )

        # Niveaux B+D
        enriched = self._compute_regional_scores(
            _alt, rg_grid, ft_grid, enrichable,
        )

        n_final = int(enrichable.sum())
        tree_scores[enrichable] = enriched[enrichable]

        # Propager dans _raw_tree_species
        raw = getattr(grid, "_raw_tree_species", None)
        if raw is not None and isinstance(raw, np.ndarray):
            raw[enrichable] = enriched[enrichable]

        mean_s = float(enriched[enrichable].mean()) if n_final > 0 else 0.0
        logger.info(
            "  Niveaux B+D : %d cellules, score moyen=%.3f", n_final, mean_s,
        )

    # ═══════════════════════════════════════════════════════════════
    # SCORES RÉGIONAUX × TYPE FORÊT × ALTITUDE
    # ═══════════════════════════════════════════════════════════════

    def _compute_regional_scores(
        self,
        altitude: np.ndarray,
        region_grid: np.ndarray,
        forest_type_grid: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        result = np.full_like(altitude, 0.25, dtype=np.float32)

        alt_idx = np.zeros_like(altitude, dtype=np.int8)
        for i, (_name, a_min, a_max) in enumerate(_ALT_BANDS):
            alt_idx[(altitude >= a_min) & (altitude < a_max)] = i

        # Pré-calculer le lookup
        lookup: dict[tuple[str, int, int], float] = {}
        all_keys = set(self._region_key_map.values()) | {"_default"}

        for rk in all_keys:
            props = _REGIONAL_PROPORTIONS.get(rk, _REGIONAL_PROPORTIONS["_default"])
            for ft in (0, 1, 2, 3):
                filtered = self._filter_by_forest_type(props, ft)
                for bi in range(len(_ALT_BANDS)):
                    lookup[(rk, ft, bi)] = self._weighted_morel_score(filtered, bi)

        # Appliquer
        for reg_int, reg_key in self._region_key_map.items():
            for ft in (0, 1, 2, 3):
                for bi in range(len(_ALT_BANDS)):
                    m = (
                        mask
                        & (region_grid == reg_int)
                        & (forest_type_grid == ft)
                        & (alt_idx == bi)
                    )
                    if m.any():
                        result[m] = lookup.get(
                            (reg_key, ft, bi),
                            lookup.get(("_default", ft, bi), 0.25),
                        )

        # Fallback
        still = mask & np.isclose(result, 0.25, atol=0.01)
        if still.any():
            for ft in (0, 1, 2, 3):
                for bi in range(len(_ALT_BANDS)):
                    m = still & (forest_type_grid == ft) & (alt_idx == bi)
                    if m.any():
                        result[m] = lookup.get(("_default", ft, bi), 0.25)

        return result

    @staticmethod
    def _filter_by_forest_type(
        proportions: dict[str, float],
        forest_type: int,
    ) -> dict[str, float]:
        """Filtre par type (1=feuillus, 2=conifères, 3/0=tout)."""
        if forest_type in (0, 3):
            return proportions

        filtered: dict[str, float] = {}
        total = 0.0
        target = _DECIDUOUS if forest_type == 1 else _CONIFEROUS

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
    # GRILLE TYPE DE FORÊT
    # ═══════════════════════════════════════════════════════════════

    def _build_forest_type_grid(
        self,
        forest_gdf: Any,
        ny: int,
        nx: int,
    ) -> np.ndarray:
        if self._forest_type_grid is not None:
            return self._forest_type_grid

        ft_grid = np.zeros((ny, nx), dtype=np.int8)

        if (
            forest_gdf is not None
            and _HAS_GEO
            and "forest_type" in getattr(forest_gdf, "columns", [])
        ):
            try:
                from rasterio.features import rasterize  # type: ignore[import-untyped]
                from rasterio.transform import from_bounds  # type: ignore[import-untyped]

                bbox = dict(config.BBOX)
                transform = from_bounds(
                    bbox["xmin"], bbox["ymin"],
                    bbox["xmax"], bbox["ymax"],
                    nx, ny,
                )
                shapes = [
                    (geom, int(ft))
                    for geom, ft in zip(
                        forest_gdf.geometry,
                        forest_gdf["forest_type"],
                        strict=False,
                    )
                    if int(ft) > 0
                ]
                if shapes:
                    ft_grid = np.asarray(
                        rasterize(
                            shapes,
                            out_shape=(ny, nx),
                            transform=transform,
                            fill=0,
                            dtype="int8",
                        )
                    )
                    logger.info(
                        "  Forest type : feuillus=%d, conifères=%d, mixte=%d",
                        int((ft_grid == 1).sum()),
                        int((ft_grid == 2).sum()),
                        int((ft_grid == 3).sum()),
                    )
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

            rf["DEP"] = rf["DEP"].astype(str).str.strip().str.split(".").str[0]
            rf38 = rf[rf["DEP"] == "38"].copy()

            if rf38.empty:
                logger.warning("  Aucune région DEP=38")
                return None

            if rf38.crs is None:
                rf38 = rf38.set_crs(epsg=2154)
            elif rf38.crs.to_epsg() != 2154:
                rf38 = rf38.to_crs(epsg=2154)

            bbox = dict(config.BBOX)
            clip = box(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
            rf38 = rf38[rf38.intersects(clip)]

            if rf38.empty:
                logger.warning("  Aucune région intersecte bbox")
                return None

            # Construire mapping REGD → int id + region_key
            next_id = 1
            regd_to_int: dict[str, int] = {}

            for regd_val in rf38["REGD"].unique():
                regd_str = str(regd_val).strip()
                reg_key = _REGD_TO_KEY.get(regd_str, "_default")
                regd_to_int[regd_str] = next_id
                self._region_key_map[next_id] = reg_key
                _sub_df = rf38[rf38["REGD"] == regd_val]
                regiond = str(_sub_df["REGIOND"].iloc[0]) if len(_sub_df) > 0 else "?"
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
                )
            )

            n_mapped = int((result > 0).sum())
            logger.info(
                "  Régions rasterisées : %d/%d cellules (%.1f%%)",
                n_mapped, result.size, n_mapped / max(result.size, 1) * 100,
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
        self._region_key_map.update({
            10: "gresivaudan", 11: "chartreuse",
            12: "belledonne",  13: "vercors",
            14: "bas_drac",     0: "_default",
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
                cnt = int((region == rid).sum())
                if cnt > 0:
                    logger.info("    %s : %d cellules", rk, cnt)

        return region

    # ═══════════════════════════════════════════════════════════════
    # OBSERVATIONS TERRAIN (NIVEAU C)
    # ═══════════════════════════════════════════════════════════════

    def _load_observations(self, path: Path) -> None:
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
                & (xx >= obs.get("xmin", 0)) & (xx <= obs.get("xmax", 0))
                & (yy >= obs.get("ymin", 0)) & (yy <= obs.get("ymax", 0))
            )
            if not zone.any():
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
            n = int(zone.sum())
            count += n
            logger.info("    Obs '%s' : %d cells → %.3f", obs.get("nom", "?"), n, score)

        return count

    # ═══════════════════════════════════════════════════════════════
    # PARSEUR TFV
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_tfv(tfv_code: Any) -> tuple[str, int]:
        """
        Parse CODE_TFV → (essence, forest_type).

        Exemples réels Isère :
          FF1-00     → (unknown, 1)      FF1-09-09  → (hêtre, 1)
          FF2G61-61  → (sapin, 2)        FF31       → (unknown, 3)
          LA4        → (unknown, 0)      FP         → (peuplier, 1)
        """
        if not isinstance(tfv_code, str) or len(tfv_code) < 2:
            return ("unknown", 0)

        code = tfv_code.strip().upper()

        # Formations non forestières
        if code.startswith(("LA", "FH")):
            return ("unknown", 0)

        # Peupleraie
        if code.startswith("FP"):
            return ("peuplier", 1)

        # Type forêt depuis 3ème caractère
        forest_type = 0
        if code.startswith(("FF", "FO")) and len(code) > 2:
            c = code[2]
            if c in ("1", "0"):
                forest_type = 1
            elif c == "2":
                forest_type = 2
            elif c == "3":
                forest_type = 3

        # Extraire codes espèces après le préfixe
        clean = code.replace("G", "-")
        parts = clean.split("-")

        for part in parts[1:]:
            digits = "".join(c for c in part if c.isdigit())
            if len(digits) >= 2:
                sp_code = digits[:2]
                if sp_code != "00":
                    species = _TFV_SPECIES.get(sp_code, "unknown")
                    if species != "unknown":
                        return (species, forest_type)

        return ("unknown", forest_type)

    # ═══════════════════════════════════════════════════════════════
    # STATISTIQUES
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self, grid: Any) -> dict[str, Any]:
        ts = grid.scores.get("tree_species")
        if ts is None or not isinstance(ts, np.ndarray):
            return {"status": "no_tree_scores"}

        total = int(ts.size)
        n_nf = int(np.isclose(ts, 0.05, atol=0.02).sum())
        n_unk = int(np.isclose(ts, 0.25, atol=0.02).sum())
        n_known = total - n_nf - n_unk

        return {
            "total_cells": total,
            "forest_cells": total - n_nf,
            "species_known": n_known,
            "species_unknown": n_unk,
            "no_forest": n_nf,
            "pct_known": round(n_known / max(total - n_nf, 1) * 100, 1),
            "observations": len(self._observations),
        }