#!/usr/bin/env python3
"""
🍄 CARTOMORILLES — Construction du maillage et calcul des scores par critère

Grille régulière en Lambert 93 (EPSG:2154), résolution configurable (défaut 5m).
Chaque critère produit un array 2D de scores [0, 1] dans self.scores[nom].

Colonnes attendues des GeoDataFrames (normalisées par data_loader.py) :
  - forest:  ESSENCE, essence_canonical, tree_score, TFV
  - geology: LITHO, geology_canonical, geology_score, DESCR
  - hydro:   NOM, water_type, water_type_key, water_bonus
  - urban:   urban_type, name

Attributs exposés pour scoring.py / visualize.py :
  - scores: dict[str, np.ndarray]  (score [0,1] par critère)
  - altitude, slope, aspect, roughness: np.ndarray (valeurs brutes)
  - nodata_mask, urban_mask, water_mask: np.ndarray (bool)
  - dist_water_grid: np.ndarray (mètres)
  - nx, ny, cell_size, transform, x_coords, y_coords, xx, yy
  - bbox, to_wgs84

v2.3.0 — Fixes #12-17, #24-25
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    uniform_filter,
    zoom,
)
import _accel
import config
from config import (
    ALTITUDE_OPTIMAL,
    ALTITUDE_RANGE,
    ASPECT_SCORES,
    DIST_WATER_OPTIMAL,
    DIST_WATER_MAX,
    GEOLOGY_SCORES,
    SLOPE_MAX,
    SLOPE_MODERATE,
    SLOPE_OPTIMAL,
    SLOPE_STEEP,
    TREE_SCORES,
    TWI_DRY_FLOOR,
    TWI_DRY_LIMIT,
    TWI_OPTIMAL,
    TWI_WATERLOG,
    TWI_WET_FLOOR,
    TWI_WET_LIMIT,
    get_geology_score,
    get_tree_score,
    LANDCOVER_FOREST_FLOOR,
    DIST_WATER_FOREST_FLOOR,
    URBAN_DIST_ELIMINATORY,
    URBAN_DIST_PENALTY,
    URBAN_DIST_FULL,
    URBAN_PROXIMITY_FLOOR,
)

try:
    from _twi_numba import _accumulate_d8
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

logger = logging.getLogger("cartomorilles.grid_builder")

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES MODULE
# ═══════════════════════════════════════════════════════════════

# Fill values pour les zones non couvertes par les couches vectorielles
FILL_NO_FOREST: float = 0.05
FILL_NO_GEOLOGY: float = float(GEOLOGY_SCORES.get("unknown", 0.30))
FILL_DEFAULT_BONUS: float = 1.0

# Rugosité terrain — non présent dans config.py v2.2.0, défini localement
ROUGHNESS_OPTIMAL: float = 3.0
ROUGHNESS_MAX: float = 12.0
ROUGHNESS_WINDOW: int = 5

# Seuil de pente pour classifier "plat" (aspect)
FLAT_SLOPE_THRESHOLD: float = 5.0

# Seuil de forte pente pour sol instable (ground_cover)
STEEP_SLOPE_THRESHOLD: float = 45.0

# Référence angulaire pour le scoring continu d'aspect
_ASPECT_REF_DEG: float = 170.0
_ASPECT_BASE: float = 0.65
_ASPECT_AMPLITUDE: float = 0.35

# v2.2.0 — Score canopy par défaut pour terrain ouvert hors forêt
_CANOPY_OPEN_FIELD: float = 0.10
# v2.2.0 — Score canopy intérieur forêt (dense, loin lisières)
_CANOPY_FOREST_INTERIOR: float = 0.55

# v2.2.0 — Critères modulés par green_score dans apply_landcover_mask
_VEGETATION_CRITERIA: frozenset[str] = frozenset({
 "canopy_openness", "ground_cover", "disturbance",
})

# ── Canopy terrain correlation (v2.3.6) ─────────────────────────
_CANOPY_BASE_INTERIOR: float = 0.45       # was 0.55 — marge pour bonus
_CANOPY_SLOPE_BONUS: float = 0.10         # trouées naturelles 10-30°
_CANOPY_ALT_BONUS: float = 0.05           # structure montagnarde 300-800m
_CANOPY_SPECIES_BONUS: float = 0.05       # essences favorables (score > 0.40)

__all__ = ("GridBuilder",)


# ═══════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ═══════════════════════════════════════════════════════════════

class GridBuilder:
    """
    Construit le maillage spatial et calcule les scores par critère.

    Usage :
        grid = GridBuilder()
        grid.compute_terrain(dem_data)
        grid.score_altitude()
        grid.score_slope()
        ...
        grid.validate_scores()
        grid.get_score_summary()
    """

    def __init__(self) -> None:
        self.bbox: dict[str, float] = dict(config.BBOX)
        self.cell_size: float = config.CELL_SIZE

        self.nx: int = int(
            (self.bbox["xmax"] - self.bbox["xmin"]) / self.cell_size
        )
        self.ny: int = int(
            (self.bbox["ymax"] - self.bbox["ymin"]) / self.cell_size
        )

        logger.info(
            "📐 Grille : %d × %d = %s cellules de %.0fm",
            self.nx,
            self.ny,
            f"{self.nx * self.ny:,}",
            self.cell_size,
        )
        logger.info(
            "   Emprise : %.0fm × %.0fm",
            self.bbox["xmax"] - self.bbox["xmin"],
            self.bbox["ymax"] - self.bbox["ymin"],
        )

        # Coordonnées des centres de cellules (Lambert 93)
        self.x_coords: np.ndarray = np.linspace(
            self.bbox["xmin"] + self.cell_size / 2,
            self.bbox["xmax"] - self.cell_size / 2,
            self.nx,
        )
        # ── Fix #12 : centres DANS le bbox (était +/- cell_size/2 inversé) ──
        self.y_coords: np.ndarray = np.linspace(
            self.bbox["ymax"] - self.cell_size / 2,
            self.bbox["ymin"] + self.cell_size / 2,
            self.ny,
        )
        self.xx: np.ndarray
        self.yy: np.ndarray
        self.xx, self.yy = np.meshgrid(self.x_coords, self.y_coords)

        # Transform rasterio pour rasterize()
        self.transform: Affine = from_bounds(
            self.bbox["xmin"],
            self.bbox["ymin"],
            self.bbox["xmax"],
            self.bbox["ymax"],
            self.nx,
            self.ny,
        )

        self.to_wgs84: Transformer = Transformer.from_crs(
            "EPSG:2154", "EPSG:4326", always_xy=True
        )

        # Scores par critère (chaque valeur = array (ny, nx) dans [0, 1])
        self.scores: dict[str, np.ndarray] = {}

        # Masques
        self.nodata_mask: np.ndarray | None = None
        self.urban_mask: np.ndarray | None = None
        self.water_mask: np.ndarray | None = None

        # Données terrain brutes
        self.altitude: np.ndarray | None = None
        self.slope: np.ndarray | None = None
        self.aspect: np.ndarray | None = None
        self.roughness: np.ndarray | None = None
        self.twi: np.ndarray | None = None 
        self.dist_water_grid: np.ndarray | None = None

        # Rasters intermédiaires (pour diagnostics)
        self.tree_raster: np.ndarray | None = None
        self.geology_raster: np.ndarray | None = None
        self.forest_mask: np.ndarray | None = None

        # Confiance par critère (0=synthétique, 0.5=OSM, 1.0=données fiables)
        self.score_confidence: dict[str, float] = {}

        # Attributs attendus par scoring.py (interface)
        self.eliminatory_species_mask: np.ndarray | None = None
        self.eliminatory_geology_mask: np.ndarray | None = None
        self._landcover_green_score: np.ndarray | None = None

        # Fix #13 — score tree_species brut AVANT modulation landcover
        self._raw_tree_species: np.ndarray | None = None

        # Flag terrain calculé
        self._terrain_computed: bool = False

    def __repr__(self) -> str:
        n_scores = len(self.scores)
        return (
            f"GridBuilder(nx={self.nx}, ny={self.ny}, "
            f"cell_size={self.cell_size}, n_scores={n_scores})"
        )

    # ═══════════════════════════════════════════════════════
    #  HELPERS INTERNES
    # ═══════════════════════════════════════════════════════

    def _require_terrain(self) -> None:
        """Vérifie que compute_terrain() a été appelé."""
        if not self._terrain_computed:
            raise RuntimeError(
                "compute_terrain() doit être appelé avant les scores. "
                "Appelez grid.compute_terrain(dem_data) d'abord."
            )

    def _ensure_l93(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Reprojette en Lambert 93 si nécessaire (garde-fou)."""
        if len(gdf) == 0:
            return gdf
        crs = gdf.crs
        if crs is None:
            logger.warning("GeoDataFrame sans CRS → assumé EPSG:2154")
            return gdf.set_crs("EPSG:2154")
        if crs.to_epsg() != 2154:
            return gdf.to_crs("EPSG:2154")
        return gdf

    def _apply_nodata(self, score: np.ndarray) -> np.ndarray:
        """Met à 0 les cellules NoData du DEM."""
        if self.nodata_mask is not None and self.nodata_mask.any():
            score = score.copy()
            score[self.nodata_mask] = 0.0
        return score

    def _fill_nan_dem(self, dem: np.ndarray) -> np.ndarray:
        """
        Remplit les NaN du DEM par le plus proche voisin valide.
        Utilisé uniquement pour le calcul des dérivées (gradient, rugosité).
        """
        nan_mask = np.isnan(dem)
        if not nan_mask.any():
            return dem
        from scipy.ndimage import distance_transform_edt as _scipy_edt
        indices: np.ndarray = np.asarray(
            _scipy_edt(
                nan_mask, return_distances=False, return_indices=True,
            )
        )
        filled = dem.copy()
        filled[nan_mask] = filled[tuple(indices)][nan_mask]
        return filled

    def _log_score_stats(self, name: str, score: np.ndarray) -> None:
        """Affiche des statistiques homogènes pour chaque score."""
        valid = (
            score[~self.nodata_mask]
            if (self.nodata_mask is not None and self.nodata_mask.any())
            else score
        )
        if valid.size == 0:
            logger.info("✅ Score %-20s : aucune cellule valide", name)
            return
        logger.info(
            "✅ Score %-20s : min=%.2f  max=%.2f  moy=%.2f  "
            "med=%.2f  >0.7=%.1f%%  =0=%.1f%%",
            name,
            float(np.min(valid)),
            float(np.max(valid)),
            float(np.mean(valid)),
            float(np.median(valid)),
            float(np.sum(valid >= 0.7)) / valid.size * 100,
            float(np.sum(valid == 0)) / valid.size * 100,
        )

    def _skip_zero_weight(self, name: str) -> bool:
        """Retourne True si le critère a un poids nul → skip le calcul.

        Enregistre quand même un score neutre (0.5) pour que le pipeline
        ne casse pas en aval (validate_scores, scoring, visualize).
        """
        weight = config.WEIGHTS.get(name, 0.0)
        if weight <= 0.0:
            logger.info(
                "⏭️  Score %-20s : poids=0 → skip (neutre 0.5)", name,
            )
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.5, dtype=np.float32,
            )
            self.score_confidence[name] = 0.0
            return True
        return False

    def _rasterize_max(
        self,
        shapes_scores: list[tuple[Any, float]],
        fill: float = 0.0,
        all_touched: bool = True,
    ) -> np.ndarray:
        """
        Rasterise des géométries avec scores en prenant le MAX
        en cas de chevauchement (au lieu du dernier-gagne).
        """
        if not shapes_scores:
            return np.full((self.ny, self.nx), fill, dtype=np.float32)

        result = np.full((self.ny, self.nx), fill, dtype=np.float32)

        # Grouper par score pour réduire le nombre de rasterize
        score_groups: dict[float, list[Any]] = {}
        for geom, score_val in shapes_scores:
            score_val = round(float(score_val), 3)
            if score_val not in score_groups:
                score_groups[score_val] = []
            score_groups[score_val].append(geom)

        for score_val, geoms in score_groups.items():
            shapes = [(g, 1) for g in geoms]
            mask = np.asarray(rasterize(
                shapes,
                out_shape=(self.ny, self.nx),
                transform=self.transform,
                fill=0,
                dtype=np.uint8,
                all_touched=all_touched,
            )).astype(bool)
            result[mask] = np.maximum(result[mask], score_val)

        return result

    # ═══════════════════════════════════════════════════════
    #  TERRAIN (MNT)
    # ═══════════════════════════════════════════════════════

    def compute_terrain(self, dem_data: dict[str, Any]) -> GridBuilder:
        """
        Calcule altitude, pente, exposition, rugosité et TWI depuis le MNT.

        Optimisation v2.4.0 : calcul à résolution native du DEM, puis
        resample vers la grille cible. Évite le zoom ×N du DEM brut
        qui multiplie le coût du TWI D8 par N² et crée des artefacts
        de micro-drainage fictifs sur le DEM interpolé.
        """
        # Warmup accélérateurs au premier appel
        if not hasattr(GridBuilder, "_accel_warmed"):
            _accel.warmup()
            GridBuilder._accel_warmed = True

        dem_raw = np.asarray(dem_data["data"]).astype(np.float32)
        dem_ny, dem_nx = dem_raw.shape

        # ── Résolution native du DEM ──
        dem_res_x: float | None = dem_data.get("res_x")
        dem_res_y: float | None = dem_data.get("res_y")
        if dem_res_x is not None and dem_res_y is not None:
            dem_cell_size = float((abs(dem_res_x) + abs(dem_res_y)) / 2)
        else:
            # Estimation depuis emprise / shape
            bbox_dx = self.bbox["xmax"] - self.bbox["xmin"]
            bbox_dy = self.bbox["ymax"] - self.bbox["ymin"]
            dem_cell_size = float(
                (bbox_dx / dem_nx + bbox_dy / dem_ny) / 2,
            )

        needs_resample = (dem_ny != self.ny) or (dem_nx != self.nx)
        ratio = self.cell_size / dem_cell_size if dem_cell_size > 0 else 1.0

        if needs_resample:
            logger.info(
                "   DEM natif : %d×%d @ %.1fm → grille %d×%d @ %.1fm "
                "(ratio ×%.1f)",
                dem_nx, dem_ny, dem_cell_size,
                self.nx, self.ny, self.cell_size,
                ratio,
            )
            logger.info(
                "   ⚡ Terrain calculé à résolution native "
                "(%s cells au lieu de %s)",
                f"{dem_ny * dem_nx:,}",
                f"{self.ny * self.nx:,}",
            )
        # ── Altitude : resample vers grille cible (interpolation) ──
        if needs_resample:
            nan_mask_raw = np.isnan(dem_raw)
            altitude = self._zoom_grid(dem_raw, nan_mask_raw, order=1)
        else:
            altitude = dem_raw.copy()
        self.altitude = altitude

        # ── Masque NoData (sur grille cible) ──
        nodata_mask: np.ndarray = np.isnan(altitude)
        self.nodata_mask = nodata_mask
        n_nodata = int(nodata_mask.sum())
        if n_nodata > 0:
            logger.debug(
                "   %d cellules NoData (%.1f%%)",
                n_nodata,
                n_nodata / altitude.size * 100,
            )

        # ── Calcul terrain à résolution NATIVE du DEM ──
        nan_mask_native = np.isnan(dem_raw)
        dem_filled_native = (
            self._fill_nan_dem(dem_raw)
            if np.any(nan_mask_native)
            else dem_raw
        )

        # Pente + aspect à résolution native
        native_slope, native_aspect = self._compute_slope_aspect_from(
            dem_filled_native, dem_cell_size,
        )

        # Rugosité à résolution native
        native_roughness = self._compute_roughness_from(native_slope)

        # TWI à résolution native (le gros gain de perf)
        native_twi = self._compute_twi(dem_filled_native, dem_cell_size)

        # ── Restaurer NaN dans les dérivées natives ──
        if np.any(nan_mask_native):
            native_slope[nan_mask_native] = np.nan
            native_aspect[nan_mask_native] = np.nan
            native_roughness[nan_mask_native] = np.nan
            native_twi[nan_mask_native] = np.nan

        # ── Resample vers grille cible ──
        if needs_resample:
            self.slope = self._zoom_grid(
                native_slope, nan_mask_native, order=1,
            )
            self.aspect = self._zoom_grid(
                native_aspect, nan_mask_native, order=0,
            )
            self.roughness = self._zoom_grid(
                native_roughness, nan_mask_native, order=1,
            )
            self.twi = self._zoom_grid(
                native_twi, nan_mask_native, order=1,
            )
        else:
            self.slope = native_slope
            self.aspect = native_aspect
            self.roughness = native_roughness
            self.twi = native_twi

        # P3 : narrowing pour Pylance
        assert self.slope is not None
        assert self.aspect is not None
        assert self.roughness is not None
        assert self.twi is not None

        # ── Restaurer NaN sur grille cible ──
        if self.nodata_mask is not None and np.any(self.nodata_mask):
            self.slope[self.nodata_mask] = np.nan
            self.aspect[self.nodata_mask] = np.nan
            self.roughness[self.nodata_mask] = np.nan
            self.twi[self.nodata_mask] = np.nan

        self._terrain_computed = True
        self._log_terrain_stats()

        logger.info(
            "   TWI       : %.1f–%.1f (moy=%.1f)",
            float(np.nanmin(self.twi)),
            float(np.nanmax(self.twi)),
            float(np.nanmean(self.twi)),
        )

        return self

    def _zoom_grid(
        self,
        grid: np.ndarray,
        nan_mask: np.ndarray,
        order: int = 1,
    ) -> np.ndarray:
        """Resample une grille native vers (self.ny, self.nx) en gérant NaN.

        Parameters
        ----------
        grid : grille à résolution native du DEM.
        nan_mask : masque NaN à la même résolution.
        order : 0=nearest (catégoriel), 1=bilinéaire (continu).
        """
        zy = self.ny / grid.shape[0]
        zx = self.nx / grid.shape[1]

        if np.any(nan_mask):
            filled = np.where(nan_mask, 0.0, grid)
            zoomed: np.ndarray = np.asarray(
                zoom(filled, (zy, zx), order=order),
            ).astype(np.float32)
            nan_zoomed: np.ndarray = (
                np.asarray(
                    zoom(nan_mask.astype(np.float32), (zy, zx), order=0),
                )
                > 0.5
            )
            zoomed[nan_zoomed] = np.nan
            return zoomed

        return np.asarray(
            zoom(grid, (zy, zx), order=order),
        ).astype(np.float32)

    @staticmethod
    def _compute_roughness_from(slope: np.ndarray) -> np.ndarray:
        """Rugosité = écart-type local de la pente.

        Version statique pour calcul à résolution native.
        """
        return _accel.compute_roughness(slope, ROUGHNESS_WINDOW)

    @staticmethod
    def _compute_slope_aspect_from(
        dem: np.ndarray,
        dx: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pente (°) et aspect (°) via Horn — accéléré GPU/CPU."""
        return _accel.compute_slope_aspect(dem, dx)

    def _compute_roughness(self) -> None:
        """Rugosité — wrapper rétrocompatible."""
        _slope = self.slope
        assert _slope is not None, "slope required for roughness"
        self.roughness = self._compute_roughness_from(_slope)

    @staticmethod
    def _compute_twi(
        dem: np.ndarray,
        cell_size: float,
    ) -> np.ndarray:
        """
        Calcul du Topographic Wetness Index — TWI = ln(a / tan(β)).

        D1 : algorithme D8 (pas D∞/MFD).

        Optimisation v2.4.0 :
        - flow_dir parallélisé via Numba (indépendant par pixel)
        - accumulation séquentielle (dépendance topologique)
        """
        ny, nx = dem.shape
        cell_area = cell_size * cell_size

        # ── 1. Pente locale β (radians), floor pour éviter div/0 ──
        dy, dx = np.gradient(dem, cell_size)
        slope_rad: np.ndarray = np.arctan(
            np.sqrt(dx**2 + dy**2),
        ).astype(np.float32)
        slope_rad = np.maximum(slope_rad, np.radians(0.1))

        # ── 2. Direction d'écoulement D8 ──
        flow_dir = _accel.compute_flow_dir_d8(
            dem.astype(np.float64), cell_size,
        )
        logger.debug("   D8 flow_dir via _accel")

        # ── 3. Aire drainée par accumulation D8 ──
        flat_idx = np.argsort(dem.ravel())[::-1].astype(np.int64)

        if _HAS_NUMBA:
            from _twi_numba import _accumulate_d8

            acc = _accumulate_d8(flat_idx, flow_dir, cell_area, ny, nx)
            logger.debug("   D8 accumulation via numba")
        else:
            d8_dr = (-1, -1, 0, 1, 1, 1, 0, -1)
            d8_dc = (0, 1, 1, 1, 0, -1, -1, -1)
            logger.debug(
                "   D8 accumulation Python pur (lent sur %d cellules)",
                ny * nx,
            )
            acc = np.ones((ny, nx), dtype=np.float64) * cell_area
            for pixel in flat_idx:
                r = pixel // nx
                c = pixel % nx
                d = flow_dir[r, c]
                if d < 0:
                    continue
                nr = r + d8_dr[d]
                nc = c + d8_dc[d]
                if 0 <= nr < ny and 0 <= nc < nx:
                    acc[nr, nc] += acc[r, c]

        # ── 4. TWI = ln(a / tan(β)) ──
        specific_area = acc / cell_size
        tan_beta = np.tan(slope_rad)
        ratio = np.maximum(specific_area / tan_beta, 1e-6)
        twi: np.ndarray = np.log(ratio).astype(np.float32)

        return twi

    def _log_terrain_stats(self) -> None:
        """Affiche les statistiques terrain."""
        _alt = self.altitude
        assert _alt is not None
        _slope = self.slope
        assert _slope is not None
        _rough = self.roughness
        assert _rough is not None

        logger.info("✅ Terrain :")
        logger.info(
            "   Altitude  : %.0f–%.0fm",
            float(np.nanmin(_alt)),
            float(np.nanmax(_alt)),
        )
        logger.info(
            "   Pente     : 0–%.1f° (moy=%.1f°)",
            float(np.nanmax(_slope)),
            float(np.nanmean(_slope)),
        )
        logger.info(
            "   Rugosité  : 0–%.1f° (moy=%.1f°)",
            float(np.nanmax(_rough)),
            float(np.nanmean(_rough)),
        )
        valid = (
            _slope[~self.nodata_mask]
            if (self.nodata_mask is not None and self.nodata_mask.any())
            else _slope
        )
        n = valid.size
        if n > 0:
            logger.info(
                "   Répartition : plat(<8°) %.0f%% | modéré(8-15°) %.0f%% "
                "| raide(15-25°) %.0f%% | extrême(>25°) %.0f%%",
                float((valid < 8).sum()) / n * 100,
                float(((valid >= 8) & (valid < 15)).sum()) / n * 100,
                float(((valid >= 15) & (valid < 25)).sum()) / n * 100,
                float((valid >= 25).sum()) / n * 100,
            )


    # ═══════════════════════════════════════════════════════
    #  SCORES TERRAIN
    # ═══════════════════════════════════════════════════════

    def score_altitude(self) -> GridBuilder:
        """
        Score d'altitude — v2.2.0 : optimal 200-600m, bonus à 350m.
        """
        if self._skip_zero_weight("altitude"):
            return self
        self._require_terrain()
        _alt = self.altitude
        assert _alt is not None

        alt_min, alt_max = ALTITUDE_OPTIMAL
        range_min, range_max = ALTITUDE_RANGE

        score = np.zeros_like(_alt, dtype=np.float32)

        # Zone optimale → 1.0
        mask_opt = (_alt >= alt_min) & (_alt <= alt_max)
        score[mask_opt] = 1.0

        # Bonus micro-humidité centré sur 350m (Vouillants)
        if mask_opt.any():
            center = config.ALTITUDE_ALLUVIAL_CENTER
            alluvial_bonus = np.exp(
                -((_alt - center) ** 2) / (2 * 100.0**2)
            ) * 0.05
            score[mask_opt] = np.minimum(
                score[mask_opt] + alluvial_bonus[mask_opt], 1.0
            )

        # Sous l'optimum
        mask_low = (_alt >= range_min) & (_alt < alt_min)
        if mask_low.any():
            score[mask_low] = (_alt[mask_low] - range_min) / (
                alt_min - range_min
            )

        # Au-dessus de l'optimum
        mask_high = (_alt > alt_max) & (_alt <= range_max)
        if mask_high.any():
            score[mask_high] = 1.0 - (
                (_alt[mask_high] - alt_max) / (range_max - alt_max)
            )

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["altitude"] = score
        self.score_confidence["altitude"] = 1.0
        self._log_score_stats("altitude", score)
        return self

    def score_slope(self) -> GridBuilder:
        """Score de pente — v2.3.4 fix #38.

        Calibré Alpes Isère : morilles confirmées 20-35° (ND-de-Mésage,
        Chartreuse, Vouillants). Piecewise, transitions douces.

          0–15°  optimal     → 1.0
         15–30°  modéré      → 1.0 → 0.55
         30–40°  raide       → 0.55 → 0.15
         40–50°  très raide  → 0.15 → 0.0
           >50°  éliminatoire → 0.0
        """
        if self._skip_zero_weight("slope"):
            return self        
        self._require_terrain()
        _slope = self.slope
        assert _slope is not None

        opt_max  = SLOPE_OPTIMAL[1]    # 15°
        moderate = SLOPE_MODERATE       # 30°
        steep    = SLOPE_STEEP          # 40°
        maximum  = SLOPE_MAX            # 50°

        score = np.ones_like(_slope, dtype=np.float32)

        # 15–30° : modéré (1.0 → 0.55)
        mask = (_slope > opt_max) & (_slope <= moderate)
        if mask.any():
            t = (_slope[mask] - opt_max) / (moderate - opt_max)
            score[mask] = 1.0 - 0.45 * t

        # 30–40° : raide (0.55 → 0.15)
        mask = (_slope > moderate) & (_slope <= steep)
        if mask.any():
            t = (_slope[mask] - moderate) / (steep - moderate)
            score[mask] = 0.55 - 0.40 * (t ** 0.7)

        # 40–50° : très raide (0.15 → 0.0)
        mask = (_slope > steep) & (_slope <= maximum)
        if mask.any():
            t = (_slope[mask] - steep) / (maximum - steep)
            score[mask] = 0.15 * (1.0 - t)

        # >50° : éliminatoire
        score[_slope > maximum] = 0.0

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["slope"] = score
        self.score_confidence["slope"] = 0.9
        self._log_score_stats("slope", score)
        return self
    
    def score_terrain_roughness(self) -> GridBuilder:
        """Score de rugosité terrain — pénalise les zones accidentées."""

        if self._skip_zero_weight("terrain_roughness"):
            return self        
        self._require_terrain()
        _rough = self.roughness
        assert _rough is not None

        r_opt = ROUGHNESS_OPTIMAL
        r_max = ROUGHNESS_MAX

        score = np.ones_like(_rough, dtype=np.float32)

        mask = (_rough > r_opt) & (_rough <= r_max)
        if mask.any():
            t = (_rough[mask] - r_opt) / (r_max - r_opt)
            score[mask] = 1.0 - t**0.8

        score[_rough > r_max] = 0.0

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["terrain_roughness"] = score
        self.score_confidence["terrain_roughness"] = 0.8
        self._log_score_stats("terrain_roughness", score)
        return self

    def score_aspect(self) -> GridBuilder:
        """Score d'exposition — fonction continue sinusoïdale."""
        if self._skip_zero_weight("aspect"):
            return self        
        self._require_terrain()

        _aspect_val = self.aspect
        assert _aspect_val is not None, "aspect is None"
        _aspect: np.ndarray = _aspect_val

        _slope_val = self.slope
        assert _slope_val is not None, "slope is None"
        _slope: np.ndarray = _slope_val

        # Score continu sinusoïdal : max à _ASPECT_REF_DEG (≈Sud)
        aspect_rad = np.radians(_aspect - _ASPECT_REF_DEG)
        score = (_ASPECT_BASE + _ASPECT_AMPLITUDE * np.cos(aspect_rad)).astype(
            np.float32
        )

        # Zones plates → score flat (overwrite)
        flat_mask = _slope < FLAT_SLOPE_THRESHOLD
        score[flat_mask] = ASPECT_SCORES["flat"]

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["aspect"] = score
        self.score_confidence["aspect"] = 0.9
        self._log_score_stats("aspect", score)
        return self

    def score_twi(self) -> GridBuilder:
        """Score TWI — courbe contrastée avec plateau optimal marqué."""
        if self._skip_zero_weight("twi"):
            return self        
        self._require_terrain()
        _twi = self.twi
        assert _twi is not None

        twi_arr = np.asarray(_twi, dtype=np.float32)

        opt_lo, opt_hi = TWI_OPTIMAL
        dry_limit = TWI_DRY_LIMIT
        wet_limit = TWI_WET_LIMIT
        waterlog = TWI_WATERLOG
        dry_floor = TWI_DRY_FLOOR
        wet_floor = TWI_WET_FLOOR

        score = np.full_like(twi_arr, np.nan, dtype=np.float32)
        valid = np.isfinite(twi_arr)

        if not np.any(valid):
            self.scores["twi"] = score
            self.score_confidence["twi"] = 0.0
            return self

        # ── Early-out : extraire uniquement les valeurs valides ──
        twi_v = twi_arr[valid]
        score_v = np.empty(twi_v.shape, dtype=np.float32)

        # Plateau optimal [opt_lo, opt_hi] → 1.0
        m_opt = (twi_v >= opt_lo) & (twi_v <= opt_hi)
        score_v[m_opt] = 1.0

        # Zone sèche : dry_limit → opt_lo (concave)
        m_dry_mid = (twi_v >= dry_limit) & (twi_v < opt_lo)
        if np.any(m_dry_mid):
            t = (twi_v[m_dry_mid] - dry_limit) / (opt_lo - dry_limit)
            score_v[m_dry_mid] = dry_floor + (1.0 - dry_floor) * t**0.6

        # Très sec : < dry_limit → dry_floor
        m_very_dry = twi_v < dry_limit
        score_v[m_very_dry] = dry_floor

        # Zone humide : opt_hi → wet_limit (convexe)
        m_wet_mid = (twi_v > opt_hi) & (twi_v <= wet_limit)
        if np.any(m_wet_mid):
            t = (twi_v[m_wet_mid] - opt_hi) / (wet_limit - opt_hi)
            score_v[m_wet_mid] = 1.0 - (1.0 - wet_floor) * t**1.5

        # Très humide : wet_limit → waterlog
        m_wet_high = (twi_v > wet_limit) & (twi_v <= waterlog)
        if np.any(m_wet_high):
            t = (twi_v[m_wet_high] - wet_limit) / (waterlog - wet_limit)
            score_v[m_wet_high] = wet_floor * (1.0 - t)

        # Engorgement : > waterlog → 0.0
        m_waterlog = twi_v > waterlog
        score_v[m_waterlog] = 0.0

        np.clip(score_v, 0, 1, out=score_v)
        score[valid] = score_v

        score = self._apply_nodata(score)
        self.scores["twi"] = score
        self.score_confidence["twi"] = 0.85
        self._log_score_stats("twi", score)

        # Stats distribution
        logger.info(
            "  TWI distribution — min=%.1f  med=%.1f  max=%.1f  "
            "dry(<%.0f)=%d  optimal(%s–%s)=%d  "
            "wet(>%.0f)=%d  waterlog(>%.0f)=%d",
            float(np.min(twi_v)),
            float(np.median(twi_v)),
            float(np.max(twi_v)),
            dry_limit,
            int(np.sum(m_very_dry)),
            opt_lo, opt_hi,
            int(np.sum(m_opt)),
            wet_limit,
            int(np.sum(m_wet_mid)) + int(np.sum(m_wet_high)),
            waterlog,
            int(np.sum(m_waterlog)),
        )

        return self

    def get_twi_raw(self) -> np.ndarray | None:
        """Retourne le raster TWI brut (valeurs hydrologiques). None si pas calculé."""
        return self.twi if hasattr(self, "twi") and self.twi is not None else None

    # ═══════════════════════════════════════════════════════
    #  SCORES ÉCOLOGIQUES (données vectorielles)
    # ═══════════════════════════════════════════════════════

    def score_distance_water(
        self, hydro_gdf: gpd.GeoDataFrame | None
    ) -> GridBuilder:
        """Score de distance aux cours d'eau et plans d'eau."""
        if self._skip_zero_weight("dist_water"):
            return self        
        self._require_terrain()

        if hydro_gdf is None or hydro_gdf.empty:
            logger.warning("⚠️ Pas de données hydro → score neutre 0.3")
            self.scores["dist_water"] = np.full(
                (self.ny, self.nx), 0.3, dtype=np.float32
            )
            self.dist_water_grid = np.full(
                (self.ny, self.nx), 9999.0, dtype=np.float32
            )
            self.water_mask = np.zeros((self.ny, self.nx), dtype=bool)
            self.score_confidence["dist_water"] = 0.0
            return self

        hydro_gdf = self._ensure_l93(hydro_gdf)

        # ── Rasteriser TOUTES les entités hydro ──
        water_shapes: list[tuple[Any, int]] = []
        for _, row in hydro_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type in ("LineString", "MultiLineString"):
                try:
                    buffered = geom.buffer(2.0)
                    if buffered.is_valid and not buffered.is_empty:
                        water_shapes.append((buffered, 1))
                except Exception:
                    continue
            elif geom.geom_type in ("Polygon", "MultiPolygon"):
                water_shapes.append((geom, 1))

        if not water_shapes:
            logger.warning("⚠️ Aucune géométrie hydro valide")
            self.scores["dist_water"] = np.full(
                (self.ny, self.nx), 0.3, dtype=np.float32
            )
            self.dist_water_grid = np.full(
                (self.ny, self.nx), 9999.0, dtype=np.float32
            )
            self.water_mask = np.zeros((self.ny, self.nx), dtype=bool)
            return self

        water_raster: np.ndarray = np.asarray(rasterize(
            water_shapes,
            out_shape=(self.ny, self.nx),
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        ))
        self.water_mask = water_raster.astype(bool)

        # Distance euclidienne en mètres (EDT)
        dist_grid: np.ndarray = (
            _accel.distance_transform_edt(~self.water_mask)
            * self.cell_size
        )
        self.dist_water_grid = dist_grid

        # ── Score de distance (courbe continue) ──
        opt_min, opt_max = DIST_WATER_OPTIMAL
        d_max = DIST_WATER_MAX

        score = np.zeros_like(dist_grid, dtype=np.float32)

        # Sur l'eau (d=0) → 0.3
        score[dist_grid == 0] = 0.3

        # 0-5m : trop proche, transition 0.3→0.5
        mask = (dist_grid > 0) & (dist_grid < 5)
        if mask.any():
            score[mask] = 0.3 + 0.2 * (dist_grid[mask] / 5.0)

        # 5m-opt_min (5-15m) : transition 0.5→1.0
        mask = (dist_grid >= 5) & (dist_grid < opt_min)
        if mask.any():
            t = (dist_grid[mask] - 5.0) / (opt_min - 5.0)
            score[mask] = 0.5 + 0.5 * t

        # opt_min-opt_max (15-50m) : optimal 1.0
        mask = (dist_grid >= opt_min) & (dist_grid <= opt_max)
        score[mask] = 1.0

        # opt_max-d_max (50-300m) : décroissance progressive
        mask = (dist_grid > opt_max) & (dist_grid <= d_max)
        if mask.any():
            t = (dist_grid[mask] - opt_max) / (d_max - opt_max)
            score[mask] = 1.0 - t**0.8

        # > d_max : 0.0
        score[dist_grid > d_max] = 0.0

        # ── Bonus par type de cours d'eau ──
        score = self._apply_water_type_bonus(score, hydro_gdf)

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["dist_water"] = score

        # Confiance
        source = (
            str(hydro_gdf["source"].iloc[0])
            if "source" in hydro_gdf.columns
            else "unknown"
        )
        self.score_confidence["dist_water"] = {
            "wfs_ign": 1.0,
            "osm": 0.7,
            "synthetic": 0.3,
            "file": 0.8,
        }.get(source, 0.5)

        _wm: np.ndarray = self.water_mask
        _not_wm = ~_wm
        _dist_min: float = (
            float(np.min(dist_grid[_not_wm]))
            if _not_wm.any()
            else 0.0
        )
        logger.info(
            "   Distance eau : %.0f–%.0fm, %d cellules en eau",
            _dist_min,
            float(np.max(dist_grid)),
            int(_wm.sum()),
        )
        self._log_score_stats("dist_water", score)
        return self

    def _apply_water_type_bonus(
        self, score: np.ndarray, hydro_gdf: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Applique le bonus par type de cours d'eau."""
        if "water_bonus" not in hydro_gdf.columns:
            return score

        bonus_raster = np.ones((self.ny, self.nx), dtype=np.float32)

        for bonus_val, group in hydro_gdf.groupby("water_bonus"):
            bonus_f = float(bonus_val)  # type: ignore[arg-type]
            if abs(bonus_f - 1.0) < 0.01:
                continue

            shapes: list[tuple[Any, int]] = []
            for geom in group.geometry:
                if geom is None or geom.is_empty:
                    continue
                try:
                    buffered = geom.buffer(100.0)
                    if buffered.is_valid and not buffered.is_empty:
                        shapes.append((buffered, 1))
                except Exception:
                    continue

            if not shapes:
                continue

            influence = np.asarray(rasterize(
                shapes,
                out_shape=(self.ny, self.nx),
                transform=self.transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            )).astype(bool)

            bonus_raster[influence] = np.maximum(
                bonus_raster[influence], bonus_f
            )

        result = score * bonus_raster
        n_modified = int((bonus_raster != 1.0).sum())
        if n_modified > 0:
            logger.debug("   Bonus hydro appliqué à %d cellules", n_modified)

        return result

    def score_tree_species(
        self, forest_gdf: gpd.GeoDataFrame | None,
    ) -> GridBuilder:
        """
        Score essences forestières — raster int-codé + lookup vectorisé.

        Rasterise les polygones forestiers en raster int16 catégoriel via
        parallel_rasterize_categorical (bandes parallèles + cache disque),
        puis convertit en scores [0,1] par np.take sur lookup code→score.

        Le tri par score croissant garantit que la meilleure essence
        prévaut en cas de chevauchement de polygones (last wins).
        """
        name = "tree_species"
        if self._skip_zero_weight(name):
            return self

        if forest_gdf is None or forest_gdf.empty:
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.5, dtype=np.float32,
            )
            self.score_confidence[name] = 0.0
            logger.info("⏭️  Score %-20s : pas de données forêt", name)
            return self

        self._require_terrain()

        # ── Identifier la colonne essence ──────────────────────────
        col: str | None = None
        for candidate in (
            "essence", "ESSENCE", "LIBELLE", "lib_frt", "CODE_TFV",
        ):
            if candidate in forest_gdf.columns:
                col = candidate
                break

        if col is None:
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.5, dtype=np.float32,
            )
            self.score_confidence[name] = 0.0
            logger.warning(
                "⚠️  Score %-20s : aucune colonne essence trouvée", name,
            )
            return self

        # ── Préparer GDF (CRS + géom valides) ─────────────────────
        gdf = self._ensure_l93(forest_gdf)
        valid_mask = (
            gdf.geometry.notna()
            & gdf.geometry.is_valid
            & (~gdf.geometry.is_empty)
        )
        gdf = gdf.loc[valid_mask].copy()

        if gdf.empty:
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.5, dtype=np.float32,
            )
            self.score_confidence[name] = 0.0
            logger.warning(
                "⚠️  Score %-20s : 0 géom valides après filtrage", name,
            )
            return self

        # ── Résolution essences (vectorisé par unique) ─────────────
        raw_unique = gdf[col].unique()
        resolve_map: dict[Any, str] = {
            r: config.resolve_tree_name(r) for r in raw_unique
        }
        gdf["_essence_canon"] = gdf[col].map(resolve_map)

        unique_essences: list[str] = sorted(gdf["_essence_canon"].unique())
        essence_to_code: dict[str, int] = {
            e: i + 1 for i, e in enumerate(unique_essences)
        }
        code_to_essence: dict[int, str] = {
            v: k for k, v in essence_to_code.items()
        }

        # ── Lookup table code → score ─────────────────────────────
        max_code = max(essence_to_code.values()) + 1
        lookup = np.full(max_code, 0.5, dtype=np.float32)
        for ess, code in essence_to_code.items():
            lookup[code] = config.get_tree_score(ess)
        lookup[0] = 0.5  # nodata = neutre

        logger.debug(
            "   tree_species : %d essences uniques, %d géom, col=%s",
            len(unique_essences), len(gdf), col,
        )

        # ── Assign codes + tri score croissant (best wins last) ───
        gdf["_burn_code"] = (
            gdf["_essence_canon"].map(essence_to_code).astype(np.int16)
        )
        gdf["_burn_score"] = np.take(
            lookup,
            np.asarray(gdf["_burn_code"].values, dtype=np.int16),
        )
        gdf = gdf.sort_values(
            "_burn_score", ascending=True,
        ).reset_index(drop=True)

        geometries: list[Any] = gdf.geometry.tolist()
        codes = np.asarray(gdf["_burn_code"].values, dtype=np.int16)

        # ── Cache (hash du mapping score → invalidation auto) ─────
        import hashlib

        lookup_hash = hashlib.md5(
            lookup.tobytes(), usedforsecurity=False,
        ).hexdigest()[:8]
        cache_path = _accel.raster_cache_path(
            "tree_species",
            f"{col}_{lookup_hash}",
            len(geometries),
            self.cell_size,
            (self.ny, self.nx),
        )
        cached = _accel.raster_cache_load(cache_path)

        if cached is not None:
            int_raster = np.asarray(cached, dtype=np.int16)
            logger.info(
                "✅ Score %-20s : cache disque (%s)",
                name, cache_path.name,
            )
        else:
            int_raster = _accel.parallel_rasterize_categorical(
                geometries=geometries,
                category_codes=codes,
                out_shape=(self.ny, self.nx),
                transform=self.transform,
                all_touched=True,
                nodata=0,
            )
            _accel.raster_cache_save(cache_path, int_raster)

        # ── Code → score vectorisé ────────────────────────────────
        safe_codes = np.clip(int_raster, 0, max_code - 1)
        score = np.take(lookup, safe_codes).astype(np.float32)
        score = np.clip(score, 0.0, 1.0)

        # ── Stocker résultats + métadonnées aval ──────────────────
        self._tree_species_int_raster = int_raster
        self._tree_code_to_name = code_to_essence
        self._tree_score_lookup = lookup

        self.scores[name] = score
        coverage = float(np.count_nonzero(int_raster > 0)) / max(
            int_raster.size, 1,
        )
        self.score_confidence[name] = min(coverage, 1.0)

        self._log_score_stats(name, score)
        return self

    def _score_from_any_column(
        self,
        gdf: gpd.GeoDataFrame,
        score_fn: Any,
        columns: list[str],
    ) -> np.ndarray:
        """
        Fallback vectorisé : résout le meilleur score depuis plusieurs colonnes.

        Fix #25 : vectorisé (était row-by-row).
        """
        default = float(TREE_SCORES["unknown"])
        best = np.full(len(gdf), default, dtype=np.float32)

        for col in columns:
            if col not in gdf.columns:
                continue
            mask = gdf[col].notna()
            if not mask.any():
                continue
            col_scores = (
                gdf.loc[mask, col]
                .astype(str)
                .map(lambda v: float(score_fn(v)))
            )
            idx = col_scores.index
            # P5 : np.asarray() pour éviter ExtensionArray
            vals = np.asarray(col_scores.values, dtype=np.float32)
            positions = gdf.index.get_indexer(idx)
            valid_pos = positions >= 0
            if valid_pos.any():
                best[positions[valid_pos]] = np.maximum(
                    best[positions[valid_pos]], vals[valid_pos]
                )

        return best

    def score_geology(
        self, geology_gdf: gpd.GeoDataFrame | None,
    ) -> GridBuilder:
        """
        Score géologie — raster int-codé + lookup vectorisé.

        Rasterise les polygones géologiques en raster int16 catégoriel via
        parallel_rasterize_categorical (bandes parallèles + cache disque),
        puis convertit en scores [0,1] par np.take sur lookup code→score.

        Résolution : DESCR prioritaire sur NOTATION (D2).
        Le tri par score croissant garantit que la meilleure catégorie
        prévaut en cas de chevauchement (last wins).
        """
        name = "geology"
        if self._skip_zero_weight(name):
            return self

        if geology_gdf is None or geology_gdf.empty:
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.5, dtype=np.float32,
            )
            self.score_confidence[name] = 0.0
            logger.info("⏭️  Score %-20s : pas de données géologie", name)
            return self

        self._require_terrain()

        # ── Identifier la colonne — DESCR prioritaire (D2) ────────
        col: str | None = None
        for candidate in (
            "DESCR", "descr", "DESCRIPTION", "description",
            "NOTATION", "notation", "CODE", "code",
        ):
            if candidate in geology_gdf.columns:
                col = candidate
                break

        if col is None:
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.5, dtype=np.float32,
            )
            self.score_confidence[name] = 0.0
            logger.warning(
                "⚠️  Score %-20s : aucune colonne géologie trouvée", name,
            )
            return self

        # ── Préparer GDF (CRS + géom valides) ─────────────────────
        gdf = self._ensure_l93(geology_gdf)
        valid_mask = (
            gdf.geometry.notna()
            & gdf.geometry.is_valid
            & (~gdf.geometry.is_empty)
        )
        gdf = gdf.loc[valid_mask].copy()

        if gdf.empty:
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.5, dtype=np.float32,
            )
            self.score_confidence[name] = 0.0
            logger.warning(
                "⚠️  Score %-20s : 0 géom valides après filtrage", name,
            )
            return self

        # ── Résolution catégories géologiques (vectorisé) ──────────
        raw_unique = gdf[col].unique()
        resolve_map: dict[Any, str] = {
            r: config.resolve_geology(r) for r in raw_unique
        }
        gdf["_geo_category"] = gdf[col].map(resolve_map)

        unique_cats: list[str] = sorted(gdf["_geo_category"].unique())
        cat_to_code: dict[str, int] = {
            c: i + 1 for i, c in enumerate(unique_cats)
        }
        code_to_cat: dict[int, str] = {
            v: k for k, v in cat_to_code.items()
        }

        # ── Lookup table code → score ─────────────────────────────
        max_code = max(cat_to_code.values()) + 1
        lookup = np.full(max_code, 0.5, dtype=np.float32)
        for cat, code in cat_to_code.items():
            lookup[code] = config.get_geology_score(cat)
        lookup[0] = 0.5  # nodata = neutre

        # ── Masque éliminatoire (codes granite/gneiss/siliceux) ────
        eliminatory_codes: frozenset[int] = frozenset(
            cat_to_code[cat]
            for cat in unique_cats
            if cat in config.ELIMINATORY_GEOLOGY
        )

        logger.debug(
            "   geology : %d catégories uniques, %d géom, col=%s, "
            "%d éliminatoires (%s)",
            len(unique_cats), len(gdf), col,
            len(eliminatory_codes),
            ", ".join(
                code_to_cat[c] for c in sorted(eliminatory_codes)
            ) or "aucune",
        )

        # ── Assign codes + tri score croissant (best wins last) ───
        gdf["_burn_code"] = (
            gdf["_geo_category"].map(cat_to_code).astype(np.int16)
        )
        gdf["_burn_score"] = np.take(
            lookup,
            np.asarray(gdf["_burn_code"].values, dtype=np.int16),
        )
        gdf = gdf.sort_values(
            "_burn_score", ascending=True,
        ).reset_index(drop=True)

        geometries: list[Any] = gdf.geometry.tolist()
        codes = np.asarray(gdf["_burn_code"].values, dtype=np.int16)

        # ── Cache (hash du mapping score → invalidation auto) ─────
        import hashlib

        lookup_hash = hashlib.md5(
            lookup.tobytes(), usedforsecurity=False,
        ).hexdigest()[:8]
        cache_path = _accel.raster_cache_path(
            "geology_cat",
            f"{col}_{lookup_hash}",
            len(geometries),
            self.cell_size,
            (self.ny, self.nx),
        )
        cached = _accel.raster_cache_load(cache_path)

        if cached is not None:
            int_raster = np.asarray(cached, dtype=np.int16)
            logger.info(
                "✅ Score %-20s : cache disque (%s)",
                name, cache_path.name,
            )
        else:
            int_raster = _accel.parallel_rasterize_categorical(
                geometries=geometries,
                category_codes=codes,
                out_shape=(self.ny, self.nx),
                transform=self.transform,
                all_touched=True,
                nodata=0,
            )
            _accel.raster_cache_save(cache_path, int_raster)

        # ── Code → score vectorisé ────────────────────────────────
        safe_codes = np.clip(int_raster, 0, max_code - 1)
        score = np.take(lookup, safe_codes).astype(np.float32)
        score = np.clip(score, 0.0, 1.0)

        # ── Stocker résultats + métadonnées aval ──────────────────
        self._geology_int_raster = int_raster
        self._geology_code_to_name = code_to_cat
        self._geology_score_lookup = lookup
        self._geology_eliminatory_codes = eliminatory_codes

        self.scores[name] = score
        coverage = float(np.count_nonzero(int_raster > 0)) / max(
            int_raster.size, 1,
        )
        self.score_confidence[name] = min(coverage, 1.0)

        self._log_score_stats(name, score)
        return self
        
    def _score_geology_from_any_column(
        self, gdf: gpd.GeoDataFrame
    ) -> np.ndarray:
        """Fallback : résout le score géologie (vectorisé, fix perf)."""
        columns = [
            "NOTATION", "notation", "CODE", "code", "LITHO",
            "DESCR", "descr", "LITHOLOGIE", "lithologie", "CODE_LEG",
        ]
        present = [c for c in columns if c in gdf.columns]
        default = float(GEOLOGY_SCORES["unknown"])

        if not present:
            return np.full(len(gdf), default, dtype=np.float32)

        best = np.full(len(gdf), default, dtype=np.float32)

        for col in present:
            vals = gdf[col].fillna("").astype(str)
            # Lookup unique → O(n_unique) appels au lieu de O(n_rows)
            unique_vals = vals.unique()
            score_map: dict[str, float] = {
                v: float(get_geology_score(v)) for v in unique_vals
            }
            col_scores = np.asarray(
                vals.map(score_map).values, dtype=np.float32,
            )
            best = np.maximum(best, col_scores)

        return best
    # ═══════════════════════════════════════════════════════
    #  SCORES MICRO-HABITAT
    # ═══════════════════════════════════════════════════════

    def score_canopy_openness(
        self, canopy_data: np.ndarray | None = None
    ) -> GridBuilder:
        """Score d'ouverture de la canopée — v2.3.6."""
        if self._skip_zero_weight("canopy_openness"):
            return self        
        self._require_terrain()

        if canopy_data is not None:
            tcd = canopy_data
            score = np.where(
                tcd < 20,
                0.4,
                np.where(
                    tcd < 40,
                    0.8,
                    np.where(tcd < 70, 1.0, np.where(tcd < 90, 0.6, 0.2)),
                ),
            ).astype(np.float32)
            self.score_confidence["canopy_openness"] = 0.9
        elif self.forest_mask is not None and self.forest_mask.any():
            score = self._estimate_canopy_from_edges()
            self.score_confidence["canopy_openness"] = 0.5
        else:
            score = np.full(
                (self.ny, self.nx), _CANOPY_OPEN_FIELD, dtype=np.float32
            )
            self.score_confidence["canopy_openness"] = 0.1

        score = self._apply_nodata(np.clip(score, 0, 1).astype(np.float32))
        self.scores["canopy_openness"] = score
        self._log_score_stats("canopy_openness", score)
        return self

    def _estimate_canopy_from_edges(self) -> np.ndarray:
        """
        Estime l'ouverture de canopée — v2.3.6.

        Fix #7  : terrain ouvert → _CANOPY_OPEN_FIELD (0.10)
        Fix #14 : transition continue 0.70 → 0.10 (plus de saut à 30m)
        v2.3.6  : corrélation pente/altitude/essences — corrige biais plaine.

        Logique :
          - Intérieur forêt : base 0.45 + bonus pente/altitude/essences
          - Lisière forêt (<30m du bord, côté intérieur) : 0.90 → base
          - Juste hors forêt (<30m du bord) : 0.70 → 0.10 (transition)
          - Terrain ouvert (>30m de toute forêt) : 0.10
        """
        _fm_val = self.forest_mask
        assert _fm_val is not None, "forest_mask is None"
        _fm: np.ndarray = _fm_val

        if not _fm.any():
            return np.full(
                (self.ny, self.nx), _CANOPY_OPEN_FIELD, dtype=np.float32
            )

        # ── Base intérieure dynamique (v2.3.6) ──
        interior_base = np.full(
            (self.ny, self.nx), _CANOPY_BASE_INTERIOR, dtype=np.float32
        )

        # Bonus pente 10-30° : trouées naturelles par la topographie
        _slope = self.slope
        if _slope is not None:
            moderate_slope = (_slope >= 10.0) & (_slope <= 30.0) & _fm
            if moderate_slope.any():
                interior_base[moderate_slope] += _CANOPY_SLOPE_BONUS

        # Bonus altitude 300-800m : structure forestière montagnarde
        _alt = self.altitude
        if _alt is not None:
            alt_bonus_mask = (_alt >= 300.0) & (_alt <= 800.0) & _fm
            if alt_bonus_mask.any():
                interior_base[alt_bonus_mask] += _CANOPY_ALT_BONUS

        # Bonus essences favorables (score enrichi > 0.40)
        ts = self.scores.get("tree_species")
        if (
            ts is not None
            and isinstance(ts, np.ndarray)
            and ts.shape == _fm.shape
        ):
            good_species = (ts > 0.40) & _fm
            if good_species.any():
                interior_base[good_species] += _CANOPY_SPECIES_BONUS

        interior_base = np.clip(interior_base, 0.0, 0.90)

        # ── Lisière = bordure intérieure de la forêt ──
        eroded: np.ndarray = np.asarray(
            binary_erosion(
                _fm,
                iterations=max(1, int(20.0 / self.cell_size)),
            )
        )
        edge_mask = _fm & ~eroded

        # Distance au pixel forêt le plus proche (pour hors-forêt)
        dist_to_forest: np.ndarray = (
            _accel.distance_transform_edt(~_fm)
            * self.cell_size
        )

        # Distance à la lisière (pour intérieur forêt)
        dist_to_edge: np.ndarray = (
            _accel.distance_transform_edt(~edge_mask)
            * self.cell_size
        )

        # ── Défaut : terrain ouvert ──
        score = np.full(
            (self.ny, self.nx), _CANOPY_OPEN_FIELD, dtype=np.float32
        )

        # ── Intérieur forêt : base dynamique ──
        score[_fm] = interior_base[_fm]

        # ── Lisière intérieure (<30m du bord) : 0.90 → interior_base ──
        near_edge_in = _fm & (dist_to_edge < 30)
        if near_edge_in.any():
            t = dist_to_edge[near_edge_in] / 30.0
            score[near_edge_in] = (
                0.90 - (0.90 - interior_base[near_edge_in]) * t
            )

        # ── Fix #14 : transition extérieure continue 0.70 → _CANOPY_OPEN_FIELD ──
        near_forest_out = (~_fm) & (dist_to_forest < 30)
        if near_forest_out.any():
            t = dist_to_forest[near_forest_out] / 30.0
            score[near_forest_out] = (
                0.70 - (0.70 - _CANOPY_OPEN_FIELD) * t
            )

        n_bonus = int(np.sum((interior_base > _CANOPY_BASE_INTERIOR) & _fm))
        logger.debug(
            "   Canopy terrain-corr : %d cellules bonifiées"
            " (pente/alt/essences)",
            n_bonus,
        )

        return score

    def score_ground_cover(self) -> GridBuilder:
        """
        Score de couverture au sol — v2.3.6.

        Fix #8  : bonus humidité uniquement en forêt, hors forêt → 0.20.
        Fix #16 : deep_forest ne pénalise plus les zones humides.
        v2.3.6  : gradient altitudinal + bonus géologie calcaire + essences.
                   Corrige biais plaine vs montagne.
        """
        if self._skip_zero_weight("ground_cover"):
            return self        
        self._require_terrain()
        _alt = self.altitude
        assert _alt is not None

        _fm_val = self.forest_mask
        has_forest = _fm_val is not None and _fm_val.any()

        # ── Base : gradient altitudinal en forêt (v2.3.6) ──
        if has_forest:
            assert _fm_val is not None
            _fm: np.ndarray = _fm_val

            score = np.full((self.ny, self.nx), 0.20, dtype=np.float32)

            in_f = _fm
            score[in_f & (_alt < 200)] = 0.35
            score[in_f & (_alt >= 200) & (_alt < 400)] = 0.50
            score[in_f & (_alt >= 400) & (_alt < 700)] = 0.65
            score[in_f & (_alt >= 700) & (_alt < 1000)] = 0.55
            score[in_f & (_alt >= 1000)] = 0.40

            # ── Bonus géologie calcaire : meilleur pH litière ──
            geo = self.scores.get("geology")
            if (
                geo is not None
                and isinstance(geo, np.ndarray)
                and geo.shape == score.shape
            ):
                calc_bonus = (geo > 0.70) & _fm
                if calc_bonus.any():
                    score[calc_bonus] += 0.15
                    logger.debug(
                        "   Ground cover : +0.15 géologie calcaire"
                        " → %d cellules",
                        int(calc_bonus.sum()),
                    )

            # ── Bonus essences favorables ──
            ts = self.scores.get("tree_species")
            if (
                ts is not None
                and isinstance(ts, np.ndarray)
                and ts.shape == score.shape
            ):
                good_sp = (ts > 0.40) & _fm
                if good_sp.any():
                    score[good_sp] += 0.10
                    logger.debug(
                        "   Ground cover : +0.10 essences favorables"
                        " → %d cellules",
                        int(good_sp.sum()),
                    )
        else:
            score = np.full((self.ny, self.nx), 0.20, dtype=np.float32)

        # ── Bonus humidité : uniquement EN FORÊT près de l'eau ──
        ideal_humid = np.zeros((self.ny, self.nx), dtype=bool)

        if self.dist_water_grid is not None and has_forest:
            assert _fm_val is not None
            _fm2: np.ndarray = _fm_val

            # Trop humide (inondable) — basse altitude uniquement
            too_wet = (self.dist_water_grid < 10) & (_alt < 300) & _fm2
            score[too_wet] = 0.25

            # Idéal : forêt + humidité résiduelle (15-80m de l'eau)
            ideal_humid = (
                (self.dist_water_grid >= 15)
                & (self.dist_water_grid <= 80)
                & _fm2
            )
            score[ideal_humid] += 0.10

        # ── Fix #16 : Forêt profonde SÈCHE → cap adaptatif ──
        if has_forest:
            assert _fm_val is not None
            _fm3: np.ndarray = _fm_val
            deep_forest: np.ndarray = np.asarray(
                binary_erosion(
                    _fm3,
                    iterations=max(1, int(40.0 / self.cell_size)),
                )
            )
            deep_forest_dry = deep_forest & ~ideal_humid

            # v2.3.6 : cap plus haut si géologie favorable
            cap = np.full((self.ny, self.nx), 0.45, dtype=np.float32)
            geo = self.scores.get("geology")
            if (
                geo is not None
                and isinstance(geo, np.ndarray)
                and geo.shape == cap.shape
            ):
                cap[geo > 0.70] = 0.55

            if deep_forest_dry.any():
                score[deep_forest_dry] = np.minimum(
                    score[deep_forest_dry], cap[deep_forest_dry]
                )

        # ── Forte pente = sol instable ──
        _slope_val = self.slope
        if _slope_val is not None:
            _slope: np.ndarray = _slope_val
            steep = _slope >= STEEP_SLOPE_THRESHOLD
            score[steep] = np.minimum(score[steep], 0.15)

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["ground_cover"] = score
        self.score_confidence["ground_cover"] = 0.3
        self._log_score_stats("ground_cover", score)
        return self

    def score_disturbance(
        self, disturbance_data: np.ndarray | None = None
    ) -> GridBuilder:
        """
        Score de perturbation du sol — v2.3.6.

        v2.3.6 : bonus pente (chablis/exploitation) + bande altitudinale.
                  Corrige biais plaine vs montagne.

        Note : le bonus proximité urbaine nécessite que apply_urban_mask()
        ait été appelé AVANT cette méthode (Fix #15 — corrigé dans main.py).
        """
        if self._skip_zero_weight("disturbance"):
            return self        
        self._require_terrain()

        if disturbance_data is not None:
            score = np.clip(disturbance_data.astype(np.float32), 0, 1)
            self.score_confidence["disturbance"] = 0.7
        elif self.forest_mask is not None and self.forest_mask.any():
            _fm: np.ndarray = self.forest_mask
            eroded: np.ndarray = np.asarray(
                binary_erosion(_fm, iterations=1)
            )
            edge_mask = _fm & ~eroded

            dist_to_edge: np.ndarray = (
                _accel.distance_transform_edt(~edge_mask)
                * self.cell_size
            )

            score = np.full((self.ny, self.nx), 0.3, dtype=np.float32)
            near_edge = dist_to_edge < 15
            if near_edge.any():
                t = dist_to_edge[near_edge] / 15.0
                score[near_edge] = 0.7 - 0.4 * t

            # ── v2.3.6 : bonus pente 15-35° en forêt (chablis naturels) ──
            _slope = self.slope
            if _slope is not None:
                slope_disturb = (_slope >= 15.0) & (_slope <= 35.0) & _fm
                if slope_disturb.any():
                    score[slope_disturb] += 0.15
                    logger.debug(
                        "   Disturbance : bonus pente 15-35°"
                        " → %d cellules",
                        int(slope_disturb.sum()),
                    )

            # ── v2.3.6 : bonus altitude 300-800m (exploitation forestière) ──
            _alt = self.altitude
            if _alt is not None:
                exploit_band = (_alt >= 300.0) & (_alt <= 800.0) & _fm
                if exploit_band.any():
                    score[exploit_band] += 0.10
                    logger.debug(
                        "   Disturbance : bonus altitude 300-800m"
                        " → %d cellules",
                        int(exploit_band.sum()),
                    )

            # ── Bonus proximité urbaine (Fix #15) ──
            _um_val = self.urban_mask
            if _um_val is not None and _um_val.any():
                _um: np.ndarray = _um_val
                urban_edge: np.ndarray = (
                    _accel.distance_transform_edt(~_um)
                    * self.cell_size
                )
                near_urban = (urban_edge < 30) & (~_um)
                score[near_urban] = np.maximum(score[near_urban], 0.5)
                logger.debug(
                    "   Disturbance : bonus urbain appliqué à %d cellules",
                    int(near_urban.sum()),
                )
            elif _um_val is None:
                logger.debug(
                    "   Disturbance : urban_mask=None — bonus urbain ignoré."
                    " Appeler apply_urban_mask() AVANT score_disturbance()."
                )

            self.score_confidence["disturbance"] = 0.3
        else:
            score = np.full((self.ny, self.nx), 0.3, dtype=np.float32)
            self.score_confidence["disturbance"] = 0.1

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["disturbance"] = score
        self._log_score_stats("disturbance", score)
        return self

    # ═══════════════════════════════════════════════════════
    #  CRITÈRES BONUS (optionnels)
    # ═══════════════════════════════════════════════════════

    def score_forest_edge_distance(self) -> GridBuilder:
        """
        Score de distance aux lisières forestières.

        Favorise les zones en lisière (≤5m) et pénalise l'intérieur profond (>50m).
        Basé sur BD Forêt v2 + EDT — actif même sans landcover OSM.
        Clé dans scores : "forest_edge" (poids 0.04 dans config.WEIGHTS).
        """
        if self._skip_zero_weight("edge_distance"):
            return self        
        self._require_terrain()

        if self.forest_mask is None or not self.forest_mask.any():
            self.scores["forest_edge"] = np.full(
                (self.ny, self.nx), 0.3, dtype=np.float32
            )
            self.score_confidence["forest_edge"] = 0.1
            return self

        _fm: np.ndarray = self.forest_mask

        eroded: np.ndarray = np.asarray(
            binary_erosion(_fm, iterations=1)
        )
        edge_mask = _fm & ~eroded

        dist: np.ndarray = (
            _accel.distance_transform_edt(~edge_mask)
            * self.cell_size
        )

        score = np.zeros_like(dist, dtype=np.float32)

        score[dist <= 5] = 1.0

        mask = (dist > 5) & (dist <= 20)
        if mask.any():
            score[mask] = 1.0 - 0.5 * ((dist[mask] - 5) / 15.0)

        mask = (dist > 20) & (dist <= 50)
        if mask.any():
            score[mask] = 0.5 - 0.4 * ((dist[mask] - 20) / 30.0)

        score[dist > 50] = 0.1

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["forest_edge"] = score
        self.score_confidence["forest_edge"] = 0.6
        self._log_score_stats("forest_edge", score)
        return self

    def score_geology_contact_distance(
        self, geo_lines_gdf: Any
    ) -> GridBuilder:
        """
        Score de distance aux contacts géologiques (BDCharm-50 L_FGEOL).

        Les transitions entre formations géologiques (calcaire/marne, alluvions/
        substrat) créent des conditions édaphiques favorables aux morilles.
        Score maximal ≤50m du contact, décroit jusqu'à 0.2 au-delà de 500m.
        Clé dans scores : "geology_contact" (poids 0.02 dans config.WEIGHTS).
        """
        name = "geology_contact"
        if self._skip_zero_weight(name):
            return self
        self._require_terrain()

        if geo_lines_gdf is None or (
            hasattr(geo_lines_gdf, "empty") and geo_lines_gdf.empty
        ):
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.2, dtype=np.float32
            )
            self.score_confidence[name] = 0.0
            logger.info("⏭️  Score %-20s : pas de données contacts géo", name)
            return self

        gdf = self._ensure_l93(geo_lines_gdf)
        valid = gdf[
            gdf.geometry.notna()
            & gdf.geometry.is_valid
            & (~gdf.geometry.is_empty)
        ]
        if valid.empty:
            self.scores[name] = np.full(
                (self.ny, self.nx), 0.2, dtype=np.float32
            )
            self.score_confidence[name] = 0.0
            return self

        # Rasteriser les lignes (all_touched=True pour ne pas rater les lignes fines)
        line_raster: np.ndarray = np.asarray(rasterize(
            valid.geometry,
            out_shape=(self.ny, self.nx),
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        ))
        contact_mask = line_raster.astype(bool)

        # EDT — distance en mètres
        dist: np.ndarray = (
            _accel.distance_transform_edt(~contact_mask)
            * self.cell_size
        )

        # Courbe de score
        score = np.full_like(dist, 0.2, dtype=np.float32)

        # ≤ 50m : optimal
        score[dist <= 50] = 1.0

        # 50–200m : décroissance 1.0 → 0.5
        m1 = (dist > 50) & (dist <= 200)
        if m1.any():
            score[m1] = np.float32(1.0 - 0.5 * ((dist[m1] - 50.0) / 150.0))

        # 200–500m : décroissance 0.5 → 0.2
        m2 = (dist > 200) & (dist <= 500)
        if m2.any():
            score[m2] = np.float32(0.5 - 0.3 * ((dist[m2] - 200.0) / 300.0))

        score = self._apply_nodata(np.clip(score, 0.0, 1.0))
        self.scores[name] = score
        self.score_confidence[name] = 0.7
        self._log_score_stats(name, score)
        return self

    def score_favorable_density(self, radius_m: float = 30.0) -> GridBuilder:
        """Score densité locale — NON INCLUS dans WEIGHTS (exploratoire)."""
        if self._skip_zero_weight("favorable_density"):
            return self        
        from scipy.ndimage import uniform_filter  # import local car optionnel

        if "tree_species" not in self.scores:
            logger.debug("   Densité : score tree_species requis, skip")
            return self

        tree = self.scores["tree_species"]
        favorable = (tree >= 0.7).astype(np.float32)

        kernel_size = max(3, int(2 * radius_m / self.cell_size) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        density: np.ndarray = _accel.uniform_filter(
            favorable, size=kernel_size,
        )
        score = self._apply_nodata(np.clip(density, 0, 1))
        self.scores["favorable_density"] = score
        self.score_confidence["favorable_density"] = self.score_confidence.get(
            "tree_species", 0.5
        )
        self._log_score_stats("favorable_density", score)
        return self

    # ═══════════════════════════════════════════════════════
    #  MASQUES D'EXCLUSION
    # ═══════════════════════════════════════════════════════

    # Buffers par type urbain (P10 : constante module)
    _URBAN_BUFFER_DEFAULTS: tuple[tuple[str, int], ...] = (
        ("batiment", 10),
        ("building", 10),
        ("residential", 15),
        ("commercial", 20),
        ("industrial", 25),
        ("route", 0),
        ("voie_ferree", 0),
        ("parking", 15),
        ("school", 15),
        ("pitch", 10),
        ("sports_centre", 15),
        ("cemetery", 5),
    )

    def apply_urban_mask(
        self,
        urban_gdf: gpd.GeoDataFrame | None,
        buffer_m: int = 10,
        *,
        min_urban_density: int = 30,
        density_radius: int = 30,
    ) -> GridBuilder:
        """Masque des zones urbanisées.

        Parameters
        ----------
        min_urban_density : int
            Nombre minimum de pixels urbains dans le voisinage
            pour qu'un pixel soit considéré comme zone urbaine.
            Filtre les bâtiments isolés en forêt.
        density_radius : int
            Rayon en mètres du voisinage pour le calcul de densité.
        """
        if urban_gdf is None or urban_gdf.empty:
            self.urban_mask = np.zeros((self.ny, self.nx), dtype=bool)
            logger.info("⚠️ Pas de données urbaines → aucun masque")
            return self

        urban_gdf = self._ensure_l93(urban_gdf)

        # ── Cache disque ──
        cache_path = _accel.raster_cache_path(
            "urban_mask",
            urban_gdf.attrs.get("source", "unknown"),
            len(urban_gdf),
            self.cell_size,
            (self.ny, self.nx),
        )
        cached = _accel.raster_cache_load(cache_path)
        if cached is not None:
            self.urban_mask = cached.astype(bool)
            _um: np.ndarray = self.urban_mask
            urban_cells = int(_um.sum())
            urban_pct = urban_cells / _um.size * 100
            urban_ha = urban_cells * (self.cell_size**2) / 10000
            logger.info(
                "✅ Masque urbain (cache) : %s cellules exclues (%.1f%%, %.1f ha)",
                f"{urban_cells:,}",
                urban_pct,
                urban_ha,
            )
            return self

        # Construire buffer_map depuis constante + surcharge buffer_m
        buffer_map: dict[str, int] = {
            k: (v if k not in ("batiment", "building") else buffer_m)
            for k, v in self._URBAN_BUFFER_DEFAULTS
        }

        # ── Rasterisation parallèle (base) ──
        all_geoms: list[Any] = [
            g for g in urban_gdf.geometry
            if g is not None and not g.is_empty
        ]

        if not all_geoms:
            self.urban_mask = np.zeros((self.ny, self.nx), dtype=bool)
            return self

        base_raster = _accel.parallel_rasterize_mask(
            all_geoms,
            out_shape=(self.ny, self.nx),
            transform=self.transform,
        )

        # ── Filtre densité : éliminer les bâtiments isolés ──
        if min_urban_density > 1:
            density_px = max(2, round(density_radius / self.cell_size))
            kernel_size = 2 * density_px + 1
            kernel_area = kernel_size * kernel_size
            base_fraction = 0.61
            fraction = base_fraction * min(1.0, 10.0 / self.cell_size)
            adaptive_threshold = max(1, round(fraction * kernel_area))

            neighbor_count = (
                _accel.uniform_filter(
                    base_raster.astype(np.float32), size=kernel_size,
                )
                * kernel_area
            )

            isolated = base_raster & (neighbor_count < adaptive_threshold)
            base_raster = base_raster & ~isolated
            n_filtered = int(isolated.sum())
            if n_filtered > 0:
                logger.info(
                    "   🏚️ Filtre densité urbaine : %d pixels isolés supprimés "
                    "(seuil=%d/%d dans rayon %dm, cell=%dm)",
                    n_filtered,
                    adaptive_threshold,
                    kernel_area,
                    density_radius,
                    int(self.cell_size),
                )

        if "urban_type" in urban_gdf.columns:
            combined_mask = np.zeros((self.ny, self.nx), dtype=bool)

            for utype, group in urban_gdf.groupby("urban_type"):
                buf = buffer_map.get(str(utype).lower(), buffer_m)

                type_geoms: list[Any] = [
                    g for g in group.geometry
                    if g is not None and not g.is_empty
                ]
                if not type_geoms:
                    continue

                type_raster = _accel.parallel_rasterize_mask(
                    type_geoms,
                    out_shape=(self.ny, self.nx),
                    transform=self.transform,
                )

                # ── Filtre densité par type ──
                if min_urban_density > 1:
                    type_neighbor = (
                        _accel.uniform_filter(
                            type_raster.astype(np.float32),
                            size=kernel_size,
                        )
                        * kernel_area
                    )
                    type_raster = type_raster & (
                        type_neighbor >= adaptive_threshold
                    )

                if buf > 0:
                    iterations = max(0, round(buf / self.cell_size))
                    if iterations > 0:
                        type_raster = np.asarray(
                            binary_dilation(type_raster, iterations=iterations),
                        ).astype(bool)

                combined_mask |= type_raster

            self.urban_mask = combined_mask
        else:
            if buffer_m > 0:
                iterations = max(0, round(buffer_m / self.cell_size))
                if iterations > 0:
                    base_raster = np.asarray(
                        binary_dilation(base_raster, iterations=iterations),
                    ).astype(bool)
            self.urban_mask = base_raster

        # ── Sauvegarder en cache ──
        _accel.raster_cache_save(cache_path, self.urban_mask)

        _um = self.urban_mask
        urban_cells = int(_um.sum())
        urban_pct = urban_cells / _um.size * 100
        urban_ha = urban_cells * (self.cell_size**2) / 10000
        logger.info(
            "✅ Masque urbain : %s cellules exclues (%.1f%%, %.1f ha)",
            f"{urban_cells:,}",
            urban_pct,
            urban_ha,
        )
        return self

    def score_urban_proximity(self) -> GridBuilder:
        """Score de proximité urbaine — pénalise les zones proches de l'urbanisation."""
        if self._skip_zero_weight("urban_proximity"):
            return self        
        _um = getattr(self, "urban_mask", None)
        if not isinstance(_um, np.ndarray) or not np.any(_um):
            logger.info("⚠️ Pas de masque urbain → score neutre 1.0")
            self.scores["urban_proximity"] = np.ones(
                (self.ny, self.nx), dtype=np.float32
            )
            self.dist_urban_grid = np.full(
                (self.ny, self.nx), 9999.0, dtype=np.float32
            )
            self.score_confidence["urban_proximity"] = 0.0
            return self

        # Distance euclidienne en mètres depuis le bord du masque urbain
        dist_grid: np.ndarray = (
            _accel.distance_transform_edt(~_um)
            * self.cell_size
        )
        self.dist_urban_grid = dist_grid

        d_elim = URBAN_DIST_ELIMINATORY
        d_pen = URBAN_DIST_PENALTY
        d_full = URBAN_DIST_FULL
        floor = URBAN_PROXIMITY_FLOOR

        score = np.zeros_like(dist_grid, dtype=np.float32)

        # < d_elim (dans masque ou très proche) → 0.0 (déjà zeros_like)

        # d_elim..d_pen → rampe [FLOOR..0.6]
        mask = (dist_grid >= d_elim) & (dist_grid < d_pen)
        if np.any(mask):
            t = (dist_grid[mask] - d_elim) / (d_pen - d_elim)
            score[mask] = floor + (0.6 - floor) * t

        # d_pen..d_full → rampe [0.6..1.0]
        mask = (dist_grid >= d_pen) & (dist_grid < d_full)
        if np.any(mask):
            t = (dist_grid[mask] - d_pen) / (d_full - d_pen)
            score[mask] = 0.6 + 0.4 * t

        # > d_full → 1.0
        score[dist_grid >= d_full] = 1.0

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["urban_proximity"] = score
        self.score_confidence["urban_proximity"] = 0.8

        _not_um = ~_um
        _dist_min: float = (
            float(np.min(dist_grid[_not_um]))
            if np.any(_not_um)
            else 0.0
        )
        logger.info(
            "   Distance urbain : %.0f–%.0fm, %d cellules en zone urbaine",
            _dist_min,
            float(np.max(dist_grid)),
            int(_um.sum()),
        )
        self._log_score_stats("urban_proximity", score)
        return self

    def apply_water_mask(self) -> GridBuilder:
        """
        Applique le masque des plans d'eau — Fix #24.

        Met les scores à 0 pour les cellules en eau.
        """
        if self.water_mask is not None and self.water_mask.any():
            n = int(self.water_mask.sum())
            for name in self.scores:
                self.scores[name][self.water_mask] = 0.0
            logger.info(
                "✅ Masque plans d'eau : %d cellules → score 0", n
            )
        else:
            logger.debug("   Pas de masque plans d'eau")
        return self

    def apply_landcover_mask(
        self, landcover_data: dict[str, Any] | None
    ) -> GridBuilder:
        """
        Applique le masque landcover — v2.3.4.

        Fix #13  : sauvegarde _raw_tree_species AVANT modulation.
        Fix #36  : forest floor green_clip pour cellules BD Forêt v2.
        Fix #37  : tree_species retiré de _VEGETATION_CRITERIA (plus modulé).
        Fix #39  : dist_water plancher en forêt (humidité sol forestier).
        """
        if landcover_data is None:
            logger.info("⚠️ Pas de données landcover")
            return self

        urban_mask_lc: np.ndarray = landcover_data["urban_mask"]
        green_score: np.ndarray = landcover_data["green_score"]

        # ── Redimensionner si nécessaire ──
        if urban_mask_lc.shape != (self.ny, self.nx):
            urban_mask_lc = (
                np.asarray(
                    zoom(
                        urban_mask_lc.astype(np.float32),
                        (
                            self.ny / urban_mask_lc.shape[0],
                            self.nx / urban_mask_lc.shape[1],
                        ),
                        order=0,
                    )
                )
                > 0.5
            )
            green_score = np.asarray(
                zoom(
                    green_score.astype(np.float32),
                    (
                        self.ny / green_score.shape[0],
                        self.nx / green_score.shape[1],
                    ),
                    order=1,
                )
            ).astype(np.float32)

        # ── Fusionner masque urbain ──
        _um_val = self.urban_mask
        if _um_val is not None:
            _um: np.ndarray = _um_val
            old_count = int(_um.sum())
            merged: np.ndarray = _um | urban_mask_lc
            self.urban_mask = merged
            new_count = int(merged.sum())
            added = new_count - old_count
            logger.info(
                "   🔗 Fusion masques : %s (vecteur) + %s (couleur) = %s",
                f"{old_count:,}",
                f"{added:,}",
                f"{new_count:,}",
            )
        else:
            self.urban_mask = urban_mask_lc

        # ── Fix #13 : Sauvegarder tree_species brut AVANT modulation ──
        if "tree_species" in self.scores:
            self._raw_tree_species = self.scores["tree_species"].copy()

        # ── Moduler les scores végétation par green_score ──
        green_clip = np.clip(green_score, 0.0, 1.0)

        # ── FIX #36 v2.3.3 : forest floor ───────────────────────
        _FILL_NO_FOREST = 0.05
        _raw_ts: np.ndarray | None = getattr(
            self, "_raw_tree_species", None
        )  # P4
        if _raw_ts is not None:
            forest_mask: np.ndarray = _raw_ts > _FILL_NO_FOREST
            n_forest = int(np.sum(forest_mask))

            # Green floor pour critères végétation
            below_floor = forest_mask & (green_clip < LANDCOVER_FOREST_FLOOR)
            n_floored = int(np.sum(below_floor))
            green_clip[below_floor] = LANDCOVER_FOREST_FLOOR
            logger.info(
                "   🌲 Forest floor : %s cellules forêt, %s rehaussées "
                "(green≥%.2f)",
                f"{n_forest:,}",
                f"{n_floored:,}",
                LANDCOVER_FOREST_FLOOR,
            )

            # ── FIX #39 v2.3.4 : dist_water plancher en forêt ───
            if "dist_water" in self.scores:
                dw = self.scores["dist_water"]
                dw_below = forest_mask & (dw < DIST_WATER_FOREST_FLOOR)
                n_dw = int(np.sum(dw_below))
                self.scores["dist_water"] = np.where(
                    forest_mask,
                    np.maximum(dw, DIST_WATER_FOREST_FLOOR),
                    dw,
                ).astype(np.float32)
                logger.info(
                    "   💧 dist_water forest floor : %s cellules forêt "
                    "rehaussées (≥%.2f)",
                    f"{n_dw:,}",
                    DIST_WATER_FOREST_FLOOR,
                )
            # ── FIN FIX #39 ─────────────────────────────────────
        else:
            logger.warning(
                "   ⚠️ _raw_tree_species absent — forest floors ignorés"
            )
        # ── FIN FIX #36 ─────────────────────────────────────────

        # ── Fix #37 v2.3.4 : tree_species N'EST PLUS dans ──
        # ── _VEGETATION_CRITERIA → pas de double pénalisation ──
        n_modulated = 0
        for criterion in _VEGETATION_CRITERIA:
            if criterion in self.scores:
                old_score = self.scores[criterion]
                self.scores[criterion] = (old_score * green_clip).astype(
                    np.float32
                )
                n_modulated += 1
                delta = float(
                    np.mean(old_score) - np.mean(self.scores[criterion])
                )
                logger.debug(
                    "   🌿 %s modulé par green_score (Δmoy=%.3f)",
                    criterion,
                    delta,
                )

        # ── Stocker green_clip AVEC floor (pas raw green_score) ──
        self._landcover_green_score = green_clip

        # ── Stats finales ──
        _urban_final_val = self.urban_mask
        assert _urban_final_val is not None
        _urban_final: np.ndarray = _urban_final_val

        urban_cells = int(_urban_final.sum())
        urban_pct = urban_cells / _urban_final.size * 100
        green_cells = int((green_clip > 0.3).sum())
        green_pct = green_cells / green_clip.size * 100

        logger.info("✅ Landcover appliqué (v2.3.4) :")
        logger.info(
            "   🏘️ Urbain total : %s cellules (%.1f%%)",
            f"{urban_cells:,}",
            urban_pct,
        )
        logger.info(
            "   🌿 Végétation : %s cellules (%.1f%%)",
            f"{green_cells:,}",
            green_pct,
        )
        logger.info(
            "   📊 %d critères modulés par green_score "
            "(tree_species exclu — fix #37)",
            n_modulated,
        )
        return self

    # ═══════════════════════════════════════════════════════
    #  VALIDATION & DIAGNOSTICS
    # ═══════════════════════════════════════════════════════

    def validate_scores(self) -> bool:
        """
        Vérifie que tous les critères de config.WEIGHTS ont un score calculé,
        et que tous les scores sont dans [0, 1].
        """
        ok = True

        missing = set(config.WEIGHTS.keys()) - set(self.scores.keys())
        if missing:
            logger.error(
                "❌ Scores manquants pour les critères : %s",
                ", ".join(sorted(missing)),
            )
            ok = False

        for name, score in self.scores.items():
            s_min = float(np.nanmin(score))
            s_max = float(np.nanmax(score))
            if s_min < -0.001 or s_max > 1.001:
                logger.error(
                    "❌ Score '%s' hors [0,1] : min=%.4f, max=%.4f",
                    name,
                    s_min,
                    s_max,
                )
                ok = False

            if score.shape != (self.ny, self.nx):
                logger.error(
                    "❌ Score '%s' mauvaise forme : %s ≠ (%d, %d)",
                    name,
                    score.shape,
                    self.ny,
                    self.nx,
                )
                ok = False

        extra = set(self.scores.keys()) - set(config.WEIGHTS.keys())
        if extra:
            logger.info(
                "ℹ️ Scores calculés mais hors WEIGHTS : %s",
                ", ".join(sorted(extra)),
            )

        if ok:
            logger.info(
                "✅ Validation scores : %d/%d critères OK",
                len(config.WEIGHTS),
                len(config.WEIGHTS),
            )

        return ok

    def get_score_summary(self) -> dict[str, dict[str, float]]:
        """Retourne un résumé structuré de tous les scores."""
        summary: dict[str, dict[str, float]] = {}
        for name, score in self.scores.items():
            valid = (
                score[~self.nodata_mask]
                if (self.nodata_mask is not None and self.nodata_mask.any())
                else score
            )

            summary[name] = {
                "min": round(float(np.nanmin(valid)), 4),
                "max": round(float(np.nanmax(valid)), 4),
                "mean": round(float(np.nanmean(valid)), 4),
                "median": round(float(np.nanmedian(valid)), 4),
                "std": round(float(np.nanstd(valid)), 4),
                "pct_zero": round(
                    float(np.sum(valid == 0)) / valid.size * 100, 2
                ),
                "pct_high": round(
                    float(np.sum(valid >= 0.7)) / valid.size * 100, 2
                ),
                "weight": float(config.WEIGHTS.get(name, 0.0)),
                "confidence": self.score_confidence.get(name, 0.5),
            }

        return summary

    def get_cell_info(self, ix: int, iy: int) -> dict[str, Any]:
        """Retourne toutes les informations d'une cellule donnée."""
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return {"error": "hors grille"}

        info: dict[str, Any] = {
            "ix": ix,
            "iy": iy,
            "x_l93": float(self.x_coords[ix]),
            "y_l93": float(self.y_coords[iy]),
        }

        if self.altitude is not None:
            info["altitude"] = float(self.altitude[iy, ix])
        if self.slope is not None:
            info["slope"] = float(self.slope[iy, ix])
        if self.aspect is not None:
            info["aspect"] = float(self.aspect[iy, ix])
        if self.roughness is not None:
            info["roughness"] = float(self.roughness[iy, ix])
        if self.dist_water_grid is not None:
            info["dist_water_m"] = float(self.dist_water_grid[iy, ix])
        if self.nodata_mask is not None:
            info["is_nodata"] = bool(self.nodata_mask[iy, ix])
        if self.urban_mask is not None:
            info["is_urban"] = bool(self.urban_mask[iy, ix])
        if self.water_mask is not None:
            info["is_water"] = bool(self.water_mask[iy, ix])

        for name, score in self.scores.items():
            info[f"score_{name}"] = round(float(score[iy, ix]), 4)

        return info