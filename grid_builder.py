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
    distance_transform_edt,
    uniform_filter,
    zoom,
)

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
    DIST_WATER_FOREST_FLOOR
)

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
        indices: np.ndarray = np.asarray(
            distance_transform_edt(
                nan_mask, return_distances=False, return_indices=True
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
        Gère les NaN proprement (NoData du DEM).
        """
        dem = dem_data["data"].astype(np.float32)

        # ── Redimensionner si nécessaire ──
        if dem.shape != (self.ny, self.nx):
            logger.debug(
                "   Redimensionnement MNT : %s → (%d, %d)",
                dem.shape,
                self.ny,
                self.nx,
            )
            dem = self._zoom_dem(dem)

        self.altitude = dem.copy()

        # ── Masque NoData ──
        self.nodata_mask = np.isnan(dem)
        n_nodata = int(self.nodata_mask.sum())
        if n_nodata > 0:
            logger.debug(
                "   %d cellules NoData (%.1f%%)",
                n_nodata,
                n_nodata / dem.size * 100,
            )

        # ── Pente et aspect (sur DEM sans NaN) ──
        dem_filled = self._fill_nan_dem(dem)
        self._compute_slope_aspect(dem_filled)

        # P3 : narrowing pour Pylance
        _slope = self.slope
        assert _slope is not None, "slope not set after _compute_slope_aspect"
        _aspect = self.aspect
        assert _aspect is not None, "aspect not set after _compute_slope_aspect"

        # ── Rugosité ──
        self._compute_roughness()
        _roughness = self.roughness
        assert _roughness is not None, "roughness not set after _compute_roughness"

        # ── Restaurer NaN dans les dérivées ──
        if self.nodata_mask.any():
            _slope[self.nodata_mask] = np.nan
            _aspect[self.nodata_mask] = np.nan
            _roughness[self.nodata_mask] = np.nan

        self._terrain_computed = True
        self._log_terrain_stats()

    # ── TWI — Topographic Wetness Index (fix #46 v2.3.5) ──
        self._twi = self._compute_twi(dem_filled, self.cell_size)
        logger.info(
            "   TWI       : %.1f–%.1f (moy=%.1f)",
            float(np.nanmin(self._twi)),
            float(np.nanmax(self._twi)),
            float(np.nanmean(self._twi)),
        )

        return self

    def _zoom_dem(self, dem: np.ndarray) -> np.ndarray:
        """Redimensionne le DEM en gérant les NaN."""
        nan_mask = np.isnan(dem)
        zy = self.ny / dem.shape[0]
        zx = self.nx / dem.shape[1]
        if nan_mask.any():
            dem_filled = self._fill_nan_dem(dem)
            dem_zoomed: np.ndarray = np.asarray(
                zoom(dem_filled, (zy, zx), order=1)
            ).astype(np.float32)
            nan_zoomed: np.ndarray = (
                np.asarray(
                    zoom(nan_mask.astype(np.float32), (zy, zx), order=0)
                )
                > 0.5
            )
            dem_zoomed[nan_zoomed] = np.nan
            return dem_zoomed
        return np.asarray(zoom(dem, (zy, zx), order=1)).astype(np.float32)

    def _compute_slope_aspect(self, dem_filled: np.ndarray) -> None:
        """Calcule pente et aspect depuis un DEM sans NaN."""
        dy, dx = np.gradient(dem_filled, self.cell_size)
        self.slope = np.degrees(
            np.arctan(np.sqrt(dx**2 + dy**2))
        ).astype(np.float32)
        self.aspect = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(
            np.float32
        )

    def _compute_roughness(self) -> None:
        """Rugosité = écart-type local de la pente."""
        _slope = self.slope
        assert _slope is not None, "slope required for roughness"
        w = ROUGHNESS_WINDOW
        slope_mean: np.ndarray = np.asarray(uniform_filter(_slope, size=w))
        slope_sq_mean: np.ndarray = np.asarray(
            uniform_filter(_slope**2, size=w)
        )
        self.roughness = np.sqrt(
            np.maximum(slope_sq_mean - slope_mean**2, 0)
        ).astype(np.float32)

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

    @staticmethod
    def _compute_twi(
        dem: np.ndarray,
        cell_size: float,
    ) -> np.ndarray:
        """
        Calcul du Topographic Wetness Index — TWI = ln(a / tan(β)).

        Utilise un algorithme D8 simplifié pour l'aire drainée.
        L'aire drainée spécifique a = A_drainée / largeur_pixel.

        Fix #46 v2.3.5.
        """
        ny, nx = dem.shape
        cell_area = cell_size * cell_size

        # ── 1. Pente locale β (radians), floor pour éviter div/0 ──
        dy, dx = np.gradient(dem, cell_size)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        # Floor : pente minimale 0.1° pour éviter ln(inf) en terrain plat
        slope_rad = np.maximum(slope_rad, np.radians(0.1))

        # ── 2. Direction d'écoulement D8 ──
        # 8 voisins : (drow, dcol) dans l'ordre conventionnel D8
        d8_offsets: tuple[tuple[int, int], ...] = (
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1),
        )
        # Distances (1.0 pour cardinal, √2 pour diagonal)
        d8_dist: np.ndarray = np.array(
            [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414],
            dtype=np.float32,
        ) * cell_size

        # Direction : index du voisin avec la plus forte pente descendante
        flow_dir = np.full((ny, nx), -1, dtype=np.int8)

        for idx, (dr, dc) in enumerate(d8_offsets):
            # Coordonnées du voisin
            r_from = max(0, -dr)
            r_to = ny - max(0, dr)
            c_from = max(0, -dc)
            c_to = nx - max(0, dc)

            r_nb_from = max(0, dr)
            r_nb_to = ny + min(0, dr) if dr < 0 else ny
            c_nb_from = max(0, dc)
            c_nb_to = nx + min(0, dc) if dc < 0 else nx

            # Pente vers ce voisin
            drop = (
                dem[r_from:r_to, c_from:c_to]
                - dem[r_nb_from:r_nb_to, c_nb_from:c_nb_to]
            )
            slope_to_nb = drop / d8_dist[idx]

            # Mettre à jour si c'est la pente max
            current_best = np.full((r_to - r_from, c_to - c_from), -np.inf)

            # Recalculer le meilleur courant
            for prev_idx, (pdr, pdc) in enumerate(d8_offsets):
                if prev_idx >= idx:
                    break
                pr_from = max(0, -pdr)
                pr_to = ny - max(0, pdr)
                pc_from = max(0, -pdc)
                pc_to = nx - max(0, pdc)
                pr_nb_from = max(0, pdr)
                pr_nb_to = ny + min(0, pdr) if pdr < 0 else ny
                pc_nb_from = max(0, pdc)
                pc_nb_to = nx + min(0, pdc) if pdc < 0 else nx

                prev_drop = (
                    dem[pr_from:pr_to, pc_from:pc_to]
                    - dem[pr_nb_from:pr_nb_to, pc_nb_from:pc_nb_to]
                )
                prev_slope = prev_drop / d8_dist[prev_idx]

                # Intersect avec la fenêtre actuelle
                # Trop complexe avec les fenêtres variables → approche alternative

            # Simplification : boucle pixel serait trop lente.
            # → Approche vectorisée en 2 passes.
            pass

        # ── Approche alternative optimisée : D8 vectorisé ──
        # Réinitialisation
        flow_dir = np.full((ny, nx), -1, dtype=np.int8)
        max_slope = np.full((ny, nx), 0.0, dtype=np.float32)

        # Padding du DEM pour accès voisins sans bounds check
        dem_pad = np.pad(dem, 1, mode="edge")

        for idx, (dr, dc) in enumerate(d8_offsets):
            # Voisin dans le DEM paddé
            nb = dem_pad[1 + dr : ny + 1 + dr, 1 + dc : nx + 1 + dc]
            drop = dem - nb
            slope_nb = drop / d8_dist[idx]

            # Mise à jour : garder la direction avec la plus forte pente
            better = slope_nb > max_slope
            flow_dir[better] = idx
            max_slope[better] = slope_nb[better]

        # Cellules sans écoulement (puits) : marquer comme -1
        flow_dir[max_slope <= 0] = -1

        # ── 3. Aire drainée par accumulation D8 ──
        # Tri topologique par altitude décroissante
        flat_idx = np.argsort(dem.ravel())[::-1]  # du plus haut au plus bas
        acc = np.ones((ny, nx), dtype=np.float64) * cell_area

        for pixel in flat_idx:
            r = pixel // nx
            c = pixel % nx
            d = flow_dir[r, c]
            if d < 0:
                continue
            dr, dc = d8_offsets[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < ny and 0 <= nc < nx:
                acc[nr, nc] += acc[r, c]

        # ── 4. Aire drainée spécifique ──
        specific_area = acc / cell_size  # m²/m = m

        # ── 5. TWI = ln(a / tan(β)) ──
        tan_beta = np.tan(slope_rad)
        # Clip pour éviter log(0) ou log(négatif)
        ratio = np.maximum(specific_area / tan_beta, 1e-6)
        twi: np.ndarray = np.log(ratio).astype(np.float32)

        return twi

    # ═══════════════════════════════════════════════════════
    #  SCORES TERRAIN
    # ═══════════════════════════════════════════════════════

    def score_altitude(self) -> GridBuilder:
        """
        Score d'altitude — v2.2.0 : optimal 200-600m, bonus à 350m.
        """
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

        # --- Plateau optimal [opt_lo, opt_hi] → 1.0 ---
        m_opt = valid & (twi_arr >= opt_lo) & (twi_arr <= opt_hi)
        score[m_opt] = 1.0

        # --- Zone sèche : dry_limit → opt_lo ---
        # Concave (exposant < 1) : remonte vite depuis dry_floor
        m_dry_mid = valid & (twi_arr >= dry_limit) & (twi_arr < opt_lo)
        if np.any(m_dry_mid):
            t = (twi_arr[m_dry_mid] - dry_limit) / (opt_lo - dry_limit)
            score[m_dry_mid] = dry_floor + (1.0 - dry_floor) * t**0.6

        # --- Zone très sèche : < dry_limit → dry_floor
        m_very_dry = valid & (twi_arr < dry_limit)
        score[m_very_dry] = dry_floor

        # --- Zone humide : opt_hi → wet_limit ---
        # Convexe (exposant > 1) : descend doucement puis accélère
        m_wet_mid = valid & (twi_arr > opt_hi) & (twi_arr <= wet_limit)
        if np.any(m_wet_mid):
            t = (twi_arr[m_wet_mid] - opt_hi) / (wet_limit - opt_hi)
            score[m_wet_mid] = 1.0 - (1.0 - wet_floor) * t**1.5

        # --- Zone très humide : wet_limit → waterlog ---
        m_wet_high = valid & (twi_arr > wet_limit) & (twi_arr <= waterlog)
        if np.any(m_wet_high):
            t = (twi_arr[m_wet_high] - wet_limit) / (waterlog - wet_limit)
            score[m_wet_high] = wet_floor * (1.0 - t)

        # --- Engorgement : > waterlog → 0.0 ---
        m_waterlog = valid & (twi_arr > waterlog)
        score[m_waterlog] = 0.0

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["twi"] = score
        self.score_confidence["twi"] = 0.85
        self._log_score_stats("twi", score)

        # --- Stats distribution pour diagnostic ---
        if valid.any():
            twi_valid = twi_arr[valid]
            logger.info(
                "  TWI distribution — min=%.1f  med=%.1f  max=%.1f  "
                "dry(<%.0f)=%d  optimal(%s–%s)=%d  wet(>%.0f)=%d  waterlog(>%.0f)=%d",
                float(np.nanmin(twi_valid)),
                float(np.nanmedian(twi_valid)),
                float(np.nanmax(twi_valid)),
                dry_limit,
                int(np.sum(m_very_dry)),
                opt_lo,
                opt_hi,
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
            np.asarray(distance_transform_edt(~self.water_mask)).astype(
                np.float32
            )
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
        self, forest_gdf: gpd.GeoDataFrame | None
    ) -> GridBuilder:
        """Score des essences forestières."""
        self._require_terrain()

        if forest_gdf is None or forest_gdf.empty:
            logger.warning(
                "⚠️ Pas de données forêt → score %.2f", FILL_NO_FOREST
            )
            self.scores["tree_species"] = np.full(
                (self.ny, self.nx), FILL_NO_FOREST, dtype=np.float32
            )
            self.forest_mask = np.zeros((self.ny, self.nx), dtype=bool)
            self.score_confidence["tree_species"] = 0.0
            return self

        forest_gdf = self._ensure_l93(forest_gdf)

        # ── Résoudre les scores si pas pré-calculés ──
        gdf = forest_gdf.copy()

        if "tree_score" not in gdf.columns:
            if "essence_canonical" in gdf.columns:
                gdf["tree_score"] = gdf["essence_canonical"].apply(
                    lambda c: TREE_SCORES.get(c, TREE_SCORES["unknown"])
                )
            elif "ESSENCE" in gdf.columns:
                gdf["tree_score"] = gdf["ESSENCE"].apply(get_tree_score)
            else:
                gdf["tree_score"] = self._score_from_any_column(
                    gdf,
                    get_tree_score,
                    [
                        "ESSENCE",
                        "TFV",
                        "TFVF",
                        "essence",
                        "tfv",
                        "NOM_TYPN",
                        "LIB_TFV",
                        "libelle",
                    ],
                )

        # ── Rasteriser avec max ──
        shapes_scores: list[tuple[Any, float]] = [
            (geom, float(sc))
            for geom, sc in zip(gdf.geometry, gdf["tree_score"])
            if geom is not None and not geom.is_empty
        ]

        tree_raster: np.ndarray = self._rasterize_max(
            shapes_scores,
            fill=FILL_NO_FOREST,
            all_touched=True,
        )
        self.tree_raster = tree_raster

        # Masque forêt (P5 : np.asarray autour de rasterize)
        forest_presence: list[tuple[Any, int]] = [
            (geom, 1)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]
        if forest_presence:
            fm: np.ndarray = np.asarray(rasterize(
                forest_presence,
                out_shape=(self.ny, self.nx),
                transform=self.transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            ))
            self.forest_mask = fm.astype(bool)
        else:
            self.forest_mask = np.zeros((self.ny, self.nx), dtype=bool)

        score = self._apply_nodata(tree_raster.copy())
        self.scores["tree_species"] = score

        # ── Fix #17 : Masque éliminatoire explicite (essences) ──────
        elim_species: set[str] = set(
            getattr(config, "ELIMINATORY_SPECIES", set())
        )
        if elim_species and "essence_canonical" in gdf.columns:
            elim_shapes: list[tuple[Any, int]] = [
                (geom, 1)
                for geom, ess in zip(gdf.geometry, gdf["essence_canonical"])
                if geom is not None
                and not geom.is_empty
                and str(ess) in elim_species
            ]
            if elim_shapes:
                self.eliminatory_species_mask = np.asarray(rasterize(
                    elim_shapes,
                    out_shape=(self.ny, self.nx),
                    transform=self.transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                )).astype(bool)
                logger.info(
                    "   Masque éliminatoire essences : %d cellules (%s)",
                    int(self.eliminatory_species_mask.sum()),
                    ", ".join(sorted(elim_species)),
                )
        # ── Fix #26 : Raster catégoriel pour hotspot enrichment ──────
        if "essence_canonical" in gdf.columns:
            _cats = sorted(
                c for c in gdf["essence_canonical"].dropna().unique()
                if isinstance(c, str)
            )
            _cat_to_int: dict[str, int] = {
                c: i + 1 for i, c in enumerate(_cats)
            }
            self._int_to_essence_canonical: dict[int, str] = {
                v: k for k, v in _cat_to_int.items()
            }
            _cat_shapes: list[tuple[Any, int]] = [
                (geom, _cat_to_int[str(ess)])
                for geom, ess in zip(
                    gdf.geometry, gdf["essence_canonical"], strict=False,
                )
                if geom is not None
                and not geom.is_empty
                and str(ess) in _cat_to_int
            ]
            if _cat_shapes:
                self._raster_essence_canonical: np.ndarray = np.asarray(
                    rasterize(
                        _cat_shapes,
                        out_shape=(self.ny, self.nx),
                        transform=self.transform,
                        fill=0,
                        dtype=np.int16,
                        all_touched=True,
                    )
                )
                logger.debug(
                    "   Raster catégoriel essences : %d catégories",
                    len(_cats),
                )
        # Confiance
        source = (
            str(gdf["source"].iloc[0]) if "source" in gdf.columns else "unknown"
        )

        if "essence_canonical" in gdf.columns:
            n_unknown = int((gdf["essence_canonical"] == "unknown").sum())
        else:
            n_unknown = len(gdf)
        pct_unknown = n_unknown / len(gdf) * 100 if len(gdf) > 0 else 100

        self.score_confidence["tree_species"] = {
            "wfs_ign": 0.9,
            "osm": 0.5,
            "synthetic": 0.3,
            "file": 0.8,
        }.get(source, 0.5)

        if pct_unknown > 50:
            self.score_confidence["tree_species"] *= 0.5
            logger.warning(
                "⚠️ %.0f%% des polygones forestiers sans essence connue",
                pct_unknown,
            )

        _fm_val = self.forest_mask
        _fm_sum = float(_fm_val.sum()) if _fm_val is not None else 0.0
        _fm_size = _fm_val.size if _fm_val is not None else 1
        logger.info(
            "   Forêt : %d polygones, %.0f%% de couverture",
            len(gdf),
            _fm_sum / _fm_size * 100,
        )
        self._log_score_stats("tree_species", score)
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
        self, geology_gdf: gpd.GeoDataFrame | None
    ) -> GridBuilder:
        """Score géologique."""
        self._require_terrain()

        if geology_gdf is None or geology_gdf.empty:
            logger.warning(
                "⚠️ Pas de données géologie → score %.2f", FILL_NO_GEOLOGY
            )
            self.scores["geology"] = np.full(
                (self.ny, self.nx), FILL_NO_GEOLOGY, dtype=np.float32
            )
            self.score_confidence["geology"] = 0.0
            return self

        geology_gdf = self._ensure_l93(geology_gdf)
        gdf = geology_gdf.copy()

        # ── Résoudre les scores si pas pré-calculés ──
        if "geology_score" not in gdf.columns:
            if "geology_canonical" in gdf.columns:
                gdf["geology_score"] = gdf["geology_canonical"].apply(
                    lambda c: GEOLOGY_SCORES.get(
                        c, GEOLOGY_SCORES["unknown"]
                    )
                )
            elif "LITHO" in gdf.columns:
                gdf["geology_score"] = gdf["LITHO"].apply(get_geology_score)
            else:
                gdf["geology_score"] = self._score_geology_from_any_column(gdf)

        # ── Rasteriser avec max ──
        shapes_scores: list[tuple[Any, float]] = [
            (geom, float(sc))
            for geom, sc in zip(gdf.geometry, gdf["geology_score"])
            if geom is not None and not geom.is_empty
        ]

        geology_raster: np.ndarray = self._rasterize_max(
            shapes_scores,
            fill=FILL_NO_GEOLOGY,
            all_touched=True,
        )
        self.geology_raster = geology_raster

        score = self._apply_nodata(geology_raster.copy())
        self.scores["geology"] = score

        # ── Fix #17 : Masque éliminatoire explicite (géologie) ──────
        elim_geology: set[str] = set(
            getattr(config, "ELIMINATORY_GEOLOGY", set())
        )
        if elim_geology and "geology_canonical" in gdf.columns:
            elim_shapes: list[tuple[Any, int]] = [
                (geom, 1)
                for geom, geo in zip(gdf.geometry, gdf["geology_canonical"])
                if geom is not None
                and not geom.is_empty
                and str(geo) in elim_geology
            ]
            if elim_shapes:
                self.eliminatory_geology_mask = np.asarray(rasterize(
                    elim_shapes,
                    out_shape=(self.ny, self.nx),
                    transform=self.transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                )).astype(bool)
                logger.info(
                    "   Masque éliminatoire géologie : %d cellules (%s)",
                    int(self.eliminatory_geology_mask.sum()),
                    ", ".join(sorted(elim_geology)),
                )
        # ── Fix #26 : Raster catégoriel pour hotspot enrichment ──────
        if "geology_canonical" in gdf.columns:
            _gcats = sorted(
                c for c in gdf["geology_canonical"].dropna().unique()
                if isinstance(c, str)
            )
            _gcat_to_int: dict[str, int] = {
                c: i + 1 for i, c in enumerate(_gcats)
            }
            self._int_to_geology_canonical: dict[int, str] = {
                v: k for k, v in _gcat_to_int.items()
            }
            _gcat_shapes: list[tuple[Any, int]] = [
                (geom, _gcat_to_int[str(geo)])
                for geom, geo in zip(
                    gdf.geometry, gdf["geology_canonical"], strict=False,
                )
                if geom is not None
                and not geom.is_empty
                and str(geo) in _gcat_to_int
            ]
            if _gcat_shapes:
                self._raster_geology_canonical: np.ndarray = np.asarray(
                    rasterize(
                        _gcat_shapes,
                        out_shape=(self.ny, self.nx),
                        transform=self.transform,
                        fill=0,
                        dtype=np.int16,
                        all_touched=True,
                    )
                )
                logger.debug(
                    "   Raster catégoriel géologie : %d catégories",
                    len(_gcats),
                )

        # Confiance
        source = (
            str(gdf["source"].iloc[0]) if "source" in gdf.columns else "unknown"
        )
        self.score_confidence["geology"] = {
            "wfs_brgm": 0.9,
            "osm": 0.4,
            "synthetic": 0.3,
            "file": 0.8,
        }.get(source, 0.5)

        logger.info("   Géologie : %d polygones", len(gdf))
        if "geology_canonical" in gdf.columns:
            for cat, cnt in (
                gdf["geology_canonical"].value_counts().head(5).items()
            ):
                logger.debug("     • %s: %d", cat, cnt)
        self._log_score_stats("geology", score)
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
        """Score d'ouverture de la canopée."""
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
            self.score_confidence["canopy_openness"] = 0.4
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
        Estime l'ouverture de canopée — v2.3.0.

        Fix #7  : terrain ouvert → _CANOPY_OPEN_FIELD (0.10)
        Fix #14 : transition continue 0.70 → 0.10 (plus de saut à 30m)

        Logique :
          - Intérieur forêt dense (>20m des lisières) : 0.55
          - Lisière forêt (<30m du bord, côté intérieur) : 0.90→0.55
          - Juste hors forêt (<30m du bord) : 0.70→0.10 (transition continue)
          - Terrain ouvert (>30m de toute forêt) : 0.10
        """
        _fm_val = self.forest_mask
        assert _fm_val is not None, "forest_mask is None"
        _fm: np.ndarray = _fm_val

        if not _fm.any():
            return np.full(
                (self.ny, self.nx), _CANOPY_OPEN_FIELD, dtype=np.float32
            )

        # Lisière = bordure intérieure de la forêt
        eroded: np.ndarray = np.asarray(
            binary_erosion(
                _fm,
                iterations=max(1, int(20.0 / self.cell_size)),
            )
        )
        edge_mask = _fm & ~eroded

        # Distance au pixel forêt le plus proche (pour hors-forêt)
        dist_to_forest: np.ndarray = (
            np.asarray(distance_transform_edt(~_fm)).astype(np.float32)
            * self.cell_size
        )

        # Distance à la lisière (pour intérieur forêt)
        dist_to_edge: np.ndarray = (
            np.asarray(distance_transform_edt(~edge_mask)).astype(np.float32)
            * self.cell_size
        )

        # ── Défaut : terrain ouvert ──
        score = np.full(
            (self.ny, self.nx), _CANOPY_OPEN_FIELD, dtype=np.float32
        )

        # ── Intérieur forêt dense ──
        score[_fm] = _CANOPY_FOREST_INTERIOR

        # ── Lisière intérieure (<30m du bord) : 0.90 → 0.55 ──
        near_edge_in = _fm & (dist_to_edge < 30)
        if near_edge_in.any():
            t = dist_to_edge[near_edge_in] / 30.0
            score[near_edge_in] = 0.90 - (0.90 - _CANOPY_FOREST_INTERIOR) * t

        # ── Fix #14 : transition extérieure continue 0.70 → _CANOPY_OPEN_FIELD ──
        near_forest_out = (~_fm) & (dist_to_forest < 30)
        if near_forest_out.any():
            t = dist_to_forest[near_forest_out] / 30.0
            score[near_forest_out] = (
                0.70 - (0.70 - _CANOPY_OPEN_FIELD) * t
            )

        return score

    def score_ground_cover(self) -> GridBuilder:
        """
        Score de couverture au sol — v2.3.0.

        Fix #8  : bonus humidité uniquement en forêt, hors forêt → 0.20.
        Fix #16 : deep_forest ne pénalise plus les zones humides.
        """
        self._require_terrain()
        _alt = self.altitude
        assert _alt is not None

        _fm_val = self.forest_mask
        has_forest = _fm_val is not None and _fm_val.any()

        # ── Défaut selon présence forêt ──
        if has_forest:
            assert _fm_val is not None
            _fm: np.ndarray = _fm_val
            # En forêt → 0.50 (litière mixte), hors forêt → 0.20 (herbe)
            score = np.where(
                _fm, 0.50, 0.20
            ).astype(np.float32)
        else:
            score = np.full((self.ny, self.nx), 0.20, dtype=np.float32)

        # ── Bonus humidité : uniquement EN FORÊT près de l'eau ──
        # On garde un masque des cellules humides pour le fix #16
        ideal_humid = np.zeros((self.ny, self.nx), dtype=bool)

        if self.dist_water_grid is not None and has_forest:
            assert _fm_val is not None
            _fm2: np.ndarray = _fm_val

            # Trop humide (inondable)
            too_wet = (self.dist_water_grid < 10) & (_alt < 300) & _fm2
            score[too_wet] = 0.25

            # Idéal : forêt + humidité résiduelle (15-80m de l'eau)
            ideal_humid = (
                (self.dist_water_grid >= 15)
                & (self.dist_water_grid <= 80)
                & (_alt < 600)
                & _fm2
            )
            score[ideal_humid] = 0.85

        # ── Fix #16 : Forêt profonde SÈCHE → malus (ne pénalise PAS la zone humide) ──
        if has_forest:
            assert _fm_val is not None
            _fm3: np.ndarray = _fm_val
            deep_forest: np.ndarray = np.asarray(
                binary_erosion(
                    _fm3,
                    iterations=max(1, int(40.0 / self.cell_size)),
                )
            )
            # Seulement les cellules deep forest ET PAS en zone humide idéale
            deep_forest_dry = deep_forest & ~ideal_humid
            score[deep_forest_dry] = np.minimum(
                score[deep_forest_dry], 0.40
            )

        # ── Forte pente = sol instable ──
        _slope_val = self.slope
        if _slope_val is not None:
            _slope: np.ndarray = _slope_val
            steep = _slope >= STEEP_SLOPE_THRESHOLD
            score[steep] = np.minimum(score[steep], 0.15)

        score = self._apply_nodata(np.clip(score, 0, 1))
        self.scores["ground_cover"] = score
        self.score_confidence["ground_cover"] = 0.2
        self._log_score_stats("ground_cover", score)
        return self

    def score_disturbance(
        self, disturbance_data: np.ndarray | None = None
    ) -> GridBuilder:
        """
        Score de perturbation du sol.

        Note : le bonus proximité urbaine nécessite que apply_urban_mask()
        ait été appelé AVANT cette méthode (Fix #15 — corrigé dans main.py).
        """
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
                np.asarray(distance_transform_edt(~edge_mask)).astype(
                    np.float32
                )
                * self.cell_size
            )

            score = np.full((self.ny, self.nx), 0.3, dtype=np.float32)
            near_edge = dist_to_edge < 15
            if near_edge.any():
                t = dist_to_edge[near_edge] / 15.0
                score[near_edge] = 0.7 - 0.4 * t

            # ── Bonus proximité urbaine (Fix #15 : fonctionne si
            #    apply_urban_mask() a été appelé avant) ──
            _um_val = self.urban_mask
            if _um_val is not None and _um_val.any():
                _um: np.ndarray = _um_val
                urban_edge: np.ndarray = (
                    np.asarray(
                        distance_transform_edt(~_um)
                    ).astype(np.float32)
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
                    "   Disturbance : urban_mask=None — bonus urbain ignoré. "
                    "Appeler apply_urban_mask() AVANT score_disturbance()."
                )

            self.score_confidence["disturbance"] = 0.2
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

        NON INCLUS dans WEIGHTS — redondant avec canopy_openness + disturbance.
        Disponible pour analyse exploratoire : grid.score_forest_edge_distance()
        Le score sera dans grid.scores["forest_edge"] mais ignoré par
        compute_weighted_score() sauf ajout explicite à config.WEIGHTS.
        """
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
            np.asarray(distance_transform_edt(~edge_mask)).astype(np.float32)
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

    def score_favorable_density(self, radius_m: float = 30.0) -> GridBuilder:
        """Score densité locale — NON INCLUS dans WEIGHTS (exploratoire)."""
        from scipy.ndimage import uniform_filter  # import local car optionnel

        if "tree_species" not in self.scores:
            logger.debug("   Densité : score tree_species requis, skip")
            return self

        tree = self.scores["tree_species"]
        favorable = (tree >= 0.7).astype(np.float32)

        kernel_size = max(3, int(2 * radius_m / self.cell_size) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        density: np.ndarray = np.asarray(
            uniform_filter(favorable, size=kernel_size)
        ).astype(np.float32)

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
    ) -> GridBuilder:
        """Masque des zones urbanisées."""
        if urban_gdf is None or urban_gdf.empty:
            self.urban_mask = np.zeros((self.ny, self.nx), dtype=bool)
            logger.info("⚠️ Pas de données urbaines → aucun masque")
            return self

        urban_gdf = self._ensure_l93(urban_gdf)

        # Construire buffer_map depuis constante + surcharge buffer_m
        buffer_map: dict[str, int] = {
            k: (v if k not in ("batiment", "building") else buffer_m)
            for k, v in self._URBAN_BUFFER_DEFAULTS
        }

        shapes: list[tuple[Any, int]] = []
        for _, row in urban_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            shapes.append((geom, 1))

        if not shapes:
            self.urban_mask = np.zeros((self.ny, self.nx), dtype=bool)
            return self

        base_raster = np.asarray(rasterize(
            shapes,
            out_shape=(self.ny, self.nx),
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )).astype(bool)

        if "urban_type" in urban_gdf.columns:
            combined_mask = np.zeros((self.ny, self.nx), dtype=bool)

            for utype, group in urban_gdf.groupby("urban_type"):
                buf = buffer_map.get(str(utype).lower(), buffer_m)

                type_shapes: list[tuple[Any, int]] = [
                    (geom, 1)
                    for geom in group.geometry
                    if geom is not None and not geom.is_empty
                ]
                if not type_shapes:
                    continue

                type_raster = np.asarray(rasterize(
                    type_shapes,
                    out_shape=(self.ny, self.nx),
                    transform=self.transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                )).astype(bool)

                if buf > 0:
                    iterations = max(1, int(buf / self.cell_size))
                    type_raster = np.asarray(
                        binary_dilation(type_raster, iterations=iterations)
                    ).astype(bool)

                combined_mask |= type_raster

            self.urban_mask = combined_mask
        else:
            if buffer_m > 0:
                iterations = max(1, int(buffer_m / self.cell_size))
                base_raster = np.asarray(
                    binary_dilation(base_raster, iterations=iterations)
                ).astype(bool)
            self.urban_mask = base_raster

        _um: np.ndarray = self.urban_mask
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