# visualize.py
"""visualize.py — Cartomorilles v2.4.0

Visualisation interactive (Folium ImageOverlay raster), export GeoTIFF et
GeoPackage.

v2.4.0 :
  - PNG compress_level=1 (×3 encoding speed vs optimize=True)
  - ThreadPoolExecutor pour encoding PNG parallèle (PIL relâche le GIL)
  - _accel.gaussian_filter remplace scipy.ndimage.gaussian_filter
  - Constantes render dédupliquées au niveau module
  - _encode_rgba helper centralisé
  - Hooks pour GPU reprojection (étape 2)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import folium  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from branca.element import Element  # type: ignore[import-untyped]  # noqa: E402
from folium.plugins import MeasureControl, MiniMap, MousePosition  # noqa: E402
from folium.raster_layers import ImageOverlay  # noqa: E402
from PIL import Image  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_bounds  # noqa: E402
from rasterio.warp import (  # noqa: E402
    Resampling,
    calculate_default_transform,
    reproject,
)
from shapely.geometry import box as shapely_box  # noqa: E402

import _accel  # noqa: E402
import config  # noqa: E402
from config import TWI_OPTIMAL, TWI_WATERLOG  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from scoring import MorilleScoring

logger = logging.getLogger("cartomorilles.visualize")

__all__ = ["MorilleVisualizer"]

# ───────────────────────────────────────────────────────────
# Constantes module
# ───────────────────────────────────────────────────────────
_NODATA: float = -9999.0
_MAX_HOTSPOT_MARKERS: int = 1000

# Résolution de rendu — minimum ~25 m/px pour éviter pixels carrés visibles
_RENDER_MAX_CELL_M: float = 25.0
# Taille max du raster WGS84 reprojeté (par axe)
_RENDER_MAX_PX: int = 4000

# PNG : compress_level=1 ≈ ×3 plus rapide que optimize=True, ~20 % plus gros
_PNG_COMPRESS: int = 1
# Nombre de workers pour encoding PNG parallèle
_PNG_WORKERS: int = 4

# Lissage masques éliminatoires (sigma en pixels-grille)
_ELIM_SMOOTH_SIGMA: float = 1.5
# Seuil post-reprojection bilinéaire (< 0.5 → couverture légèrement élargie)
_ELIM_MASK_THRESHOLD: float = 0.3

_CMAP_STOPS: tuple[tuple[float, str], ...] = (
    (0.00, "#d73027"),
    (0.15, "#f46d43"),
    (0.30, "#fdae61"),
    (0.45, "#fee08b"),
    (0.60, "#a6d96a"),
    (0.75, "#1a9850"),
    (1.00, "#006837"),
)

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "morilles",
    [(pos, col) for pos, col in _CMAP_STOPS],
)

_CLASS_COLORS: tuple[str, ...] = (
    "#d73027",
    "#f46d43",
    "#fdae61",
    "#fee08b",
    "#1a9850",
    "#006837",
)

_ELIM_LABEL_MAP: dict[str, tuple[str, str]] = {
    "urban": ("🏠 Zones urbaines", "#555555"),
    "water": ("💧 Cours d'eau", "#0077cc"),
    "nodata": ("❓ NoData", "#000000"),
    "species": ("🌳 Espèces éliminatoires", "#8b0000"),
    "geology": ("🪨 Géologie éliminatoire", "#800080"),
    "slope": ("⛰️ Pente éliminatoire", "#ff8c00"),
    "altitude": ("📏 Altitude éliminatoire", "#8b4513"),
}

# Encodage data-PNG (score/altitude/pente dans RGBA)
_SCORE_SCALE: int = 200
_SCORE_NAN: int = 255
_ALT_OFFSET: int = 500
_SLOPE_OFFSET: int = 1
_SLOPE_SCALE: float = 2.0


# ───────────────────────────────────────────────────────────
# Helpers module-level
# ───────────────────────────────────────────────────────────
def _default_landmarks() -> list[dict[str, Any]]:
    """Points de repère par défaut pour l'Isère."""
    return [
        {
            "name": "Grenoble",
            "lat": 45.1885,
            "lon": 5.7245,
            "info": "Préfecture de l'Isère",
            "icon": "home",
            "icon_color": "blue",
        },
        {
            "name": "Voiron",
            "lat": 45.3650,
            "lon": 5.5910,
            "info": "Porte de la Chartreuse",
            "icon": "info-sign",
            "icon_color": "green",
        },
        {
            "name": "Vizille",
            "lat": 45.0770,
            "lon": 5.7720,
            "info": "Vallée de la Romanche",
            "icon": "info-sign",
            "icon_color": "green",
        },
        {
            "name": "La Mure",
            "lat": 44.9040,
            "lon": 5.7840,
            "info": "Plateau matheysin",
            "icon": "info-sign",
            "icon_color": "green",
        },
        {
            "name": "Sassenage",
            "lat": 45.2130,
            "lon": 5.6620,
            "info": "Cuves de Sassenage — calcaire",
            "icon": "info-sign",
            "icon_color": "darkgreen",
        },
    ]


def _encode_rgba(
    rgba: np.ndarray,
    *,
    compress_level: int = _PNG_COMPRESS,
) -> bytes:
    """Encode un array RGBA uint8 en PNG bytes.

    Centralise la compression rapide (compress_level=1 par défaut).
    PIL relâche le GIL pendant la compression → thread-safe.
    """
    img = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════
# Classe principale
# ═══════════════════════════════════════════════════════════
class MorilleVisualizer:
    """Génère les sorties visuelles à partir d'un MorilleScoring terminé."""

    def __init__(
        self,
        scoring_model: MorilleScoring,
        *,
        hotspots: list[dict[str, Any]] | None = None,
        landmarks: list[dict[str, Any]] | None = None,
        max_hotspot_markers: int = _MAX_HOTSPOT_MARKERS,
        hydro_gdf: gpd.GeoDataFrame | None = None,
        urban_gdf: gpd.GeoDataFrame | None = None,
        inat_observations: list[dict[str, Any]] | None = None,
        weather_days: list[Any] | None = None,
    ) -> None:
        self._validate_model(scoring_model)
        self.model: MorilleScoring = scoring_model
        self.grid = scoring_model.grid
        self.hotspots: list[dict[str, Any]] = hotspots or []
        self.landmarks: list[dict[str, Any]] = landmarks or _default_landmarks()
        self.max_hotspot_markers = max_hotspot_markers
        self._hydro_gdf = hydro_gdf
        self._urban_gdf = urban_gdf
        self._inat_observations: list[dict[str, Any]] = inat_observations or []
        self._weather_days: list[Any] = weather_days or []

        _fs = scoring_model.final_score
        _pc = scoring_model.probability_classes
        _em = scoring_model.elimination_mask
        _ed = scoring_model.elimination_detail
        assert _fs is not None, "final_score is None after validation"
        assert _pc is not None, "probability_classes is None after validation"
        assert _em is not None, "elimination_mask is None after validation"
        assert _ed is not None, "elimination_detail is None after validation"
        self._final_score: np.ndarray = _fs
        self._prob_classes: np.ndarray = _pc
        self._elim_mask: np.ndarray = _em
        self._elim_detail: dict[str, Any] = _ed

        self._to_wgs84 = Transformer.from_crs(
            "EPSG:2154",
            "EPSG:4326",
            always_xy=True,
        )

        half = config.CELL_SIZE / 2.0
        self._xmin_l93 = float(np.min(self.grid.x_coords)) - half
        self._xmax_l93 = float(np.max(self.grid.x_coords)) + half
        self._ymin_l93 = float(np.min(self.grid.y_coords)) - half
        self._ymax_l93 = float(np.max(self.grid.y_coords)) + half

        # Détection orientation y pour flip
        _yc = np.asarray(self.grid.y_coords)
        self._y_ascending: bool = bool(_yc.size > 1 and _yc[-1] > _yc[0])

        # Reprojection L93 → WGS84 (paramètres)
        self._src_transform = from_bounds(
            self._xmin_l93,
            self._ymin_l93,
            self._xmax_l93,
            self._ymax_l93,
            self.grid.nx,
            self.grid.ny,
        )
        self._src_crs = CRS.from_epsg(2154)
        self._dst_crs = CRS.from_epsg(4326)

        # Résolution de rendu (découplée de la grille)
        render_scale = max(1.0, config.CELL_SIZE / _RENDER_MAX_CELL_M)
        render_nx = min(int(self.grid.nx * render_scale), _RENDER_MAX_PX)
        render_ny = min(int(self.grid.ny * render_scale), _RENDER_MAX_PX)

        # Emprise WGS84 exacte depuis pyproj (même transformation que les
        # vecteurs → plus de décalage TWI ↔ cours d'eau).
        _corners_x = [self._xmin_l93, self._xmax_l93, self._xmin_l93, self._xmax_l93]
        _corners_y = [self._ymin_l93, self._ymin_l93, self._ymax_l93, self._ymax_l93]
        _cx, _cy = self._to_wgs84.transform(_corners_x, _corners_y)
        self._west = float(min(_cx))
        self._east = float(max(_cx))
        self._south = float(min(_cy))
        self._north = float(max(_cy))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            self._src_crs,
            self._dst_crs,
            render_nx,
            render_ny,
            left=self._xmin_l93,
            bottom=self._ymin_l93,
            right=self._xmax_l93,
            top=self._ymax_l93,
        )
        assert dst_width is not None
        assert dst_height is not None

        # Recalculer le dst_transform depuis les bounds WGS84 exactes pour
        # garantir l'alignement pixel-parfait avec les vecteurs reprojetés.
        self._dst_width: int = dst_width
        self._dst_height: int = dst_height
        self._dst_transform = from_bounds(
            self._west, self._south,
            self._east, self._north,
            dst_width, dst_height,
        )

        # Bounds Folium (réutilisé partout)
        self._bounds: list[list[float]] = [
            [self._south, self._west],
            [self._north, self._east],
        ]

        logger.debug(
            "Emprise WGS84 : S=%.5f N=%.5f W=%.5f E=%.5f",
            self._south, self._north, self._west, self._east,
        )
        logger.debug(
            "Raster WGS84 : %dx%d px", self._dst_width, self._dst_height,
        )

    # ──────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _validate_model(model: Any) -> None:
        required = [
            "final_score",
            "probability_classes",
            "grid",
            "elimination_mask",
            "elimination_detail",
        ]
        missing = [a for a in required if not hasattr(model, a)]
        if missing:
            raise AttributeError(
                f"MorilleScoring incomplet — manquants : {missing}"
            )
        for attr in required[:2] + required[3:]:
            if getattr(model, attr) is None:
                raise ValueError(
                    f"MorilleScoring.{attr} is None — pipeline incomplet ?"
                )
        grid_attrs = [
            "x_coords", "y_coords", "transform", "nx", "ny",
            "urban_mask", "nodata_mask", "scores",
        ]
        grid_missing = [a for a in grid_attrs if not hasattr(model.grid, a)]
        if grid_missing:
            raise AttributeError(
                f"GridBuilder incomplet — manquants : {grid_missing}"
            )

    # ──────────────────────────────────────────────────────
    # Coordonnées & orientation
    # ──────────────────────────────────────────────────────
    def _l93_to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lon), float(lat)

    def _orient_for_overlay(self, arr: np.ndarray) -> np.ndarray:
        """Oriente un array 2D pour que row 0 = nord."""
        if self._y_ascending:
            return np.asarray(arr[::-1, :])
        return np.asarray(arr)

    def _reproject_to_wgs84(
        self,
        arr: np.ndarray,
        *,
        is_mask: bool = False,
    ) -> np.ndarray:
        """Reprojette un array 2D L93 → WGS84.

        Bilinéaire systématique. Masques seuillés à 0.5 après reprojection.
        """
        oriented = self._orient_for_overlay(arr)
        src = oriented.astype(np.float32)
        src_h, src_w = src.shape

        src_transform = from_bounds(
            self._xmin_l93, self._ymin_l93,
            self._xmax_l93, self._ymax_l93,
            src_w, src_h,
        )

        dst = np.full(
            (self._dst_height, self._dst_width), np.nan, dtype=np.float32,
        )
        reproject(
            source=src,
            destination=dst,
            src_transform=src_transform,
            src_crs=self._src_crs,
            dst_transform=self._dst_transform,
            dst_crs=self._dst_crs,
            resampling=Resampling.bilinear,
        )
        if is_mask:
            return np.asarray(np.nan_to_num(dst, nan=0.0) > 0.5)
        return np.asarray(dst)

    # ──────────────────────────────────────────────────────
    # Renderers statiques (thread-safe, pas d'état mutable)
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _colorize_score(
        score: np.ndarray,
        cmap: mcolors.LinearSegmentedColormap,
        thresholds: list[float],
    ) -> np.ndarray:
        """Colorise un array 2D de scores → RGBA uint8."""
        arr = np.asarray(score, dtype=np.float32)
        valid = np.asarray(np.isfinite(arr))
        safe = np.where(valid, arr, 0.0)

        rgba_f = np.asarray(cmap(safe))  # (H, W, 4) float [0,1]

        alpha = np.zeros(arr.shape, dtype=np.float32)
        steps = np.linspace(0.15, 0.65, len(thresholds) + 1)[1:]
        for i, th in enumerate(thresholds):
            alpha[safe >= th] = float(steps[i])
        if thresholds:
            alpha[safe >= thresholds[-1]] = 0.70
        alpha[~valid] = 0.0
        rgba_f[:, :, 3] = alpha

        return np.clip(rgba_f * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _colorize_mask(
        mask: np.ndarray,
        color_hex: str,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Convertit un masque bool 2D → RGBA uint8 monochrome."""
        m = np.asarray(mask, dtype=bool)
        r_f, g_f, b_f = mcolors.hex2color(color_hex)
        h, w = m.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[m, 0] = int(r_f * 255)
        rgba[m, 1] = int(g_f * 255)
        rgba[m, 2] = int(b_f * 255)
        rgba[m, 3] = int(alpha * 255)
        return rgba

    @staticmethod
    def _colorize_twi_raw(reprojected: np.ndarray) -> np.ndarray:
        """Colorise TWI brut → RGBA uint8 (RdYlBu)."""
        vmin, vmax = 0.0, float(TWI_WATERLOG) + 2.0
        valid = np.isfinite(reprojected)
        norm = np.zeros_like(reprojected, dtype=np.float32)
        norm[valid] = np.clip(
            (reprojected[valid] - vmin) / (vmax - vmin), 0.0, 1.0,
        )
        cmap = matplotlib.colormaps["RdYlBu"]
        colored = cmap(norm)
        rgba = np.zeros((*reprojected.shape, 4), dtype=np.uint8)
        rgba[..., :3] = (colored[..., :3] * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 200, 0).astype(np.uint8)
        return rgba

    @staticmethod
    def _colorize_twi_score(reprojected: np.ndarray) -> np.ndarray:
        """Colorise score TWI [0,1] → RGBA uint8 (RdYlGn)."""
        valid = np.isfinite(reprojected) & (reprojected > 0)
        norm = np.clip(reprojected, 0.0, 1.0)
        cmap = matplotlib.colormaps["RdYlGn"]
        colored = cmap(norm)
        rgba = np.zeros((*reprojected.shape, 4), dtype=np.uint8)
        rgba[..., :3] = (colored[..., :3] * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 180, 0).astype(np.uint8)
        return rgba

    @staticmethod
    def _colorize_waterlog(reprojected: np.ndarray) -> np.ndarray:
        """Masque engorgement → RGBA uint8 rouge."""
        active = reprojected > 0.5
        rgba = np.zeros((*reprojected.shape, 4), dtype=np.uint8)
        rgba[active, 0] = 220
        rgba[active, 1] = 40
        rgba[active, 2] = 40
        rgba[active, 3] = 180
        return rgba

    @staticmethod
    def _png_to_data_uri(png_bytes: bytes) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # ──────────────────────────────────────────────────────
    # Prepare overlays (reprojection séquentielle → render closures)
    # ──────────────────────────────────────────────────────
    # Chaque _prepare_* retourne une liste de tuples :
    #   (name, render_fn, show, opacity, zindex, log_msg)
    # render_fn: Callable[[], bytes]  — thread-safe, pas d'état mutable.
    # ──────────────────────────────────────────────────────

    def _prepare_probability(
        self,
    ) -> list[tuple[str, Callable[[], bytes], bool, float, int, str]]:
        """Prépare la couche probabilité (reprojection + closure render)."""
        score: np.ndarray = self._final_score
        valid: np.ndarray = np.asarray(np.isfinite(score))

        if not np.any(valid):
            logger.warning("Score entièrement NaN — overlay ignoré")
            return []

        # Upsampling source avant reprojection
        scale = max(1.0, config.CELL_SIZE / _RENDER_MAX_CELL_M)
        if scale > 1.0:
            from scipy.ndimage import zoom  # upsample only — pas dans _accel
            safe = np.where(valid, score, 0.0)
            safe_up = np.asarray(zoom(safe, scale, order=1), dtype=np.float32)
            valid_up = np.asarray(
                zoom(valid.astype(np.float32), scale, order=0),
            ) > 0.5
            safe_up[~valid_up] = np.nan
            score_up = safe_up
        else:
            score_up = score

        reprojected = self._reproject_to_wgs84(score_up)
        h_src, w_src = score.shape
        h_dst, w_dst = reprojected.shape
        thresholds = list(config.PROBABILITY_THRESHOLDS)

        def render() -> bytes:
            rgba = MorilleVisualizer._colorize_score(
                reprojected, _CMAP, thresholds,
            )
            return _encode_rgba(rgba)

        msg = (
            f"Couche probabilité : {w_src}×{h_src}"
            f" → {w_dst}×{h_dst} px WGS84"
        )
        return [
            ("🍄 Probabilité morilles", render, True, 1.0, 1, msg),
        ]

    def _prepare_elimination(
        self,
    ) -> list[tuple[str, Callable[[], bytes], bool, float, int, str]]:
        """Prépare les couches éliminatoires (lissage + reprojection)."""
        if not self._elim_detail:
            return []

        specs: list[tuple[str, Callable[[], bytes], bool, float, int, str]] = []

        for key, mask_arr in self._elim_detail.items():
            if not isinstance(mask_arr, np.ndarray):
                continue
            mask = np.asarray(mask_arr, dtype=bool)
            if not np.any(mask):
                continue

            # Lissage en espace source via _accel
            mask_f = np.asarray(
                _accel.gaussian_filter(
                    mask.astype(np.float32),
                    sigma=_ELIM_SMOOTH_SIGMA,
                    mode="reflect",
                ),
            )

            # Reprojection bilinéaire (float, pas is_mask)
            reprojected_f = self._reproject_to_wgs84(mask_f, is_mask=False)
            reprojected = np.asarray(
                np.nan_to_num(reprojected_f, nan=0.0) > _ELIM_MASK_THRESHOLD,
            )

            label, color = _ELIM_LABEL_MAP.get(key, (f"❌ {key}", "#ff0000"))
            n = int(np.sum(mask))

            # Capture explicite pour closure dans boucle
            def render(
                _r: np.ndarray = reprojected,
                _c: str = color,
            ) -> bytes:
                rgba = MorilleVisualizer._colorize_mask(_r, _c, alpha=0.45)
                return _encode_rgba(rgba)

            specs.append(
                (label, render, False, 1.0, 2, f"{label} : {n} cellules"),
            )

        logger.info("✅ %d couches éliminatoires préparées", len(specs))
        return specs

    def _prepare_urban_raster(
        self,
    ) -> list[tuple[str, Callable[[], bytes], bool, float, int, str]]:
        """Prépare la couche urbaine raster adoucie."""
        if not hasattr(self.grid, "urban_mask") or self.grid.urban_mask is None:
            return []

        mask = np.asarray(self.grid.urban_mask, dtype=bool)
        if not np.any(mask):
            return []

        # Reprojection bilinéaire suffit pour lisser le masque —
        # pas besoin d'upscale + gaussian sur grille doublée
        mask_f = _accel.gaussian_filter(
            mask.astype(np.float32), sigma=1.5, mode="reflect",
        )
        reprojected = self._reproject_to_wgs84(mask_f, is_mask=False)
        reprojected = np.asarray(
            np.nan_to_num(reprojected, nan=0.0) > 0.3,
        )
        n = int(np.sum(self.grid.urban_mask))

        def render() -> bytes:
            rgba = MorilleVisualizer._colorize_mask(
                reprojected, "#888888", alpha=0.35,
            )
            return _encode_rgba(rgba)

        return [
            ("🏘️ Zones urbaines", render, False, 1.0, 2,
             f"Urbain raster : {n} cellules"),
        ]

    def _prepare_twi(
        self,
    ) -> list[tuple[str, Callable[[], bytes], bool, float, int, str]]:
        """Prépare les couches TWI (brut, score, engorgement)."""
        twi_data = self.model.get_twi_display_data()
        if not twi_data["has_data"]:
            logger.debug("   Pas de données TWI")
            return []

        specs: list[tuple[str, Callable[[], bytes], bool, float, int, str]] = []

        # TWI brut
        twi_raw = twi_data["raw"]
        if twi_raw is not None:
            rep_raw = self._reproject_to_wgs84(np.asarray(twi_raw, dtype=np.float32))
            rng = f"[{float(np.nanmin(twi_raw)):.1f}, {float(np.nanmax(twi_raw)):.1f}]"

            def render_raw(_r: np.ndarray = rep_raw) -> bytes:
                return _encode_rgba(MorilleVisualizer._colorize_twi_raw(_r))

            specs.append((
                "🌊 TWI brut (valeurs hydrologiques)",
                render_raw, False, 0.65, 1,
                f"TWI brut — range {rng}",
            ))

        # TWI score
        twi_score = twi_data["score"]
        if isinstance(twi_score, np.ndarray):
            rep_score = self._reproject_to_wgs84(
                np.asarray(twi_score, dtype=np.float32),
            )
            valid_s = twi_score[np.isfinite(twi_score)]
            stats = ""
            if valid_s.size > 0:
                stats = (
                    f" mean={float(np.mean(valid_s)):.3f}"
                    f" median={float(np.median(valid_s)):.3f}"
                )

            def render_score(_r: np.ndarray = rep_score) -> bytes:
                return _encode_rgba(
                    MorilleVisualizer._colorize_twi_score(_r),
                )

            specs.append((
                "📊 TWI score [0–1]",
                render_score, False, 0.65, 1,
                f"TWI score —{stats}",
            ))

        # Engorgement
        waterlog = twi_data["waterlog_mask"]
        if waterlog is not None and np.any(waterlog):
            rep_wl = self._reproject_to_wgs84(
                np.asarray(waterlog, dtype=np.float32),
            )
            n_wl = int(np.sum(waterlog))

            def render_wl(_r: np.ndarray = rep_wl) -> bytes:
                return _encode_rgba(MorilleVisualizer._colorize_waterlog(_r))

            specs.append((
                f"⚠️ TWI engorgement (>{TWI_WATERLOG}) — {n_wl} cellules",
                render_wl, False, 0.70, 2,
                f"Engorgement TWI : {n_wl} cellules (>{TWI_WATERLOG:.0f})",
            ))
        else:
            logger.info(
                "   Aucune cellule engorgée (TWI > %.0f)", TWI_WATERLOG,
            )

        return specs

    # ──────────────────────────────────────────────────────
    # Data PNG (score + altitude + pente encodés RGBA)
    # ──────────────────────────────────────────────────────
    def _build_data_png_bytes(self) -> bytes:
        """Construit le PNG RGB opaque pour tooltip mousemove.

        R = score (0–200 → 0.00–1.00, 255 = NaN)
        G = altitude low byte (alt + 500)
        B = altitude high byte

        Fix B1 : RGB opaque — pas de pré-multiplication alpha navigateur.
        La pente est encodée dans un PNG séparé (_build_slope_png_bytes).
        """
        score_wgs = self._reproject_to_wgs84(self._final_score)
        h, w = score_wgs.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # R : score
        valid_s = np.asarray(np.isfinite(score_wgs))
        r_ch = np.full((h, w), _SCORE_NAN, dtype=np.uint8)
        r_ch[valid_s] = np.clip(
            (score_wgs[valid_s] * _SCORE_SCALE).astype(np.int32),
            0, _SCORE_SCALE,
        ).astype(np.uint8)
        rgb[:, :, 0] = r_ch

        # G, B : altitude
        alt_grid = getattr(self.grid, "altitude", None)
        if isinstance(alt_grid, np.ndarray) and alt_grid.shape == self._final_score.shape:
            alt_wgs = self._reproject_to_wgs84(alt_grid)
            valid_a = np.asarray(np.isfinite(alt_wgs))
            alt_enc = np.zeros((h, w), dtype=np.uint16)
            alt_enc[valid_a] = np.clip(
                (alt_wgs[valid_a] + _ALT_OFFSET).astype(np.int32),
                0, 65535,
            ).astype(np.uint16)
            rgb[:, :, 1] = (alt_enc & 0xFF).astype(np.uint8)
            rgb[:, :, 2] = ((alt_enc >> 8) & 0xFF).astype(np.uint8)
        else:
            logger.debug("   Altitude non disponible pour data PNG")

        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=6)
        png_bytes = buf.getvalue()
        size_kb = len(png_bytes) / 1024
        logger.info(
            "✅ Data PNG : %dx%d px (%.0f KB) — score+alt",
            w, h, size_kb,
        )
        return png_bytes

    def _build_slope_png_bytes(self) -> bytes:
        """Construit le PNG RGB opaque contenant la pente par pixel.

        R = pente encodée (1–181 → 0–90°, 0 = NaN)
        G = 0, B = 0 (réservés)

        Fix B1 : séparé du data PNG pour éviter la pré-multiplication alpha.
        """
        score_wgs = self._reproject_to_wgs84(self._final_score)
        h, w = score_wgs.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        slp_grid = getattr(self.grid, "slope", None)
        if isinstance(slp_grid, np.ndarray) and slp_grid.shape == self._final_score.shape:
            slp_wgs = self._reproject_to_wgs84(slp_grid)
            valid_p = np.asarray(np.isfinite(slp_wgs))
            r_ch = np.zeros((h, w), dtype=np.uint8)
            r_ch[valid_p] = np.clip(
                (slp_wgs[valid_p] * _SLOPE_SCALE + _SLOPE_OFFSET).astype(
                    np.int32,
                ),
                1, 181,
            ).astype(np.uint8)
            rgb[:, :, 0] = r_ch
        else:
            logger.debug("   Pente non disponible pour slope PNG")

        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=6)
        png_bytes = buf.getvalue()
        size_kb = len(png_bytes) / 1024
        logger.info(
            "✅ Slope PNG : %dx%d px (%.0f KB) — pente",
            w, h, size_kb,
        )
        return png_bytes

    # ──────────────────────────────────────────────────────
    # JS consolidé — panneaux draggable, tooltip, filtres
    # ──────────────────────────────────────────────────────
    def _add_interactive_controls(
        self,
        folium_map: folium.Map,
        data_uri: str,
        slope_uri: str = "",
    ) -> None:
        """Injecte CSS + JS : panneaux draggable, opacité, filtre, légende,
        tooltip mousemove, hotspot slider."""

        n_hotspots = getattr(self, "_hotspot_count_on_map", 0)

        # ── Template JS (inline) ──
        js_block = r"""
        <style>
        .cartom-panel {
            background: rgba(255,255,255,0.95);
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,.35);
            font: 12px/1.5 'Segoe UI', Arial, sans-serif;
            position: fixed;
            z-index: 10000;
            pointer-events: auto;
            max-width: 260px;
        }
        .cartom-panel .cartom-titlebar {
            cursor: grab;
            background: linear-gradient(135deg, #2d6a2e, #3a9e3f);
            color: white;
            padding: 5px 10px;
            border-radius: 8px 8px 0 0;
            font-weight: bold;
            font-size: 12px;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .cartom-panel .cartom-titlebar:active { cursor: grabbing; }
        .cartom-panel .cartom-body { padding: 8px 12px; }
        .cartom-panel .cartom-minimize {
            cursor: pointer; font-size: 16px; line-height: 1; opacity: 0.8;
        }
        .cartom-panel .cartom-minimize:hover { opacity: 1; }
        .cartom-panel.minimized .cartom-body { display: none; }

        #cartom-legend   { bottom: 30px;  left: 30px; }
        #cartom-colorbar { top: 10px;     right: 10px; left: auto; }
        #cartom-opacity  { top: 80px;     left: 10px; }
        #cartom-filter   { top: 190px;    left: 10px; }
        #cartom-hotspots { top: 310px;    left: 10px; }
        #cartom-info     { bottom: 30px;  right: 10px; min-width: 200px; }

        .cartom-slider-row {
            display: flex; align-items: center; gap: 6px; margin: 4px 0;
        }
        .cartom-slider-row label {
            min-width: 30px; font-size: 11px; color: #555;
        }
        .cartom-slider-row input[type=range] { flex: 1; }
        .cartom-slider-row .cartom-val {
            min-width: 32px; text-align: right; font-size: 11px;
            font-family: monospace; color: #333;
        }

        .cartom-hotspot-hidden { display: none !important; }
        </style>

        <div id="cartom-legend" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🍄 Probabilité morilles</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body" id="cartom-legend-body"></div>
        </div>

        <div id="cartom-colorbar" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🎨 Échelle</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body" id="cartom-colorbar-body"></div>
        </div>

        <div id="cartom-opacity" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🍄 Opacité</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body">
                <input type="range" min="0" max="100" value="70"
                       style="width:100%;" id="cartom_opacity_slider">
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#666;">
                    <span>0%</span><span id="cartom_opval">70%</span><span>100%</span>
                </div>
            </div>
        </div>

        <div id="cartom-filter" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🎚️ Filtre probabilité</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body">
                <div class="cartom-slider-row">
                    <label>Min</label>
                    <input type="range" min="0" max="100" value="0"
                           id="cartom_filter_min">
                    <span class="cartom-val" id="cartom_fmin_val">0.00</span>
                </div>
                <div class="cartom-slider-row">
                    <label>Max</label>
                    <input type="range" min="0" max="100" value="100"
                           id="cartom_filter_max">
                    <span class="cartom-val" id="cartom_fmax_val">1.00</span>
                </div>
                <div style="text-align:center;font-size:10px;color:#888;margin-top:4px;">
                    Affiche les cellules dans [Min, Max]
                </div>
            </div>
        </div>

        <div id="cartom-hotspots" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🎯 Hotspots visibles</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body">
                <div class="cartom-slider-row">
                    <label>Top</label>
                    <input type="range" min="0" max="%N_HOTSPOTS%"
                           value="%N_HOTSPOTS%" step="1"
                           id="cartom_hotspot_slider">
                    <span class="cartom-val" id="cartom_hs_val">%N_HOTSPOTS%</span>
                </div>
                <div style="text-align:center;font-size:10px;color:#888;margin-top:4px;">
                    Affiche les N meilleurs hotspots (sur %N_HOTSPOTS%)
                </div>
            </div>
        </div>

        <div id="cartom-info" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>📍 Cellule</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body" id="cartom-info-body">
                <span style="color:#888;">Survolez la carte…</span>
            </div>
        </div>

        <img id="cartom-data-img" src="%DATA_URI%" style="display:none;">
        <canvas id="cartom-data-canvas" style="display:none;"></canvas>
        <canvas id="cartom-render-canvas" style="display:none;"></canvas>
        <img id="cartom-slope-img" src="%SLOPE_URI%" style="display:none;">
        <canvas id="cartom-slope-canvas" style="display:none;"></canvas>

        <script>
        document.addEventListener('DOMContentLoaded', function(){

            /* ═══════ DRAG ═══════ */
            document.querySelectorAll('.cartom-panel').forEach(function(panel){
                var bar = panel.querySelector('.cartom-titlebar');
                if (!bar) return;
                bar.addEventListener('mousedown', function(e){
                    if (e.target.classList.contains('cartom-minimize')) return;
                    e.preventDefault();
                    var mx = e.clientX, my = e.clientY;
                    var rect = panel.getBoundingClientRect();
                    panel.style.top  = rect.top + 'px';
                    panel.style.left = rect.left + 'px';
                    panel.style.bottom = 'auto';
                    panel.style.right  = 'auto';
                    function onMove(ev){
                        var dx = ev.clientX - mx, dy = ev.clientY - my;
                        mx = ev.clientX; my = ev.clientY;
                        var t = Math.max(0, Math.min(parseInt(panel.style.top)+dy, window.innerHeight-40));
                        var l = Math.max(0, Math.min(parseInt(panel.style.left)+dx, window.innerWidth-40));
                        panel.style.top = t+'px'; panel.style.left = l+'px';
                    }
                    function onUp(){ document.removeEventListener('mousemove',onMove); document.removeEventListener('mouseup',onUp); }
                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup', onUp);
                });
                L.DomEvent.disableClickPropagation(panel);
                L.DomEvent.disableScrollPropagation(panel);
            });

            /* ═══════ LÉGENDE ═══════ */
            var legendBody = document.getElementById('cartom-legend-body');
            if (legendBody) {
                var labels = %LABELS_JSON%;
                var thresholds = %THRESHOLDS_JSON%;
                var colors = %COLORS_JSON%;
                var cellSize = %CELL_SIZE%;
                var html = '';
                for (var i = labels.length-1; i >= 0; i--) {
                    var range;
                    if (i===0) range='< '+thresholds[0].toFixed(2);
                    else if (i===labels.length-1) range='≥ '+thresholds[thresholds.length-1].toFixed(2);
                    else range=thresholds[i-1].toFixed(2)+' – '+thresholds[i].toFixed(2);
                    html+='<div style="margin:2px 0;">'
                        +'<span style="display:inline-block;width:18px;height:13px;background:'+colors[i]
                        +';border-radius:2px;margin-right:6px;vertical-align:middle;"></span>'
                        +'<span style="vertical-align:middle;">'+labels[i]+' ('+range+')</span></div>';
                }
                html+='<div style="margin-top:6px;color:#888;font-size:10px;">Résolution : '+cellSize+'m × '+cellSize+'m</div>';
                legendBody.innerHTML = html;
            }

            /* ═══════ COLORBAR ═══════ */
            var cbBody = document.getElementById('cartom-colorbar-body');
            if (cbBody) {
                var stops = %CMAP_STOPS_JSON%;
                var grad = stops.map(function(s){ return s[1]+' '+(s[0]*100)+'%'; }).join(', ');
                cbBody.innerHTML =
                    '<div style="height:14px;border-radius:3px;background:linear-gradient(to right, '+grad+');margin-bottom:3px;"></div>'
                    +'<div style="display:flex;justify-content:space-between;font-size:10px;color:#666;">'
                    +'<span>0.0</span><span>0.3</span><span>0.5</span><span>0.8</span><span>1.0</span></div>'
                    +'<div style="text-align:center;font-size:10px;color:#888;margin-top:2px;">Score de probabilité morilles</div>';
            }

            /* ═══════ FIND MAP ═══════ */
            var map = null;
            for (var k in window) {
                try { if (window[k] instanceof L.Map) { map = window[k]; break; } }
                catch(e){}
            }

            /* ═══════ COLORMAP JS ═══════ */
            var cmapStops = %CMAP_STOPS_JSON%;

            function hexToRgb(hex) {
                var r = parseInt(hex.slice(1,3),16);
                var g = parseInt(hex.slice(3,5),16);
                var b = parseInt(hex.slice(5,7),16);
                return [r, g, b];
            }

            function scoreToRgb(s) {
                if (s <= cmapStops[0][0]) return hexToRgb(cmapStops[0][1]);
                if (s >= cmapStops[cmapStops.length-1][0]) return hexToRgb(cmapStops[cmapStops.length-1][1]);
                for (var i = 1; i < cmapStops.length; i++) {
                    if (s <= cmapStops[i][0]) {
                        var t = (s - cmapStops[i-1][0]) / (cmapStops[i][0] - cmapStops[i-1][0]);
                        var c0 = hexToRgb(cmapStops[i-1][1]);
                        var c1 = hexToRgb(cmapStops[i][1]);
                        return [
                            Math.round(c0[0]+(c1[0]-c0[0])*t),
                            Math.round(c0[1]+(c1[1]-c0[1])*t),
                            Math.round(c0[2]+(c1[2]-c0[2])*t)
                        ];
                    }
                }
                return hexToRgb(cmapStops[cmapStops.length-1][1]);
            }

            var thresholdsA = %THRESHOLDS_JSON%;
            var alphaSteps = [];
            for (var ai = 1; ai <= thresholdsA.length + 1; ai++) {
                alphaSteps.push(0.15 + (0.65 - 0.15) * ai / (thresholdsA.length + 1));
            }

            function scoreToAlpha(s) {
                var a = 0;
                for (var i = 0; i < thresholdsA.length; i++) {
                    if (s >= thresholdsA[i]) a = alphaSteps[i];
                }
                if (thresholdsA.length > 0 && s >= thresholdsA[thresholdsA.length-1]) a = 0.70;
                return a;
            }

            /* ═══════ DATA & RENDER CANVAS ═══════ */
            var dataImg    = document.getElementById('cartom-data-img');
            var dataCanvas = document.getElementById('cartom-data-canvas');
            var renderCanvas = document.getElementById('cartom-render-canvas');
            var slopeImg    = document.getElementById('cartom-slope-img');
            var slopeCanvas = document.getElementById('cartom-slope-canvas');
            var infoBody   = document.getElementById('cartom-info-body');
            var dataCtx    = null;
            var renderCtx  = null;
            var slopeCtx   = null;
            var dataW = 0, dataH = 0;

            var south = %SOUTH%, north = %NORTH%;
            var west  = %WEST%,  east  = %EAST%;
            var SCORE_SCALE = %SCORE_SCALE%;
            var SCORE_NAN   = %SCORE_NAN%;
            var ALT_OFFSET  = %ALT_OFFSET%;
            var SLOPE_OFFSET = %SLOPE_OFFSET%;
            var SLOPE_SCALE  = %SLOPE_SCALE%;
            var labelsT = %LABELS_JSON%;
            var thresholdsT = %THRESHOLDS_JSON%;

            var probOverlay = null;
            var renderTimer = null;

            function initCanvases() {
                if (!dataImg || !dataCanvas || !renderCanvas) return false;
                dataW = dataImg.naturalWidth;
                dataH = dataImg.naturalHeight;
                if (dataW === 0 || dataH === 0) return false;

                dataCanvas.width = dataW;
                dataCanvas.height = dataH;
                dataCtx = dataCanvas.getContext('2d', {willReadFrequently: true});
                dataCtx.drawImage(dataImg, 0, 0);

                renderCanvas.width = dataW;
                renderCanvas.height = dataH;
                renderCtx = renderCanvas.getContext('2d');

                /* Slope canvas — Fix B1 */
                if (slopeImg && slopeCanvas && slopeImg.naturalWidth > 0) {
                    slopeCanvas.width  = slopeImg.naturalWidth;
                    slopeCanvas.height = slopeImg.naturalHeight;
                    slopeCtx = slopeCanvas.getContext('2d', {willReadFrequently: true});
                    slopeCtx.drawImage(slopeImg, 0, 0);
                }

                return true;
            }

            function renderOverlay(filterMin, filterMax) {
                if (!dataCtx || !renderCtx) return;
                var src = dataCtx.getImageData(0, 0, dataW, dataH);
                var dst = renderCtx.createImageData(dataW, dataH);
                var sd = src.data, dd = dst.data;
                var len = dataW * dataH;

                for (var i = 0; i < len; i++) {
                    var off = i * 4;
                    var rVal = sd[off];

                    if (rVal === SCORE_NAN) { dd[off+3] = 0; continue; }

                    var score = rVal / SCORE_SCALE;
                    if (score < filterMin || score > filterMax) {
                        dd[off+3] = 0;
                        continue;
                    }

                    var rgb = scoreToRgb(score);
                    var alpha = scoreToAlpha(score);
                    dd[off]   = rgb[0];
                    dd[off+1] = rgb[1];
                    dd[off+2] = rgb[2];
                    dd[off+3] = Math.round(alpha * 255);
                }

                renderCtx.putImageData(dst, 0, 0);

                var url = renderCanvas.toDataURL('image/png');
                if (probOverlay) {
                    probOverlay.setUrl(url);
                } else if (map) {
                    probOverlay = L.imageOverlay(url,
                        [[south, west], [north, east]],
                        {opacity: 0.7, interactive: false, zIndex: 1}
                    );
                    probOverlay.addTo(map);

                    var controls = document.querySelectorAll('.leaflet-control-layers-overlays');
                    if (controls.length > 0) {
                        var container = controls[0];
                        var lbl = document.createElement('label');
                        lbl.innerHTML = '<span><input type="checkbox" class="leaflet-control-layers-selector" checked>'
                            + ' <span> 🍄 Probabilité morilles</span></span>';
                        container.insertBefore(lbl, container.firstChild);
                        var cb = lbl.querySelector('input');
                        cb.addEventListener('change', function(){
                            if (this.checked) probOverlay.addTo(map);
                            else map.removeLayer(probOverlay);
                        });
                    }
                }
            }

            function renderDebounced() {
                if (renderTimer) clearTimeout(renderTimer);
                renderTimer = setTimeout(function(){
                    var fmin = parseInt(document.getElementById('cartom_filter_min').value) / 100.0;
                    var fmax = parseInt(document.getElementById('cartom_filter_max').value) / 100.0;
                    renderOverlay(fmin, fmax);
                }, 60);
            }

            function onReady() {
                if (!initCanvases()) return;
                renderOverlay(0.0, 1.0);
            }

            if (dataImg && dataImg.complete && dataImg.naturalWidth > 0) {
                onReady();
            } else if (dataImg) {
                dataImg.addEventListener('load', onReady);
            }
            if (slopeImg && !slopeImg.complete) {
                slopeImg.addEventListener('load', function(){ initCanvases(); });
            }

            /* ═══════ FILTRE SLIDERS ═══════ */
            var sliderMin = document.getElementById('cartom_filter_min');
            var sliderMax = document.getElementById('cartom_filter_max');
            var valMin = document.getElementById('cartom_fmin_val');
            var valMax = document.getElementById('cartom_fmax_val');

            if (sliderMin && sliderMax) {
                sliderMin.addEventListener('input', function(){
                    var v = parseInt(this.value);
                    var mx = parseInt(sliderMax.value);
                    if (v > mx) { v = mx; this.value = v; }
                    valMin.textContent = (v/100).toFixed(2);
                    renderDebounced();
                });
                sliderMax.addEventListener('input', function(){
                    var v = parseInt(this.value);
                    var mn = parseInt(sliderMin.value);
                    if (v < mn) { v = mn; this.value = v; }
                    valMax.textContent = (v/100).toFixed(2);
                    renderDebounced();
                });
            }

            /* ═══════ OPACITÉ ═══════ */
            var opSlider = document.getElementById('cartom_opacity_slider');
            var opVal = document.getElementById('cartom_opval');
            if (opSlider) {
                opSlider.addEventListener('input', function(e){
                    var v = parseInt(e.target.value) / 100.0;
                    if (opVal) opVal.textContent = e.target.value + '%';
                    if (!map) return;
                    map.eachLayer(function(layer){
                        if (layer instanceof L.TileLayer) return;
                        if (layer instanceof L.ImageOverlay) { layer.setOpacity(v); return; }
                        if (layer.eachLayer) layer.eachLayer(function(sub){
                            if (sub instanceof L.ImageOverlay) sub.setOpacity(v);
                        });
                    });
                });
            }

            /* ═══════ HOTSPOT SLIDER ═══════ */
            var hsSlider = document.getElementById('cartom_hotspot_slider');
            var hsVal    = document.getElementById('cartom_hs_val');
            var nHotspots = %N_HOTSPOTS%;

            function updateHotspotVisibility(maxVisible) {
                if (!map) return;
                map.eachLayer(function(layer) {
                    if (!layer.eachLayer) return;
                    layer.eachLayer(function(marker) {
                        if (!marker.options || !marker.options.className) return;
                        var cls = marker.options.className;
                        if (cls.indexOf('cartom-hotspot') === -1) return;
                        var match = cls.match(/cartom-rank-(\d+)/);
                        if (!match) return;
                        var rank = parseInt(match[1]);
                        var el = null;
                        if (marker._icon) el = marker._icon;
                        else if (marker.getElement) el = marker.getElement();
                        if (!el) return;
                        if (rank < maxVisible) {
                            el.classList.remove('cartom-hotspot-hidden');
                            el.style.display = '';
                        } else {
                            el.classList.add('cartom-hotspot-hidden');
                            el.style.display = 'none';
                        }
                        if (marker._shadow) {
                            marker._shadow.style.display = rank < maxVisible ? '' : 'none';
                        }
                    });
                });
            }

            if (hsSlider) {
                hsSlider.addEventListener('input', function() {
                    var v = parseInt(this.value);
                    if (hsVal) hsVal.textContent = v;
                    updateHotspotVisibility(v);
                });
            }

            /* ═══════ TOOLTIP ═══════ */
            function getClass(score) {
                for (var i = thresholdsT.length - 1; i >= 0; i--) {
                    if (score >= thresholdsT[i]) return labelsT[i + 1];
                }
                return labelsT[0];
            }

            if (map && infoBody) {
                map.on('mousemove', function(e) {
                    if (!dataCtx) { if (!initCanvases()) return; }
                    var lat = e.latlng.lat, lng = e.latlng.lng;

                    if (lat < south || lat > north || lng < west || lng > east) {
                        infoBody.innerHTML = '<span style="color:#888;">Hors emprise</span>';
                        return;
                    }

                    var px = Math.floor((lng - west) / (east - west) * dataW);
                    var py = Math.floor((north - lat) / (north - south) * dataH);
                    px = Math.max(0, Math.min(px, dataW - 1));
                    py = Math.max(0, Math.min(py, dataH - 1));

                    var pixel = dataCtx.getImageData(px, py, 1, 1).data;
                    var r = pixel[0], g = pixel[1], b = pixel[2];

                    if (r === SCORE_NAN || (r===0 && g===0 && b===0)) {
                        infoBody.innerHTML = '<span style="color:#888;">Pas de données</span>';
                        return;
                    }

                    var score = r / SCORE_SCALE;
                    var cls = getClass(score);
                    var h = '<b>🍄 '+(score*100).toFixed(1)+'%</b> — <em>'+cls+'</em><br>';

                    /* Altitude — Fix B1 : RGB opaque, plus de pré-multiplication */
                    var altRaw = g + (b << 8);
                    if (altRaw > 0) h += '⛰️ '+(altRaw - ALT_OFFSET)+' m<br>';

                    /* Pente — Fix B1 : lue depuis slope PNG séparé */
                    var slopeR = slopeCtx ? slopeCtx.getImageData(px, py, 1, 1).data[0] : 0;
                    if (slopeR > 0) {
                        var slope = (slopeR - SLOPE_OFFSET) / SLOPE_SCALE;
                        h += '📐 '+slope.toFixed(1)+'°<br>';
                    }

                    h += '<span style="font-size:9px;color:#aaa;">'
                        +lat.toFixed(5)+', '+lng.toFixed(5)+'</span>';
                    infoBody.innerHTML = h;
                });

                map.on('mouseout', function(){
                    infoBody.innerHTML = '<span style="color:#888;">Survolez la carte…</span>';
                });
            }
        });
        </script>
        """

        labels_json = json.dumps(list(config.PROBABILITY_LABELS))
        thresholds_json = json.dumps(list(config.PROBABILITY_THRESHOLDS))
        colors_json = json.dumps(
            list(_CLASS_COLORS[: len(config.PROBABILITY_LABELS)]),
        )
        cmap_stops_json = json.dumps([[pos, col] for pos, col in _CMAP_STOPS])

        js_final = (
            js_block.replace("%DATA_URI%", data_uri)
            .replace("%SLOPE_URI%", slope_uri)
            .replace("%SOUTH%", f"{self._south:.8f}")
            .replace("%NORTH%", f"{self._north:.8f}")
            .replace("%WEST%", f"{self._west:.8f}")
            .replace("%EAST%", f"{self._east:.8f}")
            .replace("%SCORE_SCALE%", str(_SCORE_SCALE))
            .replace("%SCORE_NAN%", str(_SCORE_NAN))
            .replace("%ALT_OFFSET%", str(_ALT_OFFSET))
            .replace("%SLOPE_OFFSET%", str(_SLOPE_OFFSET))
            .replace("%SLOPE_SCALE%", str(_SLOPE_SCALE))
            .replace("%LABELS_JSON%", labels_json)
            .replace("%THRESHOLDS_JSON%", thresholds_json)
            .replace("%COLORS_JSON%", colors_json)
            .replace("%CELL_SIZE%", str(config.CELL_SIZE))
            .replace("%CMAP_STOPS_JSON%", cmap_stops_json)
            .replace("%N_HOTSPOTS%", str(n_hotspots))
        )

        folium_map.get_root().html.add_child(  # type: ignore[attr-defined]
            Element(js_final),
        )

    # ──────────────────────────────────────────────────────
    # Fonds de carte
    # ──────────────────────────────────────────────────────
    def _add_basemaps(self, folium_map: folium.Map) -> None:
        for name, cfg in config.BASEMAPS.items():
            is_default = name == config.DEFAULT_BASEMAP
            tiles = cfg["tiles"]
            if tiles in {
                "OpenStreetMap",
                "CartoDB positron",
                "CartoDB dark_matter",
            }:
                folium.TileLayer(
                    tiles=tiles,
                    name=name,
                    show=is_default,
                ).add_to(folium_map)
            else:
                folium.TileLayer(
                    tiles=tiles,
                    attr=cfg.get("attr", ""),
                    name=cfg.get("name", name),
                    max_zoom=cfg.get("max_zoom", 18),
                    show=is_default,
                ).add_to(folium_map)

    # ──────────────────────────────────────────────────────
    # Couche hydro vectorielle
    # ──────────────────────────────────────────────────────
    def _add_hydro_vector_layer(self, folium_map: folium.Map) -> None:
        if self._hydro_gdf is None or self._hydro_gdf.empty:
            logger.debug("   Pas de données hydro vectorielles")
            return

        gdf = self._hydro_gdf.copy()
        if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs(epsg=4326)

        style = {"color": "#0077cc", "weight": 2, "opacity": 0.7}
        fg = folium.FeatureGroup(name="💧 Cours d'eau", show=False)
        folium.GeoJson(
            gdf.geometry.__geo_interface__,
            style_function=lambda _feat, _s=style: _s,
        ).add_to(fg)
        fg.add_to(folium_map)
        logger.info("✅ Cours d'eau vectoriels : %d entités", len(gdf))

    # ──────────────────────────────────────────────────────
    # Couche iNaturalist Morchella
    # ──────────────────────────────────────────────────────
    def _add_inaturalist_layer(self, folium_map: folium.Map) -> None:
        if not self._inat_observations:
            return

        fg = folium.FeatureGroup(name="🍄 iNaturalist Morchella", show=True)
        for obs in self._inat_observations:
            lat = obs.get("lat")
            lng = obs.get("lng")
            if lat is None or lng is None:
                continue
            quality = str(obs.get("quality_grade", "needs_id"))
            date_str = str(obs.get("observed_on", "?"))
            taxon = str(obs.get("taxon_name", "Morchella"))
            obs_id = obs.get("id", "?")

            color = "#e67300" if quality == "research" else "#cc9900"
            popup_html = (
                f"<b>iNaturalist #{obs_id}</b><br>"
                f"Taxon : {taxon}<br>"
                f"Date : {date_str}<br>"
                f"Qualité : {quality}"
            )
            folium.CircleMarker(
                location=[float(lat), float(lng)],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(fg)

        fg.add_to(folium_map)
        logger.info(
            "✅ iNaturalist : %d observations affichées",
            len(self._inat_observations),
        )

    # ──────────────────────────────────────────────────────
    # Panneau météo
    # ──────────────────────────────────────────────────────
    def _add_weather_panel(self, folium_map: folium.Map) -> None:
        if not self._weather_days:
            return

        rows = ""
        best_day = None
        best_score = -1.0
        for d in self._weather_days:
            score = float(d.score)
            label = str(d.label)
            date_str = str(d.date_fr)
            # Icône selon le niveau
            if score >= 0.7:
                icon = "&#9733;"  # star
                cls = "wx-good"
            elif score >= 0.4:
                icon = "&#9679;"  # circle
                cls = "wx-ok"
            else:
                icon = "&#9675;"  # empty circle
                cls = "wx-bad"
            rows += (
                f'<tr class="{cls}">'
                f"<td>{date_str}</td>"
                f"<td>{icon} {label}</td>"
                f"<td>{score:.0%}</td>"
                f"</tr>"
            )
            if score > best_score:
                best_score = score
                best_day = date_str

        burst_html = ""
        for d in self._weather_days:
            if getattr(d, "burst", None) is not None:
                burst_html = (
                    '<div class="wx-burst">'
                    "&#x1F4A7; Signal fructification détecté"
                    "</div>"
                )
                break

        best_html = ""
        if best_day:
            best_html = (
                f'<div class="wx-best">'
                f"Meilleur jour : <b>{best_day}</b>"
                f"</div>"
            )

        html = f"""
        <div id="wx-panel" style="
            position:fixed; bottom:30px; right:10px; z-index:1000;
            background:rgba(255,255,255,0.92); border-radius:8px;
            padding:10px 14px; font-size:12px; font-family:sans-serif;
            box-shadow:0 2px 8px rgba(0,0,0,0.3); max-width:260px;
            border:1px solid #ccc;">
          <style>
            #wx-panel table {{ border-collapse:collapse; width:100%; }}
            #wx-panel td {{ padding:2px 6px; }}
            #wx-panel .wx-good {{ color:#2d8a4e; font-weight:bold; }}
            #wx-panel .wx-ok {{ color:#b8860b; }}
            #wx-panel .wx-bad {{ color:#999; }}
            #wx-panel .wx-burst {{ color:#0066cc; margin-top:4px; font-weight:bold; }}
            #wx-panel .wx-best {{ margin-top:4px; color:#333; }}
          </style>
          <b>&#127780; Météo prospection</b>
          <table>{rows}</table>
          {burst_html}
          {best_html}
        </div>
        """
        folium_map.get_root().html.add_child(Element(html))
        logger.info("✅ Panneau météo : %d jours affichés", len(self._weather_days))

    # ──────────────────────────────────────────────────────
    # Hotspots
    # ──────────────────────────────────────────────────────
    def _add_hotspot_markers(self, folium_map: folium.Map) -> None:
        if not self.hotspots:
            self._hotspot_count_on_map = 0
            return

        hg = folium.FeatureGroup(name="🎯 Hotspots à prospecter")
        n_total = min(len(self.hotspots), self.max_hotspot_markers)

        for rank, h in enumerate(self.hotspots[:n_total]):
            lon, lat = self._l93_to_wgs84(h["x_l93"], h["y_l93"])
            ms: float = float(h.get("mean_score", 0.0))

            color = (
                "darkgreen" if ms >= 0.75
                else ("green" if ms >= 0.60 else "orange")
            )
            icon_name = (
                "star" if ms >= 0.75
                else ("ok-sign" if ms >= 0.60 else "question-sign")
            )

            popup_lines: list[str] = [
                f"<b>Hotspot #{h.get('id', '?')}</b><br>",
                f"Score moyen : <b>{ms:.3f}</b><br>",
                f"Surface : {h.get('size_m2', 0):.0f} m²<br>",
            ]
            if h.get("altitude") is not None:
                popup_lines.append(f"Altitude : {h['altitude']:.0f} m<br>")
            if h.get("mean_slope") is not None:
                popup_lines.append(
                    f"Pente moy. : {h['mean_slope']:.1f}°<br>",
                )
            if h.get("dominant_tree"):
                popup_lines.append(
                    f"Essence dom. : {h['dominant_tree']}<br>",
                )
            if h.get("geology"):
                popup_lines.append(f"Géologie : {h['geology']}<br>")

            marker = folium.Marker(
                location=[lat, lon],
                popup=folium.Popup("".join(popup_lines), max_width=300),
                tooltip=f"🍄 Score : {ms:.2f}",
                icon=folium.Icon(color=color, icon=icon_name),
            )
            marker.options["className"] = (
                f"cartom-hotspot cartom-rank-{rank}"
            )
            marker.add_to(hg)

        hg.add_to(folium_map)
        self._hotspot_count_on_map = n_total
        logger.info("✅ Hotspots affichés : %d", n_total)

    # ──────────────────────────────────────────────────────
    # Landmarks
    # ──────────────────────────────────────────────────────
    def _add_landmarks(self, folium_map: folium.Map) -> None:
        lg = folium.FeatureGroup(name="📌 Points de repère", show=True)
        for lm in self.landmarks:
            popup_html = (
                f"<b>{lm['name']}</b><br>{lm.get('info', '')}<br>"
                f"<small>{lm['lat']:.4f}°N, {lm['lon']:.4f}°E</small>"
            )
            folium.Marker(
                location=[lm["lat"], lm["lon"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=lm["name"],
                icon=folium.Icon(
                    color=lm.get("icon_color", "blue"),
                    icon=lm.get("icon", "info-sign"),
                ),
            ).add_to(lg)
        lg.add_to(folium_map)

    # ══════════════════════════════════════════════════════
    # CARTE FOLIUM — point d'entrée principal
    # ══════════════════════════════════════════════════════
    def create_folium_map(
        self,
        output: str | Path = "output/carte_morilles.html",
        *,
        grid_threshold: float | None = None,  # conservé pour compat API
        show_elimination: bool = True,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        m = folium.Map(
            location=[config.MAP_CENTER["lat"], config.MAP_CENTER["lon"]],
            zoom_start=14,
            tiles=None,
            control_scale=True,
        )
        self._add_basemaps(m)

        # ── Phase 1 : préparer les overlays (reprojection séquentielle) ──
        t_prep = time.perf_counter()
        overlay_specs: list[
            tuple[str, Callable[[], bytes], bool, float, int, str]
        ] = []

        overlay_specs.extend(self._prepare_probability())

        if show_elimination:
            overlay_specs.extend(self._prepare_elimination())

        overlay_specs.extend(self._prepare_urban_raster())
        overlay_specs.extend(self._prepare_twi())

        logger.info(
            "⏱️  Préparation overlays (reproj.) : %.2fs — %d couches",
            time.perf_counter() - t_prep,
            len(overlay_specs),
        )

        # ── Phase 2 : encoding PNG parallèle + data PNG ──
        t_enc = time.perf_counter()
        n_workers = min(_PNG_WORKERS, len(overlay_specs) + 1)

        overlay_results: list[
            tuple[str, str, bool, float, int, str]
        ] = []
        data_uri: str = ""

        with ThreadPoolExecutor(
            max_workers=max(1, n_workers),
            thread_name_prefix="png",
        ) as pool:
            # Soumettre les overlays
            future_to_idx = {}
            for idx, (name, render_fn, show, opa, zi, msg) in enumerate(
                overlay_specs,
            ):
                fut = pool.submit(render_fn)
                future_to_idx[fut] = (idx, name, show, opa, zi, msg)

            # Soumettre le data PNG et le slope PNG
            data_future = pool.submit(self._build_data_png_bytes)
            slope_future = pool.submit(self._build_slope_png_bytes)

            # Collecter les overlays (ordre de complétion)
            indexed_results: dict[
                int, tuple[str, str, bool, float, int, str]
            ] = {}
            for fut in as_completed(future_to_idx):
                idx, name, show, opa, zi, msg = future_to_idx[fut]
                png_bytes = fut.result()
                uri = self._png_to_data_uri(png_bytes)
                indexed_results[idx] = (name, uri, show, opa, zi, msg)

            # Restituer l'ordre original
            for idx in sorted(indexed_results):
                overlay_results.append(indexed_results[idx])

            # Data PNG + Slope PNG
            data_png_bytes = data_future.result()
            data_uri = self._png_to_data_uri(data_png_bytes)
            slope_uri = self._png_to_data_uri(slope_future.result())

        logger.info(
            "⏱️  Encoding PNG parallèle : %.2fs — %d workers",
            time.perf_counter() - t_enc,
            n_workers,
        )

        # ── Phase 3 : ajout des ImageOverlay au map ──
        for name, uri, show, opa, zi, msg in overlay_results:
            ImageOverlay(
                image=uri,
                bounds=self._bounds,
                opacity=opa,
                name=name,
                show=show,
                interactive=False,
                zindex=zi,
            ).add_to(m)
            logger.info("✅ %s", msg)

        # ── Couches vectorielles + marqueurs ──
        self._add_hydro_vector_layer(m)
        self._add_inaturalist_layer(m)
        self._add_hotspot_markers(m)
        self._add_landmarks(m)

        # ── Panneau météo (HTML overlay) ──
        self._add_weather_panel(m)

        # ── Plugins Folium ──
        MiniMap(toggle_display=True).add_to(m)
        MousePosition(
            position="bottomright",
            separator=" | ",
            prefix="Curseur :",
        ).add_to(m)
        MeasureControl(
            position="topleft",
            primary_length_unit="meters",
            primary_area_unit="sqmeters",
        ).add_to(m)
        folium.LayerControl(collapsed=False, position="topright").add_to(m)

        # ── Contrôles interactifs (JS + data PNG) ──
        self._add_interactive_controls(m, data_uri, slope_uri)

        m.fit_bounds(self._bounds)

        # ── Sauvegarde ──
        try:
            m.save(str(output))
            size_mb = output.stat().st_size / (1024 * 1024)
            dt = time.perf_counter() - t0
            logger.info(
                "🗺️  Carte sauvegardée : %s (%.1f MB) en %.1fs",
                output, size_mb, dt,
            )
        except OSError:
            logger.exception("Impossible de sauvegarder la carte HTML")
            raise

        return output

    # ══════════════════════════════════════════════════════
    # EXPORT GEOTIFF
    # ══════════════════════════════════════════════════════
    def export_geotiff(
        self,
        output: str | Path = "output/morilles_probability.tif",
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        score: np.ndarray = self._final_score.copy()
        nodata_mask: np.ndarray = np.asarray(~np.isfinite(score))
        score[nodata_mask] = _NODATA

        transform = from_bounds(
            self._xmin_l93, self._ymin_l93,
            self._xmax_l93, self._ymax_l93,
            self.grid.nx, self.grid.ny,
        )

        try:
            with rasterio.open(
                str(output),
                "w",
                driver="GTiff",
                height=self.grid.ny,
                width=self.grid.nx,
                count=1,
                dtype=np.float32,
                crs="EPSG:2154",
                transform=transform,
                nodata=_NODATA,
                compress="lzw",
            ) as dst:
                dst.write(score.astype(np.float32), 1)
                dst.update_tags(
                    DESCRIPTION="Cartomorilles — probabilité morilles",
                    CELL_SIZE=str(config.CELL_SIZE),
                    CRS="EPSG:2154",
                    NODATA=str(_NODATA),
                )
            logger.info("✅ GeoTIFF : %s", output)
        except OSError:
            logger.exception("Impossible d'écrire le GeoTIFF")
            raise

        return output

    # ══════════════════════════════════════════════════════
    # EXPORT GEOPACKAGE
    # ══════════════════════════════════════════════════════
    def export_gpkg_grid(
        self,
        output: str | Path = "output/morilles_grid.gpkg",
        threshold: float = 0.15,
    ) -> Path:
        """Export GeoPackage vectorisé (numpy batch)."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Export GeoPackage (seuil=%.2f)…", threshold)

        score: np.ndarray = self._final_score
        prob_classes: np.ndarray = self._prob_classes
        elim_mask: np.ndarray = self._elim_mask
        half = config.CELL_SIZE / 2.0
        labels = list(config.PROBABILITY_LABELS)

        above = np.isfinite(score) & (score >= threshold)
        iy_arr, ix_arr = np.where(above)

        if iy_arr.size == 0:
            logger.warning("Aucune cellule au-dessus du seuil %.2f", threshold)
            return output

        n = iy_arr.size
        logger.info("  %d cellules à exporter", n)

        cx_arr = np.asarray(self.grid.x_coords)[ix_arr]
        cy_arr = np.asarray(self.grid.y_coords)[iy_arr]

        geometries = [
            shapely_box(cx - half, cy - half, cx + half, cy + half)
            for cx, cy in zip(cx_arr.tolist(), cy_arr.tolist())
        ]

        score_vals = score[iy_arr, ix_arr]
        cls_vals = prob_classes[iy_arr, ix_arr].astype(int)

        data: dict[str, Any] = {
            "score": np.round(score_vals, 4),
            "prob_class": cls_vals,
            "prob_label": [
                labels[c] if 0 <= c < len(labels) else "?"
                for c in cls_vals.tolist()
            ],
            "eliminated": elim_mask[iy_arr, ix_arr].astype(bool),
        }

        alt_grid = getattr(self.grid, "altitude", None)
        if isinstance(alt_grid, np.ndarray) and alt_grid.shape == score.shape:
            av = alt_grid[iy_arr, ix_arr].astype(np.float64)
            av[~np.isfinite(av)] = np.nan
            data["altitude"] = np.round(av, 1)

        slp_grid = getattr(self.grid, "slope", None)
        if isinstance(slp_grid, np.ndarray) and slp_grid.shape == score.shape:
            sv = slp_grid[iy_arr, ix_arr].astype(np.float64)
            sv[~np.isfinite(sv)] = np.nan
            data["slope"] = np.round(sv, 2)

        conf_dict: dict[str, Any] = getattr(
            self.grid, "score_confidence", {},
        )
        if isinstance(conf_dict, dict):
            for crit, conf_val in conf_dict.items():
                if isinstance(conf_val, (int, float)):
                    data[f"conf_{crit}"] = round(float(conf_val), 2)

        scores_dict: dict[str, Any] = getattr(self.grid, "scores", {})
        if isinstance(scores_dict, dict):
            for crit_name, crit_arr in scores_dict.items():
                if (
                    isinstance(crit_arr, np.ndarray)
                    and crit_arr.shape == score.shape
                ):
                    cv = crit_arr[iy_arr, ix_arr].astype(np.float64)
                    cv[~np.isfinite(cv)] = np.nan
                    data[f"s_{crit_name}"] = np.round(cv, 3)

        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:2154")

        try:
            gdf.to_file(str(output), driver="GPKG")
            logger.info(
                "✅ GeoPackage : %s (%s cellules)",
                output, f"{len(gdf):,}",
            )
        except OSError:
            logger.exception("Impossible d'écrire le GeoPackage")
            raise

        return output