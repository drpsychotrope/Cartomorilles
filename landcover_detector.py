"""
landcover_detector.py — Détection landcover par tuiles OSM raster.

Télécharge les tuiles OpenStreetMap couvrant la zone d'étude, analyse
les couleurs HSV pour classifier chaque pixel (végétation, urbain, eau,
route), puis rééchantillonne à la résolution de la grille cible.

Interface attendue par grid_builder.apply_landcover_mask() :
    {
        "urban_mask":  np.ndarray bool    (ny, nx),
        "green_score": np.ndarray float32 (ny, nx),
        "landcover":   np.ndarray int8    (ny, nx),
        "quality":     dict  — métriques qualité de la détection,
    }

Limitations connues :
- Les labels texte OSM (noms de villes, rivières) créent du bruit.
  Préférer un serveur de tuiles sans labels si disponible.
- Calibré pour le style OSM Carto standard.

v2.3.0 — Fix #18 : detect_from_cache() cache-only (aucun téléchargement)
         Fix #19 : _get_altitude_mask() intégré dans _build_masks()
         Fix #20 : uniform_filter supprimé (import inutilisé)
         Fix #21 : compteurs tuiles thread-safe (threading.Lock)
         Fix #22 : session HTTP close() + context manager __enter__/__exit__
         Bonus   : _reset_counters() en début de pipeline, logging allégé
v2.2.0 — Fix #11 : farmland 0.10, background 0.05
         Fix duplicats : _CLASS_NAMES/_GREEN_SCORES/_DEBUG_COLORS unifiés
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import rgb_to_hsv  # vectorisé numpy
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    generate_binary_structure,
    median_filter,
    zoom as ndizoom,
)

import config

logger = logging.getLogger("cartomorilles.landcover")

__all__ = ["LandcoverDetector"]

# ── Constantes ─────────────────────────────────────────────────────

_DATA_DIR: Path = Path(getattr(config, "DATA_DIR", "data"))
_OUTPUT_DIR: Path = Path(getattr(config, "OUTPUT_DIR", "output"))

# ── Serveurs de tuiles ────────────────────────────────────────────
#  Ordre de préférence :
#    1. OSM standard avec sous-domaines (a/b/c)
#    2. CartoDB Voyager sans labels
#    3. CartoDB Positron (light)
#
#  wmflabs.org est MORT — ne plus l'utiliser.

_TILE_SERVERS: tuple[str, ...] = (
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
)

_OSM_SUBDOMAINS: tuple[str, ...] = ("a", "b", "c")

_USER_AGENT: str = getattr(
    config,
    "TILE_USER_AGENT",
    "Cartomorilles/2.0 (https://github.com/cartomorilles; contact@example.com)",
)

_TILE_SIZE: int = 256
_MAX_MOSAIC_TILES: int = getattr(config, "MAX_MOSAIC_TILES", 1000)
_MAX_DOWNLOAD_WORKERS: int = 2  # Respecter la politique OSM
_RETRY_COUNT: int = 3
_RETRY_BACKOFF: float = 2.0
_RATE_LIMIT_DELAY: float = 0.15  # secondes entre requêtes réussies

# Taille minimale d'un fichier PNG valide (~67 octets pour un PNG 1×1)
_MIN_TILE_FILE_SIZE: int = 100

# Structure 3×3 (8-connexité) pour dilatation morphologique
_STRUCT_8CONN = generate_binary_structure(2, 2)

# ── Règles de classification HSV ──────────────────────────────────
# Priorité = ordre dans la liste : premier match gagne.
#
# Classes (i+1 car result[match]=i+1) :
#   0 = other (indéterminé)          5 = road_major
#   1 = green_dark (forêt)           6 = road_major2 (hue wrap)
#   2 = green_light (prairie, parc)  7 = road_yellow
#   3 = green_pale (jardin pâle)     8 = farmland
#   4 = water                        9 = background
#  -1 = nodata (alpha invalide)

_COLOR_RULES: tuple[tuple[str, float, float, float, float, float, float], ...] = (
    # (nom,        H_min, H_max, S_min, S_max, V_min, V_max)
    ("green_dark",    70, 170, 0.10, 1.00, 0.15, 0.70),   # forêt
    ("green_light",   60, 160, 0.08, 0.60, 0.70, 0.95),   # prairie, parc
    ("green_pale",    60, 150, 0.03, 0.15, 0.85, 1.00),   # jardin pâle OSM
    ("water",        180, 260, 0.10, 1.00, 0.30, 1.00),   # eau
    ("road_major",   330, 360, 0.05, 0.50, 0.70, 1.00),   # route rose/rouge
    ("road_major2",    0,  15, 0.05, 0.50, 0.70, 1.00),   # route rose (hue wrap)
    ("road_yellow",   35,  55, 0.10, 0.70, 0.80, 1.00),   # route secondaire jaune
    ("farmland",      25,  65, 0.02, 0.12, 0.85, 1.00),   # champs beige pâle OSM
    ("background",     0, 360, 0.00, 0.05, 0.80, 1.00),   # fond blanc/gris clair
)

# ── Mapping unique classe_id → nom (9 classes + other) ────────────
_CLASS_NAMES: dict[int, str] = {
    0: "other",
    1: "green_dark",
    2: "green_light",
    3: "green_pale",
    4: "water",
    5: "road_major",
    6: "road_major2",
    7: "road_yellow",
    8: "farmland",
    9: "background",
}

# ── Scores de verdure par classe — Fix #11 v2.2.0 ────────────────
_GREEN_SCORES: dict[int, float] = {
    0: 0.30,   # indéterminé — neutre
    1: 1.00,   # forêt certaine
    2: 0.80,   # prairie
    3: 0.50,   # jardin, parc
    4: 0.10,   # eau
    5: 0.00,   # route
    6: 0.00,   # route
    7: 0.00,   # route
    8: 0.10,   # champs — Fix #11 (était 0.55)
    9: 0.05,   # fond de carte — Fix #11 (était 0.20)
}

# Classes considérées comme routes pour le masque urbain
_ROAD_CLASSES: frozenset[int] = frozenset({5, 6, 7})

# ── Couleurs pour l'image de debug ────────────────────────────────
_DEBUG_COLORS: dict[int, tuple[int, int, int]] = {
    0: (200, 200, 200),    # other
    1: (0, 100, 0),        # green_dark
    2: (100, 200, 100),    # green_light
    3: (180, 230, 180),    # green_pale
    4: (50, 50, 200),      # water
    5: (255, 100, 100),    # road_major
    6: (255, 100, 100),    # road_major2
    7: (255, 200, 50),     # road_yellow
    8: (230, 220, 180),    # farmland
    9: (240, 240, 240),    # background
}


# ═══════════════════════════════════════════════════════════════════
class LandcoverDetector:
    """
    Détecte les zones naturelles vs urbaines par analyse colorimétrique
    des tuiles OpenStreetMap.

    Calibré pour le style OSM Carto standard et CartoDB Positron.
    Utilisable en context manager (fermeture auto de la session HTTP).

    Exemple::

        with LandcoverDetector(zoom=14, debug=True) as lcd:
            lcd.set_terrain_grids(altitude=alt_arr, slope=slope_arr)
            result = lcd.detect(target_shape=(ny, nx))
    """

    def __init__(
        self,
        zoom: int | None = None,
        tile_url: str | None = None,
        prefer_no_labels: bool = True,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.prefer_no_labels = prefer_no_labels

        # Zoom auto-adaptatif
        if zoom is not None:
            self.zoom = zoom
        else:
            self.zoom = self._optimal_zoom(
                getattr(config, "CELL_SIZE", 5.0),
                getattr(config, "MAP_CENTER", {"lat": 45.235}).get(
                    "lat", 45.235
                )
                if isinstance(getattr(config, "MAP_CENTER", None), dict)
                else 45.235,
            )

        # URL du serveur — construire la liste de fallback
        if tile_url is not None:
            self._tile_servers: tuple[str, ...] = (tile_url,)
        else:
            self._tile_servers = _TILE_SERVERS

        # Cache
        self._tile_cache_dir = _DATA_DIR / "osm_tiles"
        self._session: Any = None  # Lazy init — Fix #22

        # Compteurs thread-safe — Fix #21
        self._counter_lock = threading.Lock()
        self._tiles_total: int = 0
        self._tiles_downloaded: int = 0
        self._tiles_cached: int = 0
        self._tiles_failed: int = 0

    # ── Session HTTP — Fix #22 ────────────────────────────────

    def _get_session(self) -> Any:
        """Retourne un requests.Session réutilisable."""
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": _USER_AGENT,
                    "Accept": "image/png,image/*;q=0.9,*/*;q=0.8",
                    "Accept-Language": "fr,en;q=0.5",
                    "Referer": "https://www.openstreetmap.org/",
                }
            )
        return self._session

    def close(self) -> None:
        """Ferme la session HTTP et libère les ressources."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        logger.debug("Session HTTP fermée")

    def __enter__(self) -> LandcoverDetector:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── Zoom auto ─────────────────────────────────────────────

    @staticmethod
    def _optimal_zoom(cell_size: float, lat: float) -> int:
        """Calcule le zoom OSM optimal pour une résolution cible."""
        meters_per_pixel_z0 = 40075016.0 * math.cos(math.radians(lat)) / 256
        for z in range(1, 20):
            mpp = meters_per_pixel_z0 / (2**z)
            if mpp <= cell_size:
                return z
        return 18

    # ── Compteurs thread-safe — Fix #21 ──────────────────────

    def _reset_counters(self) -> None:
        """Remet les compteurs à zéro en début de pipeline."""
        with self._counter_lock:
            self._tiles_total = 0
            self._tiles_downloaded = 0
            self._tiles_cached = 0
            self._tiles_failed = 0

    def _inc_downloaded(self) -> None:
        with self._counter_lock:
            self._tiles_downloaded += 1

    def _inc_cached(self) -> None:
        with self._counter_lock:
            self._tiles_cached += 1

    def _inc_failed(self) -> None:
        with self._counter_lock:
            self._tiles_failed += 1

    # ═══════════════════════════════════════════════════════════════
    # PIPELINE PRINCIPAL
    # ═══════════════════════════════════════════════════════════════

    def detect(
        self,
        bbox_wgs84: dict[str, float] | None = None,
        bbox_l93: dict[str, float] | None = None,
        target_shape: tuple[int, int] | None = None,
        *,
        _cache_only: bool = False,
    ) -> dict[str, Any]:
        """
        Pipeline complet de détection landcover.

        Parameters
        ----------
        bbox_wgs84 : dict, optional
            Emprise WGS84 (west, east, south, north).
        bbox_l93 : dict, optional
            Emprise Lambert-93 (xmin, xmax, ymin, ymax).
        target_shape : tuple (ny, nx), optional
            Shape cible pour le rééchantillonnage.
        _cache_only : bool
            Si True, aucun téléchargement réseau (Fix #18).

        Returns
        -------
        dict avec urban_mask, green_score, landcover, quality
        """
        # Reset compteurs en début de pipeline
        self._reset_counters()

        # Résolution des arguments
        if bbox_wgs84 is None:
            bbox_wgs84 = dict(config.BBOX_WGS84)
        if bbox_l93 is None:
            bbox_l93 = dict(config.BBOX)

        # Shape cible
        if target_shape is not None:
            ny, nx = target_shape
        else:
            cell_size = getattr(config, "CELL_SIZE", 5.0)
            nx = int((bbox_l93["xmax"] - bbox_l93["xmin"]) / cell_size)
            ny = int((bbox_l93["ymax"] - bbox_l93["ymin"]) / cell_size)

        logger.info(
            "Détection landcover par tuiles OSM (zoom=%d, cache_only=%s)",
            self.zoom,
            _cache_only,
        )

        # Vérification réseau — Fix #18 : skip si cache-only
        if not _cache_only and not self._check_network():
            logger.warning("Réseau indisponible — retour de masques neutres")
            return self._neutral_result(ny, nx)

        # Créer le répertoire de cache
        self._tile_cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Tuiles nécessaires
        tiles = self._get_tile_coords(bbox_wgs84)
        if not tiles:
            logger.warning("Aucune tuile à télécharger — BBOX trop petit ?")
            return self._neutral_result(ny, nx)

        with self._counter_lock:
            self._tiles_total = len(tiles)
        logger.info("  %d tuiles nécessaires", len(tiles))

        # Limite mémoire
        if len(tiles) > _MAX_MOSAIC_TILES:
            logger.warning(
                "  Trop de tuiles (%d > %d) — réduction du zoom",
                len(tiles),
                _MAX_MOSAIC_TILES,
            )
            self.zoom = max(10, self.zoom - 1)
            tiles = self._get_tile_coords(bbox_wgs84)
            with self._counter_lock:
                self._tiles_total = len(tiles)

        # 2. Télécharger et assembler
        mosaic, alpha_mask = self._download_and_assemble(
            tiles, cache_only=_cache_only
        )
        if mosaic is None:
            logger.error("Mosaïque vide — retour de masques neutres")
            return self._neutral_result(ny, nx)

        logger.info(
            "  Mosaïque : %d×%d px (%.1f Mo)",
            mosaic.shape[1],
            mosaic.shape[0],
            mosaic.nbytes / 1024 / 1024,
        )

        # Vérification mosaïque non-blanche
        if self._is_blank_mosaic(mosaic, alpha_mask):
            logger.warning("Mosaïque entièrement blanche/vide — masques neutres")
            return self._neutral_result(ny, nx)

        # 3. Géoréférencer et cropper (Mercator)
        mosaic_bounds = self._tiles_to_bounds(tiles)
        cropped, cropped_alpha = self._crop_to_bbox_mercator(
            mosaic,
            alpha_mask,
            mosaic_bounds,
            bbox_wgs84,
        )
        logger.info("  Crop : %d×%d px", cropped.shape[1], cropped.shape[0])

        # 4. Classifier les couleurs
        classification = self._classify_colors(cropped, cropped_alpha)

        # 5. Rééchantillonner à la grille L93
        class_resized = self._resample_to_grid(classification, ny, nx)

        # 6. Construire les masques — Fix #19 : altitude intégrée
        urban_mask, green_score = self._build_masks(
            class_resized, ny, nx, bbox_l93
        )

        # 7. Métriques qualité
        quality = self._compute_quality_metrics(classification, class_resized)

        # 8. Debug
        if self.debug:
            self._save_debug_images(cropped, classification, class_resized)

        result: dict[str, Any] = {
            "urban_mask": urban_mask,
            "green_score": green_score,
            "landcover": class_resized,
            "quality": quality,
        }

        logger.info(
            "  Résultat : urbain=%.1f%%, vert(>0.5)=%.1f%%",
            urban_mask.sum() / max(urban_mask.size, 1) * 100,
            (green_score > 0.5).sum() / max(green_score.size, 1) * 100,
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # RÉSULTAT NEUTRE (fallback)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _neutral_result(ny: int, nx: int) -> dict[str, Any]:
        """Retourne des masques neutres quand la détection échoue."""
        return {
            "urban_mask": np.zeros((ny, nx), dtype=bool),
            "green_score": np.full((ny, nx), 0.3, dtype=np.float32),
            "landcover": np.zeros((ny, nx), dtype=np.int8),
            "quality": {
                "tiles_total": 0,
                "tiles_ok": 0,
                "tiles_failed": 0,
                "pct_classified": 0.0,
                "confidence": 0.0,
            },
        }

    # ═══════════════════════════════════════════════════════════════
    # RÉSEAU
    # ═══════════════════════════════════════════════════════════════

    def _check_network(self) -> bool:
        """Vérifie l'accès réseau via les serveurs configurés."""
        test_urls = [
            "https://a.tile.openstreetmap.org/0/0/0.png",
            "https://basemaps.cartocdn.com/rastertiles/voyager/0/0/0.png",
        ]
        session = self._get_session()
        for url in test_urls:
            try:
                resp = session.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.debug("  Réseau OK via %s", url.split("/")[2])
                    return True
            except Exception:
                continue
        return False

    # ═══════════════════════════════════════════════════════════════
    # TUILES — COORDONNÉES
    # ═══════════════════════════════════════════════════════════════

    def _get_tile_coords(
        self,
        bbox_wgs84: dict[str, float],
    ) -> list[tuple[int, int, int]]:
        """Calcule les coordonnées de tuiles couvrant le BBOX."""
        z = self.zoom
        x_min = self._lon_to_tile_x(bbox_wgs84["west"], z)
        x_max = self._lon_to_tile_x_ceil(bbox_wgs84["east"], z)
        y_min = self._lat_to_tile_y(bbox_wgs84["north"], z)
        y_max = self._lat_to_tile_y_ceil(bbox_wgs84["south"], z)

        tiles = [
            (z, tx, ty)
            for tx in range(x_min, x_max + 1)
            for ty in range(y_min, y_max + 1)
        ]
        return tiles

    @staticmethod
    def _lon_to_tile_x(lon: float, zoom: int) -> int:
        return int((lon + 180.0) / 360.0 * (2**zoom))

    @staticmethod
    def _lon_to_tile_x_ceil(lon: float, zoom: int) -> int:
        return math.ceil((lon + 180.0) / 360.0 * (2**zoom)) - 1

    @staticmethod
    def _lat_to_tile_y(lat: float, zoom: int) -> int:
        lat_rad = math.radians(lat)
        return int(
            (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
            / 2
            * (2**zoom)
        )

    @staticmethod
    def _lat_to_tile_y_ceil(lat: float, zoom: int) -> int:
        lat_rad = math.radians(lat)
        raw = (
            (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
            / 2
            * (2**zoom)
        )
        return math.ceil(raw) - 1

    @staticmethod
    def _tile_to_lon(x: int, zoom: int) -> float:
        return x / (2**zoom) * 360.0 - 180.0

    @staticmethod
    def _tile_to_lat(y: int, zoom: int) -> float:
        n = math.pi - 2 * math.pi * y / (2**zoom)
        return math.degrees(math.atan(math.sinh(n)))

    @staticmethod
    def _lat_to_mercator_y(lat: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    def _tiles_to_bounds(
        self,
        tiles: list[tuple[int, int, int]],
    ) -> dict[str, float]:
        z = tiles[0][0]
        xs = [t[1] for t in tiles]
        ys = [t[2] for t in tiles]
        return {
            "west": self._tile_to_lon(min(xs), z),
            "east": self._tile_to_lon(max(xs) + 1, z),
            "north": self._tile_to_lat(min(ys), z),
            "south": self._tile_to_lat(max(ys) + 1, z),
        }

    # ═══════════════════════════════════════════════════════════════
    # TÉLÉCHARGEMENT ET ASSEMBLAGE
    # ═══════════════════════════════════════════════════════════════

    def _resolve_tile_url(self, template: str, z: int, x: int, y: int) -> str:
        """Résout un template de tuile avec rotation de sous-domaines."""
        return template.format(z=z, x=x, y=y, s=random.choice(_OSM_SUBDOMAINS))

    def _download_and_assemble(
        self,
        tiles: list[tuple[int, int, int]],
        *,
        cache_only: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Télécharge les tuiles et les assemble en mosaïque."""
        z = tiles[0][0]
        x_set = sorted({t[1] for t in tiles})
        y_set = sorted({t[2] for t in tiles})
        x_idx = {x: i for i, x in enumerate(x_set)}
        y_idx = {y: i for i, y in enumerate(y_set)}

        h_total = len(y_set) * _TILE_SIZE
        w_total = len(x_set) * _TILE_SIZE
        mosaic = np.full((h_total, w_total, 3), 255, dtype=np.uint8)
        alpha = np.zeros((h_total, w_total), dtype=bool)

        def _fetch_tile(
            tile_coord: tuple[int, int, int],
        ) -> tuple[int, int, np.ndarray, bool]:
            _, tx, ty = tile_coord
            img_arr, is_valid = self._download_single_tile(
                z, tx, ty, cache_only=cache_only
            )
            return (tx, ty, img_arr, is_valid)

        with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as executor:
            futures = {executor.submit(_fetch_tile, t): t for t in tiles}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                if done_count % 50 == 0 or done_count == len(tiles):
                    logger.info(
                        "  Tuiles : %d/%d (%.0f%%)",
                        done_count,
                        len(tiles),
                        done_count / len(tiles) * 100,
                    )
                try:
                    tx, ty, img_arr, is_valid = future.result()
                except Exception as exc:
                    logger.warning("  Erreur tuile : %s", exc)
                    self._inc_failed()
                    continue

                ix = x_idx[tx]
                iy = y_idx[ty]
                y0 = iy * _TILE_SIZE
                x0 = ix * _TILE_SIZE
                h = min(img_arr.shape[0], _TILE_SIZE)
                w = min(img_arr.shape[1], _TILE_SIZE)
                mosaic[y0 : y0 + h, x0 : x0 + w] = img_arr[:h, :w, :3]
                if is_valid:
                    alpha[y0 : y0 + h, x0 : x0 + w] = True

        with self._counter_lock:
            dl = self._tiles_downloaded
            ca = self._tiles_cached
            fa = self._tiles_failed
        logger.info(
            "  Téléchargement : %d ok, %d cache, %d échecs",
            dl,
            ca,
            fa,
        )

        if not alpha.any():
            return None, None

        return mosaic, alpha

    def _download_single_tile(
        self,
        z: int,
        tx: int,
        ty: int,
        *,
        cache_only: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """
        Télécharge une tuile avec fallback multi-serveurs.

        Fix #18 : si *cache_only* est True, ne tente aucun téléchargement
        réseau ; retourne un blank si la tuile n'est pas en cache.
        """
        cache_file = self._tile_cache_dir / f"{z}_{tx}_{ty}.png"

        # ── Cache ─────────────────────────────────────────────
        if cache_file.exists() and cache_file.stat().st_size >= _MIN_TILE_FILE_SIZE:
            try:
                img = Image.open(cache_file)
                arr = self._image_to_rgb(img)
                self._inc_cached()
                return arr, True
            except Exception:
                logger.debug("  Cache corrompu : %s", cache_file)
                cache_file.unlink(missing_ok=True)

        # ── Fix #18 : cache-only → pas de téléchargement ─────
        if cache_only:
            self._inc_failed()
            blank = np.full((_TILE_SIZE, _TILE_SIZE, 3), 255, dtype=np.uint8)
            return blank, False

        # ── Téléchargement avec fallback multi-serveurs ───────
        session = self._get_session()

        for server_idx, server_template in enumerate(self._tile_servers):
            for attempt in range(_RETRY_COUNT):
                try:
                    url = self._resolve_tile_url(server_template, z, tx, ty)
                    resp = session.get(url, timeout=10)

                    if (
                        resp.status_code == 200
                        and len(resp.content) >= _MIN_TILE_FILE_SIZE
                    ):
                        cache_file.write_bytes(resp.content)
                        img = Image.open(cache_file)
                        arr = self._image_to_rgb(img)
                        self._inc_downloaded()
                        time.sleep(_RATE_LIMIT_DELAY)
                        return arr, True

                    if resp.status_code == 429:
                        wait = _RETRY_BACKOFF * (2**attempt)
                        logger.debug(
                            "  429 tuile %d/%d/%d srv#%d — %.1fs",
                            z,
                            tx,
                            ty,
                            server_idx,
                            wait,
                        )
                        time.sleep(wait)
                        continue

                    # 4xx/5xx → tenter serveur suivant directement
                    if resp.status_code >= 400:
                        logger.debug(
                            "  HTTP %d tuile %d/%d/%d srv#%d",
                            resp.status_code,
                            z,
                            tx,
                            ty,
                            server_idx,
                        )
                        break  # serveur suivant

                except Exception as exc:
                    wait = _RETRY_BACKOFF * (2**attempt)
                    logger.debug(
                        "  Erreur tuile %d/%d/%d srv#%d att%d: %s",
                        z,
                        tx,
                        ty,
                        server_idx,
                        attempt + 1,
                        exc,
                    )
                    time.sleep(wait)

        # ── Échec total ───────────────────────────────────────
        logger.warning(
            "  Tuile %d/%d/%d : échec sur %d serveurs",
            z,
            tx,
            ty,
            len(self._tile_servers),
        )
        self._inc_failed()
        blank = np.full((_TILE_SIZE, _TILE_SIZE, 3), 255, dtype=np.uint8)
        return blank, False

    @staticmethod
    def _image_to_rgb(img: Image.Image) -> np.ndarray:
        """Convertit une image PIL en array RGB."""
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr[:, :, :3]

    # ═══════════════════════════════════════════════════════════════
    # CROP — CORRIGÉ MERCATOR
    # ═══════════════════════════════════════════════════════════════

    def _crop_to_bbox_mercator(
        self,
        mosaic: np.ndarray,
        alpha: np.ndarray | None,
        mosaic_bounds: dict[str, float],
        target_bbox: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        h, w = mosaic.shape[:2]

        x_frac_min = (target_bbox["west"] - mosaic_bounds["west"]) / (
            mosaic_bounds["east"] - mosaic_bounds["west"]
        )
        x_frac_max = (target_bbox["east"] - mosaic_bounds["west"]) / (
            mosaic_bounds["east"] - mosaic_bounds["west"]
        )

        merc_north = self._lat_to_mercator_y(mosaic_bounds["north"])
        merc_south = self._lat_to_mercator_y(mosaic_bounds["south"])
        merc_target_north = self._lat_to_mercator_y(target_bbox["north"])
        merc_target_south = self._lat_to_mercator_y(target_bbox["south"])

        merc_range = merc_north - merc_south
        if abs(merc_range) < 1e-10:
            return mosaic, alpha

        y_frac_min = (merc_north - merc_target_north) / merc_range
        y_frac_max = (merc_north - merc_target_south) / merc_range

        x0 = max(0, int(x_frac_min * w))
        x1 = min(w, int(x_frac_max * w))
        y0 = max(0, int(y_frac_min * h))
        y1 = min(h, int(y_frac_max * h))

        if x1 <= x0 or y1 <= y0:
            logger.warning("Crop invalide — retour mosaïque complète")
            return mosaic, alpha

        cropped = mosaic[y0:y1, x0:x1]
        cropped_alpha = alpha[y0:y1, x0:x1] if alpha is not None else None
        return cropped, cropped_alpha

    # ═══════════════════════════════════════════════════════════════
    # CLASSIFICATION HSV
    # ═══════════════════════════════════════════════════════════════

    def _classify_colors(
        self,
        rgb_image: np.ndarray,
        alpha_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Classifie chaque pixel en catégorie landcover via HSV."""
        rgb_norm = rgb_image.astype(np.float32) / 255.0
        hsv: np.ndarray = np.asarray(rgb_to_hsv(rgb_norm))  # P5

        hue = hsv[:, :, 0] * 360.0
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        img_h, img_w = rgb_image.shape[0], rgb_image.shape[1]
        result = np.zeros((img_h, img_w), dtype=np.int8)

        for i, (_name, h_min, h_max, s_min, s_max, v_min, v_max) in enumerate(
            _COLOR_RULES
        ):
            if h_min <= h_max:
                h_match = (hue >= h_min) & (hue <= h_max)
            else:
                h_match = (hue >= h_min) | (hue <= h_max)

            match = (
                h_match
                & (sat >= s_min)
                & (sat <= s_max)
                & (val >= v_min)
                & (val <= v_max)
            )
            # Premier match gagne — ne pas écraser les pixels déjà classés
            result[match & (result == 0)] = np.int8(i + 1)

        # Nodata là où alpha indique des tuiles manquantes
        if alpha_mask is not None:
            alpha_resized = alpha_mask
            if alpha_resized.shape != result.shape:
                alpha_resized = (
                    np.asarray(
                        ndizoom(
                            alpha_mask.astype(np.float32),
                            (
                                img_h / alpha_mask.shape[0],
                                img_w / alpha_mask.shape[1],
                            ),
                            order=0,
                        )
                    )
                    > 0.5
                )
            result[~alpha_resized] = -1

        # Logging allégé : résumé uniquement
        total_valid = int((result >= 0).sum())
        if total_valid > 0:
            classified = int((result > 0).sum())
            logger.info(
                "  Classification : %d/%d px classés (%.1f%%)",
                classified,
                total_valid,
                classified / total_valid * 100,
            )
            if logger.isEnabledFor(logging.DEBUG):
                for cls_id, cls_name in _CLASS_NAMES.items():
                    cnt = int((result == cls_id).sum())
                    if cnt > 0:
                        logger.debug(
                            "    %-15s : %8d px (%.1f%%)",
                            cls_name,
                            cnt,
                            cnt / total_valid * 100,
                        )
                nodata_cnt = int((result == -1).sum())
                if nodata_cnt > 0:
                    logger.debug("    %-15s : %8d px", "nodata", nodata_cnt)

        return result

    # ═══════════════════════════════════════════════════════════════
    # RÉÉCHANTILLONNAGE
    # ═══════════════════════════════════════════════════════════════

    def _resample_to_grid(
        self,
        classification: np.ndarray,
        ny: int,
        nx: int,
    ) -> np.ndarray:
        if classification.shape == (ny, nx):
            return classification

        resized = np.asarray(
            ndizoom(
                classification.astype(np.float32),
                (ny / classification.shape[0], nx / classification.shape[1]),
                order=0,
            ),
            dtype=np.int8,
        )
        return resized

    # ═══════════════════════════════════════════════════════════════
    # CONSTRUCTION DES MASQUES — Fix #19 : altitude intégrée
    # ═══════════════════════════════════════════════════════════════

    def _build_masks(
        self,
        class_resized: np.ndarray,
        ny: int,
        nx: int,
        bbox_l93: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construit urban_mask et green_score depuis la classification.

        Stratégie corrigée :
        - urban_mask ← UNIQUEMENT routes détectées (dilatées 1px)
        - La détection urbaine réelle vient d'OSM vecteur (apply_urban_mask)
        - "other" et "background" restent neutres — PAS convertis en urbain

        Fix #19 : le masque altitude/pente (si terrain fourni via
        set_terrain_grids) protège les zones de montagne en forçant un
        green_score minimum — évite que les pixels haute altitude soient
        classés « autre » par manque de couleur verte sur les tuiles.
        """
        # ── Green score ───────────────────────────────────────
        green_score = np.full((ny, nx), 0.3, dtype=np.float32)
        for cls_id, score in _GREEN_SCORES.items():
            green_score[class_resized == cls_id] = score
        green_score[class_resized == -1] = 0.3

        # ── Urban mask ← routes uniquement ────────────────────
        urban_mask = np.zeros((ny, nx), dtype=bool)
        for road_cls in _ROAD_CLASSES:
            urban_mask[class_resized == road_cls] = True

        # Dilatation 1px autour des routes (emprise réelle)
        if urban_mask.any():
            urban_mask = np.asarray(
                binary_dilation(
                    urban_mask,
                    structure=_STRUCT_8CONN,
                    iterations=1,
                ),
                dtype=bool,
            )

        # ── Lissage morphologique ─────────────────────────────
        urban_mask = np.asarray(
            median_filter(urban_mask.astype(np.uint8), size=5),
            dtype=np.uint8,
        ).astype(bool)
        green_score = np.asarray(
            median_filter(green_score, size=3),
            dtype=np.float32,
        )

        # ── Fix #19 : protection altitude/pente ──────────────
        # Si set_terrain_grids() a été appelé, on booste green_score
        # dans les zones haute altitude / forte pente non classées
        # comme route/eau — ces zones ont souvent des couleurs ternes
        # sur les tuiles OSM mais sont probablement forestières.
        alt_mask = self._get_altitude_mask(ny, nx, bbox_l93)
        if alt_mask is not None:
            # Ne pas booster les routes ni l'eau
            protect = alt_mask & ~urban_mask & (class_resized != 4)
            green_score[protect] = np.maximum(green_score[protect], 0.4)
            logger.debug(
                "  Protection altitude : %d cellules boostées (%.1f%%)",
                int(protect.sum()),
                protect.sum() / max(protect.size, 1) * 100,
            )

        green_score = np.clip(green_score, 0.0, 1.0)
        green_score[urban_mask] = 0.0

        return urban_mask, green_score

    def _get_altitude_mask(
        self,
        ny: int,
        nx: int,
        bbox_l93: dict[str, float],
    ) -> np.ndarray | None:
        """
        Retourne un masque booléen des cellules en zone de montagne
        (altitude > seuil OU pente > seuil).

        Nécessite un appel préalable à set_terrain_grids().
        Retourne None si les grilles terrain ne sont pas disponibles.
        """
        alt_threshold = getattr(config, "LANDCOVER_ALT_PROTECTION", 400.0)
        slope_threshold = getattr(config, "LANDCOVER_SLOPE_PROTECTION", 20.0)

        alt = getattr(self, "_altitude_grid", None)
        slope = getattr(self, "_slope_grid", None)

        if alt is not None and isinstance(alt, np.ndarray) and alt.shape == (ny, nx):
            mask = np.isfinite(alt) & (alt > alt_threshold)
            if (
                slope is not None
                and isinstance(slope, np.ndarray)
                and slope.shape == (ny, nx)
            ):
                mask |= np.isfinite(slope) & (slope > slope_threshold)
            return mask

        return None

    def set_terrain_grids(
        self,
        altitude: np.ndarray | None = None,
        slope: np.ndarray | None = None,
    ) -> LandcoverDetector:
        """Fournit les grilles terrain pour la modulation contextuelle."""
        if altitude is not None:
            self._altitude_grid = altitude
        if slope is not None:
            self._slope_grid = slope
        return self

    # ═══════════════════════════════════════════════════════════════
    # VÉRIFICATIONS
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _is_blank_mosaic(
        mosaic: np.ndarray,
        alpha: np.ndarray | None,
    ) -> bool:
        if alpha is not None and not alpha.any():
            return True
        white_px = np.all(mosaic > 250, axis=-1).sum()
        return bool(white_px > 0.95 * mosaic.shape[0] * mosaic.shape[1])

    # ═══════════════════════════════════════════════════════════════
    # MÉTRIQUES QUALITÉ
    # ═══════════════════════════════════════════════════════════════

    def _compute_quality_metrics(
        self,
        classification_full: np.ndarray,
        classification_grid: np.ndarray,
    ) -> dict[str, Any]:
        total = int((classification_full >= 0).sum())
        classified = int((classification_full > 0).sum())
        nodata = int((classification_full == -1).sum())

        pct_classified = classified / total * 100 if total > 0 else 0.0

        with self._counter_lock:
            dl = self._tiles_downloaded
            ca = self._tiles_cached
            fa = self._tiles_failed
            tt = self._tiles_total

        tile_success_rate = (dl + ca) / tt if tt > 0 else 0.0
        confidence = min(1.0, tile_success_rate * (pct_classified / 100))

        return {
            "tiles_total": tt,
            "tiles_downloaded": dl,
            "tiles_cached": ca,
            "tiles_failed": fa,
            "pixels_total": total,
            "pixels_classified": classified,
            "pixels_nodata": nodata,
            "pct_classified": round(pct_classified, 1),
            "confidence": round(confidence, 3),
        }

    # ═══════════════════════════════════════════════════════════════
    # CACHE
    # ═══════════════════════════════════════════════════════════════

    def clear_cache(self) -> int:
        """Supprime toutes les tuiles en cache."""
        if not self._tile_cache_dir.exists():
            return 0
        count = 0
        for f in self._tile_cache_dir.glob("*.png"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("Cache tuiles vidé : %d fichiers supprimés", count)
        return count

    # ═══════════════════════════════════════════════════════════════
    # DÉTECTION DEPUIS CACHE — Fix #18
    # ═══════════════════════════════════════════════════════════════

    def detect_from_cache(
        self,
        bbox_wgs84: dict[str, float] | None = None,
        bbox_l93: dict[str, float] | None = None,
        target_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """
        Détection landcover en n'utilisant que les tuiles déjà en cache.

        Fix #18 : utilise le flag ``_cache_only=True`` pour désactiver
        tout téléchargement réseau — les tuiles absentes du cache sont
        comptées comme échecs et remplies en blanc (nodata).
        """
        logger.info("Détection landcover depuis cache uniquement")
        return self.detect(
            bbox_wgs84=bbox_wgs84,
            bbox_l93=bbox_l93,
            target_shape=target_shape,
            _cache_only=True,
        )

    # ═══════════════════════════════════════════════════════════════
    # DEBUG
    # ═══════════════════════════════════════════════════════════════

    def _save_debug_images(
        self,
        rgb_image: np.ndarray,
        classification_full: np.ndarray,
        classification_grid: np.ndarray,
    ) -> None:
        output_dir = _OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.fromarray(rgb_image)
            img.save(output_dir / "landcover_osm_tiles.png")
        except Exception as exc:
            logger.warning("Erreur sauvegarde debug RGB : %s", exc)

        self._save_class_image(
            classification_full,
            output_dir / "landcover_classification_full.png",
        )
        self._save_class_image(
            classification_grid,
            output_dir / "landcover_classification_grid.png",
        )
        logger.info("  Debug images sauvegardées dans %s/", output_dir)

    @staticmethod
    def _save_class_image(classification: np.ndarray, path: Path) -> None:
        try:
            h, w = classification.shape
            debug_rgb = np.full((h, w, 3), 128, dtype=np.uint8)
            for cls_id, color in _DEBUG_COLORS.items():
                mask = classification == cls_id
                debug_rgb[mask] = color
            Image.fromarray(debug_rgb).save(path)
        except Exception as exc:
            logger.warning("Erreur sauvegarde debug classe : %s", exc)