#!/usr/bin/env python3
"""
🍄 CARTOMORILLES — Chargement des données géographiques

Sources de données, par ordre de priorité :
  1. Fichier local fourni par l'utilisateur (--dem, --forest, etc.)
  2. Cache local (data/*.gpkg, data/*.tif) avec hash BBOX
  3. WFS IGN BD Forêt V2 / BRGM Géologie / Copernicus DEM
  4. OSM Overpass (forêt, hydro, urbain)
  5. Données synthétiques (fallback ultime)

Toutes les données sont systématiquement reprojetées en EPSG:2154 (Lambert 93).
Les colonnes de sortie sont normalisées pour grid_builder.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.errors import RasterioIOError
from rasterio.windows import Window  # type: ignore[call-arg]
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.validation import make_valid

import config
from config import (
    BBOX,
    BBOX_WGS84,
    CELL_SIZE,
    DATA_BUFFER,
    WATER_TYPE_BONUS,
    get_geology_score,
    get_tree_score,
    resolve_geology,
    resolve_tree_name,
)

# ═══════════════════════════════════════════════════════════════
#  MODULE-LEVEL
# ═══════════════════════════════════════════════════════════════

logger = logging.getLogger("cartomorilles.data_loader")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TARGET_CRS = "EPSG:2154"

_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

OVERPASS_SERVERS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

# Taille max par téléchargement HTTP (500 Mo)
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024

# ── WFS layer names to try (IGN Géoplateforme) ──
IGN_FORET_LAYERS: tuple[str, ...] = (
    "BDFORET_V2:formation_vegetale",
    "VEGETATION.FORMATION_VEGETALE",
    "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale",
)

IGN_HYDRO_LAYERS: tuple[str, ...] = (
    "BDTOPO_V3:cours_d_eau",
    "HYDROGRAPHY.HYDROGRAPHY:cours_d_eau",
)

BRGM_GEOLOGY_LAYERS: tuple[str, ...] = (
    "GEOLOGIE.CARTE_GEOL_HARM_50.V1:polygones",
    "GEL_S_FGEOL_GEOL_HARM50_000",
    "ms:GEL_S_FGEOL_GEOL_HARM50_000",
)

# ── OSM waterway type → config WATER_TYPE_BONUS key ──
WATERWAY_TYPE_MAP: MappingProxyType[str, str] = MappingProxyType({
    "river": "riviere",
    "stream": "ruisseau",
    "canal": "canal",
    "drain": "canal",
    "ditch": "canal",
    "riverbank": "riviere",
})

WATER_AREA_TYPE_MAP: MappingProxyType[str, str] = MappingProxyType({
    "lake": "plan_eau",
    "pond": "plan_eau",
    "reservoir": "plan_eau",
    "oxbow": "bras_mort",
    "basin": "plan_eau",
})

# ── Filet de sécurité : patterns non couverts par _GEOLOGY_KEYWORD_MAP ──
# Utilisé en étape 2 de _normalize_geology() pour les DESCR résiduelles.
_BDCHARM_DESCR_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("sidérose",          "siliceux"),
    ("sidérite",          "siliceux"),
    ("radiolarite",       "siliceux"),
    ("serpentinite",      "siliceux"),
    ("amphibolite",       "gneiss"),
    ("migmatite",         "gneiss"),
    ("pegmatite",         "granite"),
    ("rhyolite",          "granite"),
    ("dacite",            "granite"),
    ("andésite",          "basalte"),
    ("trachyte",          "basalte"),
    ("phonolite",         "basalte"),
    ("lignite",           "marne"),
    ("houille",           "marne"),
    ("gypse",             "calcaire_lacustre"),
    ("cargneule",         "dolomie"),
    ("cornieule",         "dolomie"),
    ("spilite",           "basalte"),
    ("prasinite",         "schiste"),
    ("cipolin",           "calcaire"),
    ("quartzite",         "siliceux"),
)

# ═══════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES MODULE
# ═══════════════════════════════════════════════════════════════

def _bbox_hash(bbox_wgs84: Mapping[str, float]) -> str:
    """Hash court du BBOX pour identifier les caches de manière unique."""
    raw = (
        f"{bbox_wgs84['west']:.4f}_{bbox_wgs84['south']:.4f}_"
        f"{bbox_wgs84['east']:.4f}_{bbox_wgs84['north']:.4f}"
    )
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _overpass_query(query: str, timeout: int = 90) -> dict[str, Any]:
    """Requête Overpass avec fallback multi-serveurs et gestion rate-limit."""
    for server in OVERPASS_SERVERS:
        server_name = server.split("//")[1].split("/")[0]
        try:
            logger.debug("Overpass: %s...", server_name)
            resp = requests.post(
                server,
                data={"data": query},
                timeout=timeout,
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                # Vérifier les erreurs Overpass dans le JSON
                if "remark" in data and "error" in str(data["remark"]).lower():
                    logger.warning("Overpass erreur: %s", str(data["remark"])[:200])
                    continue
                n = len(data.get("elements", []))
                logger.debug("   ✅ %d éléments depuis %s", n, server_name)
                return data
            elif resp.status_code == 429:
                logger.debug("   ⏳ Rate limited: %s", server_name)
                time.sleep(5)
                continue
            elif resp.status_code == 504:
                logger.debug("   ⏱️ Timeout: %s", server_name)
                continue
            else:
                logger.debug("   ⚠️ HTTP %d: %s", resp.status_code, server_name)
                continue
        except requests.exceptions.Timeout:
            logger.debug("   ⏱️ Timeout: %s", server_name)
        except requests.exceptions.ConnectionError:
            logger.debug("   ❌ Connexion impossible: %s", server_name)
        except Exception as e:
            logger.debug("   ❌ %s: %s", server_name, e)
    raise RuntimeError("Tous les serveurs Overpass ont échoué")


def _wfs_request(
    url: str,
    params: dict[str, Any],
    timeout: int = 60,
    retries: int = 2,
) -> dict[str, Any]:
    """Requête WFS/WCS robuste avec retries."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                if attempt < retries:
                    time.sleep(2)
                    continue
                raise ValueError(f"HTTP {resp.status_code}")
            try:
                data: dict[str, Any] = resp.json()
            except json.JSONDecodeError:
                raise ValueError(
                    f"Réponse non-JSON ({len(resp.content)} octets)"
                )
            if "features" not in data:
                raise ValueError(
                    f"Pas de 'features'. Clés: {list(data.keys())}"
                )
            return data
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(3)
                continue
            raise
        except requests.exceptions.ConnectionError:
            if attempt < retries:
                time.sleep(3)
                continue
            raise
    raise RuntimeError(f"WFS échoué après {retries + 1} tentatives: {url}")


def _osm_element_to_polygons(element: dict[str, Any]) -> list[Polygon]:
    """
    Convertit un élément OSM (way ou relation) en liste de Polygons.
    Gère les multipolygones et les inner rings.
    """
    polygons: list[Polygon] = []

    if element["type"] == "way" and "geometry" in element:
        ring = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
        if len(ring) >= 4:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            try:
                poly = Polygon(ring)
                poly = _try_make_valid(poly)
                if poly is not None and not poly.is_empty:
                    if poly.geom_type == "MultiPolygon":
                        polygons.extend(poly.geoms)
                    else:
                        polygons.append(poly)
            except Exception:
                pass

    elif element["type"] == "relation" and "members" in element:
        outers: list[list[tuple[Any, Any]]] = []
        inners: list[list[tuple[Any, Any]]] = []
        for member in element["members"]:
            if "geometry" not in member:
                continue
            ring = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
            if len(ring) < 4:
                continue
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            role = member.get("role", "outer")
            if role == "outer":
                outers.append(ring)
            elif role == "inner":
                inners.append(ring)

        for outer_ring in outers:
            try:
                poly = Polygon(outer_ring, inners)
                poly = _try_make_valid(poly)
                if poly is not None and not poly.is_empty:
                    if poly.geom_type == "MultiPolygon":
                        polygons.extend(poly.geoms)
                    else:
                        polygons.append(poly)
            except Exception:
                # Réessayer sans inners en cas d'erreur
                try:
                    poly = Polygon(outer_ring)
                    poly = _try_make_valid(poly)
                    if poly is not None and not poly.is_empty:
                        polygons.append(poly)
                except Exception:
                    pass

    return polygons


def _try_make_valid(geom: Any) -> Any | None:
    """Tente de réparer une géométrie invalide."""
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    try:
        fixed = make_valid(geom)
        if fixed is not None and not fixed.is_empty:
            return fixed
    except Exception:
        pass
    return None


def _osm_tags_to_essence(tags: dict[str, str]) -> tuple[str, str]:
    """
    Extrait l'essence d'arbre depuis les tags OSM.
    Utilise config.resolve_tree_name() pour la résolution canonique.

    Returns:
        (display_name, canonical_key)
    """
    # Candidats à tester, par ordre de spécificité
    candidates = [
        tags.get("species", ""),
        tags.get("genus", ""),
        tags.get("taxon", ""),
        tags.get("trees", ""),
        tags.get("produce", ""),
        tags.get("wood", ""),
        tags.get("name", ""),
    ]

    for raw in candidates:
        if not raw:
            continue
        canonical = resolve_tree_name(raw)
        if canonical != "unknown":
            return raw.title(), canonical

    # Fallback par type de feuille
    leaf_type = tags.get("leaf_type", "")
    leaf_cycle = tags.get("leaf_cycle", "")

    if "broadleaved" in leaf_type or "deciduous" in leaf_cycle:
        return "Feuillu indéterminé", "unknown"
    if "needleleaved" in leaf_type or "evergreen" in leaf_cycle:
        return "Conifère indéterminé", "unknown"

    return "Essence inconnue", "unknown"


# ═══════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ═══════════════════════════════════════════════════════════════

class DataLoader:
    """
    Chargeur de données géospatiales pour Cartomorilles.
    Gère le chargement depuis fichier local, cache, WFS, OSM et synthétique.
    """

    def __init__(
        self,
        bbox: Mapping[str, float] = BBOX,
        bbox_wgs84: Mapping[str, float] = BBOX_WGS84,
    ) -> None:
        self.bbox = bbox
        self.bbox_wgs84 = bbox_wgs84
        self._cache_id = _bbox_hash(bbox_wgs84)

        # BBOX avec buffer pour charger les données au-delà de la zone
        # (évite les artefacts de bord sur les critères à fenêtre spatiale)
        self.bbox_buffered: dict[str, float] = {
            "xmin": bbox["xmin"] - DATA_BUFFER,
            "ymin": bbox["ymin"] - DATA_BUFFER,
            "xmax": bbox["xmax"] + DATA_BUFFER,
            "ymax": bbox["ymax"] + DATA_BUFFER,
        }
        buf_deg = DATA_BUFFER / 111000  # approximation degrés
        self.bbox_wgs84_buffered: dict[str, float] = {
            "west": bbox_wgs84["west"] - buf_deg,
            "south": bbox_wgs84["south"] - buf_deg,
            "east": bbox_wgs84["east"] + buf_deg,
            "north": bbox_wgs84["north"] + buf_deg,
        }

        logger.debug(
            "DataLoader initialisé — BBOX hash=%s, buffer=%dm",
            self._cache_id,
            DATA_BUFFER,
        )

    def __repr__(self) -> str:
        return (
            f"DataLoader(bbox_hash={self._cache_id}, "
            f"buffer={DATA_BUFFER}m)"
        )

    # ─── Cache management ─────────────────────────────────

    def _cache_path(self, name: str, ext: str = "gpkg") -> Path:
        """Chemin de cache avec hash BBOX intégré."""
        return DATA_DIR / f"{name}_{self._cache_id}.{ext}"

    def _safe_read_cache(self, path: Path) -> gpd.GeoDataFrame | None:
        """Lecture de cache avec gestion d'erreur."""
        if not path.exists():
            return None
        try:
            gdf = gpd.read_file(path)
            if gdf is not None and len(gdf) > 0:
                logger.info("✅ Cache: %s (%d entités)", path.name, len(gdf))
                return self._ensure_l93(gdf)
        except Exception as e:
            logger.warning("⚠️ Cache corrompu %s: %s — suppression", path.name, e)
            try:
                path.unlink()
            except OSError:
                pass
        return None

    def _save_cache(self, gdf: gpd.GeoDataFrame, path: Path) -> None:
        """Sauvegarde en cache avec métadonnées."""
        try:
            gdf.to_file(path, driver="GPKG")
            logger.debug("💾 Cache sauvé: %s", path.name)
        except Exception as e:
            logger.warning("⚠️ Cache non sauvé: %s", e)

    def clear_cache(self) -> int:
        """Supprime tous les fichiers de cache. Retourne le nombre supprimé."""
        count = 0
        for f in DATA_DIR.glob(f"*_{self._cache_id}.*"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        for f in DATA_DIR.glob("dem_*_cache.tif"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        logger.info("🗑️ %d fichiers de cache supprimés", count)
        return count

    # ─── CRS helpers ──────────────────────────────────────

    def _ensure_l93(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Vérifie et reprojette en Lambert 93 si nécessaire.
        Répare les géométries invalides.
        """
        if len(gdf) == 0:
            return gdf

        # Reprojection CRS
        if gdf.crs is None:
            logger.warning("⚠️ GeoDataFrame sans CRS → assumé EPSG:2154")
            gdf = gdf.set_crs(TARGET_CRS)
        elif str(gdf.crs).upper() != TARGET_CRS:
            src_crs = str(gdf.crs)
            gdf = gdf.to_crs(TARGET_CRS)
            logger.debug("   Reprojeté %s → %s", src_crs, TARGET_CRS)

        # Réparation des géométries invalides
        invalid_count = int((~gdf.geometry.is_valid).sum())
        if invalid_count > 0:
            logger.debug(
                "   Réparation de %d géométries invalides", invalid_count
            )
            gdf.geometry = gdf.geometry.apply(
                lambda g: make_valid(g) if g is not None and not g.is_valid else g  # type: ignore[arg-type]
            )

        return gdf

    def _wgs84_bbox_str(self, buffered: bool = True) -> str:
        """Retourne le BBOX au format Overpass (S,W,N,E)."""
        b = self.bbox_wgs84_buffered if buffered else self.bbox_wgs84
        return f"{b['south']},{b['west']},{b['north']},{b['east']}"

    # ═══════════════════════════════════════════════════════
    # MNT (Modèle Numérique de Terrain)
    # ═══════════════════════════════════════════════════════

    def load_dem(self, filepath: str | None = None) -> dict[str, Any]:
        """
        Charge le MNT. Priorité: fichier local → cache → Copernicus → synthétique.

        Returns:
            {"data": np.ndarray (float32, NaN pour nodata),
             "transform": rasterio.Affine,
             "crs": str}
        """
        # 1. Fichier local
        if filepath and Path(filepath).exists():
            result = self._read_dem_file(filepath)
            if result is not None:
                return result

        # 2. Cache
        cache_path = self._cache_path("dem", "tif")
        result = self._read_dem_file(str(cache_path), label="cache")
        if result is not None:
            return result

        # 3. Copernicus DEM
        try:
            result = self._download_copernicus_dem()
            self._save_dem_cache(result, cache_path)
            return result
        except Exception as e:
            logger.warning("⚠️ Copernicus DEM échoué: %s", e)

        # 4. Synthétique
        logger.warning("⚠️ MNT → génération synthétique")
        return self._generate_synthetic_dem()

    def _read_dem_file(
        self, filepath: str, label: str = "fichier"
    ) -> dict[str, Any] | None:
        """Lit un fichier DEM, crop à l'emprise projet, gère NoData → NaN."""
        path = Path(filepath)
        if not path.exists():
            return None
        try:
            with rasterio.open(path) as src:
                is_l93 = src.crs is not None and "2154" in str(src.crs)

                if is_l93:
                    # ── Fenêtrage à l'emprise projet via index pixel ──
                    # src.index(x, y) → (row, col) — y-axis inversé
                    r_top, c_left = src.index(
                        self.bbox["xmin"], self.bbox["ymax"],
                    )
                    r_bot, c_right = src.index(
                        self.bbox["xmax"], self.bbox["ymin"],
                    )

                    # Clamp aux limites du raster
                    r0 = max(0, min(r_top, r_bot))
                    r1 = min(src.height, max(r_top, r_bot))
                    c0 = max(0, min(c_left, c_right))
                    c1 = min(src.width, max(c_left, c_right))

                    window = Window(c0, r0, c1 - c0, r1 - r0)  # type: ignore

                    dem = src.read(1, window=window).astype(np.float32)
                    transform = src.window_transform(window)

                    if dem.shape != (src.height, src.width):
                        logger.info(
                            "   Fenêtrage DEM : %d×%d → %d×%d"
                            " (crop emprise projet)",
                            src.height,
                            src.width,
                            dem.shape[0],
                            dem.shape[1],
                        )
                else:
                    dem = src.read(1).astype(np.float32)
                    transform = src.transform

                # NoData → NaN
                nodata = src.nodata
                if nodata is not None:
                    dem[dem == nodata] = np.nan
                dem[(dem < -100) | (dem > 9000)] = np.nan

                n_nan = int(np.isnan(dem).sum())
                pct_nan = n_nan / dem.size * 100

                logger.info(
                    "✅ MNT %s: %s — %d×%dpx, alt %.0f–%.0fm%s",
                    label,
                    path.name,
                    dem.shape[1],
                    dem.shape[0],
                    float(np.nanmin(dem)),
                    float(np.nanmax(dem)),
                    f" ({pct_nan:.1f}% NaN)" if pct_nan > 0 else "",
                )

                result: dict[str, Any] = {
                    "data": dem,
                    "transform": transform,
                    "crs": str(src.crs),
                }

                # Reprojection si pas en L93
                if not is_l93 and src.crs:
                    logger.info(
                        "   Reprojection %s → EPSG:2154...", src.crs
                    )
                    result = self._reproject_dem(result)

                return result
        except RasterioIOError as e:
            logger.warning("⚠️ MNT illisible %s: %s", path.name, e)
            if label == "cache":
                try:
                    path.unlink()
                except OSError:
                    pass
            return None

    def _reproject_dem(self, dem_data: dict[str, Any]) -> dict[str, Any]:
        """Reprojette un DEM en Lambert 93 sur la grille cible."""
        nx = int((self.bbox["xmax"] - self.bbox["xmin"]) / CELL_SIZE)
        ny = int((self.bbox["ymax"] - self.bbox["ymin"]) / CELL_SIZE)
        dst_transform = from_bounds(
            self.bbox["xmin"],
            self.bbox["ymin"],
            self.bbox["xmax"],
            self.bbox["ymax"],
            nx,
            ny,
        )
        dem_l93 = np.full((ny, nx), np.nan, dtype=np.float32)

        reproject(
            source=dem_data["data"],
            destination=dem_l93,
            src_transform=dem_data["transform"],
            src_crs=dem_data["crs"],
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return {"data": dem_l93, "transform": dst_transform, "crs": TARGET_CRS}

    def _save_dem_cache(self, dem_data: dict[str, Any], cache_path: Path) -> None:
        """Sauvegarde le DEM en cache GeoTIFF."""
        try:
            dem = dem_data["data"]
            with rasterio.open(
                cache_path,
                "w",
                driver="GTiff",
                height=dem.shape[0],
                width=dem.shape[1],
                count=1,
                dtype=np.float32,
                crs=TARGET_CRS,
                transform=dem_data["transform"],
                compress="lzw",
                nodata=np.nan,
            ) as dst:
                dst.write(dem, 1)
            logger.debug("💾 Cache DEM: %s", cache_path.name)
        except Exception as e:
            logger.warning("⚠️ Cache DEM non sauvé: %s", e)

    def _download_copernicus_dem(self) -> dict[str, Any]:
        """
        Télécharge le Copernicus GLO-30 DSM.

        NOTE: C'est un DSM (surface), pas un DTM (sol).
        En forêt, l'altitude est surestimée de 10-30m.
        Préférer le RGE ALTI 5m IGN si disponible.
        """
        logger.info("🌍 Téléchargement Copernicus DEM GLO-30...")

        b = self.bbox_wgs84_buffered

        # Identifier les tuiles nécessaires
        lat_min = int(np.floor(b["south"]))
        lat_max = int(np.floor(b["north"]))
        lon_min = int(np.floor(b["west"]))
        lon_max = int(np.floor(b["east"]))

        tile_paths: list[Path] = []
        for lat in range(lat_min, lat_max + 1):
            for lon in range(lon_min, lon_max + 1):
                lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
                tile_name = (
                    f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
                )
                cache_tile = DATA_DIR / f"{tile_name}.tif"

                if not cache_tile.exists():
                    url = (
                        f"https://copernicus-dem-30m.s3.amazonaws.com/"
                        f"{tile_name}/{tile_name}.tif"
                    )
                    logger.info("   📥 %s...", tile_name)
                    self._download_file(url, cache_tile)
                else:
                    logger.debug("   ✅ Cache: %s", cache_tile.name)

                tile_paths.append(cache_tile)

        # Vérifier l'intégrité des tuiles
        valid_paths: list[Path] = []
        for tp in tile_paths:
            try:
                with rasterio.open(tp) as src:
                    _ = src.read(1, window=Window(col_off=0, row_off=0, width=1, height=1))  # type: ignore[call-arg]
                valid_paths.append(tp)
            except Exception as e:
                logger.warning("⚠️ Tuile corrompue %s: %s", tp.name, e)
                try:
                    tp.unlink()
                except OSError:
                    pass

        if not valid_paths:
            raise RuntimeError("Aucune tuile Copernicus valide")

        # Merger si plusieurs tuiles
        src_crs: Any
        src_transform: Any
        if len(valid_paths) > 1:
            from rasterio.merge import merge

            datasets = [rasterio.open(p) for p in valid_paths]
            try:
                dem_merged, merged_transform = merge(datasets)
                dem_wgs84 = dem_merged[0].astype(np.float32)
                src_crs = datasets[0].crs
                src_transform = merged_transform
            finally:
                for ds in datasets:
                    ds.close()
        else:
            with rasterio.open(valid_paths[0]) as src:
                from rasterio.windows import from_bounds as window_from_bounds

                buf = 0.005
                window = window_from_bounds(
                    b["west"] - buf,
                    b["south"] - buf,
                    b["east"] + buf,
                    b["north"] + buf,
                    src.transform,
                ).round_offsets().round_lengths()
                dem_wgs84 = src.read(1, window=window).astype(np.float32)
                src_transform = src.window_transform(window)
                src_crs = src.crs

        # Reprojection en L93 (bilinear = pas d'artefact sur falaises)
        nx = int((self.bbox["xmax"] - self.bbox["xmin"]) / CELL_SIZE)
        ny = int((self.bbox["ymax"] - self.bbox["ymin"]) / CELL_SIZE)
        dst_transform = from_bounds(
            self.bbox["xmin"],
            self.bbox["ymin"],
            self.bbox["xmax"],
            self.bbox["ymax"],
            nx,
            ny,
        )
        dem_l93 = np.full((ny, nx), np.nan, dtype=np.float32)

        reproject(
            source=dem_wgs84,
            destination=dem_l93,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        # NoData → NaN (au lieu de remplacer par la moyenne !)
        dem_l93[(dem_l93 <= 0) | (dem_l93 > 9000)] = np.nan

        logger.info(
            "✅ Copernicus DEM: %d×%d, alt %.0f–%.0fm "
            "(⚠️ DSM — surestimé en forêt)",
            nx,
            ny,
            float(np.nanmin(dem_l93)),
            float(np.nanmax(dem_l93)),
        )
        return {"data": dem_l93, "transform": dst_transform, "crs": TARGET_CRS}

    def _download_file(self, url: str, dest: Path, timeout: int = 120) -> None:
        """Télécharge un fichier avec limite de taille et vérification."""
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(65536):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    dest.unlink(missing_ok=True)
                    raise ValueError(
                        f"Fichier trop gros (>{MAX_DOWNLOAD_BYTES // 1_000_000} Mo)"
                    )
                f.write(chunk)
        logger.debug("   ✅ Téléchargé: %s (%.1f Mo)", dest.name, total / 1e6)

    def _generate_synthetic_dem(self) -> dict[str, Any]:
        """Génère un MNT synthétique réaliste par interpolation RBF."""
        from scipy.interpolate import RBFInterpolator
        from scipy.ndimage import gaussian_filter

        nx = int((self.bbox["xmax"] - self.bbox["xmin"]) / CELL_SIZE)
        ny = int((self.bbox["ymax"] - self.bbox["ymin"]) / CELL_SIZE)

        # Points de contrôle calibrés sur la topographie réelle
        ctrl = [
            (5.670, 45.215, 210),
            (5.690, 45.215, 211),
            (5.720, 45.215, 213),
            (5.683, 45.230, 215),
            (5.693, 45.239, 330),
            (5.688, 45.238, 251),
            (5.700, 45.240, 420),
            (5.708, 45.245, 700),
            (5.712, 45.250, 1050),
            (5.715, 45.255, 1280),
            (5.675, 45.230, 220),
            (5.680, 45.240, 280),
            (5.670, 45.210, 208),
            (5.720, 45.210, 213),
            (5.670, 45.260, 500),
            (5.720, 45.260, 1100),
        ]
        pts = np.array([_TO_L93.transform(lon, lat) for lon, lat, _ in ctrl])
        vals = np.array([a for *_, a in ctrl])

        x_g = np.linspace(
            self.bbox["xmin"] + CELL_SIZE / 2,
            self.bbox["xmax"] - CELL_SIZE / 2,
            nx,
        )
        y_g = np.linspace(
            self.bbox["ymin"] + CELL_SIZE / 2,
            self.bbox["ymax"] - CELL_SIZE / 2,
            ny,
        )
        xx, yy = np.meshgrid(x_g, y_g)

        rbf = RBFInterpolator(
            pts, vals, kernel="thin_plate_spline", smoothing=5
        )
        dem = (
            rbf(np.column_stack([xx.ravel(), yy.ravel()]))
            .reshape(ny, nx)
            .astype(np.float32)
        )

        # Bruit topographique proportionnel à l'altitude
        np.random.seed(42)
        af = np.clip((dem - 200) / 800, 0, 1)
        noise_large: np.ndarray = np.asarray(
            gaussian_filter(np.random.randn(ny, nx), 30)
        )
        noise_small: np.ndarray = np.asarray(
            gaussian_filter(np.random.randn(ny, nx), 8)
        )
        dem += noise_large * 15 * af
        dem += noise_small * 6 * af
        dem = np.asarray(gaussian_filter(dem, 1.0)).astype(np.float32)
        dem = np.clip(dem, 195, 1350).astype(np.float32)

        transform = from_bounds(
            self.bbox["xmin"],
            self.bbox["ymin"],
            self.bbox["xmax"],
            self.bbox["ymax"],
            nx,
            ny,
        )
        logger.info(
            "✅ MNT synthétique: %d×%d, alt %.0f–%.0fm",
            nx,
            ny,
            float(dem.min()),
            float(dem.max()),
        )
        return {"data": dem, "transform": transform, "crs": TARGET_CRS}

    # ═══════════════════════════════════════════════════════
    # FORÊT
    # Colonnes normalisées:
    #   ESSENCE (str), essence_canonical (str),
    #   TFV (str), source (str)
    # ═══════════════════════════════════════════════════════

    def load_forest(
        self, filepath: str | None = None
    ) -> gpd.GeoDataFrame | None:
        """
        Charge la couche forêt.
        Priorité: fichier → cache → WFS IGN → OSM → synthétique.
        """
        # 1. Fichier local
        if filepath and Path(filepath).exists():
            gdf = self._read_vector_file(filepath, "Forêt")
            return (
                self._normalize_forest(gdf, source="file")
                if gdf is not None
                else None
            )

        # 2. Cache
        cache = self._cache_path("foret")
        gdf = self._safe_read_cache(cache)
        if gdf is not None:
            return gdf

        # 3. WFS IGN BD Forêt V2
        try:
            gdf = self._download_forest_wfs()
            if gdf is not None and len(gdf) > 0:
                gdf = self._normalize_forest(gdf, source="wfs_ign")
                self._save_cache(gdf, cache)
                return gdf
        except Exception as e:
            logger.debug("WFS BD Forêt: %s", e)

        # 4. OSM Overpass
        try:
            gdf = self._download_forest_osm()
            if gdf is not None and len(gdf) > 0:
                gdf = self._normalize_forest(gdf, source="osm")
                self._save_cache(gdf, cache)
                return gdf
        except Exception as e:
            logger.warning("⚠️ OSM forêt: %s", e)

        # 5. Synthétique
        logger.warning("⚠️ Forêt → synthétique")
        gdf = self._generate_synthetic_forest()
        return self._normalize_forest(gdf, source="synthetic")

    def _download_forest_wfs(self) -> gpd.GeoDataFrame | None:
        """Tente de charger la BD Forêt V2 via WFS Géoplateforme."""
        bbox_str = (
            f"{self.bbox_buffered['xmin']},{self.bbox_buffered['ymin']},"
            f"{self.bbox_buffered['xmax']},{self.bbox_buffered['ymax']}"
        )
        url = "https://data.geopf.fr/wfs/ows"

        for layer in IGN_FORET_LAYERS:
            try:
                logger.info("🌲 WFS IGN BD Forêt: %s...", layer)
                params: dict[str, Any] = {
                    "service": "WFS",
                    "version": "2.0.0",
                    "request": "GetFeature",
                    "typeName": layer,
                    "outputFormat": "application/json",
                    "srsName": TARGET_CRS,
                    "bbox": bbox_str + f",{TARGET_CRS}",
                    "count": "10000",
                }
                data = _wfs_request(url, params, timeout=60)
                gdf = gpd.GeoDataFrame.from_features(
                    data["features"], crs=TARGET_CRS
                )
                logger.info("✅ BD Forêt WFS: %d polygones", len(gdf))
                return gdf
            except Exception as e:
                logger.debug("   Layer %s: %s", layer, e)
                continue

        raise RuntimeError("Aucune couche BD Forêt disponible via WFS")

    def _download_forest_osm(self) -> gpd.GeoDataFrame | None:
        """Télécharge les données forestières depuis OSM Overpass."""
        bbox_str = self._wgs84_bbox_str(buffered=True)
        logger.info("🌲 Forêt OSM...")

        query = f"""
        [out:json][timeout:60];
        (
          way["natural"="wood"]({bbox_str});
          way["landuse"="forest"]({bbox_str});
          way["landuse"="orchard"]({bbox_str});
          way["natural"="tree_row"]({bbox_str});
          relation["natural"="wood"]({bbox_str});
          relation["landuse"="forest"]({bbox_str});
        );
        out geom;
        """
        data = _overpass_query(query, timeout=60)

        geometries: list[Any] = []
        essences: list[str] = []
        canonicals: list[str] = []
        types: list[str] = []

        for el in data.get("elements", []):
            tags: dict[str, Any] = el.get("tags", {})

            # Alignements d'arbres → buffer en LineString
            if tags.get("natural") == "tree_row" and el["type"] == "way":
                if "geometry" in el:
                    coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
                    if len(coords) >= 2:
                        try:
                            line = LineString(coords)
                            if line.is_valid and line.length > 0:
                                display, canonical = _osm_tags_to_essence(tags)
                                geometries.append(line)
                                essences.append(display)
                                canonicals.append(canonical)
                                types.append("tree_row")
                        except Exception:
                            pass
                continue

            # Polygones forêt / verger
            polys = _osm_element_to_polygons(el)
            if not polys:
                continue

            display, canonical = _osm_tags_to_essence(tags)

            # Tags spécifiques vergers
            if tags.get("landuse") == "orchard" and canonical == "unknown":
                produce = str(tags.get("produce", "")).lower()
                trees = str(tags.get("trees", "")).lower()
                for kw, name in [
                    ("apple", "pommier"),
                    ("pear", "poirier"),
                    ("cherry", "unknown"),
                    ("plum", "unknown"),
                    ("walnut", "noisetier"),
                ]:
                    if kw in produce or kw in trees:
                        canonical = name
                        display = kw.title()
                        break
                if canonical == "unknown":
                    canonical = "pommier"  # verger générique → pommier
                    display = "Verger (probablement pommier)"

            tfv = str(tags.get("landuse", tags.get("natural", "forest")))

            for poly in polys:
                geometries.append(poly)
                essences.append(display)
                canonicals.append(canonical)
                types.append(tfv)

        if not geometries:
            raise ValueError("Aucune entité forestière OSM")

        gdf = gpd.GeoDataFrame(
            {
                "ESSENCE": essences,
                "essence_canonical": canonicals,
                "TFV": types,
            },
            geometry=geometries,
            crs="EPSG:4326",
        )
        gdf = self._ensure_l93(gdf)

        # Les tree_row linéaires → buffer 5m en L93 pour devenir des polygones
        mask_line = gdf.geometry.geom_type.isin(
            ["LineString", "MultiLineString"]
        )
        if mask_line.any():
            gdf.loc[mask_line, "geometry"] = gdf.loc[
                mask_line, "geometry"
            ].buffer(5)
            logger.debug(
                "   %d alignements d'arbres bufferisés (5m)", mask_line.sum()
            )

        logger.info("✅ Forêt OSM: %d entités", len(gdf))

        # Résumé essences
        for ess, cnt in gdf["ESSENCE"].value_counts().head(8).items():
            logger.debug("   • %s: %d", ess, cnt)

        # Avertissement si trop d'inconnus
        n_unknown = int((gdf["essence_canonical"] == "unknown").sum())
        pct_unknown = n_unknown / len(gdf) * 100
        if pct_unknown > 50:
            logger.warning(
                "⚠️ %.0f%% des polygones forestiers OSM sans essence connue "
                "— considérer la BD Forêt IGN",
                pct_unknown,
            )

        return gdf

    def _normalize_forest(
        self, gdf: gpd.GeoDataFrame, source: str
    ) -> gpd.GeoDataFrame:
        """
        Normalise les colonnes de la couche forêt pour grid_builder.py.
        Colonnes garanties: ESSENCE, essence_canonical, TFV, source
        """
        if len(gdf) == 0:
            return gdf

        gdf = self._ensure_l93(gdf)

        # Ajouter source
        gdf["source"] = source

        # Normaliser la colonne ESSENCE
        if "ESSENCE" not in gdf.columns:
            # BD Forêt V2 IGN : colonne 'tfv' ou 'libelle'
            for col in ["tfv", "libelle", "TFV", "LIBELLE", "lib", "essence"]:
                if col in gdf.columns:
                    gdf["ESSENCE"] = gdf[col].astype(str)
                    break
            else:
                gdf["ESSENCE"] = "Essence inconnue"

        # Résoudre les noms canoniques via config.resolve_tree_name()
        if "essence_canonical" not in gdf.columns:
            gdf["essence_canonical"] = gdf["ESSENCE"].apply(resolve_tree_name)

        # Colonne TFV (type de formation végétale)
        if "TFV" not in gdf.columns:
            gdf["TFV"] = "forest"

        # Score pré-calculé (pour diagnostics)
        gdf["tree_score"] = gdf["essence_canonical"].apply(
            lambda c: config.TREE_SCORES.get(c, config.TREE_SCORES["unknown"])
        )

        logger.debug(
            "   Forêt normalisée (%s): %d entités, %d essences distinctes",
            source,
            len(gdf),
            gdf["essence_canonical"].nunique(),
        )

        return gdf

    def _generate_synthetic_forest(self) -> gpd.GeoDataFrame:
        """
        Génère des zones forestières synthétiques cohérentes
        avec la topographie locale.
        """
        xmin, ymin = float(self.bbox["xmin"]), float(self.bbox["ymin"])
        xmax, ymax = float(self.bbox["xmax"]), float(self.bbox["ymax"])
        dx, dy = xmax - xmin, ymax - ymin

        zones: list[tuple[Polygon, str, str, str]] = [
            # Plaine alluviale Isère — zone cible morilles
            (
                box(xmin, ymin, xmin + dx * 0.40, ymin + dy * 0.12),
                "Peuplier / Frêne",
                "peuplier",
                "Ripisylve Isère",
            ),
            (
                box(xmin + dx * 0.40, ymin, xmin + dx * 0.70, ymin + dy * 0.10),
                "Frêne commun",
                "frene",
                "Frênaie alluviale",
            ),
            (
                box(xmin + dx * 0.70, ymin, xmax, ymin + dy * 0.10),
                "Orme / Frêne",
                "orme",
                "Boisement alluvial est",
            ),
            # Ripisylve Vence
            (
                box(
                    xmin + dx * 0.28,
                    ymin + dy * 0.20,
                    xmin + dx * 0.38,
                    ymin + dy * 0.38,
                ),
                "Frêne / Aulne",
                "aulne",
                "Ripisylve Vence",
            ),
            # Transition pied de pente
            (
                box(
                    xmin + dx * 0.20,
                    ymin + dy * 0.15,
                    xmin + dx * 0.40,
                    ymin + dy * 0.30,
                ),
                "Frêne / Noisetier",
                "noisetier",
                "Pied Néron",
            ),
            (
                box(
                    xmin + dx * 0.10,
                    ymin + dy * 0.20,
                    xmin + dx * 0.25,
                    ymin + dy * 0.35,
                ),
                "Pommier / Frêne",
                "pommier",
                "Vergers abandonnés",
            ),
            # Pentes du Néron — défavorables
            (
                box(
                    xmin + dx * 0.45,
                    ymin + dy * 0.30,
                    xmin + dx * 0.80,
                    ymin + dy * 0.55,
                ),
                "Chêne pubescent",
                "chene_pubescent",
                "Chênaie + buis",
            ),
            (
                box(
                    xmin + dx * 0.15,
                    ymin + dy * 0.30,
                    xmin + dx * 0.30,
                    ymin + dy * 0.45,
                ),
                "Châtaignier",
                "chataignier",
                "Châtaigneraie Champy",
            ),
            # Altitude — défavorable à M. esculenta
            (
                box(
                    xmin + dx * 0.50,
                    ymin + dy * 0.60,
                    xmin + dx * 0.90,
                    ymin + dy * 0.80,
                ),
                "Hêtre",
                "hetre",
                "Hêtraie montagnarde",
            ),
            (
                box(
                    xmin + dx * 0.35,
                    ymin + dy * 0.45,
                    xmin + dx * 0.42,
                    ymin + dy * 0.55,
                ),
                "Épicéa commun",
                "epicea",
                "Plantation épicéa",
            ),
            # Perturbé
            (
                box(
                    xmin + dx * 0.05,
                    ymin + dy * 0.12,
                    xmin + dx * 0.18,
                    ymin + dy * 0.22,
                ),
                "Robinier",
                "robinier",
                "Robinier perturbé",
            ),
        ]

        records: list[dict[str, Any]] = []
        for geom, display, canonical, description in zones:
            records.append(
                {
                    "ESSENCE": display,
                    "essence_canonical": canonical,
                    "TFV": description,
                    "tree_score": config.TREE_SCORES.get(
                        canonical, config.TREE_SCORES["unknown"]
                    ),
                    "geometry": geom,
                }
            )

        gdf = gpd.GeoDataFrame(records, crs=TARGET_CRS)
        logger.info("✅ Forêt synthétique: %d zones", len(gdf))
        return gdf

    # ═══════════════════════════════════════════════════════
    # GÉOLOGIE
    # Colonnes normalisées:
    #   LITHO (str), geology_canonical (str),
    #   DESCR (str), source (str)
    # ═══════════════════════════════════════════════════════

    def load_geology(
        self, filepath: str | None = None
    ) -> gpd.GeoDataFrame | None:
        """
        Charge la couche géologie.
        Priorité: fichier → cache → WFS BRGM → synthétique.
        """
        # 1. Fichier local
        if filepath and Path(filepath).exists():
            gdf = self._read_vector_file(filepath, "Géologie")
            return (
                self._normalize_geology(gdf, source="file")
                if gdf is not None
                else None
            )

        # 2. Cache
        cache = self._cache_path("geologie")
        gdf = self._safe_read_cache(cache)
        if gdf is not None:
            return gdf

        # 3. WFS BRGM
        try:
            gdf = self._download_geology_brgm()
            if gdf is not None and len(gdf) > 0:
                gdf = self._normalize_geology(gdf, source="wfs_brgm")
                self._save_cache(gdf, cache)
                return gdf
        except Exception as e:
            logger.debug("WFS BRGM: %s", e)

        # 4. Synthétique
        logger.warning("⚠️ Géologie → synthétique")
        gdf = self._generate_synthetic_geology()
        return self._normalize_geology(gdf, source="synthetic")

    def _download_geology_brgm(self) -> gpd.GeoDataFrame | None:
        """Tente de charger la carte géologique harmonisée BRGM."""
        bbox_str = (
            f"{self.bbox_buffered['xmin']},{self.bbox_buffered['ymin']},"
            f"{self.bbox_buffered['xmax']},{self.bbox_buffered['ymax']}"
        )

        # Essayer plusieurs URLs et layers
        urls_layers: list[tuple[str, list[str]]] = [
            ("https://geoservices.brgm.fr/geologie", list(BRGM_GEOLOGY_LAYERS)),
            (
                "https://data.geopf.fr/wfs/ows",
                ["GEOLOGIE.CARTE_GEOL_HARM_50.V1:polygones"],
            ),
        ]

        for base_url, layers in urls_layers:
            for layer in layers:
                try:
                    logger.info(
                        "🪨 WFS BRGM: %s @ %s...",
                        layer,
                        base_url.split("//")[1][:30],
                    )
                    params: dict[str, Any] = {
                        "service": "WFS",
                        "version": "2.0.0",
                        "request": "GetFeature",
                        "typeName": layer,
                        "outputFormat": "application/json",
                        "srsName": TARGET_CRS,
                        "bbox": bbox_str + f",{TARGET_CRS}",
                        "count": "5000",
                    }
                    data = _wfs_request(base_url, params, timeout=60)
                    gdf = gpd.GeoDataFrame.from_features(
                        data["features"], crs=TARGET_CRS
                    )
                    if len(gdf) > 0:
                        logger.info(
                            "✅ Géologie BRGM WFS: %d polygones", len(gdf)
                        )
                        return gdf
                except Exception as e:
                    logger.debug("   Layer %s: %s", layer, e)
                    continue

        raise RuntimeError("Aucune couche géologique BRGM disponible via WFS")

    def _normalize_geology(
        self, gdf: gpd.GeoDataFrame, source: str,
    ) -> gpd.GeoDataFrame:
        """Normalise geology_canonical via cascade 3 étapes.

        Fix #35 v2.3.3 — DESCR prioritaire (99.9% vs NOTATION 15.7%).
        Pylance clean : np.asarray (P5), assert (P3), no .values on ndarray.
        """
        if len(gdf) == 0:
            return gdf

        gdf = self._ensure_l93(gdf)  # type: ignore[attr-defined]
        gdf["source"] = source

        # ── Auto-détection colonnes (case-insensitive) ───────────
        descr_col: str | None = None
        for col in ("DESCR", "descr", "DESCRIPTIO", "description",
                     "DESCRIPTION", "LIBELLE", "libelle"):
            if col in gdf.columns:
                descr_col = col
                break

        nota_col: str | None = None
        for col in ("NOTATION", "notation", "NOTA"):
            if col in gdf.columns:
                nota_col = col
                break

        # ── LITHO = meilleure colonne descriptive ────────────────
        if "LITHO" not in gdf.columns:
            if descr_col is not None:
                gdf["LITHO"] = gdf[descr_col].astype(str)
            elif nota_col is not None:
                gdf["LITHO"] = gdf[nota_col].astype(str)
            else:
                for col in ("code", "CODE", "litho", "lithologie",
                            "LITHOLOGIE"):
                    if col in gdf.columns:
                        gdf["LITHO"] = gdf[col].astype(str)
                        break
                else:
                    gdf["LITHO"] = "unknown"

        logger.info("  Colonnes géologie : descr=%s, notation=%s, LITHO←%s",
                     descr_col, nota_col,
                     descr_col or nota_col or "fallback")

        # ── DESCR garantie ───────────────────────────────────────
        if "DESCR" not in gdf.columns:
            if descr_col is not None and descr_col != "DESCR":
                gdf["DESCR"] = gdf[descr_col].astype(str)
            else:
                gdf["DESCR"] = gdf["LITHO"]


        # ── Étape 1 : resolve_geology() sur DESCR (vectorisé) ───
        n = len(gdf)
        resolved = np.zeros(n, dtype=bool)

        raw_descr = gdf["DESCR"].fillna("").astype(str)
        unique_descr = raw_descr.unique()

        descr_map: dict[str, str] = {}
        for v in unique_descr:
            cleaned = str(v).lower().strip()
            if not cleaned:
                continue
            cat = resolve_geology(cleaned)
            if cat != "unknown":
                descr_map[v] = cat

        mask_1: np.ndarray = np.asarray(raw_descr.isin(descr_map))  # P5
        gdf["geology_canonical"] = "unknown"
        if bool(np.any(mask_1)):                                      # P5
            gdf.loc[mask_1, "geology_canonical"] = raw_descr[mask_1].map(
                descr_map,
            )
            resolved |= mask_1

        n1 = int(np.sum(mask_1))                                      # P5
        logger.info("  Étape 1 (DESCR → resolve_geology) : %d/%d (%.1f%%)",
                     n1, n, 100 * n1 / n)

        # ── Étape 2 : _BDCHARM_DESCR_OVERRIDES (substring) ──────
        if not bool(np.all(resolved)):
            descr_lower = raw_descr.str.lower().str.strip()
            n2_before = int(np.sum(resolved))

            for pattern, category in _BDCHARM_DESCR_OVERRIDES:
                if bool(np.all(resolved)):
                    break
                hits_s = descr_lower.str.contains(
                    pattern, na=False, regex=False,
                )
                hits: np.ndarray = np.asarray(hits_s)                 # P5
                combined = (~resolved) & hits
                if bool(np.any(combined)):
                    gdf.loc[combined, "geology_canonical"] = category
                    resolved |= combined

            n2 = int(np.sum(resolved)) - n2_before
            logger.info(
                "  Étape 2 (overrides substring)    : +%d → %d/%d (%.1f%%)",
                n2, int(np.sum(resolved)), n,
                100 * float(np.sum(resolved)) / n,
            )

        # ── Étape 3 : NOTATION fallback ─────────────────────────
        if nota_col is not None and not bool(np.all(resolved)):
            raw_nota = gdf[nota_col].fillna("").astype(str)
            unique_nota = raw_nota[~resolved].unique()

            nota_map: dict[str, str] = {}
            for v in unique_nota:
                vs = str(v).strip()
                if not vs:
                    continue
                cat = (config.GEOLOGY_BRGM_MAP.get(vs)
                       or config.GEOLOGY_BRGM_MAP.get(vs.lower())
                       or resolve_geology(vs))
                if cat is not None and cat != "unknown":
                    nota_map[v] = cat

            mask_3: np.ndarray = (
                (~resolved) & np.asarray(raw_nota.isin(nota_map))     # P5
            )
            if bool(np.any(mask_3)):
                gdf.loc[mask_3, "geology_canonical"] = (
                    raw_nota[mask_3].map(nota_map)
                )
                resolved |= mask_3

            n3 = int(np.sum(mask_3))
            logger.info(
                "  Étape 3 (NOTATION fallback)      : +%d → %d/%d (%.1f%%)",
                n3, int(np.sum(resolved)), n,
                100 * float(np.sum(resolved)) / n,
            )

        # ── Score pré-calculé ────────────────────────────────────
        gdf["geology_score"] = gdf["geology_canonical"].map(
            lambda c: config.GEOLOGY_SCORES.get(
                str(c), config.GEOLOGY_SCORES["unknown"],
            ),
        )

        total = int(np.sum(resolved))
        logger.info("🪨 Géologie normalisée (%s) : %d/%d résolus (%.1f%%)",
                     source, total, n, 100 * total / n)

        if total < n:
            unresolv = gdf.loc[~resolved, "DESCR"].value_counts().head(10)
            logger.warning("  Non résolues (%d) :", n - total)
            for val, cnt in unresolv.items():
                logger.warning("    [%d] %r", cnt, val)

        return gdf

    def _generate_synthetic_geology(self) -> gpd.GeoDataFrame:
        """Génère des zones géologiques synthétiques."""
        xmin, ymin = float(self.bbox["xmin"]), float(self.bbox["ymin"])
        xmax, ymax = float(self.bbox["xmax"]), float(self.bbox["ymax"])
        dx, dy = xmax - xmin, ymax - ymin

        zones: list[tuple[Polygon, str, str, str]] = [
            (
                box(xmin, ymin, xmax, ymin + dy * 0.20),
                "Fz",
                "alluvions_recentes",
                "Alluvions récentes calcaires — plaine Isère",
            ),
            (
                box(xmin, ymin + dy * 0.20, xmin + dx * 0.40, ymin + dy * 0.30),
                "Fy",
                "alluvions",
                "Alluvions anciennes — terrasses würmiennes",
            ),
            (
                box(
                    xmin + dx * 0.25,
                    ymin + dy * 0.20,
                    xmin + dx * 0.38,
                    ymin + dy * 0.50,
                ),
                "Fz",
                "alluvions_recentes",
                "Alluvions de la Vence",
            ),
            (
                box(
                    xmin + dx * 0.30,
                    ymin + dy * 0.30,
                    xmin + dx * 0.70,
                    ymin + dy * 0.55,
                ),
                "j4",
                "calcaire_marneux",
                "Marno-calcaires Jurassique",
            ),
            (
                box(
                    xmin + dx * 0.10,
                    ymin + dy * 0.30,
                    xmin + dx * 0.30,
                    ymin + dy * 0.45,
                ),
                "Gx",
                "moraine",
                "Moraine glaciaire",
            ),
            (
                box(xmin + dx * 0.45, ymin + dy * 0.55, xmax, ymax),
                "j6",
                "calcaire",
                "Calcaire urgonien — falaises Néron",
            ),
        ]

        records: list[dict[str, Any]] = []
        for geom, code, canonical, descr in zones:
            records.append(
                {
                    "LITHO": code,
                    "geology_canonical": canonical,
                    "DESCR": descr,
                    "geology_score": config.GEOLOGY_SCORES.get(
                        canonical, config.GEOLOGY_SCORES["unknown"]
                    ),
                    "geometry": geom,
                }
            )

        gdf = gpd.GeoDataFrame(records, crs=TARGET_CRS)
        logger.info("✅ Géologie synthétique: %d zones", len(gdf))
        return gdf

    # ═══════════════════════════════════════════════════════
    # HYDROGRAPHIE
    # Colonnes normalisées:
    #   NOM (str), water_type (str), water_type_key (str),
    #   water_bonus (float), source (str)
    # ═══════════════════════════════════════════════════════

    def load_hydro(
        self, filepath: str | None = None
    ) -> gpd.GeoDataFrame | None:
        """
        Charge la couche hydrographie.
        Priorité: fichier → cache → WFS IGN → OSM → synthétique.
        """
        # 1. Fichier local
        if filepath and Path(filepath).exists():
            gdf = self._read_vector_file(filepath, "Hydro")
            return (
                self._normalize_hydro(gdf, source="file")
                if gdf is not None
                else None
            )

        # 2. Cache
        cache = self._cache_path("hydro")
        gdf = self._safe_read_cache(cache)
        if gdf is not None:
            return gdf

        # 3. WFS IGN
        try:
            gdf = self._download_hydro_wfs()
            if gdf is not None and len(gdf) > 0:
                gdf = self._normalize_hydro(gdf, source="wfs_ign")
                self._save_cache(gdf, cache)
                return gdf
        except Exception as e:
            logger.debug("WFS hydro: %s", e)

        # 4. OSM Overpass
        try:
            gdf = self._download_hydro_osm()
            if gdf is not None and len(gdf) > 0:
                gdf = self._normalize_hydro(gdf, source="osm")
                self._save_cache(gdf, cache)
                return gdf
        except Exception as e:
            logger.warning("⚠️ OSM hydro: %s", e)

        # 5. Synthétique
        logger.warning("⚠️ Hydro → synthétique")
        gdf = self._generate_synthetic_hydro()
        return self._normalize_hydro(gdf, source="synthetic")

    def _download_hydro_wfs(self) -> gpd.GeoDataFrame | None:
        """Tente de charger l'hydrographie via WFS IGN BD Topo."""
        bbox_str = (
            f"{self.bbox_buffered['xmin']},{self.bbox_buffered['ymin']},"
            f"{self.bbox_buffered['xmax']},{self.bbox_buffered['ymax']}"
        )
        url = "https://data.geopf.fr/wfs/ows"

        for layer in IGN_HYDRO_LAYERS:
            try:
                logger.info("💧 WFS IGN Hydro: %s...", layer)
                params: dict[str, Any] = {
                    "service": "WFS",
                    "version": "2.0.0",
                    "request": "GetFeature",
                    "typeName": layer,
                    "outputFormat": "application/json",
                    "srsName": TARGET_CRS,
                    "bbox": bbox_str + f",{TARGET_CRS}",
                    "count": "5000",
                }
                data = _wfs_request(url, params, timeout=60)
                gdf = gpd.GeoDataFrame.from_features(
                    data["features"], crs=TARGET_CRS
                )
                if len(gdf) > 0:
                    logger.info("✅ Hydro WFS: %d entités", len(gdf))
                    return gdf
            except Exception as e:
                logger.debug("   Layer %s: %s", layer, e)
                continue

        raise RuntimeError("Aucune couche hydro IGN via WFS")

    def _download_hydro_osm(self) -> gpd.GeoDataFrame | None:
        """
        Télécharge hydrographie depuis OSM.
        Inclut: cours d'eau linéaires + plans d'eau surfaciques.
        """
        bbox_str = self._wgs84_bbox_str(buffered=True)
        logger.info("💧 Hydro OSM...")

        query = f"""
        [out:json][timeout:45];
        (
          way["waterway"~"river|stream|canal|drain|ditch"]({bbox_str});
          way["natural"="water"]({bbox_str});
          way["water"~"lake|pond|reservoir|oxbow|basin"]({bbox_str});
          relation["natural"="water"]({bbox_str});
        );
        out geom;
        """
        data = _overpass_query(query, timeout=45)

        geometries: list[Any] = []
        noms: list[str] = []
        water_types: list[str] = []

        for el in data.get("elements", []):
            if "geometry" not in el and "members" not in el:
                continue
            tags: dict[str, Any] = el.get("tags", {})

            # Filtrer les cours d'eau souterrains
            if tags.get("tunnel") == "yes" or str(
                tags.get("layer", "0")
            ).startswith("-"):
                logger.debug(
                    "   Ignoré souterrain: %s", tags.get("name", "?")
                )
                continue

            nom = str(tags.get("name", "Sans nom"))

            # Linéaire (waterway)
            if "waterway" in tags:
                wtype = str(tags["waterway"])
                if el["type"] == "way":
                    coords = [
                        (pt["lon"], pt["lat"]) for pt in el["geometry"]
                    ]
                    if len(coords) >= 2:
                        try:
                            line = LineString(coords)
                            if line.is_valid and line.length > 0:
                                geometries.append(line)
                                noms.append(nom)
                                water_types.append(wtype)
                        except Exception:
                            pass

            # Surfacique (plans d'eau)
            elif tags.get("natural") == "water" or "water" in tags:
                wtype = str(tags.get("water", "lake"))
                polys = _osm_element_to_polygons(el)
                for poly in polys:
                    geometries.append(poly)
                    noms.append(nom)
                    water_types.append(wtype)

        if not geometries:
            raise ValueError("Aucun cours d'eau/plan d'eau OSM")

        gdf = gpd.GeoDataFrame(
            {"NOM": noms, "water_type": water_types},
            geometry=geometries,
            crs="EPSG:4326",
        )
        gdf = self._ensure_l93(gdf)

        logger.info("✅ Hydro OSM: %d entités", len(gdf))
        for nom_val in gdf.loc[gdf["NOM"] != "Sans nom", "NOM"].unique()[:8]:
            logger.debug("   • %s", nom_val)

        return gdf

    def _normalize_hydro(
        self, gdf: gpd.GeoDataFrame, source: str
    ) -> gpd.GeoDataFrame:
        """
        Normalise la couche hydro pour grid_builder.py.
        Colonnes garanties: NOM, water_type, water_type_key, water_bonus, source
        """
        if len(gdf) == 0:
            return gdf

        gdf = self._ensure_l93(gdf)
        gdf["source"] = source

        # NOM
        if "NOM" not in gdf.columns:
            for col in ["nom", "name", "NAME", "toponyme", "TOPONYME"]:
                if col in gdf.columns:
                    gdf["NOM"] = gdf[col].astype(str)
                    break
            else:
                gdf["NOM"] = "Sans nom"

        # Type
        if "water_type" not in gdf.columns:
            for col in ["type", "nature", "NATURE", "regime"]:
                if col in gdf.columns:
                    gdf["water_type"] = gdf[col].astype(str)
                    break
            else:
                gdf["water_type"] = "unknown"

        # Clé pour le bonus config
        _combined_water_map: dict[str, str] = {
            **dict(WATERWAY_TYPE_MAP),
            **dict(WATER_AREA_TYPE_MAP),
        }

        def _resolve_water_type(wt: str) -> str:
            wt_lower = wt.lower().strip()
            if wt_lower in WATERWAY_TYPE_MAP:
                return WATERWAY_TYPE_MAP[wt_lower]
            if wt_lower in WATER_AREA_TYPE_MAP:
                return WATER_AREA_TYPE_MAP[wt_lower]
            # Recherche partielle
            for keyword, key in _combined_water_map.items():
                if keyword in wt_lower:
                    return key
            return "unknown"

        gdf["water_type_key"] = gdf["water_type"].apply(_resolve_water_type)
        gdf["water_bonus"] = gdf["water_type_key"].apply(
            lambda k: WATER_TYPE_BONUS.get(k, WATER_TYPE_BONUS.get("unknown", 0.9))
        )

        logger.debug(
            "   Hydro normalisée (%s): %d entités, types: %s",
            source,
            len(gdf),
            dict(gdf["water_type_key"].value_counts()),
        )

        return gdf

    def _generate_synthetic_hydro(self) -> gpd.GeoDataFrame:
        """Génère un réseau hydrographique synthétique."""
        xmin, ymin = float(self.bbox["xmin"]), float(self.bbox["ymin"])
        xmax, ymax = float(self.bbox["xmax"]), float(self.bbox["ymax"])
        dx, dy = xmax - xmin, ymax - ymin

        records: list[dict[str, Any]] = [
            {
                "NOM": "L'Isère",
                "water_type": "river",
                "geometry": LineString(
                    [
                        (xmin, ymin + dy * 0.06),
                        (xmin + dx * 0.25, ymin + dy * 0.06),
                        (xmin + dx * 0.55, ymin + dy * 0.06),
                        (xmax, ymin + dy * 0.07),
                    ]
                ),
            },
            {
                "NOM": "La Vence",
                "water_type": "stream",
                "geometry": LineString(
                    [
                        (xmin + dx * 0.60, ymin + dy * 0.85),
                        (xmin + dx * 0.48, ymin + dy * 0.65),
                        (xmin + dx * 0.38, ymin + dy * 0.45),
                        (xmin + dx * 0.30, ymin + dy * 0.30),
                        (xmin + dx * 0.22, ymin + dy * 0.15),
                        (xmin + dx * 0.20, ymin + dy * 0.08),
                    ]
                ),
            },
            {
                "NOM": "Affluent Vence",
                "water_type": "stream",
                "geometry": LineString(
                    [
                        (xmin + dx * 0.55, ymin + dy * 0.50),
                        (xmin + dx * 0.43, ymin + dy * 0.50),
                    ]
                ),
            },
            {
                "NOM": "Canal",
                "water_type": "canal",
                "geometry": LineString(
                    [
                        (xmin + dx * 0.10, ymin + dy * 0.12),
                        (xmin + dx * 0.50, ymin + dy * 0.10),
                        (xmin + dx * 0.70, ymin + dy * 0.11),
                    ]
                ),
            },
            {
                "NOM": "Bras mort Isère",
                "water_type": "oxbow",
                "geometry": Polygon(
                    [
                        (xmin + dx * 0.30, ymin + dy * 0.04),
                        (xmin + dx * 0.35, ymin + dy * 0.03),
                        (xmin + dx * 0.38, ymin + dy * 0.04),
                        (xmin + dx * 0.35, ymin + dy * 0.05),
                        (xmin + dx * 0.30, ymin + dy * 0.04),
                    ]
                ),
            },
        ]

        gdf = gpd.GeoDataFrame(records, crs=TARGET_CRS)
        logger.info("✅ Hydro synthétique: %d entités", len(gdf))
        return gdf

    # ═══════════════════════════════════════════════════════
    # ZONES URBAINES
    # Colonnes normalisées:
    #   urban_type (str), name (str), source (str)
    # ═══════════════════════════════════════════════════════

    def load_urban(
        self, filepath: str | None = None
    ) -> gpd.GeoDataFrame | None:
        """
        Charge les zones urbaines.
        Priorité: fichier → cache → OSM → synthétique.
        """
        # 1. Fichier local
        if filepath and Path(filepath).exists():
            gdf = self._read_vector_file(filepath, "Urbain")
            if gdf is not None:
                if "urban_type" not in gdf.columns:
                    gdf["urban_type"] = "unknown"
                gdf["source"] = "file"
                return gdf

        # 2. Cache
        cache = self._cache_path("urban")
        gdf = self._safe_read_cache(cache)
        if gdf is not None:
            return gdf

        # 3. OSM
        try:
            gdf = self._download_urban_osm()
            if gdf is not None and len(gdf) > 0:
                self._save_cache(gdf, cache)
                return gdf
        except Exception as e:
            logger.warning("⚠️ OSM urbain: %s", e)

        # 4. Synthétique
        logger.warning("⚠️ Urbain → synthétique")
        return self._generate_synthetic_urban()

    def _download_urban_osm(self) -> gpd.GeoDataFrame | None:
        """
        Télécharge zones urbaines OSM par lots avec gestion rate-limit.
        Inclut: bâtiments, landuse, routes principales, parkings, etc.
        """
        b = self.bbox_wgs84_buffered
        bbox_str = self._wgs84_bbox_str(buffered=True)

        logger.info("🏘️ Urbain OSM (par lots)...")

        # Découper en quadrants pour les bâtiments (volume élevé)
        mid_lat = (b["south"] + b["north"]) / 2
        mid_lon = (b["west"] + b["east"]) / 2
        quadrants = [
            f"{b['south']},{b['west']},{mid_lat},{mid_lon}",
            f"{b['south']},{mid_lon},{mid_lat},{b['east']}",
            f"{mid_lat},{b['west']},{b['north']},{mid_lon}",
            f"{mid_lat},{mid_lon},{b['north']},{b['east']}",
        ]

        all_elements: list[dict[str, Any]] = []

        # Bâtiments par quadrant
        for i, quad_bbox in enumerate(quadrants):
            try:
                logger.debug("   Bâtiments quadrant %d/4...", i + 1)
                q = f"""
                [out:json][timeout:30];
                way["building"]({quad_bbox});
                out geom;
                """
                data = _overpass_query(q, timeout=45)
                all_elements.extend(data.get("elements", []))
                time.sleep(3)
            except Exception as e:
                logger.debug("   Quadrant %d: %s", i + 1, e)
                time.sleep(5)

        # Landuse + parkings + sport + cimetières
        try:
            logger.debug("   Zones d'occupation du sol...")
            time.sleep(3)
            q_landuse = f"""
            [out:json][timeout:30];
            (
              way["landuse"~"residential|commercial|industrial|retail"]({bbox_str});
              way["amenity"~"parking|school"]({bbox_str});
              way["leisure"~"pitch|sports_centre"]({bbox_str});
              way["landuse"="cemetery"]({bbox_str});
              relation["landuse"~"residential|commercial|industrial"]({bbox_str});
            );
            out geom;
            """
            data = _overpass_query(q_landuse, timeout=45)
            all_elements.extend(data.get("elements", []))
        except Exception as e:
            logger.debug("   Landuse: %s", e)

        # Routes principales
        try:
            logger.debug("   Routes principales...")
            time.sleep(3)
            q_roads = f"""
            [out:json][timeout:20];
            (
              way["highway"~"motorway|trunk|primary|secondary"]({bbox_str});
              way["railway"~"rail|light_rail"]({bbox_str});
            );
            out geom;
            """
            data = _overpass_query(q_roads, timeout=30)
            all_elements.extend(data.get("elements", []))
        except Exception as e:
            logger.debug("   Routes: %s", e)

        if not all_elements:
            raise ValueError("Aucun élément urbain OSM")

        # Déduplication par ID OSM
        seen_ids: set[tuple[str, Any]] = set()
        unique_elements: list[dict[str, Any]] = []
        for el in all_elements:
            eid = (str(el.get("type", "")), el.get("id", ""))
            if eid not in seen_ids:
                seen_ids.add(eid)
                unique_elements.append(el)

        logger.debug(
            "   %d éléments bruts → %d uniques",
            len(all_elements),
            len(unique_elements),
        )

        # Parser les géométries
        geometries: list[Any] = []
        urban_types: list[str] = []

        for el in unique_elements:
            tags: dict[str, Any] = el.get("tags", {})

            # Déterminer le type urbain
            if "building" in tags:
                utype = "batiment"
            elif "landuse" in tags:
                utype = str(tags["landuse"])
            elif "highway" in tags:
                utype = "route"
            elif "railway" in tags:
                utype = "voie_ferree"
            elif "amenity" in tags:
                utype = str(tags["amenity"])
            elif "leisure" in tags:
                utype = str(tags["leisure"])
            else:
                utype = "autre"

            # Routes et voies ferrées → LineString → buffer en L93
            if utype in ("route", "voie_ferree"):
                if el["type"] == "way" and "geometry" in el:
                    coords = [
                        (pt["lon"], pt["lat"]) for pt in el["geometry"]
                    ]
                    if len(coords) >= 2:
                        try:
                            line = LineString(coords)
                            if line.is_valid and line.length > 0:
                                geometries.append(line)
                                urban_types.append(utype)
                        except Exception:
                            pass
            # Polygones
            else:
                polys = _osm_element_to_polygons(el)
                for poly in polys:
                    geometries.append(poly)
                    urban_types.append(utype)

        if not geometries:
            raise ValueError("Aucune géométrie urbaine parsée")

        gdf = gpd.GeoDataFrame(
            {"urban_type": urban_types},
            geometry=geometries,
            crs="EPSG:4326",
        )
        gdf = self._ensure_l93(gdf)

        # Buffer des routes et voies ferrées en L93 (mètres, pas degrés)
        mask_line = gdf.geometry.geom_type.isin(
            ["LineString", "MultiLineString"]
        )
        if mask_line.any():
            buffer_widths: dict[str, int] = {
                "route": 8,
                "voie_ferree": 10,
            }
            for utype_key, buf_m in buffer_widths.items():
                mask = mask_line & (gdf["urban_type"] == utype_key)
                if mask.any():
                    gdf.loc[mask, "geometry"] = gdf.loc[
                        mask, "geometry"
                    ].buffer(buf_m)

        # Simplification pour performance
        n_before = len(gdf)
        try:
            gdf_dissolved = gdf.dissolve(by="urban_type").reset_index()
            # Éclater les MultiPolygons pour simplification
            gdf_dissolved = gdf_dissolved.explode(index_parts=False)
            gdf_dissolved.geometry = gdf_dissolved.geometry.simplify(
                tolerance=2.0
            )
            gdf = gdf_dissolved
            logger.debug(
                "   Simplifié: %d → %d géométries", n_before, len(gdf)
            )
        except Exception:
            logger.debug("   Simplification échouée, conservation brute")

        gdf["source"] = "osm"
        gdf["name"] = ""

        logger.info("✅ Urbain OSM: %d polygones", len(gdf))
        for ut, cnt in gdf["urban_type"].value_counts().items():
            logger.debug("   • %s: %d", ut, cnt)

        return gdf

    def _generate_synthetic_urban(self) -> gpd.GeoDataFrame:
        """Génère des zones urbaines synthétiques."""
        urban_zones: list[tuple[tuple[float, float, float, float], str, str]] = [
            ((5.672, 45.226, 5.686, 45.237), "residential", "Saint-Égrève centre"),
            ((5.680, 45.213, 5.700, 45.222), "commercial", "Zone commerciale sud"),
            ((5.688, 45.217, 5.698, 45.223), "residential", "Fiancey"),
            ((5.683, 45.227, 5.693, 45.233), "residential", "Monta / Clapières"),
            ((5.677, 45.222, 5.685, 45.228), "residential", "Cuvilleux"),
            ((5.670, 45.213, 5.720, 45.216), "route", "D1075 / voie ferrée"),
            ((5.698, 45.224, 5.704, 45.228), "residential", "Champy"),
            (
                (5.670, 45.210, 5.720, 45.214),
                "industrial",
                "Bord Isère / ZI",
            ),
        ]

        records: list[dict[str, Any]] = []
        for (w, s, e, n), utype, name in urban_zones:
            sw_x, sw_y = _TO_L93.transform(w, s)
            ne_x, ne_y = _TO_L93.transform(e, n)
            records.append(
                {
                    "urban_type": utype,
                    "name": name,
                    "source": "synthetic",
                    "geometry": box(sw_x, sw_y, ne_x, ne_y),
                }
            )

        gdf = gpd.GeoDataFrame(records, crs=TARGET_CRS)
        logger.info("✅ Urbain synthétique: %d zones", len(gdf))
        for _, row in gdf.iterrows():
            logger.debug("   • %s (%s)", row["name"], row["urban_type"])
        return gdf

    # ═══════════════════════════════════════════════════════
    # LECTURE FICHIER VECTORIEL GÉNÉRIQUE
    # ═══════════════════════════════════════════════════════

    def _read_vector_file(
        self, filepath: str, label: str = "Couche"
    ) -> gpd.GeoDataFrame | None:
        """
        Lecture d'un fichier vectoriel avec reprojection automatique.
        Filtre spatial par BBOX buffered.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("⚠️ Fichier introuvable: %s", filepath)
            return None

        try:
            # Lire d'abord le CRS sans charger toutes les données
            import fiona

            with fiona.open(path) as src:
                file_crs = src.crs_wkt if src.crs_wkt else None
                file_epsg = src.crs.get("init", "") if src.crs else ""

            # Déterminer le BBOX de filtre dans le bon CRS
            if file_crs and "2154" in str(file_crs) + str(file_epsg):
                filter_bbox = (
                    self.bbox_buffered["xmin"],
                    self.bbox_buffered["ymin"],
                    self.bbox_buffered["xmax"],
                    self.bbox_buffered["ymax"],
                )
            else:
                # Fichier probablement en WGS84 → filtre en WGS84
                filter_bbox = (
                    self.bbox_wgs84_buffered["west"],
                    self.bbox_wgs84_buffered["south"],
                    self.bbox_wgs84_buffered["east"],
                    self.bbox_wgs84_buffered["north"],
                )
                logger.debug(
                    "   Fichier non-L93 (%s) → filtre WGS84", file_epsg
                )
        except Exception:
            # Pas de fiona ou erreur → tenter sans filtre CRS
            filter_bbox = (
                self.bbox_buffered["xmin"],
                self.bbox_buffered["ymin"],
                self.bbox_buffered["xmax"],
                self.bbox_buffered["ymax"],
            )

        try:
            gdf = gpd.read_file(path, bbox=filter_bbox)
            gdf = self._ensure_l93(gdf)
            logger.info(
                "✅ %s fichier: %s (%d entités)", label, path.name, len(gdf)
            )
            return gdf
        except Exception as e:
            logger.error("❌ Lecture %s échouée: %s", filepath, e)
            return None

    # ═══════════════════════════════════════════════════════
    # UTILITAIRES
    # ═══════════════════════════════════════════════════════

    def check_network(self, timeout: int = 5) -> bool:
        """Test rapide de connectivité réseau."""
        try:
            requests.head("https://data.geopf.fr", timeout=timeout)
            return True
        except Exception:
            return False

    def discover_wfs_layers(
        self,
        url: str = "https://data.geopf.fr/wfs/ows",
        keywords: list[str] | None = None,
    ) -> list[str]:
        """
        Liste les couches WFS pertinentes pour la mycologie/SIG.
        """
        import xml.etree.ElementTree as ET

        if keywords is None:
            keywords = [
                "foret",
                "forest",
                "vegetation",
                "landcover",
                "hydro",
                "cours",
                "eau",
                "water",
                "bdtopo",
                "geol",
                "batiment",
                "building",
            ]

        logger.info("🔍 Découverte WFS: %s", url)

        try:
            resp = requests.get(
                url,
                params={
                    "service": "WFS",
                    "version": "2.0.0",
                    "request": "GetCapabilities",
                },
                timeout=30,
            )
            resp.raise_for_status()

            root = ET.fromstring(resp.content)

            # Namespace-agnostic search
            fts: list[Any] = []
            for elem in root.iter():
                if elem.tag.endswith("FeatureType"):
                    fts.append(elem)

            matching: list[str] = []
            for ft in fts:
                name, title = "?", ""
                for child in ft:
                    if child.tag.endswith("Name"):
                        name = child.text or "?"
                    elif child.tag.endswith("Title"):
                        title = child.text or ""
                combined = (name + " " + title).lower()
                if any(kw in combined for kw in keywords):
                    line = f"  ✓ {name} — {title}"
                    matching.append(line)

            logger.info(
                "   %d couches, %d pertinentes:", len(fts), len(matching)
            )
            for m in matching[:30]:
                logger.info(m)

            return matching

        except Exception as e:
            logger.error("   ❌ %s", e)
            return []