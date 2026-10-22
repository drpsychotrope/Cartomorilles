#!/usr/bin/env python3
"""
🍄 CARTOMORILLES — Cartes de probabilité de présence de champignons
   Morille (Morchella esculenta) et cèpe (Boletus edulis) — Isère (38),
   département élargi (107×131 km, Lambert-93).

   Analyse multicritère spatialisée : 14 critères pondérés par espèce,
   masques éliminatoires, lissage. Météo du jour (Open-Meteo) affichée à part.
   Maille automatique selon l'emprise : 50 m pour le département
   (~5,6 M cellules) ; --cell-size pour la forcer.
   Sources : MNT Copernicus, IGN BD Forêt v2, BRGM BDCharm-50, IFN 1997
   (enrichissement essences), WFS IGN / OSM (hydrographie, urbain).

Usage :
    python main.py                              # Trois espèces, maille auto
    python main.py --open                       # Idem + ouvre la carte
    python main.py --species esculenta          # Morille grise seule
    python main.py --dry-run                    # Vérif données seulement
    python main.py --cell-size 25 --verbose     # Test rapide, logs détaillés

Console : étapes [1/4]…[4/4] puis résumé — fichiers produits, lien vers
la carte, top hotspots et concordance terrain de chaque espèce. Le détail
complet va dans <output-dir>/cartomorilles.log (à l'écran avec --verbose).

v2.4.0 — Extension emprise Isère 38 (74.5×98.8 km, ×18.4 vs v2.3)
         DEM Copernicus 30m, CELL_SIZE 10m, géologie geologie_38/
         Fix RÉSUMÉ FINAL dead code, _resolve_data_path rglob
v2.3.2 — Intégration géologie BDCharm-50 (1086 polygones, 99.9% résolus)
         Fix #26 : rasters catégoriels pour hotspot enrichment
         Default paths BD Forêt + BDCharm-50 + régions IFN
         Vectorisation export_gpkg_grid
v2.3.1 — Intégration SpeciesEnricher (BD Forêt v2 + IFN 1997)
         Fix châtaignier (checkpoint terrain mis à jour)
         LandcoverDetector context manager (fix #22)
v2.3.0 — Pipeline réordonné (Option 2), 7 fixes (#15, A-F)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import signal
import sys
import time
import webbrowser
import xml.etree.ElementTree as ET
from bisect import bisect_right
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import FrameType
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import requests

import config
from config import (
    BBOX,
    BBOX_WGS84,
    CELL_SIZE,
    CONFIG_VERSION,
    PROBABILITY_LABELS,
    PROBABILITY_THRESHOLDS,
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_MEDIUM,
    SUPPORTED_SPECIES,
    WEIGHTS,
    LANDCOVER_AUTO_MAX_KM2,
    CELL_SIZE_AUTO_THRESHOLDS,
    CELL_SIZE_AUTO_FALLBACK,
)
from data_loader import DataLoader
from grid_builder import GridBuilder
from scoring import MorilleScoring
from species_enricher import SpeciesEnricher
from utils import TO_L93, compute_array_stats, hotspot_to_export_dict
from visualize import HD_MAX_PIXELS, MorilleVisualizer
from visualize import _display_value as display_label  # libellés de la carte
from weather import WeatherChecker

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES MODULE
# ═══════════════════════════════════════════════════════════════

VERSION = CONFIG_VERSION

# ── Chemins par défaut des données locales ─────────────────────
_DATA_DIR = Path(__file__).resolve().parent / "data"
# DEM : bbox dept élargi couvre 6 tuiles Copernicus (N44-45 × E004-006) —
# load_dem() tombe sur _download_copernicus_dem() qui merge automatiquement
# les tuiles locales. Un chemin unique cropperait à la fenêtre d'une seule tuile.
_DEFAULT_DEM: str | None = None
_DEFAULT_BD_FORET = str(_DATA_DIR / "FORMATION_VEGETALE.shp")
_DEFAULT_GEOLOGY = str(
    _DATA_DIR / "geologie_38" / "GEO050K_HARM_038_S_FGEOL_2154.shp",
)
_DEFAULT_REGIONS = str(_DATA_DIR / "rfifn250_l93.shp")

# ── Zone (calculée depuis BBOX pour bannières/rapports) ────────
_ZONE_DX_KM = (BBOX["xmax"] - BBOX["xmin"]) / 1000
_ZONE_DY_KM = (BBOX["ymax"] - BBOX["ymin"]) / 1000
_ZONE_LABEL = f"Isère 38 — {_ZONE_DX_KM:.0f}×{_ZONE_DY_KM:.0f} km"

# ── Garde mémoire (cellules max sans --force) ─────────────────
_MAX_CELLS = 200_000_000  # 200M — protège contre cell_size=5m accidentel
_WARN_CELLS = 50_000_000  # 50M — avertit sur la mémoire et la durée
# Mémoire de pointe mesurée (trois espèces, carte HD) : 13,0 Go pour 35,2 M
# cellules (20 m), 36,1 Go pour 140,9 M (10 m) ≈ 5,3 Go fixes + 220 octets
# par cellule (D62) ; terrain calculé sur le RGE ALTI à 10 m : 38,2 Go à
# 10 m (D63), d'où ~2 Go fixes de plus ; pertes de couvert (année par
# cellule, perturbation de l'étape 2 gardée à part) : 39,0 Go (D64)
_PEAK_FIXED_BYTES = 7.3e9
_PEAK_BYTES_PER_CELL = 226

# ── Résumé console / carte ────────────────────────────────────
_CONSOLE_HOTSPOTS_DEFAULT = 5   # hotspots listés par espèce en fin de run
_MAP_HOTSPOTS_DEFAULT = 20      # marqueurs numérotés à l'ouverture de la carte

# Points de contrôle terrain — MORILLE (prospection réelle mars 2026)
# v2.3.1 : châtaignier corrigé — plus éliminatoire
TERRAIN_CHECKPOINTS_ESCULENTA: tuple[dict[str, Any], ...] = (
    # ── Contrôles positifs (morilles attendues) ──────────────────
    {
        "name": "Champy – châtaigneraie",
        "lat": 45.24308, "lon": 5.69736, "expected": 0.55,
        "obs": "châtaignier favorable (M. elata), sol frais versant",
    },
    {
        "name": "Terrasse plan d'eau + conifères",
        "lat": 45.24087, "lon": 5.69484, "expected": 0.50,
        "obs": "meilleur spot trouvé, M. elata possible",
    },
    # ── Contrôles négatifs locaux (même zone, micro-habitat) ────
    {
        "name": "Berge Vence – trop humide",
        "lat": 45.24588, "lon": 5.69744, "expected": 0.05,
        "obs": "lierre dense, scolopendres — LIMITATION micro-habitat",
        "tolerance": 0.50,
    },
    {
        "name": "Ravine au-dessus Vence",
        "lat": 45.24637, "lon": 5.70049, "expected": 0.05,
        "obs": "Trop humide",
        "tolerance": 0.50,
    },
    {
        "name": "Mi-pente Néron chênaie+buis",
        "lat": 45.24467, "lon": 5.69871, "expected": 0.10,
        "obs": "Trop sec, chênaie buissonnante",
        "tolerance": 0.50,
    },
    {
        "name": "Robinier + hêtres secteur Champy",
        "lat": 45.24242, "lon": 5.69658, "expected": 0.15,
        "obs": "Espèce eliminatoire (robinier), hêtres peu favorables",
        "tolerance": 0.50,
    },
    # ── Contrôles négatifs forts (vrai négatif attendu=0) ───────
    {
        "name": "Centre-ville Grenoble",
        "lat": 45.1885, "lon": 5.7245, "expected": 0.00,
        "obs": "urbain dense — doit être éliminé",
    },
    {
        "name": "Belledonne granite 1880m",
        "lat": 45.11849, "lon": 5.88389, "expected": 0.00,
        "obs": "granite éliminatoire — substrat acide",
    },
    {
        "name": "Sommet Néron 1299m",
        "lat": 45.23732, "lon": 5.70991, "expected": 0.00,
        "obs": "altitude > 1400m ou roche nue",
    },
    {
        "name": "Isère lit majeur",
        "lat": 45.20003, "lon": 5.74186, "expected": 0.00,
        "obs": "plan d'eau — masque eau",
    },
    # ── Contrôle positif éloigné (diversité géo) ────────────────
    {
        "name": "Vouillants forêt calcaire 350m",
        "lat": 45.18824, "lon": 5.66543, "expected": 0.70,
        "obs": "forêt calcaire optimale, altitude idéale",
    },
)

# Alias rétro-compat (modules tiers / scripts ancien nom)
TERRAIN_CHECKPOINTS = TERRAIN_CHECKPOINTS_ESCULENTA

# Points de contrôle terrain — CÈPE (Boletus edulis, biblio)
# v2.6.0 : sites typés cèpe — montagnard, mycorhize Pinaceae/Fagaceae,
# pas d'élimination granite (cèpe ubiquiste).
# D58 : les coordonnées rondes d'origine ne tombaient pas sur l'habitat
# décrit (crête à 2 504 m pour « Belledonne 1500m », châtaigneraie à 586 m
# pour « Plaine Bièvre », hors Isère pour « Lautaret » et « Vercors »…) ;
# chaque point est déplacé sur la cellule la plus proche qui correspond à
# sa description (essence BD Forêt publiée, étage, géologie, pente ≤ 30°,
# au cœur du peuplement, dans la zone étudiée), choisie sans regarder le
# score du modèle. Anciennes coordonnées en commentaire.
TERRAIN_CHECKPOINTS_EDULIS: tuple[dict[str, Any], ...] = (
    # ── Contrôles positifs (cèpes attendus) ──────────────────────
    {
        # avant : 45.0660, 5.4630 (hors Isère, aucune forêt à 600 m)
        "name": "Vercors hêtraie-sapinière 1100m",
        "lat": 45.08589, "lon": 5.44140, "expected": 0.70,
        "obs": "mélange hêtre-sapin (BD Forêt « Mixte »), calcaire, 1 134 m",
    },
    {
        # avant : 45.3500, 5.8000 (pente de 52°, éliminée)
        "name": "Chartreuse sapinière 1100m",
        "lat": 45.34998, "lon": 5.80190, "expected": 0.65,
        "obs": "sapinière-pessière sur éboulis calcaire, 1 032 m",
    },
    {
        # avant : 45.1700, 5.9500 (crête à 2 504 m, hors forêt)
        "name": "Belledonne pessière siliceuse 1500m",
        "lat": 45.17943, "lon": 5.91695, "expected": 0.65,
        "obs": "pessière sur schiste (sol acide), 1 547 m — INVERSE morille",
    },
    {
        # avant : 44.8500, 5.6500 (pinède à 574 m)
        "name": "Trièves hêtraie 750m",
        "lat": 44.84409, "lon": 5.67280, "expected": 0.55,
        "obs": "hêtraie sur calcaire, versant de montagne basse",
        "tolerance": 0.50,
    },
    {
        # avant : 45.4500, 5.4000 (hors forêt, 66 m d'un peuplement)
        "name": "Châtaigneraie Bas-Dauphiné 570m",
        "lat": 45.44982, "lon": 5.40170, "expected": 0.60,
        "obs": "châtaigneraie, sol acide — bon habitat cèpe",
        "tolerance": 0.50,
    },
    # ── Contrôles négatifs (cèpes peu probables) ────────────────
    {
        # avant : 45.4500, 5.2500 (châtaigneraie à 586 m)
        "name": "Plaine Bièvre peupleraie 340m",
        "lat": 45.38068, "lon": 5.12254, "expected": 0.15,
        "obs": "peupleraie de plaine alluviale, trop basse",
        "tolerance": 0.50,
    },
    {
        "name": "Centre-ville Grenoble",
        "lat": 45.1885, "lon": 5.7245, "expected": 0.00,
        "obs": "urbain dense — éliminé",
    },
    {
        # remplace « Lautaret alpin 2050m » (45.0350, 6.4050, hors Isère)
        "name": "Alpage sans arbre 2100m (Écrins)",
        "lat": 44.89517, "lon": 6.30222, "expected": 0.00,
        "obs": "pelouse alpine, aucun arbre hôte à 300 m — éliminé (D57)",
    },
    {
        "name": "Isère lit majeur",
        "lat": 45.20003, "lon": 5.74186, "expected": 0.00,
        "obs": "plan d'eau — masque eau",
    },
)


# Dispatch checkpoints par espèce
TERRAIN_CHECKPOINTS_BY_SPECIES: dict[str, tuple[dict[str, Any], ...]] = {
    "esculenta": TERRAIN_CHECKPOINTS_ESCULENTA,
    "elata": (),   # pas encore de points terrain : validation sautée (D49)
    "edulis": TERRAIN_CHECKPOINTS_EDULIS,
}

# Fichiers partiels à nettoyer en cas d'interruption
_cleanup_files: list[str] = []

logger = logging.getLogger("cartomorilles")


# ═══════════════════════════════════════════════════════════════
#  CACHE — purge conditionnelle
# ═══════════════════════════════════════════════════════════════


def _purge_cache() -> None:
    """Purge le cache des tuiles et données temporaires."""
    import shutil

    for d in [Path("data/cache"), Path("data/osm_tiles")]:
        if d.exists():
            shutil.rmtree(d)
            logger.info("🗑️ Cache purgé : %s", d)


# ═══════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════


class _ConsoleFilter(logging.Filter):
    """Console concise : les messages du pipeline (logger « cartomorilles »
    de main.py) et les avertissements de tous les modules. Le détail des
    modules (INFO/DEBUG) ne va que dans le fichier log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or record.name == logger.name


class _ConsoleFormatter(logging.Formatter):
    """Message seul, sans horodatage ; ⚠️ / ❌ devant les avertissements
    qui n'ont pas déjà leur pictogramme."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        body = text.lstrip()
        if record.levelno < logging.WARNING or not body[:1].isalnum():
            return text
        mark = "❌" if record.levelno >= logging.ERROR else "⚠️ "
        return f"{text[: len(text) - len(body)]}{mark} {body}"


def _ensure_utf8_stdout() -> None:
    """Sortie redirigée (fichier, pipe) sous Windows : cp1252 ne sait pas
    écrire les pictogrammes de la console et de --help → UTF-8."""
    stream = sys.stdout
    if (
        isinstance(stream, io.TextIOWrapper)
        and stream.encoding.lower().replace("-", "") != "utf8"
    ):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def setup_logging(output_dir: Path, verbose: bool = False) -> Path:
    """Configure le logging et retourne le chemin du fichier log.

    Console : étapes et résumé seulement (_ConsoleFilter), ou tout le
    détail DEBUG horodaté avec --verbose. Fichier : détail DEBUG complet.
    """
    root = logging.getLogger("cartomorilles")
    root.setLevel(logging.DEBUG)

    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(fmt)
    else:
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(_ConsoleFilter())
        console_handler.setFormatter(_ConsoleFormatter("%(message)s"))
    root.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "cartomorilles.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return log_path


# ═══════════════════════════════════════════════════════════════
#  FORMATAGE CONSOLE (mêmes conventions que la carte HTML)
# ═══════════════════════════════════════════════════════════════


def _fr_num(value: float, decimals: int = 0) -> str:
    """Nombre à la française (espace des milliers, virgule décimale),
    arrondi au demi supérieur comme le toLocaleString('fr-FR') de la carte."""
    q = Decimal(str(value)).quantize(
        Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP,
    )
    return f"{q:,.{decimals}f}".replace(",", " ").replace(".", ",")


def _fr_duration(seconds: float) -> str:
    """12,3 s · 4 min 05 s."""
    if seconds < 60:
        return f"{_fr_num(round(seconds, 1), 1)} s"
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"{minutes} min {sec:02d} s"


def _fr_size(n_bytes: int) -> str:
    """Taille de fichier : 820 ko · 21,8 Mo."""
    if n_bytes < 1024 * 1024:
        return f"{_fr_num(max(min(n_bytes, 1), round(n_bytes / 1024)))} ko"
    return f"{_fr_num(round(n_bytes / 1024 / 1024, 1), 1)} Mo"


def _fr_area(size_m2: float) -> str:
    """Surface d'un hotspot, arrondie comme dans son popup sur la carte."""
    ha = round(size_m2 / 10_000.0, 2)
    if ha >= 10:
        return f"{_fr_num(ha)} ha"
    if ha >= 1:
        return f"{_fr_num(ha, 1)} ha"
    return f"{_fr_num(ha * 10_000)} m²"


def _enumerate_fr(items: Sequence[str]) -> str:
    """« a », « a et b », « a, b et c »."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


def _species_label(species: str) -> str:
    """« Morille grise (Morchella esculenta) » — sans emoji d'espèce : 🍄 est
    l'amanite tue-mouches, 🌰 une châtaigne (essence du modèle)."""
    params = config.SPECIES_PARAMS.get(species, {})
    return (
        f"{params.get('common_name', species)} "
        f"({params.get('scientific_name', species)})"
    )


# Index de la classe « Bon » : son seuil inférieur est le seuil par défaut
# des hotspots de chaque espèce (D52)
_BON_INDEX: int = PROBABILITY_LABELS.index("Bon")


def _study_area_mask(grid: Any, geology_source: str) -> np.ndarray | None:
    """Zone étudiée sur la grille L93 (D39) : cellules couvertes par la
    géologie BDCharm-50 départementale. None sans BDCharm-50 (toute la
    grille compte alors pour les classes par rang)."""
    int_raster = getattr(grid, "_geology_int_raster", None)
    if geology_source != "bdcharm50" or int_raster is None:
        return None
    return np.asarray(int_raster) > 0


# ═══════════════════════════════════════════════════════════════
#  RÉSOLUTION INTELLIGENTE DES CHEMINS DE DONNÉES
# ═══════════════════════════════════════════════════════════════


def _resolve_data_path(
    cli_arg: str | None,
    default: str | None,
    label: str,
    glob_pattern: str | None,
    level: int = logging.DEBUG,
) -> str | None:
    """Résout un chemin de données : CLI > défaut > glob récursif.

    Parameters
    ----------
    cli_arg : argument CLI explicite (prioritaire).
    default : chemin par défaut absolu (None = pas de défaut fichier unique,
              ex. DEM multi-tuiles géré par load_dem).
    label : nom humain pour le logging.
    glob_pattern : pattern glob fallback dans data/ et sous-dossiers
                   (None = pas de glob).
    level : niveau de log des chemins trouvés (INFO en --dry-run).
    """
    # 1. Argument CLI explicite
    if cli_arg is not None:
        p = Path(cli_arg)
        if p.exists():
            logger.log(level, "   ✅ %s (CLI) : %s", label, p)
            return str(p)
        logger.warning("   ❌ %s (CLI) introuvable : %s", label, p)

    # 2. Chemin par défaut
    if default is not None:
        dp = Path(default)
        if dp.exists():
            logger.log(level, "   ✅ %s : %s", label, dp.name)
            return str(dp)
    else:
        dp = None

    # 3. Glob récursif dans data/ (couvre sous-dossiers type geologie_38/),
    #    à côté de main.py puis dans le dossier courant : data_loader lit
    #    data/ dans le dossier courant (MNT, caches), un main.py lancé
    #    depuis une autre copie du dépôt doit trouver les mêmes fichiers
    if glob_pattern is not None:
        for data_dir in dict.fromkeys(
            (_DATA_DIR, (Path.cwd() / "data").resolve()),
        ):
            candidates = sorted(data_dir.rglob(glob_pattern)) if data_dir.is_dir() else []
            if candidates:
                chosen = candidates[0]
                logger.log(level, "   ✅ %s (glob) : %s", label, chosen)
                return str(chosen)

    if default is None and glob_pattern is None:
        logger.log(
            level, "   ℹ️ %s : aucun défaut local (géré par loader)", label,
        )
    else:
        logger.warning(
            "   ❌ %s introuvable (défaut=%s, glob=%s)",
            label,
            dp.name if dp else "None",
            glob_pattern,
        )
    return None


# ═══════════════════════════════════════════════════════════════
#  VALIDATION & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def _auto_cell_size(area_km2: float) -> float:
    """Résolution automatique adaptée à l'emprise.

    Évite le suréchantillonnage inutile : le DEM Copernicus est à 25-30m,
    BD Forêt et géologie sont à l'échelle hectométrique.
    """
    for max_area, cell in CELL_SIZE_AUTO_THRESHOLDS:
        if area_km2 <= max_area:
            return cell
    return CELL_SIZE_AUTO_FALLBACK

def validate_weights() -> bool:
    """Vérifie la somme des poids (détail dans le log)."""
    total = sum(WEIGHTS.values())
    ok = abs(total - 1.0) < 0.01

    logger.debug("Pondérations du modèle :")
    for key, val in sorted(WEIGHTS.items(), key=lambda x: -x[1]):
        logger.debug("  %-25s %.2f", key, val)
    logger.debug("  %s ────", "─" * 25)
    logger.debug(
        "  %-25s %.2f  %s", "TOTAL", total, "✅" if ok else "⚠️ ≠ 1.0",
    )

    if not ok:
        logger.warning("Somme des poids = %.4f, attendu 1.0000", total)
    return ok


def estimate_grid_size(cell_size: float) -> dict[str, Any]:
    """Estime le nombre de cellules, la surface et la RAM nécessaire."""
    dx = BBOX["xmax"] - BBOX["xmin"]
    dy = BBOX["ymax"] - BBOX["ymin"]
    nx = int(dx / cell_size)
    ny = int(dy / cell_size)
    n_cells = nx * ny
    mem_mb = (_PEAK_FIXED_BYTES + n_cells * _PEAK_BYTES_PER_CELL) / 1e6  # D62

    return {
        "dx_m": dx, "dy_m": dy,
        "nx": nx, "ny": ny,
        "n_cells": n_cells,
        "area_km2": dx * dy / 1e6,
        "mem_estimate_mb": mem_mb,
    }


def summarize_data(
    dem_data: dict[str, Any],
    forest_gdf: Any,
    geology_gdf: Any,
    hydro_gdf: Any,
    urban_gdf: Any,
    landcover_data: dict[str, Any] | None,
    enricher_stats: dict[str, Any] | None = None,
    geology_source: str | None = None,
    level: int = logging.DEBUG,
) -> None:
    """Tableau récapitulatif des données chargées (log ; console en
    --dry-run via level=INFO)."""
    logger.log(level, "")
    logger.log(level, "─" * 55)
    logger.log(level, "📋 Résumé des données chargées :")

    alt = dem_data["data"]
    logger.log(
        level, "   MNT       : %d×%dpx, alt %.0f–%.0fm",
        alt.shape[1], alt.shape[0],
        np.nanmin(alt), np.nanmax(alt),
    )

    layers: tuple[tuple[str, Any], ...] = (
        ("Forêt", forest_gdf),
        ("Géologie", geology_gdf),
        ("Hydro", hydro_gdf),
        ("Urbain", urban_gdf),
    )
    for name, gdf in layers:
        if gdf is not None and len(gdf) > 0:
            src = ""
            if hasattr(gdf, "columns") and "source" in gdf.columns:
                sources = gdf["source"].unique()
                src = f" ({', '.join(str(s) for s in sources[:3])})"
            logger.log(level, "   %-10s: %5d entités%s", name, len(gdf), src)
        else:
            logger.log(level, "   %-10s: ❌ (absente ou vide)", name)

    if geology_source:
        logger.log(level, "   Géo source: %s", geology_source)

    lc = "✅ analyse couleur OSM" if landcover_data else "❌"
    logger.log(level, "   Landcover : %s", lc)

    if enricher_stats:
        logger.log(
            level, "   Essences  : %s → %d polygones",
            enricher_stats.get("source", "?"),
            enricher_stats.get("bd_foret_polygons", 0),
        )

    logger.log(level, "─" * 55)


def compute_statistics(
    final_score: np.ndarray,
    threshold: float,
    class_thresholds: list[float] | None = None,
    inside: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calcule les statistiques du score final.

    `classes` = répartition dans les 6 classes de la carte, avec les seuils
    de l'espèce (`class_thresholds`, par rang depuis D52 ; seuils absolus
    config.PROBABILITY_THRESHOLDS à défaut), en % des cellules avec données
    de toute la grille ; `classes_study_area` = la même chose sur la zone
    étudiée (`inside`), ce que montre la carte. Les clés n_high / n_medium /
    n_low (seuils 0.7 / 0.4) restent dans le rapport JSON pour comparer avec
    les calibrations passées (D35).
    """
    thresholds = list(
        class_thresholds if class_thresholds is not None
        else config.PROBABILITY_THRESHOLDS,
    )
    base = compute_array_stats(final_score, decimals=4)
    finite = np.isfinite(final_score)
    n_valid = int(np.count_nonzero(finite))
    bounds = [0.0, *thresholds, 1.0]

    def class_shares(values: np.ndarray) -> list[dict[str, Any]]:
        counts = np.bincount(
            np.digitize(values, thresholds),
            minlength=len(config.PROBABILITY_LABELS),
        )
        return [
            {
                "label": label,
                "min": round(lo, 4),
                "max": round(hi, 4),
                "n": int(n),
                "pct": round(float(n) / max(values.size, 1) * 100, 2),
            }
            for label, lo, hi, n in zip(
                config.PROBABILITY_LABELS, bounds[:-1], bounds[1:], counts,
            )
        ]

    classes = class_shares(final_score[finite])
    classes_study_area = (
        class_shares(final_score[finite & inside]) if inside is not None
        else None
    )
    return {
        "n_cells": int(final_score.size),
        "n_valid": n_valid,
        "score_min": base["min"],
        "score_max": base["max"],
        "score_mean": base["mean"],
        "score_median": base["median"],
        "score_std": base["std"],
        "class_thresholds": [round(t, 4) for t in thresholds],
        "class_rank_quantiles": list(config.CLASS_RANK_QUANTILES),
        "classes": classes,
        "classes_study_area": classes_study_area,
        "n_high": int(np.sum(final_score >= SCORE_THRESHOLD_HIGH)),
        "n_medium": int(np.sum(
            (final_score >= SCORE_THRESHOLD_MEDIUM)
            & (final_score < SCORE_THRESHOLD_HIGH)
        )),
        "n_low": int(np.sum(final_score < SCORE_THRESHOLD_MEDIUM)),
        "pct_high": round(
            float(np.sum(final_score >= SCORE_THRESHOLD_HIGH))
            / final_score.size * 100, 2,
        ),
        "pct_medium": round(
            float(np.sum(
                (final_score >= SCORE_THRESHOLD_MEDIUM)
                & (final_score < SCORE_THRESHOLD_HIGH)
            )) / final_score.size * 100, 2,
        ),
        "hotspot_threshold": threshold,
    }


def _class_range(lo: float, hi: float) -> str:
    """Bornes d'une classe, comme la légende de la carte."""
    if lo <= 0.0:
        return f"< {hi:.2f}"
    if hi >= 1.0:
        return f"≥ {lo:.2f}"
    return f"{lo:.2f}–{hi:.2f}"


def _score_class(score: float, thresholds: list[float] | None = None) -> str:
    """Classe de probabilité (libellé) d'un score, avec les seuils de son
    espèce (par rang, D52) ou, à défaut, les seuils absolus."""
    idx = int(np.digitize(
        score, thresholds if thresholds is not None
        else config.PROBABILITY_THRESHOLDS,
    ))
    return config.PROBABILITY_LABELS[idx]


def display_statistics(stats: dict[str, Any]) -> None:
    """Statistiques du score final d'une espèce (log) : mêmes 6 classes
    que la légende de la carte, de la meilleure à la plus faible."""
    logger.debug("")
    logger.debug("📊 Statistiques du score final [%s] :", stats.get("species", "?"))
    logger.debug(
        "   Min=%.3f  Max=%.3f  Moy=%.3f  Méd=%.3f  σ=%.3f",
        stats["score_min"], stats["score_max"],
        stats["score_mean"], stats["score_median"],
        stats["score_std"],
    )
    for i, c in reversed(list(enumerate(stats["classes"]))):
        label = "Nul ou exclu" if i == 0 else c["label"]
        logger.debug(
            "   %-13s %-10s : %12s cellules (%5.1f %%) %s",
            label, _class_range(c["min"], c["max"]), f"{c['n']:,}",
            c["pct"], "█" * int(c["pct"] / 2),
        )
    n_nodata = stats["n_cells"] - stats["n_valid"]
    if n_nodata > 0:
        logger.debug(
            "   %-24s : %12s cellules", "Sans données", f"{n_nodata:,}",
        )


# ═══════════════════════════════════════════════════════════════
#  VALIDATION TERRAIN
# ═══════════════════════════════════════════════════════════════


class TerrainConcordance(NamedTuple):
    """Accord du modèle avec les points de contrôle terrain d'une espèce."""

    rate: float      # 0–1
    n_ok: int        # points dans la tolérance
    n_points: int    # points évalués (dans la grille)


def validate_against_terrain(
    model: MorilleScoring,
    checkpoints: tuple[dict[str, Any], ...] | None = None,
) -> TerrainConcordance | None:
    """
    Compare le score modèle aux observations de terrain (détail par point
    dans le log). Retourne la concordance, ou None si impossible.

    v2.6.0 : checkpoints species-aware (TERRAIN_CHECKPOINTS_BY_SPECIES
    [model.species]) ; une espèce sans points de contrôle saute la
    validation (None), sans retomber sur ceux d'une autre espèce (D49).
    """
    if checkpoints is None:
        checkpoints = TERRAIN_CHECKPOINTS_BY_SPECIES.get(model.species, ())
    if not checkpoints:
        logger.debug(
            "📍 Validation croisée terrain [%s] : aucun point de contrôle",
            model.species,
        )
        return None
    try:
        grid = model.grid

        _scores = model.final_score
        if _scores is None:
            logger.debug("Score final non calculé, validation impossible.")
            return None
        scores: np.ndarray = _scores

        concordance_list: list[float] = []
        logger.debug("")
        logger.debug("📍 Validation croisée terrain [%s] :", model.species)

        if not hasattr(grid, "x_coords") or not hasattr(grid, "y_coords"):
            logger.debug("   Grille sans coordonnées, validation impossible.")
            return None
        xs = np.asarray(grid.x_coords, dtype=np.float64)
        ys = np.asarray(grid.y_coords, dtype=np.float64)
        half = float(getattr(grid, "cell_size", abs(xs[1] - xs[0]))) / 2.0
        x_lo, x_hi = min(xs[0], xs[-1]) - half, max(xs[0], xs[-1]) + half
        y_lo, y_hi = min(ys[0], ys[-1]) - half, max(ys[0], ys[-1]) + half

        for pt in checkpoints:
            x, y = TO_L93.transform(pt["lon"], pt["lat"])
            # argmin rend toujours un indice : un point hors de la grille
            # était noté sur la cellule du bord la plus proche (D60)
            if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
                logger.debug("   ⚠️  %s hors grille — ignoré", pt["name"])
                continue
            ix = int(np.argmin(np.abs(xs - x)))
            iy = int(np.argmin(np.abs(ys - y)))
            if not np.isfinite(scores[iy, ix]):
                logger.debug("   ⚠️  %s sans score (NoData) — ignoré", pt["name"])
                continue

            if 0 <= iy < scores.shape[0] and 0 <= ix < scores.shape[1]:
                predicted = float(scores[iy, ix])
                expected: float = float(pt["expected"])
                diff = abs(predicted - expected)
                tolerance: float = float(pt.get("tolerance", 0.25))
                ok = diff < tolerance

                symbol = "✅" if ok else "❌"
                concordance_list.append(1.0 if ok else 0.0)

                logger.debug(
                    "   %s %-40s attendu=%.2f modèle=%.2f Δ=%.2f",
                    symbol, pt["name"], expected, predicted, diff,
                )
            else:
                logger.debug("   ⚠️  %s hors grille", pt["name"])

        if concordance_list:
            n_ok = sum(1 for c in concordance_list if c)
            rate = n_ok / len(concordance_list)
            logger.debug(
                "   Concordance : %.0f%% (%d/%d points)",
                rate * 100, n_ok, len(concordance_list),
            )
            return TerrainConcordance(rate, n_ok, len(concordance_list))

    except Exception as e:
        logger.debug("Validation terrain impossible : %s", e)

    return None


# ═══════════════════════════════════════════════════════════════
#  RÉSUMÉ PAR ESPÈCE (hotspots + concordance terrain)
# ═══════════════════════════════════════════════════════════════


def _hotspot_facts(
    h: dict[str, Any], thresholds: list[float] | None = None,
) -> str:
    """Score et classe, surface, altitude, pente, essence et géologie d'un
    hotspot, avec les arrondis de son popup sur la carte."""
    score = round(float(h.get("mean_score", 0.0)), 3)
    label = PROBABILITY_LABELS[
        bisect_right(thresholds or list(PROBABILITY_THRESHOLDS), score)
    ]
    parts: list[str] = [
        f"score {_fr_num(score, 2)} ({label})",
        _fr_area(float(h.get("size_m2", 0.0))),
    ]
    alt = h.get("altitude")
    if alt is not None and np.isfinite(alt):
        parts.append(f"alt. {int(round(float(alt)))} m")
    slope = h.get("mean_slope")
    if slope is not None and np.isfinite(slope):
        parts.append(f"pente {_fr_num(round(float(slope), 1))}°")
    # Libellés de la carte (config) ; essence ou géologie inconnue : rien
    for key, label_of in (
        ("dominant_species", config.tree_label),
        ("dominant_geology", config.geology_label),
    ):
        value = h.get(key)
        if str(value).strip().lower() in {"unknown", "inconnu", "?"}:
            continue
        text = label_of(value)
        if text:
            parts.append(text)
    return " · ".join(parts)


def display_species_summary(
    species: str,
    hotspots: list[dict[str, Any]],
    concordance: TerrainConcordance | None,
    *,
    threshold: float,
    n_show: int,
    inside: np.ndarray | None = None,
    thresholds: list[float] | None = None,
) -> None:
    """Top hotspots d'une espèce puis sa concordance terrain.

    Le rang (n°1, n°2…) est celui du CSV et le numéro du marqueur sur la
    carte, comme dans son popup « hotspot n°1 » (hotspots triés par score
    décroissant). `inside` (masque aligné sur `hotspots`) écarte, comme la
    carte, les hotspots hors de la zone étudiée (D39) ; les rangs gardent
    alors leurs trous."""
    logger.info("")
    if not hotspots:
        logger.info(
            "%s — aucun hotspot au-dessus du seuil %s",
            _species_label(species), _fr_num(threshold, 2),
        )
    else:
        ranked = [
            (rank, h) for rank, h in enumerate(hotspots, 1)
            if inside is None or bool(inside[rank - 1])
        ]
        n_out = len(hotspots) - len(ranked)
        out = (
            f" ; {_fr_num(n_out)} hors zone étudiée écarté"
            f"{'s' if n_out > 1 else ''}" if n_out else ""
        )
        n = min(max(n_show, 0), len(ranked))
        logger.info(
            "%s — %s hotspots (score ≥ %s%s)%s",
            _species_label(species), _fr_num(len(hotspots)),
            _fr_num(threshold, 2), out, f", top {n} :" if n else "",
        )
        for rank, h in ranked[:n]:
            pt = hotspot_to_export_dict(h)
            logger.info("   n°%-2d %s", rank, _hotspot_facts(h, thresholds))
            logger.info(
                "        %.5f, %.5f · %s",
                pt["lat"], pt["lon"], pt["google_maps"],
            )

    if concordance is None:
        return
    if concordance.rate >= 0.8:
        level, mark, verdict = logging.INFO, "✅", "bonne"
    elif concordance.rate >= 0.5:
        level, mark, verdict = (
            logging.INFO, "⚠️ ", "moyenne, recalibrer les poids ?",
        )
    else:
        level, mark, verdict = logging.WARNING, "❌", "faible, modèle à revoir"
    logger.log(
        level,
        "   %s Concordance terrain : %s %% (%d/%d points de contrôle) — %s",
        mark, _fr_num(concordance.rate * 100), concordance.n_ok,
        concordance.n_points, verdict,
    )


# ═══════════════════════════════════════════════════════════════
#  EXPORT HOTSPOTS CSV
# ═══════════════════════════════════════════════════════════════


def save_hotspots_csv(
    output_dir: Path,
    hotspots: list[dict[str, Any]],
    species: str = "esculenta",
) -> Path:
    """Exporte tous les hotspots en CSV triés par score décroissant.

    v2.6.0 : nom CSV templaté par espèce — `{species}_hotspots.csv`.
    Colonnes `lat`, `lon`, `x_l93`, `y_l93`, `google_maps` : la cellule de
    l'amas au score maximal (`max_score`), le point à viser — pas le
    centroïde de l'amas (D50).
    """
    csv_path = output_dir / f"{species}_hotspots.csv"

    fields = [
        "rank", "id", "mean_score", "max_score",
        "n_cells", "size_m2", "altitude", "mean_slope",
        "dominant_species", "dominant_geology", "compactness", "confidence",
        "lat", "lon", "x_l93", "y_l93", "google_maps",
    ]

    rows: list[dict[str, Any]] = []
    for rank, h in enumerate(hotspots, 1):
        enriched = hotspot_to_export_dict(h, rank=rank)
        rows.append({
            "rank": rank,
            "id": h.get("id", ""),
            "mean_score": round(float(h.get("mean_score", 0.0)), 4),
            "max_score": round(float(h.get("max_score", 0.0)), 4),
            "n_cells": h.get("n_cells", 0),
            "size_m2": h.get("size_m2", 0),
            "altitude": round(float(h.get("altitude", 0.0)), 1),
            "mean_slope": round(float(h.get("mean_slope", 0.0)), 1),
            "dominant_species": h.get("dominant_species", "unknown"),
            "dominant_geology": h.get("dominant_geology", "unknown"),
            "compactness": round(float(h.get("compactness", 0.0)), 3),
            "confidence": round(float(h.get("confidence", 0.0)), 2),
            "lat": enriched["lat"],
            "lon": enriched["lon"],
            "x_l93": round(float(h["x_l93"]), 1),
            "y_l93": round(float(h["y_l93"]), 1),
            "google_maps": enriched["google_maps"],
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.debug("📍 Hotspots CSV : %s (%d lignes)", csv_path, len(rows))
    return csv_path


# ═══════════════════════════════════════════════════════════════
#  EXPORT HOTSPOTS GPX
# ═══════════════════════════════════════════════════════════════

_GPX_NS = "http://www.topografix.com/GPX/1/1"
# Autant de points que de hotspots affichés à l'ouverture de la carte HTML
GPX_MAX_WAYPOINTS = 20


def _fr_decimal(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _area_text(size_m2: float) -> str:
    """Surface lisible, comme dans les popups de la carte."""
    ha = size_m2 / 10_000.0
    if ha >= 10:
        return f"{ha:.0f} ha"
    if ha >= 1:
        return f"{_fr_decimal(ha, 1)} ha"
    return f"{size_m2:.0f} m²"


def save_hotspots_gpx(
    output_dir: Path,
    hotspots: list[dict[str, Any]],
    species: str = "esculenta",
    max_waypoints: int = GPX_MAX_WAYPOINTS,
    inside: np.ndarray | None = None,
    thresholds: list[float] | None = None,
) -> Path:
    """Exporte les meilleurs hotspots en GPX 1.1 (un waypoint chacun) pour
    les applis GPS — `{species}_hotspots.gpx`, à côté du CSV.

    Le waypoint « Morille grise n°3 » est le hotspot n°3 de la carte et du
    CSV ; sa position est la cellule la mieux notée de l'amas (D50), comme
    les colonnes lat/lon du CSV et le marqueur de la carte.
    `inside` (masque booléen aligné sur `hotspots`) écarte, comme la carte,
    les hotspots hors de la zone étudiée (D39) ; les numéros restent ceux
    du CSV.
    """
    gpx_path = output_dir / f"{species}_hotspots.gpx"
    common = str(config.get_species_params(species).get("common_name", species))
    keep = (
        np.ones(len(hotspots), dtype=bool) if inside is None
        else np.asarray(inside, dtype=bool)
    )
    ranked = [(rank, h) for rank, h in enumerate(hotspots, 1) if keep[rank - 1]]
    best = ranked[:max_waypoints]
    n_out = len(hotspots) - len(ranked)

    def tag(name: str) -> str:
        return f"{{{_GPX_NS}}}{name}"

    ET.register_namespace("", _GPX_NS)
    root = ET.Element(
        tag("gpx"), {"version": "1.1", "creator": f"Cartomorilles {VERSION}"},
    )
    meta = ET.SubElement(root, tag("metadata"))
    ET.SubElement(meta, tag("name")).text = (
        f"Cartomorilles — {common} : meilleurs hotspots "
        f"({len(best)} sur {len(ranked)})"
    )
    ET.SubElement(meta, tag("desc")).text = (
        "Triés par score moyen décroissant ; numéros identiques à la "
        "carte HTML et au CSV. Chaque point est la cellule la mieux notée "
        "de son hotspot, pas son centre."
        + (f" {n_out} hotspot(s) hors zone étudiée écarté(s), comme sur la "
           "carte." if n_out else "")
    )
    ET.SubElement(meta, tag("time")).text = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # Ordre des enfants imposé par le schéma GPX 1.1 : ele, name, desc, type
    for rank, h in best:
        e = hotspot_to_export_dict(h, rank=rank)
        wpt = ET.SubElement(
            root, tag("wpt"), {"lat": f"{e['lat']:.5f}", "lon": f"{e['lon']:.5f}"},
        )
        alt = h.get("altitude")
        if alt is not None and np.isfinite(alt):
            ET.SubElement(wpt, tag("ele")).text = f"{float(alt):.0f}"
        ET.SubElement(wpt, tag("name")).text = f"{common} n°{rank}"

        score = float(h.get("mean_score", 0.0))
        facts = [
            f"Score moyen {_fr_decimal(score, 2)} "
            f"({_score_class(score, thresholds)})",
            _area_text(float(h.get("size_m2", 0.0))),
        ]
        slope = h.get("mean_slope")
        if slope is not None and np.isfinite(slope):
            facts.append(f"pente {float(slope):.0f}°")
        tree = display_label(h.get("dominant_species"))
        if tree:
            facts.append(f"essence dominante : {tree}")
        geology = display_label(h.get("dominant_geology"))
        if geology:
            facts.append(f"géologie : {geology}")
        ET.SubElement(wpt, tag("desc")).text = " · ".join(facts)
        ET.SubElement(wpt, tag("type")).text = common

    ET.indent(root)
    ET.ElementTree(root).write(gpx_path, encoding="utf-8", xml_declaration=True)
    logger.debug("🧭 Hotspots GPX : %s (%d points)", gpx_path, len(best))
    return gpx_path


# ═══════════════════════════════════════════════════════════════
#  RAPPORT JSON
# ═══════════════════════════════════════════════════════════════


def _json_safe(obj: Any) -> Any:
    """Copie de `obj` où NaN et ±inf deviennent None (null en JSON), ainsi
    que les flottants numpy ; tuples et ensembles deviennent des listes."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_report(
    output_dir: Path,
    stats: dict[str, Any],
    hotspots: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
    duration: float,
    concordance: float | None,
    enrichment_stats: dict[str, Any] | None = None,
    species: str = "esculenta",
    n_evaluated: int | None = None,
) -> Path:
    """Sauvegarde un rapport JSON complet et reproductible.

    v2.6.0 : nom rapport templaté par espèce — `rapport_{species}.json`.
    `n_evaluated` : points de contrôle notés (dans la grille, avec score).
    """
    hotspots_export: list[dict[str, Any]] = [
        hotspot_to_export_dict(h) for h in hotspots[:30]
    ]
    sp_checkpoints = TERRAIN_CHECKPOINTS_BY_SPECIES.get(
        species, TERRAIN_CHECKPOINTS_ESCULENTA,
    )

    report: dict[str, Any] = {
        "tool": "Cartomorilles",
        "version": VERSION,
        "species": species,
        "generated_at": datetime.now().isoformat(),
        "duration_seconds": round(duration, 1),
        "zone": {
            "name": _ZONE_LABEL,
            "bbox_wgs84": dict(BBOX_WGS84),
            "bbox_lambert93": {k: round(v, 1) for k, v in BBOX.items()},
            "area_km2": round(_ZONE_DX_KM * _ZONE_DY_KM, 1),
        },
        "config": config_snapshot,
        "statistics": stats,
        "terrain_validation": {
            "concordance_rate": concordance,
            "n_checkpoints": len(sp_checkpoints),
            "n_evaluated": n_evaluated,
        },
        "enrichment": enrichment_stats,
        "hotspots": hotspots_export,
    }

    report_path = output_dir / f"rapport_{species}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        # NaN (essence inconnue, D46) → null : JSON strict lisible partout
        json.dump(
            _json_safe(report), f, indent=2, ensure_ascii=False,
            default=str, allow_nan=False,
        )

    logger.debug("📝 Rapport sauvegardé [%s] : %s", species, report_path)
    return report_path


# ═══════════════════════════════════════════════════════════════
#  MÉTÉO & RÉSUMÉ DE FIN DE RUN
# ═══════════════════════════════════════════════════════════════


def _weather_failure_reason(exc: BaseException) -> str:
    """Cause courte d'un échec météo, pour le résumé console."""
    if isinstance(exc, requests.Timeout):
        return "délai dépassé"
    if isinstance(exc, requests.ConnectionError):
        return "pas de connexion"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, RuntimeError):
        return str(exc)[:80]
    return type(exc).__name__


def _layer_label(gdf: Any, name: str) -> str | None:
    """Nom d'une couche chargée pour la ligne de fin d'étape 1 (None si
    absente) ; signale une couche de substitution synthétique."""
    if gdf is None or len(gdf) == 0:
        return None
    if (
        hasattr(gdf, "columns")
        and "source" in gdf.columns
        and bool((gdf["source"] == "synthetic").all())
    ):
        return f"{name} synthétique ⚠️"
    return name


# Mentions affichées à côté de certains fichiers du résumé
_OUTPUT_NOTES: dict[str, str] = {
    "morilles_probability.tif": "même fichier que esculenta_probability.tif",
    "cartomorilles_hd.html": "carte HD pour ordinateur",
    "cartomorilles.log": "journal détaillé",
}


def display_outputs(output_dir: Path, files: list[Path]) -> None:
    """Fichiers écrits par ce run — pas ceux qu'un run précédent a laissés
    dans le répertoire de sortie."""
    present = [f for f in dict.fromkeys(files) if f.is_file()]
    if not present:
        return
    stats = [f.stat() for f in present]
    sizes = [s.st_size for s in stats]
    # Un lien physique (alias morilles_probability.tif) ne compte qu'une fois
    total = sum({(s.st_dev, s.st_ino): s.st_size for s in stats}.values())
    width = max(len(f.name) for f in present)
    logger.info("")
    logger.info(
        "📁 Fichiers produits dans %s (%s) :", output_dir, _fr_size(total),
    )
    for f, size in zip(present, sizes):
        note = _OUTPUT_NOTES.get(f.name)
        logger.info(
            "   %-*s %9s%s", width, f.name, _fr_size(size),
            f"   {note}" if note else "",
        )


def _open_in_browser(uri: str) -> None:
    """--open : ouvre la carte dans le navigateur par défaut."""
    try:
        opened = webbrowser.open(uri)
    except webbrowser.Error as exc:
        logger.debug("webbrowser : %s", exc)
        opened = False
    if not opened:
        logger.warning(
            "Navigateur introuvable — ouvrez la carte à la main : %s", uri,
        )


# ═══════════════════════════════════════════════════════════════
#  SIGNAL HANDLER (Ctrl+C)
# ═══════════════════════════════════════════════════════════════


def _on_interrupt(signum: int, frame: FrameType | None) -> None:
    """Nettoyage propre sur interruption Ctrl+C."""
    logger.warning("\n⛔ Interruption (Ctrl+C)")
    for filepath in _cleanup_files:
        p = Path(filepath)
        if p.exists() and p.stat().st_size == 0:
            try:
                p.unlink()
                logger.warning(
                    "   Supprimé fichier vide/partiel : %s", p.name,
                )
            except OSError:
                pass
    sys.exit(130)


signal.signal(signal.SIGINT, _on_interrupt)


# ═══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════


def main(args: argparse.Namespace) -> int:
    """
    Pipeline principal Cartomorilles.

    Ordre (v2.6.0) :
      1. Chargement données
         a. DEM (Copernicus 30m)
         b. BD Forêt v2 → SpeciesEnricher (essences + TFV + IFN régional)
         c. Géologie BDCharm-50 (prioritaire) ou WFS/synthétique
         d. Hydrographie, urbain (WFS IGN)
         e. Landcover (analyse couleur OSM)
      2. Grille + scores :
         a. compute_terrain → score_altitude/slope/roughness/aspect/twi
         b. score_distance_water
         c. score_tree_species → enricher.enrich_grid_scores (niveaux B+C+D)
         d. score_geology
         e. apply_urban_mask + score_urban_proximity
         f. score_canopy_openness / score_ground_cover / score_disturbance
         g. apply_water_mask
         h. apply_landcover_mask
      3. Scoring multicritère — une passe par espèce : critères species-aware
         recalculés puis essences ré-enrichies (D47) ; météo J+0 à part (D45)
      4. Exports, puis résumé console (fichiers, carte, hotspots par espèce)

    Returns:
        0 = succès, 1 = erreur config, 2 = erreur données
    """
    t_start = time.time()
    output_dir = Path(args.output_dir)

    # ── Logging ────────────────────────────────────────────────
    log_path = setup_logging(output_dir, verbose=args.verbose)
    # Détail chemins/données : à l'écran en --dry-run, sinon dans le log
    detail = logging.INFO if args.dry_run else logging.DEBUG

    species_list: list[str] = list(getattr(args, "species", ["esculenta"]))
    species_names = ", ".join(
        str(config.get_species_params(sp)["common_name"]) for sp in species_list
    )

    logger.info("=" * 60)
    logger.info("🍄 CARTOMORILLES v%s — %s", VERSION, _ZONE_LABEL)
    logger.info("=" * 60)

    # ── Configuration : tables et paramètres par espèce cohérents (D49) ──
    if not config.validate_config():
        logger.error("❌ Configuration incohérente — détail dans %s", log_path)
        return 1

    # ── Purge cache (conditionnel) ────────────────────────────
    if args.purge_cache:
        _purge_cache()

    # ── Reproductibilité ──────────────────────────────────────
    seed: int = args.seed
    np.random.seed(seed)  # type: ignore[arg-type]
    logger.debug("🎲 Seed : %d", seed)

    # ── Résolution auto-scale ─────────────────────────────────
    bbox = BBOX
    emprise_km2 = (
        (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"]) / 1e6
    )
    if args.cell_size is not None:
        cell_size = args.cell_size
    else:
        cell_size = _auto_cell_size(emprise_km2)
    if cell_size != CELL_SIZE:
        config.CELL_SIZE = cell_size  # type: ignore[misc]
        logger.debug(
            "📐 Maille %.0fm (%s, emprise %.0f km²) au lieu du défaut "
            "config %.0fm",
            cell_size,
            "auto-scale" if args.cell_size is None else "--cell-size",
            emprise_km2, CELL_SIZE,
        )

    # ── Validation config ─────────────────────────────────────
    if not validate_weights():
        if not args.force:
            logger.error(
                "❌ Poids incohérents. Corrigez config.py ou --force.",
            )
            return 1
        logger.warning("--force : on continue malgré poids ≠ 1.0")

    # ── Estimation grille ─────────────────────────────────────
    est = estimate_grid_size(cell_size)
    logger.debug(
        "📐 Grille estimée : %d×%d = %s cellules  "
        "(%.1f km², ~%.0f Mo RAM)",
        est["nx"], est["ny"], f"{est['n_cells']:,}",
        est["area_km2"], est["mem_estimate_mb"],
    )
    logger.info(
        "   Maille %s m (%s) · %s cellules · %s",
        _fr_num(cell_size, 0 if float(cell_size).is_integer() else 1),
        "auto" if args.cell_size is None else "--cell-size",
        _fr_num(est["n_cells"]), species_names,
    )

    if est["n_cells"] > _MAX_CELLS and not args.force:
        logger.error(
            "❌ >%s cellules (%s). "
            "Essayez --cell-size %d ou --force.",
            f"{_MAX_CELLS:,}",
            f"{est['n_cells']:,}",
            int(cell_size * 2),
        )
        return 1

    if est["n_cells"] > _WARN_CELLS:
        logger.warning(
            "⚠️  %s cellules — mémoire de pointe ~%.0f Go, calcul long",
            f"{est['n_cells']:,}", est["mem_estimate_mb"] / 1000,
        )

    # ── Discovery WFS (mode spécial) ─────────────────────────
    if args.discover_wfs:
        loader = DataLoader()
        for url in (
            "https://data.geopf.fr/wfs/ows",
            "https://geoservices.brgm.fr/geologie",
        ):
            if hasattr(loader, "discover_wfs_layers"):
                found = loader.discover_wfs_layers(url)  # type: ignore[attr-defined]
                logger.info("")
                logger.info(
                    "🔍 %s — %d couches pertinentes%s :", url, len(found),
                    " (30 premières)" if len(found) > 30 else "",
                )
                for line in found[:30]:
                    logger.info(line)
            else:
                logger.warning(
                    "DataLoader n'a pas de méthode discover_wfs_layers",
                )
        return 0

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 1 — CHARGEMENT DES DONNÉES
    # ══════════════════════════════════════════════════════════

    logger.info("")
    logger.info("📂 [1/4] Chargement des données…")
    t0 = time.time()

    loader = DataLoader()

    # ── 1a. Résolution des chemins ────────────────────────────
    logger.log(detail, "   Résolution des chemins de données :")
    # DEM : pas de glob (une tuile unique ne couvre pas le bbox élargi) —
    # load_dem() merge les 6 tuiles Copernicus locales si filepath=None.
    dem_path = _resolve_data_path(
        args.dem, _DEFAULT_DEM, "DEM", None, detail,
    )
    bd_foret_path = _resolve_data_path(
        args.bd_foret, _DEFAULT_BD_FORET, "BD Forêt",
        "FORMATION_VEGETALE.shp", detail,
    )
    geology_path = _resolve_data_path(
        args.geology, _DEFAULT_GEOLOGY, "Géologie",
        "*S_FGEOL*2154.shp", detail,
    )
    regions_path = _resolve_data_path(
        args.regions, _DEFAULT_REGIONS, "Régions IFN",
        "rfifn250_l93.shp", detail,
    )

    # ── 1b. MNT ───────────────────────────────────────────────
    dem_data = loader.load_dem(filepath=dem_path)

    # ── 1c. Forêt — BD Forêt v2 prioritaire, fallback WFS ───
    enricher = SpeciesEnricher(
        bd_foret_path=bd_foret_path,
        regions_shp_path=regions_path,
        observations_path=args.obs,
    )

    forest_gdf = enricher.load_bd_foret()
    enricher_load_stats: dict[str, Any] = {}

    if forest_gdf is not None:
        n_known = int(
            (forest_gdf["essence_canonical"] != "unknown").sum(),
        )
        enricher_load_stats = {
            "source": "bd_foret_v2",
            "bd_foret_polygons": len(forest_gdf),
            "species_known_pct": round(
                n_known / max(len(forest_gdf), 1) * 100, 1,
            ),
        }
        logger.log(
            detail,
            "   🌳 BD Forêt v2 : %d polygones, "
            "%d essences connues (%.1f%%)",
            len(forest_gdf), n_known,
            enricher_load_stats["species_known_pct"],
        )
        forest_label = "BD Forêt v2"
    else:
        logger.warning(
            "   🌳 BD Forêt v2 indisponible → couche forêt WFS/fichier",
        )
        forest_gdf = loader.load_forest(args.forest)
        enricher_load_stats = {"source": "wfs_fallback"}
        forest_label = "forêt WFS/fichier"

    # ── 1d. Géologie — BDCharm-50 prioritaire ────────────────
    geology_source: str = "unknown"
    if geology_path is not None:
        geology_gdf = loader.load_geology(filepath=geology_path)
        if geology_gdf is not None and len(geology_gdf) > 0:
            geology_source = "bdcharm50"
            n_geo = len(geology_gdf)
            n_resolved = 0
            if (
                hasattr(geology_gdf, "columns")
                and "geology_canonical" in geology_gdf.columns
            ):
                n_resolved = int(
                    (geology_gdf["geology_canonical"] != "unknown").sum(),
                )
            logger.log(
                detail,
                "   🪨 Géologie BDCharm-50 : %d polygones, "
                "%d résolus (%.1f%%)",
                n_geo, n_resolved,
                n_resolved / max(n_geo, 1) * 100,
            )
        else:
            logger.warning(
                "   🪨 BDCharm-50 chargé mais vide → fallback",
            )
            geology_gdf = loader.load_geology()
            geology_source = "wfs_or_synthetic"
    else:
        geology_gdf = loader.load_geology()
        geology_source = (
            "wfs_brgm"
            if geology_gdf is not None
            and hasattr(geology_gdf, "columns")
            and "source" in geology_gdf.columns
            and (geology_gdf["source"] == "wfs_brgm").any()
            else "synthetic"
        )
    geology_label = {
        "bdcharm50": "géologie BDCharm-50",
        "wfs_brgm": "géologie WFS BRGM",
    }.get(geology_source, "géologie")

    # ── 1d'. Contacts géologiques linéaires (L_FGEOL) ────────
    geo_lines_gdf = loader.load_geology_lines()

    # ── 1e. Autres couches vectorielles ───────────────────────
    hydro_gdf = loader.load_hydro(args.hydro)
    # Surfaces en eau BD TOPO : rives des lacs et des rivières larges (D55)
    water_surfaces_gdf = loader.load_water_surfaces()
    # Pertes de couvert forestier (Hansen GFC) : coupes, chablis (D64)
    forest_loss = loader.load_forest_loss()

    urban_gdf = None
    if not args.no_urban:
        urban_gdf = loader.load_urban(args.urban)
    else:
        logger.log(detail, "   ℹ️  Masque urbain désactivé (--no-urban)")

    # ── 1f. Détection landcover par couleur OSM ───────────────
    landcover_data: dict[str, Any] | None = None
    area_km2 = est["area_km2"]

    if args.no_landcover:
        logger.log(detail, "   ℹ️  Landcover désactivé (--no-landcover)")
    elif not args.landcover and area_km2 > LANDCOVER_AUTO_MAX_KM2:
        logger.log(
            detail,
            "   ℹ️  Landcover auto-skip : %.0f km² > %.0f km² "
            "(utiliser --landcover pour forcer)",
            area_km2, LANDCOVER_AUTO_MAX_KM2,
        )
    else:
        try:
            from landcover_detector import LandcoverDetector

            with LandcoverDetector(zoom=None) as detector:
                landcover_data = detector.detect()
        except ImportError:
            logger.warning("Module landcover_detector introuvable.")
        except ConnectionError as e:
            logger.warning("Landcover — erreur réseau : %s", e)
        except Exception as e:
            logger.warning("Landcover échoué : %s", e)

    # ── Validation MNT (obligatoire) ──────────────────────────
    if dem_data is None or dem_data.get("data") is None:
        logger.error("❌ MNT indisponible. Impossible de continuer.")
        return 2

    alt: np.ndarray = dem_data["data"]

    # Traitement NoData
    nodata_mask: np.ndarray = (alt < -100) | (~np.isfinite(alt))
    n_nodata = int(np.sum(nodata_mask))
    if n_nodata > 0:
        pct_nodata = n_nodata / alt.size * 100
        logger.warning(
            "⚠️  MNT : %s pixels NoData (%.1f%%) → NaN",
            f"{n_nodata:,}", pct_nodata,
        )
        alt[nodata_mask] = np.nan
        if pct_nodata > 30 and not args.force:
            logger.error("❌ Trop de NoData (>30%%). Vérifiez le MNT.")
            return 2

    loaded = [
        "MNT", forest_label,
        _layer_label(geology_gdf, geology_label),
        _layer_label(hydro_gdf, "hydrographie"),
        _layer_label(water_surfaces_gdf, "surfaces en eau"),
        _layer_label(urban_gdf, "urbain"),
        "pertes de couvert" if forest_loss else None,
        "occupation du sol OSM" if landcover_data else None,
    ]
    logger.info(
        "   ✓ %s — %s",
        " · ".join(x for x in loaded if x), _fr_duration(time.time() - t0),
    )

    # Résumé
    summarize_data(
        dem_data, forest_gdf, geology_gdf,
        hydro_gdf, urban_gdf, landcover_data,
        enricher_load_stats if enricher_load_stats else None,
        geology_source=geology_source,
        level=detail,
    )

    # ── Dry run ───────────────────────────────────────────────
    if args.dry_run:
        logger.info("")
        logger.info(
            "🔍 --dry-run : données vérifiées en %s, "
            "arrêt avant le calcul du modèle.",
            _fr_duration(time.time() - t_start),
        )
        return 0

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 2 — CONSTRUCTION DE LA GRILLE & SCORES PAR CRITÈRE
    # ══════════════════════════════════════════════════════════

    logger.info("")
    logger.info(
        "📐 [2/4] Grille %s m et %d critères…",
        _fr_num(cell_size, 0 if float(cell_size).is_integer() else 1),
        len(WEIGHTS),
    )
    t0 = time.time()

    grid = GridBuilder()

    # ── 2a. Terrain de base (MNT → altitude, pente, aspect) ──
    # Cours d'eau et surfaces en eau gravés dans le MNT de routage du TWI
    grid.compute_terrain(
        dem_data, streams=[hydro_gdf, water_surfaces_gdf],
    )
    # Coupes, chablis récents : critère perturbation de chaque espèce (D64)
    grid.set_forest_loss(forest_loss)

    # ── 2b. Scores terrain ────────────────────────────────────
    # Tables de la 1re espèce modélisée, comme l'enrichissement (2c') :
    # les critères dépendant de l'espèce sont recalculés à l'étape 3 (D60)
    first_species = species_list[0]
    logger.debug("   ▸ Altitude, pente, rugosité, exposition...")
    grid.score_altitude(species=first_species)
    grid.score_slope(species=first_species)
    grid.score_terrain_roughness()
    grid.score_aspect(species=first_species)
    # ── TWI (fix #46 v2.3.5) ──
    grid.score_twi(species=first_species)

    # ── 2c. Scores écologiques (données vectorielles) ─────────
    logger.debug("   ▸ Distance eau...")
    grid.score_distance_water(hydro_gdf, water_surfaces_gdf)

    logger.debug("   ▸ Essences forestières...")
    grid.score_tree_species(forest_gdf, species=first_species)
    # Peuplements d'arbres hôtes : élimination « sans arbre hôte » (D57)
    grid.compute_host_forest(forest_gdf)
    # Plancher dist_water en forêt (D9), sur le masque forêt (D61)
    grid.apply_forest_water_floor()

    logger.debug("   ▸ Géologie (%s)...", geology_source)
    grid.score_geology(geology_gdf, species=first_species)

    # ── 2c'. Enrichissement essences (niveaux B+C+D) ─────────
    # Table de la 1re espèce modélisée ; rejoué par espèce à l'étape 3 (D47).
    # Ce passage sert aux planchers forêt du masque landcover
    # (_raw_tree_species) et aux statistiques d'enrichissement.
    logger.debug("   ▸ Enrichissement essences (IFN 1997 × altitude)...")
    enricher.enrich_grid_scores(
        grid, forest_gdf=forest_gdf, geology_gdf=geology_gdf,
        species=first_species,
    )
    enrich_stats = enricher.get_stats(grid)
    logger.debug(
        "   ▸ Essences : %d/%d connues (%.1f%%), "
        "%d inconnues restantes",
        enrich_stats.get("species_known", 0),
        enrich_stats.get("forest_cells", 0),
        enrich_stats.get("pct_known", 0.0),
        enrich_stats.get("species_unknown", 0),
    )

    # ── 2d. Masque urbain (AVANT micro-habitats) ─────────────
    logger.debug("   ▸ Application masque urbain...")
    urban_buffer: int = args.urban_buffer if not args.no_urban else 0
    grid.apply_urban_mask(urban_gdf, buffer_m=urban_buffer)
    # Zones bâties détectées par couleur : avant proximité et perturbation
    grid.merge_landcover_urban(landcover_data)

    # ── 2d'. Score proximité urbaine (EDT sur masque urbain) ──
    logger.debug("   ▸ Score proximité urbaine...")
    grid.score_urban_proximity()

    # ── 2e. Scores micro-habitat ──────────────────────────────
    logger.debug(
        "   ▸ Lisière forestière, contacts géologiques, ouverture canopée, "
        "couvert sol, perturbation...",
    )
    grid.score_forest_edge_distance()
    grid.score_geology_contact_distance(geo_lines_gdf)
    grid.score_canopy_openness()
    grid.score_ground_cover()
    grid.score_disturbance()

    # ── 2f. Masque plans d'eau (scores eau → 0) ──────────────
    logger.debug("   ▸ Application masque plans d'eau...")
    grid.apply_water_mask()

    # ── 2g. Modulation landcover (APRÈS tous les scores) ──────
    logger.debug("   ▸ Application masque landcover (×green_score)...")
    grid.apply_landcover_mask(landcover_data)

    n_cells = grid.scores["altitude"].size
    logger.info(
        "   ✓ %s cellules (%s km²) — %s",
        _fr_num(n_cells), _fr_num(round(n_cells * cell_size**2 / 1e6)),
        _fr_duration(time.time() - t0),
    )

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 3 — MODÈLE MULTICRITÈRE (multi-espèces v2.6.0)
    # ══════════════════════════════════════════════════════════

    logger.info("")
    logger.info("🧮 [3/4] Modèle multicritère — %s…", species_names)
    t0 = time.time()

    # ── Météo J+0 : information à part, jamais mêlée au score (D45) ──
    weather_days: list[Any] | None = None
    weather_today: dict[str, Any] | None = None
    if getattr(args, "no_weather", False):
        weather_status = "disabled"
        weather_summary = ["🌤️  Météo non consultée (--no-weather)"]
        logger.debug("🌤️  Météo non consultée (--no-weather)")
    else:
        days: list[Any] = []
        reason = "aucune prévision"
        try:
            days = WeatherChecker().evaluate()
        except Exception as exc:
            reason = _weather_failure_reason(exc)
            logger.debug("Météo indisponible : %r", exc)
        if days:
            weather_days = days
            today = days[0]
            weather_today = {
                "label": str(today.label), "score": round(float(today.score), 3),
            }
            weather_status = "available"
            weather_summary = [
                f"🌤️  Météo du jour : {today.label} "
                f"({_fr_num(today.score, 2)}) — à part du score, qui reste "
                "le potentiel du terrain (détail 7 jours sur la carte)",
            ]
            logger.info(
                "   🌤️  Météo du jour : %s (%s) — hors du score",
                today.label, _fr_num(today.score, 2),
            )
            logger.debug(WeatherChecker.format_report(days))
        else:
            weather_status = "unavailable"
            weather_summary = [f"🌤️  Météo indisponible ({reason})"]
            logger.info("   🌤️  Météo indisponible (%s)", reason)

    # ── Boucle scoring par espèce ──
    models: dict[str, MorilleScoring] = {}
    hotspots_by_species: dict[str, list[dict[str, Any]]] = {}
    hotspot_threshold_by_species: dict[str, float] = {}
    stats_by_species: dict[str, dict[str, Any]] = {}
    concordance_by_species: dict[str, TerrainConcordance | None] = {}

    for sp_idx, species in enumerate(species_list):
        sp_t0 = time.time()
        logger.debug("")
        logger.debug("─" * 70)
        sp_params = config.get_species_params(species)
        logger.debug(
            "PASS %d/%d — %s (%s)", sp_idx + 1, len(species_list),
            sp_params["common_name"], sp_params["scientific_name"],
        )
        logger.debug("─" * 70)

        # Recompute des 6 critères species-aware (grid.scores écrasé in-place)
        # 1ère itération esculenta : redondant avec score_* initiaux mais cache hit ~0s
        grid.recompute_species_scores(
            species, forest_gdf=forest_gdf, geology_gdf=geology_gdf,
        )
        # score_tree_species réécrit le critère depuis la table brute :
        # rejouer l'enrichissement IFN (niveaux B+C+C'+D, lande) avec la
        # table de l'espèce, sinon il n'atteint jamais le scoring (D47)
        enricher.enrich_grid_scores(
            grid, forest_gdf=forest_gdf, geology_gdf=geology_gdf,
            species=species,
        )
        # Ouverture de canopée et couvert du sol lisent l'essence et la
        # géologie de l'espèce (bonus « essence favorable », « calcaire ») :
        # recalculés après l'enrichissement de l'espèce (D60)
        grid.recompute_habitat_scores(species)

        sp_model = MorilleScoring(grid, species=species)
        sp_model.compute_weighted_score()
        sp_model.apply_eliminatory_factors()
        sp_model.apply_monotony_penalty()
        sp_model.apply_calcdry_penalty()
        sp_model.apply_spatial_smoothing(sigma=args.smoothing_sigma)
        # Classes par rang sur les cellules notées de la zone étudiée (D52)
        study_mask = _study_area_mask(grid, geology_source)
        sp_model.set_rank_thresholds(inside=study_mask)
        sp_model.classify_probability()

        # Hotspots : seuil de la classe Bon de l'espèce, sauf seuil imposé
        sp_threshold = (
            args.hotspot_threshold if args.hotspot_threshold is not None
            else sp_model.class_thresholds[_BON_INDEX - 1]
        )
        hotspot_threshold_by_species[species] = sp_threshold
        sp_hotspots = sp_model.get_hotspots(threshold=sp_threshold)
        hotspots_by_species[species] = sp_hotspots

        # ── Statistiques par espèce ──
        _final = sp_model.final_score
        assert _final is not None
        sp_stats = compute_statistics(
            _final, sp_threshold,
            class_thresholds=sp_model.class_thresholds, inside=study_mask,
        )
        sp_stats["n_hotspots"] = len(sp_hotspots)
        sp_stats["species"] = species
        stats_by_species[species] = sp_stats
        display_statistics(sp_stats)

        # ── Validation terrain par espèce ──
        concordance_by_species[species] = validate_against_terrain(sp_model)

        # Mémoire : critères de l'espèce et masques par facteur ne servent
        # plus qu'à l'export vecteur et aux couches de diagnostic (8 grilles
        # float32 et 9 masques par espèce : ~5 Go par espèce à 10 m, D62)
        if not (args.export_vector or args.debug_layers):
            sp_model.scores = {}
            sp_model.elimination_detail = {}

        models[species] = sp_model
        logger.info(
            "   ✓ %s : %s hotspots — %s",
            _species_label(species), _fr_num(len(sp_hotspots)),
            _fr_duration(time.time() - sp_t0),
        )

    logger.debug(
        "   ⏱️  Scoring TOTAL (%d espèces) : %.1fs",
        len(species_list), time.time() - t0,
    )

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 4 — EXPORTS
    # ══════════════════════════════════════════════════════════

    logger.info("")
    logger.info("🗺️  [4/4] Exports — carte, GeoTIFF, CSV, rapports…")
    t0 = time.time()

    # Fichiers écrits par CE run (le résumé ignore ceux d'un run précédent)
    produced: list[Path] = []

    # Récupérer les observations iNaturalist depuis l'enrichisseur
    _inat_obs = getattr(enricher, "inat_observations", None) or []

    # ── Visualizer multi-espèces (v2.6.0) ──
    # Accepte un dict de modèles + dict de hotspots. Le 1er species du dict
    # est visible par défaut, les autres sont accessibles via toggle JS.
    viz = MorilleVisualizer(
        models,
        hotspots=hotspots_by_species,
        hydro_gdf=hydro_gdf,
        urban_gdf=urban_gdf,
        inat_observations=_inat_obs,
        weather_days=weather_days,
        debug_layers=args.debug_layers,
        # BDCharm-50 départemental = Isère : hors de cette emprise, ni
        # géologie ni BD Forêt → pas de score affiché (D39)
        study_area=geology_gdf if geology_source == "bdcharm50" else None,
        default_hotspots_shown=args.map_hotspots,
        forest_gdf=forest_gdf,
    )
    # Carte Folium HTML — UN SEUL fichier multi-espèces avec toggle
    html_path = output_dir / "cartomorilles.html"
    _cleanup_files.append(str(html_path))
    produced.append(viz.create_folium_map(html_path))
    # Carte HD pour ordinateur (D62) : mêmes données, images jusqu'à 64 Mpx
    hd_path: Path | None = None
    if args.hd_map:
        hd_viz = MorilleVisualizer(
            models,
            hotspots=hotspots_by_species,
            hydro_gdf=hydro_gdf,
            urban_gdf=urban_gdf,
            inat_observations=_inat_obs,
            weather_days=weather_days,
            debug_layers=args.debug_layers,
            study_area=geology_gdf if geology_source == "bdcharm50" else None,
            default_hotspots_shown=args.map_hotspots,
            forest_gdf=forest_gdf,
            max_render_pixels=HD_MAX_PIXELS,
        )
        hd_path = output_dir / "cartomorilles_hd.html"
        _cleanup_files.append(str(hd_path))
        produced.append(hd_viz.create_folium_map(hd_path))
        del hd_viz

    # GeoTIFF raster — UN par espèce
    for sp in species_list:
        try:
            tif_path = output_dir / f"{sp}_probability.tif"
            _cleanup_files.append(str(tif_path))
            produced.append(viz.export_geotiff(tif_path, species=sp))
        except (IOError, OSError) as e:
            logger.warning("GeoTIFF [%s] I/O échoué : %s", sp, e)
        except Exception as e:
            logger.warning("GeoTIFF [%s] échoué : %s", sp, e)

    # Alias compat morilles_probability.tif (esculenta de CE run uniquement)
    src = output_dir / "esculenta_probability.tif"
    if src in produced:
        try:
            import os
            import shutil
            dst = output_dir / "morilles_probability.tif"
            dst.unlink(missing_ok=True)
            # Lien physique (même fichier, pas de 2e copie de ~100–450 Mo) ;
            # copie si le système de fichiers ne le permet pas
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            _cleanup_files.append(str(dst))
            produced.append(dst)
        except Exception as e:
            logger.debug("Alias morilles_probability.tif non créé : %s", e)

    # GPKG vecteur (optionnel) — UN par espèce
    if args.export_vector:
        for sp in species_list:
            try:
                gpkg_path = output_dir / f"{sp}_grid.gpkg"
                _cleanup_files.append(str(gpkg_path))
                produced.append(
                    viz.export_gpkg_grid(gpkg_path, threshold=0.3, species=sp),
                )
            except (IOError, OSError) as e:
                logger.warning("GPKG [%s] I/O échoué : %s", sp, e)
            except Exception as e:
                logger.warning("GPKG [%s] échoué : %s", sp, e)

    # ══════════════════════════════════════════════════════════
    #  RAPPORT & CONFIG SNAPSHOT
    # ══════════════════════════════════════════════════════════

    config_snapshot: dict[str, Any] = {
        "cell_size": cell_size,
        "seed": seed,
        "smoothing_sigma": args.smoothing_sigma,
        "hotspot_threshold": args.hotspot_threshold,
        "urban_buffer_m": urban_buffer if not args.no_urban else None,
        "no_urban": args.no_urban,
        "no_landcover": args.no_landcover,
        "weather": weather_status,
        "weather_today": weather_today,
        "weights": dict(WEIGHTS),
        # Tables d'essences et de géologie, optimums d'altitude et de pente :
        # ceux de l'espèce, sous species_params (ici ils étaient ceux de la
        # morille grise dans les trois rapports — D60)
        "geology_source": geology_source,
        "dem_source": getattr(grid, "dem_source", None),
        # Âge des coupes compté depuis l'année de référence : le critère
        # perturbation en dépend (D64)
        "forest_loss": (
            {"source": f"Hansen {forest_loss.get('version')}",
             "reference_year": config.forest_loss_reference_year()}
            if forest_loss else None
        ),
        "bd_foret_path": bd_foret_path,
        "geology_path": geology_path,
        "regions_path": regions_path,
        "species_modeled": species_list,
    }

    full_enrich_stats: dict[str, Any] = {**enricher_load_stats, **enrich_stats}

    duration = time.time() - t_start

    # ── Rapport JSON par espèce + CSV hotspots par espèce ──
    for sp in species_list:
        sp_conc = concordance_by_species[sp]
        # Snapshot enrichi avec params espèce
        sp_snapshot = dict(config_snapshot)
        sp_snapshot["species"] = sp
        # Seuil réellement appliqué (classe Bon de l'espèce si non imposé)
        sp_snapshot["hotspot_threshold"] = round(
            hotspot_threshold_by_species[sp], 4,
        )
        sp_snapshot["class_thresholds"] = [
            round(t, 4) for t in models[sp].class_thresholds
        ]
        sp_snapshot["species_params"] = {
            k: (list(v) if isinstance(v, tuple)
                else dict(v) if hasattr(v, "items")
                else sorted(v) if isinstance(v, frozenset)
                else v)
            for k, v in config.SPECIES_PARAMS[sp].items()
        }
        produced.append(save_report(
            output_dir, stats_by_species[sp], hotspots_by_species[sp],
            sp_snapshot, duration, sp_conc.rate if sp_conc else None,
            enrichment_stats=full_enrich_stats,
            species=sp,
            n_evaluated=sp_conc.n_points if sp_conc else None,
        ))
        produced.append(
            save_hotspots_csv(output_dir, hotspots_by_species[sp], species=sp),
        )
        try:
            produced.append(save_hotspots_gpx(
                output_dir, hotspots_by_species[sp], species=sp,
                max_waypoints=args.map_hotspots,
                inside=viz.inside_study_area(hotspots_by_species[sp]),
                thresholds=models[sp].class_thresholds,
            ))
        except OSError as e:
            logger.warning("GPX [%s] échoué : %s", sp, e)

    logger.info("   ✓ %s", _fr_duration(time.time() - t0))

    # ══════════════════════════════════════════════════════════
    #  RÉSUMÉ FINAL
    # ══════════════════════════════════════════════════════════

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Terminé en %s", _fr_duration(time.time() - t_start))
    logger.info("=" * 60)
    for line in weather_summary:
        logger.info(line)

    # Top hotspots + concordance terrain, pour chaque espèce
    for sp in species_list:
        display_species_summary(
            sp, hotspots_by_species[sp], concordance_by_species[sp],
            threshold=hotspot_threshold_by_species[sp],
            n_show=args.console_hotspots,
            inside=viz.inside_study_area(hotspots_by_species[sp]),
            thresholds=models[sp].class_thresholds,
        )

    display_outputs(output_dir, [*produced, log_path])

    map_uri = html_path.resolve().as_uri()
    logger.info("")
    if hd_path is not None:
        logger.info(
            "🗺️  Carte HD (ordinateur) : %s", hd_path.resolve().as_uri(),
        )
    if args.open:
        logger.info("🗺️  Carte : %s", map_uri)
        _open_in_browser(map_uri)
    else:
        logger.info("🗺️  Carte : %s  (--open pour l'ouvrir)", map_uri)

    hunted = _enumerate_fr([
        "aux " + str(config.get_species_params(sp).get("common_name_plural", sp))
        for sp in species_list
    ])
    logger.info("")
    logger.info("🧺 Bonne chasse %s !", hunted)

    _cleanup_files.clear()

    return 0


# ═══════════════════════════════════════════════════════════════
#  CLI — ARGUMENTS
# ═══════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser d'arguments CLI."""
    area_km2 = _ZONE_DX_KM * _ZONE_DY_KM
    auto_cell = _auto_cell_size(area_km2)
    auto_cells_m = estimate_grid_size(auto_cell)["n_cells"] / 1e6
    parser = argparse.ArgumentParser(
        prog="cartomorilles",
        description=(
            "🍄 Cartomorilles — cartes de probabilité de présence de la\n"
            "morille (Morchella esculenta) et du cèpe (Boletus edulis) en\n"
            "Isère : carte HTML interactive, GeoTIFF, CSV des hotspots et\n"
            "rapport JSON par espèce."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zone : {_ZONE_LABEL} ({_fr_num(area_km2)} km²)
Maille : automatique selon l'emprise — {auto_cell:.0f} m ici
         (~{_fr_num(auto_cells_m, 1)} M cellules) ; --cell-size pour la forcer
Sorties (--output-dir) : cartomorilles.html (toutes espèces),
         {{espèce}}_probability.tif, {{espèce}}_hotspots.csv,
         rapport_{{espèce}}.json, cartomorilles.log (journal détaillé)
Console : étapes [1/4]…[4/4] puis résumé ; --verbose pour tout le détail

Exemples d'utilisation :

  # Les deux espèces, maille auto, météo du jour
  python main.py

  # Idem, puis ouvrir la carte dans le navigateur
  python main.py --open

  # Morille seule, 10 hotspots listés en fin de run
  python main.py --species esculenta --console-hotspots 10

  # Avec MNT local (GeoTIFF)
  python main.py --dem data/mnt_local.tif

  # Mode rapide (maille 25m) avec logs détaillés
  python main.py --cell-size 25 --verbose

  # Vérification données seulement (pas de calcul)
  python main.py --dry-run

  # Sans masques, avec export vectoriel
  python main.py --no-urban --no-landcover --export-vector

  # Données personnalisées
  python main.py --bd-foret data/mon_foret.shp \\
                 --geology data/ma_geologie.shp \\
                 --obs observations.json

  # Purger le cache avant exécution
  python main.py --purge-cache

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """,
    )

    # ── Données d'entrée ──
    data_group = parser.add_argument_group("📂 Données d'entrée")
    data_group.add_argument(
        "--dem", type=str, default=None,
        help="Chemin MNT raster (.tif). Défaut : merge Copernicus N44-45/E004-006.",
    )
    data_group.add_argument(
        "--forest", type=str, default=None,
        help="Chemin couche forêt (.shp/.gpkg). Défaut : WFS IGN.",
    )
    data_group.add_argument(
        "--geology", type=str, default=None,
        help=(
            "Chemin couche géologie (.shp/.gpkg). "
            "Défaut : geologie_38/GEO050K...S_FGEOL."
        ),
    )
    data_group.add_argument(
        "--hydro", type=str, default=None,
        help="Chemin couche hydrographie (.shp/.gpkg). Défaut : WFS IGN.",
    )
    data_group.add_argument(
        "--urban", type=str, default=None,
        help="Chemin couche zones urbaines (.shp/.gpkg). Défaut : WFS.",
    )

    # ── Enrichissement essences ──
    enrich_group = parser.add_argument_group("🌳 Enrichissement essences")
    enrich_group.add_argument(
        "--bd-foret", type=str, default=None,
        help="Chemin BD Forêt v2 SHP (défaut: FORMATION_VEGETALE.shp).",
    )
    enrich_group.add_argument(
        "--regions", type=str, default=None,
        help="Chemin régions forestières IFN SHP (défaut: rfifn250_l93.shp).",
    )
    enrich_group.add_argument(
        "--obs", type=str, default=None,
        help="Chemin observations terrain JSON (optionnel).",
    )

    # ── Masques & filtres ──
    mask_group = parser.add_argument_group("🚫 Masques & filtres")

    urban_excl = mask_group.add_mutually_exclusive_group()
    urban_excl.add_argument(
        "--no-urban", action="store_true",
        help="Désactiver totalement le masque urbain.",
    )
    urban_excl.add_argument(
        "--urban-buffer", type=int, default=10, metavar="M",
        help="Buffer autour des zones urbaines en mètres (défaut: 10).",
    )

    mask_group.add_argument(
        "--no-landcover", action="store_true",
        help="Désactiver la détection couleur OSM.",
    )
    mask_group.add_argument(
        "--landcover", action="store_true",
        help=(
            "Forcer la détection couleur OSM même sur grandes emprises "
            "(auto-skip au-delà de %.0f km²)."
            % LANDCOVER_AUTO_MAX_KM2
        ),
    )

    # ── Paramètres modèle ──
    model_group = parser.add_argument_group("🧮 Paramètres du modèle")
    model_group.add_argument(
        "--cell-size", type=float, default=None, metavar="M",
        help=(
            "Taille de cellule en mètres (défaut : automatique selon "
            f"l'emprise — {auto_cell:.0f} m pour la zone)."
        ),
    )
    model_group.add_argument(
        "--hotspot-threshold", type=float, default=None, metavar="T",
        help=(
            "Seuil de score des hotspots (défaut : seuil de la classe Bon de "
            "chaque espèce, soit ses 10 %% de cellules les mieux notées de "
            "la zone étudiée)."
        ),
    )
    model_group.add_argument(
        "--smoothing-sigma", type=float, default=0.5, metavar="S",
        help="Sigma du lissage gaussien spatial (défaut: 0.5 — préserve mieux les pics).",
    )
    model_group.add_argument(
        "--seed", type=int, default=42,
        help="Seed pour reproductibilité des critères synthétiques.",
    )
    model_group.add_argument(
        "--species", nargs="+",
        default=list(SUPPORTED_SPECIES),
        choices=list(SUPPORTED_SPECIES),
        metavar="SPECIES",
        help=(
            "Espèces à modéliser (défaut : toutes, avec bascule d'espèce "
            "sur la carte). Choix : %s." % ", ".join(
                f"{sp} ({config.get_species_params(sp)['common_name'].lower()})"
                for sp in SUPPORTED_SPECIES
            )
        ),
    )

    # ── Sorties ──
    output_group = parser.add_argument_group("📁 Sorties")
    output_group.add_argument(
        "--output-dir", type=str, default="output", metavar="DIR",
        help="Répertoire de sortie (défaut: output/).",
    )
    output_group.add_argument(
        "--export-vector", action="store_true",
        help="Exporter la grille en GPKG vecteur (volumineux).",
    )
    output_group.add_argument(
        "--hd-map", action="store_true",
        help="Écrit aussi cartomorilles_hd.html : images jusqu'à 64 Mpx "
             "(~15 m sur l'Isère), pour ordinateur ; la carte par défaut "
             "reste sous 16,8 Mpx pour s'ouvrir sur téléphone (D62)",
    )
    output_group.add_argument(
        "--open", action="store_true",
        help="Ouvrir la carte dans le navigateur à la fin du run.",
    )
    output_group.add_argument(
        "--console-hotspots", type=int, default=_CONSOLE_HOTSPOTS_DEFAULT,
        metavar="N",
        help=(
            "Hotspots listés par espèce dans le résumé de fin de run "
            "(défaut : %(default)s ; 0 = aucun). N'affecte ni la détection, "
            "ni la carte, ni le CSV (qui contient tous les hotspots)."
        ),
    )
    # Ancien nom de --console-hotspots, gardé pour les scripts existants
    output_group.add_argument(
        "--max-hotspots", type=int, dest="console_hotspots",
        default=argparse.SUPPRESS, help=argparse.SUPPRESS,
    )
    output_group.add_argument(
        "--map-hotspots", type=int, default=_MAP_HOTSPOTS_DEFAULT,
        metavar="N",
        help=(
            "Hotspots numérotés affichés à l'ouverture de la carte, par "
            "espèce (défaut : %(default)s ; réglable ensuite dans le "
            "panneau)."
        ),
    )
    output_group.add_argument(
        "--debug-layers", action="store_true",
        help=(
            "Ajouter à la carte les couches de diagnostic (zones exclues, "
            "TWI brut) — fichier HTML plus lourd."
        ),
    )

    # ── Modes spéciaux ──
    special_group = parser.add_argument_group("🔧 Modes spéciaux")
    special_group.add_argument(
        "--dry-run", action="store_true",
        help="Charger et vérifier les données, sans lancer le modèle.",
    )
    special_group.add_argument(
        "--discover-wfs", action="store_true",
        help="Lister les couches WFS disponibles et quitter.",
    )
    special_group.add_argument(
        "--purge-cache", action="store_true",
        help="Purger le cache tuiles/données avant exécution.",
    )
    special_group.add_argument(
        "--force", action="store_true",
        help="Ignorer les avertissements (poids, taille grille...).",
    )
    special_group.add_argument(
        "-v", "--verbose", action="store_true",
        help=(
            "Afficher tout le détail (DEBUG, horodaté) à l'écran — par "
            "défaut, seulement les étapes, les avertissements et le résumé ; "
            "le détail va toujours dans cartomorilles.log."
        ),
    )
    special_group.add_argument(
        "--no-weather", action="store_true", dest="no_weather",
        help=(
            "Ne pas consulter la météo Open-Meteo (utile hors ligne) : le "
            "résumé et le panneau de la carte n'affichent alors pas les "
            "conditions du jour. Le score n'en dépend jamais."
        ),
    )

    return parser


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args()

    if args.console_hotspots < 0:
        parser.error("--console-hotspots doit être ≥ 0")
    if args.map_hotspots < 1:
        parser.error("--map-hotspots doit être ≥ 1")

    if args.no_urban and hasattr(args, "urban_buffer"):
        args.urban_buffer = 0

    exit_code = main(args)
    sys.exit(exit_code)