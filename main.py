#!/usr/bin/env python3
"""
🍄 CARTOMORILLES — Modèle de probabilité de présence de morilles
   Zone : Isère (38) — emprise limitée par données disponibles
   DEM Copernicus N45_E005 ∩ BD Forêt v2 ∩ Géologie BDCharm-50

   Analyse multicritère spatialisée avec maillage configurable (défaut 10×10m).
   Sources : IGN BD Forêt v2, BD Topo, BRGM BDCharm-50, MNT, OSM landcover.
   Enrichissement essences : BD Forêt TFV + IFN 1997 régional × altitude.

Usage :
    python main.py                              # Mode auto
    python main.py --dem mnt.tif                # Avec MNT local
    python main.py --dry-run                    # Vérif données seulement
    python main.py --cell-size 25 --verbose     # Test rapide, logs détaillés

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
import json
import logging
import signal
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np

import config
from config import (
    ALTITUDE_OPTIMAL,
    ALTITUDE_RANGE,
    BBOX,
    BBOX_WGS84,
    CELL_SIZE,
    CONFIG_VERSION,
    GEOLOGY_SCORES,
    N_SCORING_GRIDS,
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_MEDIUM,
    SLOPE_MAX,
    SLOPE_OPTIMAL,
    SUPPORTED_SPECIES,
    TREE_SCORES,
    WEIGHTS,
    LANDCOVER_AUTO_MAX_KM2,
    CELL_SIZE_AUTO_THRESHOLDS,
    CELL_SIZE_AUTO_FALLBACK,
)
from data_loader import DataLoader
from grid_builder import GridBuilder
from scoring import MorilleScoring
from species_enricher import SpeciesEnricher
from utils import TO_L93, TO_WGS84, compute_array_stats, hotspot_to_export_dict
from visualize import MorilleVisualizer
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
_WARN_CELLS = 50_000_000  # 50M — avertit sur temps de calcul TWI D8

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
TERRAIN_CHECKPOINTS_EDULIS: tuple[dict[str, Any], ...] = (
    # ── Contrôles positifs (cèpes attendus) ──────────────────────
    {
        "name": "Vercors hêtraie-sapinière 1100m",
        "lat": 45.0660, "lon": 5.4630, "expected": 0.70,
        "obs": "hêtre + sapin altitude optimale, sol calcaire",
    },
    {
        "name": "Chartreuse sapinière 1100m",
        "lat": 45.3500, "lon": 5.8000, "expected": 0.65,
        "obs": "sapinière calcaire montagne moyenne",
    },
    {
        "name": "Belledonne épicéa-pin granite 1500m",
        "lat": 45.1700, "lon": 5.9500, "expected": 0.65,
        "obs": "pessière granite — INVERSE morille (granite OK cèpe)",
    },
    {
        "name": "Trièves chênaie-hêtraie 800m",
        "lat": 44.8500, "lon": 5.6500, "expected": 0.55,
        "obs": "chêne + hêtre versant montagne basse",
        "tolerance": 0.50,
    },
    {
        "name": "Châtaigneraie Bas-Dauphiné 500m",
        "lat": 45.4500, "lon": 5.4000, "expected": 0.60,
        "obs": "châtaignier acide — bon habitat cèpe",
        "tolerance": 0.50,
    },
    # ── Contrôles négatifs (cèpes peu probables) ────────────────
    {
        "name": "Plaine Bièvre peuplier 250m",
        "lat": 45.4500, "lon": 5.2500, "expected": 0.15,
        "obs": "peuplier saturé, alt trop basse",
        "tolerance": 0.50,
    },
    {
        "name": "Centre-ville Grenoble",
        "lat": 45.1885, "lon": 5.7245, "expected": 0.00,
        "obs": "urbain dense — éliminé",
    },
    {
        "name": "Lautaret alpin 2050m",
        "lat": 45.0350, "lon": 6.4050, "expected": 0.00,
        "obs": "altitude > 2200m éliminatoire (limite supérieure)",
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


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure le logging : console (INFO ou DEBUG) + fichier (DEBUG)."""
    root = logging.getLogger("cartomorilles")
    root.setLevel(logging.DEBUG)

    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        output_dir / "cartomorilles.log", mode="w", encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ═══════════════════════════════════════════════════════════════
#  RÉSOLUTION INTELLIGENTE DES CHEMINS DE DONNÉES
# ═══════════════════════════════════════════════════════════════


def _resolve_data_path(
    cli_arg: str | None,
    default: str | None,
    label: str,
    glob_pattern: str | None,
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
    """
    # 1. Argument CLI explicite
    if cli_arg is not None:
        p = Path(cli_arg)
        if p.exists():
            logger.info("   ✅ %s (CLI) : %s", label, p)
            return str(p)
        logger.warning("   ❌ %s (CLI) introuvable : %s", label, p)

    # 2. Chemin par défaut
    if default is not None:
        dp = Path(default)
        if dp.exists():
            logger.info("   ✅ %s : %s", label, dp.name)
            return str(dp)
    else:
        dp = None

    # 3. Glob récursif dans data/ (couvre sous-dossiers type geologie_38/)
    if glob_pattern is not None:
        candidates = sorted(_DATA_DIR.rglob(glob_pattern))
        if candidates:
            chosen = candidates[0]
            logger.info("   ✅ %s (glob) : %s", label, chosen.name)
            return str(chosen)

    if default is None and glob_pattern is None:
        logger.info("   ℹ️ %s : aucun défaut local (géré par loader)", label)
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
    """Vérifie la somme des poids et affiche le détail."""
    total = sum(WEIGHTS.values())
    ok = abs(total - 1.0) < 0.01

    logger.info("Pondérations du modèle :")
    for key, val in sorted(WEIGHTS.items(), key=lambda x: -x[1]):
        logger.info("  %-25s %.2f", key, val)
    logger.info("  %s ────", "─" * 25)
    logger.info(
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
    mem_mb = n_cells * 4 * N_SCORING_GRIDS / 1024 / 1024  # float32

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
) -> None:
    """Affiche un tableau récapitulatif des données chargées."""
    logger.info("")
    logger.info("─" * 55)
    logger.info("📋 Résumé des données chargées :")

    alt = dem_data["data"]
    logger.info(
        "   MNT       : %d×%dpx, alt %.0f–%.0fm",
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
            logger.info("   %-10s: %5d entités%s", name, len(gdf), src)
        else:
            logger.info("   %-10s: ❌ (absente ou vide)", name)

    if geology_source:
        logger.info("   Géo source: %s", geology_source)

    lc = "✅ analyse couleur OSM" if landcover_data else "❌"
    logger.info("   Landcover : %s", lc)

    if enricher_stats:
        logger.info(
            "   Essences  : %s → %d polygones",
            enricher_stats.get("source", "?"),
            enricher_stats.get("bd_foret_polygons", 0),
        )

    logger.info("─" * 55)


def compute_statistics(
    final_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calcule les statistiques du score final.

    `classes` = répartition dans les 6 classes de la carte
    (config.PROBABILITY_*), en % des cellules avec données. Les clés
    n_high / n_medium / n_low (seuils 0.7 / 0.4) restent dans le rapport
    JSON pour comparer avec les calibrations passées (D35).
    """
    base = compute_array_stats(final_score, decimals=4)
    finite = np.isfinite(final_score)
    n_valid = int(np.count_nonzero(finite))
    counts = np.bincount(
        np.digitize(final_score[finite], config.PROBABILITY_THRESHOLDS),
        minlength=len(config.PROBABILITY_LABELS),
    )
    bounds = [0.0, *config.PROBABILITY_THRESHOLDS, 1.0]
    classes = [
        {
            "label": label,
            "min": lo,
            "max": hi,
            "n": int(n),
            "pct": round(float(n) / max(n_valid, 1) * 100, 2),
        }
        for label, lo, hi, n in zip(
            config.PROBABILITY_LABELS, bounds[:-1], bounds[1:], counts,
        )
    ]
    return {
        "n_cells": int(final_score.size),
        "n_valid": n_valid,
        "score_min": base["min"],
        "score_max": base["max"],
        "score_mean": base["mean"],
        "score_median": base["median"],
        "score_std": base["std"],
        "classes": classes,
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


def _score_class(score: float) -> str:
    """Classe de probabilité (libellé) d'un score."""
    idx = int(np.digitize(score, config.PROBABILITY_THRESHOLDS))
    return config.PROBABILITY_LABELS[idx]


def display_statistics(stats: dict[str, Any]) -> None:
    """Affiche les statistiques en console/log : mêmes 6 classes que la
    légende de la carte, de la meilleure à la plus faible."""
    logger.info("")
    logger.info("📊 Statistiques du score final :")
    logger.info(
        "   Min=%.3f  Max=%.3f  Moy=%.3f  Méd=%.3f  σ=%.3f",
        stats["score_min"], stats["score_max"],
        stats["score_mean"], stats["score_median"],
        stats["score_std"],
    )
    for i, c in reversed(list(enumerate(stats["classes"]))):
        label = "Nul ou exclu" if i == 0 else c["label"]
        logger.info(
            "   %-13s %-10s : %12s cellules (%5.1f %%) %s",
            label, _class_range(c["min"], c["max"]), f"{c['n']:,}",
            c["pct"], "█" * int(c["pct"] / 2),
        )
    n_nodata = stats["n_cells"] - stats["n_valid"]
    if n_nodata > 0:
        logger.info(
            "   %-24s : %12s cellules", "Sans données", f"{n_nodata:,}",
        )


# ═══════════════════════════════════════════════════════════════
#  VALIDATION TERRAIN
# ═══════════════════════════════════════════════════════════════


def validate_against_terrain(
    model: MorilleScoring,
    checkpoints: tuple[dict[str, Any], ...] | None = None,
) -> float | None:
    """
    Compare le score modèle aux observations de terrain.
    Retourne un taux de concordance (0–1) ou None si impossible.

    v2.6.0 : checkpoints species-aware — fallback sur les checkpoints de
    l'espèce du modèle (TERRAIN_CHECKPOINTS_BY_SPECIES[model.species]).
    """
    if checkpoints is None:
        checkpoints = TERRAIN_CHECKPOINTS_BY_SPECIES.get(
            model.species, TERRAIN_CHECKPOINTS_ESCULENTA,
        )
    try:
        grid = model.grid

        _scores = model.final_score
        if _scores is None:
            logger.debug("Score final non calculé, validation impossible.")
            return None
        scores: np.ndarray = _scores

        concordance_list: list[float] = []
        logger.info("")
        logger.info("📍 Validation croisée terrain [%s] :", model.species)

        for pt in checkpoints:
            x, y = TO_L93.transform(pt["lon"], pt["lat"])

            if not hasattr(grid, "x_coords") or not hasattr(grid, "y_coords"):
                logger.debug(
                    "   Grille sans coordonnées, validation impossible.",
                )
                return None

            ix = int(np.argmin(np.abs(grid.x_coords - x)))
            iy = int(np.argmin(np.abs(grid.y_coords - y)))

            if 0 <= iy < scores.shape[0] and 0 <= ix < scores.shape[1]:
                predicted = float(scores[iy, ix])
                expected: float = float(pt["expected"])
                diff = abs(predicted - expected)
                tolerance: float = float(pt.get("tolerance", 0.25))
                ok = diff < tolerance

                symbol = "✅" if ok else "❌"
                concordance_list.append(1.0 if ok else 0.0)

                logger.info(
                    "   %s %-40s attendu=%.2f modèle=%.2f Δ=%.2f",
                    symbol, pt["name"], expected, predicted, diff,
                )
            else:
                logger.debug("   ⚠️  %s hors grille", pt["name"])

        if concordance_list:
            rate = sum(concordance_list) / len(concordance_list)
            logger.info(
                "   Concordance : %.0f%% (%d/%d points)",
                rate * 100,
                sum(1 for c in concordance_list if c),
                len(concordance_list),
            )
            return rate

    except Exception as e:
        logger.debug("Validation terrain impossible : %s", e)

    return None


# ═══════════════════════════════════════════════════════════════
#  AFFICHAGE HOTSPOTS
# ═══════════════════════════════════════════════════════════════


def display_hotspots(
    hotspots: list[dict[str, Any]], max_display: int = 8,
) -> None:
    """Affiche les meilleurs hotspots avec coordonnées GPS et lien Maps."""
    if not hotspots:
        logger.warning("Aucun hotspot détecté au-dessus du seuil.")
        return

    n_show = min(max_display, len(hotspots))
    logger.info(
        "\n🎯 Top %d hotspots (sur %d détectés) :", n_show, len(hotspots),
    )

    # Numéro = rang (comme les marqueurs de la carte et la colonne rank du CSV)
    for rank, h in enumerate(hotspots[:n_show], 1):
        lon, lat = TO_WGS84.transform(h["x_l93"], h["y_l93"])

        parts: list[str] = [
            f"Score {h['mean_score']:.2f} ({_score_class(h['mean_score'])})",
            f"{h['size_m2']:.0f}m²",
        ]
        if h.get("altitude") is not None:
            parts.append(f"alt={h['altitude']:.0f}m")
        if h.get("mean_slope") is not None:
            parts.append(f"pente={h['mean_slope']:.0f}°")
        tree = config.tree_label(h.get("dominant_species"))
        if tree:
            parts.append(f"🌳 {tree}")
        geology = config.geology_label(h.get("dominant_geology"))
        if geology:
            parts.append(f"🪨 {geology}")
        if h.get("compactness") is not None:
            parts.append(f"compact={h['compactness']:.2f}")

        logger.info("   n°%d — %s", rank, ", ".join(parts))
        logger.info("         📍 %.5f°N, %.5f°E", lat, lon)
        logger.info(
            "         🔗 https://www.google.com/maps?q=%.5f,%.5f", lat, lon,
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

    logger.info("📍 Hotspots CSV : %s (%d lignes)", csv_path, len(rows))
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
) -> Path:
    """Exporte les meilleurs hotspots en GPX 1.1 (un waypoint chacun) pour
    les applis GPS — `{species}_hotspots.gpx`, à côté du CSV.

    Le waypoint « Morille n°3 » est le hotspot n°3 de la carte et du CSV.
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
        "carte HTML et au CSV."
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
            f"Score moyen {_fr_decimal(score, 2)} ({_score_class(score)})",
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
    logger.info("🧭 Hotspots GPX : %s (%d points)", gpx_path, len(best))
    return gpx_path


# ═══════════════════════════════════════════════════════════════
#  RAPPORT JSON
# ═══════════════════════════════════════════════════════════════


def save_report(
    output_dir: Path,
    stats: dict[str, Any],
    hotspots: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
    duration: float,
    concordance: float | None,
    enrichment_stats: dict[str, Any] | None = None,
    species: str = "esculenta",
) -> Path:
    """Sauvegarde un rapport JSON complet et reproductible.

    v2.6.0 : nom rapport templaté par espèce — `rapport_{species}.json`.
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
        },
        "enrichment": enrichment_stats,
        "hotspots": hotspots_export,
    }

    report_path = output_dir / f"rapport_{species}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("📝 Rapport sauvegardé [%s] : %s", species, report_path)
    return report_path


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

    Ordre (v2.4.0) :
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
      3. Scoring multicritère
      4. Exports

    Returns:
        0 = succès, 1 = erreur config, 2 = erreur données
    """
    t_start = time.time()
    output_dir = Path(args.output_dir)

    # ── Logging ────────────────────────────────────────────────
    setup_logging(output_dir, verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("🍄 CARTOMORILLES v%s", VERSION)
    logger.info("   Zone : %s", _ZONE_LABEL)
    logger.info("=" * 60)

    # ── Purge cache (conditionnel) ────────────────────────────
    if args.purge_cache:
        _purge_cache()

    # ── Reproductibilité ──────────────────────────────────────
    seed: int = args.seed
    np.random.seed(seed)  # type: ignore[arg-type]
    logger.info("🎲 Seed : %d", seed)

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
        import config as _cfg
        _cfg.CELL_SIZE = cell_size  # type: ignore[misc]
        logger.info(
            "📐 Auto-scale : %.0f km² → résolution %.0fm "
            "(défaut %.0fm, --cell-size pour forcer)",
            emprise_km2, cell_size, CELL_SIZE,
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
    logger.info(
        "📐 Grille estimée : %d×%d = %s cellules  "
        "(%.1f km², ~%.0f Mo RAM)",
        est["nx"], est["ny"], f"{est['n_cells']:,}",
        est["area_km2"], est["mem_estimate_mb"],
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
            "⚠️  %s cellules — TWI D8 peut prendre >5 min",
            f"{est['n_cells']:,}",
        )

    # ── Discovery WFS (mode spécial) ─────────────────────────
    if args.discover_wfs:
        loader = DataLoader()
        for url in (
            "https://data.geopf.fr/wfs/ows",
            "https://geoservices.brgm.fr/geologie",
        ):
            if hasattr(loader, "discover_wfs_layers"):
                loader.discover_wfs_layers(url)  # type: ignore[attr-defined]
            else:
                logger.warning(
                    "DataLoader n'a pas de méthode discover_wfs_layers",
                )
        return 0

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 1 — CHARGEMENT DES DONNÉES
    # ══════════════════════════════════════════════════════════

    logger.info("\n📂 [1/4] Chargement des données...")
    t0 = time.time()

    loader = DataLoader()

    # ── 1a. Résolution des chemins ────────────────────────────
    logger.info("   Résolution des chemins de données :")
    # DEM : pas de glob (une tuile unique ne couvre pas le bbox élargi) —
    # load_dem() merge les 6 tuiles Copernicus locales si filepath=None.
    dem_path = _resolve_data_path(
        args.dem, _DEFAULT_DEM, "DEM", None,
    )
    bd_foret_path = _resolve_data_path(
        args.bd_foret, _DEFAULT_BD_FORET, "BD Forêt",
        "FORMATION_VEGETALE.shp",
    )
    geology_path = _resolve_data_path(
        args.geology, _DEFAULT_GEOLOGY, "Géologie",
        "*S_FGEOL*2154.shp",
    )
    regions_path = _resolve_data_path(
        args.regions, _DEFAULT_REGIONS, "Régions IFN",
        "rfifn250_l93.shp",
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
        logger.info(
            "   🌳 BD Forêt v2 : %d polygones, "
            "%d essences connues (%.1f%%)",
            len(forest_gdf), n_known,
            enricher_load_stats["species_known_pct"],
        )
    else:
        logger.info("   🌳 BD Forêt v2 indisponible → fallback WFS/fichier")
        forest_gdf = loader.load_forest(args.forest)
        enricher_load_stats = {"source": "wfs_fallback"}

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
            logger.info(
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

    # ── 1d'. Contacts géologiques linéaires (L_FGEOL) ────────
    geo_lines_gdf = loader.load_geology_lines()

    # ── 1e. Autres couches vectorielles ───────────────────────
    hydro_gdf = loader.load_hydro(args.hydro)

    urban_gdf = None
    if not args.no_urban:
        urban_gdf = loader.load_urban(args.urban)
    else:
        logger.info("ℹ️  Masque urbain désactivé (--no-urban)")

    # ── 1f. Détection landcover par couleur OSM ───────────────
    landcover_data: dict[str, Any] | None = None
    area_km2 = est["area_km2"]

    if args.no_landcover:
        logger.info("ℹ️  Landcover désactivé (--no-landcover)")
    elif not args.landcover and area_km2 > LANDCOVER_AUTO_MAX_KM2:
        logger.info(
            "ℹ️  Landcover auto-skip : %.0f km² > %.0f km² "
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

    logger.info("   ⏱️  Chargement : %.1fs", time.time() - t0)

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

    # Résumé
    summarize_data(
        dem_data, forest_gdf, geology_gdf,
        hydro_gdf, urban_gdf, landcover_data,
        enricher_load_stats if enricher_load_stats else None,
        geology_source=geology_source,
    )

    # ── Dry run ───────────────────────────────────────────────
    if args.dry_run:
        logger.info(
            "\n🔍 Mode --dry-run : données vérifiées, "
            "arrêt avant calcul du modèle.",
        )
        logger.info("   (durée : %.1fs)", time.time() - t_start)
        return 0

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 2 — CONSTRUCTION DE LA GRILLE & SCORES PAR CRITÈRE
    # ══════════════════════════════════════════════════════════

    logger.info(
        "\n📐 [2/4] Construction grille %.0f×%.0fm...",
        cell_size, cell_size,
    )
    t0 = time.time()

    grid = GridBuilder()

    # ── 2a. Terrain de base (MNT → altitude, pente, aspect) ──
    grid.compute_terrain(dem_data)

    # ── 2b. Scores terrain ────────────────────────────────────
    logger.info("   ▸ Altitude, pente, rugosité, exposition...")
    grid.score_altitude()
    grid.score_slope()
    grid.score_terrain_roughness()
    grid.score_aspect()
    # ── TWI (fix #46 v2.3.5) ──
    grid.score_twi()

    # ── 2c. Scores écologiques (données vectorielles) ─────────
    logger.info("   ▸ Distance eau...")
    grid.score_distance_water(hydro_gdf)

    logger.info("   ▸ Essences forestières...")
    grid.score_tree_species(forest_gdf)

    logger.info("   ▸ Géologie (%s)...", geology_source)
    grid.score_geology(geology_gdf)

    # ── 2c'. Enrichissement essences (niveaux B+C+D) ─────────
    logger.info("   ▸ Enrichissement essences (IFN 1997 × altitude)...")
    enricher.enrich_grid_scores(
        grid, forest_gdf=forest_gdf, geology_gdf=geology_gdf,
    )
    enrich_stats = enricher.get_stats(grid)
    logger.info(
        "   ▸ Essences : %d/%d connues (%.1f%%), "
        "%d inconnues restantes",
        enrich_stats.get("species_known", 0),
        enrich_stats.get("forest_cells", 0),
        enrich_stats.get("pct_known", 0.0),
        enrich_stats.get("species_unknown", 0),
    )

    # ── 2d. Masque urbain (AVANT micro-habitats) ─────────────
    logger.info("   ▸ Application masque urbain...")
    urban_buffer: int = args.urban_buffer if not args.no_urban else 0
    grid.apply_urban_mask(urban_gdf, buffer_m=urban_buffer)

    # ── 2d'. Score proximité urbaine (EDT sur masque urbain) ──
    logger.info("   ▸ Score proximité urbaine...")
    grid.score_urban_proximity()

    # ── 2e. Scores micro-habitat ──────────────────────────────
    logger.info("   ▸ Lisière forestière, contacts géologiques, ouverture canopée, couvert sol, perturbation...")
    grid.score_forest_edge_distance()
    grid.score_geology_contact_distance(geo_lines_gdf)
    grid.score_canopy_openness()
    grid.score_ground_cover()
    grid.score_disturbance()

    # ── 2f. Masque plans d'eau (scores eau → 0) ──────────────
    logger.info("   ▸ Application masque plans d'eau...")
    grid.apply_water_mask()

    # ── 2g. Modulation landcover (APRÈS tous les scores) ──────
    logger.info("   ▸ Application masque landcover (×green_score)...")
    grid.apply_landcover_mask(landcover_data)

    n_cells = grid.scores["altitude"].size
    logger.info(
        "   ⏱️  Grille : %.1fs — %s cellules (%.1f km²)",
        time.time() - t0,
        f"{n_cells:,}",
        n_cells * cell_size**2 / 1e6,
    )

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 3 — MODÈLE MULTICRITÈRE (multi-espèces v2.6.0)
    # ══════════════════════════════════════════════════════════

    species_list: list[str] = list(getattr(args, "species", ["esculenta"]))
    logger.info(
        "\n🧮 [3/4] Calcul du modèle multicritère [espèces: %s]...",
        ", ".join(species_list),
    )
    t0 = time.time()

    # ── Niveau météo (calculé 1× pour toutes espèces) ──
    weather_days: list[Any] | None = None
    weather_factor: float = 1.0
    if not getattr(args, "no_weather", False):
        try:
            _checker = WeatherChecker()
            _days = _checker.evaluate()
            if _days:
                weather_days = _days
                _today = _days[0]
                weather_factor = float(np.float32(0.4 + 0.6 * _today.score))
                logger.info(
                    "🌤️  Météo J+0 : %s  (score=%.2f → ×%.2f)",
                    _today.label, _today.score, weather_factor,
                )
                logger.info(WeatherChecker.format_report(_days))
        except Exception as _exc:
            logger.debug("Météo indisponible, skip : %s", _exc)

    # ── Boucle scoring par espèce ──
    models: dict[str, MorilleScoring] = {}
    hotspots_by_species: dict[str, list[dict[str, Any]]] = {}
    stats_by_species: dict[str, dict[str, Any]] = {}
    concordance_by_species: dict[str, float | None] = {}

    for sp_idx, species in enumerate(species_list):
        sp_t0 = time.time()
        logger.info("")
        logger.info("─" * 70)
        sp_params = config.get_species_params(species)
        logger.info(
            "PASS %d/%d — %s (%s)", sp_idx + 1, len(species_list),
            sp_params["common_name"], sp_params["scientific_name"],
        )
        logger.info("─" * 70)

        # Recompute des 6 critères species-aware (grid.scores écrasé in-place)
        # 1ère itération esculenta : redondant avec score_* initiaux mais cache hit ~0s
        grid.recompute_species_scores(
            species, forest_gdf=forest_gdf, geology_gdf=geology_gdf,
        )

        sp_model = MorilleScoring(grid, species=species)
        sp_model.compute_weighted_score()
        sp_model.apply_eliminatory_factors()
        sp_model.apply_monotony_penalty()
        sp_model.apply_calcdry_penalty()
        sp_model.apply_spatial_smoothing(sigma=args.smoothing_sigma)

        # ── Modulation météo (multiplicative, post-smoothing) ──
        if weather_factor != 1.0:
            assert sp_model.final_score is not None
            sp_model.final_score = sp_model.final_score * np.float32(weather_factor)

        sp_model.classify_probability()

        sp_hotspots = sp_model.get_hotspots(threshold=args.hotspot_threshold)
        hotspots_by_species[species] = sp_hotspots

        # ── Statistiques par espèce ──
        _final = sp_model.final_score
        assert _final is not None
        sp_stats = compute_statistics(_final, args.hotspot_threshold)
        sp_stats["n_hotspots"] = len(sp_hotspots)
        sp_stats["species"] = species
        stats_by_species[species] = sp_stats

        logger.info("   📊 Stats [%s]:", species)
        display_statistics(sp_stats)

        # ── Validation terrain par espèce ──
        sp_conc = validate_against_terrain(sp_model)
        concordance_by_species[species] = sp_conc

        models[species] = sp_model
        logger.info("   ⏱️  [%s] : %.1fs", species, time.time() - sp_t0)

    logger.info("")
    logger.info("   ⏱️  Scoring TOTAL (%d espèces) : %.1fs",
                len(species_list), time.time() - t0)

    # ── Variables compat single-species (premier species = défaut affichage) ──
    primary_species: str = species_list[0]
    model: MorilleScoring = models[primary_species]
    hotspots: list[dict[str, Any]] = hotspots_by_species[primary_species]
    stats: dict[str, Any] = stats_by_species[primary_species]
    concordance: float | None = concordance_by_species[primary_species]
    final = model.final_score
    assert final is not None

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 4 — EXPORTS
    # ══════════════════════════════════════════════════════════

    logger.info("\n🗺️  [4/4] Génération des exports...")
    t0 = time.time()

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
    )
    # Carte Folium HTML — UN SEUL fichier multi-espèces avec toggle
    html_path = str(output_dir / "cartomorilles.html")
    _cleanup_files.append(html_path)
    viz.create_folium_map(html_path)

    # GeoTIFF raster — UN par espèce
    for sp in species_list:
        try:
            tif_path = str(output_dir / f"{sp}_probability.tif")
            _cleanup_files.append(tif_path)
            viz.export_geotiff(tif_path, species=sp)
        except (IOError, OSError) as e:
            logger.warning("GeoTIFF [%s] I/O échoué : %s", sp, e)
        except Exception as e:
            logger.warning("GeoTIFF [%s] échoué : %s", sp, e)

    # Alias compat morilles_probability.tif (esculenta)
    if "esculenta" in species_list:
        try:
            import shutil
            src = output_dir / "esculenta_probability.tif"
            dst = output_dir / "morilles_probability.tif"
            if src.exists():
                shutil.copy2(src, dst)
                _cleanup_files.append(str(dst))
        except Exception as e:
            logger.debug("Alias morilles_probability.tif non créé : %s", e)

    # GPKG vecteur (optionnel) — UN par espèce
    if args.export_vector:
        for sp in species_list:
            try:
                gpkg_path = str(output_dir / f"{sp}_grid.gpkg")
                _cleanup_files.append(gpkg_path)
                viz.export_gpkg_grid(gpkg_path, threshold=0.3, species=sp)
            except (IOError, OSError) as e:
                logger.warning("GPKG [%s] I/O échoué : %s", sp, e)
            except Exception as e:
                logger.warning("GPKG [%s] échoué : %s", sp, e)

    logger.info("   ⏱️  Export : %.1fs", time.time() - t0)

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
        "weights": dict(WEIGHTS),
        "tree_scores": dict(TREE_SCORES),
        "geology_scores": dict(GEOLOGY_SCORES),
        "altitude_optimal": list(ALTITUDE_OPTIMAL),
        "altitude_range": list(ALTITUDE_RANGE),
        "slope_optimal": list(SLOPE_OPTIMAL),
        "slope_max": SLOPE_MAX,
        "geology_source": geology_source,
        "bd_foret_path": bd_foret_path,
        "geology_path": geology_path,
        "regions_path": regions_path,
        "species_modeled": species_list,
    }

    full_enrich_stats: dict[str, Any] = {**enricher_load_stats, **enrich_stats}

    duration = time.time() - t_start

    # ── Rapport JSON par espèce + CSV hotspots par espèce ──
    for sp in species_list:
        sp_stats = stats_by_species[sp]
        sp_hotspots = hotspots_by_species[sp]
        sp_conc = concordance_by_species[sp]
        # Snapshot enrichi avec params espèce
        sp_snapshot = dict(config_snapshot)
        sp_snapshot["species"] = sp
        sp_snapshot["species_params"] = {
            k: (list(v) if isinstance(v, tuple)
                else dict(v) if hasattr(v, "items")
                else sorted(v) if isinstance(v, frozenset)
                else v)
            for k, v in config.SPECIES_PARAMS[sp].items()
        }
        save_report(
            output_dir, sp_stats, sp_hotspots,
            sp_snapshot, duration, sp_conc,
            enrichment_stats=full_enrich_stats,
            species=sp,
        )
        save_hotspots_csv(output_dir, sp_hotspots, species=sp)
        try:
            save_hotspots_gpx(
                output_dir, sp_hotspots, species=sp,
                inside=viz.inside_study_area(sp_hotspots),
            )
        except OSError as e:
            logger.warning("GPX [%s] échoué : %s", sp, e)

    # ══════════════════════════════════════════════════════════
    #  RÉSUMÉ FINAL
    # ══════════════════════════════════════════════════════════

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ CARTOMORILLES v%s — Terminé en %.1fs", VERSION, duration)
    logger.info("=" * 60)

    # Fichiers générés
    logger.info("\n📁 Fichiers générés :")
    total_size = 0.0
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info("   📄 %-45s (%.1f Mo)", f.name, size_mb)
    logger.info("   %s ──────", "─" * 45)
    logger.info("   Total : %.1f Mo", total_size)

    # Hotspots
    display_hotspots(hotspots, max_display=args.max_hotspots)

    # Concordance terrain
    if concordance is not None:
        if concordance >= 0.8:
            logger.info(
                "\n✅ Bonne concordance terrain : %.0f%%",
                concordance * 100,
            )
        elif concordance >= 0.5:
            logger.info(
                "\n⚠️  Concordance terrain moyenne : %.0f%% "
                "— recalibrer les poids ?",
                concordance * 100,
            )
        else:
            logger.warning(
                "\n❌ Concordance terrain faible : %.0f%% "
                "— modèle à revoir !",
                concordance * 100,
            )

    hunted = " et aux ".join(
        str(config.get_species_params(sp).get("common_name_plural", sp))
        for sp in species_list
    )
    logger.info("\n🧺 Bonne chasse aux %s !\n", hunted)

    _cleanup_files.clear()

    return 0


# ═══════════════════════════════════════════════════════════════
#  CLI — ARGUMENTS
# ═══════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser d'arguments CLI."""
    parser = argparse.ArgumentParser(
        prog="cartomorilles",
        description=(
            "🍄 Cartomorilles — Modèle de probabilité "
            "de présence de morilles"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zone : {_ZONE_LABEL} | Maille : {CELL_SIZE:.0f}m

Exemples d'utilisation :

  # Mode auto (BD Forêt v2 + BDCharm-50 + IFN enrichissement)
  python main.py

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
        "--cell-size", type=float, default=None,
        help="Taille de cellule en mètres (auto-scale si omis).",
    )
    model_group.add_argument(
        "--hotspot-threshold", type=float, default=0.60, metavar="T",
        help="Seuil de score pour les hotspots (défaut: 0.60).",
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
            "Espèces à modéliser (défaut: toutes — esculenta + edulis pour "
            "dual-render avec toggle UI). Choix : %s." % ", ".join(SUPPORTED_SPECIES)
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
        "--max-hotspots", type=int, default=8, metavar="N",
        help="Nombre maximum de hotspots à afficher (défaut: 8).",
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
        "--verbose", action="store_true",
        help="Activer les logs de niveau DEBUG.",
    )
    special_group.add_argument(
        "--no-weather", action="store_true", dest="no_weather",
        help="Désactiver la modulation météo (mode hors-ligne).",
    )

    return parser


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.no_urban and hasattr(args, "urban_buffer"):
        args.urban_buffer = 0

    exit_code = main(args)
    sys.exit(exit_code)