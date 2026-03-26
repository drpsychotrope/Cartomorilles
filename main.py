#!/usr/bin/env python3
"""
🍄 CARTOMORILLES — Modèle de probabilité de présence de morilles
   Zone : Grenoble 20×20km (Isère, 38)

   Analyse multicritère spatialisée avec maillage configurable (défaut 10×10m).
   Sources : IGN BD Forêt v2, BD Topo, BRGM BDCharm-50, MNT, OSM landcover.
   Enrichissement essences : BD Forêt TFV + IFN 1997 régional × altitude.

Usage :
    python main.py                              # Mode auto
    python main.py --dem mnt.tif                # Avec MNT local
    python main.py --dry-run                    # Vérif données seulement
    python main.py --cell-size 10 --verbose     # Test rapide, logs détaillés

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
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
from pyproj import Transformer

from config import (
    BBOX,
    BBOX_WGS84,
    CELL_SIZE,
    WEIGHTS,
    TREE_SCORES,
    GEOLOGY_SCORES,
    ALTITUDE_OPTIMAL,
    ALTITUDE_RANGE,
    SLOPE_OPTIMAL,
    SLOPE_MAX,
)
from data_loader import DataLoader
from grid_builder import GridBuilder
from scoring import MorilleScoring
from visualize import MorilleVisualizer
from species_enricher import SpeciesEnricher

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES MODULE
# ═══════════════════════════════════════════════════════════════

VERSION = "2.3.5"

# Transformer réutilisable (L93 → WGS84, always_xy pour lon/lat)
_TO_WGS84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

# ── Chemins par défaut des données locales ─────────────────────
_DEFAULT_BD_FORET = "data/FORMATION_VEGETALE.shp"
_DEFAULT_REGIONS = "data/rfifn250_l93.shp"
_DEFAULT_GEOLOGY = "data/geologie_38/GEO050K_HARM_038_S_FGEOL_2154.shp"
_DEFAULT_DEM = "grenoble_bdalti25.tif"

# Points de contrôle terrain (prospection réelle mai 2025)
# v2.3.1 : châtaignier corrigé — plus éliminatoire
TERRAIN_CHECKPOINTS: tuple[dict[str, Any], ...] = (

        # ── Contrôles positifs (morilles attendues) ──────────────────
    {"name": "Champy – châtaigneraie",
     "lat": 45.24308, "lon": 5.69736, "expected": 0.55,
     "obs": "châtaignier favorable (M. elata), sol frais versant"},

    {"name": "Terrasse plan d'eau + conifères",
     "lat": 45.24087, "lon": 5.69484, "expected": 0.50,
     "obs": "meilleur spot trouvé, M. elata possible"},

    # ── Contrôles négatifs locaux (même zone, micro-habitat) ────
    {"name": "Berge Vence – trop humide",
     "lat": 45.24588, "lon": 5.69744, "expected": 0.05,
     "obs": "lierre dense, scolopendres — LIMITATION micro-habitat",
     "tolerance": 0.50},                      # ← tolérance élargie
    {"name": "Ravine au-dessus Vence",
     "lat": 45.24651, "lon": 5.70052, "expected": 0.05,
     "obs": "encaissé humide — LIMITATION micro-habitat",
     "tolerance": 0.50},
    {"name": "Mi-pente Néron chênaie+buis",
     "lat": 45.24418, "lon": 5.69886, "expected": 0.10,
     "obs": "trop sec thermophile — LIMITATION espèces indicatrices",
     "tolerance": 0.50},
    {"name": "Robiniers + hêtres secteur Champy",
     "lat": 45.24212, "lon": 5.69681, "expected": 0.15,
     "obs": "sol perturbé — LIMITATION sans détection invasives",
     "tolerance": 0.45},

    # ── Contrôles négatifs forts (vrai négatif attendu=0) ───────
    {"name": "Centre-ville Grenoble",
     "lat": 45.1885, "lon": 5.7245, "expected": 0.00,
     "obs": "urbain dense — doit être éliminé"},
    {"name": "Belledonne granite 1200m",
     "lat": 45.14893, "lon": 5.88069, "expected": 0.00,
     "obs": "granite éliminatoire — substrat acide"},
    {"name": "Sommet Néron 1299m",
     "lat": 45.23732, "lon": 5.70991, "expected": 0.00,
     "obs": "altitude > 1400m ou roche nue"},
    {"name": "Isère lit majeur",
     "lat": 45.20003, "lon": 5.74186, "expected": 0.00,
     "obs": "plan d'eau — masque eau"},

    # ── Contrôle positif éloigné (diversité géo) ────────────────
    {"name": "Vouillants forêt calcaire 350m",
     "lat": 45.18824, "lon": 5.66543, "expected": 0.70,
     "obs": "forêt calcaire optimale, altitude idéale"},
)

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
    cli_value: str | None,
    default: str,
    label: str,
    glob_pattern: str | None = None,
) -> str | None:
    """
    Résout un chemin de donnée avec cascade :
      1. Valeur CLI explicite (si fournie et existe)
      2. Chemin par défaut (si existe)
      3. Recherche récursive par glob dans data/ (optionnel)
      4. None

    Logge le résultat pour traçabilité.
    """
    # 1. CLI explicite
    if cli_value is not None:
        p = Path(cli_value)
        if p.exists():
            logger.info("   %s : %s (CLI)", label, p)
            return str(p)
        logger.warning("   %s : %s introuvable (CLI)", label, cli_value)

    # 2. Défaut
    p = Path(default)
    if p.exists():
        logger.info("   %s : %s (défaut)", label, p)
        return str(p)

    # 3. Glob récursif dans data/
    if glob_pattern:
        found = list(Path("data").rglob(glob_pattern))
        if found:
            logger.info("   %s : %s (découvert)", label, found[0])
            return str(found[0])

    logger.info("   %s : non trouvé → fallback WFS/synthétique", label)
    return None


# ═══════════════════════════════════════════════════════════════
#  VALIDATION & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════


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
    mem_mb = n_cells * 12 * 8 / 1024 / 1024

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
    """Calcule les statistiques du score final."""
    return {
        "n_cells": int(final_score.size),
        "score_min": round(float(np.nanmin(final_score)), 4),
        "score_max": round(float(np.nanmax(final_score)), 4),
        "score_mean": round(float(np.nanmean(final_score)), 4),
        "score_median": round(float(np.nanmedian(final_score)), 4),
        "score_std": round(float(np.nanstd(final_score)), 4),
        "n_high": int(np.sum(final_score >= 0.7)),
        "n_medium": int(np.sum((final_score >= 0.4) & (final_score < 0.7))),
        "n_low": int(np.sum(final_score < 0.4)),
        "pct_high": round(
            float(np.sum(final_score >= 0.7)) / final_score.size * 100, 2,
        ),
        "pct_medium": round(
            float(np.sum((final_score >= 0.4) & (final_score < 0.7)))
            / final_score.size * 100, 2,
        ),
        "hotspot_threshold": threshold,
    }


def display_statistics(stats: dict[str, Any]) -> None:
    """Affiche les statistiques en console/log."""
    logger.info("")
    logger.info("📊 Statistiques du score final :")
    logger.info(
        "   Min=%.3f  Max=%.3f  Moy=%.3f  Méd=%.3f  σ=%.3f",
        stats["score_min"], stats["score_max"],
        stats["score_mean"], stats["score_median"],
        stats["score_std"],
    )
    logger.info(
        "   🟢 Élevée  (≥0.7)     : %8s cellules (%.1f%%)",
        f"{stats['n_high']:,}", stats["pct_high"],
    )
    logger.info(
        "   🟡 Moyenne (0.4–0.7)  : %8s cellules (%.1f%%)",
        f"{stats['n_medium']:,}", stats["pct_medium"],
    )
    logger.info(
        "   ⚪ Faible  (<0.4)     : %8s cellules",
        f"{stats['n_low']:,}",
    )


# ═══════════════════════════════════════════════════════════════
#  VALIDATION TERRAIN
# ═══════════════════════════════════════════════════════════════


def validate_against_terrain(model: MorilleScoring) -> float | None:
    """
    Compare le score modèle aux observations de terrain.
    Retourne un taux de concordance (0–1) ou None si impossible.
    """
    try:
        grid = model.grid

        _scores = model.final_score
        if _scores is None:
            logger.debug("Score final non calculé, validation impossible.")
            return None
        scores: np.ndarray = _scores

        concordance_list: list[float] = []
        logger.info("")
        logger.info("📍 Validation croisée terrain :")

        for pt in TERRAIN_CHECKPOINTS:
            x, y = _TO_L93.transform(pt["lon"], pt["lat"])

            if not hasattr(grid, "x_coords") or not hasattr(grid, "y_coords"):
                logger.debug("   Grille sans coordonnées, validation impossible.")
                return None

            ix = int(np.argmin(np.abs(grid.x_coords - x)))
            iy = int(np.argmin(np.abs(grid.y_coords - y)))

            if 0 <= iy < scores.shape[0] and 0 <= ix < scores.shape[1]:
                predicted = float(scores[iy, ix])
                expected: float = float(pt["expected"])
                diff = abs(predicted - expected)
                ok = diff < 0.25

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

    for h in hotspots[:n_show]:
        lon, lat = _TO_WGS84.transform(h["x_l93"], h["y_l93"])

        parts: list[str] = [
            f"Score {h['mean_score']:.2f}",
            f"{h['size_m2']:.0f}m²",
        ]
        if h.get("altitude") is not None:
            parts.append(f"alt={h['altitude']:.0f}m")
        if h.get("mean_slope") is not None:
            parts.append(f"pente={h['mean_slope']:.0f}°")
        if h.get("dominant_species"):
            parts.append(f"🌳 {h['dominant_species']}")
        if h.get("dominant_geology"):
            parts.append(f"🪨 {h['dominant_geology']}")
        if h.get("compactness") is not None:
            parts.append(f"compact={h['compactness']:.2f}")

        logger.info("   #%d — %s", h["id"], ", ".join(parts))
        logger.info("         📍 %.5f°N, %.5f°E", lat, lon)
        logger.info(
            "         🔗 https://www.google.com/maps?q=%.5f,%.5f", lat, lon,
        )


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
) -> Path:
    """Sauvegarde un rapport JSON complet et reproductible."""
    hotspots_export: list[dict[str, Any]] = []
    for h in hotspots[:30]:
        lon, lat = _TO_WGS84.transform(h["x_l93"], h["y_l93"])
        entry: dict[str, Any] = {
            **h,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "google_maps": f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}",
        }
        hotspots_export.append(entry)

    report: dict[str, Any] = {
        "tool": "Cartomorilles",
        "version": VERSION,
        "generated_at": datetime.now().isoformat(),
        "duration_seconds": round(duration, 1),
        "zone": {
            "name": "Grenoble 20×20km",
            "bbox_wgs84": dict(BBOX_WGS84),
            "bbox_lambert93": {k: round(v, 1) for k, v in BBOX.items()},
        },
        "config": config_snapshot,
        "statistics": stats,
        "terrain_validation": {
            "concordance_rate": concordance,
            "n_checkpoints": len(TERRAIN_CHECKPOINTS),
        },
        "enrichment": enrichment_stats,
        "hotspots": hotspots_export,
    }

    report_path = output_dir / "rapport_cartomorilles.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info("📝 Rapport sauvegardé : %s", report_path)
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
                logger.warning("   Supprimé fichier vide/partiel : %s", p.name)
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

    Ordre (v2.3.2) :
      1. Chargement données
         a. DEM (BD ALTI 25m)
         b. BD Forêt v2 → SpeciesEnricher (essences + TFV + IFN régional)
         c. Géologie BDCharm-50 (prioritaire) ou WFS/synthétique
         d. Hydrographie, urbain (WFS IGN)
         e. Landcover (analyse couleur OSM)
      2. Grille + scores :
         a. compute_terrain → score_altitude/slope/roughness/aspect
         b. score_distance_water
         c. score_tree_species → enricher.enrich_grid_scores (niveaux B+C+D)
         d. score_geology
         e. apply_urban_mask
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
    logger.info("   Zone : Grenoble 20×20km (Isère)")
    logger.info("=" * 60)

    # ── Purge cache (conditionnel) ────────────────────────────
    if args.purge_cache:
        _purge_cache()

    # ── Reproductibilité ──────────────────────────────────────
    seed: int = args.seed
    np.random.seed(seed)  # type: ignore[arg-type]
    logger.info("🎲 Seed : %d", seed)

    # ── Cell size override ────────────────────────────────────
    cell_size: float = args.cell_size if args.cell_size else CELL_SIZE
    if cell_size != CELL_SIZE:
        import config as _cfg

        _cfg.CELL_SIZE = cell_size  # type: ignore[attr-defined]
        logger.info(
            "📐 Cell size surchargé : %.0fm (défaut: %.0fm)",
            cell_size, CELL_SIZE,
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
        "(%.2f km², ~%.0f Mo RAM)",
        est["nx"], est["ny"], f"{est['n_cells']:,}",
        est["area_km2"], est["mem_estimate_mb"],
    )

    if est["n_cells"] > 5_000_000 and not args.force:
        logger.error(
            "❌ >5M cellules (%s). "
            "Essayez --cell-size %d ou --force.",
            f"{est['n_cells']:,}", int(cell_size * 2),
        )
        return 1

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
    dem_path = _resolve_data_path(
        args.dem, _DEFAULT_DEM, "DEM", "*.tif",
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
            (forest_gdf["essence_canonical"] != "unknown").sum()
        )
        enricher_load_stats = {
            "source": "bd_foret_v2",
            "bd_foret_polygons": len(forest_gdf),
            "species_known_pct": round(
                n_known / max(len(forest_gdf), 1) * 100, 1,
            ),
        }
        logger.info(
            "   🌳 BD Forêt v2 : %d polygones, %d essences connues (%.1f%%)",
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
                    (geology_gdf["geology_canonical"] != "unknown").sum()
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

    # ── 1e. Autres couches vectorielles ───────────────────────
    hydro_gdf = loader.load_hydro(args.hydro)

    urban_gdf = None
    if not args.no_urban:
        urban_gdf = loader.load_urban(args.urban)
    else:
        logger.info("ℹ️  Masque urbain désactivé (--no-urban)")

    # ── 1f. Détection landcover par couleur OSM ───────────────
    landcover_data: dict[str, Any] | None = None
    if not args.no_landcover:
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

    # ── 2c'. Enrichissement essences (niveaux B+C+D) ─────────
    #    Remplace les scores 0.25 (unknown) par des scores
    #    régionaux pondérés par altitude et type de forêt.
    #    DOIT être APRÈS score_tree_species, AVANT apply_landcover_mask.
    logger.info("   ▸ Enrichissement essences (IFN 1997 × altitude)...")
    enricher.enrich_grid_scores(grid, forest_gdf=forest_gdf)
    enrich_stats = enricher.get_stats(grid)
    logger.info(
        "   ▸ Essences : %d/%d connues (%.1f%%), %d inconnues restantes",
        enrich_stats.get("species_known", 0),
        enrich_stats.get("forest_cells", 0),
        enrich_stats.get("pct_known", 0.0),
        enrich_stats.get("species_unknown", 0),
    )

    logger.info("   ▸ Géologie (%s)...", geology_source)
    grid.score_geology(geology_gdf)

    # ── 2d. Masque urbain (AVANT micro-habitats) ─────────────
    logger.info("   ▸ Application masque urbain...")
    urban_buffer: int = args.urban_buffer if not args.no_urban else 0
    grid.apply_urban_mask(urban_gdf, buffer_m=urban_buffer)

    # ── 2d'. Score proximité urbaine (EDT sur masque urbain) ──
    logger.info("   ▸ Score proximité urbaine...")
    grid.score_urban_proximity()

    # ── 2e. Scores micro-habitat ──────────────────────────────
    #    APRÈS apply_urban_mask (urban_mask prêt pour disturbance)
    #    AVANT apply_landcover_mask (pour être modulés par green_score)
    logger.info("   ▸ Ouverture canopée, couvert sol, perturbation...")
    grid.score_canopy_openness()
    grid.score_ground_cover()
    grid.score_disturbance()

    # ── 2f. Masque plans d'eau (scores eau → 0) ──────────────
    logger.info("   ▸ Application masque plans d'eau...")
    grid.apply_water_mask()

    # ── 2g. Modulation landcover (APRÈS tous les scores) ──────
    #    Multiplie tree_species, canopy_openness, ground_cover,
    #    disturbance par green_score.
    logger.info("   ▸ Application masque landcover (×green_score)...")
    grid.apply_landcover_mask(landcover_data)

    n_cells = grid.scores["altitude"].size
    logger.info(
        "   ⏱️  Grille : %.1fs — %s cellules (%.2f km²)",
        time.time() - t0,
        f"{n_cells:,}",
        n_cells * cell_size**2 / 1e6,
    )

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 3 — MODÈLE MULTICRITÈRE
    # ══════════════════════════════════════════════════════════

    logger.info("\n🧮 [3/4] Calcul du modèle multicritère...")
    t0 = time.time()

    model = MorilleScoring(grid)
    model.compute_weighted_score()
    model.apply_eliminatory_factors()
    model.apply_spatial_smoothing(sigma=args.smoothing_sigma)
    model.classify_probability()

    hotspots = model.get_hotspots(threshold=args.hotspot_threshold)

    logger.info("   ⏱️  Scoring : %.1fs", time.time() - t0)

    # ── Statistiques ──────────────────────────────────────────
    _final = model.final_score
    assert _final is not None, "final_score is None after scoring pipeline"
    final: np.ndarray = _final

    stats = compute_statistics(final, args.hotspot_threshold)
    stats["n_hotspots"] = len(hotspots)
    display_statistics(stats)

    # ── Validation terrain ────────────────────────────────────
    concordance = validate_against_terrain(model)

    # ══════════════════════════════════════════════════════════
    #  ÉTAPE 4 — EXPORTS
    # ══════════════════════════════════════════════════════════

    logger.info("\n🗺️  [4/4] Génération des exports...")
    t0 = time.time()

    viz = MorilleVisualizer(model, hotspots=hotspots)

    # Carte Folium HTML
    html_path = str(output_dir / "carte_morilles.html")
    _cleanup_files.append(html_path)
    viz.create_folium_map(html_path)

    # GeoTIFF raster
    try:
        tif_path = str(output_dir / "morilles_probability.tif")
        _cleanup_files.append(tif_path)
        viz.export_geotiff(tif_path)
    except (IOError, OSError) as e:
        logger.warning("GeoTIFF I/O échoué : %s", e)
    except Exception as e:
        logger.warning("GeoTIFF échoué : %s", e)

    # GPKG vecteur (optionnel)
    if args.export_vector:
        try:
            gpkg_path = str(output_dir / "morilles_grid.gpkg")
            _cleanup_files.append(gpkg_path)
            viz.export_gpkg_grid(gpkg_path, threshold=0.3)
        except (IOError, OSError) as e:
            logger.warning("GPKG I/O échoué : %s", e)
        except Exception as e:
            logger.warning("GPKG échoué : %s", e)

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
    }

    # Fusionner les stats enrichissement
    full_enrich_stats: dict[str, Any] = {**enricher_load_stats, **enrich_stats}

    duration = time.time() - t_start

    save_report(
        output_dir, stats, hotspots,
        config_snapshot, duration, concordance,
        enrichment_stats=full_enrich_stats,
    )

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
                "\n✅ Bonne concordance terrain : %.0f%%", concordance * 100,
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

    logger.info("\n🍄 Bonne chasse aux morilles !\n")

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
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exemples d'utilisation :

  # Mode auto (BD Forêt v2 + BDCharm-50 + IFN enrichissement)
  python main.py

  # Avec MNT local (GeoTIFF)
  python main.py --dem data/mnt_neron.tif

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
        help="Chemin MNT raster (.tif). Défaut : grenoble_bdalti25.tif.",
    )
    data_group.add_argument(
        "--forest", type=str, default=None,
        help="Chemin couche forêt (.shp/.gpkg). Défaut : WFS IGN.",
    )
    data_group.add_argument(
        "--geology", type=str, default=None,
        help=(
            "Chemin couche géologie (.shp/.gpkg). "
            f"Défaut : {_DEFAULT_GEOLOGY}."
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
        help=f"Chemin BD Forêt v2 SHP (défaut: {_DEFAULT_BD_FORET}).",
    )
    enrich_group.add_argument(
        "--regions", type=str, default=None,
        help=f"Chemin régions forestières IFN SHP (défaut: {_DEFAULT_REGIONS}).",
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

    # ── Paramètres modèle ──
    model_group = parser.add_argument_group("🧮 Paramètres du modèle")
    model_group.add_argument(
        "--cell-size", type=float, default=None, metavar="M",
        help="Taille de cellule en mètres (défaut: %.0fm)." % CELL_SIZE,
    )
    model_group.add_argument(
        "--hotspot-threshold", type=float, default=0.60, metavar="T",
        help="Seuil de score pour les hotspots (défaut: 0.60).",
    )
    model_group.add_argument(
        "--smoothing-sigma", type=float, default=2.0, metavar="S",
        help="Sigma du lissage gaussien spatial (défaut: 2.0).",
    )
    model_group.add_argument(
        "--seed", type=int, default=42,
        help="Seed pour reproductibilité des critères synthétiques.",
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