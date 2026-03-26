"""diag_enricher_viz.py — Visualisation de l'enrichissement species_enricher."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from config import BBOX, CELL_SIZE
from data_loader import DataLoader
from grid_builder import GridBuilder
from species_enricher import SpeciesEnricher

logger = logging.getLogger("cartomorilles.diag_enricher_viz")

# ── Constantes visuelles ────────────────────────────────────────────────
_FIGSIZE = (18, 10)
_DPI = 150
_CMAP_SCORE = "YlGn"
_CMAP_DIFF = "RdYlGn"
_EXTENT_KM_DECIMALS = 1
_OUTPUT_DEFAULT = "output/diag_enricher_viz.png"


def _km_ticks(ax: Axes, nx: int, ny: int, cell_size: float) -> None:
    """Axes en km depuis le coin SW."""
    step_x = max(1, nx * cell_size // 5000) * 5000 / cell_size
    step_y = max(1, ny * cell_size // 5000) * 5000 / cell_size
    xt = np.arange(0, nx, step_x)
    yt = np.arange(0, ny, step_y)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v * cell_size / 1000:.{_EXTENT_KM_DECIMALS}f}" for v in xt], fontsize=7)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v * cell_size / 1000:.{_EXTENT_KM_DECIMALS}f}" for v in yt], fontsize=7)
    ax.set_xlabel("km (E→)", fontsize=8)
    ax.set_ylabel("km (N→)", fontsize=8)


def _stat_box(ax: Axes, lines: list[str]) -> None:
    """Encadré texte en haut à gauche."""
    text = "\n".join(lines)
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )


def _build_pipeline() -> tuple[GridBuilder, SpeciesEnricher, np.ndarray]:
    """Exécute le pipeline jusqu'au scoring tree_species, avant et après enrichissement."""
    loader = DataLoader()

    # ── Données ─────────────────────────────────────────────────────
    dem_data = loader.load_dem(None)
    forest_gdf = loader.load_forest(None)

    # ── Grid AVANT enrichissement ───────────────────────────────────
    grid = GridBuilder()
    grid.compute_terrain(dem_data)
    grid.score_tree_species(forest_gdf)

    score_before = np.copy(grid.scores.get("tree_species", np.array([])))
    if score_before.size == 0:
        logger.error("❌ score tree_species absent après score_tree_species()")
        sys.exit(1)

    # ── Enrichissement ──────────────────────────────────────────────
    enricher = SpeciesEnricher()
    enricher.load_bd_foret()
    enricher.enrich_grid_scores(grid, forest_gdf)

    return grid, enricher, score_before


def _render(
    grid: GridBuilder,
    enricher: SpeciesEnricher,
    score_before: np.ndarray,
    output: Path,
) -> Path:
    """Génère la figure 2×2 de diagnostic."""
    score_after = np.asarray(grid.scores["tree_species"])
    ny, nx = score_after.shape
    diff = score_after - score_before
    changed_mask = np.abs(diff) > 1e-6

    n_changed = int(np.sum(changed_mask))
    n_total = int(score_after.size)
    pct = 100.0 * n_changed / max(n_total, 1)

    stats = enricher.get_stats(grid)

    logger.info(
        "📊 Enrichissement : %d/%d cellules modifiées (%.1f%%)",
        n_changed, n_total, pct,
    )

    # ── Flip pour affichage (row 0 = nord → bas de l'image) ────────
    score_before_viz = score_before[::-1]
    score_after_viz = score_after[::-1]
    diff_viz = diff[::-1]

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE, dpi=_DPI)
    fig.suptitle(
        "Cartomorilles — Diagnostic enrichissement essences",
        fontsize=13,
        fontweight="bold",
    )

    # (0,0) Score AVANT
    ax: Axes = axes[0, 0]
    im0 = ax.imshow(score_before_viz, origin="lower", cmap=_CMAP_SCORE, vmin=0, vmax=1)
    ax.set_title("tree_species — AVANT enrichissement", fontsize=9)
    _km_ticks(ax, nx, ny, CELL_SIZE)
    fig.colorbar(im0, ax=ax, shrink=0.75, label="score")
    valid_before = score_before[~np.isnan(score_before)]
    _stat_box(ax, [
        f"mean  = {np.mean(valid_before):.3f}",
        f"median= {np.median(valid_before):.3f}",
        f"zeros = {int(np.sum(valid_before == 0))}",
    ])

    # (0,1) Score APRÈS
    ax = axes[0, 1]
    im1 = ax.imshow(score_after_viz, origin="lower", cmap=_CMAP_SCORE, vmin=0, vmax=1)
    ax.set_title("tree_species — APRÈS enrichissement", fontsize=9)
    _km_ticks(ax, nx, ny, CELL_SIZE)
    fig.colorbar(im1, ax=ax, shrink=0.75, label="score")
    valid_after = score_after[~np.isnan(score_after)]
    _stat_box(ax, [
        f"mean  = {np.mean(valid_after):.3f}",
        f"median= {np.median(valid_after):.3f}",
        f"zeros = {int(np.sum(valid_after == 0))}",
    ])

    # (1,0) Diff
    ax = axes[1, 0]
    abs_max = max(float(np.nanmax(np.abs(diff))), 0.01)
    im2 = ax.imshow(
        diff_viz, origin="lower", cmap=_CMAP_DIFF,
        vmin=-abs_max, vmax=abs_max,
    )
    ax.set_title(f"Δ score (après − avant) — {n_changed} cellules ({pct:.1f}%)", fontsize=9)
    _km_ticks(ax, nx, ny, CELL_SIZE)
    fig.colorbar(im2, ax=ax, shrink=0.75, label="Δ score")
    pos_count = int(np.sum(diff > 1e-6))
    neg_count = int(np.sum(diff < -1e-6))
    _stat_box(ax, [
        f"améliorées = {pos_count}",
        f"dégradées  = {neg_count}",
        f"inchangées = {n_total - n_changed}",
        f"Δ mean     = {np.nanmean(diff):+.4f}",
    ])

    # (1,1) Stats enricher
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Statistiques enrichissement", fontsize=9)

    table_rows: list[list[str]] = [
        ["Cellules totales", f"{stats.get('total_cells', n_total):,}"],
        ["Cellules forêt", f"{stats.get('forest_cells', 0):,}"],
        ["Essences connues", f"{stats.get('species_known', 0):,}"],
        ["Essences inconnues", f"{stats.get('species_unknown', 0):,}"],
        ["Hors forêt", f"{stats.get('no_forest', 0):,}"],
        ["% résolu", f"{stats.get('pct_known', 0):.1f}%"],
        ["Observations", f"{stats.get('observations', 0):,}"],
        ["", ""],
        ["Cellules modifiées", f"{n_changed:,}"],
        ["Δ score moyen", f"{np.nanmean(diff):+.4f}"],
        ["Score moyen avant", f"{np.nanmean(score_before):.3f}"],
        ["Score moyen après", f"{np.nanmean(score_after):.3f}"],
    ]

    table = ax.table(
        cellText=table_rows,
        colLabels=["Métrique", "Valeur"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.8, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row >= len(table_rows) - 3:
            cell.set_facecolor("#E2EFDA")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("✅ Figure sauvegardée : %s", output)
    return output


def main(args: argparse.Namespace) -> int:
    """Point d'entrée principal."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("🍄 Cartomorilles — Diagnostic enrichissement essences")

    grid, enricher, score_before = _build_pipeline()
    output = Path(args.output)
    _render(grid, enricher, score_before, output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Parser CLI."""
    parser = argparse.ArgumentParser(
        description="Diagnostic visuel de l'enrichissement species_enricher.",
    )
    parser.add_argument(
        "-o", "--output",
        default=_OUTPUT_DEFAULT,
        help=f"Chemin de sortie (défaut: {_OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode debug",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main(build_parser().parse_args()))