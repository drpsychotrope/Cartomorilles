"""Classes de la carte par rang (D52) et aire drainée du calcul du TWI,
affichée en calque supplémentaire (D53).

Exécution : .venv/Scripts/python.exe -m pytest tests/test_rank_classes.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from rasterio.transform import from_origin

import config
from grid_builder import GridBuilder
from occurrences import occurrence_metrics
from scoring import rank_class_thresholds
from utils import TO_WGS84


# ── Seuils par rang (D52) ────────────────────────────────────────────
def test_thresholds_are_quantiles_of_scored_cells():
    rng = np.random.default_rng(0)
    score = rng.uniform(0.2, 1.0, 10_000).astype(np.float32).reshape(100, 100)
    th = rank_class_thresholds(score)
    assert th[0] == config.PROBABILITY_THRESHOLDS[0]
    np.testing.assert_allclose(
        th[1:], np.quantile(score, config.CLASS_RANK_QUANTILES), rtol=1e-6,
    )
    shares = np.bincount(np.digitize(score, th).ravel(), minlength=6) / score.size
    # Nul vide, puis 40 / 30 / 20 / 8 / 2 %
    assert shares[0] == 0
    np.testing.assert_allclose(shares[1:], [0.40, 0.30, 0.20, 0.08, 0.02], atol=0.002)


def test_eliminated_nan_and_outside_cells_do_not_count():
    scored = np.linspace(0.2, 1.0, 1000, dtype=np.float32)
    score = np.concatenate([
        scored,
        np.zeros(1000, np.float32),             # éliminées (Nul)
        np.full(1000, np.nan, np.float32),      # sans données
        np.full(1000, 0.99, np.float32),        # hors zone étudiée
    ])
    inside = np.concatenate([np.ones(3000, bool), np.zeros(1000, bool)])
    np.testing.assert_allclose(
        rank_class_thresholds(score, inside)[1:],
        np.quantile(scored, config.CLASS_RANK_QUANTILES), rtol=1e-6,
    )


def test_too_few_scored_cells_fall_back_to_absolute_thresholds():
    assert rank_class_thresholds(np.full(50, 0.7, np.float32)) == list(
        config.PROBABILITY_THRESHOLDS,
    )


def test_tied_scores_keep_thresholds_strictly_increasing():
    th = rank_class_thresholds(np.full(500, 0.6, np.float32))
    assert all(b > a for a, b in zip(th, th[1:]))


def test_occurrence_metrics_counts_classes_with_given_thresholds():
    """L'outil d'évaluation compte les occurrences dans les classes de la
    carte (par rang), pas dans les classes absolues."""
    transform = from_origin(900_000.0, 6_450_000.0, 50.0, 50.0)
    score = np.full((10, 10), 0.5, dtype=np.float32)
    score[4, 4] = 0.8
    x, y = transform * (4.5, 4.5)            # centre de la cellule (4, 4)
    lon, lat = TO_WGS84.transform(x, y)
    occ = [{"lat": lat, "lng": lon, "uncertainty_m": 10, "source": "inaturalist"}]
    rank = [0.15, 0.30, 0.50, 0.70, 0.90]
    absolute = occurrence_metrics(score, transform, occ)
    by_rank = occurrence_metrics(score, transform, occ, thresholds=rank)
    bon, excellent = (config.PROBABILITY_LABELS.index(c) for c in ("Bon", "Excellent"))
    assert absolute["classes"][excellent]["n_occ"] == 1   # 0,80 ≥ 0,75
    assert by_rank["classes"][bon]["n_occ"] == 1          # 0,70 ≤ 0,80 < 0,90
    assert by_rank["class_thresholds"] == rank


# ── Aire drainée D8 (D53) ────────────────────────────────────────────
def test_drained_area_accumulates_down_an_inclined_plane():
    ny, nx, cell = 6, 4, 10.0
    dem = np.tile(np.arange(ny, 0, -1, dtype=np.float64)[:, None] * 5.0, (1, nx))
    twi, acc = GridBuilder._compute_twi_and_area(dem, cell)
    # Chaque cellule reçoit toutes celles d'amont de sa colonne
    expected = np.tile(np.arange(1, ny + 1, dtype=np.float64)[:, None], (1, nx))
    np.testing.assert_allclose(acc / (cell * cell), expected)
    np.testing.assert_array_equal(twi, GridBuilder._compute_twi(dem, cell))


def test_max_before_zoom_keeps_one_pixel_thalwegs():
    """Sous-échantillonnage 30 → 18 px au plus proche voisin : une ligne
    large d'un pixel natif disparaît selon sa position ; avec le maximum
    local préalable, elle reste continue."""
    grid = SimpleNamespace(ny=18, nx=18)
    arr = np.ones((30, 30), np.float32)
    arr[13, :] = 5.0                          # ligne perdue sans maximum local
    nan = np.zeros(arr.shape, dtype=bool)
    plain = GridBuilder._zoom_grid(grid, arr, nan, order=0)  # type: ignore[arg-type]
    pooled = GridBuilder._zoom_grid(  # type: ignore[arg-type]
        grid, GridBuilder._max_before_zoom(grid, arr), nan, order=0,  # type: ignore[arg-type]
    )
    assert not (plain == 5.0).any()
    assert (pooled == 5.0).all(axis=1).any()


def test_max_before_zoom_ignores_nan_and_skips_upsampling():
    grid = SimpleNamespace(ny=4, nx=4)
    arr = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
    assert GridBuilder._max_before_zoom(grid, arr) is arr  # type: ignore[arg-type]
    grid = SimpleNamespace(ny=1, nx=1)
    pooled = GridBuilder._max_before_zoom(grid, arr)  # type: ignore[arg-type]
    assert np.nanmax(pooled) == 2.0 and np.isfinite(pooled).any()
