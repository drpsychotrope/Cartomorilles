"""Point représentatif d'un hotspot : la cellule de l'amas au score maximal,
à égalité la plus proche du centroïde (D50). Le centroïde d'un amas en L
tombe hors de l'amas, sur une cellule éliminée."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from rasterio.transform import from_origin

import _accel
import config
from scoring import MorilleScoring

CELL = 50.0
X0, Y0 = 900_000.0, 6_500_000.0  # coin nord-ouest (Lambert-93)
TRANSFORM = from_origin(X0, Y0, CELL, CELL)
# Amas en L de 11 cellules : bras vertical (col 2, lignes 2–8) et bras
# horizontal (ligne 8, cols 3–6). Centroïde (6,09 ; 2,91) → cellule (6, 3),
# hors de l'amas.
L_CELLS = [(r, 2) for r in range(2, 9)] + [(8, c) for c in range(3, 7)]


def _cell_center(row: int, col: int) -> tuple[float, float]:
    x, y = TRANSFORM * (col + 0.5, row + 0.5)
    return float(x), float(y)


def _cell_of(x: float, y: float) -> tuple[int, int]:
    col, row = ~TRANSFORM * (x, y)
    return int(np.floor(row)), int(np.floor(col))


def _l_cluster(peak: tuple[int, int] | None, base: float = 0.7, top: float = 0.9):
    score = np.zeros((12, 12), dtype=np.float32)
    labeled = np.zeros((12, 12), dtype=np.int32)
    for r, c in L_CELLS:
        score[r, c] = base
        labeled[r, c] = 1
    if peak is not None:
        score[peak] = top
    return score, labeled


def _stats(score: np.ndarray, labeled: np.ndarray, n_clusters: int):
    t = TRANSFORM
    return _accel.vectorized_cluster_stats(
        labeled=labeled, n_clusters=n_clusters, final_score=score,
        transform_params=(t.a, t.b, t.c, t.d, t.e, t.f),
        cell_size=CELL, min_cluster_size=1,
    )


def test_centroid_of_l_shaped_cluster_falls_outside_it():
    """Précondition du bug : le centroïde du L n'est pas une cellule du L."""
    rows = np.array([r for r, _ in L_CELLS], dtype=float)
    cols = np.array([c for _, c in L_CELLS], dtype=float)
    x, y = TRANSFORM * (cols.mean() + 0.5, rows.mean() + 0.5)
    assert _cell_of(x, y) not in L_CELLS


def test_point_is_the_max_score_cell_of_the_cluster():
    score, labeled = _l_cluster(peak=(2, 2))
    stats = _stats(score, labeled, n_clusters=1)
    assert (float(stats["x_l93"][0]), float(stats["y_l93"][0])) == _cell_center(2, 2)
    assert float(stats["max_score"][0]) == np.float32(0.9)


def test_tie_picks_the_cluster_cell_closest_to_the_centroid():
    score, labeled = _l_cluster(peak=None)  # plateau : toutes à 0,7
    stats = _stats(score, labeled, n_clusters=1)
    point = _cell_of(float(stats["x_l93"][0]), float(stats["y_l93"][0]))
    assert point in L_CELLS
    rows = np.array([r for r, _ in L_CELLS], dtype=float)
    cols = np.array([c for _, c in L_CELLS], dtype=float)
    d2 = (rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2
    assert point == L_CELLS[int(np.argmin(d2))]  # (6, 2), unique


def test_each_cluster_gets_its_own_peak():
    score, labeled = _l_cluster(peak=(8, 6))
    score[0:2, 9:11] = 0.65          # 2e amas : bloc 2×2 dans le coin
    score[1, 10] = 0.8
    labeled[0:2, 9:11] = 2
    stats = _stats(score, labeled, n_clusters=2)
    points = {
        int(stats["valid_ids"][i]): (float(stats["x_l93"][i]), float(stats["y_l93"][i]))
        for i in range(int(stats["n_valid"]))
    }
    assert points == {1: _cell_center(8, 6), 2: _cell_center(1, 10)}


def test_get_hotspots_aims_at_the_best_cell(monkeypatch):
    """Bout en bout : x_l93/y_l93 du hotspot = cellule de score max, et le
    score du modèle en ce point est ≥ au seuil (jamais une cellule éliminée)."""
    monkeypatch.setattr(config, "CELL_SIZE", CELL)
    score, _ = _l_cluster(peak=(4, 2))
    grid = SimpleNamespace(scores={}, transform=TRANSFORM, altitude=None, slope=None)
    model = MorilleScoring(grid)  # type: ignore[arg-type]
    model.final_score = score
    model._step_weighted = True
    (hotspot,) = model.get_hotspots(threshold=0.6)
    assert hotspot["n_cells"] == len(L_CELLS)
    assert (hotspot["x_l93"], hotspot["y_l93"]) == _cell_center(4, 2)
    assert score[_cell_of(hotspot["x_l93"], hotspot["y_l93"])] >= 0.6
