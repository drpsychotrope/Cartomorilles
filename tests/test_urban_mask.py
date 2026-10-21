"""Masque urbain en maille fine (D56).

La couche urbaine n'a que des bâtiments : en maille fine, les rues, places
et cours de moins de 2 × URBAN_GAP_CLOSE_M (80 m) sont comblées par
fermeture morphologique ; à 50 m, rien ne change. Le rastériseur parallèle
des masques brûle aussi les géométries des rangées de bord de bande.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_urban_mask.py -q
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

import _accel
import config
from grid_builder import GridBuilder


def _grid(cell: float, n: int) -> GridBuilder:
    x0, y0 = 900_000.0, 6_450_000.0
    bbox = MappingProxyType(
        {"xmin": x0, "ymin": y0, "xmax": x0 + n * cell, "ymax": y0 + n * cell},
    )
    saved = (config.BBOX, config.CELL_SIZE)
    config.BBOX, config.CELL_SIZE = bbox, cell
    try:
        return GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved


def _blocks(n: int, gap: int) -> np.ndarray:
    """Deux îlots bâtis de 8 colonnes séparés par une rue de `gap` cellules."""
    mask = np.zeros((n, n), dtype=bool)
    mask[4:16, 2:10] = True
    mask[4:16, 10 + gap:18 + gap] = True
    return mask


def test_streets_up_to_80_m_are_closed_on_a_20_m_grid():
    grid = _grid(20.0, 30)
    for gap in (1, 2, 3, 4):                     # 20 à 80 m
        mask = _blocks(30, gap)
        closed = grid._close_urban_gaps(mask)
        # Rue comblée entre les îlots (ses deux débouchés restent ouverts)
        assert closed[6:14, 10:10 + gap].all(), gap
        assert (closed | ~mask).all()            # rien n'est retiré


def test_wider_gaps_and_outer_edges_stay_open():
    grid = _grid(20.0, 30)
    mask = _blocks(30, 5)                        # 100 m : un parc
    closed = grid._close_urban_gaps(mask)
    assert not closed[4:16, 10:15].any()
    # Le bord extérieur des îlots ne bouge pas
    assert not closed[:4, :].any() and not closed[16:, :].any()


def test_nothing_changes_on_a_50_m_grid():
    grid = _grid(50.0, 30)
    mask = _blocks(30, 1)
    np.testing.assert_array_equal(grid._close_urban_gaps(mask), mask)


def test_parallel_mask_rasterizer_matches_direct_rasterize():
    """Plus de 5 000 petits bâtiments sur 240 rangées : le chemin parallèle
    (bandes) doit brûler aussi ceux des rangées de bord de bande."""
    rng = np.random.default_rng(1)
    n = 240
    transform = from_bounds(0.0, 0.0, float(n), float(n), n, n)
    xy = rng.uniform(0.5, n - 0.5, size=(6000, 2))
    geoms = [box(x - 0.15, y - 0.15, x + 0.15, y + 0.15) for x, y in xy]
    direct = rasterize(
        [(g, 1) for g in geoms], out_shape=(n, n), transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    parallel = _accel.parallel_rasterize_mask(
        geoms, out_shape=(n, n), transform=transform, n_workers=4,
    )
    np.testing.assert_array_equal(parallel, direct)
