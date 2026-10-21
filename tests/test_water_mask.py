"""Masque des plans d'eau (D54) : les lacs et retenues de la géologie
BDCharm-50 (catégorie « hydrographie ») sont éliminés ; la couche hydro IGN
n'a que des lignes, un lac y était noté comme une terre.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_water_mask.py -q
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

import config
from grid_builder import GridBuilder

_X0, _Y0 = 900_000.0, 6_450_000.0


def _grid(cell: float, n: int) -> GridBuilder:
    bbox = MappingProxyType(
        {"xmin": _X0, "ymin": _Y0, "xmax": _X0 + n * cell, "ymax": _Y0 + n * cell},
    )
    saved = (config.BBOX, config.CELL_SIZE)
    config.BBOX, config.CELL_SIZE = bbox, cell
    try:
        grid = GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved
    grid._terrain_computed = True
    return grid


def test_geology_lakes_join_the_water_mask():
    grid = _grid(50.0, 4)
    grid.water_mask = np.zeros((4, 4), dtype=bool)
    grid.water_mask[3, 0] = True                      # un cours d'eau
    grid._geology_int_raster = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 1, 1], [0, 1, 1, 1]], dtype=np.int16,
    )
    grid._geology_code_to_name = {1: "calcaire", 2: "hydrographie"}
    grid.scores = {"altitude": np.ones((4, 4), dtype=np.float32)}
    grid.apply_water_mask()
    water = grid._geology_int_raster == 2
    water[3, 0] = True
    np.testing.assert_array_equal(grid.water_mask, water)
    assert (grid.scores["altitude"][water] == 0).all()
    assert (grid.scores["altitude"][~water] == 1).all()


def test_without_geology_the_mask_is_unchanged():
    grid = _grid(50.0, 4)
    grid.water_mask = np.eye(4, dtype=bool)
    grid.scores = {"altitude": np.ones((4, 4), dtype=np.float32)}
    grid.apply_water_mask()
    np.testing.assert_array_equal(grid.water_mask, np.eye(4, dtype=bool))
