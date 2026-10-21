"""Masque des bords de l'eau.

D54 : les lacs et retenues de la géologie BDCharm-50 (catégorie
« hydrographie ») sont éliminés.

D55 : une cellule est éliminée quand l'eau et sa berge de
WATER_BANK_EXCLUSION_M (5 m) en couvrent au moins la moitié ; un cours
d'eau linéaire a une demi-largeur supposée STREAM_HALF_WIDTH_M (2 m). Une
cellule touchée par l'eau sans être éliminée est une berge, notée, à la
distance moyenne de sa partie non exclue.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_water_mask.py -q
"""
from __future__ import annotations

from types import MappingProxyType

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

import config
from grid_builder import GridBuilder

_X0, _Y0, _N = 900_000.0, 6_450_000.0, 6


def _grid(cell: float, n: int = _N) -> GridBuilder:
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


def _gdf(*geoms: object) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"source": ["file"] * len(geoms)}, geometry=list(geoms), crs=2154,
    )


def _stream(cell: float, n: int = _N) -> LineString:
    """Ruisseau au milieu de la 3e rangée en partant du bas."""
    y = _Y0 + 2.5 * cell
    return LineString([(_X0, y), (_X0 + n * cell, y)])


def _lake(cell: float) -> object:
    """Lac d'une cellule, dans le coin bas-gauche."""
    return box(_X0, _Y0, _X0 + cell, _Y0 + cell)


def test_stream_banks_stay_scored_on_a_coarse_grid():
    grid = _grid(50.0)
    grid.score_distance_water(_gdf(_stream(50.0)), _gdf(_lake(50.0)))
    assert grid.water_mask is not None and grid.dist_water_grid is not None
    # Le lac seul est éliminé (rangée du bas = dernière ligne du raster)
    expected = np.zeros((_N, _N), dtype=bool)
    expected[-1, 0] = True
    np.testing.assert_array_equal(grid.water_mask, expected)
    # Berge du ruisseau : (2 + 5 + 25) / 2 = 16 m de l'eau, dans l'optimum
    banks = grid.dist_water_grid == 16.0
    np.testing.assert_array_equal(np.flatnonzero(banks.any(axis=1)), [_N - 3])
    np.testing.assert_allclose(grid.scores["dist_water"][banks], 1.0)


def test_wide_river_and_its_banks_are_eliminated():
    """Rivière de 86 m de large (y = 102–188 m) : avec 5 m de berge, elle
    couvre deux rangées de 50 m à 100 % et 80 %, la troisième à 0 %."""
    grid = _grid(50.0)
    river = box(_X0, _Y0 + 102.0, _X0 + _N * 50.0, _Y0 + 188.0)
    grid.score_distance_water(_gdf(_stream(50.0)), _gdf(river))
    assert grid.water_mask is not None
    rows = np.flatnonzero(grid.water_mask.all(axis=1))
    np.testing.assert_array_equal(rows, [_N - 4, _N - 3])
    assert int(grid.water_mask.sum()) == 2 * _N


def test_fine_grid_eliminates_the_stream_and_five_metres_of_bank():
    grid = _grid(5.0)
    grid.score_distance_water(_gdf(_stream(5.0)), _gdf(_lake(5.0)))
    assert grid.water_mask is not None
    expected = np.zeros((_N, _N), dtype=bool)
    # Ruisseau : centres à moins de 2 + 5 m de l'axe → trois rangées
    expected[_N - 4:_N - 1, :] = True
    # Lac d'une cellule et sa berge de 5 m : sa voisine de la rangée du bas
    # (celle du dessus est déjà dans la berge du ruisseau)
    expected[-1, :2] = True
    np.testing.assert_array_equal(grid.water_mask, expected)


def test_geology_lakes_join_the_water_mask():
    grid = _grid(50.0, 4)
    grid.water_mask = np.zeros((4, 4), dtype=bool)
    grid.water_mask[3, 0] = True                      # un plan d'eau
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
