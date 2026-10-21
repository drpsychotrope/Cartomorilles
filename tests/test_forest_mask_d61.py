"""Masque forêt, plancher hydro en forêt, abords urbains (D61).

Landes et pelouses BD Forêt (ESSENCE « NC ») ne sont pas de la forêt pour
les micro-habitats ; le plancher dist_water en forêt (D9) s'applique sans
landcover ; la proximité urbaine n'est plus éliminatoire ; le masque urbain
de l'occupation du sol est fusionné avant les critères qui le lisent.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_forest_mask_d61.py -q
"""
from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import geopandas as gpd
import numpy as np
from shapely.geometry import box

import _accel
import config
from grid_builder import GridBuilder
from scoring import MorilleScoring

_X0, _Y0, _CELL, _N = 900_000.0, 6_450_000.0, 50.0, 4


def _grid() -> GridBuilder:
    bbox = MappingProxyType({
        "xmin": _X0, "ymin": _Y0,
        "xmax": _X0 + _N * _CELL, "ymax": _Y0 + _N * _CELL,
    })
    saved = (config.BBOX, config.CELL_SIZE)
    config.BBOX, config.CELL_SIZE = bbox, _CELL
    try:
        grid = GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved
    grid.altitude = np.full((_N, _N), 500.0, dtype=np.float32)
    grid.slope = np.full((_N, _N), 10.0, dtype=np.float32)
    grid._terrain_computed = True
    return grid


def test_landes_and_pelouses_are_not_forest(tmp_path, monkeypatch):
    monkeypatch.setattr(_accel, "_CACHE_DIR", tmp_path)
    grid = _grid()
    forest = gpd.GeoDataFrame(
        {"ESSENCE": ["Feuillus", "NC"]},
        geometry=[
            box(_X0, _Y0, _X0 + _CELL, _Y0 + _N * _CELL),            # colonne 0
            box(_X0 + 2 * _CELL, _Y0, _X0 + _N * _CELL, _Y0 + _N * _CELL),
        ],
        crs=2154,
    )
    grid.score_tree_species(forest)
    assert grid.forest_mask is not None
    assert grid.forest_mask[:, 0].all()
    assert not grid.forest_mask[:, 2:].any()          # lande : hors forêt
    # …mais toujours dans le raster d'essences (lande notée par l'enrichisseur)
    assert (grid._tree_species_int_raster[:, 2:] > 0).all()


def test_forest_water_floor_without_landcover():
    grid = _grid()
    grid.forest_mask = np.zeros((_N, _N), dtype=bool)
    grid.forest_mask[:, :2] = True
    grid.scores["dist_water"] = np.zeros((_N, _N), dtype=np.float32)
    grid.apply_forest_water_floor()
    dw = grid.scores["dist_water"]
    np.testing.assert_allclose(dw[:, :2], config.DIST_WATER_FOREST_FLOOR)
    np.testing.assert_allclose(dw[:, 2:], 0.0)


def test_urban_proximity_is_not_eliminatory():
    shape = (1, 3)
    grid = SimpleNamespace(
        scores={k: np.full(shape, 0.8, dtype=np.float32) for k in config.WEIGHTS},
        validate_scores=lambda: True, nodata_mask=None,
        score_confidence={}, ny=1, nx=3,
        dist_urban_grid=np.array([[5.0, 10.0, 200.0]], dtype=np.float32),
    )
    model = MorilleScoring(grid, species="esculenta")  # type: ignore[arg-type]
    model.compute_weighted_score().apply_eliminatory_factors()
    assert model.final_score is not None
    assert (model.final_score > 0).all()
    assert "urban_proximity" not in model.elimination_detail


def test_landcover_urban_merged_before_habitat_scores():
    grid = _grid()
    grid.urban_mask = np.zeros((_N, _N), dtype=bool)
    grid.urban_mask[0, 0] = True
    lc = np.zeros((_N, _N), dtype=bool)
    lc[3, 3] = True
    grid.merge_landcover_urban({"urban_mask": lc, "green_score": np.ones((_N, _N))})
    assert grid.urban_mask[0, 0] and grid.urban_mask[3, 3]
    assert int(grid.urban_mask.sum()) == 2


def test_roads_are_not_bridged_to_buildings(tmp_path, monkeypatch):
    """La fermeture des vides urbains (D56) ne relie pas une route aux
    maisons voisines : la forêt entre elles reste hors du masque."""
    monkeypatch.setattr(_accel, "_CACHE_DIR", tmp_path)
    cell, n = 20.0, 40
    bbox = MappingProxyType({
        "xmin": _X0, "ymin": _Y0, "xmax": _X0 + n * cell, "ymax": _Y0 + n * cell,
    })
    saved = (config.BBOX, config.CELL_SIZE)
    config.BBOX, config.CELL_SIZE = bbox, cell
    try:
        grid = GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved
    # Îlot bâti (colonnes 5-14) et route nord-sud à 60 m à l'est (colonne 18)
    house = box(_X0 + 5 * cell, _Y0 + 10 * cell, _X0 + 15 * cell, _Y0 + 30 * cell)
    road = box(_X0 + 18 * cell + 2, _Y0, _X0 + 18 * cell + 18, _Y0 + n * cell)
    urban = gpd.GeoDataFrame(
        {"urban_type": ["batiment", "route"], "source": ["test", "test"]},
        geometry=[house, road], crs=2154,
    )
    grid.apply_urban_mask(urban, buffer_m=0, min_urban_density=1)
    mask = grid.urban_mask
    rows = slice(n - 30, n - 10)                     # lignes de l'îlot
    assert mask[rows, 5:15].all() and mask[:, 18].all()
    assert not mask[rows, 16:18].any()               # forêt entre îlot et route


def test_twi_flow_crosses_pits_and_follows_burned_streams():
    """TWI (D61) : l'écoulement traverse une cuvette du MNS au lieu de s'y
    arrêter, et suit un cours d'eau gravé même sur un replat."""
    ny, nx = 40, 30
    rr, cc = np.mgrid[0:ny, 0:nx].astype(np.float64)
    dem = 500.0 - rr * 2.0 + np.abs(cc - 15) * 0.5   # vallée vers le sud
    dem[20, 14:17] += 30.0                            # barrage (pont, arbres)
    twi, acc = GridBuilder._compute_twi_and_area(dem, 20.0)
    # Toute la vallée rejoint l'exutoire ; sans comblement, l'eau s'arrêtait
    # au-dessus du barrage (la moitié amont manquait)
    assert acc[-1, 15] >= 0.95 * ny * nx * 20.0 * 20.0
    # Replat : sans gravure, l'écoulement ne suit pas la colonne 5
    flat = np.full((ny, nx), 100.0) - rr * 0.001
    burn = np.zeros((ny, nx), dtype=bool)
    burn[:, 5] = True
    _, acc_b = GridBuilder._compute_twi_and_area(flat, 20.0, burn=burn)
    assert acc_b[-1, 5] >= 0.5 * acc_b.max()
