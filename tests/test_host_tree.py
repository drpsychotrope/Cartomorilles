"""Pas de cèpe sans arbre hôte (D57).

Le cèpe est mycorhizien strict : une cellule à plus de
SPECIES_PARAMS["edulis"]["host_tree_max_dist_m"] (30 m) d'un peuplement BD
Forêt — landes et pelouses (« NC ») et forêts sans couvert (« NR ») exclues —
est éliminée. Les morilles n'ont pas cette règle.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_host_tree.py -q
"""
from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import geopandas as gpd
import numpy as np
from shapely.geometry import box

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
        return GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved


def test_landes_and_pelouses_are_not_host_forests():
    grid = _grid()
    forest = gpd.GeoDataFrame(
        {"ESSENCE": ["Feuillus", "NC"]},
        geometry=[
            box(_X0, _Y0, _X0 + _CELL, _Y0 + _N * _CELL),            # colonne 0
            box(_X0 + 2 * _CELL, _Y0, _X0 + _N * _CELL, _Y0 + _N * _CELL),
        ],
        crs=2154,
    )
    grid.compute_host_forest(forest)
    assert grid.host_forest_mask is not None and grid.dist_host_forest is not None
    # La lande (colonnes 2-3) n'est pas un peuplement hôte
    assert grid.host_forest_mask[:, 0].all()
    assert not grid.host_forest_mask[:, 2:].any()
    np.testing.assert_allclose(grid.dist_host_forest[0], [0.0, 50.0, 100.0, 150.0])


def test_no_forest_data_means_no_rule():
    grid = _grid()
    grid.compute_host_forest(None)
    assert grid.host_forest_mask is None and grid.dist_host_forest is None


def _scoring_grid(dist: np.ndarray) -> SimpleNamespace:
    shape = dist.shape
    scores = {k: np.full(shape, 0.8, dtype=np.float32) for k in config.WEIGHTS}
    return SimpleNamespace(
        scores=scores, validate_scores=lambda: True, nodata_mask=None,
        score_confidence={}, ny=shape[0], nx=shape[1],
        dist_host_forest=dist.astype(np.float32),
    )


def _eliminated(species: str, dist: np.ndarray) -> np.ndarray:
    model = MorilleScoring(_scoring_grid(dist), species=species)  # type: ignore[arg-type]
    model.compute_weighted_score().apply_eliminatory_factors()
    assert model.final_score is not None
    return model.final_score == 0.0


def test_porcini_far_from_host_trees_is_eliminated():
    dist = np.array([[0.0, 20.0, 30.0, 40.0, 500.0]])
    np.testing.assert_array_equal(
        _eliminated("edulis", dist), [[False, False, False, True, True]],
    )


def test_morels_keep_cells_without_host_trees():
    dist = np.array([[0.0, 20.0, 30.0, 40.0, 500.0]])
    for species in ("esculenta", "elata"):
        assert not _eliminated(species, dist).any()
