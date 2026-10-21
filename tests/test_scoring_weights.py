"""Poids par cellule d'un critère (D51) : config.WEIGHTS[critère] × facteur
∈ [0, 1] lu dans `grid.weight_factors`. Facteur 0 = critère exclu de la
moyenne comme une valeur NaN (D46), facteur 1 = poids plein (D47).

Grille 1 × 3 : tree_species = 0,20, les 13 autres critères = 0,80 ; poids
tree_species 0,20, autres 0,80, pénalité de couverture 0,85 + 0,15 × couverture
(D10). Valeurs attendues calculées à la main :
  facteur 1   : 0,8 × 0,8 + 0,2 × 0,2                          = 0,68
  facteur 0   : 0,80 × (0,85 + 0,15 × 0,8)                     = 0,776
  facteur 0,5 : (0,64 + 0,1 × 0,2) / 0,9 × (0,85 + 0,15 × 0,9) = 0,722333
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

import config
from scoring import MorilleScoring


def _grid(tree: float, factors: dict[str, np.ndarray] | None) -> SimpleNamespace:
    scores = {k: np.full((1, 3), 0.8, dtype=np.float32) for k in config.WEIGHTS}
    scores["tree_species"] = np.full((1, 3), tree, dtype=np.float32)
    grid = SimpleNamespace(
        scores=scores, validate_scores=lambda: True,
        nodata_mask=None, score_confidence={},
    )
    if factors is not None:
        grid.weight_factors = factors
    return grid


def _final(grid: SimpleNamespace) -> np.ndarray:
    model = MorilleScoring(grid)  # type: ignore[arg-type]
    model.compute_weighted_score()
    assert model.final_score is not None
    return model.final_score


def test_full_weight_without_factor():
    np.testing.assert_allclose(_final(_grid(0.2, None)), 0.68, rtol=1e-5)


def test_factor_interpolates_between_full_weight_and_exclusion():
    factor = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    final = _final(_grid(0.2, {"tree_species": factor}))
    np.testing.assert_allclose(final, [[0.68, 0.776, 0.722333]], rtol=1e-5)


def test_zero_factor_equals_d46_exclusion():
    excluded = _final(_grid(float("nan"), None))
    zero = _final(_grid(0.2, {"tree_species": np.zeros((1, 3), np.float32)}))
    np.testing.assert_allclose(zero, excluded, rtol=1e-6)
    np.testing.assert_allclose(zero, 0.776, rtol=1e-5)


def test_log_counts_nan_apart_from_reduced_weights(caplog):
    factor = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    with caplog.at_level(logging.INFO, logger="cartomorilles.scoring"):
        _final(_grid(0.2, {"tree_species": factor}))
    line = next(
        r.getMessage() for r in caplog.records
        if r.getMessage().lstrip().startswith("tree_species")
    )
    # NaN = valeurs manquantes seulement ; les poids réduits à part
    assert "NaN=0" in line
    assert "2 réduites" in line and "1 à ×0" in line
