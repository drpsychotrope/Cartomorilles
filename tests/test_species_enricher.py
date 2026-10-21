"""Tests de SpeciesEnricher.enrich_grid_scores par espèce modélisée (D47).

Exécution (sans pytest) :
    .venv/Scripts/python.exe -m unittest tests.test_species_enricher
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import species_enricher as se  # noqa: E402
from species_enricher import SpeciesEnricher  # noqa: E402

_CODE_HETRE = 1
_CODE_UNKNOWN = 2
_ALT = 400.0  # collinéen (bande 0)


def _expected_mix_score(region_key: str, species: str) -> float:
    """Σ(p × affinité × score) / Σ(p × affinité), essences sans score ignorées."""
    table = config.get_species_params(species)["tree_scores"]
    num = den = 0.0
    for essence, p in se._REGIONAL_PROPORTIONS[region_key].items():
        score = float(table.get(config.resolve_tree_name(essence), table["unknown"]))
        if np.isnan(score):
            continue
        aff = se._ALT_AFFINITY.get(essence, (1.0,))[0]
        num += p * aff * score
        den += p * aff
    return float(np.clip(num / den, 0.05, 1.0))


def _fake_grid(ny: int = 3, nx: int = 4) -> SimpleNamespace:
    """Grille minimale : ligne 0 = hêtre, lignes 1-2 = essence inconnue,
    colonne 0 = hors forêt (code 0)."""
    int_raster = np.full((ny, nx), _CODE_UNKNOWN, dtype=np.int16)
    int_raster[0, :] = _CODE_HETRE
    int_raster[:, 0] = 0
    scores = np.full((ny, nx), np.nan, dtype=np.float32)
    scores[int_raster == _CODE_HETRE] = 0.75
    scores[int_raster == 0] = 0.05
    # Près du centre heuristique de Grenoble → région « gresivaudan »
    x_coords = se._CX + np.arange(nx, dtype=np.float64) * 50.0
    y_coords = se._CY - np.arange(ny, dtype=np.float64) * 50.0
    return SimpleNamespace(
        ny=ny, nx=nx, cell_size=50.0, transform=None,
        scores={"tree_species": scores},
        _tree_species_int_raster=int_raster,
        _tree_code_to_name={_CODE_HETRE: "hetre", _CODE_UNKNOWN: "unknown"},
        altitude=np.full((ny, nx), _ALT, dtype=np.float32),
        x_coords=x_coords, y_coords=y_coords,
    )


class _Enricher(SpeciesEnricher):
    """Sans réseau : compte les appels iNaturalist au lieu de les faire."""

    def __init__(self) -> None:
        super().__init__()
        self.inat_calls = 0

    def _fetch_inaturalist(self, bbox_wgs84, *, cache_dir=None):  # type: ignore[override]
        self.inat_calls += 1
        return []


class EnrichGridScoresBySpecies(unittest.TestCase):
    def _enrich(self, species: str) -> tuple[_Enricher, SimpleNamespace]:
        grid = _fake_grid()
        enricher = _Enricher()
        enricher.enrich_grid_scores(grid, species=species)
        return enricher, grid

    def test_morille_uses_morel_table(self) -> None:
        enricher, grid = self._enrich("esculenta")
        ts = grid.scores["tree_species"]
        unknown = grid._tree_species_int_raster == _CODE_UNKNOWN
        self.assertTrue(np.all(np.isfinite(ts[unknown])))
        np.testing.assert_allclose(
            ts[unknown], _expected_mix_score("gresivaudan", "esculenta"), atol=1e-5,
        )
        self.assertEqual(enricher.inat_calls, 1)

    def test_cepe_uses_cepe_table(self) -> None:
        enricher, grid = self._enrich("edulis")
        ts = grid.scores["tree_species"]
        unknown = grid._tree_species_int_raster == _CODE_UNKNOWN
        self.assertTrue(np.all(np.isfinite(ts[unknown])))
        np.testing.assert_allclose(
            ts[unknown], _expected_mix_score("gresivaudan", "edulis"), atol=1e-5,
        )
        self.assertNotAlmostEqual(
            _expected_mix_score("gresivaudan", "edulis"),
            _expected_mix_score("gresivaudan", "esculenta"), places=2,
        )
        # Niveau C' (observations Morchella) : morille seulement
        self.assertEqual(enricher.inat_calls, 0)

    def test_known_and_no_forest_cells_untouched(self) -> None:
        for species in ("esculenta", "edulis"):
            _, grid = self._enrich(species)
            ts = grid.scores["tree_species"]
            raster = grid._tree_species_int_raster
            np.testing.assert_allclose(ts[raster == _CODE_HETRE], 0.75)
            np.testing.assert_allclose(ts[raster == 0], 0.05, rtol=1e-6)

    def test_lande_scores_0_10_for_both_species(self) -> None:
        for species in ("esculenta", "edulis"):
            grid = _fake_grid()
            enricher = _Enricher()
            ft = np.zeros((grid.ny, grid.nx), dtype=np.int8)
            ft[2, :] = se._FT_LANDE
            enricher._forest_type_grid = ft
            enricher.enrich_grid_scores(grid, species=species)
            ts = grid.scores["tree_species"]
            np.testing.assert_allclose(ts[2, 1:], se._LANDE_DEFAULT_SCORE)
            self.assertTrue(np.all(np.isfinite(ts[1, 1:])))

    def test_observations_scored_with_species_table(self) -> None:
        for species in ("esculenta", "edulis"):
            grid = _fake_grid()
            enricher = _Enricher()
            x0, y0 = float(grid.x_coords[1]), float(grid.y_coords[1])
            enricher._observations = [{
                "nom": "test", "xmin": x0 - 1, "xmax": x0 + 1,
                "ymin": y0 - 1, "ymax": y0 + 1,
                "essences": {"hêtre": 0.5, "frêne": 0.3, "feuillus": 0.2},
            }]
            enricher.enrich_grid_scores(grid, species=species)
            table = config.get_species_params(species)["tree_scores"]
            # « feuillus » : sans score (NaN) → ignoré, poids renormalisés (D46)
            expected = (0.5 * table["hetre"] + 0.3 * table["frene"]) / 0.8
            self.assertAlmostEqual(
                float(grid.scores["tree_species"][1, 1]), expected, places=5,
            )


class GetTreeScoreBySpecies(unittest.TestCase):
    def test_default_is_morel_table(self) -> None:
        self.assertEqual(config.get_tree_score("hêtre"), config.TREE_SCORES["hetre"])

    def test_species_table(self) -> None:
        edulis = config.get_species_params("edulis")["tree_scores"]
        self.assertEqual(config.get_tree_score("hêtre", "edulis"), edulis["hetre"])
        self.assertTrue(np.isnan(config.get_tree_score("feuillus", "edulis")))


if __name__ == "__main__":
    unittest.main()
