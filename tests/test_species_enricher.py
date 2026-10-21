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


def _table_var(species: str) -> float:
    """Variance des scores notés de la table d'essences de l'espèce."""
    table = config.get_species_params(species)["tree_scores"]
    return float(np.var([v for v in table.values() if not np.isnan(v)]))


def _expected_mix_strength(region_key: str, species: str) -> float:
    """Part notée × (1 − σ²/σ²_table), poids p × affinité (bande 0), D51."""
    table = config.get_species_params(species)["tree_scores"]
    scored: list[tuple[float, float]] = []
    w_all = 0.0
    for essence, p in se._REGIONAL_PROPORTIONS[region_key].items():
        w = p * se._ALT_AFFINITY.get(essence, (1.0,))[0]
        if w <= 0:
            continue
        w_all += w
        score = float(table.get(config.resolve_tree_name(essence), table["unknown"]))
        if not np.isnan(score):
            scored.append((w, score))
    w_sc = sum(w for w, _ in scored)
    mean = sum(w * s for w, s in scored) / w_sc
    var = sum(w * (s - mean) ** 2 for w, s in scored) / w_sc
    return max(0.0, w_sc / w_all * (1.0 - var / _table_var(species)))


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

    def test_morille_noire_gets_inat_level(self) -> None:
        # Niveau C' : toute espèce de taxon Morchella, pas seulement la grise
        enricher, grid = self._enrich("elata")
        ts = grid.scores["tree_species"]
        unknown = grid._tree_species_int_raster == _CODE_UNKNOWN
        self.assertTrue(np.all(np.isfinite(ts[unknown])))
        np.testing.assert_allclose(
            ts[unknown], _expected_mix_score("gresivaudan", "elata"), atol=1e-5,
        )
        self.assertEqual(enricher.inat_calls, 1)

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


class MixStrength(unittest.TestCase):
    """Fiabilité d'un mélange d'essences (D51) : part notée × (1 − σ²/σ²_table)."""

    def test_essences_of_equal_score_give_full_strength(self) -> None:
        # Tilleul et érable : 0,65 tous deux pour la morille grise → σ = 0
        score, strength = SpeciesEnricher._mix_score_strength(
            {"tilleul": 0.5, "érable_champêtre": 0.5}, 0, "esculenta",
        )
        self.assertAlmostEqual(score, 0.65, places=6)
        self.assertAlmostEqual(strength, 1.0, places=6)

    def test_split_between_extremes_gives_zero_strength(self) -> None:
        # Frêne 1,0 (aff. 1,8) / robinier 0,15 (aff. 1,5) à parts égales :
        # moyenne 0,6136, σ² = 0,179, plus que la table entière (≈ 0,082)
        score, strength = SpeciesEnricher._mix_score_strength(
            {"frêne": 0.5, "robinier": 0.5}, 0, "esculenta",
        )
        self.assertAlmostEqual(score, 0.6136, places=4)
        self.assertEqual(strength, 0.0)

    def test_unscored_share_scales_strength(self) -> None:
        # « Feuillus » : sans score (D46), affinité 1 → part notée
        # 0,75 / (0,75 + 0,5) = 0,6 ; tilleul seul noté → σ = 0
        score, strength = SpeciesEnricher._mix_score_strength(
            {"tilleul": 0.5, "feuillus": 0.5}, 0, "esculenta",
        )
        self.assertAlmostEqual(score, 0.65, places=6)
        self.assertAlmostEqual(strength, 0.6, places=6)

    def test_partial_dispersion_interpolates(self) -> None:
        # Montagnard inférieur : hêtre 0,75 (aff. 1,5), tilleul 0,65 (aff. 0,5)
        # → parts 0,75 / 0,25, moyenne 0,725, σ² = 0,001875
        score, strength = SpeciesEnricher._mix_score_strength(
            {"hêtre": 0.5, "tilleul": 0.5}, 1, "esculenta",
        )
        self.assertAlmostEqual(score, 0.725, places=6)
        self.assertAlmostEqual(
            strength, 1.0 - 0.001875 / _table_var("esculenta"), places=6,
        )
        self.assertTrue(0.9 < strength < 1.0)

    def test_no_scored_essence_gives_nan_score_and_zero_strength(self) -> None:
        score, strength = SpeciesEnricher._mix_score_strength(
            {"feuillus": 1.0}, 0, "esculenta",
        )
        self.assertTrue(np.isnan(score))
        self.assertEqual(strength, 0.0)


class EnrichGridWeightFactors(unittest.TestCase):
    """Poids par cellule de tree_species posé par l'enrichissement (D51)."""

    def test_ifn_cells_weighted_by_mix_strength(self) -> None:
        for species in ("esculenta", "elata", "edulis"):
            grid = _fake_grid()
            _Enricher().enrich_grid_scores(grid, species=species)
            unknown = grid._tree_species_int_raster == _CODE_UNKNOWN
            np.testing.assert_allclose(
                grid.weight_factors["tree_species"][unknown],
                _expected_mix_strength("gresivaudan", species), atol=1e-6,
            )

    def test_gresivaudan_collineen_mix_carries_no_weight_for_grey_morel(self) -> None:
        # σ = 0,30 du mélange (frêne 1,0 … robinier 0,15) > σ de la table :
        # le prior reste posé mais ne pèse plus (exclusion D46)
        grid = _fake_grid()
        _Enricher().enrich_grid_scores(grid, species="esculenta")
        unknown = grid._tree_species_int_raster == _CODE_UNKNOWN
        np.testing.assert_array_equal(
            grid.weight_factors["tree_species"][unknown], 0.0,
        )
        self.assertTrue(np.all(np.isfinite(grid.scores["tree_species"][unknown])))
        # Morille noire : feuillus de plaine tous médiocres → prior fiable
        grid = _fake_grid()
        _Enricher().enrich_grid_scores(grid, species="elata")
        self.assertTrue(
            np.all(grid.weight_factors["tree_species"][unknown] > 0.3),
        )

    def test_known_no_forest_lande_and_observed_cells_keep_full_weight(self) -> None:
        grid = _fake_grid()
        enricher = _Enricher()
        ft = np.zeros((grid.ny, grid.nx), dtype=np.int8)
        ft[2, :] = se._FT_LANDE
        enricher._forest_type_grid = ft
        x0, y0 = float(grid.x_coords[1]), float(grid.y_coords[1])
        enricher._observations = [{
            "nom": "test", "xmin": x0 - 1, "xmax": x0 + 1,
            "ymin": y0 - 1, "ymax": y0 + 1, "essences": {"hêtre": 1.0},
        }]
        enricher.enrich_grid_scores(grid, species="esculenta")
        w = grid.weight_factors["tree_species"]
        raster = grid._tree_species_int_raster
        np.testing.assert_array_equal(w[raster == _CODE_HETRE], 1.0)
        np.testing.assert_array_equal(w[raster == 0], 1.0)
        np.testing.assert_array_equal(w[2, 1:], 1.0)   # lande 0,10
        self.assertEqual(float(w[1, 1]), 1.0)           # observation terrain
        np.testing.assert_array_equal(w[1, 2:], 0.0)   # mélange IFN seul


class GetTreeScoreBySpecies(unittest.TestCase):
    def test_default_is_morel_table(self) -> None:
        self.assertEqual(config.get_tree_score("hêtre"), config.TREE_SCORES["hetre"])

    def test_species_table(self) -> None:
        edulis = config.get_species_params("edulis")["tree_scores"]
        self.assertEqual(config.get_tree_score("hêtre", "edulis"), edulis["hetre"])
        self.assertTrue(np.isnan(config.get_tree_score("feuillus", "edulis")))


if __name__ == "__main__":
    unittest.main()
