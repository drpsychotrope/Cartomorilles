"""Tests de la configuration par espèce : morille grise, morille noire, cèpe (D49).

Exécution : .venv/Scripts/python.exe -m unittest tests.test_config_species
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


class SpeciesTables(unittest.TestCase):
    def test_three_species_validated(self) -> None:
        self.assertEqual(
            config.SUPPORTED_SPECIES, ("esculenta", "elata", "edulis"),
        )
        self.assertTrue(config.validate_config())

    def test_meleze_in_every_table(self) -> None:
        self.assertEqual(config.resolve_tree_name("Mélèze"), "meleze")
        self.assertEqual(config.TREE_LABELS["meleze"], "Mélèze")
        for species, expected in (
            ("esculenta", 0.30), ("elata", 0.85), ("edulis", 0.70),
        ):
            self.assertAlmostEqual(
                config.get_tree_score("Mélèze", species), expected, msg=species,
            )

    def test_tables_share_keys(self) -> None:
        ref = config.get_species_params("esculenta")
        for species in config.SUPPORTED_SPECIES:
            params = config.get_species_params(species)
            self.assertEqual(
                set(params["tree_scores"]), set(ref["tree_scores"]), species,
            )
            self.assertEqual(
                set(params["geology_scores"]), set(ref["geology_scores"]),
                species,
            )

    def test_elata_params(self) -> None:
        p = config.get_species_params("elata")
        self.assertEqual(p["common_name"], "Morille noire")
        self.assertEqual(p["common_name_plural"], "morilles noires")
        self.assertEqual(p["altitude_optimal"], (800.0, 1500.0))
        self.assertEqual(p["altitude_range"], (500.0, 1900.0))
        self.assertLess(p["alt_eliminatory_min"], p["altitude_range"][0])
        self.assertGreater(p["alt_eliminatory_max"], p["altitude_range"][1])
        self.assertEqual(p["eliminatory_geology"], frozenset())
        self.assertFalse(p["apply_calcdry"])
        self.assertTrue(p["apply_monotony"])
        self.assertAlmostEqual(p["tree_scores"]["sapin"], 0.95)
        self.assertAlmostEqual(p["tree_scores"]["meleze"], 0.85)
        self.assertAlmostEqual(p["geology_scores"]["granite"], 0.25)
        self.assertAlmostEqual(p["geology_scores"]["calcaire"], 1.00)
        self.assertTrue(np.isnan(p["tree_scores"]["unknown"]))

    def test_taxon_per_species(self) -> None:
        self.assertEqual(
            config.species_with_taxon("Morchella"), ("esculenta", "elata"),
        )
        self.assertEqual(config.species_with_taxon("Boletus edulis"), ("edulis",))
        self.assertEqual(
            config.get_species_params("edulis")["taxon"],
            ("Boletus edulis", "SPECIES"),
        )
        self.assertEqual(
            config.get_species_params("elata")["taxon"], ("Morchella", "GENUS"),
        )

    def test_grey_morel_relabelled(self) -> None:
        p = config.get_species_params("esculenta")
        self.assertEqual(p["common_name"], "Morille grise")
        self.assertEqual(p["common_name_plural"], "morilles grises")
        self.assertEqual(p["altitude_optimal"], (200.0, 600.0))


if __name__ == "__main__":
    unittest.main()
