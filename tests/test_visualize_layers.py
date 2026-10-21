"""Tests de la carte : couche d'observations liée aux espèces de taxon Morchella (D49).

Exécution : .venv/Scripts/python.exe -m unittest tests.test_visualize_layers
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import visualize  # noqa: E402


class ObservationSpecies(unittest.TestCase):
    def test_both_morels_listed_in_order(self) -> None:
        self.assertEqual(
            visualize.MorilleVisualizer.observation_species(
                ["esculenta", "elata", "edulis"],
            ),
            ["esculenta", "elata"],
        )

    def test_black_morel_alone(self) -> None:
        self.assertEqual(
            visualize.MorilleVisualizer.observation_species(["elata"]), ["elata"],
        )

    def test_porcini_alone_has_no_observation_layer(self) -> None:
        self.assertEqual(
            visualize.MorilleVisualizer.observation_species(["edulis"]), [],
        )


if __name__ == "__main__":
    unittest.main()
