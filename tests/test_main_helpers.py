"""Tests des aides de main.py : validation terrain par espèce, énumération (D49).

Exécution : .venv/Scripts/python.exe -m unittest tests.test_main_helpers
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402


class TerrainValidation(unittest.TestCase):
    def test_species_without_checkpoints_skips(self) -> None:
        model = SimpleNamespace(
            species="elata",
            final_score=np.zeros((2, 2), dtype=np.float32),
            grid=None,
        )
        self.assertIsNone(main.validate_against_terrain(model))

    def test_checkpoints_registered(self) -> None:
        self.assertEqual(main.TERRAIN_CHECKPOINTS_BY_SPECIES["elata"], ())
        self.assertGreater(
            len(main.TERRAIN_CHECKPOINTS_BY_SPECIES["esculenta"]), 0,
        )


class Enumerate(unittest.TestCase):
    def test_enumerate_fr(self) -> None:
        self.assertEqual(main._enumerate_fr(["a"]), "a")
        self.assertEqual(main._enumerate_fr(["a", "b"]), "a et b")
        self.assertEqual(main._enumerate_fr(["a", "b", "c"]), "a, b et c")
        self.assertEqual(main._enumerate_fr([]), "")


if __name__ == "__main__":
    unittest.main()
