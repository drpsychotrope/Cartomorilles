"""Tests de tools/evaluate_occurrences.py : combinaison de couches (D49).

Exécution : .venv/Scripts/python.exe -m unittest tests.test_evaluate_occurrences
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import evaluate_occurrences as ev  # noqa: E402


class CombineScores(unittest.TestCase):
    def test_max_ignores_nan_unless_everywhere(self) -> None:
        a = np.array([0.2, np.nan, 0.5, np.nan], dtype=np.float32)
        b = np.array([0.4, 0.3, np.nan, np.nan], dtype=np.float32)
        out = ev.combine_scores([a, b])
        np.testing.assert_allclose(out[:3], [0.4, 0.3, 0.5])
        self.assertTrue(np.isnan(out[3]))

    def test_single_layer_unchanged(self) -> None:
        a = np.array([0.1, 0.9], dtype=np.float32)
        np.testing.assert_array_equal(ev.combine_scores([a]), a)

    def test_combination_label_and_taxon_check(self) -> None:
        self.assertEqual(ev.combination_label(["esculenta", "elata"]), "esculenta+elata")
        self.assertEqual(ev.shared_taxon(["esculenta", "elata"]), ("Morchella", "GENUS"))
        with self.assertRaises(SystemExit):
            ev.shared_taxon(["esculenta", "edulis"])


if __name__ == "__main__":
    unittest.main()
