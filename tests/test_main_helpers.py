"""Tests des aides de main.py : validation terrain par espèce, énumération (D49).

Exécution : .venv/Scripts/python.exe -m unittest tests.test_main_helpers
"""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


class ConfigValidation(unittest.TestCase):
    def test_invalid_config_stops_run_with_code_1(self) -> None:
        # validate_config() doit être appelé au démarrage : une config
        # incohérente arrête le run (code 1 = erreur config) avant les données
        args = main.build_parser().parse_args(["--no-weather"])
        with tempfile.TemporaryDirectory() as tmp:
            args.output_dir = tmp
            try:
                # Garde-fou : si la validation n'arrête pas le run, la
                # résolution des données ne doit jamais être atteinte
                with mock.patch.object(
                    main.config, "validate_config", return_value=False,
                ) as validate, mock.patch.object(
                    main, "_resolve_data_path",
                    side_effect=AssertionError("pipeline lancé malgré la config"),
                ):
                    self.assertEqual(main.main(args), 1)
                validate.assert_called_once_with()
            finally:
                root = logging.getLogger("cartomorilles")
                for handler in list(root.handlers):
                    handler.close()
                    root.removeHandler(handler)


class Enumerate(unittest.TestCase):
    def test_enumerate_fr(self) -> None:
        self.assertEqual(main._enumerate_fr(["a"]), "a")
        self.assertEqual(main._enumerate_fr(["a", "b"]), "a et b")
        self.assertEqual(main._enumerate_fr(["a", "b", "c"]), "a, b et c")
        self.assertEqual(main._enumerate_fr([]), "")


if __name__ == "__main__":
    unittest.main()
