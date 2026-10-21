"""Tests de la clé de cache des rasters catégoriels (D23, D27, D48).

La clé d'un raster catégoriel (essences, géologie) dépend de tout ce qui
détermine le raster : lookup code → score (D23), emprise (D27) et ordre de
rasterisation (D48). Un raster brûlé dans un autre ordre, même à lookup
identique, ne doit jamais être servi depuis le cache — c'est ce qui avait
laissé le raster morille d'avant `na_position="first"` (D46) en service.

Exécution (sans pytest) :
    .venv/Scripts/python.exe -m unittest tests.test_grid_builder_cache
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _accel  # noqa: E402
import config  # noqa: E402
import grid_builder  # noqa: E402
from grid_builder import GridBuilder  # noqa: E402

# Grille 4 × 4 cellules de 100 m, en Lambert 93 dans l'Isère
_XMIN, _YMIN, _CELL, _N = 900_000.0, 6_450_000.0, 100.0, 4
_XMAX, _YMAX = _XMIN + _N * _CELL, _YMIN + _N * _CELL
_BBOX = MappingProxyType(
    {"xmin": _XMIN, "ymin": _YMIN, "xmax": _XMAX, "ymax": _YMAX},
)
_SHAPE = (_N, _N)
_WHOLE = box(_XMIN, _YMIN, _XMAX, _YMAX)
_LEFT_HALF = box(_XMIN, _YMIN, _XMIN + _N * _CELL / 2, _YMAX)
_LEFT = (slice(None), slice(0, _N // 2))
_RIGHT = (slice(None), slice(_N // 2, None))


def _tiny_grid() -> GridBuilder:
    """GridBuilder 4 × 4 sans MNT : seuls les scores catégoriels sont appelés."""
    saved = (config.BBOX, config.CELL_SIZE)
    config.BBOX, config.CELL_SIZE = _BBOX, _CELL
    try:
        grid = GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved
    grid._terrain_computed = True
    return grid


def _expected_path(name: str, col: str, lookup: np.ndarray, codes: list[int]) -> Path:
    return _accel.raster_cache_path(
        name,
        f"{col}_{grid_builder._categorical_cache_key(lookup, np.asarray(codes, dtype=np.int16))}",
        2,
        _CELL,
        _SHAPE,
        bbox=(_XMIN, _YMIN, _XMAX, _YMAX),
    )


class CategoricalCacheKey(unittest.TestCase):
    """`_categorical_cache_key` : lookup (D23) et ordre de brûlage (D48)."""

    _LOOKUP = np.array([0.5, 0.75, np.nan, 0.3], dtype=np.float32)
    _CODES = np.array([2, 3, 1, 1, 3], dtype=np.int16)

    def test_same_inputs_same_key(self) -> None:
        self.assertEqual(
            grid_builder._categorical_cache_key(self._LOOKUP, self._CODES),
            grid_builder._categorical_cache_key(
                self._LOOKUP.copy(), self._CODES.copy(),
            ),
        )

    def test_burn_order_changes_key(self) -> None:
        # Même lookup, mêmes codes, autre ordre (NaN brûlé en dernier au lieu
        # de premier, comme avant D46) → autre clé : la régression D48
        reordered = np.ascontiguousarray(self._CODES[::-1])
        self.assertNotEqual(
            grid_builder._categorical_cache_key(self._LOOKUP, self._CODES),
            grid_builder._categorical_cache_key(self._LOOKUP, reordered),
        )

    def test_lookup_changes_key(self) -> None:
        other = self._LOOKUP.copy()
        other[1] = 0.80
        self.assertNotEqual(
            grid_builder._categorical_cache_key(self._LOOKUP, self._CODES),
            grid_builder._categorical_cache_key(other, self._CODES),
        )


class _CategoricalRasterCacheMixin:
    """Deux polygones qui se chevauchent : `raw_whole` couvre la grille avec
    le score le plus bas (ou NaN), `raw_left` sa moitié gauche avec le score
    le plus haut. Brûlés par score croissant (D22), la moitié gauche doit
    porter `raw_left`. Cache disque redirigé vers un dossier temporaire."""

    name: str
    cache_name: str
    col: str
    method: str
    raster_attr: str
    names_attr: str
    lookup_attr: str
    raw_whole: str
    raw_left: str
    canon_whole: str
    canon_left: str
    expected_left: float
    expected_right: float

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_dir = _accel._CACHE_DIR
        _accel._CACHE_DIR = Path(self._tmp.name)
        self._gdf = gpd.GeoDataFrame(
            {self.col: [self.raw_whole, self.raw_left]},
            geometry=[_WHOLE, _LEFT_HALF],
            crs="EPSG:2154",
        )

    def tearDown(self) -> None:
        _accel._CACHE_DIR = self._saved_dir
        self._tmp.cleanup()

    def _cache_files(self) -> list[Path]:
        return sorted(Path(self._tmp.name).glob("*.npy"))

    def _run(self, grid: GridBuilder) -> tuple[np.ndarray, list[str]]:
        with self.assertLogs("cartomorilles.grid_builder", level="INFO") as cm:  # type: ignore[attr-defined]
            getattr(grid, self.method)(self._gdf, species="esculenta")
        return getattr(grid, self.raster_attr), cm.output

    def _codes(self, grid: GridBuilder) -> tuple[int, int]:
        by_name = {v: k for k, v in getattr(grid, self.names_attr).items()}
        return by_name[self.canon_whole], by_name[self.canon_left]

    def test_highest_score_wins_overlap_then_cache_is_reused(self) -> None:
        grid = _tiny_grid()
        raster, logs = self._run(grid)
        code_whole, code_left = self._codes(grid)
        self.assertTrue(np.all(raster[_LEFT] == code_left))
        self.assertTrue(np.all(raster[_RIGHT] == code_whole))
        np.testing.assert_array_equal(
            grid.scores[self.name][_LEFT], np.float32(self.expected_left),
        )
        np.testing.assert_array_equal(
            grid.scores[self.name][_RIGHT], np.float32(self.expected_right),
        )
        self.assertFalse(any("cache disque" in line for line in logs))
        files = self._cache_files()
        self.assertEqual(len(files), 1)
        # Clé = colonne + lookup + codes dans l'ordre de brûlage + grille
        self.assertEqual(
            files[0],
            _expected_path(
                self.cache_name, self.col, getattr(grid, self.lookup_attr),
                [code_whole, code_left],
            ),
        )

        again = _tiny_grid()
        raster2, logs2 = self._run(again)
        self.assertTrue(any("cache disque" in line for line in logs2))
        np.testing.assert_array_equal(raster2, raster)
        np.testing.assert_array_equal(
            again.scores[self.name], grid.scores[self.name],
        )

    def test_raster_burned_in_another_order_is_not_served(self) -> None:
        grid = _tiny_grid()
        raster, _ = self._run(grid)
        code_whole, code_left = self._codes(grid)
        lookup = getattr(grid, self.lookup_attr)
        good_path = self._cache_files()[0]
        good_path.unlink()

        # Raster d'un ancien tri (le polygone couvrant brûlé en dernier :
        # il gagnait tout chevauchement) rangé sous la clé de cet ordre-là
        stale_path = _expected_path(
            self.cache_name, self.col, lookup, [code_left, code_whole],
        )
        self.assertNotEqual(stale_path, good_path)
        np.save(stale_path, np.full(_SHAPE, code_whole, dtype=np.int16))

        fresh = _tiny_grid()
        raster2, logs = self._run(fresh)
        self.assertFalse(any("cache disque" in line for line in logs))
        np.testing.assert_array_equal(raster2, raster)
        self.assertTrue(np.all(raster2[_LEFT] == code_left))
        self.assertTrue(good_path.exists())


class TreeSpeciesRasterCache(_CategoricalRasterCacheMixin, unittest.TestCase):
    name = "tree_species"
    cache_name = "tree_species"
    col = "ESSENCE"
    method = "score_tree_species"
    raster_attr = "_tree_species_int_raster"
    names_attr = "_tree_code_to_name"
    lookup_attr = "_tree_score_lookup"
    raw_whole = "NR"          # BD Forêt sans essence → unknown = NaN (D46)
    raw_left = "Hêtre"
    canon_whole = "unknown"
    canon_left = "hetre"
    expected_left = config.get_species_params("esculenta")["tree_scores"]["hetre"]
    expected_right = float("nan")

    def test_fixture_resolves(self) -> None:
        self.assertEqual(config.resolve_tree_name(self.raw_whole), "unknown")
        self.assertEqual(config.resolve_tree_name(self.raw_left), "hetre")


class GeologyRasterCache(_CategoricalRasterCacheMixin, unittest.TestCase):
    name = "geology"
    cache_name = "geology_cat"
    col = "DESCR"
    method = "score_geology"
    raster_attr = "_geology_int_raster"
    names_attr = "_geology_code_to_name"
    lookup_attr = "_geology_score_lookup"
    raw_whole = "Granite"
    raw_left = "Calcaire"
    canon_whole = "granite"
    canon_left = "calcaire"
    expected_left = config.get_species_params("esculenta")["geology_scores"]["calcaire"]
    expected_right = config.get_species_params("esculenta")["geology_scores"]["granite"]

    def test_fixture_resolves(self) -> None:
        self.assertEqual(config.resolve_geology(self.raw_whole), "granite")
        self.assertEqual(config.resolve_geology(self.raw_left), "calcaire")


if __name__ == "__main__":
    unittest.main()
