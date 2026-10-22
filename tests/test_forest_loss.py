"""Pertes de couvert forestier récentes (Hansen Global Forest Change, D64) :
tuiles, découpe et cache, année de perte sur la grille (plus proche voisin,
ou la plus récente à maille grossière), critère perturbation selon l'âge de
la perte : pleine pour les morilles, nulle pour le cèpe.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_forest_loss.py -q
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import config
import data_loader
from data_loader import DataLoader
from grid_builder import GridBuilder

_X0, _Y0 = 900_000.0, 6_450_000.0


@pytest.fixture(autouse=True)
def _reference_year(monkeypatch):
    monkeypatch.setattr(config, "FOREST_LOSS_REFERENCE_YEAR", 2026)


def _grid(cell: float, n: int) -> GridBuilder:
    bbox = MappingProxyType({
        "xmin": _X0, "ymin": _Y0, "xmax": _X0 + n * cell, "ymax": _Y0 + n * cell,
    })
    saved = (config.BBOX, config.CELL_SIZE)
    config.BBOX, config.CELL_SIZE = bbox, cell
    try:
        grid = GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved
    grid.altitude = np.full((n, n), 500.0, dtype=np.float32)
    grid.slope = np.full((n, n), 10.0, dtype=np.float32)
    grid._terrain_computed = True
    return grid


def _loss_l93(cell: float, years: np.ndarray) -> dict:
    """Codes de perte (0 = aucune, n = année 2000 + n) en L93, ligne 0 au
    nord, calés sur le coin sud-ouest de la grille."""
    h = years.shape[0]
    return {
        "data": years.astype(np.uint8),
        "transform": from_origin(_X0, _Y0 + h * cell, cell, cell),
        "crs": "EPSG:2154", "version": "test",
    }


def test_hansen_tiles_are_named_by_their_north_west_corner():
    isere = {"west": 4.98, "south": 44.67, "east": 6.44, "north": 45.89}
    assert DataLoader._hansen_tiles(isere) == ["50N_000E"]
    across = {"west": -1.5, "south": 39.5, "east": 0.5, "north": 40.5}
    assert DataLoader._hansen_tiles(across) == [
        "40N_010W", "40N_000E", "50N_010W", "50N_000E",
    ]
    south_west = {"west": -75.5, "south": -12.3, "east": -74.0, "north": -11.0}
    assert DataLoader._hansen_tiles(south_west) == ["10S_080W"]


def test_tiles_are_cropped_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)
    calls: list[str] = []

    def fake_download(self, url, dest, timeout=120):
        calls.append(url)
        # Tuile 50N_000E à 0,01° : code de perte = n° de bande de 1° de latitude
        data = (np.arange(1000)[:, None] // 100 + 1).repeat(1000, axis=1)
        with rasterio.open(
            dest, "w", driver="GTiff", height=1000, width=1000, count=1,
            dtype="uint8", crs="EPSG:4326",
            transform=from_origin(0.0, 50.0, 0.01, 0.01),
        ) as dst:
            dst.write(data.astype(np.uint8), 1)

    monkeypatch.setattr(DataLoader, "_download_file", fake_download)
    bbox = {"xmin": _X0, "ymin": _Y0, "xmax": _X0 + 2_000.0, "ymax": _Y0 + 1_500.0}
    loss = DataLoader(bbox=bbox).load_forest_loss()
    assert loss is not None and loss["crs"] == "EPSG:4326"
    assert calls == [config.HANSEN_GFC_URL.format(
        version=config.HANSEN_GFC_VERSION, tile="50N_000E",
    )]
    # Découpe sur l'emprise des requêtes, calée sur les pixels de la tuile
    data, tr = loss["data"], loss["transform"]
    b = DataLoader(bbox=bbox).bbox_wgs84_buffered
    assert tr.c <= b["west"] and tr.c + data.shape[1] * tr.a >= b["east"]
    assert tr.f >= b["north"] and tr.f + data.shape[0] * tr.e <= b["south"]
    assert data.shape[0] < 1000 and data.shape[1] < 1000      # découpée
    assert np.allclose((tr.c / 0.01, tr.f / 0.01), np.round((tr.c / 0.01, tr.f / 0.01)))
    lat = tr.f + (np.arange(data.shape[0]) + 0.5) * tr.e
    band = np.floor((50.0 - lat) / 0.01).astype(int) // 100 + 1
    np.testing.assert_array_equal(data, np.repeat(band[:, None], data.shape[1], axis=1))
    # Tuile supprimée, découpe en cache relue sans téléchargement
    assert not list((tmp_path / "cache" / "hansen").glob("*"))
    again = DataLoader(bbox=bbox).load_forest_loss()
    assert len(calls) == 1
    np.testing.assert_array_equal(again["data"], loss["data"])


def test_unavailable_tiles_leave_the_model_without_losses(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)

    def broken(self, url, dest, timeout=120):
        raise ConnectionError("hors ligne")

    monkeypatch.setattr(DataLoader, "_download_file", broken)
    bbox = {"xmin": _X0, "ymin": _Y0, "xmax": _X0 + 2_000.0, "ymax": _Y0 + 1_500.0}
    assert DataLoader(bbox=bbox).load_forest_loss() is None


def test_fine_grid_takes_the_nearest_loss_year():
    grid = _grid(10.0, 10)
    years = np.zeros((20, 20), np.uint8)
    years[:4, :4] = 24                       # 20 × 20 m au nord-ouest, 2024
    grid.set_forest_loss(_loss_l93(5.0, years))
    expected = np.zeros((10, 10), np.int16)
    expected[:2, :2] = 2024
    np.testing.assert_array_equal(grid.forest_loss_year, expected)
    age = grid.forest_loss_age()
    assert age[0, 0] == 2 and age[5, 5] == -1


def test_coarse_grid_keeps_the_latest_loss_of_each_cell():
    grid = _grid(50.0, 4)
    years = np.zeros((20, 20), np.uint8)    # pixels de 10 m, 5 × 5 par cellule
    years[0, 0] = 24                         # cellule (0, 0) : 4 % perdus
    years[0:3, 5:10] = 19                    # cellule (0, 1) : 60 %, il y a 7 ans
    years[0, 10], years[4, 14] = 10, 23      # cellule (0, 2) : deux pertes
    years[5:10, 5:10] = 5                    # cellule (1, 1) : rasée en 2005
    grid.set_forest_loss(_loss_l93(10.0, years))
    y = grid.forest_loss_year
    assert (y[0, 0], y[0, 1], y[0, 2], y[1, 1], y[3, 3]) == (2024, 2019, 2023, 2005, 0)


def test_recent_losses_raise_disturbance_for_morels_and_lower_it_for_porcini():
    grid = _grid(50.0, 4)
    base = np.full((4, 4), 0.4, np.float32)
    grid.scores["disturbance"] = base.copy()
    year = np.zeros((4, 4), np.int16)
    year[0, :] = [2025, 2022, 2021, 2018]   # 1, 4, 5 et 8 ans
    year[1, 0] = 2010                       # 16 ans : plus d'effet
    grid.forest_loss_year = year
    grid.apply_forest_loss_disturbance("esculenta")
    d = grid.scores["disturbance"]
    np.testing.assert_allclose(d[0], [1.0, 1.0, 0.8, 0.4], atol=1e-6)
    assert d[1, 0] == pytest.approx(0.4) and d[2, 2] == pytest.approx(0.4)
    # Cèpe : arbres hôtes coupés, perturbation nulle puis remontant ; rien
    # de l'espèce précédente ne reste
    grid.apply_forest_loss_disturbance("edulis")
    d = grid.scores["disturbance"]
    np.testing.assert_allclose(d[0], [0.0, 0.0, 0.08, 0.32], atol=1e-6)
    assert d[1, 0] == pytest.approx(0.4) and d[2, 2] == pytest.approx(0.4)
    grid.apply_forest_loss_disturbance("elata")
    assert grid.scores["disturbance"][0, 0] == pytest.approx(1.0)
    # Sans pertes de couvert : le critère de l'étape 2
    grid.forest_loss_year = None
    grid.apply_forest_loss_disturbance("esculenta")
    np.testing.assert_array_equal(grid.scores["disturbance"], base)
