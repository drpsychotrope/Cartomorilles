"""MNT RGE ALTI de l'IGN (D63), avec un WMS simulé : tuiles assemblées et
moyennées 2 × 2 sur la grille de l'emprise, reprise depuis le cache des
tuiles, MNT incomplet refusé, repli sur le MNS Copernicus.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_rgealti.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

import data_loader
from data_loader import DataLoader

_BBOX = {"xmin": 900_000.0, "ymin": 6_450_000.0, "xmax": 902_000.0, "ymax": 6_451_500.0}


def _plane(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 500.0 + 0.01 * (x - 900_000.0) + 0.02 * (y - 6_450_000.0)


def _fake_wms(calls: list):
    def wms(self, x0, y0, x1, y1, width, height):
        calls.append((x0, y0, x1, y1, width, height))
        xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
        ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height   # ligne 0 = nord
        return _plane(xs[None, :], ys[:, None]).astype(np.float32)
    return wms


@pytest.fixture
def loader(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)
    monkeypatch.setattr(DataLoader, "_RGEALTI_TILE_PX", 64)
    return DataLoader(bbox=_BBOX)


def test_tiles_are_averaged_onto_the_bbox_grid(loader, monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(DataLoader, "_wms_elevation", _fake_wms(calls))
    dem = loader._download_rgealti_dem(10.0)
    data, tr = dem["data"], dem["transform"]
    assert data.shape == (150, 200) and dem["source"] == "RGE ALTI"
    assert len(calls) == 4 * 3                      # tuiles de 64 px
    assert all(w <= 128 and h <= 128 for *_, w, h in calls)   # lues à 5 m
    cols, rows = np.meshgrid(np.arange(200) + 0.5, np.arange(150) + 0.5)
    xs, ys = tr * (cols, rows)
    np.testing.assert_allclose(data, _plane(xs, ys), atol=1e-3)
    # Tuiles de reprise supprimées une fois le MNT complet
    assert not list((tmp_path / "cache" / "rgealti").glob("*.npy"))


def test_interrupted_download_resumes_from_tile_cache(loader, monkeypatch):
    calls: list = []
    wms = _fake_wms(calls)

    def flaky(self, *args):
        if len(calls) >= 5:
            raise RuntimeError("réseau coupé")
        return wms(self, *args)

    monkeypatch.setattr(DataLoader, "_wms_elevation", flaky)
    with pytest.raises(RuntimeError):
        loader._download_rgealti_dem(10.0)
    done = len(calls)
    monkeypatch.setattr(DataLoader, "_wms_elevation", wms)
    loader._download_rgealti_dem(10.0)
    assert len(calls) == done + (12 - done)         # seules les tuiles manquantes


def test_incomplete_dem_is_rejected(loader, monkeypatch):
    def holes(self, x0, y0, x1, y1, width, height):
        return np.full((height, width), np.nan, dtype=np.float32)

    monkeypatch.setattr(DataLoader, "_wms_elevation", holes)
    with pytest.raises(ValueError, match="sans altitude"):
        loader._download_rgealti_dem(10.0)


def test_falls_back_to_copernicus(loader, monkeypatch):
    def broken(self, res):
        raise RuntimeError("WMS indisponible")

    copernicus = {"data": np.ones((3, 3), np.float32), "transform": None,
                  "crs": "EPSG:2154"}
    monkeypatch.setattr(DataLoader, "_download_rgealti_dem", broken)
    monkeypatch.setattr(DataLoader, "_download_copernicus_dem", lambda self: dict(copernicus))
    monkeypatch.setattr(DataLoader, "_save_dem_cache", lambda self, d, p: None)
    dem = loader.load_dem()
    assert dem["source"] == "Copernicus"
