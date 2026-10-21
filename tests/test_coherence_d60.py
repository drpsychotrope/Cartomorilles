"""Cohérence du modèle et de ses sorties (D60).

Pente continue de la morille noire, tables d'espèce, NoData, points de
contrôle hors grille, rapport JSON strict, mélange IFN sans le type de
peuplement, occurrences dédoublées par cellule et Boyce aux ex æquo,
tampon urbain et clé de cache, filtres CPU alignés sur scipy.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_coherence_d60.py -q
"""
from __future__ import annotations

import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.stats import spearmanr

import _accel
import config
import main
from grid_builder import GridBuilder
from occurrences import _spearman, occurrence_metrics
from scoring import MorilleScoring
from species_enricher import SpeciesEnricher
from utils import TO_L93


def _grid(cell: float, n: int) -> GridBuilder:
    x0, y0 = 900_000.0, 6_450_000.0
    bbox = MappingProxyType(
        {"xmin": x0, "ymin": y0, "xmax": x0 + n * cell, "ymax": y0 + n * cell},
    )
    saved = (config.BBOX, config.CELL_SIZE)
    config.BBOX, config.CELL_SIZE = bbox, cell
    try:
        return GridBuilder()
    finally:
        config.BBOX, config.CELL_SIZE = saved


# ── Pente : pas de saut à la fin de l'optimum (morille noire) ──────────

def test_slope_breaks_leave_room_for_each_segment():
    for sp, params in config.SPECIES_PARAMS.items():
        opt, moderate, steep, maximum = config.slope_breaks(params)
        assert opt < moderate < steep < maximum, sp
    assert config.validate_config()


@pytest.mark.parametrize("species", list(config.SPECIES_PARAMS))
def test_slope_score_is_continuous(species):
    grid = _grid(20.0, 1)
    slopes = np.linspace(0.0, 44.99, 4500, dtype=np.float32)[None, :]
    grid.ny, grid.nx = slopes.shape
    grid.slope = slopes
    grid._terrain_computed = True
    grid.score_slope(species=species)
    score = grid.scores["slope"][0]
    assert np.all(np.diff(score) <= 1e-6)            # jamais de remontée
    assert np.max(np.abs(np.diff(score))) < 0.01     # ni de marche


# ── NoData du MNT : reste NaN après l'élimination ─────────────────────

def test_nodata_stays_nan_after_elimination():
    shape = (1, 3)
    nodata = np.array([[False, True, False]])
    grid = SimpleNamespace(
        scores={k: np.full(shape, 0.8, dtype=np.float32) for k in config.WEIGHTS},
        validate_scores=lambda: True, nodata_mask=nodata,
        score_confidence={}, ny=1, nx=3,
    )
    model = MorilleScoring(grid, species="esculenta")  # type: ignore[arg-type]
    model.compute_weighted_score().apply_eliminatory_factors()
    assert model.final_score is not None
    assert np.isnan(model.final_score[0, 1])
    assert np.isfinite(model.final_score[0, [0, 2]]).all()


# ── Points de contrôle hors grille : ignorés, pas notés au bord ───────

def test_checkpoints_outside_grid_are_skipped():
    lon, lat = 5.72, 45.19
    x, y = TO_L93.transform(lon, lat)
    xs = x + np.arange(-2, 3) * 50.0
    ys = y + np.arange(2, -3, -1) * 50.0
    model = SimpleNamespace(
        species="esculenta",
        grid=SimpleNamespace(x_coords=xs, y_coords=ys, cell_size=50.0),
        final_score=np.full((5, 5), 0.8, dtype=np.float32),
    )
    inside = {"name": "dedans", "lat": lat, "lon": lon, "expected": 0.8}
    lon_out, lat_out = 5.80, 45.19     # ~6 km à l'est
    outside = {"name": "dehors", "lat": lat_out, "lon": lon_out, "expected": 0.1}
    conc = main.validate_against_terrain(model, (inside, outside))  # type: ignore[arg-type]
    assert conc is not None
    assert (conc.n_ok, conc.n_points) == (1, 1)


# ── Rapport JSON strict : NaN → null ──────────────────────────────────

def test_json_safe_maps_nan_to_null():
    data = {"a": float("nan"), "b": [1.0, np.float32("inf")], "c": (np.int64(2),)}
    text = json.dumps(main._json_safe(data), allow_nan=False)
    assert json.loads(text) == {"a": None, "b": [1.0, None], "c": [2]}


# ── Mélange IFN : type de peuplement absent de la région ──────────────

def test_absent_forest_type_gives_empty_mix():
    broadleaf_only = {"chêne_pédonculé": 0.6, "hêtre": 0.4}
    assert SpeciesEnricher._filter_by_forest_type(broadleaf_only, 2) == {}
    kept = SpeciesEnricher._filter_by_forest_type(broadleaf_only, 1)
    assert kept == pytest.approx(broadleaf_only)
    score, strength = SpeciesEnricher._mix_score_strength({}, 0, "elata")
    assert np.isnan(score) and strength == 0.0


# ── Occurrences : une présence par cellule, Boyce aux ex æquo ─────────

def _occ(lat: float, lng: float) -> dict:
    return {"lat": lat, "lng": lng, "uncertainty_m": 10.0,
            "generalized": False, "source": "gbif"}


def test_repeated_records_count_once_per_cell():
    x, y = TO_L93.transform(5.72, 45.19)
    transform = from_origin(x - 500.0, y + 500.0, 50.0, 50.0)
    score = np.linspace(0.0, 1.0, 400, dtype=np.float32).reshape(20, 20)
    occs = [_occ(45.19, 5.72), _occ(45.19, 5.72), _occ(45.1905, 5.7201)]
    met = occurrence_metrics(score, transform, occs)
    assert met["n_used"] + met["n_same_cell"] == 3
    assert met["n_used"] == 3 - met["n_same_cell"] >= 1
    assert met["n_same_cell"] >= 1


def test_boyce_uses_average_ranks_for_ties():
    ranks = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ratios = [0.0, 0.0, 0.0, 1.2, 0.8, 2.0]
    assert _spearman(ranks, ratios) == pytest.approx(
        spearmanr(ranks, ratios).statistic,
    )


# ── Masque urbain : tampon en mailles entières, clé de cache ──────────

def test_buffer_cells_floor_not_bankers_rounding():
    assert _grid(20.0, 2)._buffer_cells(10) == 0
    assert _grid(20.0, 2)._buffer_cells(30) == 1      # round(1.5) donnait 2
    assert _grid(10.0, 2)._buffer_cells(10) == 1
    assert _grid(50.0, 2)._buffer_cells(40) == 0


def test_urban_cache_key_depends_on_buffer(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box

    monkeypatch.setattr(_accel, "_CACHE_DIR", tmp_path)
    grid = _grid(20.0, 40)
    x0, y0 = grid.bbox["xmin"], grid.bbox["ymin"]
    urban = gpd.GeoDataFrame(
        {"urban_type": ["batiment"], "source": ["test"]},
        geometry=[box(x0 + 300, y0 + 300, x0 + 500, y0 + 500)], crs=2154,
    )
    grid.apply_urban_mask(urban, buffer_m=0, min_urban_density=1)
    n0 = int(grid.urban_mask.sum())
    grid.apply_urban_mask(urban, buffer_m=40, min_urban_density=1)
    assert int(grid.urban_mask.sum()) > n0          # pas le masque en cache
    assert len(list(tmp_path.glob("urban_mask_*.npy"))) == 2


# ── Filtres CPU = scipy (mode « reflect »), NaN local ─────────────────

def test_cpu_filters_match_scipy(monkeypatch):
    monkeypatch.setattr(_accel, "_GPU_AVAILABLE", False)
    rng = np.random.default_rng(0)
    data = rng.random((40, 37)).astype(np.float32)
    np.testing.assert_allclose(
        _accel.uniform_filter(data, 5),
        ndimage.uniform_filter(data, size=5, mode="reflect"),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        _accel.gaussian_filter(data, 1.5),
        ndimage.gaussian_filter(data, 1.5, mode="reflect"),
        atol=1e-5,
    )
    data[20, 18] = np.nan
    out = _accel.uniform_filter(data, 5)
    assert np.isnan(out).sum() == 25                 # la seule fenêtre 5×5


# ── Carte : la classe exacte voyage dans le canal G du data PNG ───────

def test_data_png_carries_exact_class():
    import io

    from PIL import Image

    import visualize

    viz = object.__new__(visualize.MorilleVisualizer)
    score = np.array([[0.10, 0.8529, 0.8531, np.nan]], dtype=np.float32)
    viz._final_score_by_species = {"esculenta": score}
    viz._final_score = score
    viz.primary_species = "esculenta"
    viz.models = {"esculenta": SimpleNamespace(
        class_thresholds=[0.3, 0.5, 0.7, 0.853, 0.9],
    )}
    viz._study_mask = None
    viz._reproject_to_wgs84 = lambda arr, **kw: np.asarray(arr, np.float32)
    rgb = np.asarray(Image.open(io.BytesIO(viz._build_data_png_bytes("esculenta"))))
    assert rgb.shape == (1, 4, 3)
    # R = score × 200 tronqué (0,8531 → 170 comme 0,8529), 255 = NaN
    assert rgb[0, :, 0].tolist() == [20, 170, 170, 255]
    # G = classe du score exact : Bon (3) / Excellent (4) de part et d'autre
    assert rgb[0, :, 1].tolist() == [0, 3, 4, 0]
