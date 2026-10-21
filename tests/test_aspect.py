"""Exposition (aspect) : azimut de la descente, en degrés depuis le nord,
identique sur les chemins CPU (Numba) et GPU (CuPy) — D60.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_aspect.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

import _accel

DX = 25.0
# Plans inclinés, ligne 0 = nord : (pente vers…, z(row, col)), azimut attendu
PLANES = {
    "sud": (lambda r, c: -r * 5.0, 180.0),      # descend vers le sud
    "nord": (lambda r, c: r * 5.0, 0.0),
    "est": (lambda r, c: -c * 5.0, 90.0),
    "ouest": (lambda r, c: c * 5.0, 270.0),
    "sud-est": (lambda r, c: -(r + c) * 5.0, 135.0),
}


def _plane(fn) -> np.ndarray:
    r, c = np.mgrid[0:20, 0:20].astype(np.float64)
    return fn(r, c) + 1000.0


@pytest.mark.parametrize("name", list(PLANES))
def test_cpu_aspect_points_downslope(name):
    fn, expected = PLANES[name]
    _slope, aspect = _accel._nb_slope_aspect(_plane(fn), DX)
    np.testing.assert_allclose(aspect[5:15, 5:15], expected, atol=1e-3)


@pytest.mark.skipif(not _accel.gpu_available(), reason="pas de GPU")
@pytest.mark.parametrize("name", list(PLANES))
def test_gpu_and_cpu_agree(name):
    fn, _expected = PLANES[name]
    dem = _plane(fn)
    s_cpu, a_cpu = _accel._nb_slope_aspect(dem, DX)
    s_gpu, a_gpu = _accel._gpu_slope_aspect(dem, DX)
    np.testing.assert_allclose(s_cpu[5:15, 5:15], s_gpu[5:15, 5:15], atol=1e-4)
    # Écart angulaire circulaire (0° et 360° désignent le nord)
    diff = (a_cpu[5:15, 5:15] - a_gpu[5:15, 5:15] + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(diff, 0.0, atol=1e-3)
