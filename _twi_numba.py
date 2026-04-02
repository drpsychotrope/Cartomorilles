"""_twi_numba.py — Accumulation D8 accélérée via Numba."""
from __future__ import annotations

import numba as nb  # type: ignore[import-untyped]
import numpy as np

_D8_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
_D8_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
_D8_DIST_FACTOR = np.array(
    [1.0, 1.4142135, 1.0, 1.4142135, 1.0, 1.4142135, 1.0, 1.4142135],
    dtype=np.float64,
)


@nb.njit(cache=True)
def _accumulate_d8(
    flat_idx: np.ndarray,
    flow_dir: np.ndarray,
    cell_area: float,
    ny: int,
    nx: int,
) -> np.ndarray:
    """Accumulation D8 en une passe — O(n) avec tri topologique."""
    acc = np.full(ny * nx, cell_area, dtype=np.float64)

    for i in range(flat_idx.shape[0]):
        pixel = flat_idx[i]
        r = pixel // nx
        c = pixel % nx
        d = flow_dir[r, c]
        if d < 0:
            continue
        nr = r + _D8_DR[d]
        nc = c + _D8_DC[d]
        if 0 <= nr < ny and 0 <= nc < nx:
            acc[nr * nx + nc] += acc[pixel]

    return acc.reshape(ny, nx)


@nb.njit(parallel=True, cache=True)
def _compute_flow_dir_d8(
    dem: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """Direction d'écoulement D8 — parallélisé par ligne.

    Chaque pixel est indépendant : on cherche le voisin avec la plus
    forte pente descendante parmi les 8 voisins.

    Returns
    -------
    flow_dir : int8 array (ny, nx), -1 = puits/flat.
    """
    ny = dem.shape[0]
    nx = dem.shape[1]
    flow_dir = np.full((ny, nx), np.int8(-1), dtype=np.int8)

    for r in nb.prange(ny):
        for c in range(nx):
            max_slope = 0.0
            best_dir = np.int8(-1)
            elev = dem[r, c]

            for d in range(8):
                nr = r + _D8_DR[d]
                nc = c + _D8_DC[d]
                if 0 <= nr < ny and 0 <= nc < nx:
                    drop = elev - dem[nr, nc]
                    dist = _D8_DIST_FACTOR[d] * cell_size
                    slope_nb = drop / dist
                    if slope_nb > max_slope:
                        max_slope = slope_nb
                        best_dir = np.int8(d)

            flow_dir[r, c] = best_dir

    return flow_dir