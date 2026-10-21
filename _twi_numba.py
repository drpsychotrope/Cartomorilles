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

@nb.njit(cache=True)
def _heap_push(hv: np.ndarray, hi: np.ndarray, size: int, v: float, i: int) -> int:
    """Insère (v, i) dans le tas binaire min (hv, hi) de taille `size`."""
    k = size
    hv[k] = v
    hi[k] = i
    while k > 0:
        p = (k - 1) >> 1
        if hv[p] <= hv[k]:
            break
        hv[p], hv[k] = hv[k], hv[p]
        hi[p], hi[k] = hi[k], hi[p]
        k = p
    return size + 1


@nb.njit(cache=True)
def _heap_pop(hv: np.ndarray, hi: np.ndarray, size: int) -> int:
    """Retire la racine du tas (lue avant l'appel) ; rend la nouvelle taille."""
    size -= 1
    hv[0] = hv[size]
    hi[0] = hi[size]
    k = 0
    while True:
        a = 2 * k + 1
        if a >= size:
            break
        b = a + 1
        m = b if (b < size and hv[b] < hv[a]) else a
        if hv[k] <= hv[m]:
            break
        hv[m], hv[k] = hv[k], hv[m]
        hi[m], hi[k] = hi[k], hi[m]
        k = m
    return size


@nb.njit(cache=True)
def _priority_flood(dem: np.ndarray, eps: float) -> np.ndarray:
    """Comble les cuvettes (Priority-Flood + ε, Barnes et al. 2014) : chaque
    cellule finit au moins ε au-dessus de celle par où elle se vide vers le
    bord de la grille, donc toute cellule a un voisin plus bas et l'écoulement
    D8 rejoint le bord au lieu de s'arrêter dans un creux du MNS (D61)."""
    ny, nx = dem.shape
    out = dem.copy()
    n = ny * nx
    closed = np.zeros(n, dtype=np.bool_)
    hv = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.int64)
    size = 0
    for r in range(ny):
        for c in range(nx):
            if r == 0 or r == ny - 1 or c == 0 or c == nx - 1:
                k = r * nx + c
                closed[k] = True
                size = _heap_push(hv, hi, size, out[r, c], k)
    while size > 0:
        v = hv[0]
        k = hi[0]
        size = _heap_pop(hv, hi, size)
        r = k // nx
        c = k % nx
        for d in range(8):
            rr = r + _D8_DR[d]
            cc = c + _D8_DC[d]
            if 0 <= rr < ny and 0 <= cc < nx:
                j = rr * nx + cc
                if not closed[j]:
                    closed[j] = True
                    if out[rr, cc] <= v:
                        out[rr, cc] = v + eps
                    size = _heap_push(hv, hi, size, out[rr, cc], j)
    return out
