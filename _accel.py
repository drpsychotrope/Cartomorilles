"""_accel.py — Accélération hardware pour Cartomorilles.

Backend hybride : CuPy (GPU NVIDIA) avec fallback Numba CPU parallèle.
RTX 3080 10 GB — chaque grille ~33 MB (8.25M px float32), ~300 arrays en VRAM.

Fonctions accélérées :
- slope/aspect (Horn, résolution DEM native)
- roughness (uniform_filter → integral image)
- gaussian_filter (séparable 1D)
- distance_transform_edt
- D8 flow direction (TWI)
- uniform_filter (scoring density)
- reproject_l93_to_wgs84 (bilinéaire via map_coordinates)
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    pass

logger = logging.getLogger("cartomorilles._accel")

# ═══════════════════════════════════════════════════════════════════
# GPU BACKEND DETECTION
# ═══════════════════════════════════════════════════════════════════

_cp = None
_cpx_ndi = None
_GPU_AVAILABLE = False
_GPU_NAME = "none"
_GPU_MEM_GB = 0.0

try:
    import cupy as _cp_import
    import cupyx.scipy.ndimage as _cpx_ndi_import

    _cp_import.cuda.Device(0).compute_capability
    _cp = _cp_import
    _cpx_ndi = _cpx_ndi_import
    _dev = _cp.cuda.Device(0)
    _GPU_AVAILABLE = True
    _GPU_NAME = _dev.attributes["DeviceName"] if hasattr(
        _dev, "attributes",
    ) else str(_dev)
    _GPU_MEM_GB = _dev.mem_info[1] / (1024**3)
    logger.info(
        "🟢 GPU détecté : %s (%.1f GB VRAM)", _GPU_NAME, _GPU_MEM_GB,
    )
except Exception:  # noqa: BLE001
    logger.info("🟡 CuPy indisponible — fallback CPU Numba parallèle")


def gpu_available() -> bool:
    """Retourne True si le backend GPU CuPy est fonctionnel."""
    return _GPU_AVAILABLE


def device_info() -> dict[str, str | float | bool]:
    """Informations sur le device d'accélération actif."""
    return {
        "backend": "cupy" if _GPU_AVAILABLE else "numba_cpu",
        "gpu": _GPU_AVAILABLE,
        "gpu_name": _GPU_NAME,
        "gpu_mem_gb": round(_GPU_MEM_GB, 1),
    }


def sync_gpu() -> None:
    """Synchronise le device GPU (attend fin des kernels)."""
    if _GPU_AVAILABLE:
        assert _cp is not None
        _cp.cuda.Stream.null.synchronize()


def free_gpu_memory() -> None:
    """Libère le pool mémoire CuPy."""
    if _GPU_AVAILABLE:
        assert _cp is not None
        pool = _cp.get_default_memory_pool()
        pool.free_all_blocks()
        logger.debug("   GPU memory pool libéré")


def _to_gpu(arr: np.ndarray) -> object:
    """Transfère un ndarray numpy vers le GPU."""
    assert _cp is not None
    return _cp.asarray(arr)


def _to_cpu(arr: object) -> np.ndarray:
    """Transfère un array GPU vers numpy."""
    assert _cp is not None
    return _cp.asnumpy(arr)


# ═══════════════════════════════════════════════════════════════════
# NUMBA CPU KERNELS (fallback parallèle)
# ═══════════════════════════════════════════════════════════════════


@njit(parallel=True, cache=True)
def _nb_slope_aspect(
    dem: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Horn slope/aspect — Numba parallèle.

    Retourne (slope_deg, aspect_deg) en float32.
    """
    ny, nx = dem.shape
    slope = np.empty((ny, nx), dtype=np.float32)
    aspect = np.empty((ny, nx), dtype=np.float32)
    inv_8dx = 1.0 / (8.0 * dx)

    for i in prange(ny):
        for j in range(nx):
            i0 = max(i - 1, 0)
            i2 = min(i + 1, ny - 1)
            j0 = max(j - 1, 0)
            j2 = min(j + 1, nx - 1)

            dz_dx = (
                (dem[i0, j2] + 2.0 * dem[i, j2] + dem[i2, j2])
                - (dem[i0, j0] + 2.0 * dem[i, j0] + dem[i2, j0])
            ) * inv_8dx

            dz_dy = (
                (dem[i2, j0] + 2.0 * dem[i2, j] + dem[i2, j2])
                - (dem[i0, j0] + 2.0 * dem[i0, j] + dem[i0, j2])
            ) * inv_8dx

            mag = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)
            slope[i, j] = np.degrees(np.arctan(mag))

            if mag < 1e-8:
                aspect[i, j] = -1.0
            else:
                a = np.degrees(np.arctan2(-dz_dy, dz_dx))
                if a < 0.0:
                    a += 360.0
                a = 90.0 - a
                if a < 0.0:
                    a += 360.0
                aspect[i, j] = a

    return slope, aspect


@njit(parallel=True, cache=True)
def _nb_gaussian_filter_nearest(
    data: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Filtre gaussien séparable — boundary mode 'nearest' (clamp)."""
    ny, nx = data.shape

    radius = int(4.0 * sigma + 0.5)
    if radius < 1:
        radius = 1
    ksize = 2 * radius + 1
    kernel = np.empty(ksize, dtype=np.float64)
    s = 0.0
    for k in range(ksize):
        v = np.exp(-0.5 * ((k - radius) / sigma) ** 2)
        kernel[k] = v
        s += v
    for k in range(ksize):
        kernel[k] /= s

    tmp = np.empty((ny, nx), dtype=np.float64)
    for i in prange(ny):
        for j in range(nx):
            acc = 0.0
            for k in range(ksize):
                jj = j + k - radius
                if jj < 0:
                    jj = 0
                elif jj >= nx:
                    jj = nx - 1
                acc += data[i, jj] * kernel[k]
            tmp[i, j] = acc

    out = np.empty((ny, nx), dtype=np.float32)
    for i in prange(ny):
        for j in range(nx):
            acc = 0.0
            for k in range(ksize):
                ii = i + k - radius
                if ii < 0:
                    ii = 0
                elif ii >= ny:
                    ii = ny - 1
                acc += tmp[ii, j] * kernel[k]
            out[i, j] = np.float32(acc)

    return out


@njit(parallel=True, cache=True)
def _nb_uniform_filter(data: np.ndarray, size: int) -> np.ndarray:
    """Filtre uniforme via image intégrale — Numba parallèle."""
    ny, nx = data.shape
    half = size // 2

    integ = np.empty((ny + 1, nx + 1), dtype=np.float64)
    for i in range(ny + 1):
        integ[i, 0] = 0.0
    for j in range(nx + 1):
        integ[0, j] = 0.0

    for i in range(ny):
        row_sum = 0.0
        for j in range(nx):
            row_sum += data[i, j]
            integ[i + 1, j + 1] = integ[i, j + 1] + row_sum

    out = np.empty((ny, nx), dtype=np.float32)
    for i in prange(ny):
        for j in range(nx):
            r0 = max(i - half, 0)
            r1 = min(i + half + 1, ny)
            c0 = max(j - half, 0)
            c1 = min(j + half + 1, nx)
            area = (r1 - r0) * (c1 - c0)
            s = (
                integ[r1, c1]
                - integ[r0, c1]
                - integ[r1, c0]
                + integ[r0, c0]
            )
            out[i, j] = np.float32(s / area)

    return out


@njit(parallel=True, cache=True)
def _nb_gaussian_filter(
    data: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Filtre gaussien séparable 1D — Numba parallèle."""
    ny, nx = data.shape

    radius = int(4.0 * sigma + 0.5)
    if radius < 1:
        radius = 1
    ksize = 2 * radius + 1
    kernel = np.empty(ksize, dtype=np.float64)
    s = 0.0
    for k in range(ksize):
        v = np.exp(-0.5 * ((k - radius) / sigma) ** 2)
        kernel[k] = v
        s += v
    for k in range(ksize):
        kernel[k] /= s

    tmp = np.empty((ny, nx), dtype=np.float64)
    for i in prange(ny):
        for j in range(nx):
            acc = 0.0
            for k in range(ksize):
                jj = j + k - radius
                if jj < 0:
                    jj = -jj
                if jj >= nx:
                    jj = 2 * nx - 2 - jj
                acc += data[i, jj] * kernel[k]
            tmp[i, j] = acc

    out = np.empty((ny, nx), dtype=np.float32)
    for i in prange(ny):
        for j in range(nx):
            acc = 0.0
            for k in range(ksize):
                ii = i + k - radius
                if ii < 0:
                    ii = -ii
                if ii >= ny:
                    ii = 2 * ny - 2 - ii
                acc += tmp[ii, j] * kernel[k]
            out[i, j] = np.float32(acc)

    return out


@njit(parallel=True, cache=True)
def _nb_flow_dir_d8(dem: np.ndarray, dx: float) -> np.ndarray:
    """Direction d'écoulement D8 — Numba parallèle."""
    diag = np.sqrt(2.0) * dx
    ny, nx = dem.shape
    fdir = np.full((ny, nx), np.int8(-1), dtype=np.int8)

    di = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
    dj = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dd = np.array(
        [dx, diag, dx, diag, dx, diag, dx, diag],
        dtype=np.float64,
    )

    for i in prange(ny):
        for j in range(nx):
            max_drop = 0.0
            best = np.int8(-1)
            for k in range(8):
                ni = i + di[k]
                nj = j + dj[k]
                if 0 <= ni < ny and 0 <= nj < nx:
                    drop = (dem[i, j] - dem[ni, nj]) / dd[k]
                    if drop > max_drop:
                        max_drop = drop
                        best = np.int8(k)
            fdir[i, j] = best

    return fdir


@njit(parallel=True, cache=True)
def _nb_map_coordinates_bilinear(
    source: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Interpolation bilinéaire via coordonnées fractionnaires — Numba.

    source: (src_h, src_w) float32
    row_coords, col_coords: (dst_h, dst_w) float64 — coords dans source
    fill_value: valeur hors bornes (NaN pour scores, 0 pour masques)
    """
    dst_h, dst_w = row_coords.shape
    src_h, src_w = source.shape
    out = np.empty((dst_h, dst_w), dtype=np.float32)

    for i in prange(dst_h):
        for j in range(dst_w):
            r = row_coords[i, j]
            c = col_coords[i, j]

            # Hors bornes → fill
            if r < 0.0 or r > src_h - 1.0 or c < 0.0 or c > src_w - 1.0:
                out[i, j] = np.float32(fill_value)
                continue

            r0 = int(r)
            c0 = int(c)
            r1 = min(r0 + 1, src_h - 1)
            c1 = min(c0 + 1, src_w - 1)

            dr = r - r0
            dc = c - c0

            val = (
                source[r0, c0] * (1.0 - dr) * (1.0 - dc)
                + source[r0, c1] * (1.0 - dr) * dc
                + source[r1, c0] * dr * (1.0 - dc)
                + source[r1, c1] * dr * dc
            )
            out[i, j] = np.float32(val)

    return out


# ═══════════════════════════════════════════════════════════════════
# GPU KERNELS (CuPy)
# ═══════════════════════════════════════════════════════════════════


def _gpu_slope_aspect(
    dem: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Horn slope/aspect via CuPy convolution 2D."""
    assert _cp is not None and _cpx_ndi is not None

    g = _cp.asarray(dem, dtype=_cp.float64)

    kx = _cp.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=_cp.float64,
    ) / (8.0 * dx)
    ky = _cp.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=_cp.float64,
    ) / (8.0 * dx)

    dz_dx = _cpx_ndi.convolve(g, kx)
    dz_dy = _cpx_ndi.convolve(g, ky)

    mag = _cp.sqrt(dz_dx**2 + dz_dy**2)
    slope_deg = _cp.degrees(_cp.arctan(mag))

    aspect_deg = _cp.degrees(_cp.arctan2(-dz_dy, dz_dx))
    aspect_deg = 90.0 - aspect_deg
    aspect_deg = _cp.where(aspect_deg < 0.0, aspect_deg + 360.0, aspect_deg)
    aspect_deg = _cp.where(mag < 1e-8, -1.0, aspect_deg)

    return (
        _cp.asnumpy(slope_deg.astype(_cp.float32)),
        _cp.asnumpy(aspect_deg.astype(_cp.float32)),
    )


def _gpu_uniform_filter(data: np.ndarray, size: int) -> np.ndarray:
    """Filtre uniforme CuPy."""
    assert _cp is not None and _cpx_ndi is not None

    g = _cp.asarray(data, dtype=_cp.float32)
    result = _cpx_ndi.uniform_filter(g, size=size)
    return _cp.asnumpy(result)


def _gpu_gaussian_filter(
    data: np.ndarray,
    sigma: float,
    mode: str = "reflect",
) -> np.ndarray:
    """Filtre gaussien CuPy."""
    assert _cp is not None and _cpx_ndi is not None

    g = _cp.asarray(data, dtype=_cp.float32)
    result = _cpx_ndi.gaussian_filter(g, sigma=sigma, mode=mode)
    return _cp.asnumpy(result)


def _gpu_distance_transform_edt(
    mask: np.ndarray,
    sampling: tuple[float, ...] | None = None,
) -> np.ndarray:
    """EDT via CuPy."""
    assert _cp is not None and _cpx_ndi is not None

    g = _cp.asarray(mask)
    raw = _cpx_ndi.distance_transform_edt(g, sampling=sampling)
    result: np.ndarray = np.asarray(_cp.asnumpy(raw), dtype=np.float32)
    return result


def _gpu_flow_dir_d8(dem: np.ndarray, dx: float) -> np.ndarray:
    """D8 flow direction via CuPy RawKernel."""
    assert _cp is not None

    _kernel_src = r"""
    extern "C" __global__
    void flow_dir_d8(
        const double* dem, signed char* fdir,
        int ny, int nx, double dx
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= ny * nx) return;

        int i = idx / nx;
        int j = idx % nx;

        double diag = dx * 1.4142135623730951;

        int di[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
        int dj[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        double dd[8] = {dx, diag, dx, diag, dx, diag, dx, diag};

        double max_drop = 0.0;
        signed char best = -1;

        for (int k = 0; k < 8; k++) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (ni >= 0 && ni < ny && nj >= 0 && nj < nx) {
                double drop = (dem[idx] - dem[ni * nx + nj]) / dd[k];
                if (drop > max_drop) {
                    max_drop = drop;
                    best = (signed char)k;
                }
            }
        }
        fdir[idx] = best;
    }
    """

    kernel = _cp.RawKernel(_kernel_src, "flow_dir_d8")
    ny, nx = dem.shape

    g_dem = _cp.asarray(dem, dtype=_cp.float64)
    g_fdir = _cp.full(ny * nx, -1, dtype=_cp.int8)

    block = 256
    grid = (ny * nx + block - 1) // block
    kernel((grid,), (block,), (g_dem, g_fdir, ny, nx, dx))

    return _cp.asnumpy(g_fdir.reshape(ny, nx))


def _gpu_map_coordinates_bilinear(
    source: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Interpolation bilinéaire via cupyx.scipy.ndimage.map_coordinates."""
    assert _cp is not None and _cpx_ndi is not None

    g_src = _cp.asarray(source, dtype=_cp.float32)
    g_rows = _cp.asarray(row_coords, dtype=_cp.float64)
    g_cols = _cp.asarray(col_coords, dtype=_cp.float64)

    coords = _cp.stack([g_rows, g_cols], axis=0)

    result = _cpx_ndi.map_coordinates(
        g_src,
        coords,
        order=1,
        mode="constant",
        cval=fill_value,
    )
    return _cp.asnumpy(result.astype(_cp.float32))


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — Auto-dispatch GPU / CPU
# ═══════════════════════════════════════════════════════════════════


def compute_slope_aspect(
    dem: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule slope (°) et aspect (°) via Horn.

    GPU si disponible, sinon Numba CPU parallèle.
    """
    dem = np.asarray(dem, dtype=np.float64)
    t0 = time.perf_counter()

    if _GPU_AVAILABLE:
        slope, aspect = _gpu_slope_aspect(dem, dx)
        backend = "GPU"
    else:
        slope, aspect = _nb_slope_aspect(dem, dx)
        backend = "CPU"

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ slope/aspect %dx%d → %.2fs [%s]",
        dem.shape[1], dem.shape[0], dt, backend,
    )
    return slope, aspect


def compute_roughness(
    slope: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rugosité = écart-type local de la pente."""
    slope = np.asarray(slope, dtype=np.float32)
    t0 = time.perf_counter()

    slope_mean = uniform_filter(slope, window)
    slope_sq_mean = uniform_filter(slope**2, window)
    roughness = np.sqrt(
        np.maximum(slope_sq_mean - slope_mean**2, 0),
    ).astype(np.float32)

    dt = time.perf_counter() - t0
    backend = "GPU" if _GPU_AVAILABLE else "CPU"
    logger.info(
        "⚡ roughness %dx%d w=%d → %.2fs [%s]",
        slope.shape[1], slope.shape[0], window, dt, backend,
    )
    return roughness


def uniform_filter(data: np.ndarray, size: int) -> np.ndarray:
    """Filtre uniforme (box filter) accéléré."""
    data = np.asarray(data, dtype=np.float32)
    if _GPU_AVAILABLE:
        return _gpu_uniform_filter(data, size)
    return _nb_uniform_filter(data, size)


def gaussian_filter(
    data: np.ndarray,
    sigma: float,
    mode: str = "reflect",
) -> np.ndarray:
    """Filtre gaussien accéléré."""
    data = np.asarray(data, dtype=np.float32)
    t0 = time.perf_counter()

    if _GPU_AVAILABLE:
        result = _gpu_gaussian_filter(data, sigma, mode=mode)
        backend = "GPU"
    else:
        if mode == "nearest":
            result = _nb_gaussian_filter_nearest(data, sigma)
        else:
            result = _nb_gaussian_filter(data, sigma)
        backend = "CPU"

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ gaussian σ=%.1f %dx%d → %.2fs [%s]",
        sigma, data.shape[1], data.shape[0], dt, backend,
    )
    return result


def distance_transform_edt(
    mask: np.ndarray,
    sampling: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Transform de distance euclidienne accéléré."""
    mask = np.asarray(mask)
    t0 = time.perf_counter()

    if _GPU_AVAILABLE:
        result = _gpu_distance_transform_edt(mask, sampling)
        backend = "GPU"
    else:
        from scipy.ndimage import distance_transform_edt as scipy_edt
        result = np.asarray(
            scipy_edt(mask, sampling=sampling),
            dtype=np.float32,
        )
        backend = "CPU"

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ EDT %dx%d → %.2fs [%s]",
        mask.shape[1], mask.shape[0], dt, backend,
    )
    return result


def compute_flow_dir_d8(
    dem: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Direction d'écoulement D8 accéléré."""
    dem = np.asarray(dem, dtype=np.float64)
    t0 = time.perf_counter()

    if _GPU_AVAILABLE:
        fdir = _gpu_flow_dir_d8(dem, dx)
        backend = "GPU"
    else:
        fdir = _nb_flow_dir_d8(dem, dx)
        backend = "CPU"

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ D8 flow dir %dx%d → %.2fs [%s]",
        dem.shape[1], dem.shape[0], dt, backend,
    )
    return fdir


def reproject_l93_to_wgs84(
    source: np.ndarray,
    src_bounds_l93: tuple[float, float, float, float],
    dst_transform_params: tuple[float, ...],
    dst_shape: tuple[int, int],
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Reprojette un array 2D L93 → WGS84 via interpolation bilinéaire.

    Construit la grille de mapping (WGS84 → L93 → pixel source)
    puis interpole via map_coordinates (GPU) ou Numba (CPU).

    Parameters
    ----------
    source : array 2D orienté (row 0 = nord)
    src_bounds_l93 : (xmin, ymin, xmax, ymax) en Lambert-93
    dst_transform_params : Affine coefficients (a, b, c, d, e, f)
        du raster destination WGS84
    dst_shape : (dst_height, dst_width)
    fill_value : valeur hors bornes (np.nan pour scores, 0.0 pour masques)

    Returns
    -------
    np.ndarray float32 (dst_height, dst_width)
    """
    source = np.asarray(source, dtype=np.float32)
    t0 = time.perf_counter()

    src_h, src_w = source.shape
    dst_h, dst_w = dst_shape
    xmin, ymin, xmax, ymax = src_bounds_l93
    a, b, c, d, e, f = dst_transform_params[:6]

    # ── Grille de coordonnées WGS84 (centre pixel) ──
    dst_cols = np.arange(dst_w, dtype=np.float64) + 0.5
    dst_rows = np.arange(dst_h, dtype=np.float64) + 0.5

    # WGS84 coords from affine : lon = c + col*a, lat = f + row*e
    lons_1d = c + dst_cols * a
    lats_1d = f + dst_rows * e

    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)

    # ── Projection WGS84 → L93 ──
    from pyproj import Transformer
    wgs84_to_l93 = Transformer.from_crs(
        "EPSG:4326", "EPSG:2154", always_xy=True,
    )
    x_l93, y_l93 = wgs84_to_l93.transform(lons_2d, lats_2d)

    # ── Coordonnées pixel dans le raster source ──
    # Source orienté row 0 = nord → ymax en haut
    col_coords = (x_l93 - xmin) / (xmax - xmin) * src_w - 0.5
    row_coords = (ymax - y_l93) / (ymax - ymin) * src_h - 0.5

    # ── Interpolation bilinéaire ──
    if _GPU_AVAILABLE:
        result = _gpu_map_coordinates_bilinear(
            source, row_coords, col_coords, fill_value,
        )
        backend = "GPU"
    else:
        result = _nb_map_coordinates_bilinear(
            source, row_coords, col_coords, fill_value,
        )
        backend = "CPU"

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ reproject L93→WGS84 %dx%d → %dx%d → %.2fs [%s]",
        src_w, src_h, dst_w, dst_h, dt, backend,
    )
    return np.asarray(result, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# AXE A — RASTERISATION PARALLÈLE + CACHE DISQUE
# ═══════════════════════════════════════════════════════════════════


def _rasterize_band_worker(
    wkb_list: list[bytes],
    band_height: int,
    width: int,
    transform_tuple: tuple[float, ...],
    all_touched: bool,
) -> np.ndarray:
    """Worker : rasterise une bande horizontale depuis WKB."""
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import Affine
    from shapely import from_wkb

    shapes = []
    for wkb in wkb_list:
        geom = from_wkb(wkb)
        if geom is not None and not geom.is_empty:
            shapes.append((geom, 1))

    if not shapes:
        return np.zeros((band_height, width), dtype=np.uint8)

    t = Affine(*transform_tuple[:6])
    return np.asarray(
        _rasterize(
            shapes,
            out_shape=(band_height, width),
            transform=t,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        ),
        dtype=np.uint8,
    )


def parallel_rasterize_mask(
    geometries: list[Any],
    out_shape: tuple[int, int],
    transform: Any,
    buffer_m: float = 0.0,
    all_touched: bool = True,
    n_workers: int | None = None,
) -> np.ndarray:
    """Rasterise une liste de géométries en masque bool — parallèle par bandes."""
    from rasterio.features import rasterize as _rasterize
    from shapely import STRtree, box as shapely_box, to_wkb

    t0 = time.perf_counter()
    ny, nx = out_shape

    if buffer_m > 0.0:
        import shapely
        geom_arr = np.array(geometries, dtype=object)
        buffered = shapely.buffer(geom_arr, buffer_m)
        geometries = list(buffered)

    valid: list[Any] = [
        g for g in geometries
        if g is not None and not g.is_empty  # type: ignore[union-attr]
    ]
    if not valid:
        logger.info("   Rasterize parallèle : 0 géométries valides")
        return np.zeros((ny, nx), dtype=bool)

    if len(valid) < 5000 or ny < 200:
        shapes = [(g, 1) for g in valid]
        mask: np.ndarray = np.asarray(
            _rasterize(
                shapes,
                out_shape=(ny, nx),
                transform=transform,  # type: ignore[arg-type]
                fill=0,
                dtype=np.uint8,
                all_touched=all_touched,
            ),
        ).astype(bool)
        dt = time.perf_counter() - t0
        logger.info(
            "⚡ rasterize %d géom %dx%d → %.2fs [single-thread]",
            len(valid), nx, ny, dt,
        )
        return mask

    valid_arr = np.array(valid, dtype=object)
    tree = STRtree(valid_arr)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 12)
    n_bands = n_workers

    a = transform.a  # type: ignore[union-attr]
    b = transform.b  # type: ignore[union-attr]
    c = transform.c  # type: ignore[union-attr]
    d = transform.d  # type: ignore[union-attr]
    e = transform.e  # type: ignore[union-attr]
    f = transform.f  # type: ignore[union-attr]

    band_rows = np.array_split(np.arange(ny), n_bands)
    margin = abs(e) * 2

    tasks: list[tuple[list[bytes], int, int, tuple[float, ...], bool]] = []

    for rows in band_rows:
        if len(rows) == 0:
            continue
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        band_h = r1 - r0

        band_ymax = f + r0 * e - margin
        band_ymin = f + r1 * e + margin
        if band_ymin > band_ymax:
            band_ymin, band_ymax = band_ymax, band_ymin
        band_xmin = c - margin
        band_xmax = c + nx * a + margin

        band_box = shapely_box(band_xmin, band_ymin, band_xmax, band_ymax)
        idx = tree.query(band_box)

        if len(idx) == 0:
            tasks.append(([], band_h, nx, (a, b, c, d, e, f + r0 * e), True))
            continue

        band_geoms = [valid[int(i)] for i in idx]
        band_geom_arr = np.array(band_geoms, dtype=object)
        wkb_data = [bytes(w) for w in to_wkb(band_geom_arr)]

        band_transform = (a, b, c, d, e, f + r0 * e)
        tasks.append((wkb_data, band_h, nx, band_transform, all_touched))

    results: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_rasterize_band_worker, *task)
            for task in tasks
        ]
        for fut in futures:
            results.append(fut.result())

    mask_uint8 = np.vstack(results)
    mask_final = mask_uint8.astype(bool)

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ rasterize %d géom %dx%d → %.2fs [%d workers]",
        len(valid), nx, ny, dt, n_workers,
    )
    return mask_final


# ── Cache disque rasters ──

_CACHE_DIR = Path("data/cache")


def _cache_key(
    source_name: str,
    n_geom: int,
    cell_size: float,
    out_shape: tuple[int, int],
) -> str:
    raw = f"{source_name}|{n_geom}|{cell_size}|{out_shape}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def raster_cache_path(
    name: str,
    source_name: str,
    n_geom: int,
    cell_size: float,
    out_shape: tuple[int, int],
) -> Path:
    """Chemin du fichier cache pour un masque rasterisé."""
    key = _cache_key(source_name, n_geom, cell_size, out_shape)
    return _CACHE_DIR / f"{name}_{key}.npy"


def raster_cache_load(path: Path) -> np.ndarray | None:
    """Charge un masque depuis le cache disque. None si absent."""
    if path.exists():
        arr = np.load(path)
        logger.info("⚡ Cache raster : %s (%s)", path.name, arr.shape)
        return np.asarray(arr)
    return None


def raster_cache_save(path: Path, arr: np.ndarray) -> None:
    """Sauvegarde un masque dans le cache disque."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    size_kb = path.stat().st_size / 1024
    logger.debug("   Cache raster sauvé : %s (%.0f KB)", path.name, size_kb)

# ---------------------------------------------------------------------------
# AXE A' — RASTERISATION CATÉGORIELLE PARALLÈLE
# ---------------------------------------------------------------------------


def _rasterize_categorical_band_worker(
    args: tuple[
        list[bytes],
        list[int],
        tuple[int, int],
        tuple[float, float, float, float, float, float],
        bool,
        int,
    ],
) -> np.ndarray:
    """Worker ProcessPool : rasterise une bande horizontale en int16 catégoriel."""
    geom_wkbs, burn_values, band_shape, tf_tuple, all_touched, nodata = args

    if not geom_wkbs:
        return np.full(band_shape, nodata, dtype=np.int16)

    from shapely import wkb as _wkb
    from rasterio.features import rasterize as _rio_rasterize
    from rasterio.transform import Affine

    band_tf = Affine(*tf_tuple)
    shapes = [(_wkb.loads(w), v) for w, v in zip(geom_wkbs, burn_values)]

    result: np.ndarray = np.asarray(
        _rio_rasterize(
            shapes,
            out_shape=band_shape,
            transform=band_tf,
            fill=nodata,
            dtype="int16",
            all_touched=all_touched,
        )
    )
    return result


def parallel_rasterize_categorical(
    geometries: list[Any],
    category_codes: np.ndarray,
    out_shape: tuple[int, int],
    transform: Any,
    all_touched: bool = True,
    n_workers: int | None = None,
    nodata: int = 0,
) -> np.ndarray:
    """
    Rasterise des géométries en raster int16 catégoriel — parallèle par bandes.

    Chaque pixel reçoit le code entier de la géométrie qui le couvre.
    En cas de chevauchement, la dernière géométrie (dans l'ordre fourni) gagne.
    Le caller doit trier par score croissant pour que le meilleur score prévale.

    Parameters
    ----------
    geometries : list[shapely.Geometry]
    category_codes : int array, len == len(geometries)
    out_shape : (ny, nx)
    transform : rasterio.Affine
    all_touched : brûle tous les pixels touchés
    n_workers : parallélisme (défaut = min(cpu_count, 8))
    nodata : valeur pixels non couverts (défaut 0)

    Returns
    -------
    np.ndarray int16, shape == out_shape
    """
    t0 = time.perf_counter()
    ny, nx = out_shape
    category_codes = np.asarray(category_codes, dtype=np.int16)
    n_geom = len(geometries)

    if n_geom == 0:
        logger.info(
            "⏭️  parallel_rasterize_categorical : 0 géom → nodata",
        )
        return np.full(out_shape, nodata, dtype=np.int16)

    assert len(category_codes) == n_geom, (
        f"category_codes ({len(category_codes)}) != geometries ({n_geom})"
    )

    # --- Sérialisation WKB (une seule fois, hors boucle) ---
    geom_wkbs: list[bytes] = [g.wkb for g in geometries]
    codes_list: list[int] = category_codes.tolist()

    # --- STRtree pour filtrage spatial par bande ---
    from shapely import STRtree, box as _shp_box  # type: ignore[attr-defined]

    tree = STRtree(geometries)

    # --- Paramètres transform ---
    a: float = transform.a   # cell_size_x (positif)
    b: float = transform.b   # 0
    c: float = transform.c   # xmin
    d: float = transform.d   # 0
    e: float = transform.e   # -cell_size_y (négatif)
    f: float = transform.f   # ymax

    # --- Découpage en bandes ---
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)
    n_bands = max(n_workers, 1)
    band_height = max(1, (ny + n_bands - 1) // n_bands)

    bands_args: list[
        tuple[
            list[bytes],
            list[int],
            tuple[int, int],
            tuple[float, float, float, float, float, float],
            bool,
            int,
        ]
    ] = []

    for i in range(n_bands):
        y0 = i * band_height
        y1 = min(y0 + band_height, ny)
        if y0 >= ny:
            break
        h = y1 - y0

        # BBox de la bande en coordonnées L93
        band_xmin = c
        band_xmax = c + nx * a
        band_ymax = f + y0 * e      # haut de bande (e < 0)
        band_ymin = f + y1 * e      # bas de bande

        band_box = _shp_box(band_xmin, band_ymin, band_xmax, band_ymax)
        indices = tree.query(band_box)

        # Tri croissant → préserve l'ordre original (last wins = meilleur score)
        indices_sorted = sorted(int(j) for j in indices)
        band_wkbs = [geom_wkbs[j] for j in indices_sorted]
        band_codes = [codes_list[j] for j in indices_sorted]

        band_tf_tuple = (a, b, c, d, e, f + y0 * e)
        bands_args.append((
            band_wkbs,
            band_codes,
            (h, nx),
            band_tf_tuple,
            all_touched,
            nodata,
        ))

    # --- Exécution parallèle ou séquentielle ---
    if n_geom < 500 or len(bands_args) <= 1:
        logger.debug(
            "   rasterize_categorical : séquentiel (%d géom, %d bandes)",
            n_geom, len(bands_args),
        )
        band_results = [
            _rasterize_categorical_band_worker(ba) for ba in bands_args
        ]
    else:
        logger.debug(
            "   rasterize_categorical : parallèle (%d géom, %d bandes, %d workers)",
            n_geom, len(bands_args), n_workers,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            band_results = list(pool.map(
                _rasterize_categorical_band_worker, bands_args,
            ))

    # --- Assemblage ---
    out = np.full(out_shape, nodata, dtype=np.int16)
    row = 0
    for band in band_results:
        h = band.shape[0]
        out[row : row + h, :] = band
        row += h

    dt = time.perf_counter() - t0
    n_covered = int(np.count_nonzero(out != nodata))
    logger.info(
        "✅ parallel_rasterize_categorical : %d géom → %d/%d px (%.2fs, %d bandes)",
        n_geom, n_covered, ny * nx, dt, len(bands_args),
    )
    return out

# ═══════════════════════════════════════════════════════════════════
# AXE B — HOTSPOTS : CONNECTED COMPONENTS + STATS VECTORISÉES
# ═══════════════════════════════════════════════════════════════════


def connected_components(
    mask: np.ndarray,
    structure: np.ndarray | None = None,
    closing_iterations: int = 0,
) -> tuple[np.ndarray, int]:
    """Labeling 8-connexité avec closing optionnel — GPU si dispo."""
    t0 = time.perf_counter()

    if structure is None:
        structure = np.ones((3, 3), dtype=np.int32)

    if _GPU_AVAILABLE:
        assert _cp is not None and _cpx_ndi is not None

        g_mask = _cp.asarray(mask)
        g_struct = _cp.asarray(structure)

        if closing_iterations >= 1:
            g_mask = _cpx_ndi.binary_closing(
                g_mask, structure=g_struct, iterations=closing_iterations,
            )

        _lab_result = _cpx_ndi.label(g_mask, structure=g_struct)
        labeled = _cp.asnumpy(_lab_result[0]).astype(np.int32)  # type: ignore[index]
        n_clusters = int(_lab_result[1])  # type: ignore[index]
        backend = "GPU"
    else:
        from scipy.ndimage import binary_closing as _binary_closing
        from scipy.ndimage import label as _label

        m = np.asarray(mask)
        if closing_iterations >= 1:
            m = np.asarray(
                _binary_closing(
                    m, structure=structure, iterations=closing_iterations,
                ),
            )

        _lab_result = _label(m, structure=structure)
        labeled = np.asarray(_lab_result[0], dtype=np.int32)  # type: ignore[index]
        n_clusters = int(_lab_result[1])  # type: ignore[index]
        backend = "CPU"

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ connected_components %dx%d → %d clusters, %.2fs [%s]",
        mask.shape[1], mask.shape[0], n_clusters, dt, backend,
    )
    return labeled, n_clusters


def vectorized_cluster_stats(
    labeled: np.ndarray,
    n_clusters: int,
    final_score: np.ndarray,
    transform_params: tuple[float, float, float, float, float, float],
    cell_size: float,
    min_cluster_size: int,
    altitude: np.ndarray | None = None,
    slope: np.ndarray | None = None,
) -> dict[str, Any]:
    """Stats vectorisées pour tous les clusters en une passe."""
    t0 = time.perf_counter()
    a, b, c, d, e, f = transform_params

    flat_labels = labeled.ravel().astype(np.int32)
    flat_score = np.where(
        np.isfinite(final_score), final_score, 0.0,
    ).ravel().astype(np.float64)

    ny, nx = labeled.shape

    counts = np.bincount(flat_labels, minlength=n_clusters + 1)

    valid_ids = np.where(counts[1:] >= min_cluster_size)[0] + 1
    n_valid = len(valid_ids)

    if n_valid == 0:
        return {"valid_ids": np.array([], dtype=np.int32), "n_valid": 0}

    score_sum = np.bincount(
        flat_labels, weights=flat_score, minlength=n_clusters + 1,
    )

    valid_counts = counts[valid_ids].astype(np.float64)
    valid_mean_score = score_sum[valid_ids] / valid_counts

    max_score_arr = np.full(n_clusters + 1, -np.inf, dtype=np.float64)
    np.maximum.at(max_score_arr, flat_labels, flat_score)
    valid_max_score = max_score_arr[valid_ids]

    row_idx = np.repeat(np.arange(ny, dtype=np.float64), nx)
    col_idx = np.tile(np.arange(nx, dtype=np.float64), ny)

    row_sum = np.bincount(
        flat_labels, weights=row_idx, minlength=n_clusters + 1,
    )
    col_sum = np.bincount(
        flat_labels, weights=col_idx, minlength=n_clusters + 1,
    )
    mean_row = row_sum[valid_ids] / valid_counts
    mean_col = col_sum[valid_ids] / valid_counts

    x_l93 = c + (mean_col + 0.5) * a
    y_l93 = f + (mean_row + 0.5) * e

    valid_alt: np.ndarray | None = None
    if isinstance(altitude, np.ndarray) and altitude.shape == labeled.shape:
        flat_alt = np.where(
            np.isfinite(altitude), altitude, 0.0,
        ).ravel().astype(np.float64)
        alt_sum = np.bincount(
            flat_labels, weights=flat_alt, minlength=n_clusters + 1,
        )
        valid_alt = alt_sum[valid_ids] / valid_counts

    valid_slope: np.ndarray | None = None
    if isinstance(slope, np.ndarray) and slope.shape == labeled.shape:
        flat_slope = np.where(
            np.isfinite(slope), slope, 0.0,
        ).ravel().astype(np.float64)
        slope_sum = np.bincount(
            flat_labels, weights=flat_slope, minlength=n_clusters + 1,
        )
        valid_slope = slope_sum[valid_ids] / valid_counts

    pad = np.pad(labeled, 1, mode="constant", constant_values=0)
    edge = (
        (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
        | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
        | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
        | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    edge_in_cluster = edge & (labeled > 0)
    flat_edge = edge_in_cluster.ravel().astype(np.float64)
    edge_counts = np.bincount(
        flat_labels, weights=flat_edge, minlength=n_clusters + 1,
    )
    valid_perimeter = edge_counts[valid_ids] * cell_size

    valid_area = valid_counts * cell_size * cell_size
    with np.errstate(divide="ignore", invalid="ignore"):
        valid_compactness = np.where(
            valid_perimeter > 0,
            4.0 * np.pi * valid_area / (valid_perimeter**2),
            0.0,
        )

    dt = time.perf_counter() - t0
    logger.info(
        "⚡ cluster_stats %d valid / %d total → %.2fs",
        n_valid, n_clusters, dt,
    )

    return {
        "valid_ids": valid_ids,
        "n_valid": n_valid,
        "counts": valid_counts.astype(np.int32),
        "mean_score": valid_mean_score.astype(np.float32),
        "max_score": valid_max_score.astype(np.float32),
        "x_l93": x_l93.astype(np.float64),
        "y_l93": y_l93.astype(np.float64),
        "altitude": (
            valid_alt.astype(np.float32) if valid_alt is not None else None
        ),
        "slope": (
            valid_slope.astype(np.float32) if valid_slope is not None else None
        ),
        "perimeter": valid_perimeter.astype(np.float32),
        "area": valid_area.astype(np.float32),
        "compactness": valid_compactness.astype(np.float32),
        "labeled": labeled,
    }


# ═══════════════════════════════════════════════════════════════════
# WARMUP & BENCHMARK
# ═══════════════════════════════════════════════════════════════════


def warmup() -> dict[str, float]:
    """Pré-compile les kernels Numba + chauffe le GPU."""
    logger.info("🔥 Warmup accélérateurs hardware...")
    times: dict[str, float] = {}
    dummy = np.random.rand(256, 256).astype(np.float32)
    dummy64 = dummy.astype(np.float64)

    for name, fn in (
        ("slope_aspect", lambda: compute_slope_aspect(dummy64, 25.0)),
        ("roughness", lambda: compute_roughness(dummy, 7)),
        ("gaussian", lambda: gaussian_filter(dummy, 1.5)),
        ("edt", lambda: distance_transform_edt(dummy > 0.5)),
        ("flow_dir", lambda: compute_flow_dir_d8(dummy64, 25.0)),
    ):
        t0 = time.perf_counter()
        fn()
        times[name] = round(time.perf_counter() - t0, 3)

    if _GPU_AVAILABLE:
        sync_gpu()

    total = sum(times.values())
    logger.info(
        "✅ Warmup terminé en %.2fs — backend=%s | %s",
        total,
        "GPU" if _GPU_AVAILABLE else "CPU",
        " | ".join(f"{k}={v:.3f}s" for k, v in times.items()),
    )
    return times


def benchmark(size: int = 2048) -> dict[str, dict[str, float]]:
    """Benchmark comparatif CPU vs GPU sur grille carrée."""
    logger.info("📊 Benchmark %dx%d...", size, size)
    dem = np.random.rand(size, size).astype(np.float64) * 1000
    mask = (np.random.rand(size, size) > 0.5).astype(np.float32)
    dx = 25.0

    results: dict[str, dict[str, float]] = {}

    ops = [
        ("slope_aspect", lambda: compute_slope_aspect(dem, dx)),
        ("roughness", lambda: compute_roughness(
            dem.astype(np.float32), 7,
        )),
        ("gaussian", lambda: gaussian_filter(mask, 2.0)),
        ("edt", lambda: distance_transform_edt(mask > 0.5)),
        ("flow_dir", lambda: compute_flow_dir_d8(dem, dx)),
        ("reproject", lambda: reproject_l93_to_wgs84(
            mask,
            (857571.0, 6435430.0, 932112.0, 6534209.0),
            (0.001, 0.0, 5.0, 0.0, -0.001, 46.0),
            (size, size),
        )),
    ]

    for name, fn in ops:
        fn()
        if _GPU_AVAILABLE:
            sync_gpu()

        t0 = time.perf_counter()
        fn()
        if _GPU_AVAILABLE:
            sync_gpu()
        dt = time.perf_counter() - t0

        backend = "gpu" if _GPU_AVAILABLE else "cpu"
        results[name] = {backend: round(dt, 4)}
        logger.info("   %-15s : %.4fs [%s]", name, dt, backend)

    return results