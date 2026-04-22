"""Helpers partagés : transformers CRS et exports standardisés.

Centralise :
- Transformers pyproj L93↔WGS84 (single source of truth, évite 6 instanciations)
- Construction dict hotspot enrichi (lat/lon + Google Maps)
- Statistiques numpy standard (min/max/mean/median/std avec NaN)
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from pyproj import Transformer

TO_WGS84: Final[Transformer] = Transformer.from_crs(
    "EPSG:2154", "EPSG:4326", always_xy=True,
)
TO_L93: Final[Transformer] = Transformer.from_crs(
    "EPSG:4326", "EPSG:2154", always_xy=True,
)


def hotspot_to_export_dict(
    h: dict[str, Any], rank: int | None = None,
) -> dict[str, Any]:
    """Enrichit un hotspot avec lat/lon WGS84 arrondis + lien Google Maps."""
    lon, lat = TO_WGS84.transform(h["x_l93"], h["y_l93"])
    out: dict[str, Any] = {
        **h,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "google_maps": f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}",
    }
    if rank is not None:
        out["rank"] = rank
    return out


def compute_array_stats(
    arr: np.ndarray, decimals: int = 4,
) -> dict[str, float]:
    """Retourne min/max/mean/median/std d'un array numpy (ignore NaN)."""
    return {
        "min": round(float(np.nanmin(arr)), decimals),
        "max": round(float(np.nanmax(arr)), decimals),
        "mean": round(float(np.nanmean(arr)), decimals),
        "median": round(float(np.nanmedian(arr)), decimals),
        "std": round(float(np.nanstd(arr)), decimals),
    }


__all__ = [
    "TO_WGS84",
    "TO_L93",
    "hotspot_to_export_dict",
    "compute_array_stats",
]
