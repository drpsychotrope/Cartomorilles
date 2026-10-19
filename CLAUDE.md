# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CARTOMORILLES** is a probabilistic spatial multicriteria model for mapping favorable zones for morel mushroom (*Morchella*) foraging in Isère (France). It combines DEM terrain analysis, forest species data (BD Forêt v2), geology (BDCharm-50), hydrology, and OSM-derived landcover into a weighted score grid at 10–100m resolution.

- **CRS**: EPSG:2154 (Lambert-93), covering Isère department bbox (~14 091 km², 107.4×131.2 km)
- **Default grid**: 20m cells → ~35.2M cells
- **DEM**: 6-tile Copernicus merge (N44-45 × E004-006)
- **Python**: 3.12+

## Running the Model

```bash
pip install -r requirements.txt

# Default run (auto cell-size, default data paths)
python main.py

# Quick development test (25m cells, verbose)
python main.py --cell-size 25 --verbose

# Dry-run: validate data without computing
python main.py --dry-run

# Purge cache then run
python main.py --purge-cache

# Full control
python main.py --cell-size 10 --hotspot-threshold 0.65 --output-dir ./results --export-vector

# Discover available WFS layers
python main.py --discover-wfs
```

Key flags: `--cell-size`, `--hotspot-threshold` (default 0.60), `--smoothing-sigma` (default 1.0), `--urban-buffer`, `--no-urban`, `--no-landcover`, `--landcover`, `--max-hotspots` (default 8), `--seed` (default 42), `--force` (bypass grid size/weight warnings), `--debug-layers` (diagnostic overlays in the HTML map).

Outputs go to `output/`: `carte_morilles.html`, `morilles_probability.tif`, `hotspots.csv`, `cartomorilles.log`, and optionally `morilles_grid.gpkg`.

## Pipeline Architecture

```
data_loader.py          → ingest DEM, shapefiles, WFS/OSM data
    ↓
grid_builder.py         → build spatial grid, compute 11 criteria scores
    ↓
species_enricher.py     → enrich tree species scores (BD Forêt v2 + IFN altitude zones)
    ↓
scoring.py              → weighted aggregation, eliminatory masks, hotspot detection
    ↓
visualize.py            → Folium HTML map, GeoTIFF, GeoPackage, hotspot CSV
```

`_accel.py` provides transparent GPU/CPU dispatch and parallel rasterization — callers never import cupy directly.

`prepare_context.py` auto-generates `AI_CONTEXT.md` files for the session system.

## Key Configuration (`config.py`)

All thresholds, scoring weights, tree species scores, geology scores, and spatial bounds live in `config.py`. The **`WEIGHTS`** dict (11 criteria, sum ≈ 1.0) is the primary tuning surface. Changing it invalidates cached rasters only if the lookup hash changes (MD5-keyed cache — D23).

## Scoring Model

`scoring.py` aggregates 11 criteria via weighted sum, then applies:
1. **Eliminatory masks** (7 boolean): granite/basalt/sandstone, altitude extremes, urban core, slope >45°, impassable species, waterlogged (TWI > threshold) → score = 0
2. **Monotony penalty** — uniform terrain discount
3. **Calc-dry penalty**
4. **Gaussian spatial smoothing** (sigma configurable)
5. **6-class probability classification**: Nul / Très faible / Faible / Moyen / Bon / Excellent (thresholds: 0.15, 0.30, 0.45, 0.60, 0.75)

## Architectural Decisions to Respect

These decisions are documented in `decisions.md` — do not reverse them without understanding the justification:

| ID | Rule |
|---|---|
| **D1** | TWI uses D8 algorithm (not D∞/MFD) |
| **D2** | Geology uses `DESCR` field, not `NOTATION` (0% vs 14.7% unresolved) |
| **D3** | Chestnut score = 0.80, **not eliminatory** (frequent morel post-disturbance) |
| **D4** | `tree_species` is outside `_VEGETATION_CRITERIA` to avoid double-penalty via `green_score` |
| **D5** | `apply_urban_mask()` runs **before** micro-habitat scores |
| **D6** | Species/geology scores use int-encoded rasters + lookup arrays, never string matching in the scoring loop |
| **D8** | BD Forêt cells get floor `0.80` for forest-floor landcover to avoid HSV underscoring |
| **D9** | `dist_water` has a `0.15` floor inside forest polygons (unmapped ephemeral streams) |
| **D12** | `ProcessPoolExecutor` for categorical rasterization (rasterio holds the GIL in Cython) |
| **D13** | Urban masks are disk-cached as `.npy` in `data/cache/` (446k polygons: 27s → 0s) |
| **D22** | Scores burned ascending so last-wins rasterize picks the highest score in overlaps |
| **D30** | Slope resampled from native DEM to grid with `order=0` (nearest-neighbor) to preserve peak values at cliff edges — bilinear interpolation would average extremes below the 45° eliminatory threshold |
| **D31** | Eliminatory slope overlay uses Gaussian σ=0.5 (vs σ=1.5 for other masks) to avoid eroding narrow cliff bands; the weather panel is a section of the single map sidebar (D37) |
| **D26** | Urban density filter uses **global** combined raster, not per-type. Only `batiment` type is density-filtered; area types (`residential`, `route`, etc.) skip filtering |
| **D27** | Raster cache key must include bbox (xmin,ymin,xmax,ymax) to prevent stale cache on extent change |
| **D28** | Landcover `_resample_to_grid()` uses inverse L93→WGS84 mapping with Mercator Y correction (not naive ndizoom) |
| **D29** | Folium overlays are reprojected to **EPSG:3857** (Web Mercator), not EPSG:4326 — Leaflet stretches ImageOverlay linearly in Mercator space, so pixels must be evenly spaced in Mercator Y |
| **D37** | Folium map = one JS-rendered probability layer (per-species score PNG + shared terrain PNG), one collapsible sidebar (species, class legend-as-filter, opacity, top-N numbered hotspots from JSON, layers, weather) — no floating panels, no LayerControl; overlays at grid resolution; diagnostic overlays only with `--debug-layers` |

## Session / Branch Workflow

Active development uses a multi-session system tracked in `.sessions/`:
- Each session has a JSON file with `state`, `focus_files`, `exclude_files`, and the corresponding git branch (`session/<name>`)
- `AI_CONTEXT.md` is the master integrated context; per-session contexts are `AI_CONTEXT_<session>.md`
- When starting work scoped to specific files, check `.sessions/` for active locks and in-progress sessions
- `prepare_context.py` regenerates context files — run it after significant structural changes

## Performance Notes

- **TWI** (~5-8s at 35M cells @20m; ~30s at 169M cells @10m): topologically ordered D8 — cannot be trivially parallelized (D20)
- Grid size: warn at 50M cells (TWI >5 min), hard limit at 200M unless `--force`. Bbox dept élargi @10m = 169M (sous limite mais lourd)
- Avoid adding string matching inside scoring loops — use int-coded raster lookups (D6, D25)
- PNG overlays use `compress_level=1` (parallel speed); data GeoTIFFs use level 6 (D17)
- Cache files live in `data/cache/` — keyed by MD5 of lookup arrays so config changes auto-invalidate (D23)
