# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CARTOMORILLES** is a probabilistic spatial multicriteria model for mapping favorable zones for mushroom foraging in Isère (France) — two species, modelled in one run: morel (*Morchella esculenta*, `esculenta`) and porcini (*Boletus edulis*, `edulis`), with per-species parameters in `config.SPECIES_PARAMS`. It combines DEM terrain analysis, forest species data (BD Forêt v2), geology (BDCharm-50), hydrology, OSM-derived landcover and the day's weather (Open-Meteo) into a weighted score grid.

- **CRS**: EPSG:2154 (Lambert-93), covering Isère department bbox (~14 091 km², 107.4×131.2 km)
- **Grid**: cell size chosen automatically from the bbox area (`config.CELL_SIZE_AUTO_THRESHOLDS`) → **50 m** for the department (~5.6M cells); `--cell-size` overrides it (`config.CELL_SIZE` = 20 m is only the config default, replaced at runtime)
- **DEM**: 6-tile Copernicus merge (N44-45 × E004-006)
- **Python**: 3.12+

## Running the Model

```bash
pip install -r requirements.txt

# Default run (both species, auto cell-size, default data paths, weather J+0)
python main.py

# Same, then open the map in the browser
python main.py --open

# One species only
python main.py --species esculenta

# Quick development test (25m cells, full DEBUG output on screen)
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

Key flags: `--cell-size` (default auto → 50 m), `--species` (default `esculenta edulis`), `--hotspot-threshold` (default 0.60), `--smoothing-sigma` (default 0.5), `--urban-buffer`, `--no-urban`, `--no-landcover`, `--landcover`, `--no-weather`, `--seed` (default 42), `--force` (bypass grid size/weight warnings), `--open` (open the map at the end), `--console-hotspots` (hotspots listed per species in the end-of-run summary, default 5; console only — `--max-hotspots` is its old name), `--map-hotspots` (numbered hotspots shown when the map opens, default 20), `--debug-layers` (diagnostic overlays in the HTML map), `-v/--verbose` (full DEBUG output on screen).

Outputs go to `output/` (`--output-dir`): `cartomorilles.html` (one map, species toggle), `{species}_probability.tif` (+ `morilles_probability.tif`, a copy of the esculenta one), `{species}_hotspots.csv`, `rapport_{species}.json`, `cartomorilles.log`, and optionally `{species}_grid.gpkg` (`--export-vector`). The end-of-run summary lists only the files written by that run.

Console (D41): by default only the `[1/4]…[4/4]` step lines, warnings and the final summary (weather factor, top hotspots and field-check concordance per species, files written, `file://` link to the map). Everything else — module INFO/DEBUG, weights table, per-criterion stats, per-checkpoint validation, weather report — goes to `cartomorilles.log` (or to the screen with `--verbose`). A message the user must see in a normal run is logged by `main.py` (logger `cartomorilles`) or is a WARNING; module progress/stat lines stay at INFO/DEBUG.

## Pipeline Architecture

```
data_loader.py          → ingest DEM, shapefiles, WFS/OSM data
    ↓
grid_builder.py         → build spatial grid, compute 14 criteria scores
    ↓
species_enricher.py     → enrich tree species scores (BD Forêt v2 + IFN altitude zones)
    ↓
scoring.py              → per species: weighted aggregation, eliminatory masks, hotspot detection
    ↓
visualize.py            → Folium HTML map (all species), GeoTIFF, GeoPackage
```

`main.py` runs the four steps, recomputes the 6 species-aware criteria (`grid.recompute_species_scores`) and scores each species in turn, applies the weather factor, writes the per-species CSV/JSON reports and prints the summary.

`_accel.py` provides transparent GPU/CPU dispatch and parallel rasterization — callers never import cupy directly.

`prepare_context.py` auto-generates `AI_CONTEXT.md` files for the session system.

## Key Configuration (`config.py`)

All thresholds, scoring weights, tree species scores, geology scores, and spatial bounds live in `config.py`. The **`WEIGHTS`** dict (14 criteria, sum ≈ 1.0) is the primary tuning surface; `SPECIES_PARAMS` holds the per-species tree/geology scores, altitude/slope/aspect/TWI optima and eliminatory limits. Changing them invalidates cached rasters only if the lookup hash changes (MD5-keyed cache — D23).

## Scoring Model

`scoring.py` aggregates 14 criteria via weighted sum, per species, then applies:
1. **Eliminatory masks** (8 boolean) → score = 0: urban, water bodies, DEM NoData, eliminatory geology (esculenta: granite, gneiss, siliceous; edulis: none), slope > 45°, altitude outside the species range (esculenta 100–1400 m, edulis 200–2200 m), waterlogged (TWI > species threshold), urban proximity (< 15 m)
2. **Monotony penalty** — uniform terrain discount
3. **Calc-dry penalty** (esculenta only)
4. **Gaussian spatial smoothing** (sigma configurable, default 0.5)
5. **Weather factor** (`main.py`): scores × (0.4 + 0.6 × today's Open-Meteo weather score), skipped with `--no-weather` or when the API is unavailable — reported in the console summary and in `rapport_{species}.json` (`config.weather_factor`)
6. **6-class probability classification**: Nul / Très faible / Faible / Moyen / Bon / Excellent (thresholds: 0.15, 0.30, 0.45, 0.60, 0.75)

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
| **D38** | One score scale = the 6 `config.PROBABILITY_*` classes (map legend, weather panel, console stats and the `weather.py` day classes — D44). Probability palette = one-hue amber→brown ramp with monotone lightness on light basemaps, reversed brown→yellow on dark ones (`dark: True` in `config.BASEMAPS`) |
| **D39** | Study area = BDCharm-50 departmental coverage (Isère) ∩ grid: outlined, exterior faded, no score (data PNG code 254) and no hotspot shown outside; model outputs (GeoTIFF, CSV) unchanged |
| **D40** | Map in the field: click/tap → point popup (score, copyable WGS 84 coordinates, “Itinéraire” and Google Maps links, also in hotspot popups); “Me localiser” button (`watchPosition`: centres on the first fix, then follows without recentring); browser zoom allowed (no `maximum-scale` / `user-scalable=no`); ≥ 44 px targets on touch or narrow screens; `{species}_hotspots.gpx` = the top `--map-hotspots` hotspots inside the study area (D39), numbered like the map and the CSV |
| **D41** | Concise console: the console handler shows only the `cartomorilles` logger (main.py) at INFO+ and WARNING+ from every module (`_ConsoleFilter`); module INFO/DEBUG go to `cartomorilles.log`, `--verbose` shows everything. Hotspots are referred to by rank (n°1 = CSV rank = map marker number, as in the marker popup), never by cluster id |
| **D42** | Weather = Open-Meteo `/v1/forecast` (no key, non-commercial use; CC BY 4.0 data, credited in the console report and the map weather panel) at MAP_CENTER, elevation 400 m (middle of `ALTITUDE_OPTIMAL`), 14 past + 7 forecast days. Day score includes soil temperature (~10 cm) and moisture, then a soil-warming factor ×0.5–1 (last 7 days vs the 7 before) that separates spring from autumn; a missing value is ignored, never replaced. Cache `data/openmeteo_cache.json` (1 h); offline = last cache covering today |
| **D44** | Weather days use the map's 6 classes: `ProspectingDay.level` = index in `config.PROBABILITY_LABELS` (rounded score digitized on `PROBABILITY_THRESHOLDS`), `.label` = that label, no emoji; favourable day = Bon or Excellent. The map panel and the console summary read `level` / `label` as they are — no other weather scale |

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
