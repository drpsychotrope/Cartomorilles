# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CARTOMORILLES** is a probabilistic spatial multicriteria model for mapping favorable zones for mushroom foraging in Isère (France) — three species, modelled in one run: grey morel (*Morchella esculenta*, `esculenta`), black morel (*Morchella elata* group, `elata`) and porcini (*Boletus edulis*, `edulis`), with per-species parameters in `config.SPECIES_PARAMS`. It combines DEM terrain analysis, forest species data (BD Forêt v2), geology (BDCharm-50), hydrology, recent forest-cover loss (Hansen Global Forest Change — D64), OSM-derived landcover and the day's weather (Open-Meteo) into a weighted score grid.

- **CRS**: EPSG:2154 (Lambert-93), covering Isère department bbox (~14 091 km², 107.4×131.2 km)
- **Grid**: cell size chosen automatically from the bbox area (`config.CELL_SIZE_AUTO_THRESHOLDS`) → **50 m** for the department (~5.6M cells); `--cell-size` overrides it (`config.CELL_SIZE` = 20 m is only the config default, replaced at runtime)
- **DEM**: IGN RGE ALTI® (bare-earth, from the Géoplateforme WMS, 5 m averaged to 10 m — D63); Copernicus GLO-30 DSM (6-tile merge N44-45 × E004-006) as fallback
- **Python**: 3.12+

## Running the Model

```bash
pip install -r requirements.txt

# Default run (all three species, auto cell-size, default data paths, weather J+0)
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

Key flags: `--cell-size` (default auto → 50 m), `--species` (default `esculenta elata edulis`), `--hotspot-threshold` (default: each species' Bon threshold, i.e. its top 10 % scored cells of the study area — D52), `--smoothing-sigma` (default 0.5), `--urban-buffer`, `--no-urban`, `--no-landcover`, `--landcover`, `--no-weather`, `--seed` (default 42), `--force` (bypass grid size/weight warnings), `--open` (open the map at the end), `--hd-map` (also write `cartomorilles_hd.html`, images up to 64 Mpx for desktop — D62), `--console-hotspots` (hotspots listed per species in the end-of-run summary, default 5; console only — `--max-hotspots` is its old name), `--map-hotspots` (numbered hotspots shown when the map opens, default 20), `--debug-layers` (diagnostic overlays in the HTML map), `-v/--verbose` (full DEBUG output on screen).

Outputs go to `output/` (`--output-dir`): `cartomorilles.html` (one map, species toggle; + `cartomorilles_hd.html` with `--hd-map`), `{species}_probability.tif` (+ `morilles_probability.tif`, a hard link to the esculenta one), `{species}_hotspots.csv`, `rapport_{species}.json`, `cartomorilles.log`, and optionally `{species}_grid.gpkg` (`--export-vector`). The end-of-run summary lists only the files written by that run.

Console (D41): by default only the `[1/4]…[4/4]` step lines, warnings and the final summary (weather factor, top hotspots and field-check concordance per species, files written, `file://` link to the map). Everything else — module INFO/DEBUG, weights table, per-criterion stats, per-checkpoint validation, weather report — goes to `cartomorilles.log` (or to the screen with `--verbose`). A message the user must see in a normal run is logged by `main.py` (logger `cartomorilles`) or is a WARNING; module progress/stat lines stay at INFO/DEBUG.

## Pipeline Architecture

```
data_loader.py          → ingest DEM, shapefiles, WFS/OSM data, Hansen forest loss
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

All thresholds, scoring weights, tree species scores, geology scores, and spatial bounds live in `config.py`. The **`WEIGHTS`** dict (14 criteria, sum ≈ 1.0) is the primary tuning surface; `SPECIES_PARAMS` holds the per-species tree/geology scores, altitude/slope/aspect/TWI optima and eliminatory limits. Changing them invalidates the cached categorical rasters when the lookup hash changes (MD5-keyed cache — D23); a change of polygon burn order invalidates them too (D48).

## Scoring Model

`scoring.py` aggregates 14 criteria via weighted sum, per species, then applies:
1. **Eliminatory masks** (8 boolean, `host` for edulis only) → score = 0: urban (buildings; on fine grids gaps < 80 m between them are closed — D56), water and banks (a cell is eliminated when water plus a 5 m bank covers half of it: BD TOPO water surfaces, IGN hydro streams assumed 4 m wide, BDCharm-50 `hydrographie` lakes — D54, D55), DEM NoData, eliminatory geology (esculenta: granite, gneiss, siliceous; elata and edulis: none), no host tree within 30 m (edulis only — D57), slope > 45°, altitude outside the species range (esculenta 100–1400 m, elata 300–2200 m, edulis 200–2200 m), waterlogged (TWI > species threshold; flow routed on a stream-burned, pit-filled DEM — D61); urban proximity is a criterion only (D61)
2. **Monotony penalty** — uniform terrain discount
3. **Calc-dry penalty** (esculenta only)
4. **Gaussian spatial smoothing** (sigma configurable, default 0.5)
5. **6-class probability classification, by rank** (D52): Nul = score < 0.15 (eliminated); the other five classes split the species' scored cells of the study area at the `config.CLASS_RANK_QUANTILES` (Très faible 40 %, Faible 30 %, Moyen 20 %, Bon 8 %, Excellent 2 %), so thresholds differ per species and per run (`MorilleScoring.class_thresholds`). The fixed `config.PROBABILITY_THRESHOLDS` (0.15, 0.30, 0.45, 0.60, 0.75) remain the weather day scale (D44) and the fallback under 100 scored cells

Today's Open-Meteo weather is **never multiplied into the score** (D45): the map shows the terrain potential; the weather (today's class and score, 7-day panel) is reported separately in the console summary, the map sidebar and `rapport_{species}.json` (`config.weather_today`; `--no-weather` skips the API call).

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
| **D38** | One score scale = the 6 `config.PROBABILITY_*` classes (map legend, weather panel, console stats and the `weather.py` day classes — D44). Probability palette: see D45 (the amber→brown ramp and its dark-basemap variant are gone) |
| **D39** | Study area = BDCharm-50 departmental coverage (Isère) ∩ grid: outlined, exterior faded, no score (data PNG code 254) and no hotspot shown outside; model outputs (GeoTIFF, CSV) unchanged |
| **D40** | Map in the field: click/tap → point popup (score, copyable WGS 84 coordinates, “Itinéraire” and Google Maps links, also in hotspot popups); “Me localiser” button (`watchPosition`: centres on the first fix, then follows without recentring); browser zoom allowed (no `maximum-scale` / `user-scalable=no`); ≥ 44 px targets on touch or narrow screens; `{species}_hotspots.gpx` = the top `--map-hotspots` hotspots inside the study area (D39), numbered like the map and the CSV |
| **D41** | Concise console: the console handler shows only the `cartomorilles` logger (main.py) at INFO+ and WARNING+ from every module (`_ConsoleFilter`); module INFO/DEBUG go to `cartomorilles.log`, `--verbose` shows everything. Hotspots are referred to by rank (n°1 = CSV rank = map marker number, as in the marker popup), never by cluster id |
| **D42** | Weather = Open-Meteo `/v1/forecast` (no key, non-commercial use; CC BY 4.0 data, credited in the console report and the map weather panel) at MAP_CENTER, elevation 400 m (middle of `ALTITUDE_OPTIMAL`), 14 past + 7 forecast days. Day score includes soil temperature (~10 cm) and moisture, then a soil-warming factor ×0.5–1 (last 7 days vs the 7 before) that separates spring from autumn; a missing value is ignored, never replaced. Cache `data/openmeteo_cache.json` (1 h); offline = last cache covering today |
| **D46** | Unknown tree species (`tree_scores["unknown"]`, both species) = NaN: excluded from the weighted mean (D10 coverage penalty), never scored as a mediocre species; unknown polygons are burned first so a known species wins overlaps (D22); the IFN enrichment skips unscored species. Raising weights or lowering the Excellent threshold were measured and rejected |
| **D45** | Map classes are computed on the raw terrain score: today's weather is shown in the sidebar, console and JSON report but never multiplied into `final_score` (the ×0.4–1.0 factor of D42 is gone; in October it hid every Bon/Excellent cell). Probability palette = red→yellow→green (ColorBrewer RdYlGn, `_CMAP_STOPS` in `visualize.py`), one palette for all basemaps |
| **D44** | Weather days use the map's 6 classes: `ProspectingDay.level` = index in `config.PROBABILITY_LABELS` (rounded score digitized on `PROBABILITY_THRESHOLDS`), `.label` = that label, no emoji; favourable day = Bon or Excellent. The map panel and the console summary read `level` / `label` as they are — no other weather scale |
| **D48** | Categorical raster cache key (`tree_species`, `geology_cat`) = MD5 of the lookup (D23) **and** of the code sequence in burn order (`_categorical_cache_key`), plus the bbox (D27): a change of polygon sort order (e.g. `na_position="first"`, D46) invalidates the cache by itself — never salt the key by hand. `tests/test_grid_builder_cache.py` |
| **D49** | Third modelled species `elata` (black morel, *Morchella elata* group): own tree and geology tables (fir/spruce 0.95, larch 0.85, acid rocks 0.25 and not eliminatory), altitude optimum 800–1500 m (range 500–1900, eliminatory 300 / 2200), no calc-dry penalty; `esculenta` is labelled "Morille grise"; larch (`meleze`) is scored in all three tables; each species carries its naturalist `taxon` (both morels share *Morchella*: GBIF cache, iNaturalist observations and the map's observation layer; the C' score boost was removed in D60); a species without terrain checkpoints skips the validation instead of borrowing another species' points |
| **D50** | A hotspot's x_l93/y_l93 (so lat/lon, Google Maps link, GPX waypoint and map marker) is the centre of its max-score cell (`_accel.vectorized_cluster_stats`; tie → closest to the centroid), never the centroid, which falls outside an elongated or L-shaped cluster, often on an eliminated cell; the centroid stays in the stats dict only |
| **D52** | Map classes by rank, per species: Nul = score < 0.15, then quantiles 0.40 / 0.70 / 0.90 / 0.98 of the species' scored cells inside the study area (`scoring.rank_class_thresholds`, `set_rank_thresholds` before `classify_probability`). The same thresholds drive the map legend and LUT (palette stops placed on them, each class keeps its colour), popups, CSV/GPX/console, the JSON report (`class_thresholds`, `classes_study_area`) and `tools/evaluate_occurrences.py`; default hotspot threshold = the species' Bon threshold. `PROBABILITY_THRESHOLDS` stays the weather scale (D44) — amends D38 |
| **D53** | "Calques supplémentaires": collapsible block of the sidebar's Couches section, one layer at a time above the probability (own opacity, legend with study-area share or graduated ramp, value under the cursor and in the point popup): geology categories, BD Forêt v2 `ESSENCE` as published (rasterized in visualize.py like the model, generic classes burned first), TWI, slope (terrain PNG red channel) and D8 drained area (`grid.drained_area_log10`, max-pooled before nearest downsampling). One 8-bit code PNG per layer (255 = none or outside the study area), decoded on first display (`data-src`) and coloured in JS; raw TWI left the `--debug-layers` overlays — amends D37 |
| **D54** | The IGN hydro layer has lines only, so lakes were scored as land (Paladru 0.52–0.56): `apply_water_mask` adds the BDCharm-50 `hydrographie` cells (lakes, reservoirs; `_geology_water_cells`) to `grid.water_mask`. Streams keep their 2 m buffer, all_touched rasterization — at 50 m that excludes a 50–100 m band along every stream, which the field checkpoints "Berge Vence" and "Isère lit majeur" rely on |
| **D55** | Water-edge mask: eliminated = cells at least half covered by water plus a `config.WATER_BANK_EXCLUSION_M` = 5 m bank (user's choice, not the 50 m cell); coverage measured on ≤ 10 m sub-cells (`GridBuilder._cover_fraction`). Water = BD TOPO `surface_hydrographique` polygons (lakes, reservoirs, wide river beds; `DataLoader.load_water_surfaces`, paged WFS, cache `data/hydro_surf_*.gpkg`, glaciers dropped) + IGN hydro stream lines buffered by `config.STREAM_HALF_WIDTH_M` = 2 m. At 50 m a stream (2 × 7 m band) covers ~30 % of the cell it crosses: that cell stays scored as a bank, at the mean distance of its non-excluded part for `dist_water`; a wide river and its banks are eliminated |
| **D56** | Urban mask on fine grids: the urban layer holds buildings only, so streets, squares and courtyards were scored at 20 m (Grenoble centre ~18 % of cells). `GridBuilder._close_urban_gaps` closes gaps narrower than 2 × `config.URBAN_GAP_CLOSE_M` = 80 m (disk radius floor(40 m / cell): nothing changes at 50 m), outer edges unchanged, never removes a cell. `_accel.parallel_rasterize_mask` band query box fixed (it was shrunk by 2 cells instead of grown: buildings lying in band edge rows were dropped). Urban cache key `…_v2_ferm40` |
| **D57** | Porcini needs a host tree: cells farther than `SPECIES_PARAMS["edulis"]["host_tree_max_dist_m"]` = 30 m from a BD Forêt stand (landes/pelouses `NC` and `NR` excluded; `GridBuilder.compute_host_forest`) are eliminated (factor `host`). Morels have no such rule (gardens, orchards, edges) |
| **D58** | The porcini checkpoints (literature sites with round coordinates) were moved to the nearest cell matching their description (published BD Forêt species, altitude band, geology, slope ≤ 30°, study area), chosen without looking at model scores; old coordinates kept in comments in main.py. The grey-morel checkpoints are field data and stay untouched |
| **D59** | IGN WFS downloads (forest, hydro) are paged (`_download_wfs_paged`, the service caps a response at 5 000 features); `main._resolve_data_path` also searches `data/` in the current directory |
| **D60** | Coherence audit, batch 1: the hotspot closing only links hot cells, never crosses eliminated cells and adds nothing to cluster stats (`connected_components(barrier=…)`); CPU aspect fixed (was 180° off the GPU); DEM→grid zoom edge-aligned (`grid_mode=True`); every slope segment keeps ≥ 5° (`config.slope_breaks`, elata no longer jumps at 30°); step 2 uses the first species' tables and `GridBuilder.recompute_habitat_scores` refreshes canopy/ground cover per species; per-species GPKG and strict JSON (NaN → null); checkpoints outside the grid skipped; DEM NoData stays NaN; soft transition off (`SOFT_ELIM_BUFFER` = 0); iNaturalist C' boost removed from the score (map layer only, Verpa dropped, obscured points marked); urban cache key includes buffers/density, buffer = floor(buffer/cell); data PNG G channel = exact class; occurrences one per cell, Boyce with average ranks; CPU filters match scipy. `tests/test_coherence_d60.py` |
| **D61** | Coherence audit, batch 2 (measured): `forest_mask` excludes BD Forêt landes and grassland (ESSENCE `NC`; `NR` clear-cuts stay forest); the D9 dist_water floor applies on every grid (`apply_forest_water_floor`) and D8 reads `forest_mask`; TWI routes D8 flow on a pit-filled DEM (Priority-Flood + ε, `_twi_numba._priority_flood`; slope from the raw DEM); urban proximity is no longer eliminatory; landcover urban mask merged right after `apply_urban_mask`; complete OSM urban layer (buildings, built-up landuse, parkings, sport, main roads, railways) downloaded by tiles with per-tile cache and a User-Agent, never cached when incomplete (`urban_v2_*`), roads added after the D56 gap closing; `--dem` L93 files warped onto the bbox; IGN streams and BD TOPO water surfaces burned 20 m into the TWI routing DEM (`compute_terrain(streams=…)`). `tests/test_forest_mask_d61.py` |
| **D62** | Map pixel budget: `visualize.render_size` keeps the grid resolution down to `MOBILE_MAX_PIXELS` = 4096² per image (iOS Safari canvas limit); `--hd-map` also writes `cartomorilles_hd.html` up to `HD_MAX_PIXELS` = 64 Mpx (~15 m, desktop). Map layers are canvases shown by Leaflet (`CanvasOverlay`, putImageData — no PNG re-encoding), decoded images freed after reading; data PNG G channel = 0 or 1 + exact class where the truncated score misclassifies; GeoTIFFs deflate 6 + float predictor, tiled; `morilles_probability.tif` is a hard link; per-species criteria and factor masks released after scoring unless `--export-vector`/`--debug-layers`; D56 closing via two EDTs |
| **D63** | DEM = IGN RGE ALTI® (bare-earth terrain model, Etalab 2.0) read from the Géoplateforme raster WMS (`ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES`, BIL float32) in 25 km tiles requested at 5 m and averaged 2×2 onto the bbox grid at `config.DEM_RESOLUTION_M` = 10 m (`DataLoader._download_rgealti_dem`, per-tile resume cache, cache `data/dem_rgealti10m_*.tif`); slope/aspect/roughness/TWI computed at 10 m whatever the cell size; Copernicus DSM fallback (`DEM_SOURCE`). Grey-morel checkpoints 7/11 → 10/11. `tests/test_rgealti.py` |
| **D64** | Recent forest-cover loss (Hansen Global Forest Change v1.13, CC BY 4.0; loss year 2001–2025, ~20×28 m pixels; tile `50N_000E` downloaded once, cropped pixel-aligned, cache `data/forest_loss_GFC-2025-v1.13_*.tif` — `DataLoader.load_forest_loss`) feeds the disturbance criterion per species: age = reference year (`FOREST_LOSS_REFERENCE_YEAR`, current year by default, written in the JSON report) − loss year; weight 1 at 1–4 years, then 0.8…0.2 up to 8 years; morels (`recent_loss` = "boost") max(disturbance, weight), porcini ("penalty") disturbance × (1 − weight) — a penalty, never a host elimination (a 2022 porcini lies 40–60 m inside a 2020 Hansen loss: partial cuts). Grid: nearest, or the cell's latest loss year above 30 m (`GridBuilder.set_forest_loss`); step-2 disturbance kept in `_disturbance_base`, applied by `recompute_habitat_scores(species)`. Map layer "Coupes et chablis récents". `tests/test_forest_loss.py` |

## Session / Branch Workflow

Active development uses a multi-session system tracked in `.sessions/`:
- Each session has a JSON file with `state`, `focus_files`, `exclude_files`, and the corresponding git branch (`session/<name>`)
- `AI_CONTEXT.md` is the master integrated context; per-session contexts are `AI_CONTEXT_<session>.md`
- When starting work scoped to specific files, check `.sessions/` for active locks and in-progress sessions
- `prepare_context.py` regenerates context files — run it after significant structural changes

## Performance Notes

- **TWI** is computed on the native DEM (the 20 m cache, whatever `--cell-size`): pit filling ~7 s + D8 ~5-8 s on 35M pixels; topologically ordered — cannot be trivially parallelized (D20)
- Grid size: warn at 50M cells with the peak-memory estimate (~7.3 GB + 226 bytes per cell, measured with three species and `--hd-map`: 39.0 GB / 10.4 min at 10 m with the RGE ALTI terrain and the Hansen losses), hard limit at 200M unless `--force`
- Avoid adding string matching inside scoring loops — use int-coded raster lookups (D6, D25)
- PNG overlays use `compress_level=1` (parallel speed); data GeoTIFFs use level 6 (D17)
- Cache files live in `data/cache/` — categorical rasters keyed by MD5 of the lookup array and of the burn order, so config and sort-order changes auto-invalidate (D23, D48)
