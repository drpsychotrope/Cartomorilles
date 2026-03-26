# 🍄 CARTOMORILLES — AI Context v2.3.5 | 2026-03-26 18:51 UTC | full

> Auto-généré par `prepare_context.py` — optimisé Claude Opus
> Project hash: `f91709e16f`

<role>
Tu es un expert Python géospatial et mycologue computationnel.
Tu as écrit ce code lors de sessions précédentes.
Maintiens la cohérence avec tes choix antérieurs décrits ci-dessous.
</role>

<rules>
- Code complet uniquement — jamais de fragments
- 0 explication non sollicitée (code speaks for itself)
- Sois assertif — pas de hedging ni de caveats inutiles
- Si doute technique réel → ⚠️ DOUTE: [raison], pas de noyade dans les caveats
- Logger uniquement (`logging.getLogger("cartomorilles.<mod>")`) — jamais print
- Demande confirmation avant de modifier config.py (poids/seuils/éliminatoires)
- Bug hors scope → `# TODO: [desc]` dans le code, ne PAS corriger
- Rien de plus que ce qui est demandé — pas de features spontanées
</rules>

<forbidden>
- [D1] Ne PAS contredire : TWI algorithme D8 (pas D∞/MFD)
- [D2] Ne PAS contredire : DESCR prioritaire sur NOTATION pour géologie
- [D3] Ne PAS contredire : Châtaignier score 0.80, PAS éliminatoire
- [D4] Ne PAS contredire : tree_species hors _VEGETATION_CRITERIA
- [D5] Ne PAS contredire : apply_urban_mask AVANT micro-habitats
- [D6] Ne PAS contredire : Rasters int-codés + lookups pour hotspots
- [D7] Ne PAS contredire : Landcover cache-only (pas de téléchargement)
- [D8] Ne PAS contredire : Forest floor 0.80 pour cellules BD Forêt
- [D9] Ne PAS contredire : dist_water floor 0.15 en forêt
- [D10] Ne PAS contredire : Pénalité couverture NaN-safe floor=0.5

- Ne PAS re-proposer : score_slope seuils renforcés (0,8)/(15)/(25)/(45) (rejeté : Supersédé par TWI)

- Ne PAS utiliser Optional, Dict, List (types legacy)
- Ne PAS utiliser print() au lieu du logger
- Ne PAS produire de code partiel / fragmenté
- Ne PAS ajouter de features non demandées
</forbidden>

<style_reference>
Ton style dans ce projet (à maintenir). Extraits auto-détectés :

# Extrait 1: _zoom_dem (grid_builder.py L402) [score: 8.0]
        def _zoom_dem(self, dem: np.ndarray) -> np.ndarray:
            """Redimensionne le DEM en gérant les NaN."""
            nan_mask = np.isnan(dem)
            zy = self.ny / dem.shape[0]
            zx = self.nx / dem.shape[1]
            if nan_mask.any():
                dem_filled = self._fill_nan_dem(dem)
                dem_zoomed: np.ndarray = np.asarray(
                    zoom(dem_filled, (zy, zx), order=1)
                ).astype(np.float32)
                nan_zoomed: np.ndarray = (
                    np.asarray(
                        zoom(nan_mask.astype(np.float32), (zy, zx), order=0)
                    )
                    > 0.5
                )
                dem_zoomed[nan_zoomed] = np.nan
                return dem_zoomed
            return np.asarray(zoom(dem, (zy, zx), order=1)).astype(np.float32)

# Extrait 2: score_terrain_roughness (grid_builder.py L721) [score: 7.2]
        def score_terrain_roughness(self) -> GridBuilder:
            """Score de rugosité terrain — pénalise les zones accidentées."""
            self._require_terrain()
            _rough = self.roughness
            assert _rough is not None
    
            r_opt = ROUGHNESS_OPTIMAL
            r_max = ROUGHNESS_MAX
    
            score = np.ones_like(_rough, dtype=np.float32)
    
            mask = (_rough > r_opt) & (_rough <= r_max)
            if mask.any():
                t = (_rough[mask] - r_opt) / (r_max - r_opt)
                score[mask] = 1.0 - t**0.8
    
            score[_rough > r_max] = 0.0
    
            score = self._apply_nodata(np.clip(score, 0, 1))
            self.scores["terrain_roughness"] = score
            self.score_confidence["terrain_roughness"] = 0.8
            self._log_score_stats("terrain_roughness", score)
            return self

# Extrait 3: apply_water_mask (grid_builder.py L1920) [score: 6.5]
        def apply_water_mask(self) -> GridBuilder:
            """
            Applique le masque des plans d'eau — Fix #24.
    
            Met les scores à 0 pour les cellules en eau.
            """
            if self.water_mask is not None and self.water_mask.any():
                n = int(self.water_mask.sum())
                for name in self.scores:
                    self.scores[name][self.water_mask] = 0.0
                logger.info(
                    "✅ Masque plans d'eau : %d cellules → score 0", n
                )
            else:
                logger.debug("   Pas de masque plans d'eau")
            return self

Patterns récurrents détectés :
- np.asarray() en entrée des fonctions de score (85× trouvé)
- np.full_like + masque valid pour NaN-safety (2× trouvé)
- np.clip en sortie des scores [0, 1] (41× trouvé)
- Logger avec messages descriptifs pour chaque étape (258× trouvé)
- Type hints sur toutes les signatures publiques (173× trouvé)
- Immutabilité sur les constantes (MappingProxyType, frozenset, tuple) (52× trouvé)
- from __future__ import annotations en tête (13× trouvé)
- isinstance guard avant accès .shape (32× trouvé)
</style_reference>

## IDENTITÉ DU PROJET

| Champ | Valeur |
|---|---|
| Nom | Cartomorilles |
| Objectif | Cartographie probabiliste multicritère des zones favorables aux morilles |
| Emprise | 20×20 km centrée Grenoble |
| CRS | EPSG:2154 (Lambert-93) |
| Centre L93 | (913_100, 6_458_800), rayon 10 km |
| DEM | BD ALTI 25 m, 6000×6000 px |
| Version | 2.3.5 |

**Stack** : Python 3.12+, PIL, fiona, folium, geopandas, matplotlib, numpy, rasterio, requests, scipy, shapely

## ARBORESCENCE & ÉTAT

```
D:\Download\Cartomorilles\
├── config.py                  ✅ (730L) — 🍄 CARTOMORILLES — Configuration du modèle (v2.2.0)
├── data_loader.py             ✅ (2316L) — 🍄 CARTOMORILLES — Chargement des données géogra...
├── grid_builder.py            ✅ (2217L) — 🍄 CARTOMORILLES — Construction du maillage et c...
├── scoring.py                 ✅ (974L) — scoring.py — Modèle multicritère pondéré pour C...
├── visualize.py               ✅ (1386L) — visualize.py — Cartomorilles v2.3.5
├── landcover_detector.py      ✅ (1153L) — landcover_detector.py — Détection landcover par...
├── species_enricher.py        ✅ (916L) — species_enricher.py — Enrichissement essences f...
├── main.py                    ✅ v2.3.5 (1231L) — 🍄 CARTOMORILLES — Modèle de probabilité de prés...
├── prepare_context.py         ✅ (3415L) — prepare_context.py — Générateur de contexte IA ...
├── weather.py                 ✅ (648L) — weather.py — Alertes météo pour prospection mor...
└── data/
    ├── Copernicus_DSM_COG_10_N45_00_E005_00_DEM.tif (41.6MB)
    ├── dem_10f2356d.tif                   (0.6MB)
    ├── dem_1cb931be.tif                   (3.7MB)
    ├── dem_api_cache.tif                  (3.4MB)
    ├── dem_e457400c.tif                   (1.6MB)
    ├── dem_f3ac5551.tif                   (4.2MB)
    ├── dem_final_cache.tif                (3.4MB)
    ├── FORMATION_VEGETALE.cpg             (<0.1MB)
    ├── FORMATION_VEGETALE.dbf             (3.2MB)
    ├── FORMATION_VEGETALE.prj             (<0.1MB)
    ├── FORMATION_VEGETALE.shp             (140.1MB)
    ├── FORMATION_VEGETALE.shx             (0.2MB)
    ├── FORMATION_VEGETALE_38.cpg          (<0.1MB)
    ├── FORMATION_VEGETALE_38.dbf          (0.4MB)
    ├── FORMATION_VEGETALE_38.prj          (<0.1MB)
    ├── FORMATION_VEGETALE_38.shp          (9.7MB)
    ├── FORMATION_VEGETALE_38.shx          (<0.1MB)
    ├── GEO050K_HARM_038_L_DIVERS_2154.dbf (5.1MB)
    ├── GEO050K_HARM_038_L_DIVERS_2154.prj (<0.1MB)
    ├── GEO050K_HARM_038_L_DIVERS_2154.shp (2.8MB)
    ├── GEO050K_HARM_038_L_DIVERS_2154.shx (<0.1MB)
    ├── GEO050K_HARM_038_L_FGEOL_2154.dbf  (19.9MB)
    ├── GEO050K_HARM_038_L_FGEOL_2154.prj  (<0.1MB)
    ├── GEO050K_HARM_038_L_FGEOL_2154.shp  (27.0MB)
    ├── GEO050K_HARM_038_L_FGEOL_2154.shx  (0.5MB)
    ├── GEO050K_HARM_038_L_STRUCT_2154.dbf (3.1MB)
    ├── GEO050K_HARM_038_L_STRUCT_2154.prj (<0.1MB)
    ├── GEO050K_HARM_038_L_STRUCT_2154.shp (2.4MB)
    ├── GEO050K_HARM_038_L_STRUCT_2154.shx (<0.1MB)
    ├── GEO050K_HARM_038_P_DIVERS_2154.dbf (1.4MB)
    ├── GEO050K_HARM_038_P_DIVERS_2154.prj (<0.1MB)
    ├── GEO050K_HARM_038_P_DIVERS_2154.shp (<0.1MB)
    ├── GEO050K_HARM_038_P_DIVERS_2154.shx (<0.1MB)
    ├── GEO050K_HARM_038_P_STRUCT_2154.dbf (2.4MB)
    ├── GEO050K_HARM_038_P_STRUCT_2154.prj (<0.1MB)
    ├── GEO050K_HARM_038_P_STRUCT_2154.shp (0.1MB)
    ├── GEO050K_HARM_038_P_STRUCT_2154.shx (<0.1MB)
    ├── GEO050K_HARM_038_S_FGEOL_2154.dbf  (9.1MB)
    ├── GEO050K_HARM_038_S_FGEOL_2154.prj  (<0.1MB)
    ├── GEO050K_HARM_038_S_FGEOL_2154.shp  (47.1MB)
    ├── GEO050K_HARM_038_S_FGEOL_2154.shx  (0.2MB)
    ├── rfifn250_l93.cpg                   (<0.1MB)
    ├── rfifn250_l93.dbf                   (0.2MB)
    ├── rfifn250_l93.prj                   (<0.1MB)
    ├── rfifn250_l93.shp                   (5.1MB)
    ├── rfifn250_l93.shx                   (<0.1MB)
    ├── weather_cache.json                 (<0.1MB)
    ├── grenoble_bdalti25.tif              (74.5MB)
```

## FLUX DE DONNÉES (PIPELINE)

```
DEM (.tif) + Shapefiles (data/)
    ↓
data_loader.load_dem/forest/geology/hydro/urban()
    ↓
grid_builder.compute_terrain() → altitude, slope, aspect, roughness, TWI
grid_builder.score_*() → 11 scores [0,1]
    ↓                      ↑
species_enricher ──────────┘ (remplace unknown)
    ↓
apply_urban_mask()  ← AVANT micro-habitats
score_canopy/ground/disturbance()
apply_water_mask()
apply_landcover_mask() ← forest floor + green_clip
    ↓
scoring.compute_weighted_score()
scoring.apply_eliminatory_factors() ← 7 masques
scoring.apply_spatial_smoothing()
scoring.classify_probability() → 6 classes
scoring.get_hotspots() → clusters 8-conn
    ↓
visualize → Folium + GeoTIFF + GPKG
```

## RÉSUMÉ FONCTIONNEL PAR MODULE

### config.py

_🍄 CARTOMORILLES — Configuration du modèle (v2.2.0)_

📊 `730L | 509L code | 5fn | 0cls | e498d5f5`

**Fonctions publiques :**
- `resolve_tree_name(raw_name: str | None) → str` — Normalise un nom d'essence vers la forme canonique.
- `get_tree_score(raw_name: str | None) → float` — Retourne le score d'une essence.
- `resolve_geology(raw_code: str | None) → str` — Résout un code géologique BRGM vers la catégorie simplifiée.
- `get_geology_score(raw_code: str | None) → float` — Retourne le score d'un type géologique.
- `validate_config() → bool` — Vérifie la cohérence interne de la configuration.

---

### data_loader.py

_🍄 CARTOMORILLES — Chargement des données géographiques_

📊 `2316L | 1916L code | 6fn | 1cls | 66c5c8bf`

**class DataLoader**
  _Chargeur de données géospatiales pour Cartomorilles._
  Publiques : `clear_cache(self) → int, load_dem(self, filepath: str | None) → dict[str, Any], load_forest(self, filepath: str | None) → gpd.GeoDataFrame | None, load_geology(self, filepath: str | None) → gpd.GeoDataFrame | None, load_hydro(self, filepath: str | None) → gpd.GeoDataFrame | None, load_urban(self, filepath: str | None) → gpd.GeoDataFrame | None, check_network(self, timeout: int) → bool, discover_wfs_layers(self, url: str, keywords: list[str] | None) → list[str]`
  Privées : `_cache_path, _safe_read_cache, _save_cache, _ensure_l93, _wgs84_bbox_str, _read_dem_file, _reproject_dem, _save_dem_cache, _download_copernicus_dem, _download_file, _generate_synthetic_dem, _download_forest_wfs, _download_forest_osm, _normalize_forest, _generate_synthetic_forest, _download_geology_brgm, _normalize_geology, _generate_synthetic_geology, _download_hydro_wfs, _download_hydro_osm, _normalize_hydro, _generate_synthetic_hydro, _download_urban_osm, _generate_synthetic_urban, _read_vector_file`

**Privées** : `_bbox_hash, _overpass_query, _wfs_request, _osm_element_to_polygons, _try_make_valid, _osm_tags_to_essence`

**← imports** : config

---

### grid_builder.py

_🍄 CARTOMORILLES — Construction du maillage et calcul des scores par critère_

📊 `2217L | 1743L code | 0fn | 1cls | 49967560`

**class GridBuilder**
  _Construit le maillage spatial et calcule les scores par critère._
  Publiques : `compute_terrain(self, dem_data: dict[str, Any]) → GridBuilder, score_altitude(self) → GridBuilder, score_slope(self) → GridBuilder, score_terrain_roughness(self) → GridBuilder, score_aspect(self) → GridBuilder, score_twi(self) → GridBuilder, get_twi_raw(self) → np.ndarray | None, score_distance_water(self, hydro_gdf: gpd.GeoDataFrame | None) → GridBuilder, score_tree_species(self, forest_gdf: gpd.GeoDataFrame | None) → GridBuilder, score_geology(self, geology_gdf: gpd.GeoDataFrame | None) → GridBuilder, score_canopy_openness(self, canopy_data: np.ndarray | None) → GridBuilder, score_ground_cover(self) → GridBuilder, score_disturbance(self, disturbance_data: np.ndarray | None) → GridBuilder, score_forest_edge_distance(self) → GridBuilder, score_favorable_density(self, radius_m: float) → GridBuilder, apply_urban_mask(self, urban_gdf: gpd.GeoDataFrame | None, buffer_m: int) → GridBuilder, score_urban_proximity(self) → GridBuilder, apply_water_mask(self) → GridBuilder, apply_landcover_mask(self, landcover_data: dict[str, Any] | None) → GridBuilder, validate_scores(self) → bool, get_score_summary(self) → dict[str, dict[str, float]], get_cell_info(self, ix: int, iy: int) → dict[str, Any]`
  Privées : `_require_terrain, _ensure_l93, _apply_nodata, _fill_nan_dem, _log_score_stats, _rasterize_max, _zoom_dem, _compute_slope_aspect, _compute_roughness, _log_terrain_stats, _compute_twi, _apply_water_type_bonus, _score_from_any_column, _score_geology_from_any_column, _estimate_canopy_from_edges`

**← imports** : config

---

### scoring.py

_scoring.py — Modèle multicritère pondéré pour Cartomorilles._

📊 `974L | 748L code | 0fn | 1cls | 61d77f23`

**class MorilleScoring**
  _Modèle de scoring multicritère pour la probabilité de morilles._
  Publiques : `compute_weighted_score(self) → MorilleScoring, apply_eliminatory_factors(self) → MorilleScoring, apply_spatial_smoothing(self, sigma: float) → MorilleScoring, classify_probability(self) → MorilleScoring, get_hotspots(self, threshold: float, min_cluster_size: int | None, max_hotspots: int | None) → list[dict[str, Any]], get_elimination_stats(self) → dict[str, int], get_model_metadata(self) → dict[str, Any], get_twi_display_data(self) → dict[str, Any]`
  Privées : `_require_step, _compute_global_confidence, _build_eliminatory_species_mask, _build_eliminatory_geology_mask, _apply_soft_transition, _get_dominant_value_in_cluster, _reverse_lookup_score, _estimate_perimeter`

**← imports** : config, grid_builder

---

### visualize.py

_visualize.py — Cartomorilles v2.3.5_

📊 `1386L | 1166L code | 1fn | 1cls | 9647a9d2`

**class MorilleVisualizer**
  _Génère les sorties visuelles à partir d'un MorilleScoring terminé._
  Publiques : `create_folium_map(self, output: str | Path) → Path, export_geotiff(self, output: str | Path) → Path, export_gpkg_grid(self, output: str | Path, threshold: float) → Path`
  Privées : `_validate_model, _l93_to_wgs84, _orient_for_overlay, _reproject_to_wgs84, _render_score_png, _render_mask_png, _png_to_data_uri, _build_data_png_uri, _add_interactive_controls, _add_basemaps, _add_probability_overlay, _add_elimination_layers, _add_hotspot_markers, _add_landmarks, _render_twi_raw_png, _render_twi_score_png, _render_waterlog_png, _add_twi_layers`

**Privées** : `_default_landmarks`

**← imports** : config, scoring

---

### landcover_detector.py

_landcover_detector.py — Détection landcover par tuiles OSM raster._

📊 `1153L | 880L code | 0fn | 1cls | 4be13afd`

**class LandcoverDetector**
  _Détecte les zones naturelles vs urbaines par analyse colorimétrique_
  Publiques : `close(self) → None, detect(self, bbox_wgs84: dict[str, float] | None, bbox_l93: dict[str, float] | None, target_shape: tuple[int, int] | None) → dict[str, Any], set_terrain_grids(self, altitude: np.ndarray | None, slope: np.ndarray | None) → LandcoverDetector, clear_cache(self) → int, detect_from_cache(self, bbox_wgs84: dict[str, float] | None, bbox_l93: dict[str, float] | None, target_shape: tuple[int, int] | None) → dict[str, Any]`
  Privées : `_get_session, _optimal_zoom, _reset_counters, _inc_downloaded, _inc_cached, _inc_failed, _neutral_result, _check_network, _get_tile_coords, _lon_to_tile_x, _lon_to_tile_x_ceil, _lat_to_tile_y, _lat_to_tile_y_ceil, _tile_to_lon, _tile_to_lat, _lat_to_mercator_y, _tiles_to_bounds, _resolve_tile_url, _download_and_assemble, _download_single_tile, _image_to_rgb, _crop_to_bbox_mercator, _classify_colors, _resample_to_grid, _build_masks, _get_altitude_mask, _is_blank_mosaic, _compute_quality_metrics, _save_debug_images, _save_class_image`

**← imports** : config

---

### species_enricher.py

_species_enricher.py — Enrichissement essences forestières inconnues._

📊 `916L | 713L code | 0fn | 1cls | 035acf2a`

**class SpeciesEnricher**
  _Enrichit les essences forestières via cascade A→B→C→D._
  Publiques : `load_bd_foret(self) → gpd.GeoDataFrame | None, enrich_grid_scores(self, grid: Any, forest_gdf: Any) → None, get_stats(self, grid: Any) → dict[str, Any]`
  Privées : `_compute_regional_scores, _filter_by_forest_type, _weighted_morel_score, _build_forest_type_grid, _build_region_grid, _rasterize_regions, _heuristic_regions, _load_observations, _apply_observations, _parse_tfv`

**← imports** : config

---

### main.py (v2.3.5)

_🍄 CARTOMORILLES — Modèle de probabilité de présence de morilles_

📊 `1231L | 907L code | 14fn | 0cls | 8357537c`

**Fonctions publiques :**
- `setup_logging(output_dir: Path, verbose: bool) → None` — Configure le logging : console (INFO ou DEBUG) + fichier ...
- `validate_weights() → bool` — Vérifie la somme des poids et affiche le détail.
- `estimate_grid_size(cell_size: float) → dict[str, Any]` — Estime le nombre de cellules, la surface et la RAM nécess...
- `summarize_data(dem_data: dict[str, Any], forest_gdf: Any, geology_gdf: Any, hydro_gdf: Any, urban_gdf: Any, ...) → None` — Affiche un tableau récapitulatif des données chargées.
- `compute_statistics(final_score: np.ndarray, threshold: float) → dict[str, Any]` — Calcule les statistiques du score final.
- `display_statistics(stats: dict[str, Any]) → None` — Affiche les statistiques en console/log.
- `validate_against_terrain(model: MorilleScoring) → float | None` — Compare le score modèle aux observations de terrain.
- `display_hotspots(hotspots: list[dict[str, Any]], max_display: int) → None` — Affiche les meilleurs hotspots avec coordonnées GPS et li...
- `save_report(output_dir: Path, stats: dict[str, Any], hotspots: list[dict[str, Any]], config_snapshot: dict[str, Any], duration: float, ...) → Path` — Sauvegarde un rapport JSON complet et reproductible.
- `main(args: argparse.Namespace) → int` — Pipeline principal Cartomorilles.
- `build_parser() → argparse.ArgumentParser` — Construit le parser d'arguments CLI.

**Privées** : `_purge_cache, _resolve_data_path, _on_interrupt`

**← imports** : config, data_loader, grid_builder, landcover_detector, scoring, species_enricher, visualize

---

### prepare_context.py

_prepare_context.py — Générateur de contexte IA + Gestionnaire de sessions._

📊 `3415L | 2864L code | 17fn | 25cls | ccbf5324`

**class FunctionInfo**
  _Métadonnées d'une fonction._
  Publiques : `signature_short(self) → str`

**class ClassInfo**
  _Métadonnées d'une classe._
  Publiques : `public_methods(self) → list[FunctionInfo], private_methods(self) → list[FunctionInfo]`

**class ConstantInfo**
  _Métadonnées d'une constante module-level._

**class TodoItem**
  _Un commentaire TODO/FIXME/HACK._

**class StyleSample**
  _Un extrait de code représentatif du style._

**class ModuleInfo**
  _Métadonnées complètes d'un module Python._
  Publiques : `code_lines(self) → int, public_functions(self) → list[FunctionInfo], private_functions(self) → list[FunctionInfo]`

**class DataFileInfo**
  _Métadonnées d'un fichier de données._

**class DecisionInfo**
  _Une décision d'architecture verrouillée._

**class RejectedApproach**
  _Une approche rejetée._

**class ProjectInfo**
  _Métadonnées complètes du projet._

**class SessionState(str, Enum)**

**class LockType(str, Enum)**

**class FileLock**
  _Verrou sur un fichier._

**class FileChange**
  _Un changement appliqué à un fichier pendant une session._

**class Session**
  _Une session de travail parallèle._
  Publiques : `to_dict(self) → dict[str, Any], from_dict(cls, data: dict[str, Any]) → Session`

**class MergeConflict**
  _Un conflit détecté lors de la fusion._

**class MergeResult**
  _Résultat d'une fusion._

**class ModuleParser**
  _Parse un fichier Python via AST et extrait les métadonnées._
  Publiques : `parse(self) → ModuleInfo`
  Privées : `_extract_version, _extract_imports, _extract_functions, _parse_function, _extract_classes, _extract_constants, _extract_todos, _annotation_str, _safe_value_repr`

**class StyleExtractor**
  _Identifie les fonctions les plus représentatives du style du projet._
  Publiques : `extract_best_samples(self, max_samples: int, max_lines: int) → list[StyleSample]`
  Privées : `_score_function`

**class DecisionsParser**
  _Parse DECISIONS.md pour extraire décisions verrouillées_
  Publiques : `parse(self) → tuple[list[DecisionInfo], list[RejectedApproach]]`
  Privées : `_parse_decisions, _parse_rejected`

**class ProjectAnalyzer**
  _Analyse l'ensemble du projet Cartomorilles._
  Publiques : `analyze(self) → ProjectInfo`
  Privées : `_scan_data_files, _build_dependency_graph`

**class ClaudeContextGenerator**
  _Génère AI_CONTEXT.md avec balises Claude-natives._
  Publiques : `generate(self) → str`
  Privées : `_header, _footer, _section_session_header, _section_role, _section_rules, _section_forbidden, _section_style_reference, _detect_global_patterns, _section_project_identity, _section_architecture, _section_data_flow, _section_modules, _module_compact, _module_detailed, _section_constants_dense, _section_dependencies, _section_conventions, _section_decisions, _section_bugs_todos, _section_stats, _section_focus, _section_checkpoint, _add`

**class GitHelper**
  _Interface simplifiée avec git._
  Publiques : `current_branch(self) → str, branch_exists(self, name: str) → bool, create_branch(self, name: str) → None, switch_branch(self, name: str) → None, delete_branch(self, name: str) → None, commit_all(self, message: str) → None, merge_branch(self, branch: str, no_ff: bool) → tuple[bool, str], abort_merge(self) → None, get_diff_files(self, branch_a: str, branch_b: str) → list[str], get_file_hash(self, filepath: str) → str, stash_save(self, message: str) → None, stash_pop(self) → None, has_uncommitted_changes(self) → bool`
  Privées : `_check_git, _run`

**class LockManager**
  _Gère les verrous sur les fichiers._
  Publiques : `acquire_locks(self, session_name: str, exclusive_files: list[str], read_only_files: list[str] | None) → tuple[bool, list[str]], release_locks(self, session_name: str) → int, get_session_locks(self, session_name: str) → list[FileLock], get_all_locks(self) → dict[str, FileLock], check_file_modified_since_lock(self, filename: str) → bool, detect_cross_conflicts(self) → list[MergeConflict]`
  Privées : `_ensure_dirs, _load_locks, _save_locks`

**class SessionManager**
  _Orchestrateur principal des sessions parallèles._
  Publiques : `create(self, name: str, focus_files: list[str], description: str, read_only: list[str] | None) → Session, generate_context(self, name: str) → Path, apply_changes(self, session_name: str, filename: str, new_content: str, description: str) → FileChange, merge(self, session_name: str) → MergeResult, merge_all(self) → list[MergeResult], abort(self, session_name: str) → None, extend(self, session_name: str, new_files: list[str]) → Session, status(self, session_name: str) → dict[str, Any], list_sessions(self, state: SessionState | None) → list[Session], get_history(self) → list[dict[str, Any]]`
  Privées : `_ensure_dirs, _session_file, _load_session, _save_session, _load_history, _append_history, _generate_session_context, _regenerate_main_context`

**Fonctions publiques :**
- `cli_session_create(args: argparse.Namespace) → None`
- `cli_session_list(args: argparse.Namespace) → None`
- `cli_session_status(args: argparse.Namespace) → None`
- `cli_session_context(args: argparse.Namespace) → None`
- `cli_session_apply(args: argparse.Namespace) → None`
- `cli_session_merge(args: argparse.Namespace) → None`
- `cli_session_merge_all(args: argparse.Namespace) → None`
- `cli_session_abort(args: argparse.Namespace) → None`
- `cli_session_extend(args: argparse.Namespace) → None`
- `cli_session_history(args: argparse.Namespace) → None`
- `main() → None`

**Privées** : `_project_to_json, _log_session_table, _log_locks_table, _build_context_parser, _build_session_parser, _run_context`

---

### weather.py

_weather.py — Alertes météo pour prospection morilles (Meteoblue)._

📊 `648L | 510L code | 2fn | 4cls | b9521290`

**class DayForecast**
  _Prévision journalière Meteoblue._

**class BurstSignal**
  _Signal d'explosion de fructification détecté._

**class ProspectingDay**
  _Évaluation d'un jour pour la prospection morilles._

**class WeatherChecker**
  _Check météo locale pour prospection morilles via Meteoblue._
  Publiques : `fetch(self, use_cache: bool) → list[DayForecast], evaluate(self) → list[ProspectingDay], format_report(days: list[ProspectingDay]) → str`
  Privées : `_detect_burst, _score_day, _score_temperature, _score_frost, _score_precip_day, _recent_precipitation, _score_recent_precip, _score_humidity, _score_wind, _score_to_level, _fetch_api, _parse_response, _load_cache, _save_cache`

**Fonctions publiques :**
- `check_weather(api_key: str | None, lat: float | None, lon: float | None) → list[ProspectingDay]` — Fetch + évalue + log le rapport prospection.

**Privées** : `_format_date_fr`

**← imports** : config

---

## CONSTANTES CRITIQUES (config.py)

### Altitude
| Constante | Valeur |
|---|---|
| `ALTITUDE_OPTIMAL` | `(200.0, 600.0)` |
| `ALTITUDE_RANGE` | `(150.0, 900.0)` |
| `ALTITUDE_ALLUVIAL_CENTER` | `350.0` |

### Autres
| Constante | Valeur |
|---|---|
| `CONFIG_VERSION` | `'2.2.0'` |
| `ROUGHNESS_WINDOW` | `7` |
| `ROUGHNESS_OPTIMAL` | `3.0` |
| `ROUGHNESS_MAX` | `12.0` |
| `_ASPECT_SCORES_DICT` | `{'S': 1.0, 'SE': 0.9, 'flat': 0.85, 'SW': 0.8, 'E': 0.7, ...` |
| `ASPECT_SCORES` | `MappingProxyType(_ASPECT_SCORES_DICT)` |
| `_GROUND_COVER_DICT` | `{'litiere_seche': 1.0, 'litiere_humide': 0.6, 'herbe_rase...` |
| `GROUND_COVER_PREFERENCES` | `MappingProxyType(_GROUND_COVER_DICT)` |
| `PHENOLOGY_ENABLED` | `False` |
| `PHENOLOGY_GRADIENT` | `300.0` |
| `PHENOLOGY_BASE_MONTH` | `3` |
| `PHENOLOGY_BASE_ALT` | `200.0` |
| `PROBABILITY_THRESHOLDS` | `(0.15, 0.3, 0.45, 0.6, 0.75)` |
| `PROBABILITY_LABELS` | `('Nul', 'Très faible', 'Faible', 'Moyen', 'Bon', 'Excelle...` |
| `_GEOPF_BASE` | `'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&...` |
| `_BASEMAPS_DICT` | `{'IGN Plan v2': {'tiles': f'{_GEOPF_BASE}&LAYER=GEOGRAPHI...` |
| `BASEMAPS` | `MappingProxyType(_BASEMAPS_DICT)` |
| `DEFAULT_BASEMAP` | `'IGN Plan v2'` |

### Emprise
| Constante | Valeur |
|---|---|
| `_CENTER_X_L93` | `913100.0` |
| `_CENTER_Y_L93` | `6458800.0` |
| `_RADIUS_M` | `10000.0` |
| `BBOX` | `MappingProxyType({'xmin': _CENTER_X_L93 - _RADIUS_M, 'ymi...` |
| `BBOX_WGS84` | `MappingProxyType({'west': 5.58653, 'south': 45.098498, 'e...` |
| `MAP_CENTER` | `MappingProxyType({'lat': 45.1885, 'lon': 5.7245})` |
| `CELL_SIZE` | `5.0` |

### Essences
| Constante | Valeur |
|---|---|
| `_TREE_SCORES_DICT` | `{'frene': 1.0, 'orme': 0.95, 'pommier': 0.9, 'poirier': 0...` |
| `_TREE_ALIASES_DICT` | `{'frêne': 'frene', 'fraxinus': 'frene', 'fraxinus_excelsi...` |
| `TREE_SCORES` | `MappingProxyType(_TREE_SCORES_DICT)` |
| `TREE_ALIASES` | `MappingProxyType(_TREE_ALIASES_DICT)` |

### Géologie
| Constante | Valeur |
|---|---|
| `_GEOLOGY_KEYWORD_MAP` | `(('terres noires', 'marne'), ('couches rousses', 'calcair...` |
| `_GEOLOGY_SCORES_DICT` | `{'calcaire': 0.95, 'calcaire_marneux': 0.9, 'marne': 0.85...` |
| `_GEOLOGY_BRGM_MAP_DICT` | `{'Fz': 'alluvions_recentes', 'Fy': 'alluvions', 'Fx': 'al...` |
| `GEOLOGY_SCORES` | `MappingProxyType(_GEOLOGY_SCORES_DICT)` |
| `GEOLOGY_BRGM_MAP` | `MappingProxyType(_GEOLOGY_BRGM_MAP_DICT)` |
| `ELIMINATORY_GEOLOGY` | `frozenset({'granite', 'gneiss', 'siliceux'})` |

### Hydrographie
| Constante | Valeur |
|---|---|
| `DIST_WATER_OPTIMAL` | `(5.0, 80.0)` |
| `DIST_WATER_GOOD` | `100.0` |
| `DIST_WATER_MODERATE` | `500.0` |
| `DIST_WATER_MAX` | `1000.0` |
| `_WATER_TYPE_BONUS_DICT` | `{'bras_mort': 1.3, 'plan_eau': 1.2, 'canal': 1.1, 'rivier...` |
| `WATER_TYPE_BONUS` | `MappingProxyType(_WATER_TYPE_BONUS_DICT)` |

### Landcover
| Constante | Valeur |
|---|---|
| `LANDCOVER_FOREST_FLOOR` | `0.8` |
| `DIST_WATER_FOREST_FLOOR` | `0.2` |
| `CANOPY_OPTIMAL_OPENNESS` | `0.4` |
| `CANOPY_MIN_OPENNESS` | `0.1` |
| `CANOPY_MAX_OPENNESS` | `0.9` |

### Pente
| Constante | Valeur |
|---|---|
| `SLOPE_OPTIMAL` | `(0.0, 15.0)` |
| `SLOPE_MODERATE` | `30.0` |
| `SLOPE_STEEP` | `40.0` |
| `SLOPE_MAX` | `50.0` |

### Poids
| Constante | Valeur |
|---|---|
| `_WEIGHTS_DICT` | `{'geology': 0.18, 'tree_species': 0.14, 'canopy_openness'...` |
| `WEIGHTS` | `MappingProxyType(_WEIGHTS_DICT)` |

### TWI
| Constante | Valeur |
|---|---|
| `TWI_OPTIMAL` | `(6.0, 10.0)` |
| `TWI_DRY_LIMIT` | `3.0` |
| `TWI_WET_LIMIT` | `14.0` |
| `TWI_WATERLOG` | `18.0` |
| `TWI_DRY_FLOOR` | `0.1` |
| `TWI_WET_FLOOR` | `0.1` |

### Urbain
| Constante | Valeur |
|---|---|
| `DATA_BUFFER` | `500.0` |
| `URBAN_DIST_ELIMINATORY` | `15.0` |
| `URBAN_DIST_PENALTY` | `100.0` |
| `URBAN_DIST_FULL` | `250.0` |
| `URBAN_PROXIMITY_FLOOR` | `0.05` |
| `_DISTURBANCE_DICT` | `{'coupe_recente_1_3ans': 0.9, 'chemin_forestier': 0.7, 'c...` |
| `DISTURBANCE_SCORES` | `MappingProxyType(_DISTURBANCE_DICT)` |

## DÉPENDANCES INTER-MODULES

| Module | Importe depuis |
|---|---|
| `data_loader` | `config` |
| `grid_builder` | `config` |
| `landcover_detector` | `config` |
| `main` | `config`, `data_loader`, `grid_builder`, `landcover_detector`, `scoring`, `species_enricher`, `visualize` |
| `scoring` | `config`, `grid_builder` |
| `species_enricher` | `config` |
| `visualize` | `config`, `scoring` |
| `weather` | `config` |

**Hub** : `config` (8×), `grid_builder` (2×), `scoring` (2×)

## CONVENTIONS PYLANCE (P1-P10)

- **P1** : `from __future__ import annotations` en tête
- **P2** : Types modernes : `dict` pas `Dict`, `str | None` pas `Optional`
- **P3** : Optional → variable locale + assert + type hint
- **P4** : `getattr()` + `isinstance()` pour attributs optionnels
- **P5** : `np.asarray()` autour de rasterize/gaussian/EDT. `np.any()` au lieu de `.any()`
- **P6** : Import direct si sous-module échoue
- **P7** : `# type: ignore[…]` pour faux positifs documentés
- **P8** : Scores dict : isinstance guard avant .shape
- **P9** : Logger `logging.getLogger("cartomorilles.<mod>")` — jamais print
- **P10** : Immutabilité : MappingProxyType, frozenset, tuple

## DÉCISIONS VERROUILLÉES

| # | Décision | Justification |
|---|----------|---------------|
| D1 | TWI algorithme D8 (pas D∞/MFD) | Suffisant pour 25m, complexité moindre |
| D2 | DESCR prioritaire sur NOTATION pour géologie | NOTATION = 14.7% non résolus vs 0% avec DESCR |
| D3 | Châtaignier score 0.80, PAS éliminatoire | Association fréquente morilles post-perturbation |
| D4 | tree_species hors _VEGETATION_CRITERIA | Évite double pénalisation par green_score |
| D5 | apply_urban_mask AVANT micro-habitats | Sinon disturbance recalculé sur zones masquées |
| D6 | Rasters int-codés + lookups pour hotspots | Performance vs string matching sur clusters |
| D7 | Landcover cache-only (pas de téléchargement) | Stabilité, reproductibilité |
| D8 | Forest floor 0.80 pour cellules BD Forêt | Évite sous-notation forêts par landcover HSV |
| D9 | dist_water floor 0.15 en forêt | Cours d'eau temporaires non cartographiés |
| D10 | Pénalité couverture NaN-safe floor=0.5 | Évite score=0 si quelques critères manquants |

**Approches REJETÉES :**
| Proposition | Raison |
|---|---|
| score_slope seuils renforcés (0,8)/(15)/(25)/(45) | Supersédé par TWI |

## BUGS CONNUS & TODOs

| Sév. | Tag | Fichier | L | Description |
|---|---|---|---|---|
| 🟡 | TODO | `config.py` | 125 | grid_builder.py — implémenter score_urban_proximity() |
| 🟡 | TODO | `config.py` | 131 | scoring.py — ajouter urban_proximity < URBAN_DIST_ELIMINATORY |
| 🟡 | TODO | `grid_builder.py` | 1854 | main.py — appeler grid.score_urban_proximity() après apply_urban_mask(... |
| 🟡 | TODO | `prepare_context.py` | 994 | `" |
| 🟡 | TODO | `prepare_context.py` | 1041 | [desc]` dans le code, " |
| 🟡 | TODO | `prepare_context.py` | 1731 | [desc]` mais ne PAS corriger." |

**Total** : 6 (6 TODO)

## STATISTIQUES

**Code** : 10 modules | 14,986L total | 11,956L code | 45 fn | 35 cls
**Data** : 48 fichiers | 413.4 MB
**Santé** : 6 TODOs | 10 décisions verrouillées

<current_focus>
FOCUS : [à définir dans ton premier message]
HORS SCOPE : [aucun exclusion définie — préciser si nécessaire]
Si bug détecté hors scope → `# TODO: [desc]` mais ne PAS corriger.
</current_focus>

<checkpoint>
Avant de coder, remplis exactement ce template (pas plus) :

PROJET: Cartomorilles v2.3.5
TÂCHE: ___
FICHIER: ___
DÉPEND DE: ___
INTERDIT: ___
</checkpoint>

---
_Généré le 2026-03-26 18:51 UTC | 705 lignes | ~1048 lignes restantes pour prompt + code_