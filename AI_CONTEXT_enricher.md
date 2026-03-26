# 🍄 CARTOMORILLES — AI Context v2.3.5 | 2026-03-26 13:02 UTC | full | session:enricher

> Auto-généré par `prepare_context.py` — optimisé Claude Opus
> Project hash: `5f3a1e67eb`

> 🔀 SESSION PARALLÈLE : **enricher**
> Branche git : `session/enricher`
> Créée le : 2026-03-26T13:02:57.538389+00:00
>
> **Fichiers modifiables (verrouillés) :**
> - 🔒 `species_enricher.py` (exclusif)
> **Lecture seule :**
> - 👁️ `config.py`
>
> ⚠️ Ne modifie QUE les fichiers verrouillés. Pour tout autre fichier → `# TODO:`

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
- SESSION 'enricher' : modifier UNIQUEMENT species_enricher.py
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
- Ne PAS modifier : data_loader.py, grid_builder.py, scoring.py, visualize.py, landcover_detector.py, main.py, diag_enricher.py, explore_geology.py, weather.py
</forbidden>

<style_reference>
Ton style dans ce projet (à maintenir). Extraits auto-détectés :

# Extrait 1: _zoom_dem (grid_builder.py L397) [score: 8.0]
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

# Extrait 2: score_terrain_roughness (grid_builder.py L716) [score: 7.2]
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

# Extrait 3: apply_water_mask (grid_builder.py L1857) [score: 6.5]
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
- np.asarray() en entrée des fonctions de score (68× trouvé)
- np.full_like + masque valid pour NaN-safety (2× trouvé)
- np.clip en sortie des scores [0, 1] (23× trouvé)
- Logger avec messages descriptifs pour chaque étape (255× trouvé)
- Type hints sur toutes les signatures publiques (158× trouvé)
- Immutabilité sur les constantes (MappingProxyType, frozenset, tuple) (49× trouvé)
- from __future__ import annotations en tête (14× trouvé)
- isinstance guard avant accès .shape (30× trouvé)
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
├── config.py                  ✅ (715L) — 🍄 CARTOMORILLES — Configuration du modèle (v2.2.0)
├── data_loader.py             ⏭️ (2316L) — 🍄 CARTOMORILLES — Chargement des données géogra... (hors scope)
├── grid_builder.py            ⏭️ (2154L) — 🍄 CARTOMORILLES — Construction du maillage et c... (hors scope)
├── scoring.py                 ⏭️ (928L) — scoring.py — Modèle multicritère pondéré pour C... (hors scope)
├── visualize.py               ⏭️ (1098L) — visualize.py — Cartomorilles v2.2.0 (hors scope)
├── landcover_detector.py      ⏭️ (1153L) — landcover_detector.py — Détection landcover par... (hors scope)
├── species_enricher.py        ✅ (916L) — species_enricher.py — Enrichissement essences f... ◄ SESSION
├── main.py                    ⏭️ v2.3.5 (1227L) — 🍄 CARTOMORILLES — Modèle de probabilité de prés... (hors scope)
├── prepare_context.py         ✅ (3323L) — prepare_context.py — Générateur de contexte IA ...
├── diag_enricher.py           ⏭️ (243L) — diag_enricher_viz.py — Visualisation de l'enric... (hors scope)
├── explore_geology.py         ⏭️ (101L) — explore_geology.py — Exploration BDCharm-50 Isère. (hors scope)
├── weather.py                 ⏭️ (0L) (hors scope)
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
    ├── FORMATION_VEGETALE_38.shp          (11.7MB)
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

📊 `715L | 504L code | 5fn | 0cls | cbf36e29`

**Fonctions publiques :**
- `resolve_tree_name(raw_name: str | None) → str` — Normalise un nom d'essence vers la forme canonique.
- `get_tree_score(raw_name: str | None) → float` — Retourne le score d'une essence.
- `resolve_geology(raw_code: str | None) → str` — Résout un code géologique BRGM vers la catégorie simplifiée.
- `get_geology_score(raw_code: str | None) → float` — Retourne le score d'un type géologique.
- `validate_config() → bool` — Vérifie la cohérence interne de la configuration.

---

### data_loader.py ⏭️ (hors scope cette session)

### grid_builder.py ⏭️ (hors scope cette session)

### scoring.py ⏭️ (hors scope cette session)

### visualize.py ⏭️ (hors scope cette session)

### landcover_detector.py ⏭️ (hors scope cette session)

### species_enricher.py

_species_enricher.py — Enrichissement essences forestières inconnues._

📊 `916L | 713L code | 0fn | 1cls | 035acf2a`

**class SpeciesEnricher**
  _Enrichit les essences forestières via cascade A→B→C→D._
  Publiques : `load_bd_foret(self) → gpd.GeoDataFrame | None, enrich_grid_scores(self, grid: Any, forest_gdf: Any) → None, get_stats(self, grid: Any) → dict[str, Any]`
  Privées : `_compute_regional_scores, _filter_by_forest_type, _weighted_morel_score, _build_forest_type_grid, _build_region_grid, _rasterize_regions, _heuristic_regions, _load_observations, _apply_observations, _parse_tfv`

**← imports** : config

---

### main.py ⏭️ (hors scope cette session)

### prepare_context.py

_prepare_context.py — Générateur de contexte IA + Gestionnaire de sessions._

📊 `3323L | 2788L code | 16fn | 25cls | 9b8e99b9`

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
  Publiques : `create(self, name: str, focus_files: list[str], description: str, read_only: list[str] | None) → Session, generate_context(self, name: str) → Path, apply_changes(self, session_name: str, filename: str, new_content: str, description: str) → FileChange, merge(self, session_name: str) → MergeResult, merge_all(self) → list[MergeResult], abort(self, session_name: str) → None, status(self, session_name: str) → dict[str, Any], list_sessions(self, state: SessionState | None) → list[Session], get_history(self) → list[dict[str, Any]]`
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
- `cli_session_history(args: argparse.Namespace) → None`
- `main() → None`

**Privées** : `_project_to_json, _log_session_table, _log_locks_table, _build_context_parser, _build_session_parser, _run_context`

---

### diag_enricher.py ⏭️ (hors scope cette session)

### explore_geology.py ⏭️ (hors scope cette session)

### weather.py ⏭️ (hors scope cette session)

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
| `CELL_SIZE` | `20.0` |

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
| `_WEIGHTS_DICT` | `{'geology': 0.2, 'canopy_openness': 0.14, 'tree_species':...` |
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
| `_DISTURBANCE_DICT` | `{'coupe_recente_1_3ans': 0.9, 'chemin_forestier': 0.7, 'c...` |
| `DISTURBANCE_SCORES` | `MappingProxyType(_DISTURBANCE_DICT)` |

## DÉPENDANCES INTER-MODULES

| Module | Importe depuis |
|---|---|
| `data_loader` | `config` |
| `diag_enricher` | `config`, `data_loader`, `grid_builder`, `species_enricher` |
| `explore_geology` | `config` |
| `grid_builder` | `config` |
| `landcover_detector` | `config` |
| `main` | `config`, `data_loader`, `grid_builder`, `landcover_detector`, `scoring`, `species_enricher`, `visualize` |
| `scoring` | `config`, `grid_builder` |
| `species_enricher` | `config` |
| `visualize` | `config`, `scoring` |

**Hub** : `config` (9×), `grid_builder` (3×), `scoring` (2×)

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
| 🟡 | TODO | `prepare_context.py` | 997 | `" |
| 🟡 | TODO | `prepare_context.py` | 1044 | [desc]` dans le code, " |
| 🟡 | TODO | `prepare_context.py` | 1734 | [desc]` mais ne PAS corriger." |

**Total** : 3 (3 TODO)

## STATISTIQUES

**Code** : 12 modules | 14,174L total | 11,287L code | 48 fn | 31 cls
**Data** : 44 fichiers | 415.1 MB
**Santé** : 3 TODOs | 10 décisions verrouillées

<current_focus>
FOCUS DE CETTE SESSION : enricher — species_enricher.py
DESCRIPTION : Refactor species_enricher
HORS SCOPE (ne pas toucher) : data_loader.py, diag_enricher.py, explore_geology.py, grid_builder.py, landcover_detector.py, main.py, scoring.py, visualize.py, weather.py
Si bug détecté hors scope → `# TODO: [desc]` mais ne PAS corriger.
</current_focus>

<checkpoint>
Avant de coder, remplis exactement ce template (pas plus) :

PROJET: Cartomorilles v2.3.5
TÂCHE: ___
FICHIER: ___
DÉPEND DE: ___
INTERDIT: ___
SESSION: enricher
BRANCHE: session/enricher
</checkpoint>

---
_Généré le 2026-03-26 13:02 UTC | 597 lignes | ~1156 lignes restantes pour prompt + code_