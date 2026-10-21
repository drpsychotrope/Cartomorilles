# Morille noire (troisième espèce) — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter la morille noire (`elata`) comme troisième espèce modélisée, avec ses paramètres validés, sans changer les chiffres de la morille grise.

**Architecture:** Tout ce qui est « par espèce » passe déjà par `config.SPECIES_PARAMS` et `species_list` (boucle de scoring D47, exports, carte, résumé). Le travail consiste à ajouter l'entrée `elata` et le mélèze dans `config.py`, à remplacer les trois endroits où « morille » est codé en dur (`occurrences._TAXA`, `species_enricher._INAT_SPECIES`, `visualize._INAT_SPECIES`) par un paramètre d'espèce `taxon`, à faire sauter la validation terrain pour une espèce sans points de contrôle, puis à mesurer.

**Tech Stack:** Python 3.13 (`.venv/Scripts/python.exe`), numpy, geopandas, rasterio, unittest (pas de pytest).

**Spec:** `docs/superpowers/specs/2026-10-21-morille-noire-design.md`

## Global Constraints

- Tests : `unittest`, sans réseau, lancés depuis la racine : `.venv/Scripts/python.exe -m unittest discover -s tests -v` (cwd = `D:/Download/Cartomorilles/.claude/worktrees/nice-rhodes-274c76` ; le venv est celui du checkout principal).
- Run complet : cwd `D:/Download/Cartomorilles`, `D:/Download/Cartomorilles/.venv/Scripts/python.exe <worktree>/main.py --cell-size 50 --bd-foret data/FORMATION_VEGETALE.shp --geology data/geologie_38/GEO050K_HARM_038_S_FGEOL_2154.shp --regions data/rfifn250_l93.shp --output-dir <dossier temporaire>`.
- Fichiers du dépôt en CRLF (autocrlf) : éditer sans changer les fins de ligne.
- Textes utilisateur en français ; noms de clés en ASCII (`meleze`, `elata`).
- Essence inconnue = NaN (D46) ; lookups int-codés (D6) ; rien de codé en dur par espèce hors `config.SPECIES_PARAMS`.
- La grise ne change pas : `esculenta_*`, 11 points de contrôle, paramètres, alias `morilles_probability.tif`.
- Rien n'est poussé ; un commit par tâche sur `claude/enrichissement-ifn-par-espece`.

## Review Focus

- `--species elata` seul : la couche d'observations et le niveau C' s'appliquent, la validation terrain est sautée sans erreur, la carte s'ouvre sur la noire (tests tâches 2, 3, 4, 5).
- Polygone BD Forêt « Mélèze » : essence reconnue dans les trois tables, plus NaN (test tâche 1).
- GBIF hors ligne : la noire réutilise le cache *Morchella* de la grise, pas de seconde requête (test tâche 2).
- Enveloppe d'altitude de la noire : 300 < 500 < 800–1 500 < 1 900 < 2 200, `validate_config()` vrai (test tâche 1).
- Carte avec trois espèces : `layerSpecies.inat` liste les deux morilles, la couche reste cachée pour le cèpe (test tâche 5).

---

### Task 0 : Base de comparaison — fusion D48 et run de référence

**Files:** aucun fichier modifié (fusion puis run).

- [ ] **Step 1 : Fusionner** `refs/heads/claude/zen-dubinsky-f94603` (871c65d, D48 : clé de cache sensible à l'ordre de brûlage, raster morille périmé supprimé) dans la branche de travail, après accord de l'utilisateur ; `decisions.md` : garder D47 puis D48.
- [ ] **Step 2 : Tests** — `discover -s tests` → OK (16 tests attendus après fusion).
- [ ] **Step 3 : Run de base** dans `<scratchpad>/run_base` (Global Constraints), en arrière-plan : c'est la référence « avant » de la tâche 7 (grise et cèpe sur rasters frais). Attendu : morille NaN 1 953 au scoring, Bon ≈ 15,0 %, Excellent ≈ 1,5 %.

### Task 1 : Config — espèce `elata`, mélèze, taxon par espèce

**Files:**
- Modify: `config.py` (tables `_TREE_SCORES_DICT` ~263, `_TREE_ALIASES_DICT` ~312, `TREE_LABELS`, `_TREE_SCORES_EDULIS_DICT` ~1018, `SUPPORTED_SPECIES` 1118, `_SPECIES_PARAMS_DICT` 1124, `validate_config` 1185, `__all__` 81)
- Test: `tests/test_config_species.py`

**Interfaces:**
- Produces: `SUPPORTED_SPECIES == ("esculenta", "elata", "edulis")` ; `SPECIES_PARAMS[sp]["taxon"] : tuple[str, str]` = `("Morchella", "GENUS")` pour `esculenta` et `elata`, `("Boletus edulis", "SPECIES")` pour `edulis` ; `config.species_with_taxon(taxon_name: str) -> tuple[str, ...]` (espèces de `SUPPORTED_SPECIES` dont `taxon[0] == taxon_name`, dans l'ordre) ; clé `meleze` dans les trois tables et `TREE_LABELS["meleze"] == "Mélèze"` ; `esculenta` : `common_name "Morille grise"`, `common_name_plural "morilles grises"`.

- [ ] **Step 1 : Écrire les tests**

```python
# tests/test_config_species.py
class SpeciesTables(unittest.TestCase):
    def test_three_species_validated(self):
        self.assertEqual(config.SUPPORTED_SPECIES, ("esculenta", "elata", "edulis"))
        self.assertTrue(config.validate_config())
    def test_meleze_in_every_table(self):
        self.assertEqual(config.resolve_tree_name("Mélèze"), "meleze")
        self.assertEqual(config.TREE_LABELS["meleze"], "Mélèze")
        for sp, expected in (("esculenta", 0.30), ("elata", 0.85), ("edulis", 0.70)):
            self.assertAlmostEqual(config.get_tree_score("Mélèze", sp), expected)
    def test_tables_share_keys(self):
        ref = config.get_species_params("esculenta")
        for sp in config.SUPPORTED_SPECIES:
            p = config.get_species_params(sp)
            self.assertEqual(set(p["tree_scores"]), set(ref["tree_scores"]), sp)
            self.assertEqual(set(p["geology_scores"]), set(ref["geology_scores"]), sp)
    def test_elata_params(self):
        p = config.get_species_params("elata")
        self.assertEqual(p["common_name"], "Morille noire")
        self.assertEqual(p["altitude_optimal"], (800.0, 1500.0))
        self.assertEqual(p["altitude_range"], (500.0, 1900.0))
        self.assertLess(p["alt_eliminatory_min"], p["altitude_range"][0])
        self.assertGreater(p["alt_eliminatory_max"], p["altitude_range"][1])
        self.assertEqual(p["eliminatory_geology"], frozenset())
        self.assertFalse(p["apply_calcdry"])
        self.assertAlmostEqual(p["tree_scores"]["sapin"], 0.95)
        self.assertAlmostEqual(p["geology_scores"]["granite"], 0.25)
        self.assertTrue(np.isnan(p["tree_scores"]["unknown"]))
    def test_taxon_per_species(self):
        self.assertEqual(config.species_with_taxon("Morchella"), ("esculenta", "elata"))
        self.assertEqual(config.get_species_params("edulis")["taxon"], ("Boletus edulis", "SPECIES"))
        self.assertEqual(config.get_species_params("esculenta")["common_name"], "Morille grise")
```

- [ ] **Step 2 : Lancer, vérifier l'échec** — `.venv/Scripts/python.exe -m unittest tests.test_config_species -v` → `AssertionError` (SUPPORTED_SPECIES à deux espèces), `KeyError: 'taxon'`.

- [ ] **Step 3 : Implémenter dans `config.py`**
  - `_TREE_SCORES_DICT["meleze"] = 0.30` (section conifères), `_TREE_SCORES_EDULIS_DICT["meleze"] = 0.70`, `TREE_LABELS["meleze"] = "Mélèze"`, alias `"larix": "meleze", "larix_decidua": "meleze"`.
  - Nouveaux dicts `_TREE_SCORES_ELATA_DICT` et `_GEOLOGY_SCORES_ELATA_DICT` (valeurs du design, mêmes clés que les tables existantes, `unknown` NaN / 0.35), wrappers `MappingProxyType` `_TREE_SCORES_ELATA`, `_GEOLOGY_SCORES_ELATA`, placés après les wrappers du cèpe.
  - `SUPPORTED_SPECIES = ("esculenta", "elata", "edulis")`.
  - `_SPECIES_PARAMS_DICT` : `esculenta` renommée (« Morille grise », « morilles grises ») et `"taxon": ("Morchella", "GENUS")` ; `edulis` : `"taxon": ("Boletus edulis", "SPECIES")` ; entrée `elata` : `common_name "Morille noire"`, `common_name_plural "morilles noires"`, `scientific_name "Morchella elata (groupe)"`, `tree_scores _TREE_SCORES_ELATA`, `geology_scores _GEOLOGY_SCORES_ELATA`, `eliminatory_geology frozenset()`, `altitude_optimal (800.0, 1500.0)`, `altitude_range (500.0, 1900.0)`, `alt_eliminatory_min 300.0`, `alt_eliminatory_max 2200.0`, `slope_optimal (0.0, 30.0)`, `slope_max 45.0`, `aspect_scores ASPECT_SCORES`, `aspect_ref_deg 180.0`, `twi_optimal TWI_OPTIMAL`, `twi_dry_limit TWI_DRY_LIMIT`, `twi_wet_limit TWI_WET_LIMIT`, `twi_waterlog TWI_WATERLOG`, `apply_calcdry False`, `apply_monotony True`, `"taxon": ("Morchella", "GENUS")`.
  - `def species_with_taxon(taxon_name: str) -> tuple[str, ...]` après `get_species_params` ; ajouter à `__all__`.
  - `validate_config` : chaque espèce a `taxon`, tuple de deux chaînes non vides (erreur sinon).

- [ ] **Step 4 : Lancer tous les tests** — `.venv/Scripts/python.exe -m unittest discover -s tests -v` → OK (les tests D47 restent verts).

- [ ] **Step 5 : Commit** — `git add config.py tests/test_config_species.py` ; message : « Config : morille noire (elata), mélèze reconnu, taxon par espèce ».

### Task 2 : Occurrences — taxon dérivé de la config, cache GBIF partagé

**Files:**
- Modify: `occurrences.py` (`INAT_SPECIES` 66, `_TAXA` 87-90, `occurrences_by_species` 108-131, `fetch_gbif_occurrences` 132-160, `_fetch_species` 440-470, `__all__`)
- Test: `tests/test_occurrences.py`

**Interfaces:**
- Consumes: `config.get_species_params(sp)["taxon"]`, `config.species_with_taxon("Morchella")` (tâche 1).
- Produces: `taxon_for(species: str) -> tuple[str, str]` ; `gbif_cache_path(taxon: tuple[str, str], bbox_wgs84: Mapping[str, float], cache_dir: Path) -> Path` (nom `gbif_{nom du taxon en minuscules, espaces → _}_{md5(nom|rang|bbox)[:8]}.json`, donc identique pour les deux morilles) ; `occurrences_by_species` attache les observations iNaturalist à toute espèce de taxon *Morchella* ; `fetch_gbif_occurrences` ne fait qu'une requête par taxon distinct. `INAT_SPECIES` supprimé de `__all__` (remplacé par `config.species_with_taxon`).

- [ ] **Step 1 : Écrire les tests**

```python
class TaxonPerSpecies(unittest.TestCase):
    def test_taxon_for(self):
        self.assertEqual(occurrences.taxon_for("elata"), ("Morchella", "GENUS"))
        self.assertEqual(occurrences.taxon_for("edulis"), ("Boletus edulis", "SPECIES"))
    def test_cache_path_shared_by_taxon(self):
        a = occurrences.gbif_cache_path(occurrences.taxon_for("esculenta"), config.BBOX_WGS84, Path("c"))
        b = occurrences.gbif_cache_path(occurrences.taxon_for("elata"), config.BBOX_WGS84, Path("c"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, occurrences.gbif_cache_path(occurrences.taxon_for("edulis"), config.BBOX_WGS84, Path("c")))
    def test_inat_attached_to_both_morels(self):
        obs = [{"id": 1, "lat": 45.2, "lng": 5.7, "taxon_name": "Morchella esculenta", "accuracy_m": 10}]
        with mock.patch.object(occurrences, "fetch_gbif_occurrences", return_value={}):
            out = occurrences.occurrences_by_species(["esculenta", "elata", "edulis"], obs)
        self.assertEqual([o["inat_id"] for o in out["esculenta"]], [1])
        self.assertEqual([o["inat_id"] for o in out["elata"]], [1])
        self.assertEqual(out["edulis"], [])
    def test_one_gbif_request_per_taxon(self):
        calls = []
        def fake(session, species, taxon, bbox, cache_dir, titles, *, online):
            calls.append(taxon); return [], online
        with mock.patch.object(occurrences, "_fetch_species", side_effect=fake):
            occurrences.fetch_gbif_occurrences(["esculenta", "elata", "edulis"])
        self.assertEqual(calls, [("Morchella", "GENUS"), ("Boletus edulis", "SPECIES")])
```

- [ ] **Step 2 : Lancer, vérifier l'échec** — `AttributeError: taxon_for`.

- [ ] **Step 3 : Implémenter** — `taxon_for` lit `config.get_species_params(species)["taxon"]` ; `_TAXA` supprimé ; `fetch_gbif_occurrences` groupe les espèces par taxon (`dict[tuple, list[str]]`), appelle `_fetch_species` une fois par taxon et recopie la liste pour chaque espèce du groupe ; `_fetch_species` utilise `gbif_cache_path` ; `occurrences_by_species` : `inat if species in config.species_with_taxon("Morchella") else []`.

- [ ] **Step 4 : Lancer tous les tests** → OK.

- [ ] **Step 5 : Commit** — « Occurrences : taxon par espèce depuis config, cache GBIF partagé ».

### Task 3 : Enrichissement — niveau C' pour toute morille

**Files:**
- Modify: `species_enricher.py` (`_INAT_SPECIES` 55, test `species == _INAT_SPECIES` 778)
- Test: `tests/test_species_enricher.py`

**Interfaces:**
- Consumes: `config.get_species_params(species)["taxon"]`.
- Produces: le niveau C' s'applique si `taxon[0] == _INAT_TAXON` (« Morchella ») ; `_INAT_SPECIES` supprimé.

- [ ] **Step 1 : Ajouter le test** — dans `EnrichGridScoresBySpecies` : `test_morille_noire_gets_inat_level` : `self._enrich("elata")` → `enricher.inat_calls == 1` et cellules inconnues finies, égales à `_expected_mix_score("gresivaudan", "elata")`.
- [ ] **Step 2 : Lancer, vérifier l'échec** — `inat_calls == 0`.
- [ ] **Step 3 : Implémenter** — remplacer la constante par le test du taxon ; commentaire du bloc C' : « toute espèce de taxon Morchella ».
- [ ] **Step 4 : Lancer tous les tests** → OK.
- [ ] **Step 5 : Commit** — « Enrichissement : niveau C' iNaturalist pour toute morille ».

### Task 4 : main.py — validation terrain optionnelle, énumération des espèces

**Files:**
- Modify: `main.py` (`TERRAIN_CHECKPOINTS_BY_SPECIES` 252, `validate_against_terrain` 688-702, phrase de fin 1877-1882)
- Test: `tests/test_main_helpers.py`

**Interfaces:**
- Produces: `TERRAIN_CHECKPOINTS_BY_SPECIES["elata"] == ()` ; `validate_against_terrain(model)` renvoie `None` (log DEBUG « aucun point de contrôle ») quand l'espèce n'a pas de points, sans retomber sur ceux de la grise ; `_enumerate_fr(items: Sequence[str]) -> str` (« a », « a et b », « a, b et c ») ; phrase de fin « 🧺 Bonne chasse aux morilles grises, aux morilles noires et aux cèpes ! ».

- [ ] **Step 1 : Écrire les tests**

```python
class TerrainValidation(unittest.TestCase):
    def test_species_without_checkpoints_skips(self):
        model = SimpleNamespace(species="elata", final_score=np.zeros((2, 2), dtype=np.float32), grid=None)
        self.assertIsNone(main.validate_against_terrain(model))
    def test_checkpoints_registered(self):
        self.assertEqual(main.TERRAIN_CHECKPOINTS_BY_SPECIES["elata"], ())
class Enumerate(unittest.TestCase):
    def test_enumerate_fr(self):
        self.assertEqual(main._enumerate_fr(["a"]), "a")
        self.assertEqual(main._enumerate_fr(["a", "b"]), "a et b")
        self.assertEqual(main._enumerate_fr(["a", "b", "c"]), "a, b et c")
```

- [ ] **Step 2 : Lancer, vérifier l'échec** — `KeyError: 'elata'`, `AttributeError: _enumerate_fr`.
- [ ] **Step 3 : Implémenter** — `TERRAIN_CHECKPOINTS_BY_SPECIES["elata"] = ()` ; dans `validate_against_terrain`, `checkpoints = TERRAIN_CHECKPOINTS_BY_SPECIES.get(model.species, ())` puis `if not checkpoints: logger.debug(...); return None` ; `_enumerate_fr` près de `_fr_num` ; phrase de fin : `logger.info("🧺 Bonne chasse %s !", _enumerate_fr([f"aux {plural}" for ...]))`.
- [ ] **Step 4 : Lancer tous les tests** → OK.
- [ ] **Step 5 : Commit** — « Console : validation terrain optionnelle par espèce, énumération des espèces ».

### Task 5 : Carte — couche d'observations liée aux espèces de taxon Morchella

**Files:**
- Modify: `visualize.py` (`_INAT_SPECIES` 164, `layerSpecies` 1195, JS 1742-1764, `_add_inaturalist_layer` 2170-2175)
- Test: `tests/test_visualize_layers.py`

**Interfaces:**
- Consumes: `config.species_with_taxon("Morchella")`.
- Produces: `MorilleVisualizer.observation_species(species_list: Sequence[str]) -> list[str]` (statique : espèces de la liste dont le taxon est *Morchella*, dans l'ordre) ; `CFG.layerSpecies.inat` = cette liste ; le JS traite `species` comme une liste (`indexOf(current) === -1` → couche cachée) ; `_add_inaturalist_layer` renvoie `None` si la liste est vide.

- [ ] **Step 1 : Écrire le test** — `observation_species(["esculenta", "elata", "edulis"]) == ["esculenta", "elata"]`, `observation_species(["edulis"]) == []`, `observation_species(["elata"]) == ["elata"]`.
- [ ] **Step 2 : Lancer, vérifier l'échec** — `AttributeError`.
- [ ] **Step 3 : Implémenter** — méthode statique ; `"layerSpecies": {"inat": self.observation_species(self.species_list)}` ; JS : normaliser `CFG.layerSpecies[key]` en tableau (`[].concat(...)`) et tester l'appartenance aux deux endroits (1747, 1764) ; `_add_inaturalist_layer` : condition sur la liste ; supprimer `_INAT_SPECIES` ; libellé de la couche inchangé (« Observations iNaturalist de morilles »).
- [ ] **Step 4 : Lancer tous les tests** → OK.
- [ ] **Step 5 : Commit** — « Carte : observations Morchella affichées pour les deux morilles ».

### Task 6 : Outil d'évaluation — maximum de plusieurs couches

**Files:**
- Modify: `tools/evaluate_occurrences.py`
- Test: `tests/test_evaluate_occurrences.py`

**Interfaces:**
- Produces: `combine_scores(scores: Sequence[np.ndarray]) -> np.ndarray` (`np.fmax.reduce`, NaN ignoré sauf partout NaN) ; option CLI `--combine SP [SP ...]` (répétable) : évalue le maximum des couches nommées sous le libellé `"+".join(species)`, avec les occurrences de la première espèce ; refus (`SystemExit`, message) si les taxons diffèrent.

- [ ] **Step 1 : Écrire le test** — `combine_scores([np.array([0.2, np.nan, 0.5]), np.array([0.4, 0.3, np.nan])])` → `[0.4, 0.3, 0.5]` ; `combine_scores([a])` renvoie `a`.
- [ ] **Step 2 : Lancer, vérifier l'échec** — `ImportError`.
- [ ] **Step 3 : Implémenter** — fonction + option ; la combinaison est traitée comme une « espèce » supplémentaire dans la boucle d'impression (même `print_report`).
- [ ] **Step 4 : Lancer tous les tests** → OK.
- [ ] **Step 5 : Commit** — « Évaluation : maximum de plusieurs couches (--combine) ».

### Task 7 : Run complet, mesures, documentation, décision

**Files:**
- Modify: `README.md` (lignes 3, 11, 66, 155, 157), `CLAUDE.md` (lignes 7, 44, 77, 79, tableau des décisions), `decisions.md` (ligne D49 après D47)

- [ ] **Step 1 : Run complet** (Global Constraints) dans `<scratchpad>/run_noire`, puis `grep -c Traceback` sur `cartomorilles.log` → 0 ; exit 0.
- [ ] **Step 2 : Comparer grise et cèpe** avec `<scratchpad>/run_base` (tâche 0) : classes, points de contrôle, NaN au scoring ; toute dérive doit être traçable au mélèze (cellules sans essence des régions à mélèze, au-dessus de 600 m).
- [ ] **Step 3 : Évaluer** — `tools/evaluate_occurrences.py --run <scratchpad>/run_base --run <scratchpad>/run_noire --combine esculenta elata` : AUC, P/E, part en Nul pour `esculenta`, `edulis`, `elata`, `esculenta+elata` ; vérifier que les occurrences de 900–1 625 m ne sont plus en Nul pour `elata`. Relever aussi le score de la noire aux 9 points de contrôle du cèpe (log « Validation croisée » du cèpe : mêmes coordonnées, lues dans `elata_probability.tif`), dont « Belledonne épicéa-pin granite 1500m » et « Chartreuse sapinière 1100m », comme simples lectures pour décider des leviers ensuite.
- [ ] **Step 4 : Documenter** — README et CLAUDE.md (trois espèces, masques par espèce : noire sans géologie éliminatoire, altitude 300–2 200 m, défaut `--species esculenta elata edulis`) ; `decisions.md` : D49 (décision + justification avec les chiffres des étapes 2 et 3 ; D48 est sur la branche cache).
- [ ] **Step 5 : Commit** — « Morille noire : troisième espèce modélisée (D49) », chiffres avant/après dans le message.
