# Score Uplift v2.5.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push max score from 0.7185 to ~0.78 and forest mean from 0.32 to ~0.40 by retuning tree `unknown`, calcareous geology band, landcover floor, and adoucissant les penalty floors — without inflation aveugle.

**Architecture:** Pure constants edits in `config.py` and `scoring.py`. No pipeline changes. Validation by running model and comparing stats vs baseline. Append D32–D34 to `decisions.md`.

**Tech Stack:** Python 3.12, numpy, rasterio (model). Standalone script run validates.

**Spec reference:** `docs/superpowers/specs/2026-05-06-score-uplift-design.md`

---

## File Structure

| File | Change |
|---|---|
| `config.py` | Bump `CONFIG_VERSION`, edit `_TREE_SCORES_DICT["unknown"]`, edit `_GEOLOGY_SCORES_DICT` calcareous band, edit `LANDCOVER_FOREST_FLOOR`, add v2.5.2 changelog comment |
| `scoring.py` | Edit `_COVERAGE_PENALTY_FLOOR` constant, edit `_CALCDRY_FLOOR` and `_CALCDRY_MAX_PENALTY` literals inside `apply_calcdry_penalty` |
| `decisions.md` | Append D32, D33, D34 rows |

No new files. No tests added (pure constants tuning, validated via run-and-compare).

---

## Baseline Stats (for comparison)

From `output/rapport_cartomorilles.json` run 2026-05-06 11:27 (cell=10m, dept élargi) :
- `score_max` = 0.7185
- `score_mean` = 0.3176
- `score_median` = 0.4001
- `n_high` (≥0.7) = 537 cells (0.0%)
- `n_hotspots` (≥0.6) = 8783
- Top hotspot mean = 0.67 (chataignier sur calcaire)
- Concordance Vouillants : modèle 0.52 vs attendu 0.70 (Δ=0.18)

Targets after uplift :
- `score_max` ≥ 0.75
- `score_mean` ≥ 0.38
- Concordance terrain ≥ 90% (currently 90.9%)
- Vouillants Δ ≤ 0.10

---

### Task 1: Bump version + add v2.5.2 changelog comment

**Files:**
- Modify: `config.py` (header comment lines 1-28, `CONFIG_VERSION` line 40)

- [ ] **Step 1: Update header comment**

In `config.py`, replace the block from line 1 to line 28 (the docstring) with:

```python
"""
🍄 CARTOMORILLES — Configuration du modèle (v2.5.2)
   Zone : Isère (38) — 107.4×131.2 km, emprise dept élargie est+sud

   v2.5.2 :
     - Tree score `unknown` 0.25 → 0.40 (forêt sans essence ID = 64% du couvert)
     - Géologie calcaire saturée : calcaire 0.95 → 1.00, cascade marneux/marne/etc.
     - LANDCOVER_FOREST_FLOOR 0.80 → 0.85 (D8 floor relevé)
     - scoring.py : _COVERAGE_PENALTY_FLOOR 0.75 → 0.85
     - scoring.py : _CALCDRY_FLOOR 0.75 → 0.85, _CALCDRY_MAX_PENALTY 0.25 → 0.15
     - Cible : score_max 0.72 → 0.78, mean forêt 0.32 → 0.40
     - Décisions ajoutées : D32, D33, D34

   v2.5.0 :
     - Emprise élargie : +Oisans (est, xmax 932112→965000)
                        +Trièves (sud, ymin 6435430→6403000)
     - DEM multi-tuiles : Copernicus N44-45 × E004-006 (merge auto)
     - CELL_SIZE 10m → 20m (35.2M cellules, sous seuil warn 50M)

   v2.4.0 :
     - Extension emprise : Grenoble 20×20km → Isère 74.5×98.8km
     - DEM Copernicus N45_E005 ∩ BD Forêt v2 ∩ Géologie BDCharm-50
     - CELL_SIZE 5m → 10m (73.6M cellules, ~4.4 GB RAM)
     - Chemins données : DEM_FILE, FOREST_SHAPEFILE, GEOLOGY_SHAPEFILE
     - DIST_WATER_FOREST_FLOOR 0.20 → 0.15 (cohérence D9)

   v2.3.6 :
     - Scores résineux montagnards rehaussés (sapin 0.25→0.55,
       épicéa 0.35→0.45, pin_sylvestre 0.15→0.45)
     - Poids rééquilibrés : geology 0.18→0.20, tree_species 0.14→0.16,
       canopy_openness 0.13→0.11, twi 0.11→0.07

   v2.3.5 :
     - BUG FIX : SLOPE_STEEP 4.0 → 35.0
     - ALTITUDE_OPTIMAL élargi (200, 600)
     - unknown tree score 0.15 → 0.25
"""
```

- [ ] **Step 2: Bump `CONFIG_VERSION`**

Find line with `CONFIG_VERSION: Final[str] = "2.5.0"` and replace with:

```python
CONFIG_VERSION: Final[str] = "2.5.2"
```

- [ ] **Step 3: Verify config still imports**

Run: `python -c "import config; print(config.CONFIG_VERSION)"`
Expected output: `2.5.2`

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "chore: bump version to 2.5.2 with score uplift changelog"
```

---

### Task 2: Tree score `unknown` 0.25 → 0.40

**Files:**
- Modify: `config.py` (`_TREE_SCORES_DICT`, around line 254)

**Rationale:** Targets the 24.5M/38.2M (64%) forest cells without species ID. Forest tagged but unidentified ≠ non-forest. Coherent with v2.2.0 progression 0.15→0.25.

- [ ] **Step 1: Edit `_TREE_SCORES_DICT["unknown"]`**

In `config.py`, find:

```python
    # ── Défaut — forêt sans essence connue ≠ pas de forêt ──
    "unknown":            0.25,
```

Replace with:

```python
    # ── Défaut — forêt sans essence connue ≠ pas de forêt ──
    # v2.5.2 : 0.25 → 0.40 — forêt taguée sans essence ID = 64% du couvert
    "unknown":            0.40,
```

- [ ] **Step 2: Verify `validate_config()` passes**

Run: `python -c "import config; assert config.validate_config()"`
Expected output: log line `✅ Configuration validée (v2.5.2)`, no errors.

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "tune: tree unknown score 0.25 → 0.40 (D32 — 64% forest cells)"
```

---

### Task 3: Géologie calcaire band saturée

**Files:**
- Modify: `config.py` (`_GEOLOGY_SCORES_DICT`, around lines 466-475)

**Rationale:** Top discriminant (96.5% data coverage). Saturer 1.00 = traduction directe "calcaire = optimum biblio non amélioré". Cascade marneux/marne/alluvions_calcaires/calcaire_lacustre/eboulis_calcaire/dolomie suit pour préserver hiérarchie.

- [ ] **Step 1: Edit calcareous band**

In `config.py`, find the block:

```python
    # ── Substrats calcaires = optimaux morilles ──────────────────
    "calcaire":              0.95,
    "calcaire_marneux":      0.90,
    "marne":                 0.85,
    "alluvions_calcaires":   0.85,
    "calcaire_lacustre":     0.80,
    "eboulis_calcaire":      0.75,
    "dolomie":               0.75,
```

Replace with:

```python
    # ── Substrats calcaires = optimaux morilles ──────────────────
    # v2.5.2 : saturation optimum biblio (D33) — calcaire = pic absolu
    "calcaire":              1.00,   # était 0.95
    "calcaire_marneux":      0.95,   # était 0.90
    "marne":                 0.90,   # était 0.85
    "alluvions_calcaires":   0.90,   # était 0.85
    "calcaire_lacustre":     0.85,   # était 0.80
    "eboulis_calcaire":      0.80,   # était 0.75
    "dolomie":               0.80,   # était 0.75
```

- [ ] **Step 2: Verify scores still bornés [0,1]**

Run: `python -c "import config; assert config.validate_config()"`
Expected: no errors. (validate_config checks `0.0 <= val <= 1.0` for all GEOLOGY_SCORES.)

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "tune: calcaire band saturée à 1.00 + cascade (D33)"
```

---

### Task 4: `LANDCOVER_FOREST_FLOOR` 0.80 → 0.85

**Files:**
- Modify: `config.py` (around line 259)

**Rationale:** D8 plancher pour `green_score` en cellules BD Forêt — 0.80 préservait variabilité interne mais sous-pondérait micro-habitats forestiers. 0.85 conserve principe avec marge plus généreuse.

- [ ] **Step 1: Edit constant**

In `config.py`, find:

```python
# ── Landcover : plancher green_score pour cellules BD Forêt ──
# D8 : Forest floor 0.80 pour cellules BD Forêt
LANDCOVER_FOREST_FLOOR: Final[float] = 0.80
```

Replace with:

```python
# ── Landcover : plancher green_score pour cellules BD Forêt ──
# D8 : Forest floor pour cellules BD Forêt
# v2.5.2 : 0.80 → 0.85 — relève plancher micro-habitats forêt
LANDCOVER_FOREST_FLOOR: Final[float] = 0.85
```

- [ ] **Step 2: Verify import + valid**

Run: `python -c "import config; assert config.LANDCOVER_FOREST_FLOOR == 0.85"`
Expected: silent success.

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "tune: LANDCOVER_FOREST_FLOOR 0.80 → 0.85 (D8 floor relevé)"
```

---

### Task 5: `_COVERAGE_PENALTY_FLOOR` 0.75 → 0.85

**Files:**
- Modify: `scoring.py` (line 65)

**Rationale:** Cellule à 50% couverture pénalisée -25% actuellement (floor 0.75). Trop sévère — info manquante ≠ info négative. v2.5.1 montait déjà 0.5→0.75. Étape suivante.

- [ ] **Step 1: Edit constant**

In `scoring.py`, find lines around 62-65:

```python
# Fix #23 — Pénalité de couverture pour cellules à critères manquants
# v2.5.1 : floor 0.5 → 0.75 (pénalité max 25% au lieu de 50%)
# Cellule complète (10/10 critères) : × 1.0
# Cellule à 50% de couverture       : × 0.875 (était 0.75)
_COVERAGE_PENALTY_FLOOR: float = 0.75
```

Replace with:

```python
# Fix #23 — Pénalité de couverture pour cellules à critères manquants
# v2.5.1 : floor 0.5 → 0.75 (pénalité max 25%)
# v2.5.2 : floor 0.75 → 0.85 (pénalité max 15%) — D34
# Cellule complète (10/10 critères) : × 1.0
# Cellule à 50% de couverture       : × 0.925 (était 0.875)
_COVERAGE_PENALTY_FLOOR: float = 0.85
```

- [ ] **Step 2: Verify scoring import**

Run: `python -c "import scoring; print(scoring._COVERAGE_PENALTY_FLOOR)"`
Expected output: `0.85`

- [ ] **Step 3: Commit**

```bash
git add scoring.py
git commit -m "tune: _COVERAGE_PENALTY_FLOOR 0.75 → 0.85 (D34)"
```

---

### Task 6: `_CALCDRY_FLOOR` 0.75 → 0.85 + `_CALCDRY_MAX_PENALTY` 0.25 → 0.15

**Files:**
- Modify: `scoring.py` (lines 736-741, inside `apply_calcdry_penalty`)

**Rationale:** v2.5.1 montait 0.50→0.75 (Vouillants/Chartreuse scoraient 0.50). Persiste sous-scoring résiduel (Vouillants Δ=0.18 modèle vs attendu). Adoucir floor à 0.85 réduit Δ sans annuler malus pour vraies ravines xérothermophiles.

- [ ] **Step 1: Edit floor + max penalty**

In `scoring.py`, find the block around lines 735-741:

```python
        # Malus : score × multiplier
        # v2.5.1 : calibration adoucie (Vouillants/Chartreuse scoraient 0.50)
        # severity=0   → multiplier=1.0  (pas de malus)
        # severity=0.5 → multiplier=0.875 (était 0.75)
        # severity=1.0 → multiplier=0.75  (était 0.50)
        _CALCDRY_FLOOR = 0.75
        _CALCDRY_MAX_PENALTY = 0.25  # multiplier min = 1.0 - 0.25 = 0.75
```

Replace with:

```python
        # Malus : score × multiplier
        # v2.5.1 : calibration adoucie (Vouillants/Chartreuse scoraient 0.50)
        # v2.5.2 : floor 0.75 → 0.85, max_penalty 0.25 → 0.15 (D34)
        # severity=0   → multiplier=1.0  (pas de malus)
        # severity=0.5 → multiplier=0.925 (était 0.875)
        # severity=1.0 → multiplier=0.85  (était 0.75)
        _CALCDRY_FLOOR = 0.85
        _CALCDRY_MAX_PENALTY = 0.15  # multiplier min = 1.0 - 0.15 = 0.85
```

- [ ] **Step 2: Verify scoring still imports**

Run: `python -c "import scoring; print('ok')"`
Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add scoring.py
git commit -m "tune: _CALCDRY_FLOOR 0.75 → 0.85, max_penalty 0.25 → 0.15 (D34)"
```

---

### Task 7: Append decisions D32, D33, D34

**Files:**
- Modify: `decisions.md` (append rows after D25 / before `<decisions_new>` close tag at line 30)

- [ ] **Step 1: Edit `decisions.md`**

In `decisions.md`, find line 29:

```
|D25|`np.isin()` pour masques éliminatoires, `np.bincount()`pour dominant catégoriel dans clusters |Complète D6 — supprime tout string matching et intersection spatiale dans scoring.py 
<decisions_new>
```

Replace with:

```
|D25|`np.isin()` pour masques éliminatoires, `np.bincount()`pour dominant catégoriel dans clusters |Complète D6 — supprime tout string matching et intersection spatiale dans scoring.py 
|D32|tree_scores["unknown"] = 0.40, pas 0.25 | Forêt taguée BD Forêt v2 sans essence ID = 24.5M/38.2M (64%) cellules. Sous-scoring statistiquement significatif. Cohérent progression v2.2.0 (0.15→0.25) puis v2.5.2 (0.25→0.40).
|D33|geology_scores["calcaire"] = 1.00 + cascade marneux/marne/alluvions_calcaires/calcaire_lacustre/eboulis_calcaire/dolomie +0.05 | Top discriminant (96.5% data coverage). Saturation = optimum biblio non amélioré (M. esculenta + M. elata sur sols calcaires bien drainés). Aucun substrat ne dépasse, hiérarchie préservée.
|D34|_COVERAGE_PENALTY_FLOOR 0.75→0.85 (scoring.py) + _CALCDRY_FLOOR 0.75→0.85, _CALCDRY_MAX_PENALTY 0.25→0.15 (apply_calcdry_penalty) | Pénalités résiduelles signalent incertitude sans écraser la zone. Concordance Vouillants Δ=0.18 → cible Δ≤0.10. Cohérent progression v2.5.1 (0.50→0.75 puis 0.75→0.85).
<decisions_new>
```

- [ ] **Step 2: Verify file syntax (no broken table)**

Run: `head -35 decisions.md`
Expected: D25 visible, D32-D34 visible just after, `<decisions_new>` closing tag present.

- [ ] **Step 3: Commit**

```bash
git add decisions.md
git commit -m "docs: ajouter D32, D33, D34 (score uplift v2.5.2)"
```

---

### Task 8: Smoke test — dry-run + cell-size 25

**Files:**
- None modified. Read-only validation.

- [ ] **Step 1: Dry-run sanity check**

Run: `python main.py --dry-run`
Expected:
- `✅ Configuration validée (v2.5.2)` in log
- No errors
- Reports tree_scores["unknown"] = 0.40 and geology_scores["calcaire"] = 1.00 if those are echoed

- [ ] **Step 2: Quick run cell-size 25**

Run: `python main.py --cell-size 25 --output-dir output_uplift_test --purge-cache`

Wait for completion (estimate ~3-8 min depending on hardware).

- [ ] **Step 3: Compare stats vs baseline**

Run: `python -c "import json; r = json.load(open('output_uplift_test/rapport_cartomorilles.json')); s = r['statistics']; print(f\"max={s['score_max']:.4f} mean={s['score_mean']:.4f} median={s['score_median']:.4f} n_hotspots={s['n_hotspots']}\")"`

Compare to baseline:
- baseline: max=0.7185 mean=0.3176 median=0.4001 n_hotspots=8783
- expected: max ≥ 0.74, mean ≥ 0.36 (lower bound, cell=25m grid coarser than 10m baseline)

If max < 0.74 OR mean < baseline, **STOP and investigate** — likely a typo or unintended interaction. Do not proceed to full 10m run.

- [ ] **Step 4: Inspect Vouillants concordance**

Run: `grep "Vouillants" output_uplift_test/cartomorilles.log`
Expected: line `✅ Vouillants forêt calcaire 350m attendu=0.70 modèle=X.XX Δ=Y.YY`
- Target: Δ ≤ 0.10 (was 0.18)
- If Δ > 0.10, investigate — concordance dégradée signal d'over-tuning ailleurs.

---

### Task 9: Full 10m run + final stats

**Files:**
- None modified. Read-only validation.

- [ ] **Step 1: Full run cell-size 10m**

Run: `python main.py --cell-size 10 --output-dir output --purge-cache`

Expected duration : ~25-30 min (similar to baseline 1701s). 

- [ ] **Step 2: Compare vs baseline final**

Run: `python -c "import json; r = json.load(open('output/rapport_cartomorilles.json')); s = r['statistics']; print(f\"max={s['score_max']:.4f} mean={s['score_mean']:.4f} median={s['score_median']:.4f} n_high={s['n_high']} n_hotspots={s['n_hotspots']}\")"`

Targets:
- `score_max` ≥ 0.75 (was 0.7185)
- `score_mean` ≥ 0.38 (was 0.3176)
- `n_high` (cells ≥ 0.7) > baseline 537 — should be **substantially** higher
- Class distribution should now have non-zero "Excellent" (≥ 0.75)

- [ ] **Step 3: Inspect top hotspots**

Run: `python -c "import json; r = json.load(open('output/rapport_cartomorilles.json')); [print(f\"#{h['id']} score={h['mean_score']:.3f} max={h['max_score']:.3f} {h['dominant_species']}+{h['dominant_geology']} alt={h['altitude']}m\") for h in r['hotspots'][:10]]"`

Expected: top hotspot mean ≥ 0.73, max_score ≥ 0.78. Dominant should remain coherent (calcaire substrate, forest species).

- [ ] **Step 4: Concordance terrain check**

Run: `grep -A 1 "Validation terrain" output/cartomorilles.log | head -20`
Targets:
- `concordance_rate` ≥ 0.90 (was 0.909)
- Vouillants Δ ≤ 0.10

- [ ] **Step 5: Visual inspection**

Open `output/carte_morilles.html` in a browser.
Check:
- New "Excellent" zones present (red/dark on probability layer)
- Top hotspots are on calcaire + forest, not péri-urbain plat
- No obvious anomaly (e.g., entire valley flagged Excellent without coherence)

- [ ] **Step 6: Commit run artifacts (optional)**

If output dir is tracked, do NOT commit large `.tif` / `.html`. If user wants stats reference :

```bash
git add output/rapport_cartomorilles.json
git commit -m "chore: snapshot v2.5.2 stats reference"
```

(Skip if user prefers to keep output/ untracked.)

---

## Self-Review

**Spec coverage check:**

| Spec section | Task |
|---|---|
| §1 Tree unknown 0.25→0.40 | Task 2 |
| §2 Calcaire band saturée | Task 3 |
| §3 LANDCOVER_FOREST_FLOOR 0.80→0.85 | Task 4 |
| §4 _COVERAGE_PENALTY_FLOOR 0.75→0.85 | Task 5 |
| §4 _CALCDRY_FLOOR 0.75→0.85 + max_penalty 0.25→0.15 | Task 6 |
| §5 Bump version 2.5.2 + changelog | Task 1 |
| Décisions D32, D33, D34 ajoutées | Task 7 |
| Validation : dry-run, cell=25 sanity, full 10m | Tasks 8-9 |
| Vouillants concordance check | Tasks 8.4, 9.4 |

All spec requirements covered.

**Placeholder scan:** None. Each task has exact paths, exact code, exact commands, expected outputs.

**Type/name consistency:**
- `_COVERAGE_PENALTY_FLOOR` (Task 5) and `_CALCDRY_FLOOR` (Task 6) match exact identifiers from `scoring.py`.
- `LANDCOVER_FOREST_FLOOR` (Task 4) matches `config.py`.
- `_TREE_SCORES_DICT["unknown"]` (Task 2) matches dict key in `config.py`.
- `_GEOLOGY_SCORES_DICT` keys (Task 3) match config.py exactly.
- D32/D33/D34 IDs continuous after D25 (the `<decisions_new>` block — D26-D31 cited in CLAUDE.md exist conceptually but are not in `decisions.md` file ; D32+ is the next free contiguous ID in the file).

No issues found.
