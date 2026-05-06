# Spec — Rehaussement scores probabilité morilles

**Date** : 2026-05-06
**Cible version** : v2.5.2
**Auteur** : brainstorm session
**État** : design approuvé, à implémenter

## Contexte

Le run actuel (config v2.5.0, cell=10m, dept élargi) plafonne à :
- `score_max` = 0.7185 (mean=0.318, median=0.400)
- Top hotspot mean=0.67 (chataignier sur calcaire, alt 523m, pente 14°)
- Aucun hotspot en classe "Excellent" (≥ 0.75)
- Concordance terrain Vouillants : modèle 0.52 vs attendu 0.70 (Δ=0.18)

Les scores théoriques pour une cellule "parfaite" (chataignier + calcaire + alt+slope+twi optimaux) atteignent ~0.78 avant pénalités, mais smoothing + coverage penalty + calc_dry rabotent à ~0.72.

## Objectif

Rehausser plafond ET moyenne pour qu'au moins quelques hotspots atteignent la classe "Excellent" (≥ 0.75) et que la moyenne forestière monte de 0.32 à ~0.40, **sans inflation aveugle** :
- Cibler les leviers à fort impact statistique (densité de données)
- Respecter les décisions verrouillées D2, D3, D6, D8, D9, D22, D26, D27
- Maintenir la concordance terrain (actuellement 90.9% sur 11 checkpoints)

## Hors scope

- Rehausser les scores essences fréquentes (hetre/tilleul/erable) — BD Forêt v2 ne renseigne l'essence que sur 36% des cellules forestières, impact statistique faible.
- Toucher chataignier (verrouillé D3 à 0.80).
- Modifier les pondérations `WEIGHTS` — la redistribution v2.5.0 reste valide.
- Toucher `_MONOTONY_FLOOR` — déjà à 0.85.
- Modifier le pipeline de scoring (`scoring.py`) au-delà des constantes `_COVERAGE_PENALTY_FLOOR` et `_CALCDRY_FLOOR`.

## Changements

### 1. `config.py` — Tree score `unknown`

Cible la masse de cellules forestières sans essence identifiée (24.5M / 38.2M cellules forêt = 64%).

```python
_TREE_SCORES_DICT = {
    ...
    "unknown": 0.40,   # était 0.25 — forêt sans essence connue
                       # reste favorable (cf. v2.2.0 qui montait 0.15→0.25)
}
```

**Justification** : une cellule taguée forêt par BD Forêt v2 mais sans essence identifiable n'est pas équivalente à une zone non-forestière. Le score 0.25 sous-estime systématiquement 64% du couvert forestier. Cohérent avec D8 (forest_floor 0.80 pour green_score).

### 2. `config.py` — Géologie favorable resaturée

La géologie est le critère le plus dense (96.5% des cellules ont une classification, 17.5% sont >0.7). Top discriminant biblio.

```python
_GEOLOGY_SCORES_DICT = {
    "calcaire":              1.00,   # était 0.95 — saturer optimum biblio
    "calcaire_marneux":      0.95,   # était 0.90
    "marne":                 0.90,   # était 0.85
    "alluvions_calcaires":   0.90,   # était 0.85
    "calcaire_lacustre":     0.85,   # était 0.80
    "eboulis_calcaire":      0.80,   # était 0.75
    "dolomie":               0.80,   # était 0.75 — suit éboulis calcaire
    # Substrats intermédiaires + défavorables : inchangés
    # Éliminatoires (granite, gneiss, siliceux) : inchangés à 0.00
}
```

**Justification** : la biblio (Mojžišová 2018, Pilz et al.) place les sols calcaires bien drainés au sommet de la préférence morilles (M. esculenta, M. elata sur sapinière calcaire). Le score 0.95 actuel laisse une marge inutilisée vs siliceux 0.00. Saturer 1.00 = traduction directe "calcaire = optimum non amélioré".

### 3. `config.py` — Plancher landcover forêt

```python
LANDCOVER_FOREST_FLOOR: Final[float] = 0.85   # était 0.80 (D8)
```

**Justification** : D8 reste valide en principe (les cellules BD Forêt méritent un plancher pour `green_score` afin d'éviter sous-scoring HSV). Relever de 0.80 à 0.85 pousse les scores micro-habitat (canopy_openness/ground_cover/disturbance modulés par green_score) sans abolir leur variabilité interne.

### 4. `scoring.py` — Adoucir penalty floors

```python
# Ligne 65 environ
_COVERAGE_PENALTY_FLOOR: float = 0.85   # était 0.75 (v2.5.1 montait 0.5→0.75)
                                         # Cellule à 50% couverture : ×0.925 au lieu de ×0.875
                                         # Cellule à 100% couverture : ×1.0 (inchangé)
```

```python
# Dans apply_calcdry_penalty(), ligne 740 environ
_CALCDRY_FLOOR = 0.85   # était 0.75 (v2.5.1 montait 0.50→0.75)
_CALCDRY_MAX_PENALTY = 0.15   # était 0.25 — multiplier min = 1.0 - 0.15 = 0.85
```

**Justification** :
- Coverage penalty : pénalisait jusqu'à -25% une cellule à couverture 50%. La motivation initiale (D10) reste — une cellule incomplète est moins fiable —, mais -15% suffit à signaler l'incertitude sans écraser la zone.
- Calc_dry : v2.5.1 a déjà adouci la calibration "Vouillants/Chartreuse scoraient 0.50". Persiste un sous-scoring résiduel (Δ=0.18 sur Vouillants). Relever floor 0.75→0.85 le réduit sans annuler le malus pour vraies ravines xérothermophiles.

### 5. `config.py` — Métadonnées

```python
CONFIG_VERSION: Final[str] = "2.5.2"
```

Bump version + commentaire en tête du fichier décrivant v2.5.2.

## Estimation impact

**Plafond** :
- Top hotspot actuel (chataignier + calcaire) : geology +0.05 → +0.010 contribution pondérée. Coverage floor +0.10 préserve davantage. Calc_dry floor +0.10 idem. **Estimation : 0.72 → 0.78**.

**Moyenne forestière** :
- 24.5M cellules unknown forêt : tree_species 0.25 → 0.40 = +0.15 sur 20% de poids = +0.030 / cellule. **Estimation : 0.32 → 0.40**.

**Distribution des classes** : émergence d'une classe "Excellent" non vide attendue (actuellement 537 cellules en `n_high` sur 140M, soit 0.0%).

## Risques / mitigations

| Risque | Mitigation |
|---|---|
| Sur-scoring zones forêt non-essence sur géologie médiocre | `unknown` 0.40 reste sous chataignier 0.80 et hetre 0.75 — différenciation préservée. Géologie médiocre (schiste 0.15, gres 0.25) compense. |
| Plafond saturé masque vrais optima | Calcaire 1.00 = optimum biblio non amélioré. Aucun substrat ne dépasse. Cohérent. |
| Concordance terrain dégradée | Vérifier après run que concordance ≥ 90% maintenue. Si dégradation, ajuster `unknown` à 0.35. |
| Cache invalidé | `unknown` 0.25→0.40 et géologie modifient les lookup arrays MD5 → cache `data/cache/*.npy` auto-invalidé (D23). Comportement attendu. |

## Validation

Après implémentation :
1. `python main.py --cell-size 25 --dry-run` — sanity check config valide.
2. `python main.py --cell-size 25` — run rapide, comparer `score_max`, `score_mean`, distribution classes vs baseline.
3. Si stats cohérentes (score_max ≥ 0.75, mean ≥ 0.38, concordance ≥ 90%) : run complet `--cell-size 10`.
4. Inspection visuelle `output/carte_morilles.html` — vérifier que les nouvelles zones "Excellent" sont sur calcaire+forêt, pas en zones plates péri-urbaines.

## Décisions à ajouter (`decisions.md`)

- **D32** : `tree_scores["unknown"]` = 0.40, pas 0.25 — forêt taguée sans essence ID = 64% du couvert, sous-scoring statistiquement significatif.
- **D33** : `calcaire` = 1.00, pas 0.95 — optimum biblio, saturation cohérente, données denses (96.5% couvertes).
- **D34** : `_COVERAGE_PENALTY_FLOOR` = 0.85 et `_CALCDRY_FLOOR` = 0.85 — pénalités résiduelles signalent l'incertitude sans écraser la zone (concordance Vouillants Δ=0.18 → cible Δ≤0.05).
