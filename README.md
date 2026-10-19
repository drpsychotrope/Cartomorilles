# 🍄 CARTOMORILLES

Modèle probabiliste spatial multicritère cartographiant les zones favorables à la cueillette des **morilles** (*Morchella esculenta*) et des **cèpes** (*Boletus edulis*) en **Isère** (38).

Il combine analyse de terrain (MNT), essences forestières (BD Forêt v2), géologie (BDCharm-50), hydrographie, occupation du sol (OSM) et météo dans une grille de score pondéré, puis détecte les *hotspots* exportés avec coordonnées GPS.

- **CRS** : EPSG:2154 (Lambert-93)
- **Emprise** : Isère département élargi — 107,4 × 131,2 km (~14 091 km²)
- **Maille** : automatique selon l'emprise — 50 m pour le département (~5,6 M cellules) ; `--cell-size` pour la forcer
- **Python** : 3.12+
- **Version** : 2.5.4 (multi-espèces esculenta + edulis)

---

## Installation

```bash
pip install -r requirements.txt
```

Dépendances principales : `numpy`, `scipy`, `geopandas`, `shapely`, `rasterio`, `folium`, `pyproj`, `matplotlib`, `Pillow`, `requests`.

### Données requises (`data/`)

| Couche | Fichier attendu | Source |
|---|---|---|
| MNT | `Copernicus_DSM_COG_10_N{44,45}_00_E{004,005,006}_00_DEM.tif` (6 tuiles, merge auto) | Copernicus DSM 30 m |
| Essences | `FORMATION_VEGETALE.shp` | IGN BD Forêt v2 |
| Géologie | `geologie_38/GEO050K_HARM_038_S_FGEOL_2154.shp` | BRGM BDCharm-50 |
| Régions IFN | `rfifn250_l93.shp` | IFN |

Hydrographie et zones urbaines sont récupérées via WFS IGN à défaut de fichier local. L'occupation du sol est dérivée par analyse couleur des tuiles OSM.

---

## Démarrage rapide

```bash
# Run par défaut (maille auto, les deux espèces, météo J+0)
python main.py

# Idem, puis ouvrir la carte dans le navigateur
python main.py --open

# Test rapide dev (maille 25 m, tout le détail DEBUG à l'écran)
python main.py --cell-size 25 --verbose

# Dry-run : valider les données sans calculer
python main.py --dry-run

# Une seule espèce + export vecteur
python main.py --species esculenta --export-vector

# Purge cache puis run
python main.py --purge-cache

# Lister les couches WFS disponibles
python main.py --discover-wfs
```

### Options clés

| Flag | Défaut | Rôle |
|---|---|---|
| `--cell-size` | auto (50 m) | Taille de cellule (m) |
| `--species` | `esculenta edulis` | Espèces à modéliser |
| `--hotspot-threshold` | `0.60` | Seuil de score des hotspots |
| `--smoothing-sigma` | `0.5` | Sigma du lissage gaussien |
| `--urban-buffer` | `10` | Buffer (m) autour de l'urbain |
| `--no-urban` / `--no-landcover` / `--no-weather` | off | Désactiver un masque/modulation |
| `--landcover` | off | Forcer landcover sur grande emprise |
| `--seed` | `42` | Reproductibilité |
| `--output-dir` | `output/` | Répertoire de sortie |
| `--open` | off | Ouvrir la carte dans le navigateur en fin de run |
| `--console-hotspots` | `5` | Hotspots listés par espèce dans le résumé console (ancien nom : `--max-hotspots`) ; n'affecte ni la carte ni le CSV |
| `--map-hotspots` | `20` | Hotspots numérotés affichés à l'ouverture de la carte (réglable ensuite dans le panneau) |
| `--export-vector` | off | Export GPKG (volumineux) |
| `--debug-layers` | off | Ajouter à la carte les couches de diagnostic (zones exclues, TWI brut) |
| `-v` / `--verbose` | off | Tout le détail (DEBUG) à l'écran |
| `--force` | off | Ignorer les avertissements (poids, taille grille) |

Garde-fous : avertissement à 50 M cellules (TWI > 5 min), limite dure à 200 M sans `--force`.

### Sortie console

Par défaut, la console n'affiche que les étapes `[1/4]`…`[4/4]`, les avertissements et un résumé :

- la météo du jour et le facteur appliqué aux scores (×0,4 à ×1,0), ou son indisponibilité ;
- pour chaque espèce : les meilleurs hotspots (n°1, n°2… = rang du CSV et numéro du marqueur sur la carte, surface en ha, coordonnées, lien Google Maps) et la concordance avec les points de contrôle terrain ;
- les fichiers écrits par ce run et le lien `file://` vers la carte.

Le détail (tableau des poids, statistiques par critère, validation point par point, rapport météo 7 jours…) va dans `cartomorilles.log`, ou à l'écran avec `--verbose`.

---

## Sorties (`output/`)

- `cartomorilles.html` — carte interactive : panneau latéral (espèce, légende-filtre par classe, opacité, hotspots numérotés, fonds et couches, météo) et lecture du score au survol ou au toucher
- `{species}_probability.tif` — raster GeoTIFF par espèce (+ alias `morilles_probability.tif`)
- `{species}_hotspots.csv` — hotspots triés par score, avec lien Google Maps
- `rapport_{species}.json` — rapport complet et reproductible (config, stats, validation terrain)
- `cartomorilles.log` — journal détaillé
- `{species}_grid.gpkg` — grille vecteur (avec `--export-vector`)

---

## Architecture du pipeline

```
data_loader.py       → ingestion MNT, shapefiles, WFS/OSM
    ↓
grid_builder.py      → grille spatiale + 14 scores par critère
    ↓
species_enricher.py  → enrichissement essences (BD Forêt v2 + zones altitude IFN)
    ↓
scoring.py           → agrégation pondérée, masques éliminatoires, détection hotspots
    ↓
visualize.py         → carte HTML Folium, GeoTIFF, GeoPackage, CSV hotspots
```

Modules d'appui : `weather.py` (modulation météo J+0), `landcover_detector.py` (occupation du sol OSM), `_accel.py` (dispatch GPU/CPU transparent + rastérisation parallèle), `_twi_numba.py` (TWI D8), `utils.py`. `tools/` contient `fetch_inat.py` / `validate_inat.py` (observations iNaturalist).

---

## Modèle de scoring

`scoring.py` agrège **14 critères** par somme pondérée (`config.WEIGHTS`, somme = 1.0) :

| Critère | Poids | | Critère | Poids |
|---|---|---|---|---|
| geology | 0.20 | | aspect | 0.05 |
| tree_species | 0.20 | | forest_edge | 0.05 |
| altitude | 0.08 | | ground_cover | 0.05 |
| twi | 0.08 | | canopy_openness | 0.04 |
| disturbance | 0.08 | | terrain_roughness | 0.03 |
| dist_water | 0.07 | | urban_proximity | 0.01 |
| slope | 0.05 | | geology_contact | 0.01 |

Post-traitement :
1. **Masques éliminatoires** (score → 0), par espèce : urbain et proximité urbaine (< 15 m), plans d'eau, NoData du MNT, géologie éliminatoire (morille : granite, gneiss, siliceux ; cèpe : aucune), pente > 45°, altitude hors plage (morille 100–1 400 m, cèpe 200–2 200 m), engorgement (TWI).
2. **Pénalité de monotonie** (terrain uniforme).
3. **Pénalité calc-dry** (chênaies/buis xérothermophiles, morille seulement).
4. **Lissage gaussien spatial** (sigma configurable, défaut 0,5).
5. **Modulation météo** multiplicative : scores × (0,4 + 0,6 × score météo du jour Meteoblue), sauf `--no-weather` ou API indisponible — signalée dans le résumé console et dans `rapport_{species}.json`.
6. **Classification en 6 classes** : Nul / Très faible / Faible / Moyen / Bon / Excellent (seuils 0,15 · 0,30 · 0,45 · 0,60 · 0,75).

La sortie est validée contre des points de contrôle terrain par espèce (taux de concordance reporté en fin de run).

---

## Configuration

Tous les seuils, poids, scores essences/géologie et bornes spatiales vivent dans [`config.py`](config.py). Le dict **`WEIGHTS`** (14 critères) est la principale surface de réglage. Les rasters sont cachés dans `data/cache/` (clé MD5 des tableaux de lookup → invalidation auto si la config change).

---

## Décisions d'architecture

Les choix de conception (algorithme TWI, champ géologie, planchers de score, reprojection des overlays, etc.) sont documentés dans [`decisions.md`](decisions.md) et résumés dans [`CLAUDE.md`](CLAUDE.md). **Ne pas les inverser sans comprendre la justification.**

---

🧺 *Bonne prospection !*
