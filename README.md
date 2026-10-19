# 🍄 CARTOMORILLES

Modèle probabiliste spatial multicritère cartographiant les zones favorables à la cueillette des **morilles** (*Morchella esculenta*) et des **cèpes** (*Boletus edulis*) en **Isère** (38).

Il combine analyse de terrain (MNT), essences forestières (BD Forêt v2), géologie (BDCharm-50), hydrographie, occupation du sol (OSM) et météo dans une grille de score pondéré, puis détecte les *hotspots* exportés avec coordonnées GPS.

- **CRS** : EPSG:2154 (Lambert-93)
- **Emprise** : Isère département élargi — 107,4 × 131,2 km (~14 091 km²)
- **Maille par défaut** : 20 m → ~35,2 M cellules (auto-scale selon l'emprise)
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
# Run par défaut (auto cell-size, les deux espèces, météo J+0)
python main.py

# Test rapide dev (maille 25 m, logs DEBUG)
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
| `--cell-size` | auto | Taille de cellule (m) |
| `--species` | `esculenta edulis` | Espèces à modéliser |
| `--hotspot-threshold` | `0.60` | Seuil de score des hotspots |
| `--smoothing-sigma` | `0.5` | Sigma du lissage gaussien |
| `--urban-buffer` | `10` | Buffer (m) autour de l'urbain |
| `--no-urban` / `--no-landcover` / `--no-weather` | off | Désactiver un masque/modulation |
| `--landcover` | off | Forcer landcover sur grande emprise |
| `--max-hotspots` | `8` | Hotspots affichés en console |
| `--seed` | `42` | Reproductibilité |
| `--output-dir` | `output/` | Répertoire de sortie |
| `--export-vector` | off | Export GPKG (volumineux) |
| `--debug-layers` | off | Ajouter à la carte les couches de diagnostic (zones exclues, TWI brut) |
| `--force` | off | Ignorer les avertissements (poids, taille grille) |

Garde-fous : avertissement à 50 M cellules (TWI > 5 min), limite dure à 200 M sans `--force`.

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
1. **Masques éliminatoires** (score → 0) : granite/basalte/grès, altitude extrême, cœur urbain, pente > 45°, espèces infranchissables, engorgement (TWI).
2. **Pénalité de monotonie** (terrain uniforme).
3. **Pénalité calc-dry** (chênaies/buis xérothermophiles).
4. **Lissage gaussien spatial** (sigma configurable).
5. **Modulation météo** multiplicative (J+0, sauf `--no-weather`).
6. **Classification en 6 classes** : Nul / Très faible / Faible / Moyen / Bon / Excellent (seuils 0,15 · 0,30 · 0,45 · 0,60 · 0,75).

La sortie est validée contre des points de contrôle terrain par espèce (taux de concordance reporté en fin de run).

---

## Configuration

Tous les seuils, poids, scores essences/géologie et bornes spatiales vivent dans [`config.py`](config.py). Le dict **`WEIGHTS`** (14 critères) est la principale surface de réglage. Les rasters sont cachés dans `data/cache/` (clé MD5 des tableaux de lookup → invalidation auto si la config change).

---

## Décisions d'architecture

Les choix de conception (algorithme TWI, champ géologie, planchers de score, reprojection des overlays, etc.) sont documentés dans [`decisions.md`](decisions.md) et résumés dans [`CLAUDE.md`](CLAUDE.md). **Ne pas les inverser sans comprendre la justification.**

---

🍄 *Bonne chasse aux morilles !*
