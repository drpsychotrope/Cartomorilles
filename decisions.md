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
D11	_accel.py auto-dispatch GPU/CPU, retour toujours numpy | Transparence pour les appelants, pas de cupy leak
D12	ProcessPoolExecutor (pas Thread) pour rasterize parallèle |	rasterio Cython garde le GIL
D13	Cache disque masques .npy dans data/cache/ | Urban 446k polygones = 27s → 0s au 2e run
D14	_fill_nan_dem garde scipy EDT local (return_indices) |	Pas d'équivalent dans _accel
D15	Hotspots : reuse_mask pré-alloué pour boucle dominant |	Évite 1312× np.zeros sur 3M cells
D16	bincount vectorisé pour cluster stats (pas boucle Python) |	O(N) total vs O(N×K)
D17	compress_level=1 pour overlays PNG, compress_level=6 pour data PNG tooltip | Overlays encodés en parallèle (vitesse prime), data PNG encodé 1× (taille prime — -11 MB HTML)
D18	Urbain : gaussian σ=1.5 en espace source + reprojection bilinéaire (pas d'upscale zoom) | Upscale ×2 + gaussian sur grille doublée coûtait 4.1s pour un lissage équivalent
D19	Warmup _nb_map_coordinates_bilinear exclu du warmup standard | JIT coûte 4.5s, kernel non utilisé en production (fallback rasterio actif)
D20	TWI accumulation D8 acceptée comme incompressible (~14s sur 73M px) | Dépendance topologique séquentielle, pas de parallélisation GPU triviale
<decisions_new>
 D21|`parallel_rasterize_categorical` int16 par bandes — même pattern que `parallel_rasterize_mask`|Réutilisation prouvée (urban_mask), STRtree + WKB sérialisation + ProcessPool identique 
 D22|Tri score croissant avant burn catégoriel (last wins = best score)|rasterio.features.rasterize écrase dans l'ordre → le dernier polygone prévaut, tri croissant garantit que le meilleur score gagne en chevauchement
 D23|Cache raster catégoriel invalidé par hash du lookup (pas seulement n_geom + cell_size) |
 Changement de scores dans config.py → lookup.tobytes() change → hash MD5 change → cache miss automatique
D24|Attributs `_*_int_raster` / `_*_code_to_name` / `_*_score_lookup`sur GridBuilder, consommés par scoring.py | Découplage rasterisation (grid_builder) / évaluation (scoring) — scoring n'a plus besoin du GeoDataFrame ni de config 
|D25|`np.isin()` pour masques éliminatoires, `np.bincount()`pour dominant catégoriel dans clusters |Complète D6 — supprime tout string matching et intersection spatiale dans scoring.py 
|D32|tree_scores["unknown"] = 0.40, pas 0.25 | Forêt taguée BD Forêt v2 sans essence ID = 24.5M/38.2M (64%) cellules. Sous-scoring statistiquement significatif. Cohérent progression v2.2.0 (0.15→0.25) puis v2.5.2 (0.25→0.40).
|D33|geology_scores["calcaire"] = 1.00 + cascade marneux/marne/alluvions_calcaires/calcaire_lacustre/eboulis_calcaire/dolomie +0.05 | Top discriminant (96.5% data coverage). Saturation = optimum biblio non amélioré (M. esculenta + M. elata sur sols calcaires bien drainés). Aucun substrat ne dépasse, hiérarchie préservée.
|D34|_COVERAGE_PENALTY_FLOOR 0.75→0.85 (scoring.py) + _CALCDRY_FLOOR 0.75→0.85, _CALCDRY_MAX_PENALTY 0.25→0.15 (apply_calcdry_penalty) | Pénalités résiduelles signalent incertitude sans écraser la zone. Concordance Vouillants Δ=0.18 → cible Δ≤0.10. Cohérent progression v2.5.1 (0.50→0.75 puis 0.75→0.85).
|D35|v2.5.3 — calc_dry threshold tree_species ≤ 0.30 (était ≤ 0.50) + _CANOPY_BASE_INTERIOR 0.55 (était 0.45) + forest_edge interior floor 0.65 (était 0.5) | v2.5.2 a doublé n_high (cellules ≥0.7) ×13.5 mais plafond 0.738 < cible 0.75. Trois leviers complémentaires : exclure `unknown=0.40` du calc_dry (24M cellules concernées), relever planchers micro-habitat forêt intérieure (canopy + forest_edge). Cible : max ≥ 0.78, classe Excellent non vide.
|D36|v2.5.4 — TOUTES tentatives ABANDONNÉES, retour v2.5.3. | v2.5.3 plafond 0.739 < cible 0.75 motivait 4 leviers : (a) SLOPE_OPTIMAL (0,15)→(0,20) ; (b) ALTITUDE_OPTIMAL (200,600)→(200,700) ; (c) SOFT_ELIM_BUFFER 15→5m ; (d) gaussian truncate 4.0→2.0. RÉSULTAT : tous régressent le max à ~0.68 (vs v2.5.3 0.739). Cause commune apparente : interaction inattendue avec le smoothing normalisé qui agit comme un "boost-aux-borderlines via convolution avec voisins valides". Réduire l'agressivité du smoothing OU les bornes anti-noise (buffer) RETIRE ce boost. Augmenter les scores intrinsèques (slope/alt widening) déclenche apparemment un effet similaire via redistribution des scores neighbors. APIs ajoutées CONSERVÉES (truncate kwarg, SOFT_ELIM_BUFFER constant) — utiles indépendamment. Production = v2.5.3. Diagnostic post-mortem nécessite plus de temps.
|D37|Carte HTML : une seule couche de probabilité, rendue en JS depuis un PNG de score par espèce (+ un PNG terrain commun pente/altitude) ; panneau latéral unique (espèce, légende-filtre, opacité, hotspots, couches, météo) au lieu des panneaux flottants et du LayerControl ; hotspots en JSON (top N, marqueurs numérotés) ; rendu à la résolution de la grille ; couches de diagnostic seulement avec `--debug-layers`. Remplace le volet « panneau météo déplaçable » de D31 : la cohérence passe désormais par le panneau unique. | Audit UI/UX : la couche Folium non filtrée recouvrait la couche JS (filtre et bascule d'espèce inopérants), panneaux superposés aux contrôles Leaflet, 800+ marqueurs à l'ouverture, HTML 264 Mo. Après : 23 Mo à 50 m, chargement ~1,7 s, lecture de cellule calculée en Y Mercator (D29) — erreur moyenne 0,02 au lieu de 0,09 sur 300 cellules.|
|D42|Météo : Open-Meteo `/v1/forecast` (sans clé, usage non commercial) remplace la source précédente dans weather.py. Point MAP_CENTER, `timezone=Europe/Paris`, 14 jours passés (`past_days`) + 7 jours de prévision ; `evaluate()` renvoie toujours 7 jours à partir d'aujourd'hui (`days[0]` = aujourd'hui pour main.py, formule 0,4 + 0,6 × score inchangée). Sol : moyennes journalières des variables horaires `soil_temperature_6cm`/`_18cm` (interpolées à 10 cm) et `soil_moisture_3_to_9cm`. Score du jour : sol à 10 cm, trapèze 4/8/12/18 °C (0,30) ; humidité du sol, optimum 0,20–0,40 m³/m³ (0,15) ; air 0,15 ; gel 0,15 ; pluie du jour 0,10 ; pluie J-3..J-1 0,05 ; humidité de l'air 0,05 ; vent 0,05. Puis bonus choc hydrique, puis facteur réchauffement du sol ×0,5–1 (moyenne J-6..J contre J-13..J-7 : +1,5 °C → ×1, −1,5 °C → ×0,5). Cache `data/openmeteo_cache.json` (1 h, lié au point et aux variables) ; hors ligne (réseau absent ou `offline=True`) : dernier cache tant qu'il couvre aujourd'hui. Clé `_API_KEY_DEFAULT` et paramètre `api_key` supprimés. | Clé Meteoblue codée en dur dans un dépôt public. Pousse des morilles vers 8–12 °C de sol ; Mihail, Bruhn & Bonello 2007 (Mycol. Res. 111:339-346) : la chaleur accumulée par le sol dans les ~20 jours avant la pousse est le meilleur indicateur. En octobre le sol est aussi à 8–12 °C mais il refroidit : le facteur réchauffement sépare printemps et automne. L'historique corrige un biais : J0 n'avait aucune pluie J-3..J-1 (toujours « sol sec ») et ne pouvait pas recevoir le bonus choc hydrique. Le vent Meteoblue arrivait en m/s, comparé à des seuils en km/h (critère inopérant) ; Open-Meteo donne des km/h. Semaine du 19/10/2026 : score J0 0,81 → 0,62, moyenne 7 j 0,69 → 0,53, facteur J0 0,89 → 0,77. Seuils d'humidité du sol calés sur le modèle à ce point (0,28–0,41 m³/m³ en octobre) : à recaler au printemps. Le facteur, calculé sur des critères morille, s'applique aussi au cèpe : à traiter à part.|
<decisions_new>

**Approches REJETÉES :**

| Proposition | Raison du rejet |
|---|---|
| score_slope seuils renforcés (0,8)/(15)/(25)/(45) | Supersédé par TWI |
| dist_water pénalité engorgement <5m | Supersédé par TWI waterlog |
Upscale ×2 + gaussian pour lissage masque urbain |	Coût 4.1s pour résultat équivalent à gaussian σ=1.5 + reproj bilinéaire
Warmup map_coordinates Numba au démarrage |	4.5s JIT pour un kernel non utilisé (rasterio fallback suffisant)
GPU accumulation D8 pour TWI |	Dépendance topologique empêche parallélisation CUDA
Rasterisation N passes (une par catégorie) puis merge | 1 seule passe catégorielle suffit — N passes = N× I/O rasterio inutile 
`GeoDataFrame.sjoin` pour score par cellule grille | O(n_cells × n_poly) — intenable sur 73M pixels, raster int-codé est O(1) par pixel 
nodata = -1 pour raster catégoriel | int16 signé complique les lookups `np.take` — nodata = 0 avec lookup[0] = 0.5 (neutre) est plus simple 
| Réchauffement du sol en critère additif (poids 0,10) dans le score météo | Essai du 19/10/2026 : toute la semaine en « Excellent » (J0 1,00, moyenne 0,80) fin octobre, hors saison des morilles ; le refroidissement du sol ne retirait que ~0,08. Remplacé par un facteur ×0,5–1 (D42). |