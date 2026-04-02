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