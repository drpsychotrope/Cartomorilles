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
|D38|Une seule échelle de score : les 6 classes `config.PROBABILITY_*` pour la carte (légende et panneau météo) et la console (`display_statistics`). Le rapport météo de la console garde l'échelle à 5 niveaux de `weather.py`, verrouillé par la session « weather » (`.sessions/locks.json`). Palette de probabilité séquentielle ambre → brun (une famille de teinte, luminance OKLab monotone) sur les fonds clairs ; même famille inversée brun → jaune sur les fonds sombres (`dark: True` dans `config.BASEMAPS` : photo aérienne, satellite), légende et marqueurs suivent. | RdYlGn mettait le rouge sur « Nul », le vert sur les forêts vertes du fond IGN, et ne se lisait pas en daltonisme ; la console comptait 3 classes (0,4 / 0,7) et la météo 5 classes à d'autres seuils. Mesures ΔE OKLab ×100 sur tuiles IGN réelles (alpha de classe × opacité 70 %) : Bon / Excellent ≈ 18 / 27 sur Plan v2 ; sur photo ≈ 14 / 22 avec la palette inversée contre 6 / 9 sans, protan et deutan du même ordre. Les seuils 0,4 / 0,7 ne restent que dans le rapport JSON (`n_high`, comparaison avec D35).|
|D39|Emprise d'étude = couverture BDCharm-50 départementale (Isère) ∩ grille : contour sur halo blanc et extérieur voilé (volet Leaflet au-dessus de la probabilité), ni score (code 254 dans le data PNG) ni hotspot affichés hors emprise. Le modèle, les GeoTIFF et les CSV ne changent pas. | Hors Isère il n'y a ni géologie ni BD Forêt : les scores (ex. est lyonnais 0,46 « Moyen ») n'y reposent que sur le terrain. `coverage_union_all` ≈ 1 s contre ≈ 10 s pour `union_all` ; limite IFN 1/250 000 écartée (jusqu'à 250 m d'écart).|
|D40|Carte HTML sur le terrain : clic ou toucher → popup du point (score, coordonnées WGS 84 copiables, liens « Itinéraire » et Google Maps, mêmes actions dans les popups de hotspots) ; toucher la carte quand une popup est ouverte la ferme seulement ; bouton « Me localiser » (`watchPosition` : centre au premier point puis suit sans recentrer) ; meta viewport de Folium réécrit après rendu, sans `maximum-scale` ni `user-scalable=no` ; cibles ≥ 44 px sur écran tactile ou étroit ; état replié/ouvert du panneau en `localStorage` ; `{species}_hotspots.gpx` = les 20 meilleurs hotspots, numérotés comme la carte. Dépendance au réseau (tuiles, CDN de Folium dont netdna.bootstrapcdn.com) documentée dans le README, pas embarquée. | Audit UI/UX, lot D : carte peu utilisable au doigt (cibles de 28 à 36 px, coordonnées non copiables, pas de position), zoom du navigateur bloqué (WCAG 1.4.4), aucun export pour les applis GPS, qui gardent leurs cartes hors ligne. Embarquer Leaflet et les autres bibliothèques alourdirait le HTML et toucherait au gabarit Folium : documenté plutôt que modifié.|
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