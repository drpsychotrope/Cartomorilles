# Morille noire : troisième espèce modélisée — design

Date : 2026-10-21. Branche de travail : `claude/enrichissement-ifn-par-espece` (après D47 ; la branche `claude/zen-dubinsky-f94603` porte D48, clé de cache, à fusionner avant ou après, les deux ne se recouvrent pas).

## Intention

L'utilisateur cueille les deux morilles : les **grises** (*Morchella esculenta*, *M. vulgaris* : frênaies, hêtraies calcaires de plaine et de coteau, 200–600 m) et les **noires** (groupe *M. elata* : *elata*, *conica*, *deliciosa*, *purpurascens* : sapinières, pessières, mélézins, coupes et chablis de montagne, 800–1 500 m et au-delà). Le modèle « esculenta » n'encode que l'écologie de la grise : optimum d'altitude 200–600 m, score d'altitude nul au-delà de 900 m, élimination au-dessus de 1 400 m, conifères médiocres (sapin 0,55, épicéa 0,45, mélèze non reconnu → essence inconnue).

Mesure du 21/10/2026 (tools/evaluate_occurrences.py, occurrences iNaturalist + GBIF précises à ± 250 m, genre *Morchella*) : 25 occurrences dans la bbox, dont 10 entre 900 et 1 625 m, 2 au-dessus de 1 400 m (éliminées). 40 % des morilles observées sont là où la carte dit « non ». Sur les 12 occurrences en Isère, AUC 0,53 (hasard), 4 en classe Nul (3 en village : masque urbain, 1 à 1 389 m : plafond d'altitude).

But : la carte reconnaît les deux morilles, chaque écologie avec ses propres optimums, sans dégrader la grise.

## Objectifs

1. Une troisième espèce modélisée, `elata` (« Morille noire »), dans `config.SPECIES_PARAMS`, avec les paramètres validés ci-dessous ; `SUPPORTED_SPECIES = ("esculenta", "elata", "edulis")`, défaut de `--species` = les trois.
2. La grise s'appelle désormais « Morille grise » (`common_name`, pluriel « morilles grises ») ; ses paramètres, ses sorties (`esculenta_*`, alias `morilles_probability.tif`) et ses 11 points de contrôle ne changent pas.
3. Le mélèze devient une essence reconnue (`meleze`, libellé « Mélèze ») dans les trois tables : grise 0,30, noire 0,85, cèpe 0,70. Aujourd'hui « Mélèze » (30 polygones BD Forêt) tombe en essence inconnue.
4. Tout ce qui est « par espèce » suit sans cas particulier : boucle de scoring et ré-enrichissement D47, hotspots, CSV, GPX, rapport JSON, bascule de la carte, résumé console, phrase de fin.
5. Une espèce sans points de contrôle terrain saute la validation croisée (log, concordance `None`) au lieu de retomber sur les points de la grise. `TERRAIN_CHECKPOINTS_BY_SPECIES["elata"] = ()` ; l'utilisateur pourra en ajouter.
6. Les occurrences naturalistes du genre *Morchella* servent de contrôle aux deux morilles (les observateurs ne séparent pas les deux de façon fiable) : le taxon naturaliste devient un paramètre d'espèce, `"taxon": ("Morchella", "GENUS")` pour les deux morilles, `("Boletus edulis", "SPECIES")` pour le cèpe ; `occurrences._TAXA` en dérive, et une requête GBIF est partagée par les espèces de même taxon (cache commun).
7. Le niveau C' de l'enrichissement (boost iNaturalist *Morchella*, species_enricher) et la couche d'observations de la carte s'appliquent à toute espèce dont le taxon est *Morchella*, plus seulement à `esculenta`.
8. `tools/evaluate_occurrences.py` peut évaluer le maximum de plusieurs couches (`--combine esculenta elata`) : c'est la question de l'utilisateur, « la paire reconnaît-elle les morilles observées ? ».
9. Décision D49 dans decisions.md ; README et CLAUDE.md : trois espèces, masques par espèce.

## Non-objectifs

- Une couche « toutes morilles » (maximum des deux) sur la carte : option ultérieure, l'outil d'évaluation en donne déjà la mesure.
- Retoucher les paramètres de la grise ou les poids `WEIGHTS` (communs aux espèces).
- Nouvelles sources de données (MNT fin, placettes IFN).
- Les trois autres chantiers validés le 21/10 (contrôle par occurrences dans chaque run, pondération par confiance de l'essence, malus xérique terrain) : à part, après celui-ci.

## Anti-objectifs

- Changer la grise autrement que par le mélèze : parts de classes et 11 points de contrôle comparés au run de base (D48, rasters frais). Le mélèze noté entre dans les mélanges IFN de toutes les cellules sans essence des régions qui en ont (Belledonne 2 %, Bas-Drac 3 %, Oisans 10 %, Trièves 2 %, Haut-Diois 3 %, défaut 1 %) au-dessus de 600 m : une dérive de la grise (0,30) et du cèpe (0,70) est attendue, faible, acceptée si elle est traçable au mélèze ; elle est mesurée (classes, points de contrôle, métriques d'occurrences) et consignée dans D49, sans cas particulier dans l'enrichissement (choix du concepteur après relecture).
- Une morille noire qui « éclaire » toute la montagne. Relecture du 21/10 : avec les poids actuels, une pessière sans essence BD Forêt en Belledonne (mélange IFN ≈ 0,9, granite 0,25, altitude 1,0) sort vers 0,6 (Bon) et une sapinière de Chartreuse sur calcaire vers 0,75 (Excellent). Ce n'est pas une porte d'acceptation de cette itération : un seul run, puis lecture du score de la noire aux 9 points de contrôle du cèpe (sapinières, pessières, Belledonne granite 1 500 m), des parts de classes et des métriques d'occurrences ; les leviers (granite/gneiss/siliceux, sapin-épicéa) se décident ensuite avec ces chiffres.
- Un Traceback, une sortie ou un libellé où l'espèce est codée en dur.

## Paramètres de la morille noire (validés par l'utilisateur le 21/10/2026)

| Paramètre | Valeur | Note |
|---|---|---|
| `common_name` / pluriel / `scientific_name` | « Morille noire » / « morilles noires » / « Morchella elata (groupe) » | |
| `altitude_optimal` / `altitude_range` | (800, 1 500) / (500, 1 900) | |
| `alt_eliminatory_min` / `max` | 300 / 2 200 | limite des arbres et mélézins |
| `slope_optimal` / `slope_max` | (0, 30) / 45 | |
| `aspect_scores` / `aspect_ref_deg` | ceux de la grise / 180 | choix du concepteur |
| `twi_*` | ceux de la grise (optimum 6–10, engorgement 18) | choix du concepteur |
| `apply_calcdry` / `apply_monotony` | False / True | |
| `eliminatory_geology` | ∅ | granite et gneiss notés bas, pas éliminés |
| `taxon` | ("Morchella", "GENUS") | partagé avec la grise |

Essences (`tree_scores`, clés identiques aux autres tables + `meleze`) : sapin 0,95 ; épicéa 0,95 (BD Forêt les confond : « Sapin, épicéa ») ; mélèze 0,85 ; pin sylvestre 0,60 (pins à crochets, cembro, noir, laricio y sont résolus) ; douglas 0,40 ; hêtre 0,60 ; érable 0,50 ; bouleau 0,40 ; noisetier 0,40 ; tilleul 0,35 ; frêne, orme, pommier, poirier, châtaignier, aulne, charme 0,30 ; peuplier 0,25 ; saule 0,20 ; chêne 0,20 ; chêne pubescent 0,10 ; robinier 0,10 ; platane 0,05 ; buis 0,05 ; inconnue NaN (D46).

Géologie (`geology_scores`, mêmes clés que les autres tables) : calcaire 1,00 ; calcaire marneux 0,90 ; dolomie 0,85 ; moraine 0,80 ; éboulis calcaire 0,70 ; marne 0,70 ; calcaire lacustre 0,70 ; colluvions 0,65 ; flysch 0,60 ; alluvions calcaires 0,60 ; alluvions 0,45 ; molasse 0,45 ; schiste 0,40 ; sable 0,40 ; alluvions récentes 0,35 ; grès 0,35 ; hydrographie 0,30 ; granite, gneiss, siliceux 0,25 ; remblai 0,10 ; inconnue 0,35. Les valeurs non citées dans l'échange du 21/10 (dolomie, flysch, schiste, sable, grès, molasse, hydrographie, remblai, inconnue) sont des choix du concepteur, entre la grise et le cèpe.

## Contraintes

- D6 : rasters int-codés + lookups ; la nouvelle table d'essences donne un nouveau hash de lookup, donc une rasterisation BD Forêt de plus au premier run (~30 s, puis cache). Avec D48 fusionné, la clé de cache inclut l'ordre de brûlage.
- D22, D46 : essence inconnue = NaN, brûlée en premier ; D47 : l'enrichissement IFN est rejoué par espèce avec `config.get_tree_score(nom, species)`, donc fonctionne pour `elata` dès que sa table existe.
- D38, D45 (classes et palette), D39 (emprise), D41 (console) : inchangés ; D44 : la météo, calculée sur des critères morille, s'applique telle quelle aux trois espèces.
- `validate_config` doit passer : chaque table d'essences et de géologie couverte par `TREE_LABELS` / `GEOLOGY_LABELS`, enveloppes d'altitude cohérentes (`alt_eliminatory_min < range_min < optimal < range_max < alt_eliminatory_max`).
- `resolve_tree_name` normalise « Mélèze » en `meleze` : la clé suffit ; alias `larix`, `larix_decidua`.
- Coût : une passe de plus à 50 m (~35 s) plus les exports (PNG, GeoTIFF, CSV, GPX, rapport) ; carte HTML plus lourde d'un PNG de score (~1,8 Mo).
- Rien n'est poussé ; commits sur la branche de travail.

## Vérification attendue

- Tests unitaires (`tests/`, unittest sans réseau) : `validate_config()` vrai avec trois espèces ; `get_tree_score("Mélèze", sp)` = 0,30 / 0,85 / 0,70 ; tables complètes ; enveloppe d'altitude de `elata` ; `occurrences` dérive ses taxons de `SPECIES_PARAMS`.
- Run complet 50 m, trois espèces (cwd checkout principal, `--bd-foret data/FORMATION_VEGETALE.shp --geology data/geologie_38/GEO050K_HARM_038_S_FGEOL_2154.shp --regions data/rfifn250_l93.shp`, `--output-dir` temporaire) : exit 0, aucun Traceback ; parts de classes et 11 points de la grise inchangés ; parts de classes, hotspots et Excellent de la noire relevés.
- `tools/evaluate_occurrences.py --run … --combine esculenta elata` : AUC, P/E par classe, part en Nul pour la grise, la noire et leur maximum ; les occurrences de 900 à 1 625 m doivent sortir de la classe Nul et des zones à score d'altitude nul.
- Les chiffres avant/après vont dans le message de commit et dans D49.

## Laissé au constructeur

- Détail JavaScript de la carte pour lier la couche d'observations à plusieurs espèces (`layerSpecies` accepte une liste).
- Nom et clé du cache GBIF partagé ; texte du README ; noms des tests.
- Ordre des espèces dans la carte et la console : grise, noire, cèpe.
