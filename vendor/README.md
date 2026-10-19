# Bibliothèques de la carte HTML

`visualize.py` (`_embed_libraries`) insère ces fichiers dans la carte HTML à la place des liens CDN que Folium y mettrait (décision D43). La carte fonctionne ainsi sans réseau, fonds de carte mis à part. Ce sont les fichiers des liens de Folium 0.20 : la carte se comporte et s'affiche comme avec les CDN.

| Dossier | Contenu | Source | Licence |
|---|---|---|---|
| `leaflet-1.9.3/` | Leaflet 1.9.3 : JS, CSS, images de la CSS et de l'icône par défaut | https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/ | BSD-2-Clause |
| `jquery-3.7.1/` | jQuery 3.7.1 : Folium construit ses popups avec `$(…)` | https://code.jquery.com/jquery-3.7.1.min.js | MIT |
| `leaflet-measure-2.1.7/` | leaflet-measure 2.1.7 (outil de mesure) : JS, CSS et images | https://cdn.jsdelivr.net/gh/ljagis/leaflet-measure@2.1.7/dist/ | MIT |
| `bootstrap-5.2.2/` | Bootstrap 5.2.2, feuille de base seule (`bootstrap-reboot.min.css`) | https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/ | MIT |
| `bootstrap-3.0.0/` | Bootstrap 3.0.0, normalize + scaffolding : extrait du fichier « glyphicons » que lie Folium | https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css | Apache-2.0 |

## Pourquoi deux feuilles de base Bootstrap

La page n'utilise aucune classe ni aucun script Bootstrap, mais son rendu dépend de deux feuilles que Folium charge l'une après l'autre :

- le reboot de Bootstrap 5 : `box-sizing`, `[hidden] { display: none !important }`, hauteur de ligne des titres, marges ;
- le début du fichier « glyphicons » de netdna, qui contient aussi le normalize et le scaffolding de Bootstrap 3 : `html { font-size: 62.5% }` (1rem = 10 px, d'où la taille du texte des popups, de l'attribution et de l'outil de mesure), texte 14 px #333, liens.

Vérifié en comparant les styles calculés de tous les éléments (pseudo-éléments compris) avec ceux de la page liée aux CDN : identiques à 1280 × 800 et à 375 px, panneau, popups et outil de mesure ouverts. Sans le fichier « glyphicons », le texte des popups passe de 10,8 à 17,3 px. Sans le reboot, 59 éléments changent (titres, éléments masqués, tableaux). Le JS et les composants de Bootstrap, Font Awesome et Leaflet.awesome-markers ne changent rien : repères et hotspots sont des DivIcon. Ils ne sont pas repris.

## Copies conformes

Les fichiers sont identiques à ceux publiés : `cd vendor && sha256sum -c SHA256SUMS`. Git ne convertit pas leurs fins de ligne (`.gitattributes`). Les images des feuilles de style sont mises en data URI quand la carte est générée, de même que l'icône par défaut de Leaflet : Folium crée chaque marqueur avec elle avant de lui appliquer sa propre icône, et Leaflet irait sinon la chercher à côté de la page.

Une exception : `bootstrap-3.0.0/bootstrap-glyphicons.base.css` = les 62 premières lignes du fichier publié (empreinte SHA-256 de l'original : `ca64645c22680035acdd8149902fda928c381cafbeab0b628b5542a7323ee0e4`). La suite (`@font-face` et classes `.glyphicon-*`) est retirée : la page ne s'en sert pas et ses polices ne sont pas fournies. Pour la refaire :

```bash
curl -sL https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css | head -n 62 > vendor/bootstrap-3.0.0/bootstrap-glyphicons.base.css
```

## Changer de version

Si Folium change l'URL d'une de ces bibliothèques, le run l'annonce par un avertissement (« Carte : leaflet chargé depuis https://… (version absente de vendor/), indispensable hors ligne ») et la carte garde le lien CDN pour celle-ci. Pour l'intégrer :

1. télécharger les fichiers de la nouvelle URL dans un dossier `<bibliothèque>-<version>/`, avec la licence et les images citées par la CSS ;
2. comparer leurs empreintes à celles publiées (SRI du site de la bibliothèque, ou `https://data.jsdelivr.com/v1/package/npm/<paquet>@<version>`) ;
3. mettre à jour `_VENDORED` dans `visualize.py`, puis `SHA256SUMS` ;
4. ouvrir une carte générée avec tous les hôtes externes coupés (README, « Sur le terrain : téléphone et hors ligne »), et comparer son rendu à celui de la version liée aux CDN.
