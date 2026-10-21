# visualize.py
"""visualize.py — Cartomorilles v2.4.0

Visualisation interactive (Folium ImageOverlay raster), export GeoTIFF et
GeoPackage.

v2.4.0 :
  - PNG compress_level=1 (×3 encoding speed vs optimize=True)
  - ThreadPoolExecutor pour encoding PNG parallèle (PIL relâche le GIL)
  - _accel.gaussian_filter remplace scipy.ndimage.gaussian_filter
  - Constantes render dédupliquées au niveau module
  - _encode_rgba helper centralisé
  - Hooks pour GPU reprojection (étape 2)

Refonte UI (audit UI/UX, lots A+B) :
  - Une seule couche de probabilité, rendue en JS depuis le data PNG de
    l'espèce active : filtre, opacité et bascule d'espèce agissent dessus.
  - Panneau latéral unique (espèce, légende-filtre, hotspots, couches,
    météo) à la place des panneaux flottants et du LayerControl.
  - Hotspots en JSON compact, marqueurs numérotés créés en JS (top N).
  - Rendu à la résolution de la grille (plus de suréchantillonnage).
  - Couches de diagnostic (zones exclues, TWI) seulement avec
    debug_layers=True (option CLI --debug-layers).

Cohérence (audit UI/UX, lot C) :
  - Une seule échelle de score, les 6 classes config.PROBABILITY_* — D38.
    Palette rouge → vert, la même sur tous les fonds (D45, remplace la
    palette ambre → brun de D38).
  - Emprise d'étude (Isère, couverture BDCharm-50) : contour, extérieur
    estompé, pas de score ni de hotspot affiché hors emprise — D39.
  - Libellés lisibles des essences et géologies (config.*_LABELS),
    couche iNaturalist liée à la morille, repères en gris neutre.

Terrain et téléphone (audit UI/UX, lot D) :
  - Clic ou toucher sur la carte → popup du point (score, coordonnées
    copiables, liens Itinéraire / Google Maps), comme les hotspots.
  - Bouton « Me localiser » (navigator.geolocation) : centre la carte et
    donne le score à la position de l'utilisateur, qui suit ses
    déplacements.
  - Meta viewport sans maximum-scale ni user-scalable=no (zoom du
    navigateur possible), cibles tactiles ≥ 44 px, état du panneau
    mémorisé (localStorage).

Hors ligne (D43) :
  - Leaflet, jQuery, leaflet-measure et les feuilles de base de
    Bootstrap 5 et 3 sont insérés dans la page depuis vendor/ (copies
    figées des fichiers des CDN de Folium, même rendu) ; seuls les fonds
    de carte demandent le réseau.
"""
from __future__ import annotations

import base64
import html
import io
import json
import logging
import mimetypes
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import matplotlib

matplotlib.use("Agg")

import folium  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from branca.element import (  # type: ignore[import-untyped]  # noqa: E402
    CssLink,
    Element,
    JavascriptLink,
)
from folium.elements import JSCSSMixin  # noqa: E402
from folium.plugins import MeasureControl  # noqa: E402
from folium.raster_layers import ImageOverlay  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_bounds  # noqa: E402
from rasterio.warp import (  # noqa: E402
    Resampling,
    calculate_default_transform,
    reproject,
)
from shapely.geometry import box as shapely_box  # noqa: E402

import _accel  # noqa: E402
import config  # noqa: E402
from utils import TO_MERC, TO_WGS84  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from shapely.geometry.base import BaseGeometry

    from scoring import MorilleScoring

logger = logging.getLogger("cartomorilles.visualize")

__all__ = ["MorilleVisualizer"]

# ───────────────────────────────────────────────────────────
# Constantes module
# ───────────────────────────────────────────────────────────
_NODATA: float = -9999.0
_MAX_HOTSPOT_MARKERS: int = 1000

# Taille max du raster reprojeté (par axe). Le rendu se fait à la
# résolution de la grille : le navigateur lisse l'image à l'affichage.
_RENDER_MAX_PX: int = 4000
# Hotspots affichés à l'ouverture (le panneau permet d'en montrer plus)
_DEFAULT_HOTSPOTS_SHOWN: int = 20

# PNG : compress_level=1 ≈ ×3 plus rapide que optimize=True, ~20 % plus gros
_PNG_COMPRESS: int = 1
# Nombre de workers pour encoding PNG parallèle
_PNG_WORKERS: int = 4

# Lissage masques éliminatoires (sigma en pixels-grille)
_ELIM_SMOOTH_SIGMA: float = 1.5
# Seuil post-reprojection bilinéaire (< 0.5 → couverture légèrement élargie)
_ELIM_MASK_THRESHOLD: float = 0.3

# Palette rouge → jaune → vert (ColorBrewer RdYlGn, D45) : une couleur
# par classe, la même sur tous les fonds de carte. Choix de l'utilisateur
# après la palette ambre → brun de D38.
_CMAP_STOPS: tuple[tuple[float, str], ...] = (
    (0.00, "#d73027"),
    (0.15, "#f46d43"),
    (0.30, "#fdae61"),
    (0.45, "#fee08b"),
    (0.60, "#a6d96a"),
    (0.75, "#1a9850"),
    (1.00, "#006837"),
)

_CMAP = mcolors.LinearSegmentedColormap.from_list("morilles", list(_CMAP_STOPS))

# Couches de diagnostic : zones exclues par facteur éliminatoire
_ELIM_LABEL_MAP: dict[str, tuple[str, str]] = {
    "urban": ("Exclu : zones urbaines", "#555555"),
    "urban_proximity": ("Exclu : abords urbains", "#777777"),
    "water": ("Exclu : eau et berge de 5 m", "#0077cc"),
    "nodata": ("Exclu : MNT manquant", "#000000"),
    "species": ("Exclu : essences", "#8b0000"),
    "geology": ("Exclu : géologie", "#800080"),
    "slope": ("Exclu : pente trop forte", "#ff8c00"),
    "altitude": ("Exclu : altitude", "#8b4513"),
    "twi": ("Exclu : sol engorgé (TWI)", "#1f4e9c"),
    "host": ("Exclu : sans arbre hôte", "#6d4c41"),
}

# ── Calques supplémentaires (D53) : PNG de codes 8 bits, colorés en JS ──
_EXTRA_NONE: int = 255   # pas de valeur, ou hors emprise d'étude
# Géologie : teintes des cartes géologiques (calcaires en bleus, formations
# meubles en jaunes et beiges, socle cristallin en roses)
_GEOLOGY_COLORS: dict[str, str] = {
    "calcaire": "#3f6fb5",
    "calcaire_marneux": "#76a1d6",
    "marne": "#93c9c3",
    "calcaire_lacustre": "#4aa6cc",
    "eboulis_calcaire": "#a9b7c9",
    "dolomie": "#7d6bba",
    "alluvions_calcaires": "#cfe3a0",
    "alluvions": "#f0e08a",
    "alluvions_recentes": "#fff3b5",
    "colluvions": "#e5cd9b",
    "moraine": "#bfb58d",
    "flysch": "#8c9a46",
    "sable": "#f2c35f",
    "molasse": "#e0995a",
    "gres": "#c4824c",
    "schiste": "#6e9277",
    "remblai": "#9a9a9a",
    "hydrographie": "#2c7fb8",
    "granite": "#e0607e",
    "gneiss": "#f08c6a",
    "siliceux": "#c27ba0",
    "unknown": "#d4d4d4",
}
# Essences BD Forêt v2 (champ ESSENCE) : feuillus en verts et bruns,
# conifères en bleus, classes sans essence précise en gris ou pastel
_TREE_RAW_LABELS: dict[str, tuple[str, str]] = {
    "Feuillus": ("Feuillus sans essence précise", "#b3cf99"),
    "Conifères": ("Conifères sans essence précise", "#9fb8d6"),
    "Mixte": ("Mélange feuillus et conifères", "#c7c0a3"),
    "NC": ("Lande ou pelouse", "#e8d9b0"),
    "NR": ("Forêt sans couvert arboré", "#d9d9d9"),
}
_TREE_COLORS: dict[str, str] = {
    "chene": "#4d9221",
    "chene_pubescent": "#b5a63a",
    "hetre": "#1b7837",
    "chataignier": "#a6611a",
    "robinier": "#e6ab02",
    "frene": "#66bd63",
    "orme": "#a6d96a",
    "charme": "#7fbc41",
    "erable": "#fdae61",
    "tilleul": "#d9ef8b",
    "bouleau": "#fee08b",
    "noisetier": "#dfc27d",
    "pommier": "#f1b6da",
    "poirier": "#de77ae",
    "platane": "#c2e699",
    "buis": "#00441b",
    "peuplier": "#5ab4ac",
    "aulne": "#35978f",
    "saule": "#80cdc1",
    "sapin": "#2166ac",
    "epicea": "#4393c3",
    "douglas": "#053061",
    "pin_sylvestre": "#8073ac",
    "meleze": "#92c5de",
    "unknown": "#bdbdbd",
}
_EXTRA_FALLBACK_COLORS: tuple[str, ...] = (
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf",
)
# Rampes continues : (valeur, couleur) ; repères de légende régulièrement
# espacés sur la rampe. TWI : du sec (brun) à l'engorgé (bleu nuit).
_TWI_STOPS: tuple[tuple[float, str], ...] = (
    (2.0, "#8c510a"), (5.0, "#d8b365"), (8.0, "#f6e8c3"), (11.0, "#c7eae5"),
    (14.0, "#5ab4ac"), (17.0, "#01665e"), (20.0, "#08306b"),
)
_TWI_TICKS: tuple[tuple[float, str], ...] = (
    (2.0, "2"), (8.0, "8"), (14.0, "14"), (20.0, "20"),
)
_SLOPE_STOPS: tuple[tuple[float, str], ...] = (
    (0.0, "#ffffcc"), (10.0, "#ffeda0"), (20.0, "#feb24c"), (30.0, "#f03b20"),
    (45.0, "#bd0026"), (60.0, "#54278f"),
)
_SLOPE_TICKS: tuple[tuple[float, str], ...] = (
    (0.0, "0°"), (15.0, "15°"), (30.0, "30°"), (45.0, "45°"), (60.0, "60°"),
)
# Aire drainée D8 en log10(m²) : crêtes (orangé), versants (pâle), talwegs
# et rivières (bleus)
_AREA_STOPS: tuple[tuple[float, str], ...] = (
    (3.0, "#f4a259"), (4.0, "#fff5c2"), (5.0, "#c6dbef"), (6.0, "#6baed6"),
    (7.0, "#2171b5"), (9.0, "#08306b"),
)
_AREA_TICKS: tuple[tuple[float, str], ...] = (
    (3.0, "0,1 ha"), (5.0, "10 ha"), (7.0, "10 km²"), (9.0, "1 000 km²"),
)
# Couleur des observations iNaturalist (hors de la palette des scores)
_INAT_COLOR: str = "#6a3d9a"
# Les observations iNaturalist sont des Morchella (species_enricher) :
# couche affichée quand une espèce de ce taxon est active (grise ou noire, D49)
_INAT_TAXON: str = "Morchella"
# Repères : gris neutre, forme losange (≠ hotspots ronds numérotés)
_LANDMARK_COLOR: str = "#5f6368"
# Emprise d'étude : contour sombre sur halo blanc, extérieur voilé de blanc
_AREA_LINE: str = "#2b2b2b"
_AREA_VEIL_OPACITY: float = 0.5

# Encodage data-PNG (score/altitude/pente dans RGBA)
_SCORE_SCALE: int = 200
_SCORE_OUT: int = 254   # hors emprise d'étude (pas de score affiché)
_SCORE_NAN: int = 255
_ALT_OFFSET: int = 500
_SLOPE_OFFSET: int = 1
_SLOPE_SCALE: float = 2.0

# Bibliothèques de la carte intégrées à la page (D43) : copies figées,
# dans vendor/, des fichiers que Folium chargerait depuis des CDN. La
# carte fonctionne ainsi sans réseau, fonds de carte mis à part. Clé =
# URL du lien Folium : si une version de Folium en change une, son lien
# CDN est gardé (avertissement) plutôt que de mélanger deux versions.
_VENDOR_DIR: Path = Path(__file__).resolve().parent / "vendor"
_VENDORED: dict[str, str] = {
    "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js":
        "leaflet-1.9.3/leaflet.js",
    "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css":
        "leaflet-1.9.3/leaflet.css",
    # Folium construit ses popups avec $(…)
    "https://code.jquery.com/jquery-3.7.1.min.js":
        "jquery-3.7.1/jquery.min.js",
    "https://cdn.jsdelivr.net/gh/ljagis/leaflet-measure@2.1.7/dist/"
    "leaflet-measure.min.js":
        "leaflet-measure-2.1.7/leaflet-measure.min.js",
    # jsDelivr sert ce .min.css à partir du .css, déjà minifié
    "https://cdn.jsdelivr.net/gh/ljagis/leaflet-measure@2.1.7/dist/"
    "leaflet-measure.min.css":
        "leaflet-measure-2.1.7/leaflet-measure.css",
    # Aucune classe Bootstrap sur la page, mais le rendu dépend des deux
    # feuilles de base que Folium charge : reboot de Bootstrap 5 (box-sizing,
    # [hidden] !important, titres, marges)...
    "https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css":
        "bootstrap-5.2.2/bootstrap-reboot.min.css",
    # ... puis normalize + scaffolding de Bootstrap 3, livrés dans ce
    # fichier « glyphicons » : html à 62,5 % (1rem = 10 px, d'où la taille
    # du texte des popups et des contrôles), texte 14 px #333. Copie sans
    # les glyphicons ni leurs polices, inutilisés.
    "https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css":
        "bootstrap-3.0.0/bootstrap-glyphicons.base.css",
}
# Folium crée chaque Marker avec l'icône par défaut de Leaflet, puis lui
# applique son icône (repères en DivIcon). Avec la CSS insérée, Leaflet ne
# trouve plus le dossier de ces images et les demanderait à côté de la
# page (marker-icon.png en 404) : elles lui sont données en data URI.
_LEAFLET_ICONS: dict[str, str] = {
    "iconUrl": "leaflet-1.9.3/images/marker-icon.png",
    "iconRetinaUrl": "leaflet-1.9.3/images/marker-icon-2x.png",
    "shadowUrl": "leaflet-1.9.3/images/marker-shadow.png",
}
# Liens Folium sans effet sur cette carte : JS de Bootstrap (aucun
# composant), Font Awesome et awesome-markers (repères et hotspots en
# DivIcon). Styles calculés identiques sans eux (D43).
_UNUSED_LIBS: frozenset[str] = frozenset({
    "bootstrap",
    "awesome_markers",
    "awesome_markers_font_css",
    "awesome_markers_css",
    "awesome_rotate_css",
})


# ───────────────────────────────────────────────────────────
# Helpers module-level
# ───────────────────────────────────────────────────────────
def _default_landmarks() -> list[dict[str, Any]]:
    """Points de repère par défaut pour l'Isère."""
    return [
        {"name": "Grenoble", "lat": 45.1885, "lon": 5.7245,
         "info": "Préfecture de l'Isère"},
        {"name": "Voiron", "lat": 45.3650, "lon": 5.5910,
         "info": "Porte de la Chartreuse"},
        {"name": "Vizille", "lat": 45.0770, "lon": 5.7720,
         "info": "Vallée de la Romanche"},
        {"name": "La Mure", "lat": 44.9040, "lon": 5.7840,
         "info": "Plateau matheysin"},
        {"name": "Sassenage", "lat": 45.2130, "lon": 5.6620,
         "info": "Cuves de Sassenage — calcaire"},
    ]


def _encode_rgba(
    rgba: np.ndarray,
    *,
    compress_level: int = _PNG_COMPRESS,
) -> bytes:
    """Encode un array RGBA uint8 en PNG bytes.

    Centralise la compression rapide (compress_level=1 par défaut).
    PIL relâche le GIL pendant la compression → thread-safe.
    """
    with Image.fromarray(
        np.asarray(rgba, dtype=np.uint8), mode="RGBA",
    ) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=compress_level)
        return buf.getvalue()


def _class_alphas(n_classes: int) -> list[float]:
    """Alpha de rendu par classe : 0 pour « Nul » (transparent), rampe
    0.25 → 0.55, puis 0.70 pour la meilleure classe."""
    if n_classes <= 1:
        return [0.0] * n_classes
    ramp = np.linspace(0.15, 0.65, n_classes)[1:]
    alphas = [0.0, *(round(float(a), 3) for a in ramp)]
    alphas[-1] = 0.70
    return alphas


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    """Luminance relative WCAG d'une couleur sRGB [0, 1]."""
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _class_swatches(
    thresholds: list[float],
    cmap: mcolors.Colormap = _CMAP,
) -> tuple[list[str], list[str]]:
    """Couleur de chaque classe (colormap au milieu de la classe, donc
    fidèle à la carte) et couleur de texte la plus contrastée dessus."""
    bounds = [0.0, *thresholds, 1.0]
    fills: list[str] = []
    texts: list[str] = []
    dark_lum = _relative_luminance(mcolors.to_rgb("#1b1b1b"))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rgb = mcolors.to_rgb(cmap((lo + hi) / 2.0))
        fills.append(mcolors.to_hex(rgb))
        lum = _relative_luminance(rgb)
        on_white = 1.05 / (lum + 0.05)
        on_dark = (lum + 0.05) / (dark_lum + 0.05)
        texts.append("#ffffff" if on_white >= on_dark else "#1b1b1b")
    return fills, texts


def _palette_cfg(
    stops: tuple[tuple[float, str], ...],
    cmap: mcolors.Colormap,
    thresholds: list[float],
) -> dict[str, Any]:
    """Palette telle que la consomme le JS (LUT, légende, marqueurs)."""
    swatches, swatch_text = _class_swatches(thresholds, cmap)
    return {
        "stops": [[pos, col] for pos, col in stops],
        "swatches": swatches,
        "swatchText": swatch_text,
    }


def _dissolve_study_area(study_area: Any) -> BaseGeometry | None:
    """Emprise d'étude en Lambert-93 : union des polygones fournis
    (BDCharm-50 départemental = couverture sans recouvrement, d'où
    coverage_union_all ≈ 1 s contre ≈ 10 s pour union_all)."""
    if study_area is None:
        return None
    if isinstance(study_area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        geoms = (
            study_area.geometry
            if isinstance(study_area, gpd.GeoDataFrame)
            else study_area
        )
        if geoms.crs is not None and geoms.crs.to_epsg() != 2154:
            geoms = geoms.to_crs(epsg=2154)
        arr = np.asarray(geoms[~(geoms.isna() | geoms.is_empty)].values)
        if arr.size == 0:
            return None
        try:
            geom = shapely.coverage_union_all(arr)
        except shapely.errors.GEOSException:
            geom = shapely.union_all(arr)
    else:
        geom = study_area
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return geom


def _polygons(geom: BaseGeometry) -> list[Any]:
    """Polygones d'une géométrie (Polygon, MultiPolygon, collection)."""
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _polygons(g)]
    return []


def _display_value(value: Any) -> str | None:
    """Libellé d'une clé d'essence OU de géologie quand le type n'est pas
    connu (ex. export GPX) ; None si vide ou « unknown ». Sinon, préférer
    config.tree_label / config.geology_label."""
    key = "" if value is None else str(value).strip()
    if key.lower() in {"", "unknown", "none", "nan"}:
        return None
    if key in config.TREE_LABELS:
        return config.tree_label(key)
    return config.geology_label(key)


def _js_json(obj: Any) -> str:
    """JSON injectable dans un <script> (« </ » échappé)."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


# Folium impose « maximum-scale=1.0, user-scalable=no » : le pincement
# hors carte (panneau, popups) ne pourrait plus agrandir le texte.
_VIEWPORT_RE = re.compile(r'<meta\s+name="viewport"[^>]*>', re.IGNORECASE)
_VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'


def _allow_page_zoom(html_doc: str) -> str:
    """Remplace le meta viewport de Folium par une version qui laisse le
    navigateur zoomer (accessibilité). Le pincement sur la carte reste
    capté par Leaflet."""
    fixed, n = _VIEWPORT_RE.subn(_VIEWPORT_META, html_doc, count=1)
    if not n:
        logger.warning("Meta viewport Folium introuvable : zoom navigateur non vérifié")
    return fixed


class _RawHtml(Element):  # type: ignore[misc]
    """Élément rendu tel quel : le code tiers ne passe pas par Jinja."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def render(self, **kwargs: Any) -> str:
        return self._text


_CSS_URL_RE = re.compile(r"""url\(\s*(['"]?)([^'")\s]+)\1\s*\)""")


def _data_uri(path: Path) -> str:
    """Image de vendor/ en data URI."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _vendored_tag(rel: str) -> str:
    """<script> ou <style> avec le contenu d'un fichier de vendor/ ; les
    images d'une feuille de style y sont mises en data URI."""
    path = _VENDOR_DIR / rel
    text = path.read_text(encoding="utf-8")
    tag = "script"
    if path.suffix == ".css":
        tag = "style"

        def _inline_url(m: re.Match[str]) -> str:
            ref = m.group(2)
            if ref.startswith(("data:", "#", "/", "http:", "https:")):
                return m.group(0)
            return f'url("{_data_uri(path.parent / ref)}")'

        text = _CSS_URL_RE.sub(_inline_url, text)
    if f"</{tag}" in text.lower():
        raise ValueError(f"« </{tag} » dans {rel}")
    return f"<{tag}>/* vendor/{rel} */\n{text}\n</{tag}>"


def _leaflet_icons_tag() -> str:
    """Icône par défaut de Leaflet en data URI (voir _LEAFLET_ICONS)."""
    urls = {opt: _data_uri(_VENDOR_DIR / rel) for opt, rel in _LEAFLET_ICONS.items()}
    return (
        "<script>L.Icon.Default.imagePath = '';\n"
        f"L.Icon.Default.mergeOptions({json.dumps(urls)});</script>"
    )


def _walk(element: Element) -> Iterator[Element]:
    """L'élément et ses descendants, dans l'ordre de rendu de Folium."""
    yield element
    for child in element._children.values():
        yield from _walk(child)


def _embed_libraries(folium_map: folium.Map) -> None:
    """Remplace les liens CDN des éléments Folium par le contenu des
    fichiers de vendor/ (même ordre dans <head>) et retire les
    bibliothèques inutilisées. Sans copie conforme, le lien CDN reste."""
    header = folium_map.get_root().header  # type: ignore[attr-defined]
    for element in _walk(folium_map):
        if not isinstance(element, JSCSSMixin):
            continue
        for links, link_cls in (
            (element.default_js, JavascriptLink),
            (element.default_css, CssLink),
        ):
            for name, url in links:
                if name in _UNUSED_LIBS:
                    continue
                rel = _VENDORED.get(url)
                reason = "version absente de vendor/"
                if rel is not None:
                    try:
                        tag = _vendored_tag(rel)
                        if name == "leaflet":
                            tag += "\n" + _leaflet_icons_tag()
                        header.add_child(_RawHtml(tag), name=name)
                        continue
                    except (OSError, ValueError) as exc:
                        reason = str(exc)
                logger.warning(
                    "Carte : %s chargé depuis %s (%s), indispensable hors ligne",
                    name, url, reason,
                )
                header.add_child(link_cls(url), name=name)
        # Attributs d'instance : les listes de classe de Folium restent
        # intactes pour les autres cartes.
        element.default_js = []
        element.default_css = []


class _OverlaySpec(NamedTuple):
    """Couche de diagnostic à encoder en PNG puis à ajouter à la carte."""

    key: str                     # identifiant unique (ex. "geology-esculenta")
    group: str                   # identique d'une espèce à l'autre
    label: str                   # libellé du panneau
    species: str | None          # None = commune à toutes les espèces
    render: Callable[[], bytes]  # thread-safe
    opacity: float
    swatch: str                  # pastille du panneau
    log_msg: str


class _ExtraSpec(NamedTuple):
    """Calque supplémentaire (D53) : description pour le panneau et le JS,
    et rendu de son PNG de codes (None = lu dans le PNG terrain)."""

    cfg: dict[str, Any]
    render: Callable[[], bytes] | None


# ═══════════════════════════════════════════════════════════
# Classe principale
# ═══════════════════════════════════════════════════════════
class MorilleVisualizer:
    """Génère les sorties visuelles à partir d'un MorilleScoring terminé."""

    def __init__(
        self,
        scoring_model: "MorilleScoring | dict[str, MorilleScoring]",
        *,
        hotspots: "list[dict[str, Any]] | dict[str, list[dict[str, Any]]] | None" = None,
        landmarks: list[dict[str, Any]] | None = None,
        max_hotspot_markers: int = _MAX_HOTSPOT_MARKERS,
        hydro_gdf: gpd.GeoDataFrame | None = None,
        urban_gdf: gpd.GeoDataFrame | None = None,
        inat_observations: list[dict[str, Any]] | None = None,
        weather_days: list[Any] | None = None,
        debug_layers: bool = False,
        default_hotspots_shown: int = _DEFAULT_HOTSPOTS_SHOWN,
        study_area: Any = None,
        forest_gdf: gpd.GeoDataFrame | None = None,
    ) -> None:
        """study_area : emprise d'étude (GeoDataFrame dont on fait l'union,
        ou géométrie Lambert-93). Hors emprise, la carte n'affiche ni score
        ni hotspot et estompe le fond (D39). None = pas de restriction.
        forest_gdf : BD Forêt v2, pour le calque des essences telles que
        publiées (D53) ; sans elle, essences canoniques de la grille."""
        # ── v2.6.0 — accepter dict[species, MorilleScoring] OU instance unique
        if isinstance(scoring_model, dict):
            self.models: dict[str, MorilleScoring] = dict(scoring_model)
            if not self.models:
                raise ValueError("models dict empty")
            for sp, m in self.models.items():
                self._validate_model(m)
        else:
            self._validate_model(scoring_model)
            sp_name = getattr(scoring_model, "species", "esculenta")
            self.models = {sp_name: scoring_model}

        self.species_list: list[str] = list(self.models.keys())
        self.primary_species: str = self.species_list[0]
        # Legacy access — premier modèle = défaut
        self.model: MorilleScoring = self.models[self.primary_species]
        self.grid = self.model.grid

        # ── Hotspots dict OU list (backward-compat) ──
        if isinstance(hotspots, dict):
            self.hotspots_by_species: dict[str, list[dict[str, Any]]] = dict(hotspots)
        elif hotspots is None:
            self.hotspots_by_species = {sp: [] for sp in self.species_list}
        else:
            # Liste simple → assignée à l'espèce primaire
            self.hotspots_by_species = {self.primary_species: hotspots}
        # Fill missing species avec liste vide
        for sp in self.species_list:
            self.hotspots_by_species.setdefault(sp, [])
        # Legacy : self.hotspots = liste de l'espèce primaire
        self.hotspots: list[dict[str, Any]] = self.hotspots_by_species[self.primary_species]

        self.landmarks: list[dict[str, Any]] = landmarks or _default_landmarks()
        self.max_hotspot_markers = max_hotspot_markers
        self._hydro_gdf = hydro_gdf
        self._urban_gdf = urban_gdf
        self._forest_gdf = forest_gdf
        # Genre Morchella seulement : la requête « Morchella » ramène aussi
        # Verpa bohemica (synonyme Morchella bohemica), pas une morille (D60)
        self._inat_observations: list[dict[str, Any]] = [
            o for o in (inat_observations or [])
            if str(o.get("taxon_name") or "Morchella").split()[0] == "Morchella"
        ]
        self._weather_days: list[Any] = weather_days or []
        self.debug_layers = debug_layers
        self.default_hotspots_shown = max(1, int(default_hotspots_shown))

        # Stocker raw scores par espèce pour exports per-species
        self._final_score_by_species: dict[str, np.ndarray] = {}
        self._prob_classes_by_species: dict[str, np.ndarray] = {}
        self._elim_mask_by_species: dict[str, np.ndarray] = {}
        self._elim_detail_by_species: dict[str, dict[str, Any]] = {}
        for sp, m in self.models.items():
            assert m.final_score is not None
            assert m.probability_classes is not None
            assert m.elimination_mask is not None
            assert m.elimination_detail is not None
            self._final_score_by_species[sp] = m.final_score
            self._prob_classes_by_species[sp] = m.probability_classes
            self._elim_mask_by_species[sp] = m.elimination_mask
            self._elim_detail_by_species[sp] = m.elimination_detail

        # Legacy — pointeurs sur l'espèce primaire
        self._final_score: np.ndarray = self._final_score_by_species[self.primary_species]
        self._prob_classes: np.ndarray = self._prob_classes_by_species[self.primary_species]
        self._elim_mask: np.ndarray = self._elim_mask_by_species[self.primary_species]
        self._elim_detail: dict[str, Any] = self._elim_detail_by_species[self.primary_species]

        self._to_wgs84 = TO_WGS84

        half = config.CELL_SIZE / 2.0
        self._xmin_l93 = float(np.min(self.grid.x_coords)) - half
        self._xmax_l93 = float(np.max(self.grid.x_coords)) + half
        self._ymin_l93 = float(np.min(self.grid.y_coords)) - half
        self._ymax_l93 = float(np.max(self.grid.y_coords)) + half

        # Détection orientation y pour flip
        _yc = np.asarray(self.grid.y_coords)
        self._y_ascending: bool = bool(_yc.size > 1 and _yc[-1] > _yc[0])

        # Reprojection L93 → Web Mercator (EPSG:3857) pour overlays Folium.
        # Leaflet affiche en Mercator : les pixels doivent être espacés
        # linéairement en Mercator Y, pas en latitude (EPSG:4326).
        self._src_transform = from_bounds(
            self._xmin_l93,
            self._ymin_l93,
            self._xmax_l93,
            self._ymax_l93,
            self.grid.nx,
            self.grid.ny,
        )
        self._src_crs = CRS.from_epsg(2154)
        self._dst_crs = CRS.from_epsg(3857)

        # Résolution de rendu = celle de la grille (le suréchantillonnage
        # n'ajoutait aucune information et multipliait le poids du HTML).
        render_nx = min(int(self.grid.nx), _RENDER_MAX_PX)
        render_ny = min(int(self.grid.ny), _RENDER_MAX_PX)

        # Emprise WGS84 exacte depuis pyproj (pour bounds Folium)
        _corners_x = [self._xmin_l93, self._xmax_l93, self._xmin_l93, self._xmax_l93]
        _corners_y = [self._ymin_l93, self._ymin_l93, self._ymax_l93, self._ymax_l93]
        _cx, _cy = self._to_wgs84.transform(_corners_x, _corners_y)
        self._west = float(min(_cx))
        self._east = float(max(_cx))
        self._south = float(min(_cy))
        self._north = float(max(_cy))

        # Bounds Mercator pour la destination de reprojection
        _mx, _my = TO_MERC.transform(_corners_x, _corners_y)
        _xmin_merc = float(min(_mx))
        _xmax_merc = float(max(_mx))
        _ymin_merc = float(min(_my))
        _ymax_merc = float(max(_my))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            self._src_crs,
            self._dst_crs,
            render_nx,
            render_ny,
            left=self._xmin_l93,
            bottom=self._ymin_l93,
            right=self._xmax_l93,
            top=self._ymax_l93,
        )
        assert dst_width is not None
        assert dst_height is not None

        # dst_transform depuis bounds Mercator exactes pour alignement
        # pixel-parfait avec le fond de carte Leaflet (Web Mercator).
        self._dst_width: int = dst_width
        self._dst_height: int = dst_height
        self._dst_transform = from_bounds(
            _xmin_merc, _ymin_merc,
            _xmax_merc, _ymax_merc,
            dst_width, dst_height,
        )

        # Bounds Folium en WGS84 (lat/lng) — Leaflet les convertit en Mercator
        self._bounds: list[list[float]] = [
            [self._south, self._west],
            [self._north, self._east],
        ]

        logger.debug(
            "Emprise WGS84 : S=%.5f N=%.5f W=%.5f E=%.5f",
            self._south, self._north, self._west, self._east,
        )
        logger.debug(
            "Raster WGS84 : %dx%d px", self._dst_width, self._dst_height,
        )

        # Emprise d'étude ∩ grille (D39) + masque au pas du rendu Mercator
        self._study_area: BaseGeometry | None = None
        self._study_mask: np.ndarray | None = None
        area = _dissolve_study_area(study_area)
        if area is not None:
            grid_box = shapely_box(
                self._xmin_l93, self._ymin_l93, self._xmax_l93, self._ymax_l93,
            )
            area = area.intersection(grid_box)
            # Écarter les miettes de découpe (< 10 ha)
            parts = [p for p in _polygons(area) if p.area >= 1e5]
            if parts:
                self._study_area = shapely.multipolygons(parts)
                shapely.prepare(self._study_area)
                self._study_mask = self._rasterize_study_mask(self._study_area)
                logger.info(
                    "✅ Emprise d'étude : %.0f km² (%.0f %% de la grille)",
                    self._study_area.area / 1e6,
                    self._study_area.area / grid_box.area * 100,
                )

    # ──────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _validate_model(model: Any) -> None:
        required = [
            "final_score",
            "probability_classes",
            "grid",
            "elimination_mask",
            "elimination_detail",
        ]
        missing = [a for a in required if not hasattr(model, a)]
        if missing:
            raise AttributeError(
                f"MorilleScoring incomplet — manquants : {missing}"
            )
        for attr in required[:2] + required[3:]:
            if getattr(model, attr) is None:
                raise ValueError(
                    f"MorilleScoring.{attr} is None — pipeline incomplet ?"
                )
        grid_attrs = [
            "x_coords", "y_coords", "transform", "nx", "ny",
            "urban_mask", "nodata_mask", "scores",
        ]
        grid_missing = [a for a in grid_attrs if not hasattr(model.grid, a)]
        if grid_missing:
            raise AttributeError(
                f"GridBuilder incomplet — manquants : {grid_missing}"
            )

    # ──────────────────────────────────────────────────────
    # Coordonnées & orientation
    # ──────────────────────────────────────────────────────
    def _l93_to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lon), float(lat)

    def _orient_for_overlay(self, arr: np.ndarray) -> np.ndarray:
        """Oriente un array 2D pour que row 0 = nord."""
        if self._y_ascending:
            return np.asarray(arr[::-1, :])
        return np.asarray(arr)

    def _reproject_to_wgs84(
        self,
        arr: np.ndarray,
        *,
        is_mask: bool = False,
        nearest: bool = False,
    ) -> np.ndarray:
        """Reprojette un array 2D L93 → grille de rendu Web Mercator.

        Bilinéaire, ou plus proche voisin (`nearest`, codes catégoriels).
        Masques seuillés à 0.5 après reprojection.
        """
        oriented = self._orient_for_overlay(arr)
        src = oriented.astype(np.float32)
        src_h, src_w = src.shape

        src_transform = from_bounds(
            self._xmin_l93, self._ymin_l93,
            self._xmax_l93, self._ymax_l93,
            src_w, src_h,
        )

        dst = np.full(
            (self._dst_height, self._dst_width), np.nan, dtype=np.float32,
        )
        reproject(
            source=src,
            destination=dst,
            src_transform=src_transform,
            src_crs=self._src_crs,
            dst_transform=self._dst_transform,
            dst_crs=self._dst_crs,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
            # Sans nodata, GDAL écrit 0 hors de l'emprise L93 tournée :
            # « 0 m, 0° » et score Nul y étaient lus au lieu de « pas de
            # données » (D60)
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        if is_mask:
            return np.asarray(np.nan_to_num(dst, nan=0.0) > 0.5)
        return np.asarray(dst)

    # ──────────────────────────────────────────────────────
    # Emprise d'étude (D39)
    # ──────────────────────────────────────────────────────
    def _rasterize_study_mask(self, area: BaseGeometry) -> np.ndarray:
        """Masque bool « dans l'emprise » sur la grille de rendu Mercator
        (mêmes pixels que les data PNG, D29)."""
        merc = shapely.transform(
            area, lambda xy: np.column_stack(TO_MERC.transform(xy[:, 0], xy[:, 1])),
        )
        mask = rasterize(
            [(merc, 1)],
            out_shape=(self._dst_height, self._dst_width),
            transform=self._dst_transform,
            fill=0,
            dtype="uint8",
        )
        return np.asarray(mask, dtype=bool)

    def _study_area_geojson(self) -> dict[str, Any] | None:
        """Contour (Polygon/MultiPolygon) et voile extérieur (monde troué
        par l'emprise) en GeoJSON WGS84, simplifiés à l'échelle de la
        maille et arrondis à ~1 m."""
        if self._study_area is None:
            return None
        tol = max(float(config.CELL_SIZE) / 2.0, 10.0)
        simple = self._study_area.simplify(tol, preserve_topology=True)

        def ring(coords: Any) -> list[list[float]]:
            xy = np.asarray(coords)[:, :2]
            lon, lat = self._to_wgs84.transform(xy[:, 0], xy[:, 1])
            return [
                [round(float(x), 5), round(float(y), 5)]
                for x, y in zip(lon, lat)
            ]

        polys = [
            [ring(p.exterior.coords), *(ring(h.coords) for h in p.interiors)]
            for p in _polygons(simple)
        ]
        world = [[-180.0, -85.0], [180.0, -85.0], [180.0, 85.0],
                 [-180.0, 85.0], [-180.0, -85.0]]
        return {
            "outline": {"type": "MultiPolygon", "coordinates": polys},
            "veil": {
                "type": "Polygon",
                "coordinates": [world, *(p[0] for p in polys)],
            },
        }

    @staticmethod
    def observation_species(species_list: Sequence[str]) -> list[str]:
        """Espèces de la carte dont le taxon naturaliste est Morchella : la
        couche d'observations iNaturalist est affichée quand l'une d'elles
        est active (D49)."""
        morels = set(config.species_with_taxon(_INAT_TAXON))
        return [sp for sp in species_list if sp in morels]

    def inside_study_area(self, hotspots: list[dict[str, Any]]) -> np.ndarray:
        """Hotspots (centroïdes L93) situés dans l'emprise d'étude — même
        filtre pour la carte et l'export GPX de main.py."""
        if self._study_area is None or not hotspots:
            return np.ones(len(hotspots), dtype=bool)
        xs = np.array([float(h["x_l93"]) for h in hotspots])
        ys = np.array([float(h["y_l93"]) for h in hotspots])
        return np.asarray(shapely.contains_xy(self._study_area, xs, ys))

    # ──────────────────────────────────────────────────────
    # Renderers statiques (thread-safe, pas d'état mutable)
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _colorize_mask(
        mask: np.ndarray,
        color_hex: str,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Convertit un masque bool 2D → RGBA uint8 monochrome."""
        m = np.asarray(mask, dtype=bool)
        r_f, g_f, b_f = mcolors.hex2color(color_hex)
        h, w = m.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[m, 0] = int(r_f * 255)
        rgba[m, 1] = int(g_f * 255)
        rgba[m, 2] = int(b_f * 255)
        rgba[m, 3] = int(alpha * 255)
        return rgba

    @staticmethod
    def _png_to_data_uri(png_bytes: bytes) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # ──────────────────────────────────────────────────────
    # Couches de diagnostic (option debug_layers)
    # ──────────────────────────────────────────────────────
    def _mask_render_fn(
        self,
        mask: np.ndarray,
        key: str,
        color: str,
    ) -> Callable[[], bytes]:
        """Lisse et reprojette un masque bool → fonction de rendu PNG."""
        # Sigma réduit pour la pente : préserver les bandes étroites
        # de falaises au lieu de les éroder par excès de lissage.
        sigma = 0.5 if key == "slope" else _ELIM_SMOOTH_SIGMA
        mask_f = np.asarray(
            _accel.gaussian_filter(
                mask.astype(np.float32),
                sigma=sigma,
                mode="reflect",
            ),
        )
        reprojected_f = self._reproject_to_wgs84(mask_f, is_mask=False)
        reprojected = np.asarray(
            np.nan_to_num(reprojected_f, nan=0.0) > _ELIM_MASK_THRESHOLD,
        )

        def render(_r: np.ndarray = reprojected, _c: str = color) -> bytes:
            rgba = MorilleVisualizer._colorize_mask(_r, _c, alpha=0.45)
            return _encode_rgba(rgba)

        return render

    def _prepare_diagnostics(self) -> list[_OverlaySpec]:
        """Couches de diagnostic : zones exclues par espèce (un seul
        exemplaire quand le masque est identique pour toutes les espèces,
        ex. urbain ou eau). Le TWI est un calque supplémentaire (D53)."""
        specs: list[_OverlaySpec] = []

        masks: dict[str, dict[str, np.ndarray]] = {}
        for species in self.species_list:
            detail = self._elim_detail_by_species.get(species) or {}
            for key, arr in detail.items():
                if not isinstance(arr, np.ndarray):
                    continue
                if arr.shape != self._final_score.shape:
                    continue
                mask = np.asarray(arr, dtype=bool)
                if np.any(mask):
                    masks.setdefault(key, {})[species] = mask

        for key, by_species in masks.items():
            label, color = _ELIM_LABEL_MAP.get(key, (f"Exclu : {key}", "#ff0000"))
            first = next(iter(by_species.values()))
            shared = len(by_species) == len(self.species_list) and all(
                np.array_equal(first, m) for m in by_species.values()
            )
            targets: list[tuple[str | None, np.ndarray]] = (
                [(None, first)] if shared else list(by_species.items())
            )
            for species, mask in targets:
                suffix = "" if species is None else f"-{species}"
                n = int(np.sum(mask))
                specs.append(_OverlaySpec(
                    key=f"elim-{key}{suffix}",
                    group=f"elim-{key}",
                    label=label,
                    species=species,
                    render=self._mask_render_fn(mask, key, color),
                    opacity=1.0,
                    swatch=color,
                    log_msg=(
                        f"Diagnostic {label} "
                        f"[{species or 'toutes espèces'}] : {n} cellules"
                    ),
                ))

        logger.info("✅ %d couches de diagnostic préparées", len(specs))
        return specs

    # ──────────────────────────────────────────────────────
    # Calques supplémentaires (D53) : un PNG de codes 8 bits par calque,
    # coloré par le JS ; 255 = pas de valeur ou hors emprise d'étude
    # ──────────────────────────────────────────────────────
    def _extra_codes(
        self,
        arr: np.ndarray,
        *,
        nearest: bool = False,
        vmin: float = 0.0,
        step: float = 1.0,
        max_code: int = _EXTRA_NONE - 1,
    ) -> np.ndarray:
        """Grille L93 → codes uint8 sur la grille de rendu Mercator :
        round((valeur − vmin) / step) borné à [0, max_code]."""
        rep = self._reproject_to_wgs84(
            np.asarray(arr, dtype=np.float32), nearest=nearest,
        )
        codes = np.full(rep.shape, _EXTRA_NONE, dtype=np.uint8)
        ok = np.isfinite(rep)
        codes[ok] = np.clip(
            np.rint((rep[ok] - vmin) / step), 0, max_code,
        ).astype(np.uint8)
        if self._study_mask is not None:
            codes[~self._study_mask] = _EXTRA_NONE
        return codes

    def _extra_shares(self, codes: np.ndarray) -> np.ndarray:
        """Part de la zone étudiée couverte par chaque code (sans emprise
        d'étude : part de la zone renseignée)."""
        counts = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
        total = (
            float(np.count_nonzero(self._study_mask))
            if self._study_mask is not None
            else float(counts[:_EXTRA_NONE].sum())
        )
        return counts / max(total, 1.0)

    def _categorical_extra(
        self,
        cfg: dict[str, Any],
        raster: np.ndarray,
        classes: dict[int, tuple[str, str]],
    ) -> _ExtraSpec | None:
        """Calque catégoriel : `raster` = codes > 0 sur la grille (0 = rien),
        `classes` = code → (libellé, couleur). Légende triée par part de
        la zone étudiée, classes absentes de la zone omises."""
        src = np.asarray(raster)
        if src.shape != self._final_score.shape or not classes:
            return None
        if max(classes) >= _EXTRA_NONE:
            logger.warning("Calque %s : trop de classes, ignoré", cfg["label"])
            return None
        codes = self._extra_codes(
            np.where(src > 0, src, np.nan), nearest=True,
        )
        shares = self._extra_shares(codes)
        legend = [
            {"code": int(code), "label": label, "color": color,
             "share": round(float(shares[code]), 4)}
            for code, (label, color) in classes.items()
            if shares[code] > 0
        ]
        if not legend:
            return None
        legend.sort(key=lambda c: -float(c["share"]))
        cfg = {**cfg, "kind": "cat", "none": _EXTRA_NONE, "classes": legend}
        logger.info(
            "✅ Calque %s : %d classes, %.0f %% de la zone étudiée",
            cfg["label"], len(legend), sum(float(c["share"]) for c in legend) * 100,
        )
        return _ExtraSpec(cfg, lambda _c=codes: self._encode_opaque_png(_c))

    def _numeric_extra(
        self,
        cfg: dict[str, Any],
        arr: np.ndarray | None,
        *,
        vmin: float,
        step: float,
        vmax: float,
    ) -> _ExtraSpec | None:
        """Calque continu : valeur = vmin + code × step (code ≤ 254)."""
        if not isinstance(arr, np.ndarray) or arr.shape != self._final_score.shape:
            return None
        if not np.any(np.isfinite(arr)):
            return None
        max_code = int(round((vmax - vmin) / step))
        codes = self._extra_codes(arr, vmin=vmin, step=step, max_code=max_code)
        cfg = {**cfg, "kind": "num", "none": _EXTRA_NONE, "vmin": vmin,
               "step": step, "maxCode": max_code}
        logger.info("✅ Calque %s préparé", cfg["label"])
        return _ExtraSpec(cfg, lambda _c=codes: self._encode_opaque_png(_c))

    def _tree_extra_classes(self) -> tuple[np.ndarray, dict[int, tuple[str, str]]] | None:
        """Essences BD Forêt v2 telles que publiées (champ ESSENCE : une
        essence, ou feuillus / conifères / mélange sans essence précise,
        lande ou pelouse), rasterisées comme le modèle (all_touched) ; les
        classes sans essence précise d'abord, une essence l'emporte en cas
        de chevauchement. Sans ce champ : essences canoniques du modèle."""
        gdf = self._forest_gdf
        if gdf is not None and not gdf.empty and "ESSENCE" in gdf.columns:
            if gdf.crs is not None and gdf.crs.to_epsg() != 2154:
                gdf = gdf.to_crs(epsg=2154)
            geom = gdf.geometry
            gdf = gdf.loc[geom.notna() & ~geom.is_empty & geom.is_valid]
            raw = gdf["ESSENCE"].fillna("NR").astype(str).str.strip()
            names = sorted(set(raw), key=lambda r: (r not in _TREE_RAW_LABELS, r))
            if len(names) < _EXTRA_NONE:
                code_of = {name: i + 1 for i, name in enumerate(names)}
                codes = raw.map(code_of).to_numpy(dtype=np.int16)
                order = np.argsort(codes, kind="stable")
                geoms = gdf.geometry.to_numpy()[order]
                raster = _accel.parallel_rasterize_categorical(
                    geometries=list(geoms),
                    category_codes=codes[order],
                    out_shape=(int(self.grid.ny), int(self.grid.nx)),
                    transform=self.grid.transform,
                    all_touched=True,
                    nodata=0,
                )
                fallback = iter(_EXTRA_FALLBACK_COLORS * 4)
                classes: dict[int, tuple[str, str]] = {}
                for name, code in code_of.items():
                    if name in _TREE_RAW_LABELS:
                        classes[code] = _TREE_RAW_LABELS[name]
                    else:
                        canon = config.resolve_tree_name(name)
                        classes[code] = (
                            name,
                            _TREE_COLORS.get(canon) or next(fallback),
                        )
                return np.asarray(raster), classes
        raster = getattr(self.grid, "_tree_species_int_raster", None)
        names_by_code = getattr(self.grid, "_tree_code_to_name", None) or {}
        if not isinstance(raster, np.ndarray) or not names_by_code:
            return None
        fallback = iter(_EXTRA_FALLBACK_COLORS * 4)
        return raster, {
            int(code): (
                "Essence non précisée" if name == "unknown"
                else config.tree_label(name) or str(name),
                _TREE_COLORS.get(str(name)) or next(fallback),
            )
            for code, name in names_by_code.items()
        }

    def _species_groups(self, key: Callable[[dict[str, Any]], Any]) -> list[tuple[Any, str]]:
        """Espèces de la carte regroupées par valeur de paramètre :
        [(valeur, « Morille grise et Morille noire »), …]."""
        groups: dict[Any, list[str]] = {}
        for sp in self.species_list:
            value = key(dict(config.get_species_params(sp)))
            groups.setdefault(value, []).append(self._species_info(sp)["name"])
        return [
            (value, " et ".join(
                [", ".join(names[:-1]), names[-1]] if len(names) > 1 else names,
            ))
            for value, names in groups.items()
        ]

    def _prepare_extra_layers(self) -> list[_ExtraSpec]:
        """Calques supplémentaires du panneau (D53) : nature du sol,
        essences forestières, humidité du sol (TWI), pente et aire drainée
        (la morphologie du calcul du TWI). Un seul est affiché à la fois."""
        specs: list[_ExtraSpec | None] = []

        def num(value: float) -> str:
            return f"{value:g}".replace(".", ",")

        # ── Nature du sol : catégories géologiques du modèle ──
        geo_raster = getattr(self.grid, "_geology_int_raster", None)
        geo_names = getattr(self.grid, "_geology_code_to_name", None) or {}
        if isinstance(geo_raster, np.ndarray) and geo_names:
            fallback = iter(_EXTRA_FALLBACK_COLORS * 4)
            excluded = [
                f"{names} : "
                + ", ".join(
                    (config.geology_label(g) or g).lower() for g in sorted(geos)
                )
                for geos, names in self._species_groups(
                    lambda p: frozenset(p.get("eliminatory_geology", ())),
                )
                if geos
            ]
            specs.append(self._categorical_extra(
                {
                    "key": "geology", "label": "Nature du sol (géologie)",
                    "short": "Sol",
                    "hint": "Carte géologique harmonisée BDCharm-50 (BRGM), "
                            "regroupée dans les catégories du modèle."
                            + (" Exclu pour " + " ; ".join(excluded) + "."
                               if excluded else ""),
                },
                geo_raster,
                {
                    int(code): (
                        "Non classée" if name == "unknown"
                        else config.geology_label(name) or str(name),
                        _GEOLOGY_COLORS.get(str(name)) or next(fallback),
                    )
                    for code, name in geo_names.items()
                },
            ))

        # ── Essences forestières (BD Forêt v2) ──
        trees = self._tree_extra_classes()
        if trees is not None:
            specs.append(self._categorical_extra(
                {
                    "key": "trees", "label": "Essences forestières",
                    "short": "Végétation", "noneText": "hors forêt",
                    "hint": "BD Forêt v2 (IGN) : essence du peuplement. Sans "
                            "essence précise (feuillus, conifères, mélange), "
                            "le modèle l'estime par la composition régionale "
                            "de l'inventaire forestier (IFN).",
                },
                trees[0], trees[1],
            ))

        # ── Humidité du sol : TWI ──
        twi_text = " ; ".join(
            f"{names} : optimum {num(opt[0])}–{num(opt[1])}, exclu au-delà "
            f"de {num(wl)}"
            for (opt, wl), names in self._species_groups(
                lambda p: (tuple(p.get("twi_optimal", (6.0, 10.0))),
                           float(p.get("twi_waterlog", 18.0))),
            )
        )
        specs.append(self._numeric_extra(
            {
                "key": "twi", "label": "Humidité du sol (TWI)", "unit": "twi",
                "hint": "Indice topographique d'humidité TWI = ln(aire drainée "
                        "/ tan pente) : faible sur les crêtes et les pentes "
                        "raides, fort dans les creux et replats où l'eau "
                        f"converge. {twi_text}.",
                "stops": [list(s) for s in _TWI_STOPS],
                "ticks": [list(t) for t in _TWI_TICKS],
                "ends": ["sec", "engorgé"],
            },
            getattr(self.grid, "twi", None), vmin=0.0, step=0.1, vmax=25.0,
        ))

        # ── Pente : lue dans le PNG terrain (déjà dans la page) ──
        slope = getattr(self.grid, "slope", None)
        if isinstance(slope, np.ndarray) and slope.shape == self._final_score.shape:
            opt_text = " ; ".join(
                f"{names} : optimum {num(opt[0])}–{num(opt[1])}°, exclu "
                f"au-delà de {num(smax)}°"
                for (opt, smax), names in self._species_groups(
                    lambda p: (tuple(p.get("slope_optimal", (0.0, 15.0))),
                               float(p.get("slope_max", 45.0))),
                )
            )
            specs.append(_ExtraSpec({
                "key": "slope", "label": "Pente", "unit": "deg", "kind": "num",
                "src": "terrain", "none": 0,
                "vmin": -_SLOPE_OFFSET / _SLOPE_SCALE, "step": 1 / _SLOPE_SCALE,
                "maxCode": 181,
                "hint": "Pente du MNT Copernicus (méthode de Horn), en degrés. "
                        f"{opt_text}.",
                "stops": [list(s) for s in _SLOPE_STOPS],
                "ticks": [list(t) for t in _SLOPE_TICKS],
                "ends": ["plat", "falaise"],
            }, None))

        # ── Aire drainée D8 : la morphologie du calcul du TWI ──
        specs.append(self._numeric_extra(
            {
                "key": "area", "label": "Aire drainée (morphologie)",
                "unit": "log10m2",
                "hint": "Surface en amont dont l'eau s'écoule vers chaque "
                        "point (écoulement D8 du MNT) : minimale sur les "
                        "crêtes, maximale dans les talwegs et les vallées. "
                        "Avec la pente, elle donne le TWI.",
                "stops": [list(s) for s in _AREA_STOPS],
                "ticks": [list(t) for t in _AREA_TICKS],
                "ends": ["crêtes", "talwegs, rivières"],
            },
            getattr(self.grid, "drained_area_log10", None),
            vmin=2.0, step=0.05, vmax=11.0,
        ))
        return [s for s in specs if s is not None]

    # ──────────────────────────────────────────────────────
    # Data PNG (score par espèce) + PNG terrain (pente + altitude)
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _encode_opaque_png(arr: np.ndarray) -> bytes:
        """PNG opaque (L ou RGB selon la forme) — Fix B1 : pas de canal
        alpha, donc pas de pré-multiplication par le navigateur."""
        with Image.fromarray(np.asarray(arr, dtype=np.uint8)) as img:
            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=6)
            return buf.getvalue()

    def _build_data_png_bytes(self, species: str | None = None) -> bytes:
        """PNG RGB du score d'une espèce, lu par le JS (rendu de la couche
        de probabilité + lecture de cellule).

        R = score × 200 (0–200 → 0.00–1.00), 254 = hors emprise d'étude
        (D39), 255 = NaN ; G = classe (indice dans PROBABILITY_LABELS) du
        score non quantifié sur les seuils de l'espèce (D52). Le R tronqué
        au 1/200 faisait tomber d'une classe 9 à 25 % des cellules Bon et
        Excellent dont le score est juste au-dessus d'un seuil (D60).
        Altitude et pente, communes aux espèces, sont dans le PNG terrain.

        v2.6.0 : species kwarg sélectionne le raster source. Si None → primary.
        """
        sp = species or self.primary_species
        score_src = self._final_score_by_species.get(sp, self._final_score)
        score_wgs = self._reproject_to_wgs84(score_src)
        valid_s = np.asarray(np.isfinite(score_wgs))
        rgb = np.zeros((*score_wgs.shape, 3), dtype=np.uint8)
        gray = np.full(score_wgs.shape, _SCORE_NAN, dtype=np.uint8)
        gray[valid_s] = np.clip(
            (score_wgs[valid_s] * _SCORE_SCALE).astype(np.int32),
            0, _SCORE_SCALE,
        ).astype(np.uint8)
        if self._study_mask is not None:
            gray[~self._study_mask] = _SCORE_OUT
        thresholds = np.asarray(
            [float(t) for t in getattr(
                self.models.get(sp), "class_thresholds",
                config.PROBABILITY_THRESHOLDS,
            )],
            dtype=np.float64,
        )
        cls = np.zeros(score_wgs.shape, dtype=np.uint8)
        cls[valid_s] = np.digitize(score_wgs[valid_s], thresholds).astype(
            np.uint8,
        )
        rgb[:, :, 0] = gray
        rgb[:, :, 1] = cls

        png_bytes = self._encode_opaque_png(rgb)
        h, w = gray.shape
        logger.info(
            "✅ Data PNG [%s] : %dx%d px (%.0f KB) — score",
            sp, w, h, len(png_bytes) / 1024,
        )
        return png_bytes

    def _build_terrain_png_bytes(self) -> bytes:
        """PNG RGB du terrain, commun à toutes les espèces.

        R = pente encodée (1–181 → 0–90°, 0 = NaN)
        G = altitude octet bas (alt + 500), B = octet haut (0 = NaN)
        """
        h, w = self._dst_height, self._dst_width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        shape = self._final_score.shape

        slp_grid = getattr(self.grid, "slope", None)
        if isinstance(slp_grid, np.ndarray) and slp_grid.shape == shape:
            slp_wgs = self._reproject_to_wgs84(slp_grid)
            valid_p = np.asarray(np.isfinite(slp_wgs))
            rgb[valid_p, 0] = np.clip(
                (slp_wgs[valid_p] * _SLOPE_SCALE + _SLOPE_OFFSET).astype(
                    np.int32,
                ),
                1, 181,
            ).astype(np.uint8)
        else:
            logger.debug("   Pente non disponible pour le PNG terrain")

        alt_grid = getattr(self.grid, "altitude", None)
        if isinstance(alt_grid, np.ndarray) and alt_grid.shape == shape:
            alt_wgs = self._reproject_to_wgs84(alt_grid)
            valid_a = np.asarray(np.isfinite(alt_wgs))
            alt_enc = np.zeros((h, w), dtype=np.uint16)
            alt_enc[valid_a] = np.clip(
                (alt_wgs[valid_a] + _ALT_OFFSET).astype(np.int32),
                0, 65535,
            ).astype(np.uint16)
            rgb[:, :, 1] = (alt_enc & 0xFF).astype(np.uint8)
            rgb[:, :, 2] = ((alt_enc >> 8) & 0xFF).astype(np.uint8)
        else:
            logger.debug("   Altitude non disponible pour le PNG terrain")

        png_bytes = self._encode_opaque_png(rgb)
        logger.info(
            "✅ Terrain PNG : %dx%d px (%.0f KB) — pente+altitude",
            w, h, len(png_bytes) / 1024,
        )
        return png_bytes

    # ──────────────────────────────────────────────────────
    # Panneau latéral + couche de probabilité rendue en JS
    # ──────────────────────────────────────────────────────
    def _species_info(self, species: str) -> dict[str, str]:
        # Pas d'emoji d'espèce : 🍄 est une amanite tue-mouches et 🌰 une
        # châtaigne (essence du modèle) — le nom suffit.
        params = config.get_species_params(species)
        return {
            "name": str(params.get("common_name", species)),
            "scientific": str(params.get("scientific_name", species)),
        }

    def _page_title(self) -> str:
        names = ", ".join(
            self._species_info(sp)["name"] for sp in self.species_list
        )
        return f"Cartomorilles — {names} (Isère)"

    def _add_sidebar_ui(
        self,
        folium_map: folium.Map,
        *,
        data_uris_by_species: dict[str, str],
        terrain_uri: str,
        basemaps: list[dict[str, Any]],
        layer_vars: dict[str, str | None],
        diag_layers: list[dict[str, Any]],
        hotspots: dict[str, list[dict[str, Any]]],
        weather_html: str,
        extra_layers: list[dict[str, Any]] | None = None,
        extra_uris: dict[str, str] | None = None,
    ) -> None:
        """Injecte le panneau latéral (espèce, légende-filtre, opacité,
        hotspots, couches, météo), la lecture de cellule et le JS qui rend
        l'unique couche de probabilité depuis le data PNG de l'espèce
        active."""
        labels = list(config.PROBABILITY_LABELS)
        # Couleurs des classes : palette de référence aux seuils absolus ;
        # le JS cale ses arrêts sur les seuils par rang de chaque espèce
        # (D52), donc chaque classe garde sa couleur.
        palette = _palette_cfg(
            _CMAP_STOPS, _CMAP,
            [float(t) for t in config.PROBABILITY_THRESHOLDS],
        )
        thresholds = {
            sp: [float(t) for t in getattr(
                self.models[sp], "class_thresholds",
                config.PROBABILITY_THRESHOLDS,
            )]
            for sp in self.species_list
        }
        info = {sp: self._species_info(sp) for sp in self.species_list}

        species_radios = "".join(
            f'<label><input type="radio" name="cs-species" '
            f'value="{html.escape(sp)}"'
            f'{" checked" if sp == self.primary_species else ""}>'
            f'{html.escape(info[sp]["name"])}</label>'
            for sp in self.species_list
        )
        basemap_options = "".join(
            f'<option{" selected" if b["default"] else ""}>'
            f'{html.escape(b["label"])}</option>'
            for b in basemaps
        )
        # (clé, libellé, style de pastille, cochée au départ)
        layer_defs = (
            ("area", "Limite de la zone étudiée (Isère)",
             f"background:#fff;border:2px solid {_AREA_LINE};border-radius:2px",
             True),
            ("inat",
             f"Observations iNaturalist de morilles "
             f"({len(self._inat_observations)})",
             f"background:{_INAT_COLOR}", True),
            ("hydro", "Cours d'eau", "background:#0077cc", False),
            ("landmarks", "Points de repère",
             f"background:{_LANDMARK_COLOR};border-radius:2px;"
             "transform:rotate(45deg) scale(.8)", True),
        )
        layer_checks = "".join(
            f'<label class="cs-check"><input type="checkbox" '
            f'data-layer="{key}"{" checked" if on else ""}>'
            f'<span class="cs-dot" style="{style}" '
            f'aria-hidden="true"></span><span>{html.escape(label)}</span></label>'
            for key, label, style, on in layer_defs
            if layer_vars.get(key)
        )
        n_out = {
            sp: int(np.count_nonzero(
                ~self.inside_study_area(self.hotspots_by_species.get(sp, [])),
            ))
            for sp in self.species_list
        }
        diag_block = (
            '<details class="cs-diag"><summary>Couches de diagnostic</summary>'
            '<p class="cs-hint">Zones exclues du modèle pour l\'espèce '
            'affichée.</p><div id="cs-diag-list"></div></details>'
            if diag_layers else ""
        )
        extra_layers = extra_layers or []
        extra_uris = extra_uris or {}
        extra_imgs = "\n".join(
            f'<img id="cartom-extra-img-{html.escape(key)}" data-src="{uri}" '
            f'alt="" hidden>'
            for key, uri in extra_uris.items()
        )
        data_imgs = "\n".join(
            f'<img id="cartom-data-img-{html.escape(sp)}" src="{uri}" '
            f'alt="" hidden>'
            for sp, uri in data_uris_by_species.items()
        )
        subtitle = (
            f"Isère · maille {float(config.CELL_SIZE):.0f} m · "
            f"générée le {datetime.now():%d/%m/%Y}"
        )
        # Classes par rang (D52) : rappel du principe sous la légende, si
        # toutes les espèces ont bien leurs seuils par rang
        rank_hint = ""
        if all(thresholds[sp] != [float(t) for t in config.PROBABILITY_THRESHOLDS]
               for sp in self.species_list):
            q = list(config.CLASS_RANK_QUANTILES)
            shares = [q[0], *(b - a for a, b in zip(q, q[1:])), 1.0 - q[-1]]
            pct = [f"{round(s * 100)} %" for s in shares]
            rank_hint = (
                '<p class="cs-hint">Classes par rang, propres à chaque '
                f"espèce : Excellent = les {pct[4]} de cellules notées de la "
                f"zone étudiée les mieux classées, Bon = {pct[3]} suivants, "
                f"Moyen = {pct[2]}, Faible = {pct[1]}, Très faible = "
                f"{pct[0]}.</p>"
            )
        # Calques supplémentaires (D53) : un à la fois, par-dessus la
        # probabilité, dans un volet repliable de la section Couches
        extra_radios = "".join(
            f'<label class="cs-check"><input type="radio" name="cs-extra" '
            f'value="{html.escape(str(x["key"]))}"><span>'
            f'{html.escape(str(x["label"]))}</span></label>'
            for x in extra_layers
        )
        extra_block = (
            '<details class="cs-diag cs-extra" id="cs-extra"><summary>Calques '
            'supplémentaires</summary><div class="cs-extra-list" '
            'role="radiogroup" aria-label="Calque supplémentaire">'
            '<label class="cs-check"><input type="radio" name="cs-extra" '
            f'value="" checked><span>Aucun</span></label>{extra_radios}</div>'
            '<p class="cs-hint" id="cs-extra-hint" hidden></p>'
            '<ul class="cs-legend cs-extra-legend" id="cs-extra-legend"></ul>'
            '<div class="cs-row" id="cs-extra-oprow" hidden>'
            '<label for="cs-extra-op">Opacité</label>'
            '<input type="range" id="cs-extra-op" min="0" max="100" value="80">'
            '<output id="cs-extra-op-val" for="cs-extra-op">80 %</output></div>'
            "</details>"
            if extra_layers else ""
        )
        # La carte montre le potentiel du terrain ; la météo du jour est
        # une information à part, dans la section météo du panneau (D45).
        weather_hint = ""
        if self._weather_days:
            today = self._weather_days[0]
            score_txt = f"{float(today.score):.2f}".replace(".", ",")
            weather_hint = (
                '<p class="cs-hint">Potentiel du terrain, hors météo. '
                f"Aujourd'hui : {html.escape(str(today.label))} ({score_txt}) "
                "— voir la météo plus bas.</p>"
            )

        cfg: dict[str, Any] = {
            "mapVar": folium_map.get_name(),
            "bounds": self._bounds,
            "south": self._south,
            "north": self._north,
            "west": self._west,
            "east": self._east,
            "labels": labels,
            "thresholds": thresholds,
            "rankQuantiles": list(config.CLASS_RANK_QUANTILES),
            "classAlpha": _class_alphas(len(labels)),
            "palette": palette,
            "scoreScale": _SCORE_SCALE,
            "outCode": _SCORE_OUT,
            "altOffset": _ALT_OFFSET,
            "slopeOffset": _SLOPE_OFFSET,
            "slopeScale": _SLOPE_SCALE,
            "speciesList": self.species_list,
            "primary": self.primary_species,
            "species": info,
            "hotspots": hotspots,
            "hotspotTotals": {
                sp: len(self.hotspots_by_species.get(sp, []))
                for sp in self.species_list
            },
            "hotspotsOutside": n_out,
            "hotspotsDefault": self.default_hotspots_shown,
            "basemaps": basemaps,
            "layers": layer_vars,
            "layerSpecies": {"inat": self.observation_species(self.species_list)},
            "area": self._study_area_geojson(),
            "areaStyle": {
                "line": _AREA_LINE,
                "veilOpacity": _AREA_VEIL_OPACITY,
            },
            "diag": diag_layers,
            "extra": extra_layers,
        }

        block = r"""
        <style>
        .cartom-sidebar {
            position: fixed; top: 10px; left: 10px; z-index: 1001;
            width: 300px; max-height: calc(100% - 20px);
            display: flex; flex-direction: column;
            background: rgba(255,255,255,0.97); color: #1f2a1f;
            border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,.3);
            font: 13px/1.45 'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
            overflow: hidden;
        }
        .cartom-sidebar[hidden], .cs-open[hidden] { display: none !important; }
        .cs-header {
            display: flex; align-items: center; gap: 8px; padding: 9px 10px 9px 12px;
            background: #2d6a2e; color: #fff;
        }
        .cs-title { font-weight: 700; font-size: 15px; }
        .cs-sub { font-size: 12px; }
        .cs-collapse, .cs-open {
            cursor: pointer; font: 600 14px/1 'Segoe UI', system-ui, Arial, sans-serif;
            border-radius: 6px; min-width: 34px; min-height: 34px;
        }
        .cs-collapse {
            margin-left: auto; background: transparent; color: #fff;
            border: 1px solid rgba(255,255,255,.7);
        }
        .cs-open {
            position: fixed; top: 10px; left: 10px; z-index: 1001; padding: 0 12px;
            background: #2d6a2e; color: #fff; border: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,.35);
        }
        .cs-collapse:focus-visible, .cs-open:focus-visible,
        .cartom-sidebar button:focus-visible, .cartom-sidebar select:focus-visible,
        .cartom-sidebar input:focus-visible, .cs-seg label:has(input:focus-visible) {
            outline: 2px solid #1a73e8; outline-offset: 2px;
        }
        .cs-body { overflow-y: auto; padding: 2px 12px 12px; }
        .cs-section { border-top: 1px solid #dde3dd; padding: 10px 0 4px; }
        .cs-section:first-child { border-top: 0; }
        .cs-section h2 {
            font-size: 12px; font-weight: 700; letter-spacing: .04em;
            text-transform: uppercase; color: #4a554a; margin: 0 0 6px;
        }
        .cs-hint { font-size: 12px; color: #4a554a; margin: 0 0 6px; }
        .cs-seg { display: flex; gap: 6px; }
        .cs-seg label {
            flex: 1; display: flex; align-items: center; justify-content: center;
            gap: 6px; padding: 6px 8px; min-height: 36px; margin: 0;
            border: 1px solid #cdd5cd; border-radius: 8px; cursor: pointer;
        }
        .cs-seg input { position: absolute; opacity: 0; pointer-events: none; }
        .cs-seg label:has(input:checked) {
            background: #e3f1e3; border-color: #2d6a2e; font-weight: 600;
        }
        .cs-legend { list-style: none; margin: 0 0 4px; padding: 0; }
        .cs-legend li { display: flex; }
        .cs-class, .cs-nul {
            flex: 1; display: flex; align-items: center; gap: 8px;
            padding: 3px 6px; min-height: 30px; border-radius: 6px;
        }
        .cs-class {
            border: 0; background: transparent; color: inherit; font: inherit;
            text-align: left; cursor: pointer;
        }
        .cs-class:hover { background: #eef4ee; }
        .cs-class[aria-pressed="false"] { color: #6b746b; text-decoration: line-through; }
        .cs-class[aria-pressed="false"] .cs-swatch { background: transparent !important; }
        .cs-nul { color: #4a554a; }
        .cs-swatch {
            width: 18px; height: 14px; border-radius: 3px; flex: none;
            border: 1px solid rgba(0,0,0,.3);
        }
        .cs-swatch-none { border-style: dashed; }
        .cs-swatch-out { background: #ececec; border: 2px solid %AREA_LINE%; }
        .cs-range { margin-left: auto; color: #4a554a; font-size: 12px;
                    font-variant-numeric: tabular-nums; white-space: nowrap; }
        .cs-link {
            border: 0; background: none; padding: 0; color: #1a5fb4;
            text-decoration: underline; cursor: pointer; font: inherit; font-size: 12px;
        }
        .cs-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
        .cs-row label { margin: 0; min-width: 58px; }
        .cs-row input[type=range] { flex: 1; min-width: 0; }
        .cs-row output { min-width: 40px; text-align: right;
                         font-variant-numeric: tabular-nums; }
        .cs-check {
            display: flex; align-items: center; gap: 8px; margin: 3px 0;
            min-height: 28px; cursor: pointer;
        }
        .cs-check input { margin: 0; }
        .cs-label { display: block; margin: 0 0 3px; }
        .cs-select {
            width: 100%; padding: 5px 6px; margin-bottom: 6px; font: inherit;
            border: 1px solid #cdd5cd; border-radius: 6px; background: #fff;
        }
        .cs-dot { width: 12px; height: 12px; border-radius: 50%; flex: none;
                  border: 1px solid rgba(0,0,0,.25); }
        .cs-diag summary { cursor: pointer; color: #4a554a; margin: 6px 0; }
        .cs-extra summary { color: #1f2a1f; font-weight: 600; }
        .cs-extra-list { margin: 0 0 6px; }
        .cs-legend li.cs-ramp { display: block; padding: 4px 10px 0; }
        .cs-ramp-bar {
            display: block; height: 12px; border-radius: 3px;
            border: 1px solid rgba(0,0,0,.3);
        }
        .cs-ramp-ticks {
            position: relative; display: block; height: 20px;
            font-size: 12px; color: #4a554a; font-variant-numeric: tabular-nums;
        }
        .cs-ramp-ticks span {
            position: absolute; top: 3px; transform: translateX(-50%); white-space: nowrap;
        }
        .cs-ramp-ticks span:first-child { transform: none; }
        .cs-ramp-ticks span:last-child { transform: translateX(-100%); }
        .cs-ramp-ends {
            display: flex; justify-content: space-between; gap: 8px;
            font-size: 12px; color: #4a554a;
        }
        .cs-extra-cat { image-rendering: pixelated; }
        .cs-wx table { border-collapse: collapse; width: 100%; font-size: 12px; }
        .cs-wx td { padding: 2px 4px; }
        .cs-wx .wx-good  { color: #1e6b3a; font-weight: 600; }
        .cs-wx .wx-ok    { color: #7a5a00; }
        .cs-wx .wx-bad   { color: #5f5f5f; }
        .cs-wx .wx-burst { color: #0b5394; margin-top: 4px; font-weight: 600; }
        .cs-wx .wx-best  { margin-top: 4px; }
        .cs-wx .wx-credit { margin-top: 6px; font-size: 11px; color: #4a554a; }
        .cs-wx .wx-credit a { color: #1a5fb4; }
        .cs-readout {
            position: fixed; left: 50%; bottom: 18px; transform: translateX(-50%);
            z-index: 1001; max-width: calc(100% - 20px); padding: 5px 12px;
            background: rgba(255,255,255,.96); color: #1f2a1f; border-radius: 16px;
            box-shadow: 0 1px 6px rgba(0,0,0,.3);
            font: 13px/1.4 'Segoe UI', system-ui, Arial, sans-serif;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            pointer-events: none;
        }
        .cs-prob { image-rendering: auto; }
        .cs-lm { background: transparent; border: 0; }
        .cs-lm span {
            display: block; width: 10px; height: 10px; margin: 2px;
            background: %LANDMARK_COLOR%; border: 2px solid #fff; border-radius: 2px;
            transform: rotate(45deg); box-shadow: 0 1px 3px rgba(0,0,0,.45);
        }
        .cs-hs { background: transparent; border: 0; }
        .cs-hs span {
            display: flex; align-items: center; justify-content: center;
            width: 28px; height: 28px; border-radius: 50%;
            border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,.5);
            font: 700 12px/1 'Segoe UI', system-ui, Arial, sans-serif;
        }
        .cs-coords {
            font-variant-numeric: tabular-nums; user-select: all; -webkit-user-select: all;
        }
        .cs-pt-coords { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
        /* Pas de flex-wrap : Leaflet dimensionne la popup en min-content */
        .cs-acts { display: flex; gap: 6px; margin-top: 6px; }
        .cs-copy, .leaflet-container a.cs-act {
            display: inline-flex; align-items: center; justify-content: center;
            min-height: 30px; padding: 0 10px; border: 1px solid #cdd5cd; border-radius: 6px;
            background: #fff; color: #1a5fb4; text-decoration: none; cursor: pointer;
            font: 600 12px/1 'Segoe UI', system-ui, Arial, sans-serif;
        }
        .cs-copy:hover, .leaflet-container a.cs-act:hover { background: #eef4ee; }
        .cs-locate button {
            display: flex; align-items: center; justify-content: center;
            width: 30px; height: 30px; padding: 0; border: 0; border-radius: 2px;
            background: #fff; color: #333; cursor: pointer;
        }
        .cs-locate button:hover { background: #f4f4f4; }
        .cs-locate .cs-loc-on, .cs-locate .cs-loc-wait { color: #1a73e8; }
        .cs-locate .cs-loc-wait { animation: cs-pulse .8s ease-in-out infinite alternate; }
        @keyframes cs-pulse { to { opacity: .35; } }
        @media (prefers-reduced-motion: reduce) {
            .cs-locate .cs-loc-wait { animation: none; }
        }
        .cs-copy:focus-visible, .leaflet-container a.cs-act:focus-visible,
        .cs-locate button:focus-visible { outline: 2px solid #1a73e8; outline-offset: 2px; }
        @media (max-width: 700px) {
            .cartom-sidebar { width: auto; right: 10px; max-height: 70%; }
            .cs-readout { bottom: 30px; font-size: 12px; }
        }
        /* Écran tactile ou étroit : cibles d'au moins 44 px */
        @media (pointer: coarse), (max-width: 700px) {
            .cs-collapse, .cs-open { min-width: 44px; min-height: 44px; }
            .cs-seg label, .cs-class, .cs-check, .cs-link, .cs-select,
            .cs-copy, .leaflet-container a.cs-act { min-height: 44px; }
            .cs-link { display: inline-flex; align-items: center; }
            .cs-check input { width: 20px; height: 20px; }
            .cs-row input[type=range] { height: 44px; }
            .cs-diag summary { padding: 12px 0; margin: 0; line-height: 20px; }
            /* 16 px : iOS ne zoome pas sur la liste au focus */
            .cs-select { font-size: 16px; }
            .leaflet-touch .leaflet-bar a, .cs-locate button {
                width: 44px; height: 44px; line-height: 44px;
            }
        }
        </style>

        <div id="cartom-sidebar" class="cartom-sidebar" role="region"
             aria-label="Panneau Cartomorilles">
            <div class="cs-header">
                <div>
                    <div class="cs-title">Cartomorilles</div>
                    <div class="cs-sub">%SUBTITLE%</div>
                </div>
                <button type="button" class="cs-collapse" id="cs-collapse"
                        aria-label="Replier le panneau" title="Replier le panneau">«</button>
            </div>
            <div class="cs-body">
                <section class="cs-section"%SPECIES_HIDDEN%>
                    <h2>Espèce</h2>
                    <div class="cs-seg" role="radiogroup" aria-label="Espèce">%SPECIES_RADIOS%</div>
                </section>
                <section class="cs-section">
                    <h2 id="cs-legend-title">Probabilité</h2>
                    <p class="cs-hint">Cliquez une classe pour la masquer ou l'afficher.</p>
                    <ul class="cs-legend" id="cs-legend"></ul>
                    %RANK_HINT%
                    %WEATHER_HINT%
                    <button type="button" class="cs-link" id="cs-legend-reset" hidden>Tout afficher</button>
                    <div class="cs-row">
                        <label for="cs-opacity">Opacité</label>
                        <input type="range" id="cs-opacity" min="0" max="100" value="70">
                        <output id="cs-opacity-val" for="cs-opacity">70 %</output>
                    </div>
                </section>
                <section class="cs-section">
                    <h2>Hotspots</h2>
                    <label class="cs-check"><input type="checkbox" id="cs-hs-show" checked>
                        <span>Afficher les hotspots</span></label>
                    <div class="cs-row">
                        <label for="cs-hs-n">Nombre</label>
                        <input type="range" id="cs-hs-n" min="1" max="1" value="1" step="1">
                        <output id="cs-hs-n-val" for="cs-hs-n">0</output>
                    </div>
                    <p class="cs-hint" id="cs-hs-info"></p>
                </section>
                <section class="cs-section">
                    <h2>Couches</h2>
                    <label class="cs-label" for="cs-basemap">Fond de carte</label>
                    <select id="cs-basemap" class="cs-select">%BASEMAP_OPTIONS%</select>
                    <div id="cs-layers">%LAYER_CHECKS%</div>
                    %EXTRA_BLOCK%
                    %DIAG_BLOCK%
                </section>
                %WEATHER_SECTION%
            </div>
        </div>
        <button type="button" id="cs-open" class="cs-open"
                aria-label="Ouvrir le panneau" hidden>☰ Cartomorilles</button>
        <div id="cs-readout" class="cs-readout" aria-live="polite">Chargement de la carte…</div>
        <img id="cartom-terrain-img" src="%TERRAIN_URI%" alt="" hidden>
        %DATA_IMGS%
        %EXTRA_IMGS%

        <script>
        (function () {
        function init() {
            var CFG = %CONFIG_JSON%;
            var map = window[CFG.mapVar];
            if (!map) {   /* Leaflet (CDN) non chargé : hors ligne, sans cache */
                document.getElementById('cs-readout').textContent =
                    'Carte indisponible hors ligne (Leaflet non chargé)';
                return;
            }

            /* Seuils des classes : propres à l'espèce affichée (par rang, D52) */
            var LABELS = CFG.labels, TH = CFG.thresholds[CFG.primary], ALPHA = CFG.classAlpha;
            var N = CFG.scoreScale, OUT = CFG.outCode;
            var SW = CFG.palette.swatches, SWTXT = CFG.palette.swatchText;
            var IDLE = 'Survolez ou touchez la carte';
            var current = CFG.primary;
            var visible = LABELS.map(function (_, i) { return i > 0; });

            function fmt(x, d) {
                return Number(x).toLocaleString('fr-FR',
                    {minimumFractionDigits: d, maximumFractionDigits: d});
            }
            function esc(s) {
                return String(s).replace(/[&<>"']/g, function (c) {
                    return {'&': '&amp;', '<': '&lt;', '>': '&gt;',
                            '"': '&quot;', "'": '&#39;'}[c];
                });
            }
            function classOf(s) {
                var c = 0;
                for (var i = 0; i < TH.length; i++) { if (s >= TH[i]) { c = i + 1; } }
                return c;
            }
            function hexToRgb(h) {
                return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16),
                        parseInt(h.slice(5, 7), 16)];
            }

            /* ═══════ Panneau : repli (mémorisé) + isolation des clics ═══════ */
            var sidebar = document.getElementById('cartom-sidebar');
            var openBtn = document.getElementById('cs-open');
            var collapseBtn = document.getElementById('cs-collapse');
            var readout = document.getElementById('cs-readout');
            var PANEL_KEY = 'cartomorilles.panneau';
            [sidebar, openBtn].forEach(function (el) {
                L.DomEvent.disableClickPropagation(el);
                L.DomEvent.disableScrollPropagation(el);
            });
            function setCollapsed(c, remember) {
                sidebar.hidden = c; openBtn.hidden = !c;
                if (!remember) { return; }
                try { window.localStorage.setItem(PANEL_KEY, c ? 'replie' : 'ouvert'); }
                catch (err) { /* stockage indisponible (navigation privée…) */ }
            }
            collapseBtn.addEventListener('click', function () {
                setCollapsed(true, true); openBtn.focus();
            });
            openBtn.addEventListener('click', function () {
                setCollapsed(false, true); collapseBtn.focus();
            });
            var savedPanel = null;
            try { savedPanel = window.localStorage.getItem(PANEL_KEY); }
            catch (err) { savedPanel = null; }
            if (savedPanel === 'replie' || savedPanel === 'ouvert') {
                setCollapsed(savedPanel === 'replie', false);
            } else if (window.matchMedia && window.matchMedia('(max-width: 700px)').matches) {
                setCollapsed(true, false);
            }
            L.control.scale({position: 'bottomright', imperial: false, maxWidth: 140}).addTo(map);

            /* ═══════ Couleurs : LUT score → couleur continue (classe : canal G) ═══════ */
            var lutRgb = new Uint8ClampedArray((N + 1) * 3);
            /* Arrêts de la palette calés sur les seuils de l'espèce (0, seuils,
               1) : chaque classe garde sa couleur quels que soient ses seuils. */
            function buildLut() {
                var pos = [0].concat(TH, [1]);
                var stops = CFG.palette.stops.map(function (s, i) {
                    return [pos[i], hexToRgb(s[1])];
                });
                for (var v = 0; v <= N; v++) {
                    var s = v / N, rgb = stops[stops.length - 1][1];
                    for (var i = 1; i < stops.length; i++) {
                        if (s <= stops[i][0]) {
                            var span = stops[i][0] - stops[i - 1][0];
                            var t = span > 0 ? (s - stops[i - 1][0]) / span : 1;
                            var a = stops[i - 1][1], b = stops[i][1];
                            rgb = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                                   a[2] + (b[2] - a[2]) * t];
                            break;
                        }
                    }
                    lutRgb[v * 3] = rgb[0]; lutRgb[v * 3 + 1] = rgb[1]; lutRgb[v * 3 + 2] = rgb[2];
                }
            }
            buildLut();

            /* ═══════ Données : score par espèce + terrain (R=pente, G+B=altitude) ═══════ */
            var dataPx = {}, dataW = 0, dataH = 0, terrainPx = null;
            function pixels(img) {
                var c = document.createElement('canvas');
                c.width = img.naturalWidth; c.height = img.naturalHeight;
                var ctx = c.getContext('2d', {willReadFrequently: true});
                ctx.drawImage(img, 0, 0);
                return ctx.getImageData(0, 0, c.width, c.height).data;
            }

            /* ═══════ Couche de probabilité unique ═══════ */
            var probLayer = null, probUrl = null, renderSeq = 0;
            var renderCanvas = document.createElement('canvas');
            var opInput = document.getElementById('cs-opacity');
            var opOut = document.getElementById('cs-opacity-val');
            function opacity() { return parseInt(opInput.value, 10) / 100; }

            function renderProb() {
                var src = dataPx[current];
                if (!src) { return; }
                renderCanvas.width = dataW; renderCanvas.height = dataH;
                var ctx = renderCanvas.getContext('2d');
                var out = ctx.createImageData(dataW, dataH), dd = out.data;
                var alphaByCls = ALPHA.map(function (a, i) {
                    return visible[i] ? Math.round(a * 255) : 0;
                });
                for (var i = 0, n = dataW * dataH; i < n; i++) {
                    var o = i * 4, v = src[o];
                    if (v > N) { continue; }
                    var a = alphaByCls[src[o + 1]];   /* G = classe exacte (D60) */
                    if (!a) { continue; }
                    dd[o] = lutRgb[v * 3]; dd[o + 1] = lutRgb[v * 3 + 1];
                    dd[o + 2] = lutRgb[v * 3 + 2]; dd[o + 3] = a;
                }
                ctx.putImageData(out, 0, 0);
                var seq = ++renderSeq;
                renderCanvas.toBlob(function (blob) {
                    if (!blob || seq !== renderSeq) { return; }
                    var url = URL.createObjectURL(blob), old = probUrl;
                    probUrl = url;
                    if (probLayer) {
                        if (old) { probLayer.once('load', function () { URL.revokeObjectURL(old); }); }
                        probLayer.setUrl(url);
                    } else {
                        probLayer = L.imageOverlay(url, CFG.bounds, {
                            opacity: opacity(), interactive: false, zIndex: 1,
                            className: 'cs-prob'
                        }).addTo(map);
                    }
                }, 'image/png');
            }

            opInput.addEventListener('input', function () {
                opOut.textContent = opInput.value + ' %';
                if (probLayer) { probLayer.setOpacity(opacity()); }
            });

            /* ═══════ Légende = filtre par classe ═══════ */
            var legendReset = document.getElementById('cs-legend-reset');
            function rangeText(i) {
                if (i === 0) { return '< ' + fmt(TH[0], 2); }
                if (i === LABELS.length - 1) { return '≥ ' + fmt(TH[TH.length - 1], 2); }
                return fmt(TH[i - 1], 2) + ' – ' + fmt(TH[i], 2);
            }
            function buildLegend() {
                var ul = document.getElementById('cs-legend');
                ul.innerHTML = '';
                for (var i = LABELS.length - 1; i >= 0; i--) {
                    var li = document.createElement('li');
                    if (i === 0) {
                        li.innerHTML = '<span class="cs-nul"><span class="cs-swatch cs-swatch-none"'
                            + ' aria-hidden="true"></span><span>' + esc(LABELS[0])
                            + ' ou exclu : transparent</span><span class="cs-range">'
                            + rangeText(0) + '</span></span>';
                    } else {
                        var b = document.createElement('button');
                        b.type = 'button';
                        b.className = 'cs-class';
                        b.dataset.cls = String(i);
                        b.setAttribute('aria-pressed', visible[i] ? 'true' : 'false');
                        b.innerHTML = '<span class="cs-swatch" style="background:' + SW[i]
                            + '" aria-hidden="true"></span><span>' + esc(LABELS[i])
                            + '</span><span class="cs-range">' + rangeText(i) + '</span>';
                        b.addEventListener('click', onClassClick);
                        li.appendChild(b);
                    }
                    ul.appendChild(li);
                }
                if (CFG.area) {
                    var out = document.createElement('li');
                    out.innerHTML = '<span class="cs-nul"><span class="cs-swatch cs-swatch-out"'
                        + ' aria-hidden="true"></span><span>Hors zone étudiée : estompé,'
                        + ' sans score</span></span>';
                    ul.appendChild(out);
                }
                legendReset.hidden = visible.slice(1).every(Boolean);
            }
            function onClassClick(e) {
                var i = parseInt(e.currentTarget.dataset.cls, 10);
                visible[i] = !visible[i];
                e.currentTarget.setAttribute('aria-pressed', visible[i] ? 'true' : 'false');
                legendReset.hidden = visible.slice(1).every(Boolean);
                renderProb();
            }
            legendReset.addEventListener('click', function () {
                visible = LABELS.map(function (_, i) { return i > 0; });
                buildLegend(); renderProb();
            });

            /* ═══════ Hotspots (top N de l'espèce active) ═══════ */
            var hsLayer = L.layerGroup().addTo(map);
            var hsShow = document.getElementById('cs-hs-show');
            var hsN = document.getElementById('cs-hs-n');
            var hsNOut = document.getElementById('cs-hs-n-val');
            var hsInfo = document.getElementById('cs-hs-info');
            var hsWanted = CFG.hotspotsDefault;

            function areaText(ha) {
                if (ha >= 10) { return fmt(ha, 0) + ' ha'; }
                if (ha >= 1) { return fmt(ha, 1) + ' ha'; }
                return fmt(ha * 10000, 0) + ' m²';
            }
            function hsPopup(h) {
                var sp = CFG.species[current];
                var facts = [areaText(h.a)];
                if (h.alt !== undefined) { facts.push('altitude ' + h.alt + ' m'); }
                if (h.sl !== undefined) { facts.push('pente ' + fmt(h.sl, 0) + '°'); }
                var lines = [
                    '<b>' + esc(sp.name) + ' — hotspot n°' + h.r + '</b>',
                    'Score moyen <b>' + fmt(h.s, 2) + '</b> · ' + esc(LABELS[classOf(h.s)]),
                    facts.join(' · ')
                ];
                if (h.tr) { lines.push('Essence dominante : ' + esc(h.tr)); }
                if (h.g) { lines.push('Géologie : ' + esc(h.g)); }
                return lines.join('<br>') + coordsHtml(h.lat, h.lon, true);
            }
            function renderHotspots() {
                hsLayer.clearLayers();
                var list = CFG.hotspots[current] || [], total = list.length;
                var detected = Math.max(total, CFG.hotspotTotals[current] || 0);
                var n = Math.min(hsWanted, total);
                hsN.max = String(Math.max(1, total));
                hsN.value = String(Math.max(1, n));
                hsN.disabled = total <= 1;
                hsNOut.textContent = String(n);
                var outside = CFG.hotspotsOutside[current] || 0;
                var inside = detected - outside;
                hsInfo.textContent = (total
                    ? 'Les ' + n + ' meilleurs sur ' + inside.toLocaleString('fr-FR') + ' détectés'
                      + (inside > total ? ' (carte limitée aux ' + total.toLocaleString('fr-FR')
                         + ' premiers, liste complète dans le CSV)' : '')
                    : 'Aucun hotspot détecté')
                    + (outside ? ' · ' + outside.toLocaleString('fr-FR')
                       + ' hors zone étudiée non affichés' : '');
                if (!hsShow.checked) { return; }
                for (var i = 0; i < n; i++) {
                    var h = list[i], c = classOf(h.s);
                    var icon = L.divIcon({
                        className: 'cs-hs',
                        html: '<span style="background:' + SW[c] + ';color:' + SWTXT[c] + '">'
                            + h.r + '</span>',
                        iconSize: [28, 28], iconAnchor: [14, 14], popupAnchor: [0, -14]
                    });
                    L.marker([h.lat, h.lon], {
                        icon: icon, riseOnHover: true, zIndexOffset: (n - i) * 10,
                        title: 'Hotspot n°' + h.r + ' — score ' + fmt(h.s, 2)
                    }).bindPopup(hsPopup(h), {maxWidth: 320}).addTo(hsLayer);
                }
            }
            hsShow.addEventListener('change', renderHotspots);
            hsN.addEventListener('input', function () {
                hsWanted = parseInt(hsN.value, 10);
                renderHotspots();
            });

            /* ═══════ Fond de carte + couches ═══════ */
            var baseSel = document.getElementById('cs-basemap'), baseLayer = null;
            CFG.basemaps.forEach(function (b) {
                if (b['default']) { baseLayer = window[b.var] || null; }
            });
            baseSel.addEventListener('change', function () {
                var b = CFG.basemaps[baseSel.selectedIndex], layer = b && window[b.var];
                if (!layer || layer === baseLayer) { return; }
                if (baseLayer) { map.removeLayer(baseLayer); }
                layer.addTo(map); layer.bringToBack(); baseLayer = layer;
            });

            /* Zone étudiée (D39) : contour sur halo blanc + extérieur voilé,
               dans un volet au-dessus de la probabilité, sous les marqueurs. */
            if (CFG.area) {
                map.createPane('csArea');
                var areaPane = map.getPane('csArea');
                areaPane.style.zIndex = 450;
                areaPane.style.pointerEvents = 'none';
                var AS = CFG.areaStyle;
                window.cartom_study_area = L.layerGroup([
                    L.geoJSON(CFG.area.veil, {pane: 'csArea', interactive: false, style: {
                        stroke: false, fillColor: '#ffffff', fillOpacity: AS.veilOpacity
                    }}),
                    L.geoJSON(CFG.area.outline, {pane: 'csArea', interactive: false, style: {
                        color: '#ffffff', weight: 4.5, opacity: 0.85, fill: false
                    }}),
                    L.geoJSON(CFG.area.outline, {pane: 'csArea', interactive: false, style: {
                        color: AS.line, weight: 2, opacity: 0.95, fill: false
                    }})
                ]);
            }

            /* Couches : certaines ne valent que pour une espèce (les
               observations iNaturalist sont des morilles). */
            var layerRows = [];
            document.querySelectorAll('#cs-layers input[data-layer]').forEach(function (input) {
                var key = input.dataset.layer, layer = window[CFG.layers[key]];
                var row = input.closest('label');
                if (!layer) { row.hidden = true; return; }
                var linked = CFG.layerSpecies[key];
                layerRows.push({input: input, layer: layer, row: row,
                                species: linked ? [].concat(linked) : null});
                input.addEventListener('change', syncLayers);
            });
            function syncLayers() {
                layerRows.forEach(function (l) {
                    var off = l.species !== null && l.species.indexOf(current) === -1;
                    l.row.hidden = off;
                    if (l.input.checked && !off) { map.addLayer(l.layer); }
                    else { map.removeLayer(l.layer); }
                });
            }
            syncLayers();

            /* ═══════ Couches de diagnostic (option --debug-layers) ═══════ */
            var diagOn = {};
            function syncDiag() {
                var box = document.getElementById('cs-diag-list');
                if (!box) { return; }
                box.innerHTML = '';
                CFG.diag.forEach(function (d) {
                    var layer = window[d.var];
                    if (!layer) { return; }
                    if (d.species && d.species !== current) { map.removeLayer(layer); return; }
                    var lab = document.createElement('label');
                    lab.className = 'cs-check';
                    lab.innerHTML = '<input type="checkbox"><span class="cs-dot" style="background:'
                        + d.swatch + '" aria-hidden="true"></span><span>' + esc(d.label) + '</span>';
                    var input = lab.querySelector('input');
                    input.checked = !!diagOn[d.group];
                    function sync() {
                        if (input.checked) { map.addLayer(layer); } else { map.removeLayer(layer); }
                    }
                    input.addEventListener('change', function () {
                        diagOn[d.group] = input.checked; sync();
                    });
                    sync();
                    box.appendChild(lab);
                });
            }

            /* ═══════ Calques supplémentaires (D53) : un seul à la fois ═══════ */
            /* Un PNG de codes 8 bits par calque (pente : canal R du PNG
               terrain), décodé au premier affichage et coloré ici ; même
               grille que le score, masqué hors zone étudiée (D39). */
            var EXTRA = CFG.extra || [], extraByKey = {}, extraKey = '';
            var extraPx = {}, extraLuts = {}, extraOverlays = {};
            var extraHint = document.getElementById('cs-extra-hint');
            var extraLegend = document.getElementById('cs-extra-legend');
            var extraOpRow = document.getElementById('cs-extra-oprow');
            var extraOp = document.getElementById('cs-extra-op');
            var extraOpOut = document.getElementById('cs-extra-op-val');
            EXTRA.forEach(function (x) {
                extraByKey[x.key] = x;
                if (x.kind === 'cat') {
                    x.byCode = {};
                    x.classes.forEach(function (c) { x.byCode[c.code] = c; });
                }
            });
            function extraOpacity() { return extraOp ? parseInt(extraOp.value, 10) / 100 : 0.8; }
            function rampRgb(stops, v) {
                if (v <= stops[0][0]) { return hexToRgb(stops[0][1]); }
                for (var i = 1; i < stops.length; i++) {
                    if (v <= stops[i][0]) {
                        var t = (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
                        var a = hexToRgb(stops[i - 1][1]), b = hexToRgb(stops[i][1]);
                        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                                a[2] + (b[2] - a[2]) * t];
                    }
                }
                return hexToRgb(stops[stops.length - 1][1]);
            }
            /* Code → RGBA (alpha 0 = transparent) */
            function extraLut(x) {
                if (extraLuts[x.key]) { return extraLuts[x.key]; }
                var lut = new Uint8ClampedArray(256 * 4);
                function put(k, rgb) {
                    lut[k * 4] = rgb[0]; lut[k * 4 + 1] = rgb[1];
                    lut[k * 4 + 2] = rgb[2]; lut[k * 4 + 3] = 255;
                }
                if (x.kind === 'cat') {
                    x.classes.forEach(function (c) { put(c.code, hexToRgb(c.color)); });
                } else {
                    for (var k = 0; k <= x.maxCode; k++) {
                        if (k !== x.none) { put(k, rampRgb(x.stops, x.vmin + k * x.step)); }
                    }
                }
                return (extraLuts[x.key] = lut);
            }
            function extraPixels(x) {
                if (x.src === 'terrain') { return terrainPx; }
                if (extraPx[x.key]) { return extraPx[x.key]; }
                var img = document.getElementById('cartom-extra-img-' + x.key);
                if (img && !img.getAttribute('src')) {
                    img.addEventListener('load', function () {
                        extraPx[x.key] = pixels(img);
                        if (extraKey === x.key) { renderExtra(); }
                        if (pointPopup.isOpen()) { pointPopup.update(); }
                    }, {once: true});
                    img.src = img.getAttribute('data-src');
                }
                return null;
            }
            function renderExtra() {
                var x = extraByKey[extraKey];
                Object.keys(extraOverlays).forEach(function (k) {
                    if (k !== extraKey) { map.removeLayer(extraOverlays[k]); }
                });
                if (!x) { return; }
                if (extraOverlays[x.key]) {
                    extraOverlays[x.key].setOpacity(extraOpacity()).addTo(map);
                    return;
                }
                var px = extraPixels(x), area = dataPx[current];
                if (!px || !area || px.length !== area.length) { return; }
                var lut = extraLut(x), c = document.createElement('canvas');
                c.width = dataW; c.height = dataH;
                var ctx = c.getContext('2d'), out = ctx.createImageData(dataW, dataH), dd = out.data;
                for (var i = 0, n = dataW * dataH; i < n; i++) {
                    var o = i * 4;
                    if (area[o] === OUT) { continue; }
                    var k = px[o] * 4;
                    if (!lut[k + 3]) { continue; }
                    dd[o] = lut[k]; dd[o + 1] = lut[k + 1]; dd[o + 2] = lut[k + 2]; dd[o + 3] = 255;
                }
                ctx.putImageData(out, 0, 0);
                var key = x.key;
                c.toBlob(function (blob) {
                    if (!blob || extraOverlays[key]) { return; }
                    extraOverlays[key] = L.imageOverlay(URL.createObjectURL(blob), CFG.bounds, {
                        opacity: extraOpacity(), interactive: false, zIndex: 2,
                        className: x.kind === 'cat' ? 'cs-extra-cat' : 'cs-prob'
                    });
                    if (extraKey === key) { extraOverlays[key].addTo(map); }
                }, 'image/png');
            }
            function shareText(s) {
                if (s >= 0.095) { return fmt(s * 100, 0) + ' %'; }
                if (s >= 0.001) { return fmt(s * 100, 1) + ' %'; }
                return '< 0,1 %';
            }
            function buildExtraLegend() {
                var x = extraByKey[extraKey];
                if (!extraLegend) { return; }
                extraLegend.innerHTML = '';
                extraHint.hidden = !x; extraOpRow.hidden = !x;
                if (!x) { return; }
                extraHint.textContent = x.hint || '';
                if (x.kind === 'cat') {
                    x.classes.forEach(function (c) {
                        var li = document.createElement('li');
                        li.innerHTML = '<span class="cs-nul"><span class="cs-swatch" style="background:'
                            + c.color + '" aria-hidden="true"></span><span>' + esc(c.label)
                            + '</span><span class="cs-range" title="Part de la zone étudiée">'
                            + shareText(c.share) + '</span></span>';
                        extraLegend.appendChild(li);
                    });
                    return;
                }
                var lo = x.stops[0][0], hi = x.stops[x.stops.length - 1][0];
                function pos(v) { return ((v - lo) / (hi - lo) * 100).toFixed(1) + '%'; }
                var li = document.createElement('li');
                li.className = 'cs-ramp';
                li.innerHTML = '<span class="cs-ramp-bar" aria-hidden="true" '
                    + 'style="background:linear-gradient(90deg,'
                    + x.stops.map(function (s) { return s[1] + ' ' + pos(s[0]); }).join(',')
                    + ')"></span><span class="cs-ramp-ticks">'
                    + x.ticks.map(function (t) {
                        return '<span style="left:' + pos(t[0]) + '">' + esc(t[1]) + '</span>';
                    }).join('') + '</span>'
                    + (x.ends ? '<span class="cs-ramp-ends"><span>' + esc(x.ends[0])
                       + '</span><span>' + esc(x.ends[1]) + '</span></span>' : '');
                extraLegend.appendChild(li);
            }
            function bigAreaText(m2) {
                if (m2 >= 1e6) { return fmt(m2 / 1e6, m2 >= 1e7 ? 0 : 1) + ' km²'; }
                return areaText(m2 / 1e4);
            }
            /* Valeur du calque affiché au pixel o (texte), dans la zone étudiée */
            function extraValue(o) {
                var x = extraByKey[extraKey];
                if (!x || x.unit === 'deg') { return null; }   /* pente : déjà lue */
                var px = x.src === 'terrain' ? terrainPx : extraPx[x.key];
                if (!px) { return null; }
                var k = px[o];
                if (x.kind === 'cat') {
                    var c = k === x.none ? null : x.byCode[k];
                    if (c) { return x.short + ' : ' + c.label; }
                    return x.noneText ? x.short + ' : ' + x.noneText : null;
                }
                if (k === x.none) { return null; }
                var v = x.vmin + k * x.step;
                if (x.unit === 'twi') { return 'TWI ' + fmt(v, 1); }
                if (x.unit === 'log10m2') { return 'aire drainée ' + bigAreaText(Math.pow(10, v)); }
                return fmt(v, 1);
            }
            document.querySelectorAll('input[name="cs-extra"]').forEach(function (r) {
                r.addEventListener('change', function () {
                    if (!r.checked) { return; }
                    extraKey = r.value;
                    buildExtraLegend(); renderExtra();
                    if (pointPopup.isOpen()) { pointPopup.update(); }
                });
            });
            if (extraOp) {
                extraOp.addEventListener('input', function () {
                    extraOpOut.textContent = extraOp.value + ' %';
                    var ov = extraOverlays[extraKey];
                    if (ov) { ov.setOpacity(extraOpacity()); }
                });
            }

            /* ═══════ Lecture de la cellule (survol, ou toucher sur mobile) ═══════ */
            /* L'image est linéaire en Y Mercator (overlay EPSG:3857, D29) :
               la ligne de pixel se calcule en Mercator, pas en latitude. */
            function mercY(lat) {
                var s = Math.sin(lat * Math.PI / 180);
                return 0.5 * Math.log((1 + s) / (1 - s));
            }
            var yN = mercY(CFG.north), yS = mercY(CFG.south);
            function cellAt(ll) {
                var d = dataPx[current];
                if (!d) { return {state: 'loading'}; }
                if (ll.lat < CFG.south || ll.lat > CFG.north
                    || ll.lng < CFG.west || ll.lng > CFG.east) {
                    return {state: 'out'};
                }
                var px = Math.floor((ll.lng - CFG.west) / (CFG.east - CFG.west) * dataW);
                var py = Math.floor((yN - mercY(ll.lat)) / (yN - yS) * dataH);
                px = Math.max(0, Math.min(px, dataW - 1));
                py = Math.max(0, Math.min(py, dataH - 1));
                var o = (py * dataW + px) * 4, v = d[o];
                if (v > N && v !== OUT) { return {state: 'nodata'}; }
                /* Hors zone étudiée (D39) : pas de score, mais le terrain */
                var cell = v === OUT ? {state: 'outside'}
                    : {state: 'ok', s: v / N, cls: d[o + 1]};
                if (terrainPx) {
                    var alt = terrainPx[o + 1] + (terrainPx[o + 2] << 8);
                    if (alt > 0) { cell.alt = Math.round(alt - CFG.altOffset); }
                    if (terrainPx[o] > 0) {
                        cell.slope = (terrainPx[o] - CFG.slopeOffset) / CFG.slopeScale;
                    }
                }
                if (v !== OUT) {
                    var ex = extraValue(o);
                    if (ex) { cell.extra = ex; }
                }
                return cell;
            }
            var OUT_TEXT = 'Hors zone étudiée (Isère) : pas de score';
            function hasTerrain(cell) { return cell.state === 'ok' || cell.state === 'outside'; }
            function cellParts(cell) {
                if (cell.state === 'out') { return ['Hors de la zone modélisée']; }
                if (cell.state === 'nodata') { return ['Pas de données ici']; }
                if (!hasTerrain(cell)) { return ['Chargement de la carte…']; }
                var parts = [cell.state === 'outside' ? OUT_TEXT
                             : CFG.species[current].name + ' : ' + fmt(cell.s, 2) + ' · '
                               + LABELS[cell.cls]];
                if (cell.extra) { parts.push(cell.extra); }
                if (cell.alt !== undefined) { parts.push(cell.alt + ' m'); }
                if (cell.slope !== undefined) { parts.push(fmt(cell.slope, 0) + '°'); }
                return parts;
            }
            function showReadout(ll, prefix) {
                if (!dataPx[current]) { return; }
                var cell = cellAt(ll), parts = cellParts(cell);
                if (hasTerrain(cell)) { parts.push(ll.lat.toFixed(5) + ', ' + ll.lng.toFixed(5)); }
                readout.textContent = (prefix || '') + parts.join(' · ');
            }
            map.on('mousemove click', function (e) { showReadout(e.latlng); });
            map.on('mouseout', function () { if (dataPx[current]) { readout.textContent = IDLE; } });

            /* ═══════ Popups : score du point, coordonnées copiables, liens ═══════ */
            function pointBody(ll) {
                var cell = cellAt(ll), terrain = [];
                if (!hasTerrain(cell)) { return esc(cellParts(cell)[0]); }
                var lines = [cell.state === 'outside' ? esc(OUT_TEXT)
                             : esc(CFG.species[current].name) + ' : <b>' + fmt(cell.s, 2)
                               + '</b> · ' + esc(LABELS[cell.cls])];
                if (cell.extra) { lines.push(esc(cell.extra)); }
                if (cell.alt !== undefined) { terrain.push('altitude ' + cell.alt + ' m'); }
                if (cell.slope !== undefined) { terrain.push('pente ' + fmt(cell.slope, 0) + '°'); }
                if (terrain.length) { lines.push(terrain.join(' · ')); }
                return lines.join('<br>');
            }
            /* Coordonnées WGS 84 (format accepté par les applis GPS) + bouton
               Copier ; liens Itinéraire (directions Google Maps) et Google Maps. */
            function coordsHtml(lat, lng, links) {
                var txt = lat.toFixed(5) + ', ' + lng.toFixed(5);
                var q = lat.toFixed(5) + ',' + lng.toFixed(5);
                var out = '<div class="cs-pt-coords"><span class="cs-coords" '
                    + 'title="Latitude, longitude (WGS 84)">' + txt + '</span>'
                    + '<button type="button" class="cs-copy" data-copy="' + txt
                    + '">Copier</button></div>';
                if (!links) { return out; }
                return out + '<div class="cs-acts">'
                    + '<a class="cs-act" href="https://www.google.com/maps/dir/?api=1&amp;destination='
                    + q + '" target="_blank" rel="noopener">Itinéraire</a>'
                    + '<a class="cs-act" href="https://www.google.com/maps?q=' + q
                    + '" target="_blank" rel="noopener">Google Maps</a></div>';
            }
            function copyFallback(text) {
                var ta = document.createElement('textarea');
                ta.value = text;
                ta.setAttribute('readonly', '');
                ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;font-size:16px';
                document.body.appendChild(ta);
                ta.select();
                ta.setSelectionRange(0, text.length);
                var ok = false;
                try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
                document.body.removeChild(ta);
                return ok;
            }
            function copyCoords(btn) {
                var text = btn.getAttribute('data-copy');
                function done(ok) {
                    btn.focus();
                    if (!ok) {   /* coordonnées sélectionnées : copie manuelle possible */
                        var span = btn.parentNode.querySelector('.cs-coords');
                        if (span && window.getSelection) { window.getSelection().selectAllChildren(span); }
                    }
                    btn.textContent = ok ? 'Copié ✓' : 'Copiez la sélection';
                    setTimeout(function () { btn.textContent = 'Copier'; }, 2000);
                }
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text).then(function () { done(true); },
                        function () { done(copyFallback(text)); });
                } else {
                    done(copyFallback(text));
                }
            }
            /* Capture : Leaflet arrête la propagation des clics dans les popups. */
            document.addEventListener('click', function (ev) {
                var btn = ev.target && ev.target.closest ? ev.target.closest('.cs-copy') : null;
                if (btn) { copyCoords(btn); }
            }, true);

            /* ═══════ Popup du point cliqué ou touché ═══════ */
            var pointPopup = L.popup({maxWidth: 300}).setContent(function (p) {
                var ll = p.getLatLng();
                return '<b>Point sélectionné</b><br>' + pointBody(ll) + coordsHtml(ll.lat, ll.lng, true);
            });
            var popupOpen = false, dismissClick = false, clickTimer = null;
            map.on('popupopen', function () { popupOpen = true; });
            map.on('popupclose', function () { popupOpen = false; });
            /* Toucher la carte quand une popup est ouverte la ferme seulement
               (Leaflet la ferme au « preclick », juste avant ce « click »). */
            map.on('preclick', function () {
                dismissClick = popupOpen;
                setTimeout(function () { dismissClick = false; }, 0);
            });
            map.on('click', function (e) {
                clearTimeout(clickTimer);
                if (dismissClick) { dismissClick = false; return; }
                var ll = e.latlng;
                /* Délai : un double-clic (zoom) n'ouvre pas de popup. */
                clickTimer = setTimeout(function () { pointPopup.setLatLng(ll).openOn(map); }, 250);
            });
            map.on('dblclick', function () { clearTimeout(clickTimer); });

            /* ═══════ Me localiser (navigator.geolocation) ═══════ */
            var geo = {watch: null, ll: null, acc: 0, dot: null, ring: null, center: false};
            var LocateControl = L.Control.extend({
                options: {position: 'topright'},
                onAdd: function () {
                    var bar = L.DomUtil.create('div', 'leaflet-bar cs-locate');
                    var btn = L.DomUtil.create('button', '', bar);
                    btn.type = 'button';
                    btn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" '
                        + 'aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="3.5" '
                        + 'fill="currentColor"/><circle cx="12" cy="12" r="7.5" fill="none" '
                        + 'stroke="currentColor" stroke-width="2"/><path d="M12 1.5V5M12 19v3.5'
                        + 'M1.5 12H5M19 12h3.5" stroke="currentColor" stroke-width="2" '
                        + 'stroke-linecap="round"/></svg>';
                    btn.addEventListener('click', onLocateClick);
                    L.DomEvent.disableClickPropagation(bar);
                    this.button = btn;
                    return bar;
                }
            });
            var locate = new LocateControl().addTo(map), locBtn = locate.button;
            /* Juste sous les boutons de zoom */
            var zoomBar = map.getContainer().querySelector('.leaflet-right .leaflet-control-zoom');
            if (zoomBar) { zoomBar.parentNode.insertBefore(locate.getContainer(), zoomBar.nextSibling); }
            function locState(state) {
                var label = {
                    off: 'Me localiser',
                    wait: 'Localisation en cours… (toucher pour annuler)',
                    on: 'Ma position : recentrer la carte, ou arrêter si elle est visible'
                }[state];
                locBtn.className = 'cs-loc-' + state;
                locBtn.setAttribute('aria-pressed', state === 'off' ? 'false' : 'true');
                locBtn.setAttribute('aria-label', label);
                locBtn.title = label;
            }
            locState('off');
            function accText(m) {
                return m >= 1000 ? fmt(m / 1000, 1) + ' km' : fmt(m, 0) + ' m';
            }
            function herePopup() {
                return '<b>Vous êtes ici</b> (± ' + accText(geo.acc) + ')<br>'
                    + pointBody(geo.ll) + coordsHtml(geo.ll.lat, geo.ll.lng, false);
            }
            function locStop(msg) {
                if (geo.watch !== null) { navigator.geolocation.clearWatch(geo.watch); }
                if (geo.dot) { map.removeLayer(geo.dot); map.removeLayer(geo.ring); }
                geo = {watch: null, ll: null, acc: 0, dot: null, ring: null, center: false};
                locState('off');
                if (msg) { readout.textContent = msg; }
            }
            function onLocated(pos) {
                var ll = L.latLng(pos.coords.latitude, pos.coords.longitude);
                geo.ll = ll;
                geo.acc = pos.coords.accuracy || 0;
                if (!geo.dot) {
                    geo.ring = L.circle(ll, {radius: geo.acc, color: '#1a73e8', weight: 1,
                                             fillOpacity: 0.12, interactive: false}).addTo(map);
                    geo.dot = L.circleMarker(ll, {radius: 8, color: '#fff', weight: 3,
                                                  fillColor: '#1a73e8', fillOpacity: 1,
                                                  bubblingMouseEvents: false})
                        .bindPopup(herePopup, {maxWidth: 300}).addTo(map);
                } else {
                    geo.ring.setLatLng(ll).setRadius(geo.acc);
                    geo.dot.setLatLng(ll);
                    if (geo.dot.isPopupOpen()) { geo.dot.getPopup().update(); }
                }
                locState('on');
                if (geo.center) {   /* premier point : centrer, sans suivre ensuite */
                    geo.center = false;
                    map.setView(ll, Math.max(map.getZoom(), 15));
                    geo.dot.openPopup();
                    showReadout(ll, 'Votre position · ');
                }
            }
            function onLocError(err) {
                if (geo.ll && err.code !== 1) { return; }   /* perte passagère du signal */
                locStop(err.code === 1
                    ? 'Localisation refusée : autorisez-la pour cette page'
                    : err.code === 3 ? 'Position introuvable (délai dépassé), réessayez'
                    : 'Position indisponible');
            }
            function onLocateClick() {
                if (!navigator.geolocation) {
                    readout.textContent = 'Géolocalisation non disponible dans ce navigateur';
                } else if (window.isSecureContext === false) {
                    readout.textContent = 'Localisation impossible en http : '
                        + 'ouvrez le fichier directement, ou servez-le en https';
                } else if (geo.watch === null) {
                    geo.center = true;
                    locState('wait');
                    readout.textContent = 'Localisation en cours…';
                    geo.watch = navigator.geolocation.watchPosition(onLocated, onLocError,
                        {enableHighAccuracy: true, maximumAge: 10000, timeout: 30000});
                } else if (geo.ll && !map.getBounds().contains(geo.ll)) {
                    map.setView(geo.ll, Math.max(map.getZoom(), 15));
                } else {
                    locStop(dataPx[current] ? IDLE : '');
                }
            }

            /* ═══════ Espèce ═══════ */
            function setSpecies(sp) {
                if (!CFG.species[sp]) { return; }
                current = sp;
                document.getElementById('cs-legend-title').textContent =
                    'Probabilité — ' + CFG.species[sp].name;
                /* Seuils par rang de l'espèce : LUT et bornes de la légende */
                TH = CFG.thresholds[sp];
                buildLut(); buildLegend();
                renderProb(); renderHotspots(); syncDiag(); syncLayers();
                if (dataPx[current]) { readout.textContent = IDLE; }
                /* Popups ouvertes : score de la nouvelle espèce */
                if (pointPopup.isOpen()) { pointPopup.update(); }
                if (geo.dot && geo.dot.isPopupOpen()) { geo.dot.getPopup().update(); }
            }
            document.querySelectorAll('input[name="cs-species"]').forEach(function (r) {
                r.addEventListener('change', function () { if (r.checked) { setSpecies(r.value); } });
            });

            /* ═══════ Démarrage : attendre les data PNG ═══════ */
            buildLegend();
            setSpecies(current);
            var imgs = CFG.speciesList.map(function (sp) {
                return document.getElementById('cartom-data-img-' + sp);
            });
            var terrainImg = document.getElementById('cartom-terrain-img');
            function ready(img) { return img && img.complete && img.naturalWidth > 0; }
            function start() {
                if (!imgs.every(ready)) { return; }
                CFG.speciesList.forEach(function (sp, i) { dataPx[sp] = pixels(imgs[i]); });
                dataW = imgs[0].naturalWidth; dataH = imgs[0].naturalHeight;
                if (ready(terrainImg) && terrainImg.naturalWidth === dataW) {
                    terrainPx = pixels(terrainImg);
                } else {   /* calque de pente lu dans le PNG terrain : indisponible */
                    EXTRA.forEach(function (x) {
                        if (x.src !== 'terrain') { return; }
                        var r = document.querySelector('input[name="cs-extra"][value="' + x.key + '"]');
                        if (r) { r.closest('label').hidden = true; }
                    });
                }
                readout.textContent = IDLE;
                renderProb(); renderExtra();
            }
            var waiting = imgs.concat([terrainImg]).filter(function (img) { return img && !ready(img); });
            if (!waiting.length) { start(); }
            waiting.forEach(function (img) {
                img.addEventListener('load', function () {
                    if (waiting.every(ready)) { start(); }
                }, {once: true});
            });
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
        })();
        </script>
        """

        replacements = {
            "%SUBTITLE%": html.escape(subtitle),
            "%SPECIES_HIDDEN%": " hidden" if len(self.species_list) < 2 else "",
            "%SPECIES_RADIOS%": species_radios,
            "%BASEMAP_OPTIONS%": basemap_options,
            "%LAYER_CHECKS%": layer_checks,
            "%DIAG_BLOCK%": diag_block,
            "%WEATHER_SECTION%": weather_html,
            "%WEATHER_HINT%": weather_hint,
            "%RANK_HINT%": rank_hint,
            "%EXTRA_BLOCK%": extra_block,
            "%EXTRA_IMGS%": extra_imgs,
            "%TERRAIN_URI%": terrain_uri,
            "%DATA_IMGS%": data_imgs,
            "%AREA_LINE%": _AREA_LINE,
            "%LANDMARK_COLOR%": _LANDMARK_COLOR,
            "%CONFIG_JSON%": _js_json(cfg),
        }
        for token, value in replacements.items():
            block = block.replace(token, value)

        folium_map.get_root().html.add_child(  # type: ignore[attr-defined]
            Element(block),
        )

    # ──────────────────────────────────────────────────────
    # Fonds de carte
    # ──────────────────────────────────────────────────────
    def _add_basemaps(self, folium_map: folium.Map) -> list[dict[str, Any]]:
        """Ajoute les fonds de carte (seul le défaut est affiché) et
        retourne leur description pour le sélecteur du panneau."""
        basemaps: list[dict[str, Any]] = []
        for name, cfg in config.BASEMAPS.items():
            is_default = name == config.DEFAULT_BASEMAP
            label = str(cfg.get("name", name))
            tiles = cfg["tiles"]
            if tiles in {
                "OpenStreetMap",
                "CartoDB positron",
                "CartoDB dark_matter",
            }:
                layer = folium.TileLayer(
                    tiles=tiles,
                    name=label,
                    show=is_default,
                    control=False,
                )
            else:
                layer = folium.TileLayer(
                    tiles=tiles,
                    attr=cfg.get("attr", ""),
                    name=label,
                    max_zoom=cfg.get("max_zoom", 18),
                    # Zoom maximal des tuiles servies ; au-delà, Leaflet
                    # agrandit au lieu d'afficher des cases vides
                    max_native_zoom=cfg.get("max_native_zoom"),
                    show=is_default,
                    control=False,
                )
            layer.add_to(folium_map)
            basemaps.append({
                "label": label,
                "var": layer.get_name(),
                "default": is_default,
            })
        return basemaps

    # ──────────────────────────────────────────────────────
    # Couche hydro vectorielle
    # ──────────────────────────────────────────────────────
    def _add_hydro_vector_layer(self, folium_map: folium.Map) -> str | None:
        """Cours d'eau (masqués au départ). Géométries simplifiées à
        l'échelle de la maille et arrondies à ~1 m pour alléger le HTML."""
        if self._hydro_gdf is None or self._hydro_gdf.empty:
            logger.debug("   Pas de données hydro vectorielles")
            return None

        geoms = self._hydro_gdf.geometry
        if geoms.crs is not None and geoms.crs.is_projected:
            geoms = geoms.simplify(max(float(config.CELL_SIZE) / 4.0, 5.0))
        if geoms.crs is not None and not geoms.crs.equals("EPSG:4326"):
            geoms = geoms.to_crs(epsg=4326)
        geoms = geoms.set_precision(1e-5)
        geoms = geoms[~(geoms.isna() | geoms.is_empty)]

        style = {"color": "#0077cc", "weight": 2, "opacity": 0.7}
        fg = folium.FeatureGroup(name="Cours d'eau", show=False, control=False)
        folium.GeoJson(
            geoms.__geo_interface__,
            style_function=lambda _feat, _s=style: _s,
        ).add_to(fg)
        fg.add_to(folium_map)
        logger.info("✅ Cours d'eau vectoriels : %d entités", len(geoms))
        return fg.get_name()

    # ──────────────────────────────────────────────────────
    # Couche iNaturalist Morchella
    # ──────────────────────────────────────────────────────
    def _add_inaturalist_layer(self, folium_map: folium.Map) -> str | None:
        """Observations Morchella : affichées quand une morille (grise ou
        noire) est l'espèce active (le JS masque la couche pour les autres)."""
        if not self._inat_observations or not self.observation_species(
            self.species_list,
        ):
            return None

        fg = folium.FeatureGroup(
            name="Observations iNaturalist de morilles", show=True, control=False,
        )
        for obs in self._inat_observations:
            lat = obs.get("lat")
            lng = obs.get("lng")
            if lat is None or lng is None:
                continue
            quality = str(obs.get("quality_grade", "needs_id"))
            date_str = str(obs.get("observed_on", "?"))
            taxon = str(obs.get("taxon_name", "Morchella"))
            obs_id = obs.get("id", "?")
            obscured = bool(obs.get("obscured"))
            accuracy = obs.get("accuracy_m")

            popup_html = (
                f"<b>iNaturalist #{html.escape(str(obs_id))}</b><br>"
                f"Taxon : {html.escape(taxon)}<br>"
                f"Date : {html.escape(date_str)}<br>"
                f"Qualité : {html.escape(quality)}"
            )
            # Position floutée par iNaturalist (espèce ou lieu protégés) :
            # le point n'est qu'un centre de zone, à ~20 km près (D60)
            if obscured:
                popup_html += "<br><i>Position floutée par iNaturalist</i>"
            elif accuracy is not None:
                try:
                    popup_html += f"<br>Précision : ± {float(accuracy):.0f} m"
                except (TypeError, ValueError):
                    pass
            fill = _INAT_COLOR if quality == "research" else "#9e7cc1"
            folium.CircleMarker(
                location=[float(lat), float(lng)],
                radius=6,
                color=fill if obscured else "#ffffff",
                weight=2 if obscured else 1.5,
                dash_array="3 3" if obscured else None,
                fill=not obscured,
                fill_color=fill,
                fill_opacity=0.9,
                # le clic ouvre cette popup, pas celle du point de la carte
                bubbling_mouse_events=False,
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(fg)

        fg.add_to(folium_map)
        logger.info(
            "✅ iNaturalist : %d observations affichées",
            len(self._inat_observations),
        )
        return fg.get_name()

    # ──────────────────────────────────────────────────────
    # Section météo du panneau
    # ──────────────────────────────────────────────────────
    def _weather_section_html(self) -> str:
        """Section météo du panneau ('' sans données). Même échelle que la
        carte : weather.py classe chaque jour dans les 6 classes
        config.PROBABILITY_* (D44). Crédit Open-Meteo exigé par la licence
        CC BY 4.0 des données (D42)."""
        if not self._weather_days:
            return ""

        good = config.PROBABILITY_LABELS.index("Bon")
        medium = config.PROBABILITY_LABELS.index("Moyen")
        rows = ""
        best_day = None
        best_score = -1.0
        for d in self._weather_days:
            score = float(d.score)
            level = int(d.level)
            label = html.escape(str(d.label))
            date_str = html.escape(str(d.date_fr))
            # Icône : favorable (Bon et +), moyen, défavorable
            if level >= good:
                icon = "&#9733;"  # star
                cls = "wx-good"
            elif level >= medium:
                icon = "&#9679;"  # circle
                cls = "wx-ok"
            else:
                icon = "&#9675;"  # empty circle
                cls = "wx-bad"
            # Même format que les scores de la carte (0,81), pas en %
            score_txt = f"{score:.2f}".replace(".", ",")
            rows += (
                f'<tr class="{cls}">'
                f"<td>{date_str}</td>"
                f"<td>{icon} {label}</td>"
                f"<td>{score_txt}</td>"
                f"</tr>"
            )
            if score > best_score:
                best_score = score
                best_day = date_str

        burst_html = ""
        for d in self._weather_days:
            if getattr(d, "burst", None) is not None:
                burst_html = (
                    '<div class="wx-burst">'
                    "&#x1F4A7; Signal fructification détecté"
                    "</div>"
                )
                break

        best_html = ""
        if best_day:
            best_html = (
                f'<div class="wx-best">'
                f"Meilleur jour : <b>{best_day}</b>"
                f"</div>"
            )

        credit_html = (
            '<div class="wx-credit">Données météo : '
            '<a href="https://open-meteo.com/" target="_blank" '
            'rel="noopener">Open-Meteo.com</a> '
            '(<a href="https://creativecommons.org/licenses/by/4.0/" '
            'target="_blank" rel="noopener">CC BY 4.0</a>)'
            "</div>"
        )

        logger.info("✅ Panneau météo : %d jours affichés", len(self._weather_days))
        return (
            '<section class="cs-section cs-wx">'
            "<h2>Météo pour les morilles</h2>"
            f"<table>{rows}</table>{burst_html}{best_html}{credit_html}"
            "</section>"
        )

    # ──────────────────────────────────────────────────────
    # Hotspots
    # ──────────────────────────────────────────────────────
    def _hotspots_payload(self) -> dict[str, list[dict[str, Any]]]:
        """Hotspots compacts par espèce (rang, position, attributs), triés
        comme le CSV. Position = cellule la mieux notée de l'amas, comme le
        CSV et le GPX (D50). Les marqueurs sont créés en JS pour le top N.
        Hors emprise d'étude : écartés (D39), le rang reste celui du CSV."""
        payload: dict[str, list[dict[str, Any]]] = {}
        for species in self.species_list:
            sp_hotspots = self.hotspots_by_species.get(species, [])
            inside = self.inside_study_area(sp_hotspots)
            records: list[dict[str, Any]] = []
            for rank, h in enumerate(sp_hotspots, 1):
                if len(records) >= self.max_hotspot_markers:
                    break
                if not inside[rank - 1]:
                    continue
                lon, lat = self._l93_to_wgs84(h["x_l93"], h["y_l93"])
                rec: dict[str, Any] = {
                    "r": rank,
                    "id": int(h.get("id", rank)),
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "s": round(float(h.get("mean_score", 0.0)), 3),
                    "a": round(float(h.get("size_m2", 0.0)) / 10_000.0, 2),
                }
                alt = h.get("altitude")
                if alt is not None and np.isfinite(alt):
                    rec["alt"] = int(round(float(alt)))
                slope = h.get("mean_slope")
                if slope is not None and np.isfinite(slope):
                    rec["sl"] = round(float(slope), 1)
                tree = config.tree_label(h.get("dominant_species"))
                if tree:
                    rec["tr"] = tree
                geology = config.geology_label(h.get("dominant_geology"))
                if geology:
                    rec["g"] = geology
                records.append(rec)
            payload[species] = records
            n_out = int(np.count_nonzero(~inside))
            logger.info(
                "✅ Hotspots [%s] : %d sur la carte%s", species, len(records),
                f" ({n_out} hors emprise d'étude écartés)" if n_out else "",
            )
        return payload

    # ──────────────────────────────────────────────────────
    # Landmarks
    # ──────────────────────────────────────────────────────
    def _add_landmarks(self, folium_map: folium.Map) -> str:
        """Repères : petits losanges gris neutres, pour ne pas ressembler
        aux hotspots (disques numérotés aux couleurs des classes)."""
        lg = folium.FeatureGroup(
            name="Points de repère", show=True, control=False,
        )
        for lm in self.landmarks:
            popup_html = (
                f"<b>{html.escape(str(lm['name']))}</b><br>"
                f"{html.escape(str(lm.get('info', '')))}<br>"
                f"<small>{lm['lat']:.4f}°N, {lm['lon']:.4f}°E</small>"
            )
            folium.Marker(
                location=[lm["lat"], lm["lon"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=str(lm["name"]),
                icon=folium.DivIcon(
                    html="<span></span>",
                    icon_size=(18, 18),
                    icon_anchor=(9, 9),
                    class_name="cs-lm",
                ),
            ).add_to(lg)
        lg.add_to(folium_map)
        return lg.get_name()

    # ══════════════════════════════════════════════════════
    # CARTE FOLIUM — point d'entrée principal
    # ══════════════════════════════════════════════════════
    def create_folium_map(
        self,
        output: str | Path = "output/carte_morilles.html",
        *,
        grid_threshold: float | None = None,  # conservé pour compat API
        show_elimination: bool = True,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        m = folium.Map(
            location=[config.MAP_CENTER["lat"], config.MAP_CENTER["lon"]],
            zoom_start=10,
            tiles=None,
            zoom_control="topright",
        )
        m.get_root().header.add_child(  # type: ignore[attr-defined]
            Element(f"<title>{html.escape(self._page_title())}</title>"),
        )
        basemaps = self._add_basemaps(m)

        # ── Phase 1 : couches de diagnostic (option) — reprojection ──
        t_prep = time.perf_counter()
        diag_specs = (
            self._prepare_diagnostics()
            if self.debug_layers and show_elimination
            else []
        )
        extra_specs = self._prepare_extra_layers()
        logger.info(
            "⏱️  Préparation diagnostic et calques (reproj.) : %.2fs — "
            "%d couches, %d calques",
            time.perf_counter() - t_prep,
            len(diag_specs), len(extra_specs),
        )

        # ── Phase 2 : encodage PNG parallèle (score par espèce,
        #    terrain, diagnostic, calques supplémentaires) ──
        t_enc = time.perf_counter()
        n_workers = max(
            1, min(
                _PNG_WORKERS,
                len(diag_specs) + len(extra_specs) + len(self.species_list) + 1,
            ),
        )
        diag_uris: dict[str, str] = {}
        extra_uris: dict[str, str] = {}
        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="png",
        ) as pool:
            diag_futures = {pool.submit(spec.render): spec for spec in diag_specs}
            extra_futures = {
                pool.submit(x.render): str(x.cfg["key"])
                for x in extra_specs if x.render is not None
            }
            data_futures = {
                sp: pool.submit(self._build_data_png_bytes, sp)
                for sp in self.species_list
            }
            terrain_future = pool.submit(self._build_terrain_png_bytes)

            for fut in as_completed(diag_futures):
                spec = diag_futures[fut]
                diag_uris[spec.key] = self._png_to_data_uri(fut.result())
            for fut in as_completed(extra_futures):
                extra_uris[extra_futures[fut]] = self._png_to_data_uri(fut.result())
            data_uris_by_species = {
                sp: self._png_to_data_uri(fut.result())
                for sp, fut in data_futures.items()
            }
            terrain_uri = self._png_to_data_uri(terrain_future.result())

        logger.info(
            "⏱️  Encoding PNG parallèle : %.2fs — %d workers",
            time.perf_counter() - t_enc,
            n_workers,
        )

        # ── Phase 3 : couches de diagnostic, masquées au départ ──
        diag_layers: list[dict[str, Any]] = []
        for spec in diag_specs:
            overlay = ImageOverlay(
                image=diag_uris[spec.key],
                bounds=self._bounds,
                opacity=spec.opacity,
                name=spec.label,
                show=False,
                control=False,
                interactive=False,
                pixelated=False,
                zIndex=3,
            )
            overlay.add_to(m)
            diag_layers.append({
                "key": spec.key,
                "group": spec.group,
                "label": spec.label,
                "species": spec.species,
                "var": overlay.get_name(),
                "swatch": spec.swatch,
            })
            logger.info("✅ %s", spec.log_msg)

        # ── Couches vectorielles + marqueurs fixes ──
        layer_vars: dict[str, str | None] = {
            # Créée par le JS du panneau (volet dédié au-dessus des scores)
            "area": "cartom_study_area" if self._study_area is not None else None,
            "inat": self._add_inaturalist_layer(m),
            "hydro": self._add_hydro_vector_layer(m),
            "landmarks": self._add_landmarks(m),
        }

        MeasureControl(
            position="topright",
            primary_length_unit="meters",
            secondary_length_unit="kilometers",
            primary_area_unit="hectares",
            secondary_area_unit="sqmeters",
        ).add_to(m)

        # ── Panneau latéral + couche de probabilité JS ──
        self._add_sidebar_ui(
            m,
            data_uris_by_species=data_uris_by_species,
            terrain_uri=terrain_uri,
            basemaps=basemaps,
            layer_vars=layer_vars,
            diag_layers=diag_layers,
            hotspots=self._hotspots_payload(),
            weather_html=self._weather_section_html(),
            extra_layers=[
                x.cfg for x in extra_specs
                if x.render is None or str(x.cfg["key"]) in extra_uris
            ],
            extra_uris=extra_uris,
        )

        m.fit_bounds(self._bounds)
        # Bibliothèques dans la page plutôt que sur des CDN (D43)
        _embed_libraries(m)

        # ── Sauvegarde (équivalent de m.save, meta viewport corrigé) ──
        try:
            html_doc = _allow_page_zoom(m.get_root().render())
            output.write_bytes(html_doc.encode("utf-8"))
            size_mb = output.stat().st_size / (1024 * 1024)
            dt = time.perf_counter() - t0
            logger.info(
                "🗺️  Carte sauvegardée : %s (%.1f MB) en %.1fs",
                output, size_mb, dt,
            )
        except OSError:
            logger.exception("Impossible de sauvegarder la carte HTML")
            raise

        return output

    # ══════════════════════════════════════════════════════
    # EXPORT GEOTIFF
    # ══════════════════════════════════════════════════════
    def export_geotiff(
        self,
        output: str | Path = "output/esculenta_probability.tif",
        *,
        species: str | None = None,
    ) -> Path:
        """Export GeoTIFF de la probabilité.

        v2.6.0 : species kwarg sélectionne le raster à exporter.
        Si None, utilise l'espèce primaire (1ère du dict models).
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        sp = species or self.primary_species
        score_src = self._final_score_by_species.get(sp, self._final_score)
        score: np.ndarray = score_src.copy()
        nodata_mask: np.ndarray = np.asarray(~np.isfinite(score))
        score[nodata_mask] = _NODATA

        transform = from_bounds(
            self._xmin_l93, self._ymin_l93,
            self._xmax_l93, self._ymax_l93,
            self.grid.nx, self.grid.ny,
        )

        sp_params = config.SPECIES_PARAMS.get(
            sp, config.SPECIES_PARAMS[config.DEFAULT_SPECIES],
        )
        sp_common = sp_params.get("common_name", sp)

        try:
            with rasterio.open(
                str(output),
                "w",
                driver="GTiff",
                height=self.grid.ny,
                width=self.grid.nx,
                count=1,
                dtype=np.float32,
                crs="EPSG:2154",
                transform=transform,
                nodata=_NODATA,
                compress="lzw",
            ) as dst:
                dst.write(score.astype(np.float32), 1)
                dst.update_tags(
                    DESCRIPTION=f"Cartomorilles — probabilité {sp_common} ({sp})",
                    SPECIES=sp,
                    CELL_SIZE=str(config.CELL_SIZE),
                    CRS="EPSG:2154",
                    NODATA=str(_NODATA),
                )
            logger.info("✅ GeoTIFF [%s] : %s", sp, output)
        except OSError:
            logger.exception("Impossible d'écrire le GeoTIFF")
            raise

        return output

    # ══════════════════════════════════════════════════════
    # EXPORT GEOPACKAGE
    # ══════════════════════════════════════════════════════
    def export_gpkg_grid(
        self,
        output: str | Path = "output/esculenta_grid.gpkg",
        threshold: float = 0.15,
        *,
        species: str | None = None,
    ) -> Path:
        """Export GeoPackage vectorisé (numpy batch).

        v2.6.0 : species kwarg sélectionne le raster source.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        sp = species or self.primary_species
        logger.info("Export GeoPackage [%s] (seuil=%.2f)…", sp, threshold)

        score: np.ndarray = self._final_score_by_species.get(sp, self._final_score)
        prob_classes: np.ndarray = self._prob_classes_by_species.get(sp, self._prob_classes)
        elim_mask: np.ndarray = self._elim_mask_by_species.get(sp, self._elim_mask)
        half = float(getattr(self.grid, "cell_size", config.CELL_SIZE)) / 2.0
        labels = list(config.PROBABILITY_LABELS)
        # Critères et confiances de l'espèce (instantané du modèle) : la
        # grille ne garde que ceux de la dernière espèce calculée (D60)
        model = self.models.get(sp)

        above = np.isfinite(score) & (score >= threshold)
        iy_arr, ix_arr = np.where(above)

        if iy_arr.size == 0:
            logger.warning("Aucune cellule au-dessus du seuil %.2f", threshold)
            return output

        n = iy_arr.size
        logger.info("  %d cellules à exporter", n)

        cx_arr = np.asarray(self.grid.x_coords)[ix_arr]
        cy_arr = np.asarray(self.grid.y_coords)[iy_arr]

        geometries = [
            shapely_box(cx - half, cy - half, cx + half, cy + half)
            for cx, cy in zip(cx_arr.tolist(), cy_arr.tolist())
        ]

        score_vals = score[iy_arr, ix_arr]
        cls_vals = prob_classes[iy_arr, ix_arr].astype(int)

        data: dict[str, Any] = {
            "score": np.round(score_vals, 4),
            "prob_class": cls_vals,
            "prob_label": [
                labels[c] if 0 <= c < len(labels) else "?"
                for c in cls_vals.tolist()
            ],
            "eliminated": elim_mask[iy_arr, ix_arr].astype(bool),
        }

        alt_grid = getattr(self.grid, "altitude", None)
        if isinstance(alt_grid, np.ndarray) and alt_grid.shape == score.shape:
            av = alt_grid[iy_arr, ix_arr].astype(np.float64)
            av[~np.isfinite(av)] = np.nan
            data["altitude"] = np.round(av, 1)

        slp_grid = getattr(self.grid, "slope", None)
        if isinstance(slp_grid, np.ndarray) and slp_grid.shape == score.shape:
            sv = slp_grid[iy_arr, ix_arr].astype(np.float64)
            sv[~np.isfinite(sv)] = np.nan
            data["slope"] = np.round(sv, 2)

        conf_dict: dict[str, Any] = getattr(
            model, "score_confidence", None,
        ) or getattr(self.grid, "score_confidence", {})
        if isinstance(conf_dict, dict):
            for crit, conf_val in conf_dict.items():
                if isinstance(conf_val, (int, float)):
                    data[f"conf_{crit}"] = round(float(conf_val), 2)

        scores_dict: dict[str, Any] = getattr(
            model, "scores", None,
        ) or getattr(self.grid, "scores", {})
        if isinstance(scores_dict, dict):
            for crit_name, crit_arr in scores_dict.items():
                if (
                    isinstance(crit_arr, np.ndarray)
                    and crit_arr.shape == score.shape
                ):
                    cv = crit_arr[iy_arr, ix_arr].astype(np.float64)
                    cv[~np.isfinite(cv)] = np.nan
                    data[f"s_{crit_name}"] = np.round(cv, 3)

        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:2154")

        try:
            gdf.to_file(str(output), driver="GPKG")
            logger.info(
                "✅ GeoPackage : %s (%s cellules)",
                output, f"{len(gdf):,}",
            )
        except OSError:
            logger.exception("Impossible d'écrire le GeoPackage")
            raise

        return output