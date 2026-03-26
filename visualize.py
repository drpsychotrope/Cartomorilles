"""
visualize.py — Cartomorilles v2.2.0
Visualisation interactive (Folium), export GeoTIFF et GeoPackage.

v2.2.0 :
  - Panneaux déplaçables (drag & drop) : légende, colorbar, opacité, info
  - GeoJsonTooltip sticky sur la grille
  - Panneau info cellule (bottomright)
  - Colorbar HTML custom (remplace branca LinearColormap)
  - JS consolidé en un seul bloc
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
matplotlib.use("Agg")

import folium                                                  # noqa: E402
import geopandas as gpd                                        # noqa: E402
import matplotlib.colors as mcolors                            # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import rasterio                                                # noqa: E402
from branca.element import Element  # type: ignore[import-untyped]  # noqa: E402
from folium.plugins import MeasureControl, MiniMap, MousePosition  # noqa: E402
from folium.raster_layers import ImageOverlay                  # noqa: E402
from pyproj import Transformer                                 # noqa: E402
from rasterio.transform import from_bounds                     # noqa: E402
from shapely.geometry import box as shapely_box                # noqa: E402

import config                                                  # noqa: E402

if TYPE_CHECKING:
    from scoring import MorilleScoring

logger = logging.getLogger("cartomorilles.visualize")

__all__ = ["MorilleVisualizer"]

# ───────────────────────────────────────────────────────────
# Constantes
# ───────────────────────────────────────────────────────────
_NODATA: float = -9999.0
_MAX_GRID_CELLS: int = 500_000
_MAX_HOTSPOT_MARKERS: int = 30

_CMAP_STOPS: tuple[tuple[float, str], ...] = (
    (0.00, "#d73027"),
    (0.15, "#f46d43"),
    (0.30, "#fdae61"),
    (0.45, "#fee08b"),
    (0.60, "#a6d96a"),
    (0.75, "#1a9850"),
    (1.00, "#006837"),
)

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "morilles",
    [(pos, col) for pos, col in _CMAP_STOPS],
)

_CLASS_COLORS: tuple[str, ...] = (
    "#d73027", "#f46d43", "#fdae61", "#fee08b", "#1a9850", "#006837",
)


# ═══════════════════════════════════════════════════════════
# Classe principale
# ═══════════════════════════════════════════════════════════
class MorilleVisualizer:
    """Génère les sorties visuelles à partir d'un MorilleScoring terminé."""

    def __init__(
        self,
        scoring_model: MorilleScoring,
        *,
        hotspots: list[dict[str, Any]] | None = None,
        landmarks: list[dict[str, Any]] | None = None,
        max_hotspot_markers: int = _MAX_HOTSPOT_MARKERS,
        max_grid_cells: int = _MAX_GRID_CELLS,
    ) -> None:
        self._validate_model(scoring_model)
        self.model: MorilleScoring = scoring_model
        self.grid = scoring_model.grid
        self.hotspots: list[dict[str, Any]] = hotspots or []
        self.landmarks: list[dict[str, Any]] = landmarks or _default_landmarks()
        self.max_hotspot_markers = max_hotspot_markers
        self.max_grid_cells = max_grid_cells

        _fs = scoring_model.final_score
        _pc = scoring_model.probability_classes
        _em = scoring_model.elimination_mask
        _ed = scoring_model.elimination_detail
        assert _fs is not None, "final_score is None after validation"
        assert _pc is not None, "probability_classes is None after validation"
        assert _em is not None, "elimination_mask is None after validation"
        assert _ed is not None, "elimination_detail is None after validation"
        self._final_score: np.ndarray = _fs
        self._prob_classes: np.ndarray = _pc
        self._elim_mask: np.ndarray = _em
        self._elim_detail: dict[str, Any] = _ed

        self._to_wgs84 = Transformer.from_crs(
            "EPSG:2154", "EPSG:4326", always_xy=True,
        )

        half = config.CELL_SIZE / 2.0
        self._xmin_l93 = float(np.min(self.grid.x_coords)) - half
        self._xmax_l93 = float(np.max(self.grid.x_coords)) + half
        self._ymin_l93 = float(np.min(self.grid.y_coords)) - half
        self._ymax_l93 = float(np.max(self.grid.y_coords)) + half

        corners_x = [self._xmin_l93, self._xmax_l93,
                      self._xmin_l93, self._xmax_l93]
        corners_y = [self._ymin_l93, self._ymin_l93,
                      self._ymax_l93, self._ymax_l93]
        lons, lats = self._to_wgs84.transform(corners_x, corners_y)
        self._south = float(np.min(lats))
        self._north = float(np.max(lats))
        self._west = float(np.min(lons))
        self._east = float(np.max(lons))

        logger.debug(
            "Emprise WGS84 : S=%.5f N=%.5f W=%.5f E=%.5f",
            self._south, self._north, self._west, self._east,
        )

    # ──────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _validate_model(model: Any) -> None:
        required_attrs = [
            "final_score", "probability_classes", "grid",
            "elimination_mask", "elimination_detail",
        ]
        missing = [a for a in required_attrs if not hasattr(model, a)]
        if missing:
            raise AttributeError(
                f"MorilleScoring incomplet — attributs manquants : {missing}"
            )
        for attr in ["final_score", "probability_classes",
                     "elimination_mask", "elimination_detail"]:
            if getattr(model, attr) is None:
                raise ValueError(
                    f"MorilleScoring.{attr} is None — pipeline incomplet ?"
                )
        grid_attrs = [
            "x_coords", "y_coords", "transform", "nx", "ny",
            "urban_mask", "water_mask", "nodata_mask", "scores",
        ]
        grid_missing = [a for a in grid_attrs if not hasattr(model.grid, a)]
        if grid_missing:
            raise AttributeError(
                f"GridBuilder incomplet — attributs manquants : {grid_missing}"
            )

    # ──────────────────────────────────────────────────────
    # Coordonnées
    # ──────────────────────────────────────────────────────
    def _l93_to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lon), float(lat)

    # ──────────────────────────────────────────────────────
    # JS consolidé : drag + opacité + info curseur
    # ──────────────────────────────────────────────────────
    def _add_interactive_controls(self, folium_map: folium.Map) -> None:
        """Injecte le JS unique pour : panneaux draggable, opacité, info curseur."""
        js_block = r"""
        <style>
        .cartom-panel {
            background: rgba(255,255,255,0.95);
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,.35);
            font: 12px/1.5 'Segoe UI', Arial, sans-serif;
            position: fixed;
            z-index: 10000;
            pointer-events: auto;
            max-width: 260px;
        }
        .cartom-panel .cartom-titlebar {
            cursor: grab;
            background: linear-gradient(135deg, #2d6a2e, #3a9e3f);
            color: white;
            padding: 5px 10px;
            border-radius: 8px 8px 0 0;
            font-weight: bold;
            font-size: 12px;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .cartom-panel .cartom-titlebar:active { cursor: grabbing; }
        .cartom-panel .cartom-body { padding: 8px 12px; }
        .cartom-panel .cartom-minimize {
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
            opacity: 0.8;
        }
        .cartom-panel .cartom-minimize:hover { opacity: 1; }
        .cartom-panel.minimized .cartom-body { display: none; }

        /* Positionnement initial */
        #cartom-legend   { bottom: 30px;  left: 30px; }
        #cartom-colorbar { top: 10px;     right: 10px; left: auto; }
        #cartom-opacity  { top: 80px;     left: 10px; }
        #cartom-info     { bottom: 30px;  right: 10px; }
        </style>

        <div id="cartom-legend" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🍄 Probabilité morilles</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body" id="cartom-legend-body"></div>
        </div>

        <div id="cartom-colorbar" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🎨 Échelle</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body" id="cartom-colorbar-body"></div>
        </div>

        <div id="cartom-opacity" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>🍄 Opacité</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body">
                <input type="range" min="0" max="100" value="70"
                       style="width:100%;" id="cartom_opacity_slider">
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#666;">
                    <span>0%</span><span id="cartom_opval">70%</span><span>100%</span>
                </div>
            </div>
        </div>

        <div id="cartom-info" class="cartom-panel">
            <div class="cartom-titlebar">
                <span>📍 Cellule</span>
                <span class="cartom-minimize" onclick="this.closest('.cartom-panel').classList.toggle('minimized')">−</span>
            </div>
            <div class="cartom-body" id="cartom-info-body">
                Survolez la grille…
            </div>
        </div>

        <script>
        document.addEventListener('DOMContentLoaded', function(){

            /* ═══════════════════════════════════════════
               DRAG — tous les panneaux .cartom-panel
               ═══════════════════════════════════════════ */
            document.querySelectorAll('.cartom-panel').forEach(function(panel){
                var bar = panel.querySelector('.cartom-titlebar');
                if (!bar) return;
                var dx=0, dy=0, mx=0, my=0;

                bar.addEventListener('mousedown', function(e){
                    if (e.target.classList.contains('cartom-minimize')) return;
                    e.preventDefault();
                    mx = e.clientX; my = e.clientY;

                    /* Convertir bottom/right en top/left avant drag */
                    var rect = panel.getBoundingClientRect();
                    panel.style.top  = rect.top + 'px';
                    panel.style.left = rect.left + 'px';
                    panel.style.bottom = 'auto';
                    panel.style.right  = 'auto';

                    function onMove(ev){
                        dx = ev.clientX - mx; dy = ev.clientY - my;
                        mx = ev.clientX;      my = ev.clientY;
                        var t = parseInt(panel.style.top)  + dy;
                        var l = parseInt(panel.style.left) + dx;
                        /* Clamp dans la fenêtre */
                        t = Math.max(0, Math.min(t, window.innerHeight - 40));
                        l = Math.max(0, Math.min(l, window.innerWidth - 40));
                        panel.style.top  = t + 'px';
                        panel.style.left = l + 'px';
                    }
                    function onUp(){
                        document.removeEventListener('mousemove', onMove);
                        document.removeEventListener('mouseup', onUp);
                    }
                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup', onUp);
                });

                /* Empêcher Leaflet d'intercepter les événements */
                L.DomEvent.disableClickPropagation(panel);
                L.DomEvent.disableScrollPropagation(panel);
            });

            /* ═══════════════════════════════════════════
               LÉGENDE discrète — remplir #cartom-legend-body
               ═══════════════════════════════════════════ */
            var legendBody = document.getElementById('cartom-legend-body');
            if (legendBody) {
                var labels = %LABELS_JSON%;
                var thresholds = %THRESHOLDS_JSON%;
                var colors = %COLORS_JSON%;
                var cellSize = %CELL_SIZE%;
                var html = '';
                for (var i = labels.length - 1; i >= 0; i--) {
                    var range;
                    if (i === 0) range = '< ' + thresholds[0].toFixed(2);
                    else if (i === labels.length - 1) range = '≥ ' + thresholds[thresholds.length-1].toFixed(2);
                    else range = thresholds[i-1].toFixed(2) + ' – ' + thresholds[i].toFixed(2);
                    html += '<div style="margin:2px 0;">'
                        + '<span style="display:inline-block;width:18px;height:13px;'
                        + 'background:' + colors[i] + ';border-radius:2px;'
                        + 'margin-right:6px;vertical-align:middle;"></span>'
                        + '<span style="vertical-align:middle;">'
                        + labels[i] + ' (' + range + ')</span></div>';
                }
                html += '<div style="margin-top:6px;color:#888;font-size:10px;">'
                    + 'Résolution : ' + cellSize + 'm × ' + cellSize + 'm</div>';
                legendBody.innerHTML = html;
            }

            /* ═══════════════════════════════════════════
               COLORBAR continue — remplir #cartom-colorbar-body
               ═══════════════════════════════════════════ */
            var cbBody = document.getElementById('cartom-colorbar-body');
            if (cbBody) {
                var stops = %CMAP_STOPS_JSON%;
                var grad = stops.map(function(s){ return s[1] + ' ' + (s[0]*100) + '%'; }).join(', ');
                cbBody.innerHTML =
                    '<div style="height:14px;border-radius:3px;'
                    + 'background:linear-gradient(to right, ' + grad + ');'
                    + 'margin-bottom:3px;"></div>'
                    + '<div style="display:flex;justify-content:space-between;font-size:10px;color:#666;">'
                    + '<span>0.0</span><span>0.3</span><span>0.5</span>'
                    + '<span>0.8</span><span>1.0</span></div>'
                    + '<div style="text-align:center;font-size:10px;color:#888;margin-top:2px;">'
                    + 'Score de probabilité morilles</div>';
            }

            /* ═══════════════════════════════════════════
               OPACITÉ — slider — cible TOUTES les couches
               ═══════════════════════════════════════════ */
            var slider = document.getElementById('cartom_opacity_slider');
            var opVal  = document.getElementById('cartom_opval');
            if (slider) {
                var map = null;
                for (var k in window) {
                    try { if (window[k] instanceof L.Map) { map = window[k]; break; } }
                    catch(e){}
                }
                if (!map) {
                    var containers = document.querySelectorAll('.leaflet-container');
                    for (var ci = 0; ci < containers.length; ci++) {
                        for (var k2 in window) {
                            try {
                                if (window[k2] instanceof L.Map &&
                                    window[k2].getContainer() === containers[ci]) {
                                    map = window[k2]; break;
                                }
                            } catch(e2){}
                        }
                        if (map) break;
                    }
                }

                function setLayerOpacity(layer, v) {
                    /* ImageOverlay (PNG probabilité) */
                    if (layer instanceof L.ImageOverlay) {
                        layer.setOpacity(v);
                        return;
                    }
                    /* GeoJSON feature (grille) */
                    if (layer.setStyle && layer.feature) {
                        layer.setStyle({fillOpacity: v, opacity: v});
                        return;
                    }
                    /* Rectangle (masques éliminatoires) */
                    if (layer instanceof L.Rectangle || layer instanceof L.Polygon) {
                        if (layer.setStyle) {
                            layer.setStyle({fillOpacity: v * 0.5, opacity: v});
                        }
                        return;
                    }
                    /* FeatureGroup / LayerGroup récursif */
                    if (layer.eachLayer) {
                        layer.eachLayer(function(sub){ setLayerOpacity(sub, v); });
                    }
                }

                slider.addEventListener('input', function(e){
                    var v = parseInt(e.target.value) / 100.0;
                    if (opVal) opVal.textContent = e.target.value + '%';
                    if (!map) return;
                    map.eachLayer(function(layer){
                        /* Ignorer les TileLayers (fonds de carte) */
                        if (layer instanceof L.TileLayer) return;
                        setLayerOpacity(layer, v);
                    });
                });
            }

            /* ═══════════════════════════════════════════
               INFO CELLULE — mouseover sur GeoJSON
               ═══════════════════════════════════════════ */
            var infoBody = document.getElementById('cartom-info-body');
            if (map && infoBody) {
                map.eachLayer(function(layer){
                    if (!layer.eachLayer) return;
                    layer.eachLayer(function(sub){
                        if (!sub.feature || !sub.feature.properties ||
                            sub.feature.properties.score === undefined) return;
                        sub.on('mouseover', function(ev){
                            var p = ev.target.feature.properties;
                            var ll = ev.latlng;
                            var h = '<b>🍄 ' + (p.score * 100).toFixed(1) + '%</b>';
                            if (p.classe) h += ' — <em>' + p.classe + '</em>';
                            h += '<br>';
                            if (p.alt_m !== undefined) h += '⛰️ ' + p.alt_m + ' m<br>';
                            if (p.pente_deg !== undefined) h += '📐 ' + p.pente_deg + '°<br>';
                            /* Critères individuels */
                            var crits = ['tree_species','geology','dist_water',
                                         'altitude','slope','canopy_openness'];
                            for (var ci2 = 0; ci2 < crits.length; ci2++){
                                var c = crits[ci2];
                                if (p[c] !== undefined){
                                    h += '<span style="color:#666;">'
                                        + c.replace(/_/g,' ') + ': '
                                        + p[c].toFixed(2) + '</span><br>';
                                }
                            }
                            if (ll) h += '<span style="font-size:9px;color:#aaa;">'
                                + ll.lat.toFixed(5) + ', ' + ll.lng.toFixed(5) + '</span>';
                            infoBody.innerHTML = h;
                        });
                        sub.on('mouseout', function(){
                            infoBody.innerHTML = 'Survolez la grille…';
                        });
                    });
                });
            }
        });
        </script>
        """

        import json
        labels_json = json.dumps(list(config.PROBABILITY_LABELS))
        thresholds_json = json.dumps(list(config.PROBABILITY_THRESHOLDS))
        colors_json = json.dumps(list(_CLASS_COLORS[:len(config.PROBABILITY_LABELS)]))
        cell_size_str = str(config.CELL_SIZE)
        cmap_stops_json = json.dumps([[pos, col] for pos, col in _CMAP_STOPS])

        js_final = (
            js_block
            .replace("%LABELS_JSON%", labels_json)
            .replace("%THRESHOLDS_JSON%", thresholds_json)
            .replace("%COLORS_JSON%", colors_json)
            .replace("%CELL_SIZE%", cell_size_str)
            .replace("%CMAP_STOPS_JSON%", cmap_stops_json)
        )

        folium_map.get_root().html.add_child(  # type: ignore[attr-defined]
            Element(js_final),
        )

    # ──────────────────────────────────────────────────────
    # CARTE FOLIUM
    # ──────────────────────────────────────────────────────
    def create_folium_map(
        self,
        output: str | Path = "output/carte_morilles.html",
        *,
        grid_threshold: float | None = None,
        show_elimination: bool = True,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if grid_threshold is None:
            grid_threshold = 0.15

        m = folium.Map(
            location=[config.MAP_CENTER["lat"], config.MAP_CENTER["lon"]],
            zoom_start=14,
            tiles=None,
            control_scale=True,
        )

        self._add_basemaps(m)
        self._add_probability_overlay(m, output.parent)
        self._add_detailed_grid(m, threshold=grid_threshold)

        if show_elimination:
            self._add_elimination_layers(m)

        self._add_hotspot_markers(m)
        self._add_landmarks(m)

        MiniMap(toggle_display=True).add_to(m)
        MousePosition(
            position="bottomright", separator=" | ", prefix="Curseur :",
        ).add_to(m)
        MeasureControl(
            position="topleft",
            primary_length_unit="meters",
            primary_area_unit="sqmeters",
        ).add_to(m)

        folium.LayerControl(collapsed=False, position="topright").add_to(m)

        # JS consolidé — APRÈS toutes les couches
        self._add_interactive_controls(m)

        m.fit_bounds([[self._south, self._west], [self._north, self._east]])

        try:
            m.save(str(output))
            logger.info("🗺️  Carte sauvegardée : %s", output)
        except OSError:
            logger.exception("Impossible de sauvegarder la carte HTML")
            raise

        return output

    # ──────────────────────────────────────────────────────
    # Fonds de carte
    # ──────────────────────────────────────────────────────
    def _add_basemaps(self, folium_map: folium.Map) -> None:
        for name, cfg in config.BASEMAPS.items():
            is_default = name == config.DEFAULT_BASEMAP
            tiles = cfg["tiles"]
            if tiles in {"OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"}:
                folium.TileLayer(
                    tiles=tiles, name=name, show=is_default,
                ).add_to(folium_map)
            else:
                folium.TileLayer(
                    tiles=tiles,
                    attr=cfg.get("attr", ""),
                    name=cfg.get("name", name),
                    max_zoom=cfg.get("max_zoom", 18),
                    show=is_default,
                ).add_to(folium_map)

    # ──────────────────────────────────────────────────────
    # Overlay PNG
    # ──────────────────────────────────────────────────────
    def _add_probability_overlay(
        self, folium_map: folium.Map, output_dir: Path,
    ) -> None:
        score: np.ndarray = self._final_score.copy()
        valid: np.ndarray = np.asarray(np.isfinite(score))

        if not np.any(valid):
            logger.warning("Score entièrement NaN — overlay ignoré")
            return

        score_safe: np.ndarray = np.where(valid, score, 0.0)
        colored: np.ndarray = np.asarray(_CMAP(score_safe))

        thresholds = list(config.PROBABILITY_THRESHOLDS)
        alpha = np.zeros_like(score_safe, dtype=np.float32)
        alpha_steps = np.linspace(0.15, 0.65, len(thresholds) + 1)[1:]
        for i, th in enumerate(thresholds):
            alpha[score_safe >= th] = float(alpha_steps[i])
        alpha[score_safe >= thresholds[-1]] = 0.70
        alpha[~valid] = 0.0
        colored[:, :, 3] = alpha

        img_path = output_dir / "probability_overlay.png"
        try:
            plt.imsave(str(img_path), colored)
        except OSError:
            logger.exception("Impossible d'écrire le PNG overlay")
            return

        ImageOverlay(
            image=str(img_path),
            bounds=[[self._south, self._west], [self._north, self._east]],
            opacity=0.7,
            name="🍄 Probabilité morilles",
            interactive=True,
            zindex=1,
        ).add_to(folium_map)

    # ──────────────────────────────────────────────────────
    # Masques éliminatoires
    # ──────────────────────────────────────────────────────
    def _add_elimination_layers(self, folium_map: folium.Map) -> None:
        if not self._elim_detail:
            return

        label_map: dict[str, tuple[str, str]] = {
            "urban":    ("🏠 Zones urbaines", "gray"),
            "water":    ("💧 Cours d'eau", "blue"),
            "nodata":   ("❓ NoData", "black"),
            "species":  ("🌳 Espèces éliminatoires", "darkred"),
            "geology":  ("🪨 Géologie éliminatoire", "purple"),
            "slope":    ("⛰️ Pente éliminatoire", "orange"),
            "altitude": ("📏 Altitude éliminatoire", "brown"),
        }

        half = config.CELL_SIZE / 2.0

        for key, mask_arr in self._elim_detail.items():
            if not isinstance(mask_arr, np.ndarray):
                continue
            label, color = label_map.get(key, (f"❌ {key}", "red"))
            indices: np.ndarray = np.argwhere(mask_arr)
            if indices.size == 0:
                continue

            fg = folium.FeatureGroup(name=label, show=False)
            step = max(1, len(indices) // 5000)

            for idx in indices[::step]:
                iy, ix = int(idx[0]), int(idx[1])
                cx = float(self.grid.x_coords[ix])
                cy = float(self.grid.y_coords[iy])
                sw_lon, sw_lat = self._l93_to_wgs84(cx - half, cy - half)
                ne_lon, ne_lat = self._l93_to_wgs84(cx + half, cy + half)
                folium.Rectangle(
                    bounds=[[sw_lat, sw_lon], [ne_lat, ne_lon]],
                    color=color, fill=True,
                    fill_color=color, fill_opacity=0.35,
                    weight=0,
                ).add_to(fg)

            fg.add_to(folium_map)

    # ──────────────────────────────────────────────────────
    # Grille GeoJSON
    # ──────────────────────────────────────────────────────
    def _add_detailed_grid(
        self,
        folium_map: folium.Map,
        threshold: float = 0.15,
    ) -> None:
        score: np.ndarray = self._final_score
        prob_classes: np.ndarray = self._prob_classes

        valid_mask = np.isfinite(score) & (score >= threshold)
        ys, xs = np.nonzero(valid_mask)
        if ys.size == 0:
            logger.warning("Aucune cellule ≥ %.2f — grille ignorée", threshold)
            return

        scores_flat = score[ys, xs]
        order = np.argsort(scores_flat)[::-1]
        if order.size > self.max_grid_cells:
            logger.warning(
                "Grille tronquée : %s → %s cellules",
                f"{order.size:,}", f"{self.max_grid_cells:,}",
            )
            order = order[:self.max_grid_cells]

        ys = ys[order]
        xs = xs[order]
        scores_flat = scores_flat[order]
        count = ys.size

        score_min = threshold
        score_max = max(float(scores_flat[0]), threshold + 0.01)
        half = config.CELL_SIZE / 2.0

        cx_all = self.grid.x_coords[xs]
        cy_all = self.grid.y_coords[ys]
        _to_wgs = self.grid.to_wgs84
        sw_lon_all, sw_lat_all = _to_wgs.transform(cx_all - half, cy_all - half)
        ne_lon_all, ne_lat_all = _to_wgs.transform(cx_all + half, cy_all + half)

        alt_grid = getattr(self.grid, "altitude", None)
        slp_grid = getattr(self.grid, "slope", None)
        scores_dict: dict[str, Any] = getattr(self.grid, "scores", {})

        known_criteria: tuple[str, ...] = (
            "tree_species", "geology", "dist_water", "altitude",
            "slope", "aspect", "terrain_roughness", "canopy_openness",
            "ground_cover", "disturbance",
        )

        features: list[dict[str, Any]] = []

        for k in range(count):
            iy = int(ys[k])
            ix = int(xs[k])
            s = float(scores_flat[k])

            s_norm = float(
                np.clip((s - score_min) / (score_max - score_min), 0.0, 1.0)
            )
            hex_color = mcolors.rgb2hex(_CMAP(s_norm)[:3])

            cls_idx = int(prob_classes[iy, ix])
            cls_label = (
                config.PROBABILITY_LABELS[cls_idx]
                if 0 <= cls_idx < len(config.PROBABILITY_LABELS)
                else "?"
            )

            props: dict[str, Any] = {
                "score": round(s, 3),
                "classe": cls_label,
                "_color": hex_color,
            }

            if isinstance(alt_grid, np.ndarray) and alt_grid.shape == score.shape:
                av = float(alt_grid[iy, ix])
                if math.isfinite(av):
                    props["alt_m"] = int(av)

            if isinstance(slp_grid, np.ndarray) and slp_grid.shape == score.shape:
                sv = float(slp_grid[iy, ix])
                if math.isfinite(sv):
                    props["pente_deg"] = round(sv, 1)

            for crit in known_criteria:
                arr = scores_dict.get(crit)
                if isinstance(arr, np.ndarray) and arr.shape == score.shape:
                    v = float(arr[iy, ix])
                    if math.isfinite(v):
                        props[crit] = round(v, 2)

            coords = [[
                [float(sw_lon_all[k]), float(sw_lat_all[k])],
                [float(ne_lon_all[k]), float(sw_lat_all[k])],
                [float(ne_lon_all[k]), float(ne_lat_all[k])],
                [float(sw_lon_all[k]), float(ne_lat_all[k])],
                [float(sw_lon_all[k]), float(sw_lat_all[k])],
            ]]

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": props,
            })

        geojson_data: dict[str, Any] = {
            "type": "FeatureCollection",
            "features": features,
        }

        grid_group = folium.FeatureGroup(
            name=f"📐 Grille détaillée (≥{threshold:.2f})",
            show=True,
        )

        def style_fn(feature: dict[str, Any]) -> dict[str, Any]:
            c: str = feature.get("properties", {}).get("_color", "#ff7f00")
            return {
                "fillColor": c,
                "color": c,
                "weight": 0.3,
                "fillOpacity": 0.6,
                "opacity": 0.6,
            }

        # Tooltip sticky — suit le curseur
        tooltip = folium.GeoJsonTooltip(
            fields=["score", "classe"],
            aliases=["🍄 Score :", "Classe :"],
            localize=True,
            sticky=True,
            style=(
                "background-color:rgba(255,255,255,0.92); color:#333; "
                "font-family:'Segoe UI',Arial,sans-serif; font-size:12px; "
                "padding:5px 8px; border-radius:4px; "
                "box-shadow:0 1px 4px rgba(0,0,0,.3); pointer-events:none;"
            ),
        )

        # Popup au clic
        popup_fields = ["score", "classe"]
        popup_aliases = ["🍄 Score", "Classe"]
        for crit in known_criteria:
            if any(crit in f.get("properties", {}) for f in features[:20]):
                popup_fields.append(crit)
                popup_aliases.append(crit.replace("_", " ").title())
        if any("alt_m" in f.get("properties", {}) for f in features[:20]):
            popup_fields.append("alt_m")
            popup_aliases.append("Altitude (m)")
        if any("pente_deg" in f.get("properties", {}) for f in features[:20]):
            popup_fields.append("pente_deg")
            popup_aliases.append("Pente (°)")

        folium.GeoJson(
            geojson_data,
            style_function=style_fn,
            tooltip=tooltip,
            popup=folium.GeoJsonPopup(
                fields=popup_fields,
                aliases=popup_aliases,
                max_width=280,
            ),
        ).add_to(grid_group)

        grid_group.add_to(folium_map)
        logger.info(
            "Grille détaillée : %s cellules (seuil=%.2f)",
            f"{count:,}", threshold,
        )

    # ──────────────────────────────────────────────────────
    # Hotspots
    # ──────────────────────────────────────────────────────
    def _add_hotspot_markers(self, folium_map: folium.Map) -> None:
        if not self.hotspots:
            return

        hg = folium.FeatureGroup(name="🎯 Hotspots à prospecter")

        for h in self.hotspots[:self.max_hotspot_markers]:
            lon, lat = self._l93_to_wgs84(h["x_l93"], h["y_l93"])
            ms: float = float(h.get("mean_score", 0.0))

            color = (
                "darkgreen" if ms >= 0.75
                else ("green" if ms >= 0.60 else "orange")
            )
            icon_name = (
                "star" if ms >= 0.75
                else ("ok-sign" if ms >= 0.60 else "question-sign")
            )

            popup_lines: list[str] = [
                f"<b>Hotspot #{h.get('id', '?')}</b><br>",
                f"Score moyen : <b>{ms:.3f}</b><br>",
                f"Surface : {h.get('size_m2', 0):.0f} m²<br>",
            ]
            if h.get("altitude") is not None:
                popup_lines.append(f"Altitude : {h['altitude']:.0f} m<br>")
            if h.get("mean_slope") is not None:
                popup_lines.append(f"Pente moy. : {h['mean_slope']:.1f}°<br>")
            if h.get("dominant_tree"):
                popup_lines.append(f"Essence dom. : {h['dominant_tree']}<br>")
            if h.get("geology"):
                popup_lines.append(f"Géologie : {h['geology']}<br>")

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup("".join(popup_lines), max_width=300),
                tooltip=f"🍄 Score : {ms:.2f}",
                icon=folium.Icon(color=color, icon=icon_name),
            ).add_to(hg)

        hg.add_to(folium_map)
        logger.info(
            "Hotspots affichés : %d",
            min(len(self.hotspots), self.max_hotspot_markers),
        )

    # ──────────────────────────────────────────────────────
    # Landmarks
    # ──────────────────────────────────────────────────────
    def _add_landmarks(self, folium_map: folium.Map) -> None:
        lg = folium.FeatureGroup(name="📌 Points de repère", show=True)
        for lm in self.landmarks:
            popup_html = (
                f"<b>{lm['name']}</b><br>{lm.get('info', '')}<br>"
                f"<small>{lm['lat']:.4f}°N, {lm['lon']:.4f}°E</small>"
            )
            folium.Marker(
                location=[lm["lat"], lm["lon"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=lm["name"],
                icon=folium.Icon(
                    color=lm.get("icon_color", "blue"),
                    icon=lm.get("icon", "info-sign"),
                ),
            ).add_to(lg)
        lg.add_to(folium_map)

    # ═══════════════════════════════════════════════════════
    # EXPORT GEOTIFF
    # ═══════════════════════════════════════════════════════
    def export_geotiff(
        self,
        output: str | Path = "output/morilles_probability.tif",
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        score: np.ndarray = self._final_score.copy()
        nodata_mask: np.ndarray = np.asarray(~np.isfinite(score))
        score[nodata_mask] = _NODATA

        transform = from_bounds(
            self._xmin_l93, self._ymin_l93,
            self._xmax_l93, self._ymax_l93,
            self.grid.nx, self.grid.ny,
        )

        try:
            with rasterio.open(
                str(output), "w",
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
                    DESCRIPTION="Cartomorilles — probabilité morilles",
                    CELL_SIZE=str(config.CELL_SIZE),
                    CRS="EPSG:2154",
                    NODATA=str(_NODATA),
                )
            logger.info("✅ GeoTIFF : %s", output)
        except OSError:
            logger.exception("Impossible d'écrire le GeoTIFF")
            raise

        return output

    # ═══════════════════════════════════════════════════════
    # EXPORT GEOPACKAGE
    # ═══════════════════════════════════════════════════════
    def export_gpkg_grid(
        self,
        output: str | Path = "output/morilles_grid.gpkg",
        threshold: float = 0.15,
    ) -> Path:
        """Export GeoPackage vectorisé (numpy batch au lieu de double boucle)."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Export GeoPackage (seuil=%.2f)…", threshold)

        score: np.ndarray = self._final_score
        prob_classes: np.ndarray = self._prob_classes
        elim_mask: np.ndarray = self._elim_mask
        half = config.CELL_SIZE / 2.0
        labels = list(config.PROBABILITY_LABELS)

        # ── Indices vectorisés des cellules au-dessus du seuil ──
        above = np.isfinite(score) & (score >= threshold)
        iy_arr, ix_arr = np.where(above)

        if iy_arr.size == 0:
            logger.warning("Aucune cellule au-dessus du seuil %.2f", threshold)
            return output

        n = iy_arr.size
        logger.info("  %d cellules à exporter", n)

        # Coordonnées centres (vectorisé)
        cx_arr = np.asarray(self.grid.x_coords)[ix_arr]
        cy_arr = np.asarray(self.grid.y_coords)[iy_arr]

        # Géométries batch (list comp sur arrays, pas double boucle)
        geometries = [
            shapely_box(cx - half, cy - half, cx + half, cy + half)
            for cx, cy in zip(cx_arr.tolist(), cy_arr.tolist())
        ]

        # ── Colonnes principales ──
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

        # ── Altitude / pente (vectorisé) ──
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

        # ── Confiance (scalaires → broadcast pandas) ──
        conf_dict: dict[str, Any] = getattr(self.grid, "score_confidence", {})
        if isinstance(conf_dict, dict):
            for crit, conf_val in conf_dict.items():
                if isinstance(conf_val, (int, float)):
                    data[f"conf_{crit}"] = round(float(conf_val), 2)

        # ── Scores par critère (vectorisé) ──
        scores_dict: dict[str, Any] = getattr(self.grid, "scores", {})
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
                output,
                f"{len(gdf):,}",
            )
        except OSError:
            logger.exception("Impossible d'écrire le GeoPackage")
            raise

        return output

# ═══════════════════════════════════════════════════════════
# Landmarks par défaut
# ═══════════════════════════════════════════════════════════
def _default_landmarks() -> list[dict[str, Any]]:
    return [
        {
            "name": "Pont de l'Oulle", "lat": 45.24713, "lon": 5.69889,
            "info": "Entrée gorges Vence, ~265m",
            "icon_color": "red", "icon": "flag",
        },
        {
            "name": "Cascade des Prises", "lat": 45.2454, "lon": 5.69631,
            "info": "Cascade gorges Vence, ~350m",
            "icon_color": "red", "icon": "tint",
        },
        {
            "name": "Champy", "lat": 45.24036, "lon": 5.69272,
            "info": "Hameau, ~250m",
            "icon_color": "blue", "icon": "home",
        },
        {
            "name": "Saint-Égrève centre", "lat": 45.2325, "lon": 5.6790,
            "info": "Mairie, ~210m",
            "icon_color": "blue", "icon": "info-sign",
        },
        {
            "name": "🍄 Ripisylve Vence (bas)",
            "lat": 45.2442, "lon": 5.69375,
            "info": "Frênes/aulnes, sol alluvial",
            "icon_color": "green", "icon": "leaf",
        },
        {
            "name": "Le Néron", "lat": 45.23731, "lon": 5.71002,
            "info": "Sommet 1299m",
            "icon_color": "gray", "icon": "triangle-top",
        },    
        # ── Contrôles positifs (morilles attendues) ──────────────────
    {"name": "Champy – châtaigneraie",
     "lat": 45.24308, "lon": 5.69736, "expected": 0.55,
     "obs": "châtaignier favorable (M. elata), sol frais versant"},

    {"name": "Terrasse plan d'eau + conifères",
     "lat": 45.24087, "lon": 5.69484, "expected": 0.50,
     "obs": "meilleur spot trouvé, M. elata possible"},

    # ── Contrôles négatifs locaux (même zone, micro-habitat) ────
    {"name": "Berge Vence – trop humide",
     "lat": 45.24588, "lon": 5.69744, "expected": 0.05,
     "obs": "lierre dense, scolopendres — LIMITATION micro-habitat",
     "tolerance": 0.50},                      # ← tolérance élargie
    {"name": "Ravine au-dessus Vence",
     "lat": 45.24651, "lon": 5.70052, "expected": 0.05,
     "obs": "encaissé humide — LIMITATION micro-habitat",
     "tolerance": 0.50},
    {"name": "Mi-pente Néron chênaie+buis",
     "lat": 45.24418, "lon": 5.69886, "expected": 0.10,
     "obs": "trop sec thermophile — LIMITATION espèces indicatrices",
     "tolerance": 0.50},
    {"name": "Robiniers + hêtres secteur Champy",
     "lat": 45.24212, "lon": 5.69681, "expected": 0.15,
     "obs": "sol perturbé — LIMITATION sans détection invasives",
     "tolerance": 0.45},

    # ── Contrôle positif éloigné (diversité géo) ────────────────
    {"name": "Vouillants forêt calcaire 350m",
     "lat": 45.18824, "lon": 5.66543, "expected": 0.70,
     "obs": "forêt calcaire optimale, altitude idéale"},

    ]