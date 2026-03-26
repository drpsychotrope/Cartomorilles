"""
scoring.py — Modèle multicritère pondéré pour Cartomorilles.

Calcul du score de probabilité de présence de morilles :
- Somme pondérée NaN-safe des critères normalisés [0, 1]
- Pénalité de couverture (cellules avec critères manquants)
- Facteurs éliminatoires multi-couche (urbain, eau, espèces, géologie, pente, altitude)
- Transition douce autour des zones éliminées
- Lissage spatial par convolution normalisée (respecte les frontières)
- Classification en niveaux de probabilité
- Détection de hotspots enrichis (essence, géologie, compacité, confiance)

v2.3.0 — Fix #13 : _raw_tree_species pour masque éliminatoire espèces
         Fix #23 : pénalité couverture NaN-safe
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label,
    zoom,
)

import config
from config import TWI_WATERLOG

if TYPE_CHECKING:
    from grid_builder import GridBuilder

logger = logging.getLogger("cartomorilles.scoring")

__all__ = ["MorilleScoring"]

# ── Paramètres avec fallback sur config ────────────────────────────

_PROB_THRESHOLDS: list[float] = list(
    getattr(config, "PROBABILITY_THRESHOLDS", [0.15, 0.30, 0.45, 0.60, 0.75])
)
_PROB_LABELS: list[str] = list(
    getattr(
        config,
        "PROBABILITY_LABELS",
        ["Nul", "Très faible", "Faible", "Moyen", "Bon", "Excellent"],
    )
)

_ALT_ELIM_MIN: float = getattr(config, "ALT_ELIMINATORY_MIN", 100.0)
_ALT_ELIM_MAX: float = getattr(config, "ALT_ELIMINATORY_MAX", 1400.0)
_SOFT_ELIM_BUFFER_M: float = getattr(config, "SOFT_ELIM_BUFFER", 15.0)
_MIN_CLUSTER_SIZE: int = getattr(config, "MIN_CLUSTER_SIZE", 4)
_HOTSPOT_CLOSING_RADIUS: int = getattr(config, "HOTSPOT_CLOSING_RADIUS", 1)

# Fix #23 — Pénalité de couverture pour cellules à critères manquants
# Cellule complète (10/10 critères) : × 1.0
# Cellule à 50% de couverture       : × 0.75
_COVERAGE_PENALTY_FLOOR: float = 0.5

# Structure 8-connexité réutilisée
_STRUCT_8CONN: np.ndarray = np.asarray(generate_binary_structure(2, 2))


# ═══════════════════════════════════════════════════════════════════
class MorilleScoring:
    """Modèle de scoring multicritère pour la probabilité de morilles."""

    # ── Constructeur ──────────────────────────────────────────────

    def __init__(
        self,
        grid_builder: GridBuilder,
        species: str = "esculenta",
    ) -> None:
        self.grid: GridBuilder = grid_builder
        # Copie superficielle — ne pas aliaser le dict original
        self.scores: dict[str, np.ndarray] = dict(grid_builder.scores)
        self.species: str = species

        # Résultats — initialisés à None
        self.final_score: np.ndarray | None = None
        self.elimination_mask: np.ndarray | None = None
        self.elimination_detail: dict[str, np.ndarray] = {}
        self.probability_classes: np.ndarray | None = None

        # État du pipeline
        self._step_weighted: bool = False
        self._step_eliminated: bool = False
        self._step_smoothed: bool = False
        self._step_classified: bool = False

        # Métadonnées pour le rapport
        self._criteria_used: list[str] = []
        self._criteria_missing: list[str] = []
        self._effective_weight: float = 0.0
        self._confidence_score: float | None = None

    # ── Garde-fou d'ordre ─────────────────────────────────────────

    def _require_step(self, flag: str, current_step: str) -> None:
        """Lève RuntimeError si une étape prérequise n'a pas été exécutée."""
        if not getattr(self, flag, False):
            raise RuntimeError(
                f"'{current_step}' nécessite une étape antérieure. "
                f"Ordre attendu : compute → eliminate → smooth → classify"
            )

    # ═══════════════════════════════════════════════════════════════
    # 1. SCORE PONDÉRÉ
    # ═══════════════════════════════════════════════════════════════

    def compute_weighted_score(self) -> MorilleScoring:
        """
        Calcule le score pondéré multicritère.

        Fix #23 : pénalité de couverture appliquée aux cellules dont
        certains critères sont NaN. Empêche les scores artificiellement
        élevés en bordure de couverture de données.
        """
        self.grid.validate_scores()

        shape: tuple[int, int] = next(iter(self.scores.values())).shape
        ny, nx = shape
        total = np.zeros(shape, dtype=np.float64)
        weight_sum = np.zeros(shape, dtype=np.float64)
        theoretical_weight = 0.0

        logger.info("Calcul du score pondéré [%s]", self.species)
        logger.info("-" * 62)

        for factor, weight in config.WEIGHTS.items():
            theoretical_weight += weight

            if factor not in self.scores:
                self._criteria_missing.append(factor)
                logger.warning(
                    "  %-22s | MANQUANT (w=%.2f ignoré)",
                    factor,
                    weight,
                )
                continue

            arr = self.scores[factor]

            # Redimensionnement de secours
            if arr.shape != shape:
                logger.warning(
                    "  %-22s | shape %s ≠ %s — zoom bilinéaire",
                    factor,
                    arr.shape,
                    shape,
                )
                arr = np.clip(
                    np.asarray(
                        zoom(arr, (ny / arr.shape[0], nx / arr.shape[1]), order=1)
                    ),
                    0.0,
                    1.0,
                )

            # Accumulation NaN-safe
            valid = np.isfinite(arr)
            total = np.where(valid, total + arr * weight, total)
            weight_sum = np.where(valid, weight_sum + weight, weight_sum)

            self._criteria_used.append(factor)

            v = arr[valid]
            logger.info(
                "  %-22s | w=%.2f | moy=%.3f | max=%.3f | NaN=%d",
                factor,
                weight,
                float(v.mean()) if v.size else 0.0,
                float(v.max()) if v.size else 0.0,
                int((~valid).sum()),
            )

        # Division par le cumul de poids effectif par cellule
        with np.errstate(invalid="ignore", divide="ignore"):
            self.final_score = np.where(
                weight_sum > 0,
                total / weight_sum,
                0.0,
            ).astype(np.float64)
        self.final_score = np.clip(self.final_score, 0.0, 1.0)

        # ── Fix #23 : Pénalité de couverture ──────────────────
        # Cellules avec critères NaN → score pénalisé
        # proportionnellement au ratio de poids manquants.
        if theoretical_weight > 0:
            coverage_ratio = np.where(
                weight_sum > 0,
                weight_sum / theoretical_weight,
                0.0,
            )
            # Pénalité douce : score × (floor + (1-floor) × coverage)
            # Cellule 10/10 → ×1.0 | Cellule 5/10 → ×0.75
            penalty = (
                _COVERAGE_PENALTY_FLOOR
                + (1.0 - _COVERAGE_PENALTY_FLOOR) * coverage_ratio
            )
            self.final_score *= penalty

            n_penalized = int((coverage_ratio < 0.999).sum())
            if n_penalized > 0:
                logger.info(
                    "  Pénalité couverture  | %d cellules pénalisées "
                    "(couverture < 100%%)",
                    n_penalized,
                )

        self.final_score = np.clip(self.final_score, 0.0, 1.0)

        # NoData → NaN
        nodata = getattr(self.grid, "nodata_mask", None)
        if isinstance(nodata, np.ndarray):
            self.final_score[nodata] = np.nan
            logger.info(
                "  NoData (DEM)          | %d cellules → NaN",
                int(nodata.sum()),
            )

        # Poids effectif moyen
        self._effective_weight = float(
            np.nanmean(weight_sum) if weight_sum.size else 0.0
        )

        logger.info("-" * 62)
        logger.info(
            "  SCORE FINAL           | moy=%.3f | max=%.3f | critères=%d/%d",
            float(np.nanmean(self.final_score)),
            float(np.nanmax(self.final_score)),
            len(self._criteria_used),
            len(self._criteria_used) + len(self._criteria_missing),
        )
        if self._criteria_missing:
            logger.warning(
                "  Poids effectif moyen : %.3f / %.3f théorique (%.0f%%)",
                self._effective_weight,
                theoretical_weight,
                (
                    self._effective_weight / theoretical_weight * 100
                    if theoretical_weight > 0
                    else 0
                ),
            )

        # Confiance globale pondérée
        self._compute_global_confidence()

        self._step_weighted = True
        return self

    def _compute_global_confidence(self) -> None:
        """Confiance = moyenne pondérée des confidences par critère."""
        conf_dict = getattr(self.grid, "score_confidence", None)
        if not conf_dict:
            return
        num, den = 0.0, 0.0
        for factor in self._criteria_used:
            if factor in conf_dict:
                w = float(config.WEIGHTS.get(factor, 0.0))
                num += float(conf_dict[factor]) * w
                den += w
        if den > 0:
            self._confidence_score = num / den
            logger.info("  Confiance globale     | %.2f", self._confidence_score)

    # ═══════════════════════════════════════════════════════════════
    # 2. FACTEURS ÉLIMINATOIRES
    # ═══════════════════════════════════════════════════════════════

    def apply_eliminatory_factors(self) -> MorilleScoring:
        """
        Applique les facteurs éliminatoires avec masque multi-couche.
        """
        self._require_step("_step_weighted", "apply_eliminatory_factors")
        assert self.final_score is not None, "final_score is None after weighted step"

        ny, nx = self.final_score.shape
        combined = np.zeros((ny, nx), dtype=bool)
        detail: dict[str, np.ndarray] = {}

        logger.info("Facteurs éliminatoires :")

        # ── Urbain ──
        urban = getattr(self.grid, "urban_mask", None)
        if isinstance(urban, np.ndarray) and urban.shape == (ny, nx):
            detail["urban"] = urban.copy()
            combined |= urban
            logger.info("  Urbain           : %8d cellules", int(urban.sum()))

        # ── Plans d'eau ──
        water = getattr(self.grid, "water_mask", None)
        if isinstance(water, np.ndarray) and water.shape == (ny, nx):
            detail["water"] = water.copy()
            combined |= water
            logger.info("  Plans d'eau      : %8d cellules", int(water.sum()))

        # ── NoData ──
        nodata = getattr(self.grid, "nodata_mask", None)
        if isinstance(nodata, np.ndarray) and nodata.shape == (ny, nx):
            detail["nodata"] = nodata.copy()
            combined |= nodata
            logger.info("  NoData (DEM)     : %8d cellules", int(nodata.sum()))

        # ── Géologie éliminatoire ──
        geo_mask = self._build_eliminatory_geology_mask(ny, nx)
        if geo_mask is not None:
            detail["geology"] = geo_mask
            combined |= geo_mask
            logger.info(
                "  Géologie élim.   : %8d cellules  (%s)",
                int(geo_mask.sum()),
                ", ".join(sorted(config.ELIMINATORY_GEOLOGY)),
            )

        # ── Pente ──
        slope = getattr(self.grid, "slope", None)
        slope_max = float(getattr(config, "SLOPE_MAX", 30.0))
        if isinstance(slope, np.ndarray) and slope.shape == (ny, nx):
            slope_elim = np.isfinite(slope) & (slope > slope_max)
            detail["slope"] = slope_elim
            combined |= slope_elim
            logger.info(
                "  Pente > %.0f°     : %8d cellules",
                slope_max,
                int(slope_elim.sum()),
            )

        # ── Altitude ──
        alt = getattr(self.grid, "altitude", None)
        if isinstance(alt, np.ndarray) and alt.shape == (ny, nx):
            finite = np.isfinite(alt)
            alt_elim = finite & ((alt < _ALT_ELIM_MIN) | (alt > _ALT_ELIM_MAX))
            detail["altitude"] = alt_elim
            combined |= alt_elim
            logger.info(
                "  Altitude hors [%.0f–%.0f]m : %8d cellules",
                _ALT_ELIM_MIN,
                _ALT_ELIM_MAX,
                int(alt_elim.sum()),
            )

        # ── TWI engorgement → éliminatoire ──
        twi_arr: np.ndarray | None = getattr(self.grid, "_twi", None)  # P4
        if isinstance(twi_arr, np.ndarray) and twi_arr.shape == (ny, nx):
            twi_elim = twi_arr > TWI_WATERLOG
            detail["twi"] = twi_elim
            combined |= twi_elim
            logger.info(
                "  TWI engorgé (>%.0f) : %8d cellules",
                TWI_WATERLOG,
                int(twi_elim.sum()),
            )
            
        # ── Proximité urbaine éliminatoire ──
        dist_urban = getattr(self.grid, "dist_urban_grid", None)
        if isinstance(dist_urban, np.ndarray) and dist_urban.shape == (ny, nx):
            _existing_urban = detail.get(
                "urban", np.zeros((ny, nx), dtype=bool)
            )
            urban_prox_elim = (
                np.isfinite(dist_urban)
                & (dist_urban < config.URBAN_DIST_ELIMINATORY)
                & ~_existing_urban
            )
            detail["urban_proximity"] = urban_prox_elim
            combined |= urban_prox_elim
            logger.info(
                "  Proximité urb. (<%.0fm) : %8d cellules",
                config.URBAN_DIST_ELIMINATORY,
                int(urban_prox_elim.sum()),
            )

        # ── Résumé ──
        total_elim = int(combined.sum())
        total_cells = self.final_score.size
        pct = total_elim / total_cells * 100 if total_cells > 0 else 0.0
        logger.info(
            "  TOTAL éliminé    : %8d / %d (%.1f%%)",
            total_elim,
            total_cells,
            pct,
        )

        # Appliquer l'élimination dure
        self.final_score[combined] = 0.0

        # Transition douce autour des zones éliminées
        self._apply_soft_transition(combined)

        self.elimination_mask = combined
        self.elimination_detail = detail

        self._step_eliminated = True
        return self

    # ── Helpers élimination ───────────────────────────────────────

    def _build_eliminatory_species_mask(
        self,
        ny: int,
        nx: int,
    ) -> np.ndarray | None:
        """
        Construit un masque booléen des cellules occupées par une
        essence éliminatoire (châtaignier, etc.).

        Fix #13 v2.3.0 : utilise _raw_tree_species (score PRÉ-modulation
        landcover) pour éviter les faux positifs sur les cellules farmland
        dont le score modulé (0.05 × 0.10 = 0.005) passe sous le seuil.

        Ordre de priorité :
          1. Masque rasterisé explicite (grid_builder.eliminatory_species_mask)
          2. Score brut pré-landcover (_raw_tree_species) → score = 0.0
          3. Fallback : score modulé actuel < seuil (legacy)
        """
        # ── Priorité 1 : masque explicite rasterisé (Fix #17) ──
        mask = getattr(self.grid, "eliminatory_species_mask", None)
        if isinstance(mask, np.ndarray) and mask.shape == (ny, nx):
            return mask

        # ── Priorité 2 : score brut pré-landcover (Fix #13) ──
        raw = getattr(self.grid, "_raw_tree_species", None)
        if isinstance(raw, np.ndarray) and raw.shape == (ny, nx):
            # Éliminatoire = score brut exactement 0.0            
            # (châtaignier, etc. — score 0.0 dans config.TREE_SCORES)
            result = np.isfinite(raw) & (raw <= 0.0)
            if result.any():
                return result

        # ── Priorité 3 : fallback legacy sur score modulé ──
        if "tree_species" in self.scores:
            arr = self.scores["tree_species"]
            if arr.shape == (ny, nx):
                fill = float(getattr(config, "FILL_NO_FOREST", 0.05))
                result = arr < (fill * 0.5)
                if result.any():
                    return result

        return None

    def _build_eliminatory_geology_mask(
        self,
        ny: int,
        nx: int,
    ) -> np.ndarray | None:
        """
        Construit un masque booléen des cellules sur géologie éliminatoire.

        Ordre de priorité :
          1. Masque rasterisé explicite (grid_builder.eliminatory_geology_mask)
          2. Score géologie = 0.0 dans self.scores
        """
        # ── Priorité 1 : masque explicite rasterisé (Fix #17) ──
        mask = getattr(self.grid, "eliminatory_geology_mask", None)
        if isinstance(mask, np.ndarray) and mask.shape == (ny, nx):
            return mask

        # ── Priorité 2 : fallback score ──
        if "geology" in self.scores:
            arr = self.scores["geology"]
            if arr.shape == (ny, nx):
                result = np.isfinite(arr) & (arr <= 0.0)
                if result.any():
                    return result

        return None

    def _apply_soft_transition(self, hard_mask: np.ndarray) -> None:
        """
        Applique une transition douce autour des zones éliminées.
        """
        assert self.final_score is not None, "final_score is None in soft transition"

        if _SOFT_ELIM_BUFFER_M <= 0:
            return

        cell_size = float(getattr(config, "CELL_SIZE", 5.0))
        buffer_px = max(1, int(round(_SOFT_ELIM_BUFFER_M / cell_size)))

        dist_from_elim: np.ndarray = np.asarray(
            distance_transform_edt(~hard_mask)
        ).astype(np.float64)

        in_buffer = (dist_from_elim > 0) & (dist_from_elim <= buffer_px)
        if not in_buffer.any():
            return

        fade = np.clip(dist_from_elim / buffer_px, 0.0, 1.0)
        self.final_score[in_buffer] *= fade[in_buffer]

        logger.debug(
            "  Transition douce : %d cellules atténuées (buffer=%.0fm)",
            int(in_buffer.sum()),
            _SOFT_ELIM_BUFFER_M,
        )

    # ═══════════════════════════════════════════════════════════════
    # 3. LISSAGE SPATIAL
    # ═══════════════════════════════════════════════════════════════

    def apply_spatial_smoothing(self, sigma: float = 2.0) -> MorilleScoring:
        """
        Lissage spatial par convolution normalisée.
        """
        self._require_step("_step_eliminated", "apply_spatial_smoothing")
        assert self.final_score is not None, "final_score is None before smoothing"

        if sigma <= 0:
            logger.info("Lissage désactivé (sigma=0)")
            self._step_smoothed = True
            return self

        # Masque des cellules valides (non éliminées, non NaN)
        elim = (
            self.elimination_mask
            if self.elimination_mask is not None
            else np.zeros_like(self.final_score, dtype=bool)
        )
        nan_mask = ~np.isfinite(self.final_score)
        invalid = elim | nan_mask
        valid = ~invalid

        # Convolution normalisée
        score_valid = np.where(valid, self.final_score, 0.0)
        weights = valid.astype(np.float64)

        num: np.ndarray = np.asarray(
            gaussian_filter(score_valid, sigma=sigma, mode="nearest")
        )
        den: np.ndarray = np.asarray(
            gaussian_filter(weights, sigma=sigma, mode="nearest")
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(den > 1e-10, num / den, 0.0)

        smoothed = np.clip(smoothed, 0.0, 1.0)

        # Réappliquer l'élimination
        smoothed[invalid] = 0.0

        # Rétablir NaN sur nodata
        if nan_mask.any():
            smoothed[nan_mask] = np.nan

        self.final_score = smoothed

        logger.info(
            "Lissage spatial (sigma=%.1f) — convolution normalisée, "
            "%d cellules masquées préservées",
            sigma,
            int(invalid.sum()),
        )

        self._step_smoothed = True
        return self

    # ═══════════════════════════════════════════════════════════════
    # 4. CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════

    def classify_probability(self) -> MorilleScoring:
        """
        Classification en niveaux de probabilité.
        """
        self._require_step("_step_smoothed", "classify_probability")
        assert self.final_score is not None, "final_score is None before classify"

        safe_score = np.where(
            np.isfinite(self.final_score),
            self.final_score,
            -1.0,
        )
        classes = np.digitize(safe_score, _PROB_THRESHOLDS).astype(np.int8)

        # NaN → classe -1
        classes[~np.isfinite(self.final_score)] = -1

        self.probability_classes = classes

        logger.info("Répartition des probabilités :")
        n_valid = int(np.isfinite(self.final_score).sum())
        for i, lbl in enumerate(_PROB_LABELS):
            n = int((classes == i).sum())
            pct = n / n_valid * 100 if n_valid > 0 else 0.0
            bar = "█" * int(pct / 2)
            logger.info(
                "  %-14s : %8d cellules (%5.1f%%) %s", lbl, n, pct, bar
            )

        n_nan = int((classes == -1).sum())
        if n_nan > 0:
            logger.info("  %-14s : %8d cellules", "NoData", n_nan)

        self._step_classified = True
        return self

    # ═══════════════════════════════════════════════════════════════
    # 5. HOTSPOTS
    # ═══════════════════════════════════════════════════════════════

    def get_hotspots(
        self,
        threshold: float = 0.65,
        min_cluster_size: int | None = None,
        max_hotspots: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Identifie les clusters de forte probabilité.
        """
        self._require_step("_step_weighted", "get_hotspots")
        assert self.final_score is not None, "final_score is None for hotspots"

        if min_cluster_size is None:
            min_cluster_size = _MIN_CLUSTER_SIZE

        cell_size: float = float(getattr(config, "CELL_SIZE", 5.0))
        cell_area: float = cell_size**2

        # Masque des cellules chaudes
        safe = np.where(
            np.isfinite(self.final_score), self.final_score, 0.0
        )
        hot_mask = safe >= threshold

        # Closing morphologique pour fusionner clusters proches
        if _HOTSPOT_CLOSING_RADIUS >= 1:
            hot_mask = np.asarray(
                binary_closing(
                    hot_mask,
                    structure=_STRUCT_8CONN,
                    iterations=_HOTSPOT_CLOSING_RADIUS,
                )
            )

        # Labellisation 8-connexité
        _label_result = label(hot_mask, structure=_STRUCT_8CONN)
        labeled: np.ndarray = np.asarray(_label_result[0])  # type: ignore[index]
        n_clusters: int = int(_label_result[1])  # type: ignore[index]

        hotspots: list[dict[str, Any]] = []
        for cid in range(1, n_clusters + 1):
            cm = labeled == cid
            n_cells = int(cm.sum())
            if n_cells < min_cluster_size:
                continue

            ys, xs = np.where(cm)

            # Centroïde en coordonnées L93 continues
            transform = getattr(self.grid, "transform", None)
            if transform is not None:
                x_l93 = float(
                    transform[2] + (xs.mean() + 0.5) * transform[0]
                )
                y_l93 = float(
                    transform[5] + (ys.mean() + 0.5) * transform[4]
                )
            else:
                x_coords = getattr(self.grid, "x_coords", None)
                y_coords = getattr(self.grid, "y_coords", None)
                x_l93 = (
                    float(x_coords[int(xs.mean())])
                    if isinstance(x_coords, np.ndarray)
                    else 0.0
                )
                y_l93 = (
                    float(y_coords[int(ys.mean())])
                    if isinstance(y_coords, np.ndarray)
                    else 0.0
                )

            # Score
            cluster_scores = self.final_score[cm]
            mean_score = float(np.nanmean(cluster_scores))
            max_score = float(np.nanmax(cluster_scores))

            # Altitude
            alt = getattr(self.grid, "altitude", None)
            mean_alt: float | None = (
                float(np.nanmean(alt[cm]))
                if isinstance(alt, np.ndarray) and alt.shape == cm.shape
                else None
            )

            # Pente
            slope = getattr(self.grid, "slope", None)
            mean_slope: float | None = (
                float(np.nanmean(slope[cm]))
                if isinstance(slope, np.ndarray) and slope.shape == cm.shape
                else None
            )

            # Essence dominante
            dominant_species = self._get_dominant_value_in_cluster(
                cm,
                "tree_species",
                "essence_canonical",
            )

            # Géologie dominante
            dominant_geology = self._get_dominant_value_in_cluster(
                cm,
                "geology",
                "geology_canonical",
            )

            # Compacité
            area = n_cells * cell_area
            perimeter = self._estimate_perimeter(cm, cell_size)
            compactness = (
                (4.0 * np.pi * area / (perimeter**2))
                if perimeter > 0
                else 0.0
            )

            # Confiance
            conf_dict = getattr(self.grid, "score_confidence", None)
            confidence: float | None = None
            if isinstance(conf_dict, dict):
                conf_vals = [
                    float(conf_dict[k])
                    for k in self._criteria_used
                    if k in conf_dict
                ]
                confidence = (
                    float(np.mean(conf_vals)) if conf_vals else None
                )

            hotspots.append(
                {
                    "id": cid,
                    "x_l93": float(x_l93),
                    "y_l93": float(y_l93),
                    "n_cells": n_cells,
                    "size_m2": area,
                    "mean_score": round(mean_score, 4),
                    "max_score": round(max_score, 4),
                    "altitude": (
                        round(mean_alt, 1) if mean_alt is not None else None
                    ),
                    "mean_slope": (
                        round(mean_slope, 1)
                        if mean_slope is not None
                        else None
                    ),
                    "dominant_species": dominant_species,
                    "dominant_geology": dominant_geology,
                    "compactness": round(float(compactness), 3),
                    "confidence": (
                        round(confidence, 2)
                        if confidence is not None
                        else None
                    ),
                }
            )

        hotspots.sort(key=lambda h: h["mean_score"], reverse=True)

        if max_hotspots is not None:
            hotspots = hotspots[:max_hotspots]

        # Log
        logger.info(
            "%d hotspots détectés (seuil=%.2f, min_cluster=%d) :",
            len(hotspots),
            threshold,
            min_cluster_size,
        )
        for h in hotspots[:15]:
            parts = [
                f"score={h['mean_score']:.2f}",
                f"{h['size_m2']:.0f}m²",
            ]
            if h["altitude"] is not None:
                parts.append(f"alt={h['altitude']:.0f}m")
            if h["mean_slope"] is not None:
                parts.append(f"pente={h['mean_slope']:.0f}°")
            if h["dominant_species"]:
                parts.append(f"essence={h['dominant_species']}")
            if h["dominant_geology"]:
                parts.append(f"géol={h['dominant_geology']}")
            parts.append(f"compact={h['compactness']:.2f}")
            logger.info("  #%-3d | %s", h["id"], " | ".join(parts))

        return hotspots

    # ── Helpers hotspots ──────────────────────────────────────────

    def _get_dominant_value_in_cluster(
        self,
        cluster_mask: np.ndarray,
        score_key: str,
        label_attr: str,
    ) -> str | None:
        """
        Retrouve la valeur dominante (essence ou géologie) dans un cluster.

        Fix #26 : utilise un raster int-codé + lookup dict.
        L'ancien code cherchait _raster_essence_canonical comme raster
        de strings qui n'existait jamais → fallback systématique.
        """
        # ── Méthode 1 : raster int-codé + lookup (Fix #26) ──
        raster_attr = f"_raster_{label_attr}"
        lookup_attr = f"_int_to_{label_attr}"

        raster = getattr(self.grid, raster_attr, None)
        lookup = getattr(self.grid, lookup_attr, None)

        if (
            isinstance(raster, np.ndarray)
            and raster.shape == cluster_mask.shape
            and isinstance(lookup, dict)
        ):
            values = raster[cluster_mask]
            values = values[values > 0]  # 0 = fill / hors couverture
            if values.size > 0:
                unique, counts = np.unique(values, return_counts=True)
                dominant_int = int(unique[counts.argmax()])
                result = lookup.get(dominant_int)
                if result is not None:
                    return str(result)

        # ── Méthode 2 : reverse lookup approximatif (fallback) ──
        if score_key in self.scores:
            arr = self.scores[score_key]
            if arr.shape == cluster_mask.shape:
                mean_val = float(np.nanmean(arr[cluster_mask]))
                return self._reverse_lookup_score(score_key, mean_val)

        return None

    @staticmethod
    def _reverse_lookup_score(score_key: str, value: float) -> str | None:
        """Correspondance inverse approximative score → label."""
        if score_key == "tree_species":
            best_name: str | None = None
            best_diff = 999.0
            for name, sc in config.TREE_SCORES.items():
                diff = abs(float(sc) - value)
                if diff < best_diff:
                    best_diff = diff
                    best_name = str(name)
            return best_name if best_diff < 0.05 else f"score~{value:.2f}"

        if score_key == "geology":
            best_name_g: str | None = None
            best_diff_g = 999.0
            for name, sc in config.GEOLOGY_SCORES.items():
                diff = abs(float(sc) - value)
                if diff < best_diff_g:
                    best_diff_g = diff
                    best_name_g = str(name)
            return best_name_g if best_diff_g < 0.05 else f"score~{value:.2f}"

        return None

    @staticmethod
    def _estimate_perimeter(mask: np.ndarray, cell_size: float) -> float:
        """
        Estime le périmètre d'un cluster en mètres.
        """
        eroded: np.ndarray = np.asarray(
            binary_erosion(mask, structure=_STRUCT_8CONN)
        )
        border = mask & ~eroded
        return float(border.sum()) * cell_size

    # ═══════════════════════════════════════════════════════════════
    # 6. RAPPORT / MÉTADONNÉES
    # ═══════════════════════════════════════════════════════════════

    def get_elimination_stats(self) -> dict[str, int]:
        """
        Retourne les statistiques d'élimination par facteur.
        """
        stats: dict[str, int] = {}
        for key, mask in self.elimination_detail.items():
            stats[key] = int(mask.sum())
        if self.elimination_mask is not None:
            stats["total"] = int(self.elimination_mask.sum())
        if self.final_score is not None:
            stats["total_cells"] = int(self.final_score.size)
        return stats

    def get_model_metadata(self) -> dict[str, Any]:
        """
        Retourne les métadonnées du modèle pour le rapport JSON.
        """
        meta: dict[str, Any] = {
            "species": self.species,
            "criteria_used": self._criteria_used,
            "criteria_missing": self._criteria_missing,
            "effective_weight_mean": round(self._effective_weight, 4),
            "theoretical_weight": round(
                sum(float(v) for v in config.WEIGHTS.values()),
                4,
            ),
            "confidence_score": (
                round(self._confidence_score, 3)
                if self._confidence_score is not None
                else None
            ),
            "elimination_stats": self.get_elimination_stats(),
            "pipeline_steps": {
                "weighted": self._step_weighted,
                "eliminated": self._step_eliminated,
                "smoothed": self._step_smoothed,
                "classified": self._step_classified,
            },
        }

        # Stats du score final
        if self.final_score is not None:
            finite = self.final_score[np.isfinite(self.final_score)]
            if finite.size > 0:
                meta["score_stats"] = {
                    "mean": round(float(finite.mean()), 4),
                    "std": round(float(finite.std()), 4),
                    "min": round(float(finite.min()), 4),
                    "max": round(float(finite.max()), 4),
                    "median": round(float(np.median(finite)), 4),
                    "pct_90": round(float(np.percentile(finite, 90)), 4),
                    "pct_95": round(float(np.percentile(finite, 95)), 4),
                    "pct_99": round(float(np.percentile(finite, 99)), 4),
                }

        # Distribution des classes
        if self.probability_classes is not None:
            dist: dict[str, int] = {}
            for i, lbl in enumerate(_PROB_LABELS):
                dist[lbl] = int((self.probability_classes == i).sum())
            dist["NoData"] = int((self.probability_classes == -1).sum())
            meta["class_distribution"] = dist

        return meta
    
    def get_twi_display_data(self) -> dict[str, Any]:
        """Données TWI pour affichage cartographique.

        Returns
        -------
        dict avec clés :
            - ``raw``: np.ndarray | None — valeurs TWI brutes
            - ``score``: np.ndarray | None — score TWI [0,1]
            - ``waterlog_mask``: np.ndarray | None — masque engorgement (bool)
            - ``has_data``: bool
        """
        builder = self.grid
        twi_raw = builder.get_twi_raw() if hasattr(builder, "get_twi_raw") else None
        twi_score = builder.scores.get("twi")

        waterlog_mask: np.ndarray | None = None
        if twi_raw is not None:
            valid = np.isfinite(twi_raw)
            waterlog_mask = valid & (np.asarray(twi_raw) > TWI_WATERLOG)

        return {
            "raw": twi_raw,
            "score": twi_score,
            "waterlog_mask": waterlog_mask,
            "has_data": twi_raw is not None,
        }