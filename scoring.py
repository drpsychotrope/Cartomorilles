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
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    zoom,
)
import _accel
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

# ── Malus monotonie terrain (v2.3.6) ────────────────────────────
# Pénalise les zones où altitude, pente et TWI sont simultanément
# dans leur zone optimale sur de grandes surfaces — typiquement
# plaine alluviale péri-urbaine, pas micro-habitat forestier.
_MONOTONY_RADIUS: int = 50          # cellules (~250m à 5m/cell)
_MONOTONY_FLOOR: float = 0.85       # plancher du malus (15% max)
_MONOTONY_SLOPE_THRESHOLD: float = 8.0   # pente < seuil = "plat"
_MONOTONY_ALT_STD_THRESHOLD: float = 15.0  # écart-type altitude < seuil = "homogène"

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

        v2.4.0 : float32 strict — évite promotion float64 sur 73.6M cells.
        Fix #23 : pénalité de couverture NaN-safe floor=0.5 (D10).
        """
        self.grid.validate_scores()

        shape: tuple[int, int] = next(iter(self.scores.values())).shape
        ny, nx = shape

        # ⑤ Float32 strict — 4 bytes/cell au lieu de 8
        total = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        theoretical_weight = np.float32(0.0)

        logger.info("Calcul du score pondéré [%s]", self.species)
        logger.info("-" * 62)

        for factor, weight in config.WEIGHTS.items():
            w32 = np.float32(weight)
            theoretical_weight += w32

            if factor not in self.scores:
                self._criteria_missing.append(factor)
                logger.warning(
                    "  %-22s | MANQUANT (w=%.2f ignoré)",
                    factor, weight,
                )
                continue

            arr = self.scores[factor]

            # Redimensionnement de secours
            if arr.shape != shape:
                logger.warning(
                    "  %-22s | shape %s ≠ %s — zoom bilinéaire",
                    factor, arr.shape, shape,
                )
                arr = np.clip(
                    np.asarray(
                        zoom(
                            arr,
                            (ny / arr.shape[0], nx / arr.shape[1]),
                            order=1,
                        ),
                    ),
                    0.0, 1.0,
                ).astype(np.float32)

            # Assurer float32
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)

            # Accumulation NaN-safe (float32)
            valid = np.isfinite(arr)
            total = np.where(valid, total + arr * w32, total)
            weight_sum = np.where(valid, weight_sum + w32, weight_sum)

            self._criteria_used.append(factor)

            v = arr[valid]
            logger.info(
                "  %-22s | w=%.2f | moy=%.3f | max=%.3f | NaN=%d",
                factor, weight,
                float(v.mean()) if v.size else 0.0,
                float(v.max()) if v.size else 0.0,
                int((~valid).sum()),
            )

        # Division par le cumul de poids effectif par cellule
        with np.errstate(invalid="ignore", divide="ignore"):
            final = np.where(
                weight_sum > 0,
                total / weight_sum,
                np.float32(0.0),
            ).astype(np.float32)
        np.clip(final, 0.0, 1.0, out=final)

        # ── Fix #23 / D10 : Pénalité de couverture (floor=0.5) ──
        if theoretical_weight > 0:
            coverage_ratio = np.where(
                weight_sum > 0,
                weight_sum / theoretical_weight,
                np.float32(0.0),
            ).astype(np.float32)

            penalty = (
                np.float32(_COVERAGE_PENALTY_FLOOR)
                + np.float32(1.0 - _COVERAGE_PENALTY_FLOOR) * coverage_ratio
            )
            final *= penalty

            n_penalized = int(np.sum(coverage_ratio < 0.999))
            if n_penalized > 0:
                logger.info(
                    "  Pénalité couverture  | %d cellules pénalisées "
                    "(couverture < 100%%)",
                    n_penalized,
                )

        np.clip(final, 0.0, 1.0, out=final)

        # NoData → NaN
        nodata = getattr(self.grid, "nodata_mask", None)
        if isinstance(nodata, np.ndarray) and np.any(nodata):
            final[nodata] = np.nan
            logger.info(
                "  NoData (DEM)          | %d cellules → NaN",
                int(nodata.sum()),
            )

        self.final_score = final

        # Poids effectif moyen
        self._effective_weight = float(
            np.nanmean(weight_sum) if weight_sum.size else 0.0,
        )

        logger.info("-" * 62)
        logger.info(
            "  SCORE FINAL           | moy=%.3f | max=%.3f | critères=%d/%d",
            float(np.nanmean(final)),
            float(np.nanmax(final)),
            len(self._criteria_used),
            len(self._criteria_used) + len(self._criteria_missing),
        )
        if self._criteria_missing:
            logger.warning(
                "  Poids effectif moyen : %.3f / %.3f théorique (%.0f%%)",
                self._effective_weight,
                float(theoretical_weight),
                (
                    self._effective_weight / float(theoretical_weight) * 100
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
        geo_mask = self._build_eliminatory_geology_mask()
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
        twi_arr: np.ndarray | None = getattr(self.grid, "twi", None)  # P4
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

    # ------------------------------------------------------------------
    # Masques éliminatoires — int rasters vectorisés (D6)
    # ------------------------------------------------------------------

    def _build_eliminatory_species_mask(self) -> np.ndarray:
        """
        Masque éliminatoire essences — vectorisé via int raster (D6).

        Élimine les cellules couvertes par une essence à score 0.0.
        Cellules sans couverture forêt (code 0 = nodata) → non éliminées.
        Châtaignier score 0.80 → PAS éliminatoire (D3).
        """
        grid = self.grid
        int_raster = getattr(grid, "_tree_species_int_raster", None)
        lookup = getattr(grid, "_tree_score_lookup", None)

        if int_raster is None or lookup is None:
            logger.debug(
                "   eliminatory_species : pas d'int raster → skip",
            )
            return np.zeros((grid.ny, grid.nx), dtype=bool)

        int_raster = np.asarray(int_raster, dtype=np.int16)
        lookup = np.asarray(lookup, dtype=np.float32)

        # Codes éliminatoires = couverts (code > 0) avec score == 0.0
        elim_codes = np.where(lookup == 0.0)[0]
        elim_codes = elim_codes[elim_codes > 0]

        if len(elim_codes) == 0:
            logger.debug(
                "   eliminatory_species : aucun code éliminatoire",
            )
            return np.zeros((grid.ny, grid.nx), dtype=bool)

        mask = np.isin(int_raster, elim_codes)
        n = int(np.count_nonzero(mask))

        if n > 0:
            code_to_name = getattr(grid, "_tree_code_to_name", {})
            names = [
                code_to_name.get(int(c), f"code_{c}")
                for c in elim_codes
            ]
            logger.info(
                "🚫 Masque éliminatoire essences : %d cellules (%s)",
                n, ", ".join(names),
            )
        else:
            logger.debug("   eliminatory_species : 0 cellules éliminées")

        return mask

    def _build_eliminatory_geology_mask(self) -> np.ndarray:
        """
        Masque éliminatoire géologie — vectorisé via int raster (D6).

        Utilise _geology_eliminatory_codes (frozenset[int]) pré-calculés
        dans grid_builder.score_geology(). Catégories : granite, gneiss,
        siliceux (config.ELIMINATORY_GEOLOGY).
        """
        grid = self.grid
        int_raster = getattr(grid, "_geology_int_raster", None)
        elim_codes = getattr(
            grid, "_geology_eliminatory_codes", frozenset(),
        )

        if int_raster is None:
            logger.debug(
                "   eliminatory_geology : pas d'int raster → skip",
            )
            return np.zeros((grid.ny, grid.nx), dtype=bool)

        int_raster = np.asarray(int_raster, dtype=np.int16)

        if not elim_codes:
            logger.debug(
                "   eliminatory_geology : aucun code éliminatoire",
            )
            return np.zeros((grid.ny, grid.nx), dtype=bool)

        elim_arr = np.array(sorted(elim_codes), dtype=np.int16)
        mask = np.isin(int_raster, elim_arr)
        n = int(np.count_nonzero(mask))

        if n > 0:
            code_to_name = getattr(grid, "_geology_code_to_name", {})
            names = [
                code_to_name.get(int(c), f"code_{c}")
                for c in sorted(elim_codes)
            ]
            logger.info(
                "🚫 Masque éliminatoire géologie : %d cellules (%s)",
                n, ", ".join(names),
            )
        else:
            logger.debug("   eliminatory_geology : 0 cellules éliminées")

        return mask

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

    def apply_monotony_penalty(self) -> MorilleScoring:
        """
        Malus monotonie terrain — v2.3.6.

        Détecte les zones où pente, altitude et TWI sont uniformément
        optimaux sur un grand voisinage (~250m), signe de plaine
        alluviale plutôt que de micro-habitat forestier différencié.

        Le malus est multiplicatif : score × [_MONOTONY_FLOOR .. 1.0].
        Les zones éliminées et NaN sont préservées.
        """
        self._require_step("_step_eliminated", "apply_monotony_penalty")
        assert self.final_score is not None

        _slope = getattr(self.grid, "slope", None)
        _alt = getattr(self.grid, "altitude", None)
        if _slope is None or _alt is None:
            logger.debug("   Monotonie : slope/altitude indisponible, skip")
            self._step_monotony = True
            return self

        slope: np.ndarray = np.asarray(_slope)
        alt: np.ndarray = np.asarray(_alt)

        # ── Masque "plat" : pente < seuil ──
        flat_mask = (slope < _MONOTONY_SLOPE_THRESHOLD).astype(np.float32)

        # ── Fraction de cellules plates dans le voisinage ──
        flat_frac: np.ndarray = _accel.gaussian_filter(
            flat_mask, sigma=_MONOTONY_RADIUS / 2.0, mode="nearest",
        )

        # ── Écart-type local de l'altitude (proxy homogénéité) ──
        alt_f32 = np.where(np.isfinite(alt), alt, 0.0).astype(np.float32)
        alt_mean: np.ndarray = _accel.gaussian_filter(
            alt_f32, sigma=_MONOTONY_RADIUS / 2.0, mode="nearest",
        )
        alt_sq_mean: np.ndarray = _accel.gaussian_filter(
            alt_f32 ** 2, sigma=_MONOTONY_RADIUS / 2.0, mode="nearest",
        )

        alt_var = np.maximum(alt_sq_mean - alt_mean ** 2, 0.0)
        alt_std = np.sqrt(alt_var)

        # ── Score de monotonie [0, 1] ──
        # 1.0 = zone parfaitement plate et homogène
        # flat_frac élevée ET alt_std faible → monotone
        homogeneous = (
            alt_std < _MONOTONY_ALT_STD_THRESHOLD
        ).astype(np.float32)

        monotony = flat_frac * homogeneous
        monotony = np.clip(monotony, 0.0, 1.0)

        # ── Malus multiplicatif ──
        # monotony=1.0 → penalty = _MONOTONY_FLOOR (0.85)
        # monotony=0.0 → penalty = 1.0 (pas de pénalité)
        penalty = 1.0 - (1.0 - _MONOTONY_FLOOR) * monotony

        # Préserver élimination et NaN
        elim = (
            self.elimination_mask
            if self.elimination_mask is not None
            else np.zeros_like(self.final_score, dtype=bool)
        )
        nan_mask = ~np.isfinite(self.final_score)
        preserve = elim | nan_mask

        self.final_score = np.where(
            preserve,
            self.final_score,
            self.final_score * penalty,
        )

        n_penalized = int((penalty < 0.999).sum())
        mean_penalty = float(penalty[penalty < 0.999].mean()) if n_penalized > 0 else 1.0
        logger.info(
            "Malus monotonie terrain : %d cellules pénalisées"
            " (malus moyen=%.2f, floor=%.2f)",
            n_penalized,
            mean_penalty,
            _MONOTONY_FLOOR,
        )

        self._step_monotony = True
        return self

    def apply_calcdry_penalty(self) -> MorilleScoring:
        """
        Malus multiplicatif pour versants calcaires secs à feuillus collinéens.

        Cible les chênaies-buis sur calcaire en pente (>10°, <700m)
        qui scorent artificiellement haut grâce au terrain seul.
        Le malus est gradué par rugosité et TWI :
          - rugosité forte + TWI sec → ravine rocheuse → malus fort
          - rugosité faible + TWI normal → forêt dense → malus léger
        Les zones éliminées et NaN sont préservées.
        """
        self._require_step("_step_monotony", "apply_calcdry_penalty")

        grid = self.grid
        score = self.final_score
        assert score is not None

        # ── Accès aux grilles requises ──
        substrate = getattr(grid, "substrate_grid", None)
        if substrate is None or not isinstance(substrate, np.ndarray):
            logger.info("Malus calc_dry : pas de grille substrat — ignoré")
            return self

        altitude = getattr(grid, "altitude", None)
        slope = getattr(grid, "slope", None)
        roughness = getattr(grid, "roughness", None)
        twi_raw = grid.get_twi_raw()
        ft_grid = getattr(grid, "_forest_type_grid", None)

        if altitude is None or slope is None:
            logger.info("Malus calc_dry : altitude/pente indisponible — ignoré")
            return self

        _alt = np.asarray(altitude)
        _slope = np.asarray(slope)
        _sub = np.asarray(substrate)

        # ── Masque de base : calcaire sec + feuillus collinéens en pente ──
        # substrate==1 = calc_dry (défini dans species_enricher)
        # altitude < 700m = étage collinéen élargi (inclut chênaie pubescente)
        # slope > 10° = versant (le plat est déjà reclassé marly)
        target = (
            (_sub == 1)
            & (_alt < 700.0)
            & (_slope > 10.0)
        )

        # Restreindre aux feuillus si forest_type_grid disponible
        if ft_grid is not None and isinstance(ft_grid, np.ndarray):
            _ft = np.asarray(ft_grid)
            # 1=feuillus, 0=unknown (souvent feuillus dans ce contexte)
            target = target & ((_ft == 1) | (_ft == 0))

        # Exclure les cellules déjà éliminées
        if self.elimination_mask is not None:
            target = target & ~self.elimination_mask

        # Exclure NaN
        target = target & np.isfinite(score)

        n_target = int(target.sum())
        if n_target == 0:
            logger.info("Malus calc_dry : aucune cellule ciblée")
            return self

        # ── Calcul du malus gradué ──
        # Composante rugosité : rugosité élevée = ravine → malus fort
        roughness_factor = np.ones_like(score, dtype=np.float32)
        if roughness is not None and isinstance(roughness, np.ndarray):
            _rough = np.asarray(roughness)
            # Normaliser : 0°→0, 5°→0.5, 10°+→1.0
            roughness_factor = np.clip(_rough / 10.0, 0.0, 1.0)

        # Composante TWI : TWI sec = drainage excessif → malus fort
        twi_factor = np.ones_like(score, dtype=np.float32)
        if twi_raw is not None and isinstance(twi_raw, np.ndarray):
            _twi = np.asarray(twi_raw)
            # TWI < 4 = sec, TWI 4-8 = transition, TWI > 8 = humide
            # Normaliser inversé : sec (TWI=2)→1.0, optimal (TWI=8)→0.0
            twi_factor = np.clip((8.0 - _twi) / 6.0, 0.0, 1.0)

        # Combiner : moyenne pondérée (rugosité 40%, TWI 60%)
        # TWI plus important car l'humidité du sol est plus déterminante
        severity = 0.4 * roughness_factor + 0.6 * twi_factor

        # Malus : score × multiplier
        # severity=0 → multiplier=1.0 (pas de malus)
        # severity=0.5 → multiplier=0.75
        # severity=1.0 → multiplier=0.50
        _CALCDRY_FLOOR = 0.50
        _CALCDRY_MAX_PENALTY = 0.50  # multiplier min = 1.0 - 0.50 = 0.50

        multiplier = np.ones_like(score, dtype=np.float32)
        multiplier[target] = 1.0 - _CALCDRY_MAX_PENALTY * severity[target]
        multiplier = np.clip(multiplier, _CALCDRY_FLOOR, 1.0)

        # Appliquer
        score[target] *= multiplier[target]

        # ── Stats ──
        mean_mult = float(multiplier[target].mean())
        mean_sev = float(severity[target].mean())
        n_strong = int((multiplier[target] < 0.70).sum())
        n_moderate = int(((multiplier[target] >= 0.70) & (multiplier[target] < 0.90)).sum())
        n_light = int((multiplier[target] >= 0.90).sum())

        logger.info(
            "Malus calc_dry : %d cellules (sévérité moy=%.2f, mult moy=%.2f)",
            n_target, mean_sev, mean_mult,
        )
        logger.info(
            "  Fort (<0.70)=%d | Modéré (0.70-0.90)=%d | Léger (>0.90)=%d",
            n_strong, n_moderate, n_light,
        )

        self._step_calcdry = True
        return self

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

        num: np.ndarray = _accel.gaussian_filter(
            score_valid, sigma=sigma, mode="nearest",
        )
        den: np.ndarray = _accel.gaussian_filter(
            weights, sigma=sigma, mode="nearest",
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
        """Identifie les clusters de forte probabilité."""
        self._require_step("_step_weighted", "get_hotspots")
        assert self.final_score is not None, "final_score is None for hotspots"

        if min_cluster_size is None:
            min_cluster_size = _MIN_CLUSTER_SIZE

        cell_size: float = float(getattr(config, "CELL_SIZE", 5.0))
        cell_area: float = cell_size**2

        # Masque des cellules chaudes
        safe = np.where(
            np.isfinite(self.final_score), self.final_score, 0.0,
        )
        hot_mask = safe >= threshold

        # ── Connected components GPU/CPU ──
        labeled, n_clusters = _accel.connected_components(
            hot_mask,
            structure=_STRUCT_8CONN,
            closing_iterations=_HOTSPOT_CLOSING_RADIUS,
        )

        if n_clusters == 0:
            logger.info("0 hotspots détectés (seuil=%.2f)", threshold)
            return []

        # ── Stats vectorisées ──
        transform = getattr(self.grid, "transform", None)
        if transform is not None:
            transform_params = (
                transform.a, transform.b, transform.c,
                transform.d, transform.e, transform.f,
            )
        else:
            transform_params = (cell_size, 0.0, 0.0, 0.0, -cell_size, 0.0)

        alt = getattr(self.grid, "altitude", None)
        slp = getattr(self.grid, "slope", None)

        stats = _accel.vectorized_cluster_stats(
            labeled=labeled,
            n_clusters=n_clusters,
            final_score=self.final_score,
            transform_params=transform_params,
            cell_size=cell_size,
            min_cluster_size=min_cluster_size,
            altitude=alt if isinstance(alt, np.ndarray) else None,
            slope=slp if isinstance(slp, np.ndarray) else None,
        )

        n_valid = stats["n_valid"]
        if n_valid == 0:
            logger.info("0 hotspots après filtrage taille (seuil=%d)", min_cluster_size)
            return []

        valid_ids = stats["valid_ids"]
        stat_labeled = stats["labeled"]

        # ── Confiance globale ──
        conf_dict = getattr(self.grid, "score_confidence", None)
        confidence: float | None = None
        if isinstance(conf_dict, dict):
            conf_vals = [
                float(conf_dict[k])
                for k in self._criteria_used
                if k in conf_dict
            ]
            confidence = float(np.mean(conf_vals)) if conf_vals else None
        conf_rounded = round(confidence, 2) if confidence is not None else None

        # ── Dominant species/geology par cluster (boucle légère) ──
        # Réutilise un masque pré-alloué pour éviter 1312× np.zeros
        reuse_mask = np.empty(stat_labeled.shape, dtype=bool)

        hotspots: list[dict[str, Any]] = []
        for i in range(n_valid):
            cid = int(valid_ids[i])

            # Masque cluster réutilisé (O(cluster) vs O(N))
            np.equal(stat_labeled, cid, out=reuse_mask)

            dominant_species = self._get_dominant_value_in_cluster(
                reuse_mask, "tree_species", "essence_canonical",
            )
            dominant_geology = self._get_dominant_value_in_cluster(
                reuse_mask, "geology", "geology_canonical",
            )

            h_alt = stats["altitude"]
            h_slp = stats["slope"]

            hotspots.append({
                "id": cid,
                "x_l93": float(stats["x_l93"][i]),
                "y_l93": float(stats["y_l93"][i]),
                "n_cells": int(stats["counts"][i]),
                "size_m2": float(stats["area"][i]),
                "mean_score": round(float(stats["mean_score"][i]), 4),
                "max_score": round(float(stats["max_score"][i]), 4),
                "altitude": (
                    round(float(h_alt[i]), 1) if h_alt is not None else None
                ),
                "mean_slope": (
                    round(float(h_slp[i]), 1) if h_slp is not None else None
                ),
                "dominant_species": dominant_species,
                "dominant_geology": dominant_geology,
                "compactness": round(float(stats["compactness"][i]), 3),
                "confidence": conf_rounded,
            })

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

    # ------------------------------------------------------------------
    # Hotspots — catégorie dominante via bincount sur int raster
    # ------------------------------------------------------------------

    def _get_dominant_value_in_cluster(
        self,
        cluster_mask: np.ndarray,
        field: str,
        _column_unused: str | None = None,
    ) -> str | None:
        """
        Catégorie dominante dans un cluster via bincount sur int raster (D6).

        Parameters
        ----------
        cluster_mask : bool array (ny, nx), True pour les cellules du cluster
        field : "tree_species" | "geology"
        _column_unused : ignoré — conservé pour compatibilité call sites

        Returns
        -------
        Nom canonique de la catégorie dominante, ou None si pas de données.
        """
        grid = self.grid

        if field == "tree_species":
            int_raster = getattr(grid, "_tree_species_int_raster", None)
            code_to_name = getattr(grid, "_tree_code_to_name", None)
        elif field == "geology":
            int_raster = getattr(grid, "_geology_int_raster", None)
            code_to_name = getattr(grid, "_geology_code_to_name", None)
        else:
            return None

        if int_raster is None or code_to_name is None:
            return None

        cluster_mask = np.asarray(cluster_mask, dtype=bool)
        int_raster = np.asarray(int_raster, dtype=np.int16)

        values = int_raster[cluster_mask]
        values = values[values > 0]  # exclure nodata

        if len(values) == 0:
            return None

        counts = np.bincount(values)
        dominant_code = int(counts.argmax())
        return code_to_name.get(dominant_code)

    def _reverse_lookup_score(
        self,
        score_name: str,
        score_value: float,
    ) -> str | None:
        """
        Retrouve le nom de catégorie à partir d'un score float.

        Utilise les lookup tables int-codées pour un matching exact
        plutôt que la recherche inverse dans les dicts config.
        """
        grid = self.grid

        if score_name == "tree_species":
            code_to_name = getattr(grid, "_tree_code_to_name", None)
            lookup = getattr(grid, "_tree_score_lookup", None)
        elif score_name == "geology":
            code_to_name = getattr(grid, "_geology_code_to_name", None)
            lookup = getattr(grid, "_geology_score_lookup", None)
        else:
            return None

        if code_to_name is None or lookup is None:
            return None

        lookup = np.asarray(lookup, dtype=np.float32)

        # Matching exact d'abord (tolérance float32)
        matches = np.where(np.abs(lookup - score_value) < 1e-6)[0]
        matches = matches[matches > 0]  # exclure nodata (code 0)

        if len(matches) == 0:
            return None

        # Si plusieurs codes ont le même score, retourne le premier
        best_code = int(matches[0])
        return code_to_name.get(best_code)

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
        """Données TWI pour affichage cartographique."""
        twi_raw: np.ndarray | None = getattr(self.grid, "twi", None)  # P4
        twi_score: np.ndarray | None = self.grid.scores.get("twi")

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