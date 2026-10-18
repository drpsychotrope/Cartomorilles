#!/usr/bin/env python3
"""
🍄 Cartomorilles — validate_inat.py

Validation post-hoc des cartes de probabilité par confrontation aux
observations iNaturalist research-grade.

Pipeline :
1. Lit `data/inat_{species}_isere.csv` (généré par fetch_inat.py).
2. Lit `output/{species}_probability.tif` (généré par main.py).
3. Reproject WGS84 → L93, sample raster aux points d'observation.
4. Génère un échantillon null (random) pour comparer la distribution.
5. Calcule AUC-ROC + statistiques distribution.
6. Sort un PNG histogramme + un .txt rapport.

Cible AUC : > 0.65 acceptable, > 0.75 excellent. Si AUC < 0.55, params
écologiques à revoir.

Usage :
    python tools/validate_inat.py --species edulis
    python tools/validate_inat.py --species esculenta --csv data/inat_eculenta.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from pyproj import Transformer
except ImportError as exc:
    print(f"ERREUR import : {exc}. pip install rasterio pyproj")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


logger = logging.getLogger("validate_inat")

DEFAULT_N_RANDOM = 10_000
WGS84_TO_L93 = "+proj=pipeline +step +proj=axisswap +order=2,1 +step +inv +proj=longlat +ellps=WGS84"


def read_inat_csv(csv_path: Path) -> list[tuple[float, float]]:
    """Retourne [(lon, lat), ...] depuis le CSV iNat."""
    rows: list[tuple[float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
                rows.append((lon, lat))
            except (KeyError, ValueError):
                continue
    logger.info("CSV iNat : %d obs valides depuis %s", len(rows), csv_path)
    return rows


def sample_raster_at_points(
    raster_path: Path,
    points_l93: list[tuple[float, float]],
) -> np.ndarray:
    """Sample le raster aux coordonnées L93. Retourne array float (NaN si hors)."""
    with rasterio.open(raster_path) as src:
        scores = np.array([
            list(src.sample([(x, y)]))[0][0]
            for x, y in points_l93
        ])
    nodata = -9999.0
    scores = np.where(scores == nodata, np.nan, scores)
    return scores


def random_null_sample(
    raster_path: Path,
    n: int = DEFAULT_N_RANDOM,
    rng_seed: int = 42,
) -> np.ndarray:
    """Échantillon aléatoire de cellules valides (>0) du raster."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        valid = (arr != nodata) & np.isfinite(arr) & (arr > 0)
        idx_y, idx_x = np.where(valid)
        if idx_y.size == 0:
            return np.array([])
        rng = np.random.default_rng(rng_seed)
        if idx_y.size > n:
            sel = rng.choice(idx_y.size, size=n, replace=False)
            return arr[idx_y[sel], idx_x[sel]]
        return arr[idx_y, idx_x]


def compute_metrics(
    obs_scores: np.ndarray,
    null_scores: np.ndarray,
) -> dict[str, float]:
    """Calcule stats + AUC-ROC."""
    obs_v = obs_scores[np.isfinite(obs_scores)]
    null_v = null_scores[np.isfinite(null_scores)]

    metrics: dict[str, float] = {
        "n_obs": float(obs_v.size),
        "n_null": float(null_v.size),
        "obs_mean": float(obs_v.mean()) if obs_v.size > 0 else float("nan"),
        "obs_median": float(np.median(obs_v)) if obs_v.size > 0 else float("nan"),
        "obs_std": float(obs_v.std()) if obs_v.size > 0 else float("nan"),
        "null_mean": float(null_v.mean()) if null_v.size > 0 else float("nan"),
        "null_median": float(np.median(null_v)) if null_v.size > 0 else float("nan"),
    }

    if HAS_SKLEARN and obs_v.size >= 5 and null_v.size >= 5:
        labels = np.concatenate([
            np.ones(obs_v.size, dtype=int),
            np.zeros(null_v.size, dtype=int),
        ])
        scores = np.concatenate([obs_v, null_v])
        auc = float(roc_auc_score(labels, scores))
        metrics["auc_roc"] = auc
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def write_report(
    out_dir: Path,
    species: str,
    metrics: dict[str, float],
    obs_scores: np.ndarray,
    null_scores: np.ndarray,
) -> None:
    """Écrit rapport texte + histogramme PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"validation_inat_{species}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Validation iNaturalist — Cartomorilles {species}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Observations iNat (research-grade, geoprivacy=open) : {int(metrics['n_obs'])}\n")
        f.write(f"Échantillon null (random forêt)                     : {int(metrics['n_null'])}\n\n")
        f.write("Distribution des scores prédits :\n")
        f.write(f"  Aux observations : mean={metrics['obs_mean']:.3f}  median={metrics['obs_median']:.3f}  std={metrics['obs_std']:.3f}\n")
        f.write(f"  Random null      : mean={metrics['null_mean']:.3f}  median={metrics['null_median']:.3f}\n\n")
        auc = metrics.get("auc_roc", float("nan"))
        if np.isfinite(auc):
            f.write(f"AUC-ROC : {auc:.3f}\n")
            interp = (
                "EXCELLENT (>0.75)" if auc > 0.75
                else "BON (>0.65)" if auc > 0.65
                else "ACCEPTABLE (>0.55)" if auc > 0.55
                else "FAIBLE — params à recalibrer"
            )
            f.write(f"Interprétation : {interp}\n")
        else:
            f.write("AUC-ROC : N/A (sklearn manquant ou n_obs < 5)\n")
    logger.info("📄 Rapport : %s", txt_path)

    if HAS_MPL:
        png_path = out_dir / f"validation_inat_{species}.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 21)
        obs_v = obs_scores[np.isfinite(obs_scores)]
        null_v = null_scores[np.isfinite(null_scores)]
        if obs_v.size > 0:
            ax.hist(obs_v, bins=bins, alpha=0.6, label=f"Observations iNat (n={obs_v.size})",
                    color="#2d8a4e", density=True)
        if null_v.size > 0:
            ax.hist(null_v, bins=bins, alpha=0.4, label=f"Random null (n={null_v.size})",
                    color="#888", density=True)
        ax.set_xlabel("Score de probabilité prédit")
        ax.set_ylabel("Densité")
        auc = metrics.get("auc_roc", float("nan"))
        title = f"Validation iNaturalist — {species}"
        if np.isfinite(auc):
            title += f"  (AUC={auc:.3f})"
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        logger.info("📊 Histogramme : %s", png_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="validate_inat",
        description="Validation post-hoc raster vs iNat observations.",
    )
    parser.add_argument(
        "--species", choices=["esculenta", "edulis"], required=True,
        help="Espèce à valider.",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Chemin CSV iNat (défaut: data/inat_{species}_isere.csv).",
    )
    parser.add_argument(
        "--raster", type=str, default=None,
        help="Chemin raster (défaut: output/{species}_probability.tif).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Répertoire output (défaut: output).",
    )
    parser.add_argument(
        "--n-random", type=int, default=DEFAULT_N_RANDOM,
        help="Taille échantillon null random (défaut: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (défaut: 42).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Logs DEBUG.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    csv_path = Path(args.csv) if args.csv else Path(f"data/inat_{args.species}_isere.csv")
    raster_path = Path(args.raster) if args.raster else Path(f"output/{args.species}_probability.tif")
    out_dir = Path(args.output_dir)

    if not csv_path.exists():
        logger.error("CSV iNat absent : %s. Run tools/fetch_inat.py d'abord.", csv_path)
        return 1
    if not raster_path.exists():
        logger.error("Raster absent : %s. Run main.py d'abord.", raster_path)
        return 1

    # ── Lecture CSV iNat → reprojection L93 ──
    obs_lonlat = read_inat_csv(csv_path)
    if not obs_lonlat:
        logger.error("Aucune observation valide dans le CSV.")
        return 1

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    obs_l93 = [transformer.transform(lon, lat) for lon, lat in obs_lonlat]

    # ── Sample raster aux points d'observation ──
    obs_scores = sample_raster_at_points(raster_path, obs_l93)
    n_valid = int(np.isfinite(obs_scores).sum())
    n_in_zone = int(np.isfinite(obs_scores).sum())
    logger.info(
        "Sample : %d obs avec score valide (%d hors zone)",
        n_in_zone, len(obs_l93) - n_in_zone,
    )

    if n_valid < 5:
        logger.error(
            "Trop peu d'observations valides (%d). "
            "Étendre place_id ou vérifier zone.", n_valid,
        )
        return 1

    # ── Échantillon null aléatoire ──
    null_scores = random_null_sample(
        raster_path, n=args.n_random, rng_seed=args.seed,
    )
    if null_scores.size == 0:
        logger.error("Échantillon null vide — raster sans cellules valides ?")
        return 1

    # ── Métriques ──
    metrics = compute_metrics(obs_scores, null_scores)
    auc = metrics.get("auc_roc", float("nan"))
    logger.info("=" * 60)
    logger.info("Validation [%s] :", args.species)
    logger.info("  n_obs valides   : %d", int(metrics["n_obs"]))
    logger.info("  n_null random   : %d", int(metrics["n_null"]))
    logger.info("  obs mean        : %.3f", metrics["obs_mean"])
    logger.info("  null mean       : %.3f", metrics["null_mean"])
    if np.isfinite(auc):
        logger.info("  AUC-ROC         : %.3f", auc)
    else:
        logger.info("  AUC-ROC         : N/A (sklearn manquant)")
    logger.info("=" * 60)

    write_report(out_dir, args.species, metrics, obs_scores, null_scores)

    return 0


if __name__ == "__main__":
    sys.exit(main())
