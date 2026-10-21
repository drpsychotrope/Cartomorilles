#!/usr/bin/env python3
"""
🍄 Cartomorilles — evaluate_occurrences.py

Contrôle du score par les occurrences naturalistes précises (iNaturalist,
GBIF ; module occurrences.py). Pour un ou plusieurs dossiers de sortie de
main.py, lit {species}_probability.tif, restreint le fond à l'emprise
d'étude (couverture BDCharm-50, D39) et imprime occurrence_metrics par
espèce : P/E par classe, indice de Boyce, AUC présence contre fond, part
des occurrences en Nul, parts cumulées par seuil. Avec plusieurs --run,
les colonnes se comparent côte à côte. --combine SP SP… évalue le maximum
cellule à cellule de plusieurs couches de même taxon (les deux morilles),
avec les occurrences de la première.

Usage (cwd = checkout principal, pour data/cache et la géologie) :
    .venv/Scripts/python.exe tools/evaluate_occurrences.py --run output
    .venv/Scripts/python.exe tools/evaluate_occurrences.py \\
        --run sorties_avant --run sorties_apres --species esculenta --json eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402

import config  # noqa: E402
from occurrences import occurrence_metrics, occurrences_by_species, taxon_for  # noqa: E402
from scoring import rank_class_thresholds  # noqa: E402
from species_enricher import SpeciesEnricher  # noqa: E402

_DEFAULT_GEOLOGY = "data/geologie_38/GEO050K_HARM_038_S_FGEOL_2154.shp"
logger = logging.getLogger("cartomorilles.evaluate")


def load_score(path: Path) -> tuple[np.ndarray, Any]:
    """Score final (NaN hors données) et transform affine du GeoTIFF."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.transform


def study_mask(geology: Path, shape: tuple[int, int], transform: Any) -> np.ndarray | None:
    """Cellules de l'emprise d'étude = couverture du shapefile géologique."""
    if not geology.exists():
        logger.warning("Géologie absente (%s) : fond = toute la grille", geology)
        return None
    gdf = gpd.read_file(geology)
    if gdf.crs is not None and gdf.crs.to_epsg() != 2154:
        gdf = gdf.to_crs(epsg=2154)
    geoms = gdf.geometry[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
    arr = np.asarray(geoms.values)
    try:
        geom = shapely.coverage_union_all(arr)
    except shapely.errors.GEOSException:
        geom = shapely.union_all(arr)
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    mask = geometry_mask([geom], out_shape=shape, transform=transform, invert=True)
    logger.info(
        "Emprise d'étude : %d cellules (%.0f %% de la grille)",
        int(mask.sum()), mask.mean() * 100,
    )
    return np.asarray(mask, dtype=bool)


def combine_scores(scores: Sequence[np.ndarray]) -> np.ndarray:
    """Maximum cellule à cellule de plusieurs couches ; NaN seulement là où
    toutes le sont (np.fmax)."""
    out = np.asarray(scores[0], dtype=np.float32)
    for score in scores[1:]:
        out = np.fmax(out, np.asarray(score, dtype=np.float32))
    return out


def combination_label(species: Sequence[str]) -> str:
    return "+".join(species)


def shared_taxon(species: Sequence[str]) -> tuple[str, str]:
    """Taxon commun des espèces combinées ; refus si leurs occurrences ne
    sont pas les mêmes (taxons différents)."""
    taxa = {taxon_for(sp) for sp in species}
    if len(taxa) != 1:
        sys.exit(
            f"--combine {' '.join(species)} : taxons différents "
            f"{sorted(taxa)}, occurrences non comparables",
        )
    return taxa.pop()


def evaluate_layer(
    label: str,
    tifs_by_run: dict[str, list[Path]],
    occurrences: list[dict[str, Any]],
    masks: dict[tuple[Any, ...], np.ndarray | None],
    geology: Path,
) -> dict[str, dict[str, Any]]:
    """Métriques d'une couche (ou du maximum de plusieurs) pour chaque run."""
    results: dict[str, dict[str, Any]] = {}
    for run_name, tifs in tifs_by_run.items():
        missing = [t for t in tifs if not t.exists()]
        if missing:
            logger.warning("%s absent — run ignoré pour %s", missing[0], label)
            continue
        loaded = [load_score(t) for t in tifs]
        score = combine_scores([s for s, _ in loaded])
        transform = loaded[0][1]
        key = (score.shape, tuple(np.round(np.asarray(transform)[:6], 6)))
        if key not in masks:
            masks[key] = study_mask(geology, score.shape, transform)
        # Classes de la carte : par rang dans l'emprise d'étude (D52)
        results[run_name] = occurrence_metrics(
            score, transform, occurrences, study_mask=masks[key],
            thresholds=rank_class_thresholds(score, masks[key]),
        )
    if results:
        print_report(label, results)
    return results


def _pct(value: float | None) -> str:
    return "   —  " if value is None else f"{value * 100:5.1f} %"


def _num(value: float | None, fmt: str = "{:.2f}") -> str:
    return "  —" if value is None else fmt.format(value)


def print_report(species: str, results: dict[str, dict[str, Any]]) -> None:
    runs = list(results)
    first = results[runs[0]]
    src = ", ".join(
        f"{s['label']} {s['n_precise']}/{s['n_total']}" for s in first["sources"].values()
    )
    print(f"\n=== {species} — {first['n_used']} occurrences précises utilisées "
          f"(≤ {first['precise_max_m']:.0f} m ; {src}) ===")
    width = max(12, *(len(r) for r in runs))
    head = f"{'':24s}" + "".join(f"{r:>{width + 2}s}" for r in runs)
    print(head)
    rows = [
        ("AUC présence/fond", lambda m: _num(m["auc"], "{:.3f}")),
        ("Indice de Boyce", lambda m: _num(m["boyce"], "{:.3f}")),
        ("Score médian aux occ.", lambda m: _num(m["median_score"], "{:.3f}")),
        ("Occurrences en Nul", lambda m: _pct(m["share_nul"])),
    ]
    for label, fn in rows:
        print(f"{label:24s}" + "".join(f"{fn(results[r]):>{width + 2}s}" for r in runs))
    print(f"\n{'classe':12s}" + "".join(
        f"{'surf.':>8s}{'occ.':>8s}{'P/E':>6s} |" for _ in runs))
    for r in runs:
        cuts = " | ".join(f"{t:.3f}" for t in results[r].get("class_thresholds", []))
        print(f"seuils {r} : {cuts}")
    for i, cls in enumerate(first["classes"]):
        line = f"{cls['label']:12s}"
        for r in runs:
            c = results[r]["classes"][i]
            line += f"{_pct(c['area_share']):>8s}{_pct(c['occ_share']):>8s}{_num(c['pe']):>6s} |"
        print(line)
    print(f"\n{'score ≥':12s}" + "".join(
        f"{'surf.':>8s}{'occ.':>8s}{'lift':>6s} |" for _ in runs))
    for i, band in enumerate(first["bands"]):
        line = f"{band['threshold']:<12.2f}"
        for r in runs:
            b = results[r]["bands"][i]
            line += f"{_pct(b['area_share']):>8s}{_pct(b['occ_share']):>8s}{_num(b['lift']):>6s} |"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", metavar="DIR",
                        help="dossier de sortie de main.py (répétable ; défaut : output)")
    parser.add_argument("--species", nargs="+", default=list(config.SUPPORTED_SPECIES),
                        choices=list(config.SUPPORTED_SPECIES))
    parser.add_argument("--geology", default=_DEFAULT_GEOLOGY,
                        help="shapefile BDCharm-50 (emprise d'étude)")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--combine", action="append", nargs="+", metavar="SP",
                        help="maximum de plusieurs couches de même taxon, p. ex. "
                             "--combine esculenta elata (répétable)")
    parser.add_argument("--json", metavar="PATH", help="écrit toutes les métriques en JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s", stream=sys.stderr,
    )
    runs = [Path(r) for r in (args.run or ["output"])]
    cache_dir = Path(args.cache_dir)

    combos: list[list[str]] = [list(c) for c in (args.combine or [])]
    for combo in combos:
        shared_taxon(combo)
    needed = list(dict.fromkeys([*args.species, *(sp for c in combos for sp in c)]))

    enricher = SpeciesEnricher()
    inat = enricher._fetch_inaturalist(config.BBOX_WGS84, cache_dir=cache_dir)
    occurrences = occurrences_by_species(needed, inat, cache_dir=cache_dir)

    masks: dict[tuple[Any, ...], np.ndarray | None] = {}
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    geology = Path(args.geology)
    for species in args.species:
        results = evaluate_layer(
            species,
            {run.name: [run / f"{species}_probability.tif"] for run in runs},
            occurrences[species], masks, geology,
        )
        if results:
            all_results[species] = results
    for combo in combos:
        label = combination_label(combo)
        results = evaluate_layer(
            label,
            {run.name: [run / f"{sp}_probability.tif" for sp in combo] for run in runs},
            occurrences[combo[0]], masks, geology,
        )
        if results:
            all_results[label] = results

    if args.json:
        Path(args.json).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.info("Métriques écrites : %s", args.json)
    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
