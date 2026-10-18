#!/usr/bin/env python3
"""
🍄 Cartomorilles — fetch_inat.py

Récupère les observations iNaturalist research-grade pour une espèce donnée
en Isère (38) et les exporte en CSV. Utilisé par tools/validate_inat.py
pour la validation post-hoc des cartes de probabilité.

Usage :
    python tools/fetch_inat.py --species esculenta
    python tools/fetch_inat.py --species edulis --out data/inat_edulis.csv
    python tools/fetch_inat.py --species edulis --place-id 53612
    python tools/fetch_inat.py --species edulis --place france  # élargir si peu d'obs

Aucune dépendance avec main.py / scoring.py / grid_builder.py.
Pas appelé par le pipeline principal — préprocessing manuel quand besoin.

Géoprivacy : seules les observations `geoprivacy=open` sont retenues
(coordonnées précises). Les observations obscured/private sont ignorées
car coords floutées à ~25km — inutiles pour validation cellule.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERREUR : module 'requests' manquant. pip install requests")
    sys.exit(1)

logger = logging.getLogger("fetch_inat")

# ── iNat taxon IDs ──────────────────────────────────────────────
# Boletus edulis = 48701, Morchella esculenta = 47348 (vérifier)
# Genre Morchella = 53970 — élargi (toutes espèces morilles)
TAXON_IDS: dict[str, int] = {
    "esculenta": 53970,   # Genre Morchella (élargi)
    "edulis": 48701,      # Boletus edulis (cèpe de Bordeaux)
}

# ── Place IDs iNat ──────────────────────────────────────────────
# Isère (département 38) = 53612 (à vérifier sur l'API)
# France entière = 6753
PLACE_IDS: dict[str, int] = {
    "isere": 53612,
    "france": 6753,
}

INAT_URL = "https://api.inaturalist.org/v1/observations"
PER_PAGE = 200
THROTTLE_S = 1.0   # politesse iNat (1 req/s)

CSV_FIELDS = [
    "observation_id",
    "lon",
    "lat",
    "observed_on",
    "positional_accuracy_m",
    "user_login",
    "license_code",
    "quality_grade",
    "geoprivacy",
]


def fetch_species(
    species: str,
    place_id: int,
    quality_grade: str = "research",
    max_pages: int = 50,
) -> list[dict[str, object]]:
    """Pagine l'API iNat et retourne la liste des observations."""
    if species not in TAXON_IDS:
        raise ValueError(f"species inconnue: {species}, choix: {list(TAXON_IDS)}")

    taxon_id = TAXON_IDS[species]
    rows: list[dict[str, object]] = []
    page = 1

    logger.info(
        "Fetch iNaturalist : species=%s taxon_id=%d place_id=%d",
        species, taxon_id, place_id,
    )

    while page <= max_pages:
        params = {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "quality_grade": quality_grade,
            "geoprivacy": "open",
            "per_page": PER_PAGE,
            "page": page,
            "order_by": "observed_on",
            "order": "desc",
        }
        try:
            r = requests.get(INAT_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Page %d échec : %s", page, exc)
            break

        data = r.json()
        results = data.get("results", []) or []
        if not results:
            logger.info("Page %d : 0 results — fin pagination", page)
            break

        for obs in results:
            geom = obs.get("geojson") or {}
            coords = geom.get("coordinates")
            if not coords or len(coords) != 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            user = obs.get("user") or {}
            rows.append({
                "observation_id": obs.get("id"),
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "observed_on": obs.get("observed_on"),
                "positional_accuracy_m": obs.get("positional_accuracy"),
                "user_login": user.get("login"),
                "license_code": obs.get("license_code"),
                "quality_grade": obs.get("quality_grade"),
                "geoprivacy": obs.get("geoprivacy") or "open",
            })

        logger.info(
            "Page %d : %d obs (cumul %d)", page, len(results), len(rows),
        )

        if len(results) < PER_PAGE:
            break

        page += 1
        time.sleep(THROTTLE_S)

    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    logger.info("✅ CSV écrit : %s (%d lignes)", out_path, len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_inat",
        description="Fetch iNaturalist observations research-grade.",
    )
    parser.add_argument(
        "--species", choices=list(TAXON_IDS), required=True,
        help="Espèce : esculenta (morille) ou edulis (cèpe).",
    )
    parser.add_argument(
        "--place", choices=list(PLACE_IDS), default="isere",
        help="Zone géographique (défaut: isere).",
    )
    parser.add_argument(
        "--place-id", type=int, default=None,
        help="Place ID iNat custom (override --place).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Chemin CSV de sortie (défaut: data/inat_{species}_{place}.csv).",
    )
    parser.add_argument(
        "--quality", choices=["research", "any"], default="research",
        help="Qualité observations (défaut: research).",
    )
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Limite pagination (défaut 50 = jusqu'à 10000 obs).",
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

    place_id = args.place_id if args.place_id else PLACE_IDS[args.place]
    out_path = Path(args.out) if args.out else Path(
        f"data/inat_{args.species}_{args.place}.csv",
    )

    rows = fetch_species(
        args.species,
        place_id=place_id,
        quality_grade=args.quality,
        max_pages=args.max_pages,
    )

    if not rows:
        logger.warning("Aucune observation récupérée — CSV vide non créé")
        return 1

    write_csv(rows, out_path)
    logger.info("📊 Distribution années : voir colonne observed_on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
