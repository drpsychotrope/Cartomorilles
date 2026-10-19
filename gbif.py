"""
gbif.py — Occurrences GBIF des espèces modélisées (morille, cèpe).

API publique sans clé (https://www.gbif.org/developer/summary) :
  1. species/match      → taxonKey (genre Morchella, Boletus edulis)
  2. occurrence/search  → occurrences dans config.BBOX_WGS84, pages de 300
  3. dataset/{key}      → titre du jeu de données, affiché comme source

Cache JSON par espèce dans data/cache (TTL 7 jours, comme iNaturalist).
Hors ligne : le dernier cache est réutilisé même périmé ; sans cache, la
liste est vide et le pipeline continue sans ces occurrences.

GBIF agrège aussi les observations iNaturalist « research grade » (jeu
50c9509d…). drop_inaturalist_duplicates() retire celles déjà récupérées
par l'API iNaturalist (species_enricher), sur l'identifiant d'observation.

La plupart des données naturalistes françaises (SINP, MycoflAURA) sont
diffusées au centre d'une maille de 5 km : précision ± 2,8 km
(coordinateUncertaintyInMeters), conservée dans chaque occurrence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time as _time
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import requests

import config

logger = logging.getLogger("cartomorilles.gbif")

__all__ = ["fetch_gbif_occurrences", "drop_inaturalist_duplicates"]

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════

_API: Final[str] = "https://api.gbif.org/v1"
_USER_AGENT: Final[str] = "Cartomorilles (favorabilite morille et cepe, Isere)"
# Taxon GBIF interrogé pour chaque espèce du modèle : (nom, rang)
_TAXA: Final[dict[str, tuple[str, str]]] = {
    "esculenta": ("Morchella", "GENUS"),        # toutes les morilles
    "edulis": ("Boletus edulis", "SPECIES"),
}
_PAGE_SIZE: Final[int] = 300                     # maximum de l'API
_MAX_RECORDS: Final[int] = 3000                  # plafond par espèce
_CACHE_TTL_S: Final[int] = 7 * 24 * 3600         # 7 jours
_CACHE_SCHEMA: Final[int] = 1
_TIMEOUT: Final[tuple[float, float]] = (5.0, 30.0)   # connexion, lecture
# Jeu « iNaturalist Research-grade Observations »
_INAT_DATASET_KEY: Final[str] = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
_INAT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"inaturalist\.org/observations/(\d+)",
)
# Préfixe numérique des titres SINP (« 31 - Import de données… »)
_TITLE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^\d+\s*-\s*")
_DEFAULT_TITLE: Final[str] = "jeu de données GBIF"


# ═══════════════════════════════════════════════════════════════════
# API PUBLIQUE
# ═══════════════════════════════════════════════════════════════════

def fetch_gbif_occurrences(
    species_list: Iterable[str],
    bbox_wgs84: Mapping[str, float] = config.BBOX_WGS84,
    *,
    cache_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Occurrences GBIF par espèce du modèle, dans la BBOX.

    Retourne {espèce: [occurrence, …]}. Chaque occurrence :
    {"id" (gbifID), "lat", "lng", "taxon", "date" (ISO partiel : AAAA,
    AAAA-MM ou AAAA-MM-JJ, "" si inconnue), "uncertainty_m" (None si non
    renseignée), "generalized" (coordonnées floutées par le producteur),
    "dataset" (titre du jeu), "inat_id" (None hors iNaturalist)}.

    Espèce sans taxon GBIF défini, API indisponible sans cache : [].
    """
    cache_dir = cache_dir or Path("data/cache")
    out: dict[str, list[dict[str, Any]]] = {}
    titles: dict[str, str] = {}
    online = True
    with requests.Session() as session:
        session.headers["User-Agent"] = _USER_AGENT
        for species in species_list:
            taxon = _TAXA.get(species)
            if taxon is None:
                out[species] = []
                continue
            out[species], online = _fetch_species(
                session, species, taxon, bbox_wgs84, cache_dir,
                titles, online=online,
            )
    return out


def drop_inaturalist_duplicates(
    occurrences: list[dict[str, Any]],
    inat_observations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retire les occurrences GBIF venues d'iNaturalist dont l'observation
    figure déjà dans `inat_observations` (même identifiant iNaturalist)."""
    known: set[int] = set()
    for obs in inat_observations:
        try:
            known.add(int(str(obs["id"])))
        except (KeyError, ValueError):
            continue
    kept = [o for o in occurrences if o.get("inat_id") not in known]
    n_dup = len(occurrences) - len(kept)
    if n_dup:
        logger.info(
            "GBIF : %d occurrences iNaturalist déjà présentes écartées",
            n_dup,
        )
    return kept


# ═══════════════════════════════════════════════════════════════════
# RÉCUPÉRATION + CACHE
# ═══════════════════════════════════════════════════════════════════

def _fetch_species(
    session: requests.Session,
    species: str,
    taxon: tuple[str, str],
    bbox: Mapping[str, float],
    cache_dir: Path,
    titles: dict[str, str],
    *,
    online: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Occurrences d'une espèce (cache, sinon API) et état du réseau.

    `online=False` (échec réseau sur une espèce précédente) : cache seul.
    """
    name, rank = taxon
    bbox_key = (
        f"{bbox['west']:.4f}_{bbox['south']:.4f}_"
        f"{bbox['east']:.4f}_{bbox['north']:.4f}"
    )
    digest = hashlib.md5(f"{name}|{rank}|{bbox_key}".encode()).hexdigest()[:8]
    cache_file = cache_dir / f"gbif_{species}_{digest}.json"

    cached = _read_cache(cache_file)
    if cached is not None:
        age = _time.time() - float(cached["fetched_at"])
        if age < _CACHE_TTL_S:
            logger.info(
                "✅ GBIF %s : %d occurrences depuis cache (%.0fh)",
                name, len(cached["results"]), age / 3600,
            )
            return list(cached["results"]), online

    try:
        if not online:
            raise requests.ConnectionError("réseau indisponible")
        taxon_key = _match_taxon(session, name, rank)
        raw = _search_occurrences(session, taxon_key, bbox)
        _fill_titles(session, {str(r.get("datasetKey", "")) for r in raw}, titles)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        network_down = isinstance(
            exc, (requests.ConnectionError, requests.Timeout),
        )
        if cached is not None:
            fetched = datetime.fromtimestamp(float(cached["fetched_at"]))
            logger.warning(
                "⚠️ GBIF indisponible (%s) : %d occurrences %s reprises "
                "du cache du %s",
                _failure_reason(exc), len(cached["results"]), name,
                f"{fetched:%d/%m/%Y}",
            )
            return list(cached["results"]), online and not network_down
        logger.warning(
            "⚠️ GBIF indisponible (%s) : pas d'occurrences %s",
            _failure_reason(exc), name,
        )
        return [], online and not network_down

    results = [
        occ for occ in (_normalize(r, titles) for r in raw) if occ is not None
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "fetched_at": _time.time(),
                "schema": _CACHE_SCHEMA,
                "taxon": {"name": name, "rank": rank, "key": taxon_key},
                "results": results,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    logger.info(
        "✅ GBIF %s (taxonKey %d) : %d occurrences récupérées",
        name, taxon_key, len(results),
    )
    return results, online


def _failure_reason(exc: BaseException) -> str:
    """Cause courte d'un échec, pour une ligne de console lisible."""
    if isinstance(exc, requests.Timeout):
        return "délai dépassé"
    if isinstance(exc, requests.ConnectionError):
        return "pas de connexion"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, ValueError):
        return str(exc)[:80]
    return type(exc).__name__


def _read_cache(cache_file: Path) -> dict[str, Any] | None:
    """Contenu du cache s'il est lisible et au schéma courant, sinon None."""
    if not cache_file.exists():
        return None
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        if raw.get("schema") != _CACHE_SCHEMA:
            return None
        float(raw["fetched_at"])
        if not isinstance(raw["results"], list):
            return None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return dict(raw)


def _match_taxon(session: requests.Session, name: str, rank: str) -> int:
    """taxonKey GBIF du nom au rang demandé (clé acceptée si synonyme)."""
    resp = session.get(
        f"{_API}/species/match",
        params={"name": name, "rank": rank, "kingdom": "Fungi", "strict": "true"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    key = data.get("acceptedUsageKey") or data.get("usageKey")
    if data.get("matchType") == "NONE" or key is None or data.get("rank") != rank:
        raise ValueError(f"taxon « {name} » ({rank}) introuvable dans GBIF")
    return int(key)


def _search_occurrences(
    session: requests.Session,
    taxon_key: int,
    bbox: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Occurrences brutes du taxon dans la BBOX (pagination offset/limit)."""
    params: dict[str, str | int] = {
        "taxonKey": taxon_key,
        "decimalLatitude": f"{bbox['south']},{bbox['north']}",
        "decimalLongitude": f"{bbox['west']},{bbox['east']}",
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "occurrenceStatus": "PRESENT",
        "limit": _PAGE_SIZE,
    }
    records: list[dict[str, Any]] = []
    while True:
        resp = session.get(
            f"{_API}/occurrence/search",
            params={**params, "offset": len(records)},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        page: list[dict[str, Any]] = data.get("results", [])
        records.extend(page)
        if data.get("endOfRecords", True) or not page:
            break
        if len(records) >= _MAX_RECORDS:
            logger.warning(
                "⚠️ GBIF : %d occurrences chargées sur %s (plafond)",
                len(records), data.get("count", "?"),
            )
            break
    return records


def _fill_titles(
    session: requests.Session,
    dataset_keys: set[str],
    titles: dict[str, str],
) -> None:
    """Complète `titles` {datasetKey: titre} pour les jeux pas encore vus.

    Titre indisponible : libellé générique (le réseau coupé arrête les
    requêtes restantes).
    """
    for key in sorted(dataset_keys - titles.keys()):
        title = ""
        try:
            resp = session.get(f"{_API}/dataset/{key}", timeout=_TIMEOUT)
            resp.raise_for_status()
            title = str(resp.json().get("title") or "")
        except (requests.ConnectionError, requests.Timeout):
            for k in dataset_keys - titles.keys():
                titles[k] = _DEFAULT_TITLE
            return
        except (requests.RequestException, ValueError):
            pass
        titles[key] = _TITLE_PREFIX_RE.sub("", title).strip() or _DEFAULT_TITLE


def _normalize(
    rec: Mapping[str, Any], titles: Mapping[str, str],
) -> dict[str, Any] | None:
    """Occurrence GBIF brute → dict compact (None si coordonnées absentes)."""
    try:
        lat = float(rec["decimalLatitude"])
        lng = float(rec["decimalLongitude"])
        gbif_id = int(rec["key"])
    except (KeyError, TypeError, ValueError):
        return None
    dataset_key = str(rec.get("datasetKey", ""))
    inat_id: int | None = None
    if dataset_key == _INAT_DATASET_KEY:
        match = _INAT_ID_RE.search(
            str(rec.get("occurrenceID") or rec.get("references") or ""),
        )
        if match:
            inat_id = int(match.group(1))
    uncertainty = rec.get("coordinateUncertaintyInMeters")
    return {
        "id": gbif_id,
        "lat": round(lat, 6),
        "lng": round(lng, 6),
        "taxon": str(
            rec.get("species") or rec.get("genus")
            or rec.get("scientificName") or "?",
        ),
        "date": _iso_date(rec),
        "uncertainty_m": float(uncertainty) if uncertainty is not None else None,
        "generalized": bool(rec.get("informationWithheld")),
        "dataset": titles.get(dataset_key, _DEFAULT_TITLE),
        "inat_id": inat_id,
    }


def _iso_date(rec: Mapping[str, Any]) -> str:
    """Date d'observation ISO partielle depuis year/month/day interprétés."""
    year, month, day = rec.get("year"), rec.get("month"), rec.get("day")
    if not year:
        return ""
    if not month:
        return f"{int(year):04d}"
    if not day:
        return f"{int(year):04d}-{int(month):02d}"
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
