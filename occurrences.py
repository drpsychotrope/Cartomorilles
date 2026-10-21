"""
occurrences.py — Occurrences naturalistes des espèces modélisées.

Sources, réunies au même format (une liste par espèce du modèle) :
  - iNaturalist : observations de morilles récupérées par species_enricher
    (inat_occurrences) ;
  - GBIF, API publique sans clé (https://www.gbif.org/developer/summary) :
    species/match → taxonKey (genre Morchella, Boletus edulis),
    occurrence/search dans config.BBOX_WGS84 (pages de 300), dataset/{key}
    → titre du jeu de données, affiché comme source.

Cache GBIF JSON par taxon dans data/cache (TTL 7 jours, comme
iNaturalist) ; les deux morilles partagent le genre Morchella (D49). Hors ligne : le dernier cache est repris même périmé ; sans
cache, la liste est vide et le pipeline continue.

GBIF agrège aussi les observations iNaturalist « research grade » (jeu
50c9509d…) : drop_inaturalist_duplicates() retire celles déjà récupérées
par l'API iNaturalist, sur l'identifiant d'observation.

Format d'une occurrence : {"id", "source" ("inaturalist" | "gbif"), "lat",
"lng", "taxon", "date" (ISO partiel AAAA, AAAA-MM ou AAAA-MM-JJ ; "" si
inconnue), "uncertainty_m" (None si non renseignée), "generalized"
(coordonnées floutées), "dataset", "inat_id" (None hors iNaturalist),
"quality" (qualité iNaturalist, None sinon)}.

La plupart des données naturalistes françaises (SINP, MycoflAURA) sont
diffusées au centre d'une maille de 5 km : précision ± 2,8 km. Seules les
occurrences précises (≤ PRECISE_MAX_M, non floutées) servent au contrôle
du modèle (occurrence_check) ; la carte les distingue des autres.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time as _time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import requests
from affine import Affine
from scipy.stats import rankdata

import config
from utils import TO_L93

logger = logging.getLogger("cartomorilles.occurrences")

__all__ = [
    "INAT_TAXON",
    "PRECISE_MAX_M",
    "taxon_for",
    "gbif_cache_path",
    "SOURCES",
    "occurrences_by_species",
    "fetch_gbif_occurrences",
    "inat_occurrences",
    "drop_inaturalist_duplicates",
    "is_precise",
    "occurrence_check",
    "occurrence_metrics",
]

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════

# Taxon des observations iNaturalist de species_enricher (requête genre
# Morchella) : attachées à toute espèce modélisée de ce taxon (D49)
INAT_TAXON: Final[str] = "Morchella"
# Occurrence « précise » : position connue à ± 250 m au plus, non floutée
PRECISE_MAX_M: Final[float] = 250.0
# Sources distinguées par le contrôle du modèle (et leurs libellés)
SOURCES: Final[dict[str, str]] = {
    "inaturalist": "iNaturalist",
    "gbif": "GBIF hors iNaturalist",
}
# Première classe comptée comme favorable dans le contrôle du modèle
_GOOD_LABEL: Final[str] = "Bon"
# Seuils fins de occurrence_metrics (part de surface et d'occurrences ≥ seuil)
_BANDS: Final[tuple[float, ...]] = (0.45, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)

_API: Final[str] = "https://api.gbif.org/v1"
_USER_AGENT: Final[str] = "Cartomorilles (favorabilite morille et cepe, Isere)"
# Taxon GBIF (nom, rang) de chaque espèce : config.SPECIES_PARAMS[sp]["taxon"]
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

def taxon_for(species: str) -> tuple[str, str]:
    """Taxon GBIF (nom, rang) des occurrences d'une espèce modélisée, lu
    dans config.SPECIES_PARAMS (D49)."""
    name, rank = config.get_species_params(species)["taxon"]
    return str(name), str(rank)


def gbif_cache_path(
    taxon: tuple[str, str],
    bbox_wgs84: Mapping[str, float],
    cache_dir: Path,
) -> Path:
    """Fichier de cache GBIF d'un taxon dans une bbox : commun aux espèces
    de même taxon (les deux morilles)."""
    name, rank = taxon
    bbox_key = (
        f"{bbox_wgs84['west']:.4f}_{bbox_wgs84['south']:.4f}_"
        f"{bbox_wgs84['east']:.4f}_{bbox_wgs84['north']:.4f}"
    )
    digest = hashlib.md5(f"{name}|{rank}|{bbox_key}".encode()).hexdigest()[:8]
    slug = name.lower().replace(" ", "_")
    return cache_dir / f"gbif_{slug}_{digest}.json"


def occurrences_by_species(
    species_list: Iterable[str],
    inat_observations: Iterable[Mapping[str, Any]] = (),
    *,
    bbox_wgs84: Mapping[str, float] = config.BBOX_WGS84,
    cache_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Occurrences par espèce : iNaturalist (espèces de taxon INAT_TAXON)
    puis GBIF, sans les doublons iNaturalist de GBIF."""
    species = list(species_list)
    # La requête iNaturalist « Morchella » ramène aussi Verpa bohemica
    # (synonyme Morchella bohemica) : hors genre, pas une morille.
    inat = [
        o for o in inat_occurrences(inat_observations)
        if o["taxon"].split()[0] == "Morchella"
    ]
    gbif = fetch_gbif_occurrences(species, bbox_wgs84, cache_dir=cache_dir)
    morels = set(config.species_with_taxon(INAT_TAXON))
    return {
        sp: (inat if sp in morels else [])
        + drop_inaturalist_duplicates(gbif.get(sp, []), inat)
        for sp in species
    }


def fetch_gbif_occurrences(
    species_list: Iterable[str],
    bbox_wgs84: Mapping[str, float] = config.BBOX_WGS84,
    *,
    cache_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Occurrences GBIF par espèce du modèle, dans la BBOX.

    Retourne {espèce: [occurrence, …]} (format du module ; les occurrences
    du jeu iNaturalist ont source="inaturalist").
    Une requête par taxon distinct (les deux morilles partagent le genre
    Morchella). API indisponible sans cache : [].
    """
    cache_dir = cache_dir or Path("data/cache")
    out: dict[str, list[dict[str, Any]]] = {}
    titles: dict[str, str] = {}
    online = True
    groups: dict[tuple[str, str], list[str]] = {}
    for species in species_list:
        groups.setdefault(taxon_for(species), []).append(species)
    with requests.Session() as session:
        session.headers["User-Agent"] = _USER_AGENT
        for taxon, members in groups.items():
            found, online = _fetch_species(
                session, members[0], taxon, bbox_wgs84, cache_dir,
                titles, online=online,
            )
            for species in members:
                out[species] = list(found)
    return out


def inat_occurrences(
    observations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Observations iNaturalist de species_enricher au format du module."""
    out: list[dict[str, Any]] = []
    for obs in observations:
        try:
            inat_id = int(str(obs["id"]))
            lat = float(str(obs["lat"]))
            lng = float(str(obs["lng"]))
        except (KeyError, ValueError):
            continue
        accuracy = obs.get("accuracy_m")
        out.append({
            "id": inat_id,
            "source": "inaturalist",
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            # Cache iNaturalist antérieur au schéma 2 : pas de taxon
            "taxon": str(obs.get("taxon_name") or "Morchella"),
            "date": str(obs.get("observed_on") or ""),
            "uncertainty_m": (
                float(str(accuracy)) if accuracy is not None else None
            ),
            "generalized": bool(obs.get("obscured")),
            "dataset": "iNaturalist",
            "inat_id": inat_id,
            "quality": str(obs.get("quality_grade") or "") or None,
        })
    return out


def drop_inaturalist_duplicates(
    occurrences: list[dict[str, Any]],
    inat: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retire de `occurrences` (GBIF) les observations iNaturalist déjà
    présentes dans `inat` (même identifiant iNaturalist)."""
    known = {o["inat_id"] for o in inat if o.get("inat_id") is not None}
    kept = [o for o in occurrences if o.get("inat_id") not in known]
    n_dup = len(occurrences) - len(kept)
    if n_dup:
        logger.info(
            "GBIF : %d occurrences iNaturalist déjà présentes écartées",
            n_dup,
        )
    return kept


def is_precise(occurrence: Mapping[str, Any]) -> bool:
    """Position connue à PRECISE_MAX_M près au plus, et non floutée."""
    uncertainty = occurrence.get("uncertainty_m")
    return (
        uncertainty is not None
        and float(uncertainty) <= PRECISE_MAX_M
        and not occurrence.get("generalized")
    )


def _occurrence_cells(
    occurrences: list[Mapping[str, Any]],
    transform: Affine,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Lignes et colonnes de grille des occurrences qui tombent dans la
    grille (transform : affine Lambert-93 de la grille)."""
    if not occurrences:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    xs, ys = TO_L93.transform(
        np.array([float(o["lng"]) for o in occurrences]),
        np.array([float(o["lat"]) for o in occurrences]),
    )
    cols, rows = ~transform * (np.asarray(xs), np.asarray(ys))
    cols = np.floor(np.asarray(cols)).astype(np.int64)
    rows = np.floor(np.asarray(rows)).astype(np.int64)
    inside = (
        (rows >= 0) & (rows < shape[0])
        & (cols >= 0) & (cols < shape[1])
    )
    return rows[inside], cols[inside]


def occurrence_check(
    score: np.ndarray,
    transform: Affine,
    occurrences: Iterable[Mapping[str, Any]],
    *,
    study_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Part des occurrences précises en classe Bon ou Excellent, par
    source, comparée à la part de la surface étudiée dans ces classes.

    score : score final (ny, nx), NaN hors données ; transform : affine
    Lambert-93 de la grille ; study_mask : cellules de l'emprise d'étude
    (None : toute la grille). Une occurrence compte si elle est précise et
    tombe dans une cellule valide de l'emprise.

    Retourne {"classes", "threshold", "precise_max_m", "area_share",
    "sources": {source: {"label", "n_total", "n_precise", "n_used",
    "n_good", "share", "lift"}}} ; share = n_good / n_used, lift = share /
    area_share (None si non calculable).
    """
    good_idx = config.PROBABILITY_LABELS.index(_GOOD_LABEL)
    threshold = float(config.PROBABILITY_THRESHOLDS[good_idx - 1])
    valid = np.isfinite(score)
    if study_mask is not None:
        valid &= study_mask
    good = np.zeros(score.shape, dtype=bool)
    np.greater_equal(score, threshold, out=good, where=valid)
    n_area = int(np.count_nonzero(valid))
    area_share = int(np.count_nonzero(good)) / n_area if n_area else None

    occs = list(occurrences)
    sources: dict[str, dict[str, Any]] = {}
    for source, label in SOURCES.items():
        group = [o for o in occs if o.get("source") == source]
        precise = [o for o in group if is_precise(o)]
        n_used = n_good = 0
        if precise:
            rows, cols = _occurrence_cells(precise, transform, score.shape)
            inside = valid[rows, cols]
            rows, cols = _unique_cells(rows[inside], cols[inside], score.shape)
            n_used = int(rows.size)
            n_good = int(np.count_nonzero(good[rows, cols]))
        share = n_good / n_used if n_used else None
        sources[source] = {
            "label": label,
            "n_total": len(group),
            "n_precise": len(precise),
            "n_used": n_used,
            "n_good": n_good,
            "share": round(share, 4) if share is not None else None,
            "lift": (
                round(share / area_share, 2)
                if share is not None and area_share else None
            ),
        }
    return {
        "classes": list(config.PROBABILITY_LABELS[good_idx:]),
        "threshold": threshold,
        "precise_max_m": PRECISE_MAX_M,
        "area_share": round(area_share, 4) if area_share is not None else None,
        "sources": sources,
    }


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Corrélation de Spearman, ex æquo au rang moyen (des classes sans
    occurrence ont le même P/E = 0 : argsort les départageait par ordre de
    classe — D60) ; None si moins de trois points ou variance nulle."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _unique_cells(
    rows: np.ndarray, cols: np.ndarray, shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Une présence par cellule : relevés répétés d'un même site (deux
    années d'inventaire aux mêmes coordonnées) ou observations voisines
    comptaient plusieurs fois dans l'AUC et les P/E (D60)."""
    if rows.size == 0:
        return rows, cols
    flat = np.unique(rows.astype(np.int64) * int(shape[1]) + cols)
    return flat // int(shape[1]), flat % int(shape[1])


def occurrence_metrics(
    score: np.ndarray,
    transform: Affine,
    occurrences: Iterable[Mapping[str, Any]],
    *,
    study_mask: np.ndarray | None = None,
    bands: tuple[float, ...] = _BANDS,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Qualité du score mesurée sur les occurrences précises (présence
    seule), complément de occurrence_check :

    - classes : par classe de config.PROBABILITY_LABELS (bornes
      `thresholds`, celles de la carte — par rang depuis D52 ; seuils
      absolus config.PROBABILITY_THRESHOLDS à défaut), part de la surface
      étudiée (area_share), nombre et part des occurrences (n_occ,
      occ_share), P/E = occ_share / area_share (1 = pas mieux que le
      hasard) ;
    - boyce : corrélation de Spearman entre le rang des classes et leur
      P/E (Hirzel et al. 2006 : 1 = densité d'occurrences croissante de
      classe en classe), sur les classes de surface non nulle ;
    - auc : probabilité qu'une occurrence tirée au hasard soit mieux notée
      qu'une cellule de l'emprise tirée au hasard (0,5 = hasard) ;
    - share_nul : part des occurrences dans la classe Nul, donc éliminées
      à tort ;
    - median_score : score médian aux occurrences ;
    - bands : pour chaque seuil, part de la surface et des occurrences à ce
      score ou plus, et leur rapport (lift).

    Biais connus : échantillonnage naturaliste près des routes et des
    villes ; position connue à ± PRECISE_MAX_M pour une maille de 50 m.
    """
    valid = np.isfinite(score)
    if study_mask is not None:
        valid &= study_mask
    background = np.sort(np.asarray(score[valid], dtype=np.float32))
    n_area = int(background.size)

    occs = list(occurrences)
    precise = [o for o in occs if is_precise(o)]
    rows, cols = _occurrence_cells(precise, transform, score.shape)
    kept = valid[rows, cols]
    n_in = int(np.count_nonzero(kept))
    rows, cols = _unique_cells(rows[kept], cols[kept], score.shape)
    s_occ = np.asarray(score[rows, cols], dtype=np.float32)
    n_used = int(s_occ.size)

    cuts = np.asarray(
        thresholds if thresholds is not None else config.PROBABILITY_THRESHOLDS,
        dtype=np.float32,
    )
    labels = list(config.PROBABILITY_LABELS)
    area_counts = np.bincount(
        np.digitize(background, cuts), minlength=len(labels),
    )
    occ_counts = np.bincount(
        np.digitize(s_occ, cuts), minlength=len(labels),
    )

    classes: list[dict[str, Any]] = []
    ranks: list[float] = []
    ratios: list[float] = []
    for i, label in enumerate(labels):
        area_share = area_counts[i] / n_area if n_area else None
        occ_share = occ_counts[i] / n_used if n_used else None
        pe = (
            occ_share / area_share
            if area_share and occ_share is not None else None
        )
        classes.append({
            "label": label,
            "area_share": _round(area_share),
            "n_occ": int(occ_counts[i]),
            "occ_share": _round(occ_share),
            "pe": _round(pe, 2),
        })
        if pe is not None:
            ranks.append(float(i))
            ratios.append(pe)

    auc = None
    if n_used and n_area:
        lo = np.searchsorted(background, s_occ, side="left")
        hi = np.searchsorted(background, s_occ, side="right")
        auc = float(np.mean((lo + hi) / 2.0) / n_area)

    bands_out: list[dict[str, Any]] = []
    for t in bands:
        area = (
            float(np.count_nonzero(background >= t)) / n_area
            if n_area else None
        )
        occ = (
            float(np.count_nonzero(s_occ >= t)) / n_used if n_used else None
        )
        bands_out.append({
            "threshold": float(t),
            "area_share": _round(area),
            "occ_share": _round(occ),
            "lift": _round(occ / area, 2) if area and occ is not None else None,
        })

    sources: dict[str, dict[str, Any]] = {}
    for source, label in SOURCES.items():
        group = [o for o in occs if o.get("source") == source]
        sources[source] = {
            "label": label,
            "n_total": len(group),
            "n_precise": sum(1 for o in group if is_precise(o)),
        }

    return {
        "n_total": len(occs),
        "n_precise": len(precise),
        "n_used": n_used,
        "n_same_cell": n_in - n_used,
        "n_area": n_area,
        "precise_max_m": PRECISE_MAX_M,
        "class_thresholds": [round(float(t), 4) for t in cuts],
        "classes": classes,
        "boyce": _round(_spearman(ranks, ratios), 3),
        "auc": _round(auc, 3),
        "share_nul": _round(occ_counts[0] / n_used) if n_used else None,
        "median_score": _round(float(np.median(s_occ)), 3) if n_used else None,
        "bands": bands_out,
        "sources": sources,
    }


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
    cache_file = gbif_cache_path(taxon, bbox, cache_dir)

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
        "source": "inaturalist" if inat_id is not None else "gbif",
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
        # Le jeu iNaturalist de GBIF ne contient que le « research grade »
        "quality": "research" if inat_id is not None else None,
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
