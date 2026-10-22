"""Essence des peuplements sans essence précise estimée par les peuplements
d'essence connue voisins (D65) : anneau de 1 à 5 km, même type (feuillus /
conifères), poids N / (N + N0), modulation par l'étage ; sans voisin connu,
le mélange régional IFN reste tel quel.

Exécution : .venv/Scripts/python.exe -m pytest tests/test_local_stands.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import config
import species_enricher as se
from species_enricher import SpeciesEnricher

_CELL, _N, _C = 20.0, 400, 200           # grille de 8 km, centre (200, 200)
_UNKNOWN, _CHAT, _HETRE, _SAPIN = 1, 2, 3, 4
_NAMES = {_UNKNOWN: "unknown", _CHAT: "châtaignier", _HETRE: "hêtre", _SAPIN: "sapin"}


def _block(raster: np.ndarray, code: int, row: int, col: int, half: int = 12) -> None:
    raster[row - half:row + half + 1, col - half:col + half + 1] = code


def _setup(ft_code: int = se._FT_FEUILLUS, chestnut: bool = True):
    """Peuplement sans essence au centre ; châtaigneraie de 25 ha à 3 km
    (dans l'anneau), hêtraie à 500 m (trop près), sapinière à 2 km (autre
    type pour un peuplement de feuillus)."""
    raster = np.zeros((_N, _N), dtype=np.int16)
    _block(raster, _UNKNOWN, _C, _C, half=5)
    if chestnut:
        _block(raster, _CHAT, _C, _C + 150)
    _block(raster, _HETRE, _C, _C + 25)
    _block(raster, _SAPIN, _C - 100, _C)
    mask = raster == _UNKNOWN
    ft = np.where(raster > 0, ft_code, 0).astype(np.int8)
    alt = np.full((_N, _N), 800.0, dtype=np.float32)    # montagnard inf. (bande 1)
    region = np.zeros((_N, _N), dtype=np.int32)          # « _default »
    return raster, mask, ft, alt, region


def _scores(species: str, ft_code: int = se._FT_FEUILLUS, chestnut: bool = True,
            with_local: bool = True):
    raster, mask, ft, alt, region = _setup(ft_code, chestnut)
    enricher = SpeciesEnricher()
    local = (
        enricher._local_stand_area(SimpleNamespace(cell_size=_CELL), raster, _NAMES)
        if with_local else None
    )
    score, rel = enricher._compute_regional_scores(
        alt, region, ft, mask, species=species, local=local,
    )
    return score[mask], rel[mask], local


def _expected(q: dict[str, float], species: str) -> tuple[float, float]:
    """Score et fiabilité d'un mélange déjà pondéré (formule de D51)."""
    num = den = w_all = 0.0
    pairs = []
    for s, w in q.items():
        w_all += w
        v = config.get_tree_score(s, species)
        if np.isnan(v):
            continue
        v = 0.25 if v < 0 else v
        pairs.append((w, v))
        num += w * v
        den += w
    mean = num / den
    var = sum(w * (v - mean) ** 2 for w, v in pairs) / den
    rel = den / w_all * max(0.0, 1.0 - var / se._table_variance(species))
    return float(np.clip(mean, 0.05, 1.0)), float(np.clip(rel, 0.0, 1.0))


def test_ring_counts_known_stands_between_1_and_5_km():
    _s, _r, local = _scores("esculenta")
    names, area, f = local
    assert f == 5 and set(names) == {"châtaignier", "hêtre", "sapin"}
    centre = area[:, _C // f, _C // f]
    by = dict(zip(names, centre))
    assert by["châtaignier"] == pytest.approx(25.0, rel=1e-3)   # à 3 km
    assert by["hêtre"] == pytest.approx(0.0, abs=1e-3)          # à 500 m
    assert by["sapin"] == pytest.approx(25.0, rel=1e-3)         # à 2 km


def _expected_with(species: str, ft_code: int, local_area: dict[str, float],
                   known=("châtaignier", "hêtre", "sapin", "épicéa")):
    """Mélange attendu (bande 1) : seule la part régionale des essences
    repérables est redistribuée, q = p + w·(P_K·local − p·[s ∈ K]),
    w = N / (N + N0) ; « sapin » partagé avec l'épicéa comme le mélange."""
    mix = se.SpeciesEnricher._filter_by_forest_type(
        se._REGIONAL_PROPORTIONS["_default"], ft_code,
    )
    prior = se._mix_weights(mix, 1)
    tot = sum(prior.values())
    p = {s: v / tot for s, v in prior.items()}
    n = sum(local_area.values())
    w = n / (n + config.STAND_NEIGHBOUR_N0_HA)
    mod: dict[str, float] = {}
    for s, a in local_area.items():
        if s == "sapin":
            ps, pe = mix.get("sapin", 0.0), mix.get("épicéa", 0.0)
            r = ps / (ps + pe) if ps + pe > 0 else 1.0
            mod["sapin"] = a * r * se._ALT_AFFINITY["sapin"][1]
            mod["épicéa"] = a * (1 - r) * se._ALT_AFFINITY["épicéa"][1]
        else:
            mod[s] = a * se._ALT_AFFINITY[s][1]
    mt = sum(mod.values())
    p_k = sum(v for s, v in p.items() if s in known)
    q = {s: v - w * v * (s in known) for s, v in p.items()}
    for s, v in mod.items():
        q[s] = q.get(s, 0.0) + w * p_k * v / mt
    return _expected(q, species)


@pytest.mark.parametrize("species", ["esculenta", "elata", "edulis"])
def test_broadleaf_stand_takes_its_broadleaf_neighbours(species):
    score, rel, _local = _scores(species)
    # Sapinière : autre type ; hêtraie : à moins de 1 km
    exp_score, exp_rel = _expected_with(species, se._FT_FEUILLUS, {"châtaignier": 25.0})
    np.testing.assert_allclose(score, exp_score, atol=1e-5)
    np.testing.assert_allclose(rel, exp_rel, atol=1e-5)


def test_mixed_stand_takes_every_neighbour():
    score, rel, _local = _scores("elata", ft_code=se._FT_MIXTE)
    exp_score, exp_rel = _expected_with(
        "elata", se._FT_MIXTE, {"châtaignier": 25.0, "sapin": 25.0},
    )
    np.testing.assert_allclose(score, exp_score, atol=1e-5)
    np.testing.assert_allclose(rel, exp_rel, atol=1e-5)


def test_without_known_neighbour_the_regional_mix_is_unchanged():
    # Seuls la hêtraie (trop près) et la sapinière (autre type) entourent
    # le peuplement : rien ne change
    alone, alone_rel, _l = _scores("esculenta", chestnut=False)
    regional, regional_rel, _l = _scores("esculenta", chestnut=False, with_local=False)
    np.testing.assert_array_equal(alone, regional)
    np.testing.assert_array_equal(alone_rel, regional_rel)
