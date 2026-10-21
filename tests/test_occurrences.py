"""Tests du module occurrences : taxon par espèce, cache GBIF partagé (D49).

Exécution : .venv/Scripts/python.exe -m unittest tests.test_occurrences
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import occurrences  # noqa: E402

_OBS = [{
    "id": 1, "lat": 45.2, "lng": 5.7, "taxon_name": "Morchella esculenta",
    "accuracy_m": 10, "observed_on": "2026-04-01",
    "quality_grade": "research", "obscured": False,
}]


class TaxonPerSpecies(unittest.TestCase):
    def test_taxon_for(self) -> None:
        self.assertEqual(occurrences.taxon_for("esculenta"), ("Morchella", "GENUS"))
        self.assertEqual(occurrences.taxon_for("elata"), ("Morchella", "GENUS"))
        self.assertEqual(
            occurrences.taxon_for("edulis"), ("Boletus edulis", "SPECIES"),
        )

    def test_cache_path_shared_by_taxon(self) -> None:
        cache = Path("cache")
        grise = occurrences.gbif_cache_path(
            occurrences.taxon_for("esculenta"), config.BBOX_WGS84, cache,
        )
        noire = occurrences.gbif_cache_path(
            occurrences.taxon_for("elata"), config.BBOX_WGS84, cache,
        )
        cepe = occurrences.gbif_cache_path(
            occurrences.taxon_for("edulis"), config.BBOX_WGS84, cache,
        )
        self.assertEqual(grise, noire)
        self.assertNotEqual(grise, cepe)
        self.assertEqual(grise.parent, cache)
        self.assertTrue(grise.name.startswith("gbif_morchella_"), grise.name)
        self.assertTrue(grise.name.endswith(".json"))

    def test_inat_attached_to_both_morels(self) -> None:
        with mock.patch.object(
            occurrences, "fetch_gbif_occurrences", return_value={},
        ):
            out = occurrences.occurrences_by_species(
                ["esculenta", "elata", "edulis"], _OBS,
            )
        self.assertEqual([o["inat_id"] for o in out["esculenta"]], [1])
        self.assertEqual([o["inat_id"] for o in out["elata"]], [1])
        self.assertEqual(out["edulis"], [])

    def test_one_gbif_request_per_taxon(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake(session, species, taxon, bbox, cache_dir, titles, *, online):
            calls.append(taxon)
            return [{"id": f"{taxon[0]}-1"}], online

        with mock.patch.object(occurrences, "_fetch_species", side_effect=fake):
            out = occurrences.fetch_gbif_occurrences(
                ["esculenta", "elata", "edulis"],
            )
        self.assertEqual(
            calls, [("Morchella", "GENUS"), ("Boletus edulis", "SPECIES")],
        )
        self.assertEqual(out["esculenta"], [{"id": "Morchella-1"}])
        self.assertEqual(out["elata"], out["esculenta"])
        self.assertEqual(out["edulis"], [{"id": "Boletus edulis-1"}])


if __name__ == "__main__":
    unittest.main()
