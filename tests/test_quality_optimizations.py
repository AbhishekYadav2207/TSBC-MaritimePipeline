import sys
import unittest
from pathlib import Path

# Add scripts to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

import importlib

from text_sanitizer import (
    strip_administrative_noise,
    join_words_grammatical,
    format_cargo_description,
    format_damage_description,
    format_casualty_count
)

merge_tables = importlib.import_module("05_merge_tables")
deduplicate_child_records = merge_tables.deduplicate_child_records
normalize_label = merge_tables.normalize_label

class TestCorpusQualityOptimizations(unittest.TestCase):
    
    def test_equipment_normalization(self):
        self.assertEqual(normalize_label("radar1"), "radar")
        self.assertEqual(normalize_label("Radar 2"), "radar")
        self.assertEqual(normalize_label("vhf"), "VHF radio")
        self.assertEqual(normalize_label("gps"), "GPS receiver")
        self.assertEqual(normalize_label("gyro compass"), "gyrocompass")

    def test_equipment_deduplication(self):
        raw_nav = [
            {"NavigationAidTypeDisplayEng": "Radar 1", "OnOffEnumDisplayEng": "On"},
            {"NavigationAidTypeDisplayEng": "Radar 2", "OnOffEnumDisplayEng": "On"},
            {"NavigationAidTypeDisplayEng": "Radar 3", "OnOffEnumDisplayEng": "Off"}
        ]
        deduped = deduplicate_child_records(raw_nav, "nav")
        self.assertEqual(len(deduped), 2)
        on_item = next(d for d in deduped if d["status_clean"] == "On")
        off_item = next(d for d in deduped if d["status_clean"] == "Off")
        self.assertEqual(on_item["item_count"], 2)
        self.assertEqual(off_item["item_count"], 1)

    def test_administrative_noise_removal(self):
        raw_text = "Note: formerly OccNo : 9708-26-1. The vessel was underway (data extraction status pending)."
        cleaned = strip_administrative_noise(raw_text)
        self.assertNotIn("OccNo", cleaned)
        self.assertNotIn("extraction status pending", cleaned)
        self.assertTrue(cleaned.startswith("The vessel was underway"))

    def test_cargo_formatting(self):
        self.assertEqual(format_cargo_description("General cargo"), "laden with general cargo")
        self.assertEqual(format_cargo_description("Grain"), "laden with grain cargo")
        self.assertEqual(format_cargo_description("Coal", "5000 tonnes"), "carrying 5000 tonnes of coal")

    def test_damage_formatting(self):
        self.assertEqual(format_damage_description("Damaged", "Hull"), "sustaining damage to the hull")
        self.assertEqual(format_damage_description("Substantial", "Bow"), "sustaining substantial damage to the bow")

    def test_casualty_singular_plural(self):
        self.assertEqual(format_casualty_count(1, "missing person", "missing persons"), "1 missing person")
        self.assertEqual(format_casualty_count(3, "missing person", "missing persons"), "3 missing persons")
        self.assertEqual(format_casualty_count(1, "fatality", "fatalities"), "1 fatality")

    def test_grammatical_list_joining(self):
        self.assertEqual(join_words_grammatical(["radar"]), "radar")
        self.assertEqual(join_words_grammatical(["radar", "VHF radio"]), "radar and VHF radio")
        self.assertEqual(join_words_grammatical(["radar", "gyrocompass", "VHF radio"]), "radar, gyrocompass, and VHF radio")

if __name__ == "__main__":
    unittest.main()
