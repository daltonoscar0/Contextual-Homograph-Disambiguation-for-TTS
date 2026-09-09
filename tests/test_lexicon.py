"""Lexicon tests: IPA conversion, the wordid -> ARPAbet table, CMUdict lookup."""

from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lede.data import load_wordids  # noqa: E402
from lede.lexicon import (  # noqa: E402
    PronunciationError,
    cmudict_heteronym_candidates,
    gorman_homographs,
    ipa_to_arpabet,
    lookup,
    segments,
    wordid_pronunciations,
)

ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}
ARPABET_CONSONANTS = {
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
    "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
}


class TestIpaConversion(unittest.TestCase):
    def test_stress_moves_from_before_the_syllable_to_after_the_vowel(self):
        self.assertEqual(ipa_to_arpabet("'ɹɛd"), ["R", "EH1", "D"])
        self.assertEqual(ipa_to_arpabet("'ɹiːd"), ["R", "IY1", "D"])

    def test_unmarked_syllables_get_zero_stress(self):
        self.assertEqual(ipa_to_arpabet("kən'faɪnz"), ["K", "AH0", "N", "F", "AY1", "N", "Z"])

    def test_secondary_stress(self):
        self.assertEqual(ipa_to_arpabet("'sɑːˌkeɪ"), ["S", "AA1", "K", "EY2"])

    def test_digraphs_beat_their_first_character(self):
        # "aɪ" must not be read as "a" followed by "ɪ".
        self.assertEqual(ipa_to_arpabet("'laɪv"), ["L", "AY1", "V"])
        self.assertEqual(ipa_to_arpabet("'lɪv"), ["L", "IH1", "V"])

    def test_stray_typo_characters_are_ignored(self):
        # Four upstream strings carry a digit or a length mark on a consonant.
        self.assertEqual(ipa_to_arpabet("ə0'fɪ"), ["AH0", "F", "IH1"])

    def test_unmapped_symbol_raises(self):
        with self.assertRaises(PronunciationError):
            ipa_to_arpabet("'ɹɛdЖ")


class TestWordidPronunciations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = wordid_pronunciations()
        cls.wordids = load_wordids()

    def test_every_wordid_has_a_pronunciation(self):
        self.assertEqual(set(self.wordids) - set(self.table), set())

    def test_pronunciations_are_valid_arpabet(self):
        for wordid, pron in self.table.items():
            with self.subTest(wordid=wordid):
                self.assertTrue(pron)
                for phone in pron:
                    bare = phone.rstrip("012")
                    if bare in ARPABET_VOWELS:
                        self.assertTrue(phone[-1].isdigit(), f"{wordid}: {phone}")
                    else:
                        self.assertIn(bare, ARPABET_CONSONANTS, f"{wordid}: {phone}")

    def test_readings_of_a_homograph_never_collide(self):
        """The point of the whole stage. If two readings map to the same
        phones, choosing between them cannot change what gets spoken."""
        by_homograph = defaultdict(set)
        for entry in self.wordids.values():
            by_homograph[entry.homograph].add(tuple(self.table[entry.wordid]))
        collisions = {
            homograph: prons
            for homograph, prons in by_homograph.items()
            if len(prons)
            < len({e.wordid for e in self.wordids.values() if e.homograph == homograph})
        }
        self.assertEqual(collisions, {})

    def test_known_readings(self):
        expected = {
            "read_past": "R EH1 D",
            "read_present": "R IY1 D",
            "lead_nou": "L EH1 D",
            "lead_nou-vrb": "L IY1 D",
            "live_adj": "L AY1 V",
            "live_vrb": "L IH1 V",
            "reading_geo": "R EH1 D IH0 NG",
            "reading_en": "R IY1 D IH0 NG",
            "bass": "B EY1 S",
            "bass_corp": "B AE1 S",
        }
        for wordid, pron in expected.items():
            self.assertEqual(" ".join(self.table[wordid]), pron, wordid)

    def test_stress_shift_pairs_stay_distinct(self):
        """Snapping to CMUdict compares stress-marked phones precisely because
        these pairs differ in nothing else."""
        for homograph in ("overthrow", "permit", "record", "conduct"):
            readings = [
                tuple(self.table[e.wordid])
                for e in self.wordids.values()
                if e.homograph == homograph
            ]
            self.assertEqual(len(set(readings)), len(readings), homograph)


class TestInventory(unittest.TestCase):
    def test_gorman_set_is_the_decidable_inventory(self):
        self.assertEqual(len(gorman_homographs()), 162)

    def test_cmudict_filter_does_not_reconstruct_the_curated_set(self):
        """Documented negative result: the mechanical definition of a
        heteronym over CMUdict is both far too broad and incomplete."""
        candidates = cmudict_heteronym_candidates()
        self.assertGreater(len(candidates - gorman_homographs()), 1000)
        self.assertGreater(len(gorman_homographs() - candidates), 50)

    def test_lookup_returns_empty_for_oov(self):
        self.assertEqual(lookup("zzzqqxnotaword"), [])
        self.assertTrue(lookup("band"))

    def test_segments_strips_stress(self):
        self.assertEqual(segments(["R", "EH1", "D"]), ("R", "EH", "D"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
