"""Loader tests. The important one is that every byte span round-trips."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lede.data import (  # noqa: E402
    OffsetError,
    _byte_span_to_char_span,
    check_offsets,
    group_by_homograph,
    load_cmudict,
    load_split,
    load_tsv,
    load_wordids,
)

TRAIN = load_split("train")
EVAL = load_split("eval")


class TestByteSpanConversion(unittest.TestCase):
    def test_ascii_span_is_identity(self):
        sentence = "The bass was loud."
        self.assertEqual(_byte_span_to_char_span(sentence, 4, 8), (4, 8))

    def test_multibyte_prefix_shifts_span(self):
        # "café " is 6 bytes but 5 characters, so the byte span of "bass"
        # starts two positions later than its character span.
        sentence = "café bass"
        start, end = _byte_span_to_char_span(sentence, 6, 10)
        self.assertEqual((start, end), (5, 9))
        self.assertEqual(sentence[start:end], "bass")

    def test_span_splitting_a_character_is_rejected(self):
        with self.assertRaises(OffsetError):
            _byte_span_to_char_span("café", 0, 4)

    def test_out_of_range_span_is_rejected(self):
        with self.assertRaises(OffsetError):
            _byte_span_to_char_span("bass", 0, 99)


class TestSplits(unittest.TestCase):
    def test_expected_shape(self):
        self.assertEqual(len(group_by_homograph(TRAIN)), 162)
        self.assertEqual(len(group_by_homograph(EVAL)), 162)
        self.assertGreater(len(TRAIN), len(EVAL))

    def test_every_span_round_trips(self):
        problems = check_offsets(TRAIN) + check_offsets(EVAL)
        self.assertEqual(problems, [], f"{len(problems)} bad spans, e.g. {problems[:5]}")

    def test_span_recovers_a_nonempty_substring_of_the_sentence(self):
        for example in TRAIN + EVAL:
            with self.subTest(source=example.source):
                self.assertTrue(example.surface)
                self.assertEqual(
                    example.sentence[example.start : example.end], example.surface
                )

    def test_splits_are_disjoint(self):
        train_keys = {(e.sentence, e.start) for e in TRAIN}
        eval_keys = {(e.sentence, e.start) for e in EVAL}
        self.assertEqual(train_keys & eval_keys, set())

    def test_load_order_is_stable(self):
        self.assertEqual(
            [e.source for e in load_split("eval")], [e.source for e in EVAL]
        )


class TestWordIds(unittest.TestCase):
    def setUp(self):
        self.wordids = load_wordids()

    def test_every_label_has_metadata(self):
        seen = {e.wordid for e in TRAIN} | {e.wordid for e in EVAL}
        self.assertEqual(seen - set(self.wordids), set())

    def test_every_homograph_has_at_least_two_readings(self):
        by_homograph: dict[str, set[str]] = {}
        for entry in self.wordids.values():
            by_homograph.setdefault(entry.homograph, set()).add(entry.wordid)
        thin = {h: w for h, w in by_homograph.items() if len(w) < 2}
        self.assertEqual(thin, {})


class TestAdversarialSet(unittest.TestCase):
    """The hand-written set must satisfy the same invariants as the corpus."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent.parent / "adversarial" / "adversarial.tsv"
        if not path.exists():
            raise unittest.SkipTest("adversarial set not built")
        cls.rows = load_tsv(path, "adversarial")

    def test_spans_round_trip(self):
        self.assertEqual(check_offsets(self.rows), [])

    def test_covers_enough_homographs_with_both_readings(self):
        by_homograph: dict[str, set[str]] = {}
        for row in self.rows:
            by_homograph.setdefault(row.homograph, set()).add(row.wordid)
        self.assertGreaterEqual(len(by_homograph), 15)
        both = [h for h, w in by_homograph.items() if len(w) > 1]
        self.assertGreaterEqual(len(both), 10, "too few homographs with both readings")

    def test_labels_exist_in_the_corpus(self):
        wordids = load_wordids()
        unknown = {r.wordid for r in self.rows} - set(wordids)
        self.assertEqual(unknown, set())

    def test_every_sentence_has_a_category(self):
        import csv

        path = Path(__file__).resolve().parent.parent / "adversarial" / "categories.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            categories = {
                row["sentence"]: row["category"]
                for row in csv.DictReader(handle, delimiter="\t", quotechar='"')
            }
        missing = [r.source for r in self.rows if r.sentence not in categories]
        self.assertEqual(missing, [])


class TestCmudict(unittest.TestCase):
    def test_cmudict_covers_the_homographs(self):
        cmudict = load_cmudict()
        if not cmudict:
            self.skipTest("cmudict not fetched")
        homographs = {e.homograph for e in TRAIN}
        missing = {h for h in homographs if h.lower() not in cmudict}
        # A few dataset keys are inflected or rare; require broad coverage.
        self.assertLess(len(missing), 15, f"missing from cmudict: {sorted(missing)}")

    def test_known_homographs_have_multiple_pronunciations(self):
        cmudict = load_cmudict()
        if not cmudict:
            self.skipTest("cmudict not fetched")
        for word in ("bass", "read", "live", "bow"):
            self.assertGreater(len(cmudict[word]), 1, word)


if __name__ == "__main__":
    unittest.main(verbosity=2)
