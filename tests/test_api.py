"""Contract tests for ``lede.api.run``.

The probe tests need an encoder and the shipped weights. When neither is
available -- no torch, no model download -- ``run`` is still required to return
a well-formed result via the POS fallback, and the structural tests below hold
either way. Only the tests that assert *which* reading was chosen are skipped.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lede import serve  # noqa: E402
from lede.__main__ import CANONICAL, CANONICAL_HINTS, CANONICAL_NAME_SPANS  # noqa: E402
from lede.api import run  # noqa: E402
from lede.g2p import tokenize  # noqa: E402


def _probe_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return serve.weights_path(serve.active_model()).exists()


class TestTokenizer(unittest.TestCase):
    def test_offsets_recover_the_surface(self):
        text = "I read the second Doctor Lee lead a live band."
        for token in tokenize(text):
            self.assertEqual(text[token.start : token.end], token.text)

    def test_punctuation_and_words_are_separated(self):
        kinds = {t.text: t.kind for t in tokenize("Hello, world 42!")}
        self.assertEqual(kinds["Hello"], "word")
        self.assertEqual(kinds[","], "punct")
        self.assertEqual(kinds["42"], "number")

    def test_internal_apostrophes_stay_in_one_token(self):
        self.assertEqual([t.text for t in tokenize("don't")], ["don't"])


class TestContractShape(unittest.TestCase):
    """These hold whether or not the probe can run."""

    @classmethod
    def setUpClass(cls):
        cls.result = run(CANONICAL, CANONICAL_HINTS, CANONICAL_NAME_SPANS)
        cls.blob = cls.result.as_dict()

    def test_result_is_json_serialisable(self):
        import json

        json.loads(self.result.to_json())

    def test_envelope_fields(self):
        self.assertEqual(self.blob["stage"], "lede")
        self.assertTrue(self.blob["ok"])
        self.assertEqual(self.blob["text"], CANONICAL)
        for key in ("tokens", "decisions", "meta"):
            self.assertIn(key, self.blob)

    def test_every_token_is_pronounced_or_flagged(self):
        for token in self.blob["tokens"]:
            with self.subTest(token=token["text"]):
                if token["kind"] == "punct":
                    continue
                self.assertTrue(
                    token["pron"] or token["oov"],
                    f"{token['text']} has neither a pronunciation nor an OOV flag",
                )

    def test_canonical_sentence_has_four_heteronym_decisions(self):
        tokens = [d["token"] for d in self.blob["decisions"]]
        self.assertEqual(tokens, ["read", "lead", "live", "Reading"])

    def test_every_decision_carries_its_full_candidate_set(self):
        for decision in self.blob["decisions"]:
            with self.subTest(token=decision["token"]):
                self.assertGreaterEqual(len(decision["alternatives"]), 2)
                self.assertTrue(decision["rule"])
                prons = [a["pron"] for a in decision["alternatives"]]
                self.assertIn(decision["pron"], prons)
                self.assertEqual(len(set(prons)), len(prons))

    def test_alternatives_are_sorted_by_probability(self):
        for decision in self.blob["decisions"]:
            probabilities = [a["probability"] for a in decision["alternatives"]]
            self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_meta_records_hints_spans_and_oov(self):
        meta = self.blob["meta"]
        self.assertEqual(meta["oov"], [])
        self.assertEqual(meta["pron_hints"], CANONICAL_HINTS)
        self.assertEqual(meta["name_spans"], [list(CANONICAL_NAME_SPANS[0])])
        self.assertEqual(meta["n_decisions"], 4)

    def test_decision_offsets_point_at_the_token(self):
        for decision in self.blob["decisions"]:
            self.assertEqual(
                CANONICAL[decision["start"] : decision["end"]], decision["token"]
            )


class TestFullCoverage(unittest.TestCase):
    def test_out_of_vocabulary_tokens_are_marked_not_guessed(self):
        result = run("The zzyzxian blorptastic")
        blob = result.as_dict()
        self.assertIn("zzyzxian", blob["meta"]["oov"])
        self.assertIn("blorptastic", blob["meta"]["oov"])
        self.assertIn("<oov:ZZYZXIAN>", result.pronunciation)

    def test_ordinary_words_come_from_cmudict(self):
        sources = {t["text"]: t["source"] for t in run("the band played").as_dict()["tokens"]}
        self.assertEqual(sources["band"], "cmudict")

    def test_hint_overrides_the_dictionary(self):
        blob = run("the band played", {"band": "Z Z Z"}).as_dict()
        token = next(t for t in blob["tokens"] if t["text"] == "band")
        self.assertEqual(token["pron"], "Z Z Z")
        self.assertEqual(token["source"], "hint")

    def test_a_hint_matching_a_known_reading_recovers_its_wordid(self):
        blob = run("we drove to Reading", {"Reading": "R EH1 D IH0 NG"}).as_dict()
        decision = blob["decisions"][0]
        self.assertEqual(decision["rule"], "hint")
        self.assertEqual(decision["wordid"], "reading_geo")

    def test_a_hint_matching_no_known_reading_leaves_the_wordid_empty(self):
        blob = run("we drove to Reading", {"Reading": "Z Z Z"}).as_dict()
        self.assertEqual(blob["decisions"][0]["wordid"], "")

    def test_span_hints_are_accepted_too(self):
        text = "the band played"
        hints = [{"start": 4, "end": 8, "pron": ["Z", "Z"]}]
        token = next(
            t for t in run(text, hints).as_dict()["tokens"] if t["text"] == "band"
        )
        self.assertEqual(token["pron"], "Z Z")


@unittest.skipUnless(_probe_available(), "needs an encoder and shipped weights")
class TestProbeDecisions(unittest.TestCase):
    """What the probe actually decides. Reported, never patched."""

    @classmethod
    def setUpClass(cls):
        cls.blob = run(CANONICAL, CANONICAL_HINTS, CANONICAL_NAME_SPANS).as_dict()
        cls.by_token = {d["token"]: d for d in cls.blob["decisions"]}

    def test_the_probe_was_what_decided(self):
        for token in ("read", "lead", "live"):
            self.assertTrue(self.by_token[token]["rule"].startswith("probe:"))

    def test_hint_wins_for_reading(self):
        self.assertEqual(self.by_token["Reading"]["rule"], "hint")
        self.assertEqual(self.by_token["Reading"]["pron"], "R EH1 D IH0 NG")

    def test_name_span_selects_the_proper_noun_reading(self):
        """"Lee" is inside the name span but is not a heteronym; the rule is
        exercised on a sentence where it can bite."""
        text = "We drove to Reading yesterday"
        span = [(text.index("Reading"), text.index("Reading") + len("Reading"))]
        decision = run(text, None, span).as_dict()["decisions"][0]
        self.assertEqual(decision["rule"], "name-span")
        self.assertEqual(decision["pron"], "R EH1 D IH0 NG")


if __name__ == "__main__":
    unittest.main(verbosity=2)
