"""POS-rule baseline.

For each homograph we learn, from the training split only, a map from the
coarse POS tag of the homograph token to the majority wordid observed with that
tag. At prediction time an unseen tag falls back to the homograph's overall
majority wordid. This is the classic rule-system design the Gorman et al. paper
describes as its starting point, reconstructed empirically rather than by hand.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from lede.data import Example, REPO_ROOT, group_by_homograph, load_split

CACHE_DIR = REPO_ROOT / "cache"
MODEL_PATH = CACHE_DIR / "baseline_pos.json"
TAGS_CACHE = CACHE_DIR / "pos_tags.pkl"

_NLP = None


def _nlp():
    """Load spaCy lazily; the tagger is the only pipe we need."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load(
            "en_core_web_sm", exclude=["ner", "lemmatizer", "parser", "senter"]
        )
    return _NLP


def _align_token(doc, start: int, end: int) -> "object | None":
    """Find the spaCy token covering character span ``[start, end)``.

    ``doc.char_span`` is strict about token boundaries, so we fall back to the
    token with the largest character overlap. That matters for cases where the
    tagger splits or merges around the target (hyphenation, clitics).
    """
    span = doc.char_span(start, end, alignment_mode="expand")
    if span is not None and len(span) == 1:
        return span[0]

    best, best_overlap = None, 0
    for token in doc:
        overlap = min(end, token.idx + len(token.text)) - max(start, token.idx)
        if overlap > best_overlap:
            best, best_overlap = token, overlap
    if best is not None:
        return best
    return span[0] if span is not None and len(span) else None


def tag_examples(examples: list[Example], cache_key: str) -> list[str]:
    """Return the coarse POS tag of the homograph token for each example.

    Tagging all ~16k sentences takes about a minute, so results are cached by
    split. ``"?"`` marks an example whose target could not be aligned to any
    token at all.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache: dict[str, list[str]] = {}
    if TAGS_CACHE.exists():
        cache = pickle.loads(TAGS_CACHE.read_bytes())
        cached = cache.get(cache_key)
        if cached is not None and len(cached) == len(examples):
            return cached

    nlp = _nlp()
    tags: list[str] = []
    docs = nlp.pipe((e.sentence for e in examples), batch_size=64)
    for example, doc in zip(examples, docs):
        token = _align_token(doc, example.start, example.end)
        tags.append(token.pos_ if token is not None else "?")

    cache[cache_key] = tags
    TAGS_CACHE.write_bytes(pickle.dumps(cache))
    return tags


@dataclass
class PosRuleModel:
    """tag -> wordid rules per homograph, plus a per-homograph default."""

    rules: dict[str, dict[str, str]]
    defaults: dict[str, str]

    def predict(self, homograph: str, tag: str) -> str:
        return self.rules.get(homograph, {}).get(tag) or self.defaults[homograph]

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps({"rules": self.rules, "defaults": self.defaults}, indent=2)
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "PosRuleModel":
        blob = json.loads(path.read_text())
        return cls(rules=blob["rules"], defaults=blob["defaults"])


def fit(examples: list[Example], tags: list[str]) -> PosRuleModel:
    """Learn tag -> majority wordid rules. Ties break alphabetically."""
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    totals: dict[str, Counter] = defaultdict(Counter)
    for example, tag in zip(examples, tags):
        counts[example.homograph][tag][example.wordid] += 1
        totals[example.homograph][example.wordid] += 1

    def argmax(counter: Counter) -> str:
        return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    rules = {
        homograph: {tag: argmax(by_wordid) for tag, by_wordid in by_tag.items()}
        for homograph, by_tag in counts.items()
    }
    defaults = {homograph: argmax(c) for homograph, c in totals.items()}
    return PosRuleModel(rules=rules, defaults=defaults)


def predict(model: PosRuleModel, examples: list[Example], tags: list[str]) -> list[str]:
    return [
        model.predict(example.homograph, tag) for example, tag in zip(examples, tags)
    ]


def run() -> dict[str, list[str]]:
    """Fit on train, predict on eval, and persist the model. Returns predictions."""
    train, evaluation = load_split("train"), load_split("eval")
    train_tags = tag_examples(train, "train")
    eval_tags = tag_examples(evaluation, "eval")

    model = fit(train, train_tags)
    model.save()

    predictions = predict(model, evaluation, eval_tags)
    correct = sum(p == e.wordid for p, e in zip(predictions, evaluation))
    print(f"baseline: fit {len(model.rules)} homographs from {len(train)} train rows")
    print(f"baseline: eval micro accuracy {correct / len(evaluation):.4f}")
    return {"predictions": predictions}


def mle_predictions(examples: list[Example]) -> list[str]:
    """Most-frequent-wordid baseline, the paper's 'MLE baseline' row."""
    train = load_split("train")
    defaults = {
        homograph: min(
            Counter(e.wordid for e in rows).items(), key=lambda kv: (-kv[1], kv[0])
        )[0]
        for homograph, rows in group_by_homograph(train).items()
    }
    return [defaults[e.homograph] for e in examples]


if __name__ == "__main__":
    run()
