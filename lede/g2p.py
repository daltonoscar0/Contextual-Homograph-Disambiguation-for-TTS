"""Full grapheme-to-phoneme for a normalised sentence.

``lede.probe`` answers a narrow question: given a sentence and a known
homograph span, which reading is it. A TTS front end needs more than that -- it
needs a pronunciation for *every* token, and it needs to say so explicitly when
it does not have one.

The pipeline per token, in priority order:

1. an explicit ``pron_hints`` entry wins outright; the caller knows something
   we do not;
2. a heteronym inside a ``name_spans`` region takes its proper-noun reading;
3. a heteronym is decided by the probe;
4. anything else in CMUdict takes CMUdict's first pronunciation;
5. everything else is out of vocabulary and is emitted as ``<oov:TOKEN>``.

There is deliberately no neural G2P for step 5. Guessing pronunciations for
unknown words is a separate model with its own evaluation, and a front end that
silently invents a pronunciation is worse than one that says it does not know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lede import lexicon
from lede.data import load_wordids

# Words, including internal apostrophes and hyphens so that "don't" and
# "well-read" survive as units, and everything else as its own token.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+|[^\sA-Za-z\d]")

# Readings that name a person, place, or brand rather than a common noun.
# Taken from the wordid suffix, cross-checked against the ``label`` column.
_PROPER_SUFFIXES = frozenset({"geo", "nam", "corp", "es", "bible", "jp"})
_PROPER_LABEL_HINTS = ("city", "name", "composer", "sports team", "ethnicity")

OOV_TEMPLATE = "<oov:{}>"


@dataclass
class Token:
    """One token with whatever pronunciation we could justify for it."""

    text: str
    start: int
    end: int
    kind: str  # "word" | "number" | "punct"
    pron: list[str] | None = None
    oov: bool = False
    source: str = ""  # hint | name-span | probe | pos-baseline | cmudict | oov

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "pron": " ".join(self.pron) if self.pron else None,
            "oov": self.oov,
            "source": self.source,
        }


@dataclass
class Decision:
    """One heteronym resolved, with the full candidate set behind it."""

    token: str
    start: int
    end: int
    homograph: str
    wordid: str
    pron: list[str]
    alternatives: list[dict] = field(default_factory=list)
    rule: str = ""

    def as_dict(self) -> dict:
        return {
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "homograph": self.homograph,
            "wordid": self.wordid,
            "pron": " ".join(self.pron),
            "alternatives": self.alternatives,
            "rule": self.rule,
        }


def tokenize(text: str) -> list[Token]:
    """Split into tokens carrying character offsets into ``text``."""
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        surface = match.group()
        if surface[0].isdigit():
            kind = "number"
        elif surface[0].isalpha():
            kind = "word"
        else:
            kind = "punct"
        tokens.append(Token(surface, match.start(), match.end(), kind))
    return tokens


def proper_noun_readings(homograph: str) -> list[str]:
    """Wordids of ``homograph`` that denote a proper name."""
    readings = []
    for entry in load_wordids().values():
        if entry.homograph.lower() != homograph.lower():
            continue
        suffix = entry.wordid.split("_", 1)[1] if "_" in entry.wordid else ""
        label = entry.label.lower()
        if suffix in _PROPER_SUFFIXES or any(h in label for h in _PROPER_LABEL_HINTS):
            readings.append(entry.wordid)
    return readings


def readings_of(homograph: str) -> list[str]:
    """Every wordid for ``homograph``, sorted for stable output."""
    return sorted(
        entry.wordid
        for entry in load_wordids().values()
        if entry.homograph.lower() == homograph.lower()
    )


def normalise_hints(pron_hints) -> tuple[dict[str, list[str]], list[dict]]:
    """Accept either shape of ``pron_hints`` and return both indexes.

    The contract does not pin the shape down, so both readings of it work: a
    ``{word: pron}`` mapping applied to every occurrence, and a list of
    ``{"start", "end", "pron"}`` span hints. Span hints are more specific and
    are applied first. A pron may be a string or a list of phones.
    """
    if not pron_hints:
        return {}, []

    def phones(value) -> list[str]:
        return list(value) if isinstance(value, (list, tuple)) else str(value).split()

    if isinstance(pron_hints, dict):
        return {str(k).lower(): phones(v) for k, v in pron_hints.items()}, []

    by_word: dict[str, list[str]] = {}
    spans: list[dict] = []
    for hint in pron_hints:
        if "start" in hint and "end" in hint:
            spans.append(
                {
                    "start": int(hint["start"]),
                    "end": int(hint["end"]),
                    "pron": phones(hint.get("pron") or hint.get("pronunciation")),
                }
            )
        else:
            word = str(hint.get("word") or hint.get("token", "")).lower()
            if word:
                by_word[word] = phones(hint.get("pron") or hint.get("pronunciation"))
    return by_word, spans


def normalise_name_spans(name_spans) -> list[tuple[int, int]]:
    """Accept ``[(start, end)]`` or ``[{"start", "end"}]``."""
    if not name_spans:
        return []
    spans = []
    for item in name_spans:
        if isinstance(item, dict):
            spans.append((int(item["start"]), int(item["end"])))
        else:
            start, end = item[0], item[1]
            spans.append((int(start), int(end)))
    return spans


def inside_span(token: Token, spans: list[tuple[int, int]]) -> bool:
    """True if ``token`` falls entirely within one of ``spans``."""
    return any(start <= token.start and token.end <= end for start, end in spans)


def wordid_for_pron(homograph: str, pron: list[str]) -> str:
    """The wordid whose pronunciation is ``pron``, or "" if none matches.

    A caller-supplied hint usually *is* one of the corpus readings, just
    arriving by a different route. Recovering the wordid keeps a hinted
    decision as informative as a probe-decided one; a hint that matches no
    known reading is legitimate and simply leaves the field empty.
    """
    from lede.lexicon import wordid_pronunciations

    table = wordid_pronunciations()
    for wordid in readings_of(homograph):
        if table[wordid] == list(pron):
            return wordid
    return ""


def is_heteronym(token: Token) -> bool:
    """True if this token is one of the curated, decidable heteronyms.

    The inventory is the Gorman et al. set, because those are the words for
    which we have both a probe and a gold pronunciation per reading. See
    ``lexicon.cmudict_heteronym_candidates`` and the README for why deriving
    the inventory from CMUdict alone does not work.
    """
    return token.kind == "word" and token.text.lower() in lexicon.gorman_homographs()
