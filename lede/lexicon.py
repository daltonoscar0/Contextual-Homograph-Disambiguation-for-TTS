"""Pronunciation lexicon: CMUdict lookup plus a wordid -> ARPAbet table.

Two sources feed this module and they answer different questions.

``cmudict.dict`` answers "how is this word pronounced", for every word. It is
the lexicon a TTS front end actually ships, and it is what we return for
ordinary in-vocabulary tokens.

``wordids.tsv`` answers "what are the competing readings of this homograph, and
which is which". It gives one IPA string per reading. That is the thing CMUdict
cannot tell us: CMUdict lists a word's variants but does not say which variant
goes with which sense, and for 36 of the 162 homographs it does not list the
second reading at all (``sake`` has only ``S EY1 K``, never the Japanese
``S AA1 K EY2``; ``pasty`` is absent entirely).

So the corpus IPA is primary. We convert it to ARPAbet, then *snap* the result
onto a CMUdict variant when one is within an edit distance of 1, so that the
pronunciation we emit for a heteronym is drawn from the same inventory and in
the same style as the pronunciation we emit for every other token. When nothing
in CMUdict is close, the converted IPA stands on its own.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations

from lede.data import load_cmudict, load_wordids

# --------------------------------------------------------------------------
# IPA -> ARPAbet
# --------------------------------------------------------------------------

# Longest match first: the digraphs have to be tried before their first
# character is consumed as a bare monophthong.
_VOWELS: tuple[tuple[str, str], ...] = (
    ("aɪ", "AY"), ("aʊ", "AW"), ("ɔɪ", "OY"), ("eɪ", "EY"), ("oʊ", "OW"),
    ("iː", "IY"), ("uː", "UW"), ("ɑː", "AA"), ("ɔː", "AO"), ("ɜː", "ER"),
    ("ɚ", "ER"), ("ɝ", "ER"),
    ("ɪ", "IH"), ("ɛ", "EH"), ("æ", "AE"), ("ʌ", "AH"), ("ə", "AH"),
    ("ʊ", "UH"), ("ɑ", "AA"), ("ɔ", "AO"), ("i", "IY"), ("u", "UW"),
    ("e", "EH"), ("o", "OW"), ("a", "AA"),
)

_AFFRICATES: tuple[tuple[str, str], ...] = (("tʃ", "CH"), ("dʒ", "JH"))

_CONSONANTS: dict[str, str] = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
    "ʃ": "SH", "ʒ": "ZH", "h": "HH", "m": "M", "n": "N", "ŋ": "NG",
    "l": "L", "ɹ": "R", "r": "R", "j": "Y", "w": "W", "ʧ": "CH", "ʤ": "JH",
}

# Four of the 327 upstream IPA strings carry stray characters that are clearly
# typos rather than phonology: a digit inside the transcription ("ə'bjuː1səz",
# "ə0'fɪˌliːˌeɪt") and a length mark on a consonant ("ˌɔɹːnə'mɛnt"). Dropping
# them is safe -- none of them changes which reading is being described -- and
# doing it silently would hide a real data defect, so they are listed here.
_IGNORED = set(" .-‿͡0123456789")


class PronunciationError(ValueError):
    """Raised when an IPA string cannot be converted to ARPAbet."""


def ipa_to_arpabet(ipa: str) -> list[str]:
    """Convert one ``wordids.tsv`` IPA string to stress-marked ARPAbet.

    Stress is written before the syllable in IPA and after the vowel in
    ARPAbet, so a pending stress level is carried forward and attached to the
    next vowel emitted. Unmarked syllables get 0, matching CMUdict.
    """
    phones: list[str] = []
    index, stress = 0, 0
    while index < len(ipa):
        char = ipa[index]
        if char in ("'", "ˈ"):
            stress, index = 1, index + 1
            continue
        if char == "ˌ":
            stress, index = 2, index + 1
            continue
        # A length mark only means anything after a vowel, where the digraph
        # rules above have already consumed it.
        if char in _IGNORED or char == "ː":
            index += 1
            continue

        for source, target in _AFFRICATES:
            if ipa.startswith(source, index):
                phones.append(target)
                index += len(source)
                break
        else:
            for source, target in _VOWELS:
                if ipa.startswith(source, index):
                    phones.append(f"{target}{stress}")
                    stress, index = 0, index + len(source)
                    break
            else:
                if char not in _CONSONANTS:
                    raise PronunciationError(f"unmapped IPA symbol {char!r} in {ipa!r}")
                phones.append(_CONSONANTS[char])
                index += 1
    if not phones:
        raise PronunciationError(f"empty pronunciation from {ipa!r}")
    return phones


def segments(pronunciation: list[str]) -> tuple[str, ...]:
    """Strip stress digits, leaving the bare segment sequence."""
    return tuple(phone.rstrip("012") for phone in pronunciation)


def _edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------
# wordid -> ARPAbet
# --------------------------------------------------------------------------

_SNAP_TOLERANCE = 1


def _assign(converted: list[list[str]], variants: list[list[str]]) -> list[int | None]:
    """Match each converted reading to at most one CMUdict variant.

    Distance is computed over *stress-marked* phones, not bare segments. A
    large class of heteronyms -- ``PERmit``/``perMIT``, ``OVERthrow``/
    ``overTHROW`` -- differ in nothing but stress placement, so a
    stress-insensitive comparison rates both readings identical to the same
    CMUdict entry and snapping erases the only distinction that matters.

    A per-reading nearest-neighbour search would also happily map both readings
    onto the same entry, so we choose the assignment minimising total distance
    over all injective matchings and accept an individual pairing only if it is
    within ``_SNAP_TOLERANCE``. Readings per homograph are 2 or 3, so brute
    force is free.
    """
    if not variants:
        return [None] * len(converted)

    converted_segments = [tuple(c) for c in converted]
    variant_segments = [tuple(v) for v in variants]
    slots = list(range(len(variants))) + [None] * len(converted)

    best_choice, best_cost = None, None
    for candidate in set(permutations(slots, len(converted))):
        cost = 0
        for reading, slot in zip(converted_segments, candidate):
            if slot is None:
                cost += _SNAP_TOLERANCE + 1  # cost of declining to snap
            else:
                cost += _edit_distance(reading, variant_segments[slot])
        if best_cost is None or cost < best_cost:
            best_choice, best_cost = candidate, cost

    return [
        slot
        if slot is not None
        and _edit_distance(converted_segments[i], variant_segments[slot])
        <= _SNAP_TOLERANCE
        else None
        for i, slot in enumerate(best_choice)
    ]


@lru_cache(maxsize=1)
def wordid_pronunciations() -> dict[str, list[str]]:
    """Map every wordid to a single ARPAbet pronunciation.

    Guaranteed by ``tests/test_lexicon.py``: every wordid gets a
    pronunciation, and no two readings of the same homograph get the same one.
    """
    cmudict = load_cmudict()
    by_homograph: dict[str, list] = {}
    for entry in load_wordids().values():
        by_homograph.setdefault(entry.homograph, []).append(entry)

    table: dict[str, list[str]] = {}
    for homograph, entries in by_homograph.items():
        entries = sorted(entries, key=lambda e: e.wordid)
        converted = [ipa_to_arpabet(e.pronunciation) for e in entries]
        variants = cmudict.get(homograph.lower(), [])
        chosen = [
            list(variants[slot]) if slot is not None else reading
            for reading, slot in zip(converted, _assign(converted, variants))
        ]
        # Belt and braces. Snapping is a cosmetic step; if it ever collapses two
        # readings into one string it has destroyed the distinction this whole
        # module exists to preserve, so discard it and keep the raw conversions,
        # which are distinct by construction.
        if len({tuple(c) for c in chosen}) != len(chosen):
            chosen = converted
        for entry, reading in zip(entries, chosen):
            table[entry.wordid] = reading
    return table


def pronunciation_for(wordid: str) -> list[str]:
    return wordid_pronunciations()[wordid]


# --------------------------------------------------------------------------
# Heteronym inventory
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def gorman_homographs() -> frozenset[str]:
    """The 162 curated homographs, lowercased. These are the decidable ones."""
    return frozenset(entry.homograph.lower() for entry in load_wordids().values())


def _primary_stress(pronunciation: list[str]) -> tuple[str | None, int]:
    for index, phone in enumerate(pronunciation):
        if phone.endswith("1"):
            return phone[:-1], index
    return None, -1


def _nuclei(pronunciation: list[str]) -> list[str]:
    return [phone[:-1] for phone in pronunciation if phone[-1:].isdigit()]


@lru_cache(maxsize=1)
def cmudict_heteronym_candidates() -> frozenset[str]:
    """CMUdict words whose variants differ in stress *placement* or in the
    quality of the stressed vowel.

    This is the mechanical version of "a heteronym is a word with more than one
    pronunciation". It is deliberately kept even though it does not work well
    enough to drive ``run()`` -- see ``README.md``. Variants differing only in
    stress *level*, in unstressed vowel reduction, or in syllable count are
    dropped as free variation.
    """
    candidates = set()
    for word, prons in load_cmudict().items():
        if len(prons) < 2:
            continue
        stresses = {_primary_stress(pron) for pron in prons}
        if len({index for _, index in stresses}) > 1:
            candidates.add(word)
            continue
        if len({vowel for vowel, _ in stresses}) > 1:
            candidates.add(word)
    return frozenset(candidates)


def lookup(word: str) -> list[list[str]]:
    """Every CMUdict pronunciation of ``word``, or ``[]`` if out of vocabulary."""
    return load_cmudict().get(word.lower(), [])
