"""Loaders for the Wikipedia Homograph Data set.

The upstream TSVs quote every field and give the target span as *byte* offsets
into the UTF-8 encoding of the sentence, which is not the same thing as
character offsets once a sentence contains non-ASCII text. Everything in this
module works in byte space and converts to character space exactly once, at the
boundary where we hand spans to downstream consumers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "vendor" / "WikipediaHomographData" / "data"
CMUDICT_PATH = REPO_ROOT / "data" / "vendor" / "cmudict" / "cmudict.dict"


class OffsetError(ValueError):
    """Raised when a row's byte span does not line up with its sentence."""


@dataclass(frozen=True)
class Example:
    """One labeled homograph occurrence.

    ``start``/``end`` are *character* offsets into ``sentence``, already
    converted from the on-disk byte offsets. ``sentence[start:end]`` is the
    surface form of the homograph as it appears in the text.
    """

    homograph: str
    wordid: str
    sentence: str
    start: int
    end: int
    split: str
    source: str

    @property
    def surface(self) -> str:
        return self.sentence[self.start : self.end]


@dataclass(frozen=True)
class WordId:
    """A row of ``wordids.tsv``: one pronunciation of one homograph."""

    homograph: str
    wordid: str
    label: str
    pronunciation: str
    homograph_type: str
    fine_homograph_type: str


def _byte_span_to_char_span(sentence: str, start: int, end: int) -> tuple[int, int]:
    """Convert a UTF-8 byte span to a character span.

    Raises ``OffsetError`` if either endpoint lands inside a multi-byte
    character, which would mean the annotation is inconsistent with the text.
    """
    raw = sentence.encode("utf-8")
    if not 0 <= start <= end <= len(raw):
        raise OffsetError(
            f"span [{start},{end}) out of range for {len(raw)}-byte sentence"
        )
    try:
        char_start = len(raw[:start].decode("utf-8"))
        char_end = char_start + len(raw[start:end].decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise OffsetError(f"span [{start},{end}) splits a multi-byte character") from exc
    return char_start, char_end


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quotechar='"'))


def load_split(split: str, data_root: Path = DATA_ROOT) -> list[Example]:
    """Load every example in ``train`` or ``eval``, in stable sorted order.

    Order is fixed by (homograph file name, row index) so that repeated runs
    produce identical caches and identical classifier fits.
    """
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} not found. Run scripts/fetch_data.sh first."
        )

    examples: list[Example] = []
    for tsv_path in sorted(split_dir.glob("*.tsv")):
        for line_no, row in enumerate(_read_tsv(tsv_path), start=2):
            sentence = row["sentence"]
            source = f"{split}/{tsv_path.name}:{line_no}"
            try:
                start, end = _byte_span_to_char_span(
                    sentence, int(row["start"]), int(row["end"])
                )
            except OffsetError as exc:
                raise OffsetError(f"{source}: {exc}") from exc
            examples.append(
                Example(
                    homograph=row["homograph"],
                    wordid=row["wordid"],
                    sentence=sentence,
                    start=start,
                    end=end,
                    split=split,
                    source=source,
                )
            )
    return examples


def load_wordids(data_root: Path = DATA_ROOT) -> dict[str, WordId]:
    """Map wordid -> metadata from ``wordids.tsv``."""
    rows = _read_tsv(data_root / "wordids.tsv")
    return {
        row["wordid"]: WordId(
            homograph=row["homograph"],
            wordid=row["wordid"],
            label=row["label"],
            pronunciation=row["pronunciation"],
            homograph_type=row["homograph_type"],
            # The upstream header spells this field two different ways across
            # the README and the file itself; accept either.
            fine_homograph_type=row.get("fine_homograph_type")
            or row.get("fine_homography_type", ""),
        )
        for row in rows
    }


def group_by_homograph(examples: list[Example]) -> dict[str, list[Example]]:
    grouped: dict[str, list[Example]] = {}
    for example in examples:
        grouped.setdefault(example.homograph, []).append(example)
    return grouped


def check_offsets(examples: list[Example]) -> list[str]:
    """Return a human-readable problem report for spans that look wrong.

    A span is considered good if the substring it selects matches the declared
    homograph up to case and a trailing inflectional suffix -- the data set
    labels inflected forms (``uses``, ``recorded``) with the lemma-ish
    homograph key, so exact string equality is too strict.
    """
    problems: list[str] = []
    for example in examples:
        surface = example.surface
        if not surface:
            problems.append(f"{example.source}: empty span")
            continue
        if surface != surface.strip():
            problems.append(f"{example.source}: span {surface!r} has surrounding space")
            continue
        folded, homograph = surface.casefold(), example.homograph.casefold()
        if folded == homograph:
            continue
        if folded.startswith(homograph) and len(folded) - len(homograph) <= 3:
            continue
        # A handful of homographs are labeled by their bare stem while the text
        # shows an orthographic variant (e.g. "lived" for "live"). Allow a
        # shared prefix of all but the final stem character.
        if len(homograph) > 3 and folded.startswith(homograph[:-1]):
            continue
        problems.append(
            f"{example.source}: span {surface!r} does not match homograph "
            f"{example.homograph!r}"
        )
    return problems


@lru_cache(maxsize=1)
def load_cmudict(path: Path = CMUDICT_PATH) -> dict[str, list[list[str]]]:
    """Parse CMUdict into word -> list of ARPAbet pronunciations.

    Supporting data only: we use it to sanity-check that the wordids for a
    homograph really do correspond to distinct pronunciations, never as a
    modeling target.
    """
    if not path.exists():
        return {}
    entries: dict[str, list[list[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            word, *phones = line.split()
            # cmudict marks alternate pronunciations as "word(2)".
            if word.endswith(")") and "(" in word:
                word = word[: word.rindex("(")]
            entries.setdefault(word.lower(), []).append(phones)
    return entries
