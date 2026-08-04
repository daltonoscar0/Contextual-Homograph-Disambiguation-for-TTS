"""Build adversarial/adversarial.tsv from the hand-written sentence list.

Offsets are computed here rather than typed by hand, in the same byte-offset
convention the upstream data set uses, so the shipped TSV is loadable by
lede.data without any special casing. Each sentence also carries a trap
category, written to a sidecar file so the analysis can break accuracy down by
failure mode.

Trap categories:
  misleading_ngram  the words immediately around the target point the wrong way
  long_distance     the disambiguating cue sits far from the target
  garden_path       the local parse suggests the wrong part of speech
  semantic_field    the sentence's topical vocabulary belongs to the other sense
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (homograph, wordid, category, sentence, occurrence index of the target)
SENTENCES: list[tuple[str, str, str, str, int]] = [
    # --- bass -------------------------------------------------------------
    (
        "bass",
        "bass_corp",
        "semantic_field",
        "Between the amplifier and the stacked sheet music sat a glass tank "
        "where his prize bass circled slowly.",
        0,
    ),
    (
        "bass",
        "bass",
        "semantic_field",
        "The angler's son, who had never once cared for the lake or the boats "
        "or the long cold mornings, played bass in a touring band.",
        0,
    ),
    # --- bow --------------------------------------------------------------
    (
        "bow",
        "bow_nou-knot",
        "semantic_field",
        "Standing at the very front of the ship as it cut through the swell, "
        "the archer restrung her bow.",
        0,
    ),
    (
        "bow",
        "bow_nou-ship",
        "semantic_field",
        "The quartermaster counted the arrows, set down the quiver, and went "
        "below to inspect the cracked bow.",
        0,
    ),
    # --- lead -------------------------------------------------------------
    (
        "lead",
        "lead_nou-vrb",
        "semantic_field",
        "The pipes were grey and impossibly heavy, but it was her steady lead "
        "that got the crew through the winter.",
        0,
    ),
    (
        "lead",
        "lead_nou",
        "misleading_ngram",
        "The detective had chased every promising lead for a month, which is "
        "how the barrels of lead turned up in the yard.",
        1,
    ),
    # --- live -------------------------------------------------------------
    (
        "live",
        "live_vrb",
        "misleading_ngram",
        "The bands that record in this studio rarely live in the city, "
        "preferring the quiet of the coast.",
        0,
    ),
    (
        "live",
        "live_adj",
        "long_distance",
        "The audience had come mainly to see the neighbourhood where the "
        "musicians grew up and still live, but the show went out live.",
        1,
    ),
    # --- read -------------------------------------------------------------
    (
        "read",
        "read_present",
        "long_distance",
        "Yesterday's minutes, which the secretary will read aloud at the "
        "meeting next month, ran to forty pages.",
        0,
    ),
    (
        "read",
        "read_past",
        "long_distance",
        "She kept promising she would read it tomorrow, though in truth she "
        "had read it twice already.",
        1,
    ),
    # --- tear -------------------------------------------------------------
    (
        "tear",
        "tear_vrb",
        "semantic_field",
        "He had not cried at the funeral, nor at the wake, but he would tear "
        "the letters to pieces before letting anyone read them.",
        0,
    ),
    (
        "tear",
        "tear_nou",
        "misleading_ngram",
        "The riot police had used gas and batons for hours, and not one "
        "reporter shed a single tear over the arrests.",
        0,
    ),
    # --- wind -------------------------------------------------------------
    (
        "wind",
        "wind_vrb",
        "semantic_field",
        "After the solar panels failed and the turbines stopped, the board "
        "voted to wind the project down.",
        0,
    ),
    (
        "wind",
        "wind_nou",
        "garden_path",
        "They had watched the clock wind down all afternoon before the wind "
        "finally rose off the water.",
        1,
    ),
    # --- present ----------------------------------------------------------
    (
        "present",
        "present_vrb",
        "misleading_ngram",
        "Nobody had wrapped a single gift, so the committee asked her to "
        "present the findings instead.",
        0,
    ),
    (
        "present",
        "present_adj-nou",
        "garden_path",
        "Smith has played matches for the county from 1993 to present, a run "
        "nobody expected to last.",
        0,
    ),
    # --- record -----------------------------------------------------------
    (
        "record",
        "record_vrb",
        "semantic_field",
        "The vinyl sleeves and the turntable stayed boxed up while the band "
        "went out to record the album live.",
        0,
    ),
    (
        "record",
        "record_nou",
        "garden_path",
        "They will record the session tonight, but the record they cut last "
        "spring still sells better.",
        1,
    ),
    # --- object -----------------------------------------------------------
    (
        "object",
        "object_vrb",
        "misleading_ngram",
        "The museum had catalogued each object carefully, so the curators "
        "were startled to hear the donor object to the display.",
        1,
    ),
    (
        "object",
        "object_nou",
        "garden_path",
        "Lawyers who habitually object to every question were shown a small "
        "brass object recovered from the site.",
        1,
    ),
    # --- minute -----------------------------------------------------------
    (
        "minute",
        "minute_adj",
        "misleading_ngram",
        "The talk was scheduled for exactly one hour, and she spent it on the "
        "minute differences between the two shells.",
        0,
    ),
    (
        "minute",
        "minute",
        "semantic_field",
        "The gastropod specimens were almost invisible, so the survey allowed "
        "a full minute per tray.",
        0,
    ),
    # --- desert -----------------------------------------------------------
    (
        "desert",
        "desert_vrb",
        "semantic_field",
        "The convoy had crossed the Sahara without incident, which made it "
        "stranger still that two drivers chose to desert.",
        0,
    ),
    (
        "desert",
        "desert_nou",
        "garden_path",
        "Soldiers who desert are rarely caught, but the patrol that vanished "
        "was swallowed by the desert.",
        1,
    ),
    # --- contract ---------------------------------------------------------
    (
        "contract",
        "contract_vrb",
        "misleading_ngram",
        "She had signed the contract in March, and by June she would contract "
        "the illness that ended her tour.",
        1,
    ),
    (
        "contract",
        "contract_nou",
        "garden_path",
        "Metals contract in the cold, a fact the engineers wrote into the "
        "contract before anyone signed.",
        1,
    ),
    # --- console ----------------------------------------------------------
    (
        "console",
        "console_vrb",
        "semantic_field",
        "The organ's keyboards and pedals sat silent all evening while the "
        "choirmaster tried to console the family.",
        0,
    ),
    # --- refuse -----------------------------------------------------------
    (
        "refuse",
        "refuse_nou",
        "garden_path",
        "Residents who refuse to sort their waste leave the refuse piled at "
        "the kerb.",
        1,
    ),
    # --- invalid ----------------------------------------------------------
    (
        "invalid",
        "invalid_nou",
        "misleading_ngram",
        "The clerk stamped the form invalid, then wheeled the elderly invalid "
        "back toward the ward.",
        1,
    ),
    # --- entrance ---------------------------------------------------------
    (
        "entrance",
        "entrance_vrb",
        "misleading_ngram",
        "Guests filed past the marble entrance hall, where the singer would "
        "entrance them for the better part of an hour.",
        1,
    ),
]


def _byte_span(sentence: str, homograph: str, occurrence: int) -> tuple[int, int]:
    """Byte offsets of the nth standalone occurrence of ``homograph``."""
    lowered = sentence.lower()
    target = homograph.lower()
    found, cursor = -1, 0
    for _ in range(occurrence + 1):
        while True:
            found = lowered.find(target, cursor)
            if found < 0:
                raise ValueError(f"{homograph!r} not found in {sentence!r}")
            before_ok = found == 0 or not lowered[found - 1].isalpha()
            after = found + len(target)
            after_ok = after == len(lowered) or not lowered[after].isalpha()
            cursor = found + 1
            if before_ok and after_ok:
                break
    start = len(sentence[:found].encode("utf-8"))
    end = start + len(sentence[found : found + len(target)].encode("utf-8"))
    return start, end


def main() -> None:
    rows, categories = [], []
    for homograph, wordid, category, sentence, occurrence in SENTENCES:
        start, end = _byte_span(sentence, homograph, occurrence)
        recovered = sentence.encode("utf-8")[start:end].decode("utf-8")
        if recovered.lower() != homograph.lower():
            raise ValueError(f"span recovered {recovered!r} for {homograph!r}")
        rows.append([homograph, wordid, sentence, start, end])
        categories.append([homograph, wordid, category, sentence])

    HERE.mkdir(exist_ok=True)
    with (HERE / "adversarial.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow(["homograph", "wordid", "sentence", "start", "end"])
        writer.writerows(rows)

    with (HERE / "categories.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow(["homograph", "wordid", "category", "sentence"])
        writer.writerows(categories)

    homographs = {row[0] for row in rows}
    print(f"wrote {len(rows)} sentences over {len(homographs)} homographs")


if __name__ == "__main__":
    main()
