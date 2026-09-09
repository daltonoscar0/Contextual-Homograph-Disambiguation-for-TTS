"""Run both systems over the hand-written adversarial set.

Nothing is refit here: the POS rules and the per-homograph probes are the ones
learned from the training split, applied unchanged to sentences designed to
mislead them.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from lede import baseline_pos, probe
from lede.data import REPO_ROOT, load_split, load_tsv

ADVERSARIAL_DIR = REPO_ROOT / "adversarial"
TSV_PATH = ADVERSARIAL_DIR / "adversarial.tsv"
CATEGORIES_PATH = ADVERSARIAL_DIR / "categories.tsv"


def load_categories() -> dict[str, str]:
    """Map sentence text -> trap category."""
    with CATEGORIES_PATH.open(encoding="utf-8", newline="") as handle:
        return {
            row["sentence"]: row["category"]
            for row in csv.DictReader(handle, delimiter="\t", quotechar='"')
        }


def run() -> dict:
    if not TSV_PATH.exists():
        raise SystemExit(f"{TSV_PATH} missing; run python adversarial/build.py")

    examples = load_tsv(TSV_PATH, "adversarial")
    categories = load_categories()
    train = load_split("train")

    # Baseline: reuse the rules fit on train, tag the new sentences.
    train_tags = baseline_pos.tag_examples(train, "train")
    model = baseline_pos.fit(train, train_tags)
    adversarial_tags = baseline_pos.tag_examples(examples, "adversarial")
    baseline_predictions = baseline_pos.predict(model, examples, adversarial_tags)

    # Probe: reuse the cached train embeddings and the selected configuration.
    # Both the layer mode *and* the class weighting have to come from the
    # cached choice. Refitting without the class weighting here would score a
    # different model than the one the headline number describes, which is
    # exactly the comparison this file exists to make.
    import json

    choice = json.loads(probe.CHOICE_PATH.read_text())
    mode, class_weight = choice["mode"], choice["class_weight"]
    train_embeddings = probe.extract("train", train)[mode]
    adversarial_embeddings = probe.extract("adversarial", examples)[mode]
    probes = probe.fit_probes(train, train_embeddings.vectors, class_weight=class_weight)
    probe_predictions = probe.predict_probes(
        probes, examples, adversarial_embeddings.vectors
    )

    rows = []
    for example, base, prb, tag in zip(
        examples, baseline_predictions, probe_predictions, adversarial_tags
    ):
        rows.append(
            {
                "homograph": example.homograph,
                "category": categories.get(example.sentence, "?"),
                "gold": example.wordid,
                "pos_tag": tag,
                "baseline_pred": base,
                "probe_pred": prb,
                "baseline_ok": int(base == example.wordid),
                "probe_ok": int(prb == example.wordid),
                "sentence": example.sentence,
            }
        )

    write_predictions(rows, probe.MODEL_NAME)
    print(f"adversarial: encoder {probe.MODEL_NAME} ({mode}, class_weight {class_weight!r})")
    report(rows)
    return {"rows": rows}


def _suffix(encoder: str) -> str:
    """Match ``lede.evaluate``: the default encoder owns the bare filename."""
    return "" if encoder == "bert-base-cased" else f"_{encoder.replace('/', '_')}"


def write_predictions(rows: list[dict], encoder: str = "bert-base-cased") -> None:
    path = ADVERSARIAL_DIR / f"predictions{_suffix(encoder)}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict]) -> None:
    n = len(rows)
    baseline_micro = sum(r["baseline_ok"] for r in rows) / n
    probe_micro = sum(r["probe_ok"] for r in rows) / n
    print(f"adversarial: {n} sentences")
    print(f"adversarial: POS baseline micro {baseline_micro:.4f}")
    print(f"adversarial: probe        micro {probe_micro:.4f}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    print("\nby category:")
    print(f"{'category':<18}{'n':>4}{'baseline':>10}{'probe':>8}")
    for category in sorted(by_category):
        group = by_category[category]
        base = sum(r["baseline_ok"] for r in group) / len(group)
        prb = sum(r["probe_ok"] for r in group) / len(group)
        print(f"{category:<18}{len(group):>4}{base:>10.3f}{prb:>8.3f}")


if __name__ == "__main__":
    run()
