"""Accuracy computation and the generated results tables.

Micro accuracy is the fraction of correctly classified eval examples. Macro
accuracy is the mean of the per-homograph accuracies, which is the definition
Gorman et al. use ("mean average accuracy") and which weights the 162
homographs equally regardless of how many eval sentences each contributes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from lede.data import Example, REPO_ROOT, group_by_homograph, load_wordids

RESULTS_DIR = REPO_ROOT / "results"
PAPER_NUMBERS = RESULTS_DIR / "paper_numbers.csv"


@dataclass
class Scores:
    micro: float
    macro: float
    per_homograph: dict[str, float]
    n_per_homograph: dict[str, int]
    micro_ci: tuple[float, float] = (0.0, 0.0)
    macro_ci: tuple[float, float] = (0.0, 0.0)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because accuracies here sit
    near 1.0, where the normal interval is badly skewed and can exceed 1.
    """
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def _macro_interval(
    per_homograph: dict[str, float], seed: int = 0, draws: int = 10000
) -> tuple[float, float]:
    """Percentile bootstrap over homographs for the macro average.

    Macro accuracy is a mean over 162 per-homograph rates, so its sampling
    variation comes from which homographs are in the set, not from Bernoulli
    trials; resampling homographs is the matching procedure.
    """
    import random

    values = list(per_homograph.values())
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(draws)
    )
    return (means[int(0.025 * draws)], means[int(0.975 * draws)])


def score(predictions: list[str], examples: list[Example]) -> Scores:
    if len(predictions) != len(examples):
        raise ValueError(f"{len(predictions)} predictions for {len(examples)} examples")

    correct_by_homograph: dict[str, int] = {}
    total_by_homograph: dict[str, int] = {}
    for prediction, example in zip(predictions, examples):
        total_by_homograph[example.homograph] = (
            total_by_homograph.get(example.homograph, 0) + 1
        )
        correct_by_homograph[example.homograph] = correct_by_homograph.get(
            example.homograph, 0
        ) + int(prediction == example.wordid)

    per_homograph = {
        homograph: correct_by_homograph[homograph] / total
        for homograph, total in total_by_homograph.items()
    }
    correct = sum(correct_by_homograph.values())
    return Scores(
        micro=correct / len(examples),
        macro=sum(per_homograph.values()) / len(per_homograph),
        per_homograph=per_homograph,
        n_per_homograph=total_by_homograph,
        micro_ci=wilson_interval(correct, len(examples)),
        macro_ci=_macro_interval(per_homograph),
    )


def load_paper_numbers(path: Path = PAPER_NUMBERS) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paper_headline(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """The eval-set rows we compare against, keyed by system name."""
    return {
        row["system"]: {"micro": float(row["micro"]), "macro": float(row["macro"])}
        for row in rows
        if row["scope"] == "evaluation set"
    }


def per_homograph_table(
    examples: list[Example],
    baseline: Scores,
    probe: Scores,
    mle: Scores,
) -> list[dict[str, object]]:
    """One row per homograph, sorted by how far the probe trails the baseline."""
    wordids = load_wordids()
    types = {entry.homograph: entry.homograph_type for entry in wordids.values()}

    rows = []
    for homograph in sorted(group_by_homograph(examples)):
        rows.append(
            {
                "homograph": homograph,
                "homograph_type": types.get(homograph, ""),
                "n_eval": baseline.n_per_homograph[homograph],
                "mle_acc": round(mle.per_homograph[homograph], 4),
                "pos_baseline_acc": round(baseline.per_homograph[homograph], 4),
                "probe_acc": round(probe.per_homograph[homograph], 4),
                "probe_minus_baseline": round(
                    probe.per_homograph[homograph]
                    - baseline.per_homograph[homograph],
                    4,
                ),
            }
        )
    return rows


def _suffix(encoder: str) -> str:
    """Default encoder writes the canonical filenames; others get a suffix."""
    return "" if encoder == "bert-base-cased" else f"_{encoder.replace('/', '_')}"


def write_per_homograph(
    rows: list[dict[str, object]], encoder: str = "bert-base-cased"
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fields = list(rows[0])
    sfx = _suffix(encoder)

    csv_path = RESULTS_DIR / f"per_homograph{sfx}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        "| homograph | type | n_eval | MLE | POS baseline | probe | probe - baseline |"
    )
    lines = [
        "# Per-homograph accuracy on the evaluation split",
        "",
        "Gorman et al. (2018) report only micro- and macro-averaged accuracy, so",
        "there is no per-homograph column to compare against here; see",
        "`paper_numbers.csv` and the README for the aggregate comparison.",
        "",
        header,
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['homograph']} | {row['homograph_type']} | {row['n_eval']} | "
            f"{row['mle_acc']:.3f} | {row['pos_baseline_acc']:.3f} | "
            f"{row['probe_acc']:.3f} | {row['probe_minus_baseline']:+.3f} |"
        )
    (RESULTS_DIR / f"per_homograph{sfx}.md").write_text("\n".join(lines) + "\n")


def write_summary(
    mle: Scores,
    baseline: Scores,
    probe: Scores,
    mode: str,
    rows: list[dict],
    encoder: str = "bert-base-cased",
) -> str:
    """Write results/summary.md and return it, for pasting into the README."""
    paper = paper_headline(load_paper_numbers())

    def ours(label: str, s: Scores) -> str:
        return (
            f"| {label} | {s.micro:.3f} | "
            f"[{s.micro_ci[0]:.3f}, {s.micro_ci[1]:.3f}] | "
            f"{s.macro:.3f} | [{s.macro_ci[0]:.3f}, {s.macro_ci[1]:.3f}] |"
        )

    lines = [
        "# Summary",
        "",
        "## Evaluation-split accuracy",
        "",
        "95% intervals: Wilson score for micro, percentile bootstrap over the",
        "162 homographs for macro. The paper reports point estimates only, so",
        "its rows have no interval; they are single numbers on n=1615 and carry",
        "comparable uncertainty.",
        "",
        "| system | micro | 95% CI | macro | 95% CI |",
        "|---|---:|---:|---:|---:|",
        ours("MLE baseline (ours)", mle),
        ours("POS-rule baseline (ours)", baseline),
        ours(f"Frozen probe (ours, {encoder}, {mode})", probe),
    ]
    for system in (
        "Embedded: rules",
        "Server: rules",
        "Embedded: ML",
        "Server: ML",
        "Server: rules + ML",
    ):
        entry = paper[system]
        lines.append(
            f"| {system} (Gorman et al. 2018) | {entry['micro']:.3f} | — | "
            f"{entry['macro']:.3f} | — |"
        )

    worst = sorted(rows, key=lambda r: (r["probe_minus_baseline"], r["homograph"]))[:10]
    lines += [
        "",
        "## Ten homographs where the probe most underperforms the POS baseline",
        "",
        "| homograph | type | n_eval | POS baseline | probe | delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in worst:
        lines.append(
            f"| {row['homograph']} | {row['homograph_type']} | {row['n_eval']} | "
            f"{row['pos_baseline_acc']:.3f} | {row['probe_acc']:.3f} | "
            f"{row['probe_minus_baseline']:+.3f} |"
        )

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"summary{_suffix(encoder)}.md").write_text(text)
    return text


def error_dump(
    examples: list[Example],
    baseline: list[str],
    probe: list[str],
    encoder: str = "bert-base-cased",
) -> None:
    """Every eval example either system got wrong, for error analysis."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"errors{_suffix(encoder)}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["homograph", "gold", "baseline_pred", "probe_pred", "surface", "sentence"]
        )
        for example, base, prb in zip(examples, baseline, probe):
            if base != example.wordid or prb != example.wordid:
                writer.writerow(
                    [
                        example.homograph,
                        example.wordid,
                        base,
                        prb,
                        example.surface,
                        example.sentence,
                    ]
                )
