"""Run the whole pipeline: baseline, probe, tables. One command, reproducible.

    python -m lede.run_all

Embeddings and POS tags are cached under ``cache/``, so the first run costs a
few minutes of BERT forward passes and every later run takes seconds.
"""

from __future__ import annotations

import random

import numpy as np

from lede import baseline_pos, evaluate, probe
from lede.data import check_offsets, load_split

SEED = 0


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    evaluation = load_split("eval")
    problems = check_offsets(load_split("train")) + check_offsets(evaluation)
    if problems:
        raise SystemExit(
            f"{len(problems)} rows failed the offset check, e.g. {problems[:3]}"
        )
    print(f"data: {len(evaluation)} eval rows, all spans round-trip")

    print("\n--- POS-rule baseline ---")
    baseline_predictions = baseline_pos.run()["predictions"]
    mle_predictions = baseline_pos.mle_predictions(evaluation)

    print("\n--- frozen BERT probe ---")
    probe_result = probe.run()
    probe_predictions = probe_result["predictions"]

    print("\n--- results ---")
    mle_scores = evaluate.score(mle_predictions, evaluation)
    baseline_scores = evaluate.score(baseline_predictions, evaluation)
    probe_scores = evaluate.score(probe_predictions, evaluation)

    rows = evaluate.per_homograph_table(
        evaluation, baseline_scores, probe_scores, mle_scores
    )
    evaluate.write_per_homograph(rows)
    summary = evaluate.write_summary(
        mle_scores, baseline_scores, probe_scores, probe_result["mode"], rows
    )
    evaluate.error_dump(evaluation, baseline_predictions, probe_predictions)
    print(summary)

    if probe_scores.micro < baseline_scores.micro:
        raise SystemExit(
            f"probe micro {probe_scores.micro:.4f} below baseline "
            f"{baseline_scores.micro:.4f}; check wordpiece/byte alignment"
        )
    print("wrote results/per_homograph.{csv,md}, results/summary.md, results/errors.csv")


if __name__ == "__main__":
    main()
