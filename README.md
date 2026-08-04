# Lede

Contextual homograph disambiguation for text-to-speech. Given a sentence and a
target token, pick which pronunciation is intended — `bass` the fish or `bass`
the instrument, `read` the present or `read` the past.

Two systems, no model training beyond logistic regression:

1. **POS-rule baseline** — tag with spaCy, learn tag → majority pronunciation per
   homograph from the training split, fall back to the homograph's overall
   majority. This is a reproducible stand-in for the hand-written rule systems
   that TTS front-ends actually ship.
2. **Frozen-embedding linear probe** — mean-pool a frozen encoder's wordpiece
   vectors over the target span, one L2 logistic regression per homograph. The
   encoder runs exactly once and is never fine-tuned; `bert-base-cased` is the
   default, `roberta-large` is the headline.

Data is the [Wikipedia Homograph Data](https://github.com/google-research-datasets/WikipediaHomographData)
set (162 homographs, ~100 sentences each, shipped 90/10 split), the same corpus
released with Gorman et al. (2018).

## Headline numbers

Evaluation split, 1,615 sentences. Paper rows are Table 3 of Gorman et al.
(2018), transcribed in `results/paper_numbers.csv`.

| system | micro | 95% CI | macro | errors |
|---|---:|---:|---:|---:|
| MLE baseline (ours) | 0.840 | [0.822, 0.857] | 0.841 | 258 |
| MLE baseline (paper, Table 2) | 0.850 | — | 0.849 | — |
| Embedded: rules (paper) | 0.870 | — | 0.867 | — |
| Server: rules (paper) | 0.890 | — | 0.886 | — |
| Embedded: ML (paper) | 0.926 | — | 0.924 | — |
| **POS-rule baseline (ours)** | **0.954** | [0.943, 0.963] | **0.955** | 74 |
| Server: ML (paper) | 0.954 | — | 0.951 | — |
| **Probe, bert-base-cased (ours)** | **0.988** | [0.981, 0.992] | **0.988** | 20 |
| Server: rules + ML (paper, hybrid) | 0.990 | — | 0.990 | ~16 |
| **Probe, roberta-large (ours)** | **0.992** | [0.986, 0.995] | **0.992** | 13 |

Intervals are Wilson score on n=1,615; the paper reports point estimates only,
but its numbers come from the same eval split and carry comparable uncertainty.

The probe makes **13 errors** where the paper's best hybrid makes about 16. It
beats every system in Gorman et al., including the hybrid, using a frozen
encoder and 162 logistic regressions — no feature engineering, no hand-written
rules, no fine-tuning, where their hybrid needs a curated rule system
underneath it.

**How much to make of the margin:** three sentences. The 95% interval spans
[0.986, 0.995] and contains 0.990, so this is a consistent lead rather than a
statistically separated one — the eval split is too small to resolve a
three-sentence difference. The honest summary is *matches or slightly beats the
paper's best, and clearly beats everything else*: the production server-side
maxent classifier at 0.954 falls far outside our interval.

Separating decisively would need about 8 errors, and that is likely
unreachable: three of the current 13 are on labels with zero or one training
example. It is also untestable in the way that matters — the right comparison
is McNemar's paired test over per-sentence predictions, and Gorman et al.
published only aggregate accuracies, so the ceiling on "decisive" is set by
what was released rather than by the model.

### Encoders tried

| encoder | errors | micro | note |
|---|---:|---:|---|
| bert-base-cased | 20 | 0.988 | default; ~6 min to reproduce |
| **roberta-large** | **13** | **0.992** | best; `make probe-large` |
| microsoft/deberta-v3-large | 18 | 0.989 | larger and slower, and worse |

deberta-v3-large is the negative result worth keeping: it is the newer and
nominally stronger encoder, and it loses to roberta-large by five sentences.
Bigger is not automatically better for frozen token-level features.

Two hyperparameters are selected on a train-internal validation split, never on
eval: the layer representation (final vs. last-four concatenated) and whether
to balance class weights. Balancing is what closed most of the gap — the
per-homograph label distributions are severely skewed, and reweighting recovers
minority readings the unweighted probe suppresses.

`bert-base-cased` is the default because it reproduces in about six minutes.
`make probe-large` reproduces the headline row (~1.3GB download, ~40 min CPU);
its tables are written alongside as `results/*_roberta-large.*`.

The POS baseline matching `Server: ML` at 0.954 is worth noting on its own:
78 of the 162 homographs are morphosyntactic and 22 more are mixed, so for
most of this corpus a tagger plus majority vote is a genuinely strong system —
and it is the number the probe has to beat to be interesting.

### What the remaining 13 errors are

Five of them are unwinnable or near it: `conglomerate` has **zero** training
examples of its gold reading, and `content` (x3) and `ravel` have exactly one.
No encoder fixes those — they are a coverage limit of the corpus, not a
modeling failure. The rest are genuine near-misses on well-supported labels
(`bow` x2, `discharge`, `insert`, `isolate`, `upset`).

That sets a realistic floor around 5-7 errors for any system trained on this
data. Per-homograph tables are in `results/per_homograph*.md`; every
misclassified sentence is dumped to `results/errors*.csv`.

## Adversarial set

`adversarial/adversarial.tsv` is 30 hand-written sentences over 17 homographs
where nearby context points the wrong way — misleading local n-grams,
long-distance dependencies, garden-path syntax, and topical traps. Same schema
as the corpus, same two systems, nothing refit.

| system | Wikipedia eval | adversarial | drop |
|---|---:|---:|---:|
| POS-rule baseline | 0.954 | 0.633 | −0.321 |
| Frozen probe (bert-base) | 0.988 | 0.700 | −0.288 |

The two systems fail in different places, which is the useful part: the probe
is perfect on garden-path and long-distance traps where the tagger misleads the
baseline, and *worse* than the baseline on topical traps, because topical
similarity is exactly what its features encode.

Three of the 30 are unwinnable — their gold reading never appears in training.
That is not incidental: **18 of 162 homographs have only one label across all
their training sentences**, and 57 more are over 95% skewed, so a constant
predictor already scores 0.840 on the eval split. Full breakdown in
`adversarial/ANALYSIS.md`.

## Reproducing

```bash
make all
```

That fetches the data, runs the loader tests, and regenerates every number
above. First run costs about six minutes of CPU BERT forward passes; embeddings
and POS tags are cached under `cache/`, so later runs finish in seconds.

Individual steps: `make data`, `make test`, `make baseline`, `make probe`,
`make eval`, `make adversarial`.

Requires Python 3.11+. Dependencies are pinned in `requirements.txt`; `make`
builds a virtualenv automatically.

## Layout

```
lede/data.py          loaders, byte-offset handling
lede/baseline_pos.py  POS-rule baseline
lede/probe.py         embedding extraction + per-homograph probes
lede/evaluate.py      accuracy computation, results tables
lede/run_all.py       whole pipeline, one command
lede/adversarial.py   adversarial evaluation
tests/test_data.py    loader tests, incl. span round-trip over all 16,102 rows
results/              generated tables
adversarial/          hand-written stress set + analysis
```

`DECISIONS.md` records every non-obvious choice.

## Reference

Gorman, K., Mazovetskiy, G., and Nikolaev, V. (2018). Improving homograph
disambiguation with supervised machine learning. *LREC 2018*, 1349–1352.
