# Lede

Contextual homograph disambiguation for text-to-speech. Given a sentence and a
target token, pick which pronunciation is intended — `bass` the fish or `bass`
the instrument, `read` the present or `read` the past.

Two systems, no model training beyond logistic regression:

1. **POS-rule baseline** — tag with spaCy, learn tag → majority pronunciation per
   homograph from the training split, fall back to the homograph's overall
   majority. This is a reproducible stand-in for the hand-written rule systems
   that TTS front-ends actually ship.
2. **Frozen-embedding linear probe** — mean-pool `bert-base-cased` wordpiece
   vectors over the target span, one L2 logistic regression per homograph.
   BERT is frozen and run exactly once; nothing is fine-tuned.

Data is the [Wikipedia Homograph Data](https://github.com/google-research-datasets/WikipediaHomographData)
set (162 homographs, ~100 sentences each, shipped 90/10 split), the same corpus
released with Gorman et al. (2018).

## Headline numbers

Evaluation split, 1,615 sentences. Paper rows are Table 3 of Gorman et al.
(2018), transcribed in `results/paper_numbers.csv`.

| system | micro | macro |
|---|---:|---:|
| MLE baseline (ours) | 0.840 | 0.841 |
| MLE baseline (paper, Table 2) | 0.850 | 0.849 |
| Embedded: rules (paper) | 0.870 | 0.867 |
| Server: rules (paper) | 0.890 | 0.886 |
| Embedded: ML (paper) | 0.926 | 0.924 |
| **POS-rule baseline (ours)** | **0.954** | **0.955** |
| Server: ML (paper) | 0.954 | 0.951 |
| **Frozen BERT probe (ours)** | **0.986** | **0.986** |
| Server: rules + ML (paper, hybrid) | 0.990 | 0.990 |

The probe beats every non-hybrid system in the paper, including the
production server-side maxent classifier, and lands 0.4 points below the
hybrid rules+ML system that the paper reports as its best. It does so with a
frozen encoder and 162 logistic regressions — no feature engineering, no
hand-written rules, no fine-tuning.

The POS baseline matching `Server: ML` at 0.954 is worth noting on its own:
78 of the 162 homographs are morphosyntactic and 22 more are mixed, so for
most of this corpus a tagger plus majority vote is a genuinely strong system —
and it is the number the probe has to beat to be interesting.

### Where the probe loses to the baseline

Only eight homographs, each by one or two eval sentences:

| homograph | type | n_eval | POS baseline | probe | delta |
|---|---|---:|---:|---:|---:|
| graduate | Lexical/Morphosyntactic | 10 | 1.000 | 0.800 | -0.200 |
| compress | Lexical | 10 | 1.000 | 0.900 | -0.100 |
| discharge | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| escort | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| incline | Lexical | 10 | 1.000 | 0.900 | -0.100 |
| invite | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| minute | Lexical | 10 | 1.000 | 0.900 | -0.100 |
| perfume | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |

These are cases where syntax alone settles the question and the probe's
semantic signal adds noise the rule does not have. Full table in
`results/per_homograph.md`; every misclassified sentence is dumped to
`results/errors.csv`.

## Adversarial set

`adversarial/adversarial.tsv` is 30 hand-written sentences over 17 homographs
where nearby context points the wrong way — misleading local n-grams,
long-distance dependencies, garden-path syntax, and topical traps. Same schema
as the corpus, same two systems, nothing refit.

| system | Wikipedia eval | adversarial | drop |
|---|---:|---:|---:|
| POS-rule baseline | 0.954 | 0.633 | −0.321 |
| Frozen BERT probe | 0.986 | 0.700 | −0.286 |

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
