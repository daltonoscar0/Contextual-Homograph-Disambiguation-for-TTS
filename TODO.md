# TODO

Milestones 1 and 2 (the probe and the evaluation) and milestone 3 (the
component: full G2P, the contract API, serving-sized backbones) are complete.
Every number in the README regenerates from `make all` plus `make probe-serving`
and `make probe-large`. These are the things I would do next.

## Worth doing

- **Function-word reduction.** Every non-heteronym token takes CMUdict's first
  pronunciation, which is the stressed citation form. The canonical sentence
  comes back with `at` as `AE1 T` where connected speech wants `AH0 T`. This
  affects far more tokens than the homograph problem does, and it needs a
  lexicon with reduced variants plus a rule for when to use them, not a model.
  It is the largest remaining gap between this and a usable front end.
- **A real OOV path.** `<oov:TOKEN>` is honest but it is not a pronunciation.
  A neural G2P was deliberately left out of scope; it is the obvious next
  component, and it should be its own module with its own eval, not bolted on
  here.
- **Targeted sentences for minority readings.** 18 of 162 homographs have a
  single label across their entire training split and 57 more are over 95%
  skewed. This is the single highest-value piece of work available and it is
  annotation, not modelling. It caps both the eval errors (3 of 13) and the
  adversarial set (3 of 5).
- **Second annotator on the eval split, not just the adversarial set.** Two of
  the 13 roberta-large errors (`approximate`, `discharge`) look like gold
  errors, where the model's reading is the better one. That puts a practical
  ceiling near 11 errors, and nobody can locate the real ceiling without
  re-adjudication. The adversarial gold labels are also still one person's
  judgments, checked against the corpus conventions but not independently
  verified.
- **Wider context for headline and UI-string inputs.** Three of the 13 errors
  are targets in a button label, a headline, and a truncated teaser, where
  there is no clause to read syntax off. Passing the preceding sentence and a
  document-type flag into the encoder is the cheap version of the fix.
- **A collocation list for lexical heteronyms.** "bow collector" and "bow
  window" are fixed phrases; a phrase list fixes those two errors more
  reliably and more cheaply than any encoder scaling, and roberta-large already
  beats deberta-v3-large, so scale is not the lever.
- **Grow the adversarial set.** 30 sentences is enough to show the failure
  modes exist and not enough to size them. Category-level n is 3 to 11; the
  `long_distance` row rests on three sentences.

## Tried and rejected

- **Separating from the paper's 0.990 statistically: not worth pursuing.**
  roberta-large makes 13 errors against the paper's ~16, but the interval still
  contains their number. Clearing it needs ~8 errors while three of the current
  13 sit on labels with zero or one training example. deberta-v3-large was the
  main untried lever and it lost (18 errors). More importantly the comparison
  cannot be made properly at all: McNemar's paired test needs per-sentence
  predictions and the paper published only aggregates. Remaining ideas, all
  low-expected-value: pooling more than four layers, a wider C grid.
- **Hybrid system: tried, and it failed.** Letting the POS rule pre-empt the
  probe scores 0.962, *far worse* than the probe alone. The paper's hybrid
  works because its rules are hand-curated and high-precision; ours are learned
  majority-vote rules at 0.954, so overriding a 0.990 classifier with them
  trades good answers for bad. Adding the POS tag as a probe *feature* instead
  is worth +1 sentence. An oracle that always picked the better of the two
  systems would score 0.993, so the information is there, but no
  confidence-based routing I tried recovered more than one of the eleven
  sentences the rule gets right and the probe does not.
- **Deriving the heteronym inventory from CMUdict.** Documented in the README
  and asserted in the tests. The mechanical filter is both far too broad (3,396
  candidates, 3,316 of them proper names and free variation) and materially
  incomplete (it misses 82 of 162). The curated list stays.

## Known limitations, not planned

- **Watch for eval overfitting.** Model selection has stayed on train-internal
  validation throughout, which is what keeps the eval number meaningful. But
  many configurations have now been scored on eval, and each look erodes its
  independence a little. Further tuning should be resisted unless it comes with
  a fresh held-out set.
- 18 of 162 homographs have a single label across their entire training split,
  so no system fit on this data can ever produce their minority reading. This
  is a property of the corpus. It is documented in `adversarial/ANALYSIS.md`
  and is why that file reports a winnable-subset number alongside the raw one.
- The heteronym inventory is 162 words. A production front end needs a much
  larger curated list, and building it is lexicography.
- Sentences longer than 256 wordpieces are truncated. No eval or adversarial
  row actually triggers the fallback, but the limit is real.
- The POS baseline inherits every spaCy tagging error. That is the point,
  and it is what makes it a fair stand-in for a rule system, but it means the
  baseline number moves if the spaCy version changes.
- `run()` runs one encoder forward pass per call with no batching across calls.
  Fine for a demo backend, wrong for throughput.
