# TODO

Nothing here blocked milestone 1 or 2; both are complete and every number in
the README regenerates from `make all`. These are the things I would do next.

## Worth doing

- **Separating from 0.990 statistically.** roberta-large with balanced class
  weights makes 13 errors against the paper's ~16, but the interval still
  contains their number. Clearing it needs ~8 errors, and five of the current
  13 are on labels with zero or one training example, so the reachable floor is
  around 5-7. Untried: deberta-v3-large, pooling more than four layers.
- **Hybrid system — tried, and it failed.** Letting the POS rule pre-empt the
  probe scores 0.962, *far worse* than the probe alone. The paper's hybrid
  works because its rules are hand-curated and high-precision; ours are learned
  majority-vote rules at 0.954, so overriding a 0.990 classifier with them
  trades good answers for bad. Adding the POS tag as a probe *feature* instead
  is worth +1 sentence. An oracle that always picked the better of the two
  systems would score 0.993, so the information is there — but no
  confidence-based routing I tried recovered more than one of the eleven
  sentences the rule gets right and the probe does not.
- **Grow the adversarial set.** 30 sentences is enough to show the failure
  modes exist and not enough to size them. Category-level n is 3–11; the
  `long_distance` row rests on three sentences.
- **Second annotator on the adversarial gold labels.** They are currently one
  person's judgments, checked against the corpus conventions but not
  independently verified.

## Known limitations, not planned

- 18 of 162 homographs have a single label across their entire training split,
  so no system fit on this data can ever produce their minority reading. This
  is a property of the corpus. It is documented in `adversarial/ANALYSIS.md`
  and is why that file reports a winnable-subset number alongside the raw one.
- Sentences longer than 256 wordpieces are truncated. No eval or adversarial
  row actually triggers the fallback, but the limit is real.
- The POS baseline inherits every spaCy tagging error. That is the point — it
  is what makes it a fair stand-in for a rule system — but it means the
  baseline number moves if the spaCy version changes.
