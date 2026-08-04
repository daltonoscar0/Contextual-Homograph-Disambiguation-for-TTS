# TODO

Nothing here blocked milestone 1 or 2; both are complete and every number in
the README regenerates from `make all`. These are the things I would do next.

## Worth doing

- **Hybrid system.** The paper's best result (0.990) comes from letting rules
  pre-empt the classifier. Our two systems fail in largely disjoint places —
  the probe wins on syntax, the baseline on topical traps — so a hybrid is the
  obvious next step and is likely to close the remaining 0.4 points.
- **Report a confidence interval on the eval numbers.** 1,615 sentences puts
  roughly ±0.6 points on a 0.986 micro accuracy; the README currently states
  point estimates and compares them to the paper's point estimates without
  either. The comparison to `Server: ML` (0.954) is comfortably outside that
  interval, but the comparison to the hybrid (0.990) is not.
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
