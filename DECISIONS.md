# Decisions

Non-obvious choices made while building this, one line each.

## Data

- Parsed the TSVs with `csv.DictReader(delimiter="\t", quotechar='"')` rather than `str.split("\t")`, because every field in the upstream files is quoted and some sentences contain embedded quotes.
- Treated `start`/`end` as UTF-8 **byte** offsets and converted to character offsets once, in the loader; 102 of 16,102 rows contain non-ASCII text before the target, so naive character indexing silently mislabels those spans.
- Made byte-span conversion raise rather than clamp when an endpoint lands mid-character, so a malformed row fails loudly instead of producing an off-by-one target.
- `check_offsets` accepts a span whose surface differs from the `homograph` key by case or a short suffix (`uses` for `use`, `lived` for `live`), because the data set keys inflected occurrences by a stem-like form; exact string equality would flag ~1,400 correct rows.
- Fixed load order to (filename, row index) so caches, classifier fits, and results are byte-identical across runs.
- Used the shipped 90/10 train/eval split as-is; nothing is reshuffled and eval is never touched during model selection.
- CMUdict is supporting data only: tests assert it covers the homographs and that known homographs carry multiple pronunciations, but no model reads it.
- `wordids.tsv` spells its last column `fine_homograph_type` while the README calls it `fine_homography_type`; the loader accepts either.

## POS baseline

- Reconstructed the rule system empirically (learn tag → majority wordid from train) rather than hand-writing rules, so the baseline is reproducible and not tuned by hand against eval.
- Used spaCy's coarse `pos_` rather than the fine-grained `tag_`; the fine tags split noun/verb into inflection classes and fragment the already-small per-homograph training counts.
- Excluded the parser, NER, and lemmatizer from the spaCy pipeline — only the tagger matters here, and dropping them cuts tagging time substantially.
- Aligned the target to a spaCy token by maximum character overlap, falling back from `doc.char_span`, because the tagger occasionally splits around hyphens and clitics where a strict span lookup returns `None`.
- Broke majority-vote ties alphabetically by wordid so the fitted rules are deterministic.
- Cached POS tags to `cache/pos_tags.pkl` keyed by split, since tagging 16k sentences takes about half a minute and is pure overhead on re-runs.

## Probe

- Pooled wordpieces by **character-offset overlap** with the target span rather than by re-tokenizing the homograph string, which is what keeps subword splits (`ba ##ss`) correct; a 400-row spot check reconstructs the exact surface form for every sampled example.
- Ran one forward pass and derived both layer representations from it, so comparing `final` against `last4` costs no extra BERT compute.
- Selected the layer mode on a 20% train-internal validation split: `last4` scored 0.9834 against `final`'s 0.9800, so the reported run concatenates the last four layers (3072-d).
- Tuned `C` over {0.1, 1, 10} per homograph on a per-homograph 20% internal split, falling back to `C=1` when a homograph has fewer than 10 training rows or its inner split collapses to one label.
- Standardized features before the logistic regression; per-homograph training sets are ~90 rows against 3072 dimensions, and unscaled BERT activations make the solver's convergence erratic.
- Homographs whose training rows carry a single wordid become constant predictors rather than classifiers, since `LogisticRegression` cannot fit one class.
- Suppressed numpy FP flags narrowly around solver matmuls: Apple's Accelerate BLAS raises spurious divide-by-zero/overflow warnings on both float32 and float64, and all 144 fitted models return finite coefficients. `_fit_one` asserts finiteness so a real numerical failure would still surface.
- Truncated at 256 wordpieces and mean-pooled the sentence as a fallback when a target falls outside the window; the fallback flag is recorded and no eval row actually triggers it.
- Cached embeddings as compressed `.npz` per (split, layer mode), so refitting every probe takes seconds.

## Evaluation

- Defined macro accuracy as the mean of per-homograph accuracies, matching the paper's "mean average accuracy".
- Transcribed the paper's numbers into `results/paper_numbers.csv` with the table and scope each came from; Table 2 covers the entire data set and Table 3 the evaluation split, and only Table 3 is comparable to our numbers.
- Compared at micro/macro level only: Gorman et al. report no per-homograph accuracies, so `per_homograph.md` has no paper column and says so.
- Added the MLE (most-frequent-wordid) baseline because the paper reports one, and it anchors how much work context is actually doing.
- Made `run_all` exit non-zero if the probe fails to beat the baseline on micro accuracy, so an alignment regression cannot pass silently.

- Wrote all 30 sentences by hand rather than mining Wikipedia, since the point is contexts where local cues actively mislead — those are rare in naturally occurring text.
- Reused the dataset's exact schema and byte-offset convention so the same loader and the same two systems run unmodified.
- Assigned each sentence one of four trap categories, recorded in a sidecar TSV, so accuracy can be broken down by failure mode rather than reported as one number.
