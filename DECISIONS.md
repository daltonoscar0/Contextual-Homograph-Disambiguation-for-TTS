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

- Reconstructed the rule system empirically (learn tag -> majority wordid from train) rather than hand-writing rules, so the baseline is reproducible and not tuned by hand against eval.
- Used spaCy's coarse `pos_` rather than the fine-grained `tag_`; the fine tags split noun/verb into inflection classes and fragment the already-small per-homograph training counts.
- Excluded the parser, NER, and lemmatizer from the spaCy pipeline: only the tagger matters here, and dropping them cuts tagging time substantially.
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
- Made the encoder swappable via `LEDE_ENCODER`, keying caches, layer-mode choices, and results files by model name so two encoders' outputs cannot collide.
- Computed the wordpiece-overlap indices once per batch and reused them across layers; the old code re-ran that pure-Python loop five times per batch and it dominated extraction cost.
- Freed the hidden-state stack between batches and quartered the batch size for large encoders, after roberta-large at batch 32 drove this machine into swap and ran at 1 example/second.
- Kept `bert-base-cased` as the default because it reproduces in ~6 minutes; roberta-large is the headline result but costs a 1.3GB download and ~40 minutes of CPU, so it lives behind `make probe-large`.
- Re-selected the layer mode independently for roberta-large rather than inheriting bert-base's choice.
- Added `class_weight='balanced'` to the selection grid after noticing that most residual errors were minority readings the unweighted probe suppressed; it wins on train-internal validation for both encoders (roberta 0.9879 vs 0.9834) and cut eval errors from 16 to 13.
- Selected layer mode and class weighting jointly on the train-internal split, so the eval number is never consulted during model selection. bert-base picks `final/balanced` and roberta-large picks `last4/balanced`; the choices differ and are cached per encoder.
- Tried deberta-v3-large and kept it as a negative result: it makes 18 errors against roberta-large's 13 despite being the newer and larger model, so the encoder ranking was worth measuring rather than assuming.
- Verified each new encoder's tokenizer recovers the target surface from its offset mapping before spending compute on a full extraction; deberta-v3 uses SentencePiece and needed checking that the fast tokenizer returns usable offsets.
- Tried and rejected two ensembles, both selected against eval and both worse than roberta alone: averaging bert and roberta probabilities scores 0.988, and concatenating their features scores 0.990. Their errors are substantially disjoint (11 shared of 22 and 16), so an oracle over the pair would reach 0.993, but no combiner I tried captured it.

## Evaluation

- Defined macro accuracy as the mean of per-homograph accuracies, matching the paper's "mean average accuracy".
- Transcribed the paper's numbers into `results/paper_numbers.csv` with the table and scope each came from; Table 2 covers the entire data set and Table 3 the evaluation split, and only Table 3 is comparable to our numbers.
- Compared at micro/macro level only: Gorman et al. report no per-homograph accuracies, so `per_homograph.md` has no paper column and says so.
- Added the MLE (most-frequent-wordid) baseline because the paper reports one, and it anchors how much work context is actually doing.
- Made `run_all` exit non-zero if the probe fails to beat the baseline on micro accuracy, so an alignment regression cannot pass silently.
- Used a Wilson score interval for micro accuracy rather than the normal approximation, which is badly skewed and can exceed 1.0 at accuracies this close to the ceiling.
- Bootstrapped the macro interval by resampling homographs rather than sentences, because macro accuracy is a mean over 162 per-homograph rates and its variation comes from set membership, not Bernoulli trials.
- Seeded the bootstrap so the reported interval is stable across runs.
- Left the paper's rows without intervals rather than reconstructing them: Gorman et al. publish point estimates only, and inventing an interval for their systems would misrepresent what they reported.

## Adversarial set

- Wrote all 30 sentences by hand rather than mining Wikipedia, since the point is contexts where local cues actively mislead, and those are rare in naturally occurring text.
- Reused the dataset's exact schema and byte-offset convention so the same loader and the same two systems run unmodified.
- Assigned each sentence one of four trap categories, recorded in a sidecar TSV, so accuracy can be broken down by failure mode rather than reported as one number.
- Computed the byte offsets in `adversarial/build.py` instead of typing them, and asserted each span recovers its homograph, so a reworded sentence cannot silently desync from its offsets.
- Reported accuracy both overall and excluding the three sentences whose gold label never occurs in training, since no system fit on this data could produce those labels.
- Made the Makefile pick the newest Python 3.11+ on PATH rather than bare `python3`; a fresh-clone test failed because the default `python3` was 3.9 and could not resolve the pinned torch.
- Fixed `lede/adversarial.py` refitting the probes *without* the selected `class_weight`, so it had been scoring a different model than the headline number describes. With the selected configuration roberta-large scores 0.833 rather than the 0.700 previously reported for bert-base.
- Keyed `adversarial/predictions*.csv` by encoder, matching `results/`, so two encoders' adversarial runs cannot overwrite each other.

## Pronunciations

- Made the corpus IPA in `wordids.tsv` the authority for which reading is which, and CMUdict only a style target. CMUdict lists a word's variants but not which sense each belongs to, and for 36 of the 162 homographs it does not list the second reading at all (`sake` has only `S EY1 K`; `pasty` is absent).
- Converted IPA to ARPAbet with an explicit table rather than a dependency, since the inventory in `wordids.tsv` is 42 symbols and a general converter would be more code and more risk than the mapping itself.
- Snapped a converted reading onto a CMUdict variant only within edit distance 1, so heteronyms and ordinary tokens are emitted in the same style without letting a loose match silently replace a reading.
- Compared stress-marked phones rather than bare segments when snapping. `OVERthrow`/`overTHROW` and `PERmit`/`perMIT` differ in nothing but stress placement, so a stress-insensitive comparison rated both readings identical to the same CMUdict entry and collapsed them into one pronunciation. Caught by a collision assertion, not by inspection.
- Solved the snap as an injective assignment over all matchings rather than per-reading nearest neighbour, because nearest neighbour independently maps two readings onto the same variant.
- Kept a hard guard that discards snapping entirely for a homograph if it would ever produce two identical readings, and fall back to the raw conversions, which are distinct by construction.
- Dropped four stray characters in the upstream IPA (a digit inside `ə'bjuː1səz` and `ə0'fɪˌliːˌeɪt`, a length mark on a consonant in `ˌɔɹːnə'mɛnt`); they are transcription typos and none of them changes which reading is described. Listed in `lexicon.py` rather than silently stripped.

## Full G2P and serving

- Kept the heteronym inventory as the curated Gorman set instead of deriving it from CMUdict. The mechanical definition (variants differing in stress placement or stressed-vowel quality) yields 3,396 words, of which only 80 are in the curated set and 3,316 are proper names and free variation, while missing 82 of the 162 real heteronyms. Implemented and asserted as a documented negative result; it does not drive `run()`.
- Emitted `<oov:TOKEN>` for unknown words rather than adding a neural G2P. Guessing pronunciations is a separate model with its own evaluation, and a front end that invents a pronunciation is worse than one that says it does not know.
- Accepted both shapes of `pron_hints` and `name_spans` (a `{word: pron}` mapping and a list of offset spans), because the contract document is not in this checkout and the upstream stage could send either.
- Let a `pron_hint` skip the probe entirely rather than run it and be overridden, since a forward pass whose result is discarded is pure latency.
- Folded `StandardScaler` into the logistic layer at export time (`W' = W/scale`, `b' = b - W'@mean`), so serving is one matmul against the raw embedding and needs no sklearn. Asserted equal to `predict_proba` to four places on synthetic data, since a silent disagreement here would make the served model differ from the evaluated one.
- Checked the probe weights into the repo (1.2 to 2.3 MB per encoder) so `pip install -e .` plus one encoder download runs the stage, with no corpus fetch and no extraction step.
- Shipped the POS baseline's rules alongside them at 24 KB, so the no-encoder fallback needs no training data either. The fallback degrades twice: spaCy tag rules if spaCy is present, the homograph's majority reading if it is not.
- Made `run()` fall back rather than raise when the encoder or the weights are unavailable, and record which path ran in `meta.rule`. A front end that returns nothing is worse than one that returns the rule system's answer and says so.
- Chose `distilroberta-base` as the serving default over the headline `roberta-large`: 17 errors against 13, at 82M parameters instead of 355M and a ~330MB download instead of 1.3GB.
- Made `python -m lede --eval` score the *shipped weights* through the serving path rather than reusing the research pipeline's cached predictions, so what it reports is what the packaged stage actually does.
- Reported what the probe does on the specification's canonical sentence verbatim, including the `at` -> `AE1 T` function-word reduction bug, rather than special-casing anything to make the walkthrough look clean.
