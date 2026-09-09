# Lede

Grapheme-to-phoneme for a TTS front end, with the homograph problem taken
seriously. Give it a normalised sentence; it gives back a pronunciation for
every token, and for each heteronym it tells you which reading it chose, what
the alternatives were, and how confident it was.

```console
$ python -m lede "I read the second Doctor Lee lead a live band on Reading Road at ten thirty"
```

```json
{
  "stage": "lede",
  "ok": true,
  "decisions": [
    {
      "token": "read", "start": 2, "end": 6,
      "homograph": "read", "wordid": "read_past",
      "pron": "R EH1 D",
      "alternatives": [
        {"wordid": "read_past",    "pron": "R EH1 D", "probability": 0.857685},
        {"wordid": "read_present", "pron": "R IY1 D", "probability": 0.142315}
      ],
      "rule": "probe:distilroberta-base"
    }
  ],
  "meta": {"model": "distilroberta-base", "oov": [], "n_decisions": 4}
}
```

(abridged: `tokens` and the other three decisions are omitted here.)

The disambiguator is a linear probe on frozen encoder embeddings: mean-pool the
wordpieces covering the target span, one L2 logistic regression per homograph.
Nothing is fine-tuned. With `roberta-large` it scores **0.992 micro** on the
Wikipedia Homograph Data evaluation split, against 0.990 for the best hybrid in
Gorman et al. (2018), and it does that with 162 logistic regressions and no
hand-written rules. The serving default shown above, `distilroberta-base`,
scores 0.989 at a quarter of the parameters.

## Install and use

```bash
pip install -e .
```

That plus one encoder download is enough to run the stage. The probe weights
are checked into `lede/weights/` (1.7 to 2.3 MB each), so there is no corpus
fetch and no training step in the serving path.

```python
from lede.api import run

result = run("The bass player wound the cable")
result.pronunciation
# 'DH AH0 B EY1 S P L EY1 ER0 W AW1 N D DH AH0 K EY1 B AH0 L'
result.as_dict()               # contract JSON
```

`run(text, pron_hints=None, name_spans=None)` returns a `StageResult`.
`pron_hints` overrides the dictionary for tokens the caller already knows;
`name_spans` marks proper-name regions, where a heteronym takes its
proper-noun reading. Both accept either a `{word: pron}` mapping or a list of
`{"start", "end", ...}` spans, since the upstream stage may send either.

```bash
python -m lede "text"          # JSON to stdout
python -m lede "text" --phones # just the phone string
python -m lede --canonical     # the specification sentence, with its hints
python -m lede --eval          # score the shipped weights on the eval split
```

### Choosing a backbone

`LEDE_MODEL` selects the encoder. **`distilroberta-base` is the default**,
because roberta-large is a 1.3 GB download and far too heavy for a free demo
backend. roberta-large remains the headline accuracy number and is one env var
away.

```bash
LEDE_MODEL=roberta-large python -m lede "..."
```

## What it does per token

1. An explicit `pron_hints` entry wins outright. The caller knows something we
   do not.
2. A heteronym inside a `name_spans` region takes its proper-noun reading
   (`reading_geo`, `celtic`, `ravel_nam`, and the like).
3. A heteronym is decided by the probe, in context.
4. Anything else in CMUdict takes CMUdict's first pronunciation.
5. Everything else is out of vocabulary: the token is emitted as
   `<oov:TOKEN>` and recorded in `meta.oov`.

There is deliberately no neural G2P for step 5. Guessing pronunciations for
unknown words is a separate model with its own evaluation, and a front end that
silently invents a pronunciation is worse than one that admits it does not
know.

### Which words count as heteronyms

The inventory is the 162 curated homographs from the Wikipedia Homograph Data
set. That choice is worth defending, because the obvious mechanical
alternative does not work.

If you define a heteronym as "a CMUdict entry with more than one pronunciation
differing in stress placement or in the quality of the stressed vowel", you get
**3,396** words. Only **80** of them are in the curated set. The other 3,316 are
overwhelmingly proper names and free variation (`rwanda`, `faberge`, `escrow`,
`whitehead's`), and the filter simultaneously *misses* **82 of the 162** real
heteronyms, because CMUdict lists only one pronunciation for them at all
(`sake` has `S EY1 K` and never the Japanese `S AA1 K EY2`; `pasty` is absent
entirely).

So a mechanical filter over CMUdict is both far too broad and materially
incomplete. It is implemented in `lexicon.cmudict_heteronym_candidates` and
those numbers are asserted in the tests, but it does not drive `run()`. The
curated list is the asset here, and pretending it could be derived would be
dishonest about where the work went.

### Pronunciations

`wordids.tsv` gives one IPA string per reading; CMUdict gives ARPAbet. The
corpus IPA is the authority on *which reading is which*, so it is converted to
ARPAbet and then snapped onto a CMUdict variant when one is within an edit
distance of 1, purely so heteronyms and ordinary words are emitted in the same
style.

The comparison is over stress-marked phones, not bare segments. A large class
of heteronyms (`OVERthrow`/`overTHROW`, `PERmit`/`perMIT`) differ in nothing
but stress, and a stress-insensitive match rates both readings identical to the
same CMUdict entry, collapsing them. All 326 wordids convert cleanly and no
homograph's readings collide, which the test suite asserts.

## Results

Evaluation split, 1,615 sentences. Paper rows are Table 3 of Gorman et al.
(2018), transcribed in `results/paper_numbers.csv`.

| system | micro | 95% CI | macro | errors |
|---|---:|---:|---:|---:|
| MLE baseline (ours) | 0.840 | [0.822, 0.857] | 0.841 | 258 |
| MLE baseline (paper, Table 2) | 0.850 | n/a | 0.849 | n/a |
| Embedded: rules (paper) | 0.870 | n/a | 0.867 | n/a |
| Server: rules (paper) | 0.890 | n/a | 0.886 | n/a |
| Embedded: ML (paper) | 0.926 | n/a | 0.924 | n/a |
| **POS-rule baseline (ours)** | **0.954** | [0.943, 0.963] | **0.955** | 74 |
| Server: ML (paper) | 0.954 | n/a | 0.951 | n/a |
| **Probe, distilroberta-base (ours)** | **0.989** | [0.983, 0.993] | **0.990** | 17 |
| Server: rules + ML (paper, hybrid) | 0.990 | n/a | 0.990 | ~16 |
| **Probe, roberta-base (ours)** | **0.991** | [0.986, 0.995] | **0.991** | 14 |
| **Probe, roberta-large (ours)** | **0.992** | [0.986, 0.995] | **0.992** | 13 |

### Backbones

| encoder | params | errors | micro | weights | role |
|---|---:|---:|---:|---:|---|
| **distilroberta-base** | 82M | 17 | 0.989 | 1.7 MB | **serving default** |
| roberta-base | 125M | 14 | 0.991 | 1.7 MB | middle option |
| **roberta-large** | 355M | **13** | **0.992** | 2.3 MB | **headline** |
| bert-base-cased | 108M | 20 | 0.988 | n/a | earlier default |
| microsoft/deberta-v3-large | 434M | 18 | 0.989 | n/a | negative result |

The serving story is the useful part. **distilroberta-base makes 17 errors to
roberta-large's 13**, a difference of four sentences in 1,615, and it still
beats both bert-base-cased and the much larger deberta-v3-large. Dropping from
355M parameters to 82M costs four sentences and saves about a gigabyte of
download, which is what makes a free demo backend viable at all.

deberta-v3-large is the negative result worth keeping. It is the newer and
nominally stronger encoder and it loses to roberta-large by five sentences.
Bigger is not automatically better for frozen token-level features.

Two hyperparameters are selected on a train-internal validation split, never on
eval: the layer representation (final vs. last-four concatenated) and whether
to balance class weights. Balancing is what closed most of the gap, since the
per-homograph label distributions are severely skewed.

**How much to make of the margin against the paper.** The 95% interval spans
[0.986, 0.995] and contains 0.990, so this is a consistent lead rather than a
statistically separated one; the eval split is too small to resolve a
three-sentence difference. The honest summary is *matches or slightly beats the
paper's best, and clearly beats everything else*: the production server-side
maxent classifier at 0.954 falls far outside our interval. The right comparison
would be McNemar's paired test over per-sentence predictions, and Gorman et al.
published only aggregate accuracies, so the ceiling on "decisive" is set by
what was released rather than by the model.

The POS baseline matching `Server: ML` at 0.954 is worth noting on its own:
78 of the 162 homographs are morphosyntactic and 22 more are mixed, so for most
of this corpus a tagger plus majority vote is a genuinely strong system, and it
is the number the probe has to beat to be interesting.

## Where it fails

### The canonical sentence

> "I read the second Doctor Lee lead a live band on Reading Road at ten thirty"

with `Reading` supplied as a pronunciation hint and `Doctor Lee` as a name
span, which is what the upstream stage emits.

| token | chosen | probability | rule | correct? |
|---|---|---:|---|---|
| `read` | `R EH1 D` (past) | 0.858 | probe | yes |
| `lead` | `L IY1 D` (verb) | 0.997 | probe | yes |
| `live` | `L AY1 V` (adjective) | 0.999 | probe | yes |
| `Reading` | `R EH1 D IH0 NG` (place) | n/a | hint | yes |

The probe gets all four right, with the default `distilroberta-base` backbone
and with both of the others. Probabilities for roberta-large are 0.896, 0.989,
0.999; for roberta-base 0.842, 0.997, 0.999. Nothing here is hand-patched: this
is what `python -m lede --canonical` prints.

The hardest of the four is `lead`. "Doctor Lee lead a live band" is the verb
reading and shares its context with the metal sense in two directions at once,
and it is the one the specification flagged. All three backbones get it, at
0.99 or better.

`Reading` is supplied as a hint here because that is what the upstream stage
emits, but the hint turns out to be redundant: run the same sentence with no
hint and no name span and the probe picks `reading_geo` on its own, at 0.998
(distilroberta) and 0.986 (roberta-large). Marking `Reading Road` as a name
span instead of hinting it also returns `R EH1 D IH0 NG`, by the proper-noun
rule. All three routes agree, which is the result you want but not one to
generalise from a single sentence.

One thing this walkthrough exposes that is not a heteronym problem at all:
`at` comes back as `AE1 T`, CMUdict's stressed citation form, where connected
speech wants the reduced `AH0 T`. Taking CMUdict's first pronunciation is a
placeholder for a real lexicon with function-word reduction, and every
unstressed function word in a sentence has this bug.

### The 13 roberta-large errors on the evaluation split

Every misclassified sentence is in `results/errors_roberta-large.csv`. Counts
in parentheses are how many training examples that reading has.

**A. Corpus coverage: the gold reading is barely in the training data (3)**

| homograph | gold | predicted | sentence |
|---|---|---|---|
| conglomerate | `K AH0 N G L AA1 M ER0 EY2 T` (0) | `K AH0 N G L AA1 M ER0 AH0 T` (90) | "...bonded the quartz and conglomerate together, creating Kittatinny Mountain." |
| content | `K AH0 N T EH1 N T` (1) | `K AA1 N T EH0 N T` (89) | "Not content with their own lands, they've come to steal mine!" |
| content | `K AH0 N T EH1 N T` (1) | `K AA1 N T EH0 N T` (89) | "...criticism at people content with themselves, who achieve their success by trampling on others." |

Unwinnable. `conglomerate`'s verb reading appears zero times in training, so no
classifier fit on this data can emit it; `content`'s adjective reading appears
once against 89. This is a property of the corpus, not of the model.

**B. Probably annotation error (2)**

| homograph | gold | predicted | sentence |
|---|---|---|---|
| approximate | `AH0 P R AA1 K S AH0 M AH0 T` adj (78) | `AH0 P R AA1 K S AH0 M EY2 T` vrb (11) | "All practical schedulers approximate GPS and use it as a reference to measure fairness." |
| discharge | `D IH1 S CH AA2 R JH` nou (80) | `D IH0 S CH AA1 R JH` vrb (10) | "Owners, or operators of facilities, that discharge regulated waste are then required to secure discharge permits." |

In both the model's reading is the better one. "Schedulers approximate GPS" is
a verb; "facilities that discharge regulated waste" is a relative clause with a
verb in it. These are cases where the probe went against a majority label it
had every statistical incentive to follow, and was right. They are counted as
errors because the gold says so.

**C. No sentential syntax to read: headline, UI string, or fragment (3)**

| homograph | gold | predicted | sentence |
|---|---|---|---|
| increment | `IH1 N K R AH0 M EH2 N T` vrb (11) | `IH1 N K R AH0 M AH0 N T` nou (78) | "They can then use pads marked "Increment," "Decrement," "Forward," and "Backward" to change settings." |
| isolate | `AY1 S AH0 L EY2 T` vrb (69) | `AY1 S AH0 L AH0 T` nou (21) | "Floodwaters from Storm Isolate 13 Vermont Towns; article; The New York Times online; accessed." |
| convert | `K AA1 N V ER0 T` nou (12) | `K AH0 N V ER1 T` vrb (78) | "A Nation challenged-the convert; Shoe-Bomb Suspect Fell in With Extremists." |

The target sits in a button label, a newspaper headline in headlinese, and a
truncated news teaser. There is no well-formed clause around it to read the
syntax off, and title case removes the capitalisation cue too.

**D. Genuine near-misses on well-supported labels (5)**

| homograph | gold | predicted | sentence |
|---|---|---|---|
| bow | `B OW1` knot (43) | `B AW1` ship (47) | "The trams used a mixture of bow collectors and trolley poles." |
| bow | `B OW1` knot (43) | `B AW1` ship (47) | "The gaunt honesty of those projecting concrete frames carrying boxed-out bow windows persists." |
| export | `EH1 K S P AO0 R T` nou (75) | `AH0 K S P AO1 R T` vrb (14) | "An export to blog feature allows for a 1-click publication on the web of a figure." |
| insert | `IH1 N S ER2 T` nou (47) | `IH2 N S ER1 T` vrb (43) | "Similarly easy to the insert operation, Remove() uses the Meld operation..." |
| upset | `AH0 P S EH1 T` vrb (24) | `AH1 P S EH2 T` nou (65) | "Brock was evidently upset by the news that the conspirators had been shot." |

The two `bow` failures need to know that "bow collector" (tram equipment) and
"bow window" (architecture) are fixed collocations taking `B OW1`; the
surrounding text is about trams and buildings, not archery, so topical
similarity actively points the wrong way. `export` and `insert` are nouns used
as modifiers inside compounds ("an export to blog feature", "the insert
operation") where the verb reading is locally more plausible. `upset` is a
passive participle read as a predicative adjective.

### The adversarial set

`adversarial/adversarial.tsv` is 30 hand-written sentences over 17 homographs
where nearby context points the wrong way. Nothing is refit; the probes are the
ones fit on the training split.

| system | Wikipedia eval | adversarial | winnable (n=27) |
|---|---:|---:|---:|
| POS-rule baseline | 0.954 | 0.633 | 0.704 |
| Probe, roberta-large | 0.992 | 0.833 | 0.926 |

All five misses:

| homograph | category | gold | predicted | sentence |
|---|---|---|---|---|
| object | misleading n-gram | `AH0 B JH EH1 K T` (0) | `AA1 B JH EH0 K T` (90) | "The museum had catalogued each object carefully, so the curators were startled to hear the donor object to the display." |
| desert | semantic field | `D IH0 Z ER1 T` (0) | `D EH1 Z ER0 T` (90) | "The convoy had crossed the Sahara without incident, which made it stranger still that two drivers chose to desert." |
| entrance | misleading n-gram | `AH0 N T R AE1 N S` (0) | `EH1 N T R AH0 N S` (90) | "Guests filed past the marble entrance hall, where the singer would entrance them for the better part of an hour." |
| bass | semantic field | `B AE1 S` (13) | `B EY1 S` (74) | "Between the amplifier and the stacked sheet music sat a glass tank where his prize bass circled slowly." |
| bow | semantic field | `B AW1` (47) | `B OW1` (43) | "The quartermaster counted the arrows, set down the quiver, and went below to inspect the cracked bow." |

Three of the five are the same unwinnable class as category A: `object`,
`desert`, and `entrance` have **zero** training examples of the gold reading,
so the probe is a constant predictor for them. That leaves two real failures,
both in the `semantic_field` category, and both the same shape as the `bow`
errors above: the sentence is topically loaded toward the wrong reading (a fish
tank surrounded by amplifiers and sheet music; a ship's bow surrounded by
archery equipment). Topical similarity is exactly what frozen embedding
features encode, so this is the probe's characteristic weakness rather than
bad luck.

The probe is perfect on `garden_path` (7/7) and `long_distance` (3/3), where
the tagger misleads the baseline, and level with the baseline on
`semantic_field`. The two systems fail in different places, which is the
useful part.

Full per-category breakdown in `adversarial/ANALYSIS.md`. A 30-sentence set is
small: one sentence is 3.3 points, so read the category numbers as directional.

### What each category would actually need

**A and the three adversarial zeros are a data problem, not a model problem.**
18 of 162 homographs have a single label across their entire training split and
57 more are over 95% skewed. No system fit on this corpus can produce the
missing readings. Fixing it needs sentences, specifically *targeted* sentences
for minority readings, which means either a sense-tagged corpus or deliberate
mining for the rare reading. That is the highest-value work available and it is
annotation work, not modelling work.

**B needs a second annotator**, not a better model. If the gold label is wrong,
every improvement in the model shows up as a regression. Two of 13 errors being
questionable gold sets a practical ceiling around 11 errors on this split, and
nobody can tell where the real ceiling is without re-adjudication.

**C needs context beyond the sentence.** A headline or a UI string does not
contain the syntax that disambiguates it; the disambiguating information is in
the document, the surrounding list of button labels, or the fact that it is a
headline at all. That means a model that sees a window wider than one sentence,
plus a genre or register signal. Passing the preceding sentence and a
document-type flag into the encoder is the cheap version.

**D is the only category where a better model is the answer**, and even there
the two `bow` cases are really a lexical resource question: "bow collector" and
"bow window" are collocations, and a phrase list would fix them more reliably
and more cheaply than any amount of encoder scaling. `upset` and `insert` want
syntax the mean-pooled span representation throws away, so the honest fix is to
give the probe the dependency relation, or a parse-derived feature, rather than
a bigger encoder. roberta-large already beats deberta-v3-large, so scale is not
the lever.

Realistically that puts the floor for this corpus at 5 to 7 errors, with the
last few unreachable for reasons that have nothing to do with the classifier.

## Reproducing

```bash
make all              # fetch data, run tests, regenerate every number
```

First run costs about six minutes of CPU forward passes with the default
encoder; embeddings and POS tags are cached under `cache/`, so later runs finish
in seconds.

| target | what it does |
|---|---|
| `make data` | fetch the corpus and CMUdict |
| `make test` | pytest |
| `make eval` | baseline + probe + results tables |
| `make probe-serving` | roberta-base and distilroberta-base runs |
| `make probe-large` | the headline roberta-large row |
| `make adversarial` | rebuild and score the stress set |
| `make weights` | refit and rewrite the checked-in serving weights |
| `make canonical` | the walkthrough sentence above |

Requires Python 3.11+. Dependencies are pinned in `requirements.txt`; `make`
builds a virtualenv automatically. Serving needs only `pip install -e .`
(numpy, torch, transformers); scikit-learn and spaCy are needed to *fit* the
probes, not to run them.

## Layout

```
lede/api.py           run() and StageResult, the contract entry point
lede/g2p.py           tokenisation, per-token pronunciation policy
lede/lexicon.py       CMUdict, IPA to ARPAbet, heteronym inventory
lede/serve.py         serving-time probe: exported weights, one forward pass
lede/export_weights.py  refit and write lede/weights/
lede/probe.py         embedding extraction + per-homograph probe fitting
lede/baseline_pos.py  POS-rule baseline, and the no-encoder fallback
lede/data.py          loaders, byte-offset handling
lede/evaluate.py      accuracy computation, results tables
lede/run_all.py       whole research pipeline, one command
lede/adversarial.py   adversarial evaluation
lede/weights/         checked-in probe weights and POS fallback rules
tests/                loader, lexicon, serving, and contract tests
results/              generated tables
adversarial/          hand-written stress set + analysis
```

`DECISIONS.md` records every non-obvious choice; `TODO.md` records what is left.

## A note on the contract

This stage is specified to satisfy `rime-agents/CONTRACT.md`, which is not
present in this checkout. The field names used here (`decisions`,
`alternatives`, `rule`, `meta.oov`, `meta.pron_hints`, `meta.name_spans`) come
from the stage specification; the surrounding envelope (`stage`, `ok`, `text`,
`tokens`) is this repo's own. `StageResult.as_dict` is the single
serialisation boundary, so reconciling with the real contract means changing
one method.

## Reference

Gorman, K., Mazovetskiy, G., and Nikolaev, V. (2018). Improving homograph
disambiguation with supervised machine learning. *LREC 2018*, 1349-1352.
