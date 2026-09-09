# Adversarial error analysis

30 hand-written sentences over 17 homographs, built so that context near the
target points at the wrong reading. Both systems are the ones fit on the
training split; nothing is refit here. Schema and byte-offset convention match
the upstream data set, so the same loader reads it.

Regenerate with `make adversarial` (or `python adversarial/build.py &&
python -m lede.adversarial`). Per-sentence predictions land in
`predictions.csv`, or `predictions_<encoder>.csv` for a non-default encoder.

## Headline

| system | Wikipedia eval | adversarial | drop |
|---|---:|---:|---:|
| POS-rule baseline | 0.954 | 0.633 | −0.321 |
| Frozen probe, bert-base-cased | 0.988 | 0.700 | −0.288 |
| Frozen probe, roberta-large | 0.992 | 0.833 | −0.159 |

A 30-sentence set is small (one sentence is 3.3 points), so treat the
category numbers below as directional. The size of the drop is not in doubt;
its exact value per category is.

**The encoder matters much more here than it does on the corpus.** On the
Wikipedia eval split roberta-large is worth four sentences over bert-base
(0.992 against 0.988). On this set it is worth four out of thirty, and the drop
from the eval number roughly halves. Whatever robustness to misleading local
context the larger encoder has, the corpus does not measure it, because
naturally occurring text rarely puts the cues in conflict. That is the argument
for building a set like this one.

The rest of this file analyses the **bert-base** run in detail, since that was
the default encoder when it was written. The roberta-large summary follows;
where the two disagree, the direction is the same and roberta-large is simply
further along it.

### roberta-large, same set

| system | all (n=30) | winnable (n=27) |
|---|---:|---:|
| POS-rule baseline | 0.633 | 0.704 |
| Frozen probe, roberta-large | 0.833 | 0.926 |

Winnable sentences only, by category:

| category | n | baseline | probe |
|---|---:|---:|---:|
| garden_path | 7 | 0.857 | **1.000** |
| long_distance | 3 | 0.667 | **1.000** |
| misleading_ngram | 7 | 0.429 | **1.000** |
| semantic_field | 10 | 0.800 | 0.800 |

roberta-large closes `misleading_ngram` completely, which bert-base does not
(0.571). `semantic_field` is the one category where the larger encoder buys
nothing at all: both of its winnable misses are there (`bass` in a room full of
amplifiers, `bow` surrounded by archery equipment). Topical similarity is what
these features encode, so a trap built out of topical similarity attacks the
representation itself rather than any weakness in the classifier on top of it.
That is the one failure mode scaling the encoder does not appear to touch.

## Three of the errors are unwinnable

Before reading anything into the rest: `object`, `desert`, and `entrance` have
**only one wordid across all 90 of their training sentences**. The verb
readings (`object_vrb`, `desert_vrb`, `entrance_vrb`) never appear in training
at all, so both systems are structurally incapable of producing them: the
probe literally degenerates to a constant predictor, and the POS rule table
maps every tag to the same label.

This is not a modeling failure, it is a coverage limit of the data set. It
affects **18 of 162 homographs**, and a further 57 have a training majority
above 95%. Any accuracy figure on this corpus, including the 0.988 bert-base
headline, is partly a measure of how well a system reproduces that skew.

Excluding the three unwinnable sentences:

| system | winnable (n=27) |
|---|---:|
| POS-rule baseline | 0.704 |
| Frozen probe, bert-base-cased | 0.778 |

## By trap category

Winnable sentences only.

| category | n | baseline | probe |
|---|---:|---:|---:|
| garden_path | 7 | 0.857 | **1.000** |
| long_distance | 3 | 0.667 | **1.000** |
| semantic_field | 10 | **0.800** | 0.700 |
| misleading_ngram | 7 | 0.429 | 0.571 |

The ordering is the interesting part, and it is not the one I expected.

### Where the probe wins outright: syntax under local ambiguity

The probe is perfect on `garden_path` and `long_distance`; the baseline is not.
Both categories are failures of the *tagger*, and the probe does not depend on
one.

> Yesterday's minutes, which the secretary will **read** aloud at the meeting
> next month, ran to forty pages.

The baseline says `read_past`. spaCy tags the token VERB correctly, but the
coarse tag cannot express tense, so the rule has nothing to key on and falls
back to the majority. The probe gets it right: BERT's representation of `read`
carries the `will` that governs it, four tokens back, and coarse POS throws
exactly that away.

> Smith has played matches for the county from 1993 to **present**, a run
> nobody expected to last.

This is adapted from the example Gorman et al. single out in their own error
analysis (§5.3), where they report that all six of their systems predict the
verb here and that the server model mistags `present` as a bare verb. Our
baseline reproduces the failure exactly: spaCy tags it VERB, the rule maps
VERB to `present_vrb`. The probe is unaffected, because it never consults a
tag.

That is the cleanest result in this set: the paper attributes a specific
failure to POS-tag propagation, and removing the POS dependency removes the
failure.

### Where the probe loses: the semantic field overwhelms the target

`semantic_field` is the only category where the baseline beats the probe, and
the reason is the mirror image of the above. The probe's signal *is* lexical
semantics, so stacking the sentence with vocabulary from the wrong sense
attacks it directly.

> Between the amplifier and the stacked sheet music sat a glass tank where his
> prize **bass** circled slowly.

Both systems answer `bass` (the instrument). `amplifier` and `sheet music` sit
in the left context; `glass tank` and `circled slowly` are the actual evidence
and they are ignored. Mean-pooled BERT states at the target absorb the whole
sentence's topical drift, and with ~90 training examples per homograph the
probe has no way to learn that *fish-tank* cues should dominate
*music-shop* cues.

> The quartermaster counted the arrows, set down the quiver, and went below to
> inspect the cracked **bow**.

Here the baseline is right by luck, since `bow_nou-ship` is its majority answer,
while the probe is actively pulled to `bow_nou-knot` by *arrows* and *quiver*.
Where a homograph's two senses are distinguished by topic rather than syntax,
a frozen encoder is exactly the wrong tool to make robust: it is doing topic
matching, and the trap is built out of topic.

### Where both collapse: two occurrences, one embedding each

`misleading_ngram` is the worst category for both systems (0.429 / 0.571), and
most of its sentences share a design: the homograph appears **twice**, once in
each reading.

> The museum had catalogued each **object** carefully, so the curators were
> startled to hear the donor object to the display.

> Guests filed past the marble **entrance** hall, where the singer would
> entrance them for the better part of an hour.

> The talk was scheduled for exactly one hour, and she spent it on the
> **minute** differences between the two shells.

For `object` and `entrance` these are also the unwinnable-label cases, so they
were lost twice over. But `minute` is winnable and both systems still miss it:
spaCy tags the target NOUN (it is `the minute differences`, a perfectly
ordinary adjectival reading that the tagger gets wrong), and the probe is
dragged by `scheduled for exactly one hour` toward the time sense.

roberta-large is the exception here: it gets all seven winnable
`misleading_ngram` sentences, `minute` included. Whatever lets it keep two
occurrences of the same word apart is the clearest capability gap between the
two encoders on this set.

The general pattern: when a sentence establishes a strong reading early and
then switches, both systems commit to the first reading. The baseline commits
through the tagger, which is itself influenced by the earlier occurrence. The
probe commits through the sentence-level topical signal that mean-pooling
mixes into the target vector.

## What this says about the headline number

The probe's 0.988 on the Wikipedia eval split is real, and it beats every
non-hybrid system in Gorman et al. But the adversarial set shows what that
number is and is not measuring:

1. **It is partly memorized skew.** 18 homographs have one label in training;
   57 more are >95% skewed. A constant predictor scores 0.840 on this eval set.
   The probe's genuine contribution is the 0.146 above that, not the 0.988.
2. **The probe's advantage over rules is specifically syntactic robustness.**
   It wins exactly where POS tagging is brittle: long-distance agreement,
   garden paths, the `present` case the paper itself flags. This is a real and
   transferable win.
3. **The probe's weakness is topical interference**, and it is a weakness the
   POS baseline does not share. A hybrid would want the rule to pre-empt the
   probe on cleanly morphosyntactic homographs, which is close to the
   architecture Gorman et al. arrived at for a different reason: they
   hybridized because rules had high precision and low recall, but the same
   structure would also cover this failure mode.
4. **Naturally occurring Wikipedia text is easy.** Sentences were sampled, not
   adversarially selected; the contexts are cooperative. A 29-point drop from
   hand-written traps suggests the gap between benchmark and deployment for a
   TTS front-end is wide, which is consistent with homograph disambiguation
   having been the paper's second-most-bug-reported English component despite
   good offline numbers.

## Honest limitations

- 30 sentences, written by one person who knew both systems' architectures.
  They are not a random sample of hard cases; they are targeted at
  hypothesized weaknesses, and they found some. Category-level n is 3 to 11.
- Gold labels are my own judgments. I checked each against the labeling
  conventions in the training data (e.g. that a rip-noun is `tear_vrb`), but
  they were not multiply annotated the way the upstream corpus was.
- The `long_distance` category has only 3 sentences; its 0.667/1.000 split is
  two sentences' worth of evidence and should not be quoted on its own.
