"""The contract entry point.

    from lede.api import run
    result = run("I read the second Doctor Lee lead a live band")
    result.as_dict()          # contract-valid JSON

``run`` returns a ``StageResult``: every token with a pronunciation, one
``Decision`` per heteronym carrying the full candidate set and the probe's
posterior over it, and a ``meta`` block recording what the stage could not do.

**On the shape of StageResult.** The contract document this stage is supposed
to satisfy (``rime-agents/CONTRACT.md``) is not present in this checkout, so
the field names below are taken from the stage specification -- ``decisions``,
``alternatives``, ``rule``, ``meta.oov``, ``meta.pron_hints``,
``meta.name_spans`` -- and the surrounding envelope (``stage``, ``ok``,
``text``, ``tokens``) is our own. ``as_dict`` is the serialisation boundary, so
if the real contract differs, that one method is what needs to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from lede import g2p, lexicon, serve
from lede.g2p import Decision, Token

STAGE = "lede"


@dataclass
class StageResult:
    """What this stage hands to the next one."""

    stage: str
    ok: bool
    text: str
    tokens: list[Token] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def pronunciation(self) -> str:
        """Flat phone string, with ``<oov:...>`` markers left in place."""
        parts = []
        for token in self.tokens:
            if token.pron:
                parts.append(" ".join(token.pron))
            elif token.oov:
                parts.append(g2p.OOV_TEMPLATE.format(token.text.upper()))
        return " ".join(parts)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "text": self.text,
            "tokens": [t.as_dict() for t in self.tokens],
            "decisions": [d.as_dict() for d in self.decisions],
            "meta": self.meta,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------
# Deciding heteronyms
# --------------------------------------------------------------------------


def _probe_posteriors(
    text: str, spans: list[tuple[int, int]], homographs: list[str], model: str
) -> tuple[list[dict[str, float]], str, bool, list[str]]:
    """Posterior per heteronym span, or a flag saying the probe was unavailable.

    Anything that stops the probe from running -- no torch, no weights, no
    network for the encoder download -- falls back rather than raising, because
    a front end that returns nothing is worse than one that returns the POS
    rule's answer and says so.
    """
    try:
        probes = serve.load_probes(model)
        vectors, aligned = serve.embed(text, spans, model, probes.layer_mode)
    except (ImportError, OSError, FileNotFoundError, ValueError):
        return [], "", False, []

    posteriors = []
    for homograph, vector in zip(homographs, vectors):
        if homograph in probes:
            posteriors.append(probes.probabilities(homograph, vector))
        else:
            posteriors.append({})
    # A target past the 256-wordpiece window pools the whole sentence instead of
    # the span, so its decision is much weaker than its probability suggests.
    unaligned = [h for h, ok in zip(homographs, aligned) if not ok]
    return posteriors, probes.encoder, True, unaligned


def _pos_posteriors(
    text: str, spans: list[tuple[int, int]], homographs: list[str]
) -> list[dict[str, float]]:
    """Fallback: the POS rule baseline, as a degenerate one-hot posterior.

    Two steps down. If spaCy is installed we tag the sentence and use the
    learned tag -> wordid rule; if it is not, we use the homograph's overall
    majority reading. Both come from ``lede/weights/pos_rules.json``, so this
    path needs no training data and no model download.
    """
    rules = serve.load_pos_rules()
    tags: list[str] = ["?"] * len(spans)
    try:
        from lede.baseline_pos import _align_token, _nlp

        document = _nlp()(text)
        for index, (start, end) in enumerate(spans):
            token = _align_token(document, start, end)
            tags[index] = token.pos_ if token is not None else "?"
    except (ImportError, OSError):
        pass

    posteriors = []
    for homograph, tag in zip(homographs, tags):
        chosen = rules["rules"].get(homograph, {}).get(tag) or rules["defaults"].get(
            homograph
        )
        posteriors.append({chosen: 1.0} if chosen else {})
    return posteriors


def _alternatives(homograph: str, posterior: dict[str, float]) -> list[dict]:
    """Every candidate reading with its pronunciation and probe probability.

    Readings the probe never saw in training still appear, with probability
    0.0, so the caller can see the full option set rather than only the ones
    the model happens to score.
    """
    table = lexicon.wordid_pronunciations()
    rows = [
        {
            "wordid": wordid,
            "pron": " ".join(table[wordid]),
            "probability": round(float(posterior.get(wordid, 0.0)), 6),
        }
        for wordid in g2p.readings_of(homograph)
    ]
    return sorted(rows, key=lambda r: (-r["probability"], r["wordid"]))


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def run(text: str, pron_hints=None, name_spans=None) -> StageResult:
    """Pronounce every token of ``text``, deciding heteronyms in context.

    ``pron_hints`` overrides the dictionary for the tokens it names.
    ``name_spans`` marks proper-name regions; a heteronym inside one takes its
    proper-noun reading when it has one.
    """
    model = serve.active_model()
    tokens = g2p.tokenize(text)
    hint_words, hint_spans = g2p.normalise_hints(pron_hints)
    names = g2p.normalise_name_spans(name_spans)

    # Which tokens need the probe. Hinted tokens do not: the caller has already
    # decided, and spending a forward pass to be overridden is waste.
    def hinted(token: Token) -> list[str] | None:
        for span in hint_spans:
            if span["start"] <= token.start and token.end <= span["end"]:
                return span["pron"]
        return hint_words.get(token.text.lower())

    pending = [
        index
        for index, token in enumerate(tokens)
        if g2p.is_heteronym(token) and hinted(token) is None
    ]
    spans = [(tokens[i].start, tokens[i].end) for i in pending]
    homographs = [tokens[i].text.lower() for i in pending]

    posteriors: list[dict[str, float]] = []
    rule, unaligned = "", []
    if pending:
        posteriors, encoder, ok, unaligned = _probe_posteriors(
            text, spans, homographs, model
        )
        if ok:
            rule = f"probe:{encoder}"
        else:
            posteriors = _pos_posteriors(text, spans, homographs)
            rule = "pos-baseline"

    posterior_of = dict(zip(pending, posteriors))
    table = lexicon.wordid_pronunciations()
    decisions: list[Decision] = []
    oov: list[str] = []
    undecided: list[str] = []

    for index, token in enumerate(tokens):
        if token.kind == "punct":
            token.source = "punct"
            continue

        hint = hinted(token)
        if hint is not None:
            token.pron, token.source = hint, "hint"
            if g2p.is_heteronym(token):
                decisions.append(
                    Decision(
                        token=token.text,
                        start=token.start,
                        end=token.end,
                        homograph=token.text.lower(),
                        wordid=g2p.wordid_for_pron(token.text.lower(), hint),
                        pron=hint,
                        alternatives=_alternatives(token.text.lower(), {}),
                        rule="hint",
                    )
                )
            continue

        if index in posterior_of:
            homograph = token.text.lower()
            posterior = posterior_of[index]
            token_rule = rule

            # A heteronym inside a name span takes its proper-noun reading.
            # This overrides the probe: the caller has told us something about
            # the token that the sentence alone does not carry.
            chosen = ""
            if g2p.inside_span(token, names):
                proper = [w for w in g2p.proper_noun_readings(homograph) if w in table]
                if proper:
                    chosen = max(proper, key=lambda w: posterior.get(w, 0.0))
                    token_rule = "name-span"
            if not chosen and posterior:
                chosen = max(posterior, key=posterior.get)

            if chosen:
                token.pron, token.source = list(table[chosen]), token_rule.split(":")[0]
                decisions.append(
                    Decision(
                        token=token.text,
                        start=token.start,
                        end=token.end,
                        homograph=homograph,
                        wordid=chosen,
                        pron=list(table[chosen]),
                        alternatives=_alternatives(homograph, posterior),
                        rule=token_rule,
                    )
                )
                continue
            undecided.append(token.text)

        entries = lexicon.lookup(token.text)
        if entries:
            token.pron, token.source = list(entries[0]), "cmudict"
        else:
            token.oov, token.source = True, "oov"
            oov.append(token.text)

    return StageResult(
        stage=STAGE,
        ok=True,
        text=text,
        tokens=tokens,
        decisions=decisions,
        meta={
            "model": model,
            "rule": rule,
            "oov": oov,
            "n_tokens": len(tokens),
            "n_decisions": len(decisions),
            "undecided_heteronyms": undecided,
            "truncated_spans": unaligned,
            "pron_hints": pron_hints or {},
            "name_spans": [list(s) for s in names],
        },
    )
