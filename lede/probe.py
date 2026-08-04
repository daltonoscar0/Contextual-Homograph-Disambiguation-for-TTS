"""Frozen-embedding linear probe.

Contextual embeddings come from a frozen ``bert-base-cased``: we mean-pool the
wordpieces covering the homograph's character span, and fit one L2 logistic
regression per homograph. Nothing is fine-tuned; BERT runs once, its output is
cached to disk, and every subsequent fit reads from that cache.

Two layer representations are compared on a train-internal validation split --
the final hidden layer, and the last four layers concatenated -- and the winner
is used for the reported eval run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lede.data import Example, REPO_ROOT, group_by_homograph, load_split

CACHE_DIR = REPO_ROOT / "cache"
# Frozen encoder. Override with LEDE_ENCODER to compare encoders; caches and
# the layer-mode choice are keyed by model name so they never collide.
MODEL_NAME = os.environ.get("LEDE_ENCODER", "bert-base-cased")
MODEL_SLUG = MODEL_NAME.replace("/", "_")
MAX_LENGTH = 256
BATCH_SIZE = 32
SEED = 0
C_GRID = (0.1, 1.0, 10.0)
LAYER_MODES = ("final", "last4")
CHOICE_PATH = CACHE_DIR / f"probe_choice_{MODEL_SLUG}.json"


# --------------------------------------------------------------------------
# Embedding extraction
# --------------------------------------------------------------------------


@dataclass
class Embeddings:
    """Per-example pooled vectors for one layer mode, aligned to a split."""

    vectors: np.ndarray  # (n_examples, hidden)
    aligned: np.ndarray  # bool mask, False if no wordpiece covered the span


def _cache_path(split: str, mode: str) -> Path:
    return CACHE_DIR / f"emb_{MODEL_SLUG}_{split}_{mode}.npz"


def _keep_indices(offsets, spans, attention_mask):
    """Token positions overlapping each target span, plus an alignment flag.

    ``offsets`` are the tokenizer's character offset mappings. A wordpiece
    counts as part of the target if it overlaps the span at all, which is the
    right rule for subword splits that straddle the boundary (``##ass`` in
    ``bass``). Special tokens carry (0, 0) offsets and are excluded here.

    Computed once per batch and reused across layers -- this loop is pure
    Python and re-running it per layer dominated extraction time.
    """
    keeps, aligned = [], []
    for i, (start, end) in enumerate(spans):
        token_offsets = offsets[i]
        keep = []
        for j in range(token_offsets.shape[0]):
            tok_start = int(token_offsets[j, 0])
            tok_end = int(token_offsets[j, 1])
            if tok_end <= tok_start:  # special token or empty piece
                continue
            if attention_mask[i, j] == 0:
                continue
            if tok_start < end and tok_end > start:
                keep.append(j)
        if keep:
            keeps.append(keep)
            aligned.append(True)
        else:
            # Target fell outside the truncation window; fall back to the
            # sentence mean so the row stays usable and gets flagged.
            keeps.append([j for j in range(len(token_offsets)) if attention_mask[i, j]])
            aligned.append(False)
    return keeps, aligned


def _pool(hidden, keeps):
    """Mean-pool one layer's states over precomputed token positions."""
    import torch

    return torch.stack([hidden[i, keep].mean(dim=0) for i, keep in enumerate(keeps)])


def extract(split: str, examples: list[Example] | None = None) -> dict[str, Embeddings]:
    """Embed a split under every layer mode, caching each to ``cache/``.

    One forward pass produces both representations, so this is a single sweep
    over the data regardless of how many modes we compare.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    examples = examples if examples is not None else load_split(split)
    CACHE_DIR.mkdir(exist_ok=True)

    cached = {}
    for mode in LAYER_MODES:
        path = _cache_path(split, mode)
        if path.exists():
            blob = np.load(path)
            if blob["vectors"].shape[0] == len(examples):
                cached[mode] = Embeddings(blob["vectors"], blob["aligned"])
    if len(cached) == len(LAYER_MODES):
        print(f"probe: reusing cached embeddings for {split}")
        return cached

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()

    # Hidden states scale with batch x length x width x layers. A "large"
    # encoder needs a smaller batch or the stack alone runs to hundreds of MB
    # and the machine swaps, which costs far more than the smaller batch does.
    batch_size = BATCH_SIZE
    if model.config.hidden_size > 768 or model.config.num_hidden_layers > 12:
        batch_size = max(4, BATCH_SIZE // 4)
    print(f"probe: encoder {MODEL_NAME}, batch size {batch_size}")

    out: dict[str, list] = {mode: [] for mode in LAYER_MODES}
    aligned_flags: list[bool] = []
    started = time.time()

    for begin in range(0, len(examples), batch_size):
        batch = examples[begin : begin + batch_size]
        encoded = tokenizer(
            [e.sentence for e in batch],
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        offsets = encoded.pop("offset_mapping")
        spans = [(e.start, e.end) for e in batch]
        with torch.inference_mode():
            states = model(**encoded).hidden_states
            keeps, flags = _keep_indices(offsets, spans, encoded["attention_mask"])
            pooled = [_pool(states[-k], keeps) for k in (1, 2, 3, 4)]
            final = pooled[0].clone()
            last4 = torch.cat(pooled, dim=1)
        # Drop the full hidden-state stack before the next forward pass; for a
        # large encoder it is hundreds of MB per batch and holding it while the
        # next batch allocates is what pushes this machine into swap.
        del states, pooled
        out["final"].append(final.numpy())
        out["last4"].append(last4.numpy())
        aligned_flags.extend(flags)

        done = begin + len(batch)
        if done % (batch_size * 20) < batch_size or done == len(examples):
            rate = done / max(time.time() - started, 1e-6)
            print(f"probe: {split} {done}/{len(examples)} ({rate:.0f} ex/s)", flush=True)

    result = {}
    aligned = np.array(aligned_flags, dtype=bool)
    for mode in LAYER_MODES:
        vectors = np.concatenate(out[mode], axis=0).astype(np.float32)
        np.savez_compressed(_cache_path(split, mode), vectors=vectors, aligned=aligned)
        result[mode] = Embeddings(vectors, aligned)

    unaligned = int((~aligned).sum())
    if unaligned:
        print(f"probe: warning -- {unaligned}/{len(examples)} spans not covered")
    return result


# --------------------------------------------------------------------------
# Per-homograph probes
# --------------------------------------------------------------------------


def _internal_split(n: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask selecting ~20% of rows for train-internal validation."""
    mask = np.zeros(n, dtype=bool)
    held = max(1, int(round(0.2 * n)))
    mask[rng.permutation(n)[:held]] = True
    return mask


def _predict(pipeline, x: np.ndarray) -> np.ndarray:
    """Predict under the FP-flag guard described in ``_fit_one``."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return pipeline.predict(x)


def _fit_one(x: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, random_state=SEED),
    )
    # Apple's Accelerate BLAS raises spurious divide-by-zero / overflow FP flags
    # from the matmuls inside the solver. The flags are not backed by real
    # numerical trouble -- the inputs are finite and every fitted coefficient
    # comes back finite (asserted below) -- and they fire on both float32 and
    # float64, so they are an artifact of the backend rather than of our data.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pipeline.fit(x, y)

    if not np.isfinite(pipeline[-1].coef_).all():
        raise FloatingPointError("logistic regression produced non-finite weights")
    return pipeline


def _select_C(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """Pick C on a held-out slice of this homograph's training rows.

    With a single label, or too few rows to hold anything out, C is irrelevant
    and we return the middle of the grid.
    """
    if len(set(y)) < 2 or len(y) < 10:
        return 1.0
    held = _internal_split(len(y), rng)
    if len(set(y[~held])) < 2:
        return 1.0

    best_C, best_score = 1.0, -1.0
    for C in C_GRID:
        model = _fit_one(x[~held], y[~held], C)
        score = float((_predict(model, x[held]) == y[held]).mean())
        if score > best_score:
            best_C, best_score = C, score
    return best_C


def fit_probes(
    train: list[Example], embeddings: np.ndarray, seed: int = SEED
) -> dict[str, object]:
    """One logistic regression per homograph, C tuned per homograph."""
    index_of = {id(e): i for i, e in enumerate(train)}
    models: dict[str, object] = {}
    rng = np.random.default_rng(seed)

    for homograph, rows in sorted(group_by_homograph(train).items()):
        idx = np.array([index_of[id(e)] for e in rows])
        x = embeddings[idx]
        y = np.array([e.wordid for e in rows])
        if len(set(y)) < 2:
            models[homograph] = ("constant", y[0])
            continue
        C = _select_C(x, y, rng)
        models[homograph] = ("model", _fit_one(x, y, C), C)
    return models


def predict_probes(
    models: dict[str, object], examples: list[Example], embeddings: np.ndarray
) -> list[str]:
    """Predict wordids, batching by homograph so each model is called once."""
    predictions: list[str] = [""] * len(examples)
    by_homograph: dict[str, list[int]] = {}
    for i, example in enumerate(examples):
        by_homograph.setdefault(example.homograph, []).append(i)

    for homograph, indices in by_homograph.items():
        entry = models[homograph]
        if entry[0] == "constant":
            for i in indices:
                predictions[i] = entry[1]
            continue
        idx = np.array(indices)
        for i, label in zip(indices, _predict(entry[1], embeddings[idx])):
            predictions[i] = label
    return predictions


def _micro(predictions: list[str], examples: list[Example]) -> float:
    return sum(p == e.wordid for p, e in zip(predictions, examples)) / len(examples)


def select_layer_mode(
    train: list[Example], train_embeddings: dict[str, Embeddings]
) -> tuple[str, dict[str, float]]:
    """Compare layer modes on a train-internal split; eval is never touched."""
    rng = np.random.default_rng(SEED)
    held = _internal_split(len(train), rng)
    inner = [e for e, h in zip(train, held) if not h]
    outer = [e for e, h in zip(train, held) if h]

    scores: dict[str, float] = {}
    for mode in LAYER_MODES:
        vectors = train_embeddings[mode].vectors
        models = fit_probes(inner, vectors[~held])
        # Homographs absent from the inner split can't be scored; skip them.
        scorable = [e for e in outer if e.homograph in models]
        keep = np.array(
            [i for i, (e, h) in enumerate(zip(train, held)) if h and e.homograph in models]
        )
        scores[mode] = _micro(
            predict_probes(models, scorable, vectors[keep]), scorable
        )
        print(f"probe: layer mode {mode!r} internal-val micro {scores[mode]:.4f}")

    best = max(LAYER_MODES, key=lambda m: (scores[m], m == "final"))
    return best, scores


def run() -> dict:
    """Extract, select the layer mode, fit, and predict on eval."""
    train, evaluation = load_split("train"), load_split("eval")
    train_embeddings = extract("train", train)
    eval_embeddings = extract("eval", evaluation)

    if CHOICE_PATH.exists():
        choice = json.loads(CHOICE_PATH.read_text())
        mode, scores = choice["mode"], choice["scores"]
        print(f"probe: reusing layer mode {mode!r} from cache")
    else:
        mode, scores = select_layer_mode(train, train_embeddings)
        CACHE_DIR.mkdir(exist_ok=True)
        CHOICE_PATH.write_text(json.dumps({"mode": mode, "scores": scores}, indent=2))
    print(f"probe: using layer mode {mode!r}")

    models = fit_probes(train, train_embeddings[mode].vectors)
    predictions = predict_probes(models, evaluation, eval_embeddings[mode].vectors)
    print(f"probe: eval micro accuracy {_micro(predictions, evaluation):.4f}")
    return {"predictions": predictions, "mode": mode, "scores": scores}


if __name__ == "__main__":
    run()
