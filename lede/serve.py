"""Serving-time probe: load exported weights, embed one sentence, decide.

The training path in ``lede.probe`` fits an sklearn ``StandardScaler`` +
``LogisticRegression`` pipeline per homograph, reading embeddings from a cache
of the whole corpus. None of that is wanted in a front end. This module exports
the fitted pipelines to a small ``.npz`` per encoder and reloads them as plain
arrays, so serving needs numpy and a transformer forward pass and nothing else.

The scaler folds into the linear layer exactly. For ``z = (x - mean) / scale``
and ``logit = W z + b``:

    W' = W / scale                b' = b - W' @ mean

so one matmul against the raw embedding reproduces the pipeline's output to
floating-point noise, which ``tests/test_serve.py`` asserts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from lede.data import REPO_ROOT

WEIGHTS_DIR = REPO_ROOT / "lede" / "weights"

# Serving default. roberta-large is the headline accuracy but a 1.3GB download
# and far too heavy for a free demo backend; distilroberta is 6 layers and
# ~330MB. Override with LEDE_MODEL.
DEFAULT_MODEL = "distilroberta-base"
SUPPORTED_MODELS = ("distilroberta-base", "roberta-base", "roberta-large")

MAX_LENGTH = 256


def active_model() -> str:
    """Encoder for ``run()``, from ``LEDE_MODEL`` or the serving default."""
    return os.environ.get("LEDE_MODEL", DEFAULT_MODEL)


def _slug(model: str) -> str:
    return model.replace("/", "_")


def weights_path(model: str) -> Path:
    return WEIGHTS_DIR / f"probe_{_slug(model)}.npz"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export(model: str, probes: dict[str, object], mode: str, class_weight) -> Path:
    """Write fitted per-homograph pipelines to ``lede/weights/``.

    ``probes`` is the structure ``lede.probe.fit_probes`` returns: either
    ``("constant", wordid)`` or ``("model", pipeline, C)``.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    dimension = 0

    for homograph, entry in sorted(probes.items()):
        if entry[0] == "constant":
            manifest[homograph] = {"kind": "constant", "classes": [str(entry[1])]}
            continue

        pipeline = entry[1]
        scaler, logistic = pipeline[0], pipeline[-1]
        # Same spurious-FP-flag situation as lede.probe._fit_one: Apple's
        # Accelerate BLAS raises divide/overflow flags from this matmul that are
        # not backed by real numerical trouble. Suppress narrowly and assert.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weight = logistic.coef_ / scaler.scale_
            bias = logistic.intercept_ - weight @ scaler.mean_
        if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
            raise FloatingPointError(f"folding the scaler for {homograph!r} diverged")
        dimension = weight.shape[1]

        arrays[f"W::{homograph}"] = weight.astype(np.float32)
        arrays[f"b::{homograph}"] = bias.astype(np.float32)
        manifest[homograph] = {
            "kind": "linear",
            "classes": [str(c) for c in logistic.classes_],
            "C": float(entry[2]),
        }

    path = weights_path(model)
    np.savez_compressed(
        path,
        manifest=np.array(
            json.dumps(
                {
                    "encoder": model,
                    "layer_mode": mode,
                    "class_weight": class_weight,
                    "dimension": dimension,
                    "homographs": manifest,
                }
            )
        ),
        **arrays,
    )
    return path


# --------------------------------------------------------------------------
# Load and predict
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Probes:
    """Loaded per-homograph linear probes for one encoder."""

    encoder: str
    layer_mode: str
    dimension: int
    manifest: dict[str, dict]
    arrays: dict[str, np.ndarray]

    def __contains__(self, homograph: str) -> bool:
        return homograph in self.manifest

    def classes(self, homograph: str) -> list[str]:
        return list(self.manifest[homograph]["classes"])

    def probabilities(self, homograph: str, vector: np.ndarray) -> dict[str, float]:
        """Posterior over this homograph's wordids for one pooled embedding."""
        entry = self.manifest[homograph]
        classes = entry["classes"]
        if entry["kind"] == "constant":
            return {classes[0]: 1.0}

        # Accelerate raises spurious divide/overflow/invalid flags from this
        # matmul, as it does throughout this repo. Suppress the flags but check
        # the result, so a genuinely non-finite logit still fails loudly rather
        # than silently turning argmax into a coin flip.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = (
                self.arrays[f"W::{homograph}"] @ vector + self.arrays[f"b::{homograph}"]
            )
        if not np.isfinite(logits).all():
            raise FloatingPointError(f"non-finite logits for {homograph!r}")
        if logits.shape[0] == 1:
            # sklearn stores a single row for binary problems; classes_[1] is
            # the positive class.
            positive = 1.0 / (1.0 + np.exp(-float(logits[0])))
            return {classes[0]: 1.0 - positive, classes[1]: positive}
        shifted = np.exp(logits - logits.max())
        probabilities = shifted / shifted.sum()
        return {c: float(p) for c, p in zip(classes, probabilities)}


@lru_cache(maxsize=4)
def load_probes(model: str | None = None) -> Probes:
    model = model or active_model()
    path = weights_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"no probe weights for {model!r} at {path}. "
            f"Available: {sorted(p.name for p in WEIGHTS_DIR.glob('probe_*.npz'))}. "
            f"Regenerate with `make weights`."
        )
    blob = np.load(path)
    manifest = json.loads(str(blob["manifest"]))
    arrays = {key: blob[key] for key in blob.files if key != "manifest"}
    return Probes(
        encoder=manifest["encoder"],
        layer_mode=manifest["layer_mode"],
        dimension=manifest["dimension"],
        manifest=manifest["homographs"],
        arrays=arrays,
    )


POS_RULES_PATH = WEIGHTS_DIR / "pos_rules.json"


@lru_cache(maxsize=1)
def load_pos_rules() -> dict:
    """The POS baseline's learned rules, shipped for the no-probe fallback."""
    if not POS_RULES_PATH.exists():
        return {"rules": {}, "defaults": {}}
    return json.loads(POS_RULES_PATH.read_text())


@lru_cache(maxsize=4)
def _encoder(model: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    encoder = AutoModel.from_pretrained(model, output_hidden_states=True)
    encoder.eval()
    return tokenizer, encoder


def embed(sentence: str, spans: list[tuple[int, int]], model: str, mode: str):
    """Pool encoder states over each character span of one sentence.

    Same pooling rule as training (``lede.probe._keep_indices``), reused rather
    than reimplemented so serving cannot drift from the fitted weights.
    """
    import torch

    from lede.probe import _keep_indices, _pool

    tokenizer, encoder = _encoder(model)
    encoded = tokenizer(
        [sentence] * len(spans),
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    offsets = encoded.pop("offset_mapping")
    with torch.inference_mode():
        states = encoder(**encoded).hidden_states
        keeps, aligned = _keep_indices(offsets, spans, encoded["attention_mask"])
        pooled = [_pool(states[-k], keeps) for k in (1, 2, 3, 4)]
        vectors = pooled[0] if mode == "final" else torch.cat(pooled, dim=1)
    return vectors.numpy().astype(np.float32), aligned
