"""Fit the probes for one encoder and write serving weights into the repo.

    LEDE_ENCODER=distilroberta-base python -m lede.export_weights

Reads the cached embeddings ``lede.probe`` already produced, refits on the full
training split with the layer mode and class weighting that encoder selected on
its train-internal validation, and writes ``lede/weights/probe_<encoder>.npz``.

The weights are checked in. They are a few megabytes, and shipping them is what
makes ``pip install -e .`` plus one encoder download enough to run the stage --
no corpus download, no six-minute extraction, no sklearn at serving time.
"""

from __future__ import annotations

import json
import shutil

from lede import probe, serve
from lede.baseline_pos import MODEL_PATH as POS_MODEL_PATH
from lede.data import load_split


def main() -> None:
    model = probe.MODEL_NAME
    if not probe.CHOICE_PATH.exists():
        raise SystemExit(
            f"no cached config for {model!r}; run "
            f"`LEDE_ENCODER={model} python -m lede.run_all` first"
        )
    choice = json.loads(probe.CHOICE_PATH.read_text())
    mode, class_weight = choice["mode"], choice["class_weight"]

    train = load_split("train")
    embeddings = probe.extract("train", train)[mode]
    probes = probe.fit_probes(train, embeddings.vectors, class_weight=class_weight)

    path = serve.export(model, probes, mode, class_weight)
    size = path.stat().st_size / 1e6
    print(f"wrote {path.relative_to(probe.REPO_ROOT)} ({size:.1f} MB, {mode})")

    # The POS rules ride along: they are the fallback when the encoder cannot
    # be loaded, and at 24 KB there is no reason not to ship them.
    if POS_MODEL_PATH.exists():
        serve.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(POS_MODEL_PATH, serve.POS_RULES_PATH)
        print(f"wrote {serve.POS_RULES_PATH.relative_to(probe.REPO_ROOT)}")


if __name__ == "__main__":
    main()
