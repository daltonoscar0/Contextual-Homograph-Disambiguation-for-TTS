"""Command line entry point.

    python -m lede "I read the second Doctor Lee lead a live band"
    python -m lede --eval
    python -m lede --canonical

``--eval`` scores the *shipped weights* end to end: it runs the serving path,
not the training path, over the evaluation split, so what it reports is what
the packaged stage actually does rather than what the research pipeline did.
"""

from __future__ import annotations

import argparse
import json
import sys

from lede import serve

# The sentence this stage is specified against, as it arrives from the
# normaliser. "Reading" is a place here and only the caller knows that, so it
# comes in as a pronunciation hint the way the upstream stage would send it.
CANONICAL = (
    "I read the second Doctor Lee lead a live band on Reading Road at ten thirty"
)
CANONICAL_HINTS = {"Reading": "R EH1 D IH0 NG"}
# Computed rather than typed, so the span cannot drift from the sentence.
CANONICAL_NAME_SPANS = [
    (CANONICAL.index("Doctor Lee"), CANONICAL.index("Doctor Lee") + len("Doctor Lee"))
]


def _evaluate(model: str) -> int:
    """Micro and macro accuracy of the shipped weights on the eval split."""
    from lede.data import load_split
    from lede.evaluate import score

    examples = load_split("eval")
    probes = serve.load_probes(model)
    print(
        f"lede: scoring {len(examples)} eval sentences with {probes.encoder} "
        f"({probes.layer_mode}) from lede/weights/",
        file=sys.stderr,
    )

    predictions: list[str] = []
    for index, example in enumerate(examples, 1):
        vectors, _ = serve.embed(
            example.sentence, [(example.start, example.end)], model, probes.layer_mode
        )
        posterior = probes.probabilities(example.homograph, vectors[0])
        predictions.append(max(posterior, key=posterior.get) if posterior else "")
        if index % 200 == 0 or index == len(examples):
            print(f"lede: {index}/{len(examples)}", file=sys.stderr, flush=True)

    scores = score(predictions, examples)
    errors = sum(p != e.wordid for p, e in zip(predictions, examples))
    print(
        json.dumps(
            {
                "model": probes.encoder,
                "layer_mode": probes.layer_mode,
                "n": len(examples),
                "micro": round(scores.micro, 4),
                "micro_ci": [round(scores.micro_ci[0], 4), round(scores.micro_ci[1], 4)],
                "macro": round(scores.macro, 4),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lede", description="Contextual homograph disambiguation and G2P."
    )
    parser.add_argument("text", nargs="?", help="sentence to pronounce")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="score the shipped weights on the evaluation split",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="run the specification's canonical sentence with its hints",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"encoder to use (default {serve.DEFAULT_MODEL}, or $LEDE_MODEL)",
    )
    parser.add_argument(
        "--phones", action="store_true", help="print the phone string, not JSON"
    )
    args = parser.parse_args(argv)

    if args.model:
        import os

        os.environ["LEDE_MODEL"] = args.model

    if args.eval:
        return _evaluate(serve.active_model())

    from lede.api import run

    if args.canonical:
        result = run(CANONICAL, CANONICAL_HINTS, CANONICAL_NAME_SPANS)
    elif args.text:
        result = run(args.text)
    else:
        parser.print_help()
        return 2

    print(result.pronunciation if args.phones else result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
