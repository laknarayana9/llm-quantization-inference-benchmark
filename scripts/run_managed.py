"""Run the benchmark sweep against real OpenAI-compatible endpoints.

In This targets the managed Nebius Token Factory endpoint. Requires the
API key in the environment variable named by each endpoint's `api_key_env`.

Usage:
    export NEBIUS_API_KEY=...
    python scripts/run_managed.py --sweep configs/sweep.yaml --reduced
    python scripts/run_managed.py --sweep configs/sweep.yaml
"""
from __future__ import annotations

import argparse
import asyncio

from inferbench.client import complete
from inferbench.sweep import load_sweep, run_sweep, build_endpoints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--reduced", action="store_true",
                    help="smoke run: n=10, concurrency [1, 5]")
    args = ap.parse_args()

    sweep = load_sweep(args.sweep)
    if args.reduced:
        sweep["n"] = 10
        sweep["concurrency"] = [1, 5]

    # Canonical tokenizer for prompt-shaping: configurable so every config (self-host
    # and managed) shapes from identical text. Falls back to the first endpoint's model.
    from transformers import AutoTokenizer
    tok_model = sweep.get("tokenizer_model") or sweep["endpoints"][0]["model"]
    tokenizer = AutoTokenizer.from_pretrained(tok_model)

    endpoints = build_endpoints(sweep, kind="managed")
    cells = asyncio.run(run_sweep(sweep, endpoints, complete, tokenizer, args.results_dir))
    print(f"Wrote {len(cells)} cells to {args.results_dir}/")


if __name__ == "__main__":
    main()
