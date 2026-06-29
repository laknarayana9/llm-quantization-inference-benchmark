"""Run the benchmark sweep against self-hosted vLLM endpoints (GPU host).

Assumes vLLM servers are already running (see serve/launch_*.sh) and the
self-host endpoints are uncommented in the sweep config. Captures GPU memory via
nvidia-smi for each cell.

Usage (on the GPU host):
    python scripts/run_selfhost.py --sweep configs/sweep.yaml --reduced
    python scripts/run_selfhost.py --sweep configs/sweep.yaml
"""
from __future__ import annotations

import argparse
import asyncio

from inferbench.client import complete
from inferbench.analysis.gpu import make_gpu_mem_fn
from inferbench.sweep import load_sweep, run_sweep, build_endpoints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--only", default=None,
                    help="run only the endpoint with this name (e.g. awq) — use when "
                         "serving one format at a time")
    ap.add_argument("--reduced", action="store_true",
                    help="smoke run: n=10, concurrency [1, 5]")
    args = ap.parse_args()

    sweep = load_sweep(args.sweep)
    if args.reduced:
        sweep["n"] = 10
        sweep["concurrency"] = [1, 5]

    from transformers import AutoTokenizer
    tok_model = sweep.get("tokenizer_model") or sweep["endpoints"][0]["model"]
    tokenizer = AutoTokenizer.from_pretrained(tok_model)

    endpoints = build_endpoints(sweep, kind="self_host")
    if args.only:
        endpoints = [e for e in endpoints if e.name == args.only]
    if not endpoints:
        raise SystemExit(
            f"No matching self_host endpoints (--only={args.only}). "
            "Check configs/sweep.yaml.")

    gpu_mem_fn = make_gpu_mem_fn(args.gpu_index)
    cells = asyncio.run(run_sweep(sweep, endpoints, complete, tokenizer, args.results_dir,
                                  gpu_mem_fn=gpu_mem_fn))
    print(f"Wrote {len(cells)} cells to {args.results_dir}/")


if __name__ == "__main__":
    main()
