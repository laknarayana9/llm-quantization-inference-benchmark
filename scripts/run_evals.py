"""Quality-eval run (the eval step). Two steps to fit one-server-at-a-time serving:

  # while each self-host server is up, generate its outputs (natural completions):
  python scripts/run_evals.py gen --workload rag --config bf16 --n 30
  python scripts/run_evals.py gen --workload rag --config awq  --n 30
  python scripts/run_evals.py gen --workload rag --config gptq --n 30

  # then score all collected outputs vs the baseline (judge via managed Token Factory):
  python scripts/run_evals.py score --workload rag --configs bf16,awq,gptq --baseline bf16

`gen` writes results/outputs_<workload>_<config>.json; `score` reads them all and
writes results/quality_<workload>.json. Quality scoring logic is unit-tested
(inferbench.evals.run.score_outputs); these are the integration wrappers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from inferbench.client import complete
from inferbench.evals.judge import build_judge
from inferbench.evals.run import score_outputs
from inferbench.sweep import load_sweep, build_endpoints
from inferbench.workloads import build_messages, WORKLOADS
from inferbench.datasets_loader import load_structured, load_rag, load_summary


def _items_and_refs(workload: str, n: int):
    if workload == "chat":
        items = load_structured(n)
        return items, [{"schema": it.json_schema} for it in items]
    if workload == "rag":
        items = load_rag(n)
        return items, [{"context": it.context, "question": it.question} for it in items]
    if workload == "summary":
        items = load_summary(n)
        return items, [{"document": it.document} for it in items]
    raise ValueError(workload)


def _tokenizer(sweep: dict):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(sweep.get("tokenizer_model") or sweep["endpoints"][0]["model"])


async def _generate(ep, payloads, max_tokens):
    results = []
    async with httpx.AsyncClient() as c:
        for msgs in payloads:
            # natural completions for quality: ignore_eos off, capture the text
            r = await complete(c, ep, msgs, max_tokens=max_tokens, ignore_eos=False, capture_text=True)
            results.append(r)
    return results


def _make_managed_judge(sweep: dict):
    me = build_endpoints(sweep, kind="managed")[0]
    headers = {"Authorization": f"Bearer {me.api_key}"} if me.api_key else {}

    def api_complete(prompt: str) -> str:
        with httpx.Client() as c:
            resp = c.post(me.base_url.rstrip("/") + "/chat/completions", headers=headers,
                          json={"model": me.model, "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 16, "temperature": 0.0}, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    return build_judge(api_complete)


def cmd_gen(args, sweep) -> None:
    tok = _tokenizer(sweep)
    items, _ = _items_and_refs(args.workload, args.n)
    payloads = [build_messages(args.workload, it, tok) for it in items]
    max_tokens = WORKLOADS[args.workload].target_out
    eps = {e.name: e for e in build_endpoints(sweep, kind="self_host")}
    if args.config not in eps:
        raise SystemExit(f"config {args.config} not in self_host endpoints")
    results = asyncio.run(_generate(eps[args.config], payloads, max_tokens))
    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    sample_err = next((r.error for r in results if not r.ok), None)
    if n_ok == 0:
        raise SystemExit(
            f"All {len(results)} requests FAILED for config '{args.config}'. Is its vLLM "
            f"server up on the right port (bf16:8000 awq:8001 gptq:8002)? error: {sample_err}")
    if n_fail:
        print(f"WARNING: {n_fail}/{len(results)} requests failed (sample: {sample_err})")
    outs = [r.text or "" for r in results]
    out = Path(args.results_dir) / f"outputs_{args.workload}_{args.config}.json"
    out.write_text(json.dumps(outs, indent=2))
    print(f"Wrote {len(outs)} outputs to {out} ({n_ok} ok)")


def cmd_score(args, sweep) -> None:
    _, refs = _items_and_refs(args.workload, args.n)
    names = args.configs.split(",")
    outputs: dict[str, list[str]] = {}
    for name in names:
        p = Path(args.results_dir) / f"outputs_{args.workload}_{name}.json"
        if not p.exists():
            raise SystemExit(f"Missing {p} — run `gen --config {name}` first")
        outputs[name] = json.loads(p.read_text())
    judge = None if args.workload == "chat" else _make_managed_judge(sweep)
    res = score_outputs(args.workload, outputs, refs, judge, baseline=args.baseline)
    out = Path(args.results_dir) / f"quality_{args.workload}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"Wrote {out}")
    for cfg, s in res.items():
        print(f"  {cfg}: mean={s['mean']:.3f}  retained={s['retained_pct']:.1f}%  delta={s['delta_vs_baseline']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--results-dir", default="results")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate outputs for one config (its server must be up)")
    g.add_argument("--workload", required=True, choices=["chat", "rag", "summary"])
    g.add_argument("--config", required=True, help="self-host endpoint name, e.g. bf16")
    g.add_argument("--n", type=int, default=30)

    s = sub.add_parser("score", help="score collected outputs vs baseline")
    s.add_argument("--workload", required=True, choices=["chat", "rag", "summary"])
    s.add_argument("--configs", required=True, help="comma-separated, e.g. bf16,awq,gptq")
    s.add_argument("--baseline", default="bf16")
    s.add_argument("--n", type=int, default=30)

    args = ap.parse_args()
    sweep = load_sweep(args.sweep)
    if args.cmd == "gen":
        cmd_gen(args, sweep)
    else:
        cmd_score(args, sweep)


if __name__ == "__main__":
    main()
