from __future__ import annotations

import os
from pathlib import Path

import yaml

from inferbench.models import EndpointConfig, SweepCell, CellResult
from inferbench.runner import run_cell
from inferbench.workloads import build_messages
from inferbench.datasets_loader import load_structured, load_rag, load_summary


def load_sweep(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_endpoints(sweep: dict, kind: str | None = None) -> list[EndpointConfig]:
    """Build endpoint configs from a sweep dict, optionally filtered by kind.

    Resolves each endpoint's API key from the env var named by `api_key_env`
    (self-host endpoints typically have none).
    """
    out: list[EndpointConfig] = []
    for e in sweep["endpoints"]:
        if kind is not None and e["kind"] != kind:
            continue
        api_key = os.environ.get(e["api_key_env"]) if e.get("api_key_env") else None
        if e.get("api_key_env") and not api_key:
            raise SystemExit(f"Missing env var {e['api_key_env']} for endpoint {e['name']}")
        out.append(EndpointConfig(name=e["name"], base_url=e["base_url"], model=e["model"],
                                  kind=e["kind"], api_key=api_key))
    return out


def _structured_items(n):
    return load_structured(n)


def _items_for(workload: str, n: int):
    if workload == "chat":
        return _structured_items(n)
    if workload == "rag":
        return load_rag(n)
    if workload == "summary":
        return load_summary(n)
    raise ValueError(workload)


async def run_sweep(sweep: dict, endpoints: list[EndpointConfig], complete_fn,
                    tokenizer, write_dir: str, gpu_mem_fn=None) -> list[CellResult]:
    Path(write_dir).mkdir(parents=True, exist_ok=True)
    cells: list[CellResult] = []
    for ep in endpoints:
        # GPU memory is observable only for self-hosted endpoints, not the managed black box.
        ep_gpu_mem_fn = gpu_mem_fn if ep.kind == "self_host" else None
        for workload in sweep["workloads"]:
            items = _items_for(workload, sweep["n"] + sweep["warmup"])
            payloads = [build_messages(workload, it, tokenizer) for it in items]
            for conc in sweep["concurrency"]:
                cell = SweepCell(config=ep.name, workload=workload, concurrency=conc)
                res = await run_cell(complete_fn, ep, cell, payloads,
                                     max_tokens=sweep["max_tokens"][workload],
                                     warmup=sweep["warmup"], n=sweep["n"], concurrency=conc,
                                     gpu_mem_fn=ep_gpu_mem_fn)
                out = Path(write_dir) / f"{ep.name}_{workload}_{conc}.json"
                out.write_text(res.model_dump_json(indent=2))
                cells.append(res)
    return cells
