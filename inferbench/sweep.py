from __future__ import annotations

from pathlib import Path

import yaml

from inferbench.models import EndpointConfig, SweepCell, CellResult
from inferbench.runner import run_cell
from inferbench.workloads import build_messages
from inferbench.datasets_loader import load_structured, load_rag, load_summary


def load_sweep(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


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
                    tokenizer, write_dir: str) -> list[CellResult]:
    Path(write_dir).mkdir(parents=True, exist_ok=True)
    cells: list[CellResult] = []
    for ep in endpoints:
        for workload in sweep["workloads"]:
            items = _items_for(workload, sweep["n"] + sweep["warmup"])
            payloads = [build_messages(workload, it, tokenizer) for it in items]
            for conc in sweep["concurrency"]:
                cell = SweepCell(config=ep.name, workload=workload, concurrency=conc)
                res = await run_cell(complete_fn, ep, cell, payloads,
                                     max_tokens=sweep["max_tokens"][workload],
                                     warmup=sweep["warmup"], n=sweep["n"], concurrency=conc)
                out = Path(write_dir) / f"{ep.name}_{workload}_{conc}.json"
                out.write_text(res.model_dump_json(indent=2))
                cells.append(res)
    return cells
