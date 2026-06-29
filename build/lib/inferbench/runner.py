from __future__ import annotations

import asyncio
import time

import httpx

from inferbench.models import EndpointConfig, SweepCell, CellResult
from inferbench.metrics import aggregate


async def run_cell(complete_fn, ep: EndpointConfig, cell: SweepCell,
                   payloads: list[list[dict]], max_tokens: int,
                   warmup: int, n: int, concurrency: int, gpu_mem_fn=None) -> CellResult:
    sem = asyncio.Semaphore(concurrency)

    async def one(client, messages):
        async with sem:
            return await complete_fn(client, ep, messages, max_tokens)

    def pick(k):  # cycle through payloads deterministically
        return [payloads[i % len(payloads)] for i in range(k)]

    async with httpx.AsyncClient() as client:
        # warmup (discarded)
        if warmup:
            await asyncio.gather(*(one(client, m) for m in pick(warmup)))
        # measured
        start = time.perf_counter()
        results = await asyncio.gather(*(one(client, m) for m in pick(n)))
        wall = time.perf_counter() - start

    gpu_mem = gpu_mem_fn() if gpu_mem_fn else None
    return aggregate(cell, ep.kind, list(results), wall_clock_s=wall, gpu_mem_gb=gpu_mem)
