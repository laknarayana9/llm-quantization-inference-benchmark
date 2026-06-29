from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EndpointConfig(BaseModel):
    name: str
    base_url: str
    model: str
    kind: Literal["self_host", "managed"]
    api_key: str | None = None


class RequestResult(BaseModel):
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None = None
    error: str | None = None
    text: str | None = None


class SweepCell(BaseModel):
    config: str
    workload: str
    concurrency: int


class CellResult(BaseModel):
    cell: SweepCell
    kind: str
    n: int
    p50_s: float
    p95_s: float
    p99_s: float
    ttft_p50_s: float
    ttft_p95_s: float
    output_tps: float
    req_per_s: float
    error_rate: float
    gpu_mem_gb: float | None
    raw: list[RequestResult]
