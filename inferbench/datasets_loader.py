from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_DATA = Path(__file__).resolve().parent.parent / "datasets" / "data"


class RagItem(BaseModel):
    context: str
    question: str
    answer: str


class SummaryItem(BaseModel):
    document: str
    reference: str


class StructuredItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    instruction: str
    json_schema: dict = Field(alias="schema")


def load_structured(n: int = 50) -> list[StructuredItem]:
    raw = json.loads((_DATA / "structured_prompts.json").read_text())
    items = [StructuredItem(**x) for x in raw]
    return items[:n]


def load_rag(n: int = 50, seed: int = 1234) -> list[RagItem]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    # Fixed deterministic indices: walk in order, take answerable examples.
    picked: list[RagItem] = []
    for i in range(len(ds)):
        row = ds[i]
        if row["answers"]["text"]:
            picked.append(RagItem(context=row["context"], question=row["question"],
                                  answer=row["answers"]["text"][0]))
        if len(picked) >= n:
            break
    return picked


def load_summary(n: int = 50, seed: int = 1234) -> list[SummaryItem]:
    from datasets import load_dataset

    # govreport: long government reports (often 8k+ tokens) — script-free parquet,
    # well-suited to the long-context summary workload. Columns: report, summary.
    ds = load_dataset("ccdv/govreport-summarization", split="validation")
    # Only keep documents long enough to fill the 8k-token summary workload after
    # shaping (>= ~40k chars ≈ >= ~8k tokens; shorter docs would under-fill the
    # long-context target). Deterministic: first N qualifying docs in dataset order.
    MIN_CHARS = 40_000
    picked: list[SummaryItem] = []
    for i in range(len(ds)):
        row = ds[i]
        if row["report"].strip() and len(row["report"]) >= MIN_CHARS:
            picked.append(SummaryItem(document=row["report"], reference=row["summary"]))
        if len(picked) >= n:
            break
    return picked
