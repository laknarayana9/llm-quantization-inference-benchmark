from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    target_in: int
    target_out: int


WORKLOADS = {
    "chat": WorkloadSpec("chat", 200, 100),
    "rag": WorkloadSpec("rag", 2000, 300),
    "summary": WorkloadSpec("summary", 8000, 500),
}


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


def _truncate(text: str, target: int, tokenizer) -> str:
    toks = tokenizer.encode(text)
    if len(toks) <= target:
        return text
    return tokenizer.decode(toks[:target])


def build_messages(workload: str, item, tokenizer) -> list[dict]:
    spec = WORKLOADS[workload]
    if workload == "chat":
        user = f"{item.instruction}"
        return [{"role": "system", "content": "You are a precise assistant."},
                {"role": "user", "content": user}]
    if workload == "rag":
        # Reserve ~80 tokens for the question/instruction wrapper.
        ctx = _truncate(item.context, spec.target_in - 80, tokenizer)
        user = f"Context:\n{ctx}\n\nQuestion: {item.question}\nAnswer using only the context."
        return [{"role": "system", "content": "Answer strictly from the provided context."},
                {"role": "user", "content": user}]
    if workload == "summary":
        doc = _truncate(item.document, spec.target_in - 60, tokenizer)
        user = f"Summarize the following document in a few paragraphs:\n\n{doc}"
        return [{"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": user}]
    raise ValueError(f"unknown workload {workload}")
