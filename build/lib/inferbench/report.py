from __future__ import annotations

from inferbench.models import CellResult


def results_table(cells: list[CellResult]) -> str:
    head = ("| config | workload | conc | p95 (s) | ttft_p95 (s) | out tok/s | err | gpu GB |\n"
            "|---|---|---|---|---|---|---|---|")
    rows = []
    for c in cells:
        gpu = f"{c.gpu_mem_gb:.1f}" if c.gpu_mem_gb is not None else "—"
        rows.append(f"| {c.cell.config} | {c.cell.workload} | {c.cell.concurrency} | "
                    f"{c.p95_s:.2f} | {c.ttft_p95_s:.2f} | {c.output_tps:.0f} | "
                    f"{c.error_rate:.2%} | {gpu} |")
    return "\n".join([head, *rows])


def build_report(cells: list[CellResult], recommendation: str, limitations: str) -> str:
    return "\n\n".join([
        "# LLM Quantization & Inference Benchmark — Report",
        "## Problem\nServe Qwen2.5-7B for production GenAI workloads: self-host quantized vs managed API.",
        "## Methodology\nWarmup discarded; fixed output length (ignore_eos + max_tokens); "
        "N requests/cell; pinned datasets; metrics over successful requests only.",
        "## Results\n" + results_table(cells),
        "## Recommendation\n" + recommendation,
        "## Limitations\n" + limitations,
    ])
