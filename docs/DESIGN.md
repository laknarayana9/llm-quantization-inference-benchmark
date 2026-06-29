# LLM Quantization & Inference Benchmark Lab — Design

**Author:** Lak Tuttagunta

## Problem

We need to serve an open-source LLM (Qwen2.5-7B-Instruct) for production GenAI
workloads and answer a concrete buyer's question:

> Given target latency, throughput, quality, and cost constraints, should we
> self-host a quantized model, and if so in which format — or pay for a managed
> token API?

This project builds an inference benchmark lab that answers that question with
real, reproducible measurements rather than vendor claims. It is deployment- and
measurement-focused, not an exercise in implementing quantization algorithms.

## SLO Profiles (customer-driven framing)

The benchmark is framed around customer service-level objectives, not just raw
numbers. Every measured cell is scored **pass/fail against the relevant SLO**, so
the report reads like a production readiness assessment.

| Profile | TTFT | p95 latency | Quality gate | Notes |
|---|---|---|---|---|
| Interactive chat | TTFT < T_chat | p95 < L_chat | JSON-validity > Q_chat | latency-critical |
| RAG assistant | — | p95 < L_rag | faithfulness > Q_rag | grounding-critical |
| Long-context summary | — | p95 < L_sum or async SLA | summary score > Q_sum | throughput/cost-optimized |
| Production-wide | — | — | — | error rate < E, timeout rate < TO |

Thresholds (`T_chat`, `L_chat`, `L_rag`, `L_sum`, `E`, `TO`, quality gates) are finalized
during implementation based on **realistic customer/product expectations**,
and adjusted for feasibility after the BF16 baseline run. (SLOs are driven by
requirements; the baseline only calibrates what is achievable, it does not define
the SLO.) The report then shows, per config, which SLO profiles each deployment
satisfies.

## Goal & Non-Goals

**Goals**
- Compare FP16/BF16 vs AWQ-Int4 vs GPTQ-Int4 served on self-hosted vLLM, plus a
  managed token API (Nebius Token Factory) as a first-class baseline.
- Measure latency (p50/p95/p99), TTFT, throughput, GPU memory, error rate, and
  cost per 1M tokens across realistic workloads and concurrency levels.
- Measure **quality** per config (not just speed), scored against the BF16
  baseline.
- Produce a customer-facing technical report with a clear production
  recommendation, including the **self-host vs managed cost-crossover** analysis.

**Non-Goals (v1)**
- Implementing quantization algorithms from scratch.
- Multi-GPU / tensor-parallel serving (7B fits comfortably on one GPU).
- GGUF, bitsandbytes, FP8 — explicit **stretch goals**, not v1. (SGLang is a
  scoped **v1.1** milestone — see below.)
- Locust/k6 — the async client is sufficient.

## Key Architectural Insight

Both vLLM and Nebius Token Factory expose an **OpenAI-compatible
`/v1/chat/completions` API**. A single async benchmark client therefore drives
every target identically — only `base_url` (and auth) changes. This is what makes
the self-host-vs-managed comparison rigorous instead of hand-wavy.

```
                    ┌──────────────────────────────┐
                    │  benchmark client             │
                    │  (asyncio + httpx,            │
                    │   OpenAI-compatible, streaming)│
                    └───────────────┬──────────────┘
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
   vLLM: Qwen2.5-7B BF16   vLLM: AWQ-Int4 / GPTQ   Nebius Token Factory
   (self-hosted, GPU)      (self-hosted, GPU)      (managed, Qwen2.5-7B)
              │                    │                     │
              └──── GPU metrics (nvidia-smi) ────────────┘ (self-host only)
```

## The Comparison Matrix

**Configs (4 in v1):**
| Config | Source | Checkpoint |
|---|---|---|
| BF16 baseline | self-host vLLM | `Qwen/Qwen2.5-7B-Instruct` |
| AWQ-Int4 | self-host vLLM | `Qwen/Qwen2.5-7B-Instruct-AWQ` (official) |
| GPTQ-Int4 | self-host vLLM | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (official) |
| Managed | Nebius Token Factory | Qwen2.5-7B served endpoint |

Using Qwen's **official** AWQ/GPTQ checkpoints removes "is the quantization any
good" as a confounding variable.

**Workloads (3):**
| Workload | Input tokens | Output tokens | Purpose |
|---|---|---|---|
| Short chat | ~200 | ~100 | latency-sensitive interactive |
| RAG answer | ~2,000 | ~300 | grounded answering |
| Long-context summary | ~8,000 | ~500 | prefill-heavy, KV-cache stress |

**Concurrency levels:** 1, 5, 20, 50 simultaneous users.

Full sweep = 4 configs × 3 workloads × 4 concurrency = 48 cells (managed config
omits GPU-memory metric).

## Metrics

Per cell:
- p50 / p95 / p99 end-to-end latency
- **TTFT** (time to first token, requires streaming)
- output tokens/sec (throughput)
- requests/sec
- error rate
- **GPU memory** used (self-host only — honest asymmetry: the managed GPU is not
  observable)
- **cost per 1M tokens** (see Cost Model)

## Quality Evals

Speed without quality is a trap. Every config is scored against the **BF16
baseline** per workload:

- **Short chat / structured output:** JSON-validity rate on structured-output
  prompts; deterministic schema check.
- **RAG answer:** grounding / faithfulness — is the answer supported by the
  provided context? LLM-as-judge (strong model via API) + deterministic checks.
- **Long-context summary:** summarization quality via LLM-as-judge, scored
  side-by-side vs BF16 output.

Headline form: *"AWQ cut memory ~3× and held quality within X% of BF16 —
acceptable for chat, but I'd keep BF16 for long-context summaries."*

## Methodology Rigor (what makes the numbers defensible)

- **Warmup:** discard the first K warmup requests per cell before measuring.
- **Consistent output length:** use `ignore_eos` + fixed `max_tokens` so
  tokens/sec is comparable across configs (otherwise variable EOS skews
  throughput).
- **Fixed inputs & seeds:** pinned prompt set and sampling seeds; greedy or
  fixed-temperature decoding for quality runs.
- **Statistical validity:** N requests per cell; report distribution / variance,
  not a single shot.
- **Reproducibility pinning:** record exact vLLM version, model commit hashes,
  GPU type, CUDA version, and dataset slice in the results metadata.
- **vLLM tuning is an explicit narrative:** document the tuned
  `--gpu-memory-utilization`, `--max-num-seqs`, `--max-model-len`, and chunked
  prefill settings (chunked prefill matters a lot for the 8k-context workload).
  These are the "optimal deployment shapes" that drive production serving cost.

## Datasets (prompts & ground truth)

Quality evals require real data with ground truth, not synthetic gibberish. Pin a
small, fixed slice of public datasets:
- **RAG answer:** a QA dataset providing (context, question, supported-answer)
  triples.
- **Long-context summary:** a summarization dataset providing source documents.
- **Short chat / structured:** a small curated set of structured-output prompts
  with target schemas.

The exact dataset choices and slice sizes are finalized in the implementation
plan; they must be pinned (fixed indices) for reproducibility.

## Cost Model (elevated to a headline result)

- **Self-host:** cost/1M tokens = (GPU $/hr ÷ measured sustained throughput in
  tokens/hr). This is **utilization-dependent** — at 1 concurrent user the GPU is
  mostly idle and self-host is far more expensive per token than managed; at 50
  users it is far cheaper.
- **Managed:** Token Factory's published per-token price.
- **Headline deliverable — the cost crossover:** "Self-hosting AWQ beats Token
  Factory above ~X sustained requests/sec." This utilization-driven crossover is
  the central senior insight and gets its own chart.

## Failure Modes & Debugging (deliberately reproduced)

Production inference needs debugging under load, so the project induces and
explains real failure modes rather than only reporting happy-path numbers. Each
is reproduced where feasible, with the observed symptom, root cause, and fix
documented:

- **OOM from KV-cache growth** — push 8k context × 50 concurrency until the
  server OOMs; show how `--max-num-seqs` / `--gpu-memory-utilization` / chunked
  prefill resolve it.
- **TTFT regression on long prompts** — prefill cost on the 8k workload vs short
  chat; effect of chunked prefill.
- **p95/p99 latency spikes under concurrency** — queueing behavior as users scale
  1 → 50.
- **Streaming timeout errors** — client-side timeout interacting with long
  generations.
- **Degraded quality after quantization** — caught by the quality evals, not by
  latency metrics.
- **GPU underutilization at low concurrency** — ties directly to the cost-
  crossover (idle GPU = terrible $/token).
- **Retry storms from client timeout behavior** — show how aggressive client
  retries amplify load and worsen p99.

Failure modes that cannot be cleanly reproduced are documented as "expected /
not observed" rather than fabricated.

## Managed Black-Box Honesty (stated limitation)

Token Factory may serve the model at a different precision (often FP8) on
hardware and with batching we do not control, and its GPU memory is not
observable. We are therefore measuring **delivered price/performance**, not an
identical serving config. This limitation is stated explicitly in the report;
pretending it is apples-to-apples would be a red flag.

## Visualization & Narrative Artifacts

- **Hero chart 1 — cost crossover:** $/1M tokens vs sustained requests/sec, self-
  host configs vs managed line.
- **Hero chart 2 — quality-vs-latency Pareto frontier:** which configs are
  dominated, which are on the frontier.
- Supporting charts: latency vs concurrency, throughput vs concurrency.
- **README as a customer-facing technical report:** Problem → SLO Profiles →
  Setup → Methodology → Results tables (with SLO pass/fail) → hero charts →
  **Recommendation** → **Failure Modes** → **Limitations & what I'd do next**.
- Short write-up / blog framing for communication value.

**Recommendation templates** (decision patterns in the report, backed by the
measured data):
- **Use the managed API when:** traffic is low/spiky, speed-to-market matters,
  ops team is small — self-host can't beat managed $/token below the crossover.
- **Use self-hosted AWQ when:** sustained throughput is high and memory/cost
  efficiency matters — above the crossover it wins on $/token.
- **Use BF16 when:** quality sensitivity is high, long-context accuracy matters,
  or quantization quality loss is unacceptable for the workload.
- **Use GPTQ when:** compatibility or checkpoint availability makes it the
  practical choice over AWQ.

## Infrastructure

- **GPU:** one Nebius L40S (48GB) or A100. 7B BF16 is expected to fit on a 48GB
  GPU, while Int4 should leave substantially more headroom for KV cache. The
  actual 8k-context × 50-concurrency limit is validated empirically during the
  OOM / `--max-num-seqs` experiments rather than assumed. ~$1.50–2/hr, project
  total ~$30–50.
- **Serving:** vLLM (OpenAI-compatible server), one launch config per format.
- **Client:** Python, asyncio + httpx, streaming for TTFT.

## Repo Structure

```
serve/      # vLLM launch configs/scripts per format (BF16, AWQ, GPTQ)
bench/      # async OpenAI-compatible client, workloads, metrics collection
evals/      # quality scoring (JSON validity, RAG grounding, summarization judge)
datasets/   # pinned prompt sets & ground truth (or loaders w/ fixed indices)
results/    # raw per-cell JSON + generated tables
report/     # README technical report + generated hero charts
```

A config-driven sweep (YAML) + a single driver script regenerate all results so
the benchmark is reproducible.

## v1.1 Milestone (scoped, near-term — not far-stretch)

vLLM and SGLang are both common production serving frameworks, so SGLang is a
scoped v1.1 milestone rather than an open-ended stretch:

- **v1.1 — SGLang comparison:** repeat AWQ and GPTQ serving on **SGLang** for
  **one workload** (RAG answer) at **two concurrency levels** (5 and 50), reusing
  the same OpenAI-compatible benchmark client. Yields a clean
  "vLLM vs SGLang" claim without bloating v1.

## Stretch Goals (post-v1.1)

1. **FP8** self-host (online-quantized via vLLM — very role-relevant; caveat that
   it is online-quantized).
2. SGLang expanded to the full workload/concurrency matrix.
3. Additional model family (e.g. Llama-3.1-8B) for cross-model generalization.

## Success Criteria

- Full 48-cell sweep runs from a single driver and produces pinned, reproducible
  results with recorded environment metadata.
- Quality evals produce per-config scores relative to BF16.
- README report contains both hero charts and a concrete, defensible production
  recommendation including the cost-crossover point.
- A reader can reproduce the headline numbers from the repo.
```
