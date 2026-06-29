#!/usr/bin/env bash
# AWQ-Int4 — Qwen team's official AWQ checkpoint on vLLM (port 8001).
set -euo pipefail
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --port 8001 \
  --quantization awq_marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-chunked-prefill
