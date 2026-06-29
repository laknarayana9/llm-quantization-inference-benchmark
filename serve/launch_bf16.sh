#!/usr/bin/env bash
# BF16 baseline — full-precision Qwen2.5-7B-Instruct on vLLM (port 8000).
set -euo pipefail
# Disable FlashInfer sampler: it JIT-compiles a CUDA kernel at startup and needs
# nvcc/CUDA toolkit, which GPU boxes with only the CUDA runtime lack. Native
# sampling is used instead — identical for greedy (temperature=0) decoding.
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-chunked-prefill
