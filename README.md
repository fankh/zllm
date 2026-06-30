# ZLLM

White-box LLM inference engine with zero-copy latent intercept.

## Features

- From-scratch inference engine in Rust (zero Python) — CPU (AVX-512), plus
  optional iGPU paths (raw-Vulkan `ZLLM_VK=1` / wgpu) tuned for AMD Strix Halo;
  beats llama.cpp on single-stream decode (see [BENCHMARKS.md](BENCHMARKS.md))
- Mid-layer hidden-state access (`RunnerObserver`) with a **hook write-back**
  path — activation steering and memory inject/capture edit the live residual stream
- Confidence-driven **early exit** (layer short-circuit during decode)
- **Hallucination/uncertainty detection** from the output distribution
  (predictive entropy / top-prob), opt-in via `detect_hallucination`
- Logit control: token banning (`ban:` mode). *Structured-grammar modes
  (regex/JSON/BNF) are stubs — not yet implemented.*
- Paged-KV **continuous batching** (vLLM-style: prefix cache, preemption,
  chunked prefill, batched spec-decode) on the GPU backend, `ZLLM_CB=1`
- Goal / task / status API (REST `/v1/goal/*`), disk-persisted, backed by the
  in-memory store

## Install

Prebuilt binaries for the latest release: <https://github.com/fankh/zllm/releases/latest>

| Platform | Acceleration | Archive |
|---|---|---|
| Windows x86_64 | CPU | `zllm-vX.Y.Z-windows-x86_64-cpu.zip` |
| Windows x86_64 | CUDA 12 (sm_89) | `zllm-vX.Y.Z-windows-x86_64-cuda.zip` |
| Linux x86_64 | CPU | `zllm-vX.Y.Z-linux-x86_64-cpu.tar.gz` |
| Linux x86_64 | CUDA 12 (sm_89) | `zllm-vX.Y.Z-linux-x86_64-cuda.tar.gz` |
| macOS aarch64 | Metal | `zllm-vX.Y.Z-macos-aarch64.tar.gz` |

Each archive contains the `zllm` binary, `configs/default.toml`, `README.md`, `LICENSE`, and `CHANGELOG.md`. CUDA artifacts target `sm_89` (Ada Lovelace / RTX 40-series); newer architectures are covered by PTX JIT, older cards are not. CUDA binaries are built but not runtime-tested in CI — validate on a real GPU box before relying on them.

```bash
# Linux example
tar xzf zllm-v0.1.3-linux-x86_64-cpu.tar.gz
cd zllm-v0.1.3-linux-x86_64-cpu
./zllm serve --config configs/default.toml
```

## Build from source

```bash
# Build
cargo build --release

# Build with CUDA (requires CUDA 12.x toolkit + nvcc)
cargo build --release --features cuda

# Build with Metal (macOS only)
cargo build --release --features metal

# Run server
cargo run -- serve --config configs/default.toml

# Health check
curl http://localhost:8080/health

# CLI help
cargo run -- --help
```

The build script needs `protoc` (Protocol Buffers compiler) on PATH. CUDA builds without a local GPU should set `CUDA_COMPUTE_CAP` explicitly (e.g. `CUDA_COMPUTE_CAP=89`) so `candle-kernels` doesn't try to autodetect via `nvidia-smi`.

## Architecture

See [ai-inference-engine docs](https://github.com/fankh/new-research/tree/main/ai-inference-engine) for full architecture specification.
