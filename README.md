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
  (predictive entropy / top-prob), opt-in via `detect_hallucination` — also
  available for *other* servers (llama.cpp, vLLM, ...) via the spin-off
  [zllm-probe](https://github.com/fankh/zllm-probe) drop-in proxy
- **Grammar-constrained decoding**: `regex:<pattern>` (anchored byte-DFA token
  masking — output is guaranteed to match) and `ban:<ids>` token banning.
  *JSON-schema/BNF modes not yet implemented (requests are rejected with 400).*
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

Each archive contains the `zllm` binary, `configs/default.toml`, `README.md`, `LICENSE`, and `CHANGELOG.md`. CUDA artifacts target `sm_89` (Ada Lovelace / RTX 40-series); newer architectures are covered by PTX JIT, older cards are not. CUDA binaries are built but not runtime-tested in CI — validate on a real GPU box before relying on them. The AMD iGPU paths (`vulkan`/`gpu` features, below) are **source-build only** — no prebuilt archive.

```bash
# Linux example
tar xzf zllm-vX.Y.Z-linux-x86_64-cpu.tar.gz
cd zllm-vX.Y.Z-linux-x86_64-cpu
./zllm serve --config configs/default.toml
```

## Get a model

zllm loads **GGUF** models (same files llama.cpp uses) plus a HuggingFace
`tokenizer.json` from the same model family:

1. Download a GGUF, e.g. `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
   (all-Q4 quantizations give the best GPU-path numbers — see BENCHMARKS.md).
2. Put a matching `tokenizer.json` next to it (from the model's HF repo).
3. Point the config at both:

```toml
# configs/default.toml
[model]
path = "C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
tokenizer_path = ""   # empty = tokenizer.json next to `path`
```

## Build from source

No non-Rust build dependencies (stable toolchain only).

```bash
# CPU (default; AVX-512 used automatically via target-cpu=native)
cargo build --release

# AMD iGPU, raw-Vulkan fast path (the headline single-stream decode engine)
cargo build --release --features vulkan

# AMD iGPU, wgpu path (continuous batching / serving stack)
cargo build --release --features gpu

# NVIDIA CUDA (requires CUDA 12.x toolkit + nvcc)
cargo build --release --features cuda

# Apple Metal (macOS only)
cargo build --release --features metal
```

CUDA builds without a local GPU should set `CUDA_COMPUTE_CAP` explicitly (e.g.
`CUDA_COMPUTE_CAP=89`) so `candle-kernels` doesn't try to autodetect via `nvidia-smi`.

## Run

```bash
# Server (REST + chat UI) — CPU path
./target/release/zllm serve --config configs/default.toml

# Server with the raw-Vulkan iGPU fast lane (needs --features vulkan build)
ZLLM_VK=1 ./target/release/zllm serve --config configs/default.toml

# Server with vLLM-style continuous batching (needs --features gpu build)
ZLLM_CB=1 ./target/release/zllm serve --config configs/default.toml

# Health check (default port 8080; see [server].rest_port in the config)
curl http://localhost:8080/health

# One-shot CLI generation, no server
./target/release/zllm generate \
  --model C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --tokenizer C:/models/llama-3.2-1b/tokenizer.json \
  --prompt "The capital of France is" --max-tokens 32

# CLI help
./target/release/zllm --help
```

## Configuration reference

Config file (`configs/*.toml`): `[server]` rest_port, max_concurrent; `[model]`
path (GGUF), tokenizer_path, dir (picker scan); `[engine]` backend_pool_size,
draft_model_path (spec-decode), memory_inject_alpha (0.0 = inject off),
default_temperature/top_k/top_p.

Environment flags (deployment / opt-in features):

| flag | effect |
|---|---|
| `ZLLM_VK=1` | load the raw-Vulkan iGPU fast lane (needs `--features vulkan`) |
| `ZLLM_CB=1` | continuous-batching server (needs `--features gpu`; `ZLLM_CB_SLOTS`, `ZLLM_CB_PLD`) |
| `ZLLM_HEADMAJOR_KV=1` | head-major KV layout: +5-12% long-context decode (opt-in) |
| `ZLLM_NO_PIN=1` | disable physical-core pinning of the rayon pool |
| `RUST_LOG` / `RAYON_NUM_THREADS` | standard logging / CPU worker cap |

A/B + diagnostics (regression comparison, not for production): `VK_FA3`,
`VK_SCALAR_SDPA` (attention kernel generations), `ZLLM_FLAT_COMBINE`,
`ZLLM_ONLINE_PARTIAL`, `VK_MVONLY`/`VK_NO*`/`VK_PF_*` (cost attribution),
`VK_PFTIME` (prefill timing), `ZLLM_FIRST_FIT` (cache-reclaim policy),
`ZLLM_SWAP` (swap-to-host preemption — loses on UMA, keep off),
`ZLLM_Q3_GATEUP`/`ZLLM_FUSED_QKV` (measured-parity experiments, keep off).
Runtime toggles (REST): `/v1/inspect/enabled`, `/v1/early_exit/enabled`(+`/config`),
`/v1/pld/enabled`, `/v1/spec_decode/enabled`. See [docs/TESTING.md](docs/TESTING.md) for a full playbook.

## Architecture & docs

Engineering docs live in [`docs/`](docs/): [SUMMARY](docs/SUMMARY.md) (project
on-ramp: status, techniques, dead-ends), [TESTING](docs/TESTING.md) (manual
playbook), [VULKAN_PLAN](docs/VULKAN_PLAN.md) (iGPU research notes), and the
plugin/instrumentation design plans.

Related: [**zllm-probe**](https://github.com/fankh/zllm-probe) — the
cross-engine spin-off that brings zllm's uncertainty instrumentation to any
OpenAI-compatible server (llama.cpp `llama-server`, vLLM, ...) as a drop-in
proxy; zllm is its reference engine for the deep (hidden-state) tier.

See [ai-inference-engine docs](https://github.com/fankh/new-research/tree/main/ai-inference-engine) for full architecture specification.
