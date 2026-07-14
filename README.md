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
- OpenAI-compatible **logprobs** (`logprobs: true` + `top_logprobs: N` on chat;
  integer `logprobs` on legacy completions)
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

zllm loads **GGUF** models (same files llama.cpp uses). Supported
families today: **Llama** (1/2/3.x, Mistral-arch included) and
**Qwen2/2.5**, dense variants; unknown architectures are rejected at
load — see `src/backend/arch.rs` for the registry.

For modern BPE models (Llama 3, Qwen2) **the GGUF alone is enough** —
the tokenizer and chat template embedded in the file are used, verified
token-identical to the HF `tokenizer.json`. SentencePiece models
(Llama 2, Mistral v0.3) still need a sibling `tokenizer.json`, which
always takes precedence when present.

1. Download a GGUF, e.g. `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
   (all-Q4 quantizations give the best GPU-path numbers — see BENCHMARKS.md).
2. Point the config at it:

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

### Fully-static / isolated build

zllm is a **single self-contained process** — no Python, no llama.cpp, no
external runtime or SDK. The tokenizer and chat template are read from inside
the GGUF (for byte-level BPE models), and all shaders are baked into the
binary (SPIR-V via `include_bytes!`, WGSL compiled in-process by `naga`). So
the minimal deployment is literally **one executable + one `.gguf` file**.

For a locked-down or air-gapped box, statically link the C runtime so the
binary depends on nothing but core OS libraries. Use the **CPU-only** (default)
build — the GPU fast lanes require the GPU driver's DLLs and so can't be fully
static; CPU-only is the right target for an isolated deployment anyway.

**Windows (static MSVC CRT):**

```powershell
$env:RUSTFLAGS = "-C target-feature=+crt-static -C target-cpu=native"
# separate target dir so it doesn't clobber a normal build:
$env:CARGO_TARGET_DIR = "target-static"
cargo build --release
# -> target-static\release\zllm.exe  (~13 MB)
```

The result links **only guaranteed-present core Windows DLLs** (`kernel32`,
`ntdll`, `ws2_32`, `bcrypt`, `advapi32`, …) — no `VCRUNTIME140`, no
`api-ms-win-crt-*`, no GPU DLLs. It runs on a bare Windows install with **no
MSVC redistributable** required. Verify with
`dumpbin /dependents target-static\release\zllm.exe`.

**Linux (fully-static ELF via musl — zero dynamic deps):**

```bash
rustup target add x86_64-unknown-linux-musl
RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --target x86_64-unknown-linux-musl
# -> target/x86_64-unknown-linux-musl/release/zllm
# `ldd` reports "not a dynamic executable" — copy it anywhere and run.
```

Drop that one binary plus a single `.gguf` into an empty directory and run —
no sidecar files needed for BPE models:

```bash
./zllm serve --config config.toml     # or: ./zllm generate --model model.gguf --prompt "..."
```

(For portability across machines, drop `target-cpu=native` — see the CI note
above — so the binary doesn't require the build host's exact SIMD level.)

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

# One-shot CLI generation, no server. The tokenizer is read from the
# GGUF itself for BPE models; pass --tokenizer only for SPM families.
./target/release/zllm generate \
  --model C:/models/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --prompt "The capital of France is" --max-tokens 32

# CLI help
./target/release/zllm --help
```

## Trust model

zllm has no multi-user story — it is an installed app. Accordingly:

- The server binds **`127.0.0.1` only** by default. Set `ZLLM_BIND=0.0.0.0`
  (or an interface address) to expose it — you'll get a loud startup
  warning if you do that without a key.
- Set `ZLLM_API_KEY=<secret>` to require `Authorization: Bearer <secret>`
  on every route except `/health`. This is the only auth mechanism;
  anyone holding the key controls the model AND the white-box surface
  (goals, inspection, memory).
- Backpressure is bounded: more than `[server] max_concurrent` in-flight
  generations → `503`; each request has a wall-clock budget
  (`ZLLM_REQ_TIMEOUT_SECS`, default 600) and returns partial output
  rather than holding a slot forever.

## API stability (toward 1.0)

Two tiers. **Stable** (semver applies from 1.0): `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/v1/embeddings`, `/tokenize`,
`/detokenize`, `/health`, `/metrics`. Accepted-but-unsupported OpenAI
parameters return `400` — nothing is silently ignored.
**Experimental** (may change in any release): `/v1/goal/*`,
`/v1/inspect*`, `/v1/cb/*`, `/v1/debug/*`, the `*_enabled` toggles, and
the non-OpenAI extensions (`grammar`, `detect_hallucination`,
`repeat_penalty`).

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
