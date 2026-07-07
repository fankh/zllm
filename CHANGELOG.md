# Changelog

## v0.9.0 — 2026-07-07

Performance: prefill rebuilt around WMMA attention; TTFT transformed; the
white-box feature set made real and live-verified. Head-to-head vs llama.cpp
Vulkan (same box/model/session): decode 1.11x FASTER (213.6 vs 191.8 tok/s,
bit-exact), prefill@1024 0.86x (4063 vs 4722 tok/s, was 0.49x), batched
B=8 0.16x (the remaining gap; wgpu backend). See BENCHMARKS.md.

### Prefill / TTFT (raw-Vulkan engine)
- Fused coopmat FLASH ATTENTION (default): S never materialized; online causal
  softmax in LDS between WMMA GEMM halves. Prefill SDPA 115 -> ~37 ms; 1024-tok
  prefill 341 -> 262 ms. A/B: VK_FA3 (3-phase), VK_SCALAR_SDPA (scalar).
- Chunked batched prefill at position offsets: prompts beyond one 1024 tile run
  as tiles against the resident cache (bsdpa_offset -> fused FA). 2036-tok
  prompt: ~20 s (CPU fallback) -> 0.83 s end-to-end.
- Fast-lane prompt cap 128 -> 4096 (stale gate; 902-tok TTFT 9.5 s -> 0.49 s).
- Cross-request PREFIX CACHE on the fast lane: warm system-prompt turns
  prefill only the suffix (906-tok prefix: 424 -> 48 ms).
- Coopmat GEMM: BK=32 sub-block alignment (+2%); NOLOAD probe showed the core
  is issue-bound (~40 ms total memory work) — double-buffering rejected by data.
- glslang 16.3 toolchain installed (C:/tools/glslang), byte-identical rebuilds.

### Decode / long context
- Head-major KV cache (ZLLM_HEADMAJOR_KV): contiguous per-head reads,
  +5-12% at depth, model-general (hd<=64), bit-exact, opt-in.

### Batched serving (wgpu CB)
- Packed-f16 KV block pool: half the KV bytes, 2x resident streams per GB;
  pack-scatter kernel replaces per-row copies; swap round-trip still bit-exact.
- Skinny Q4_K GEMM for M<=2 decode batches (+56% at M=1); single-stream wgpu
  decode 53 -> 80 tok/s. Skinny wall decomposed into 3 documented limits.

### White-box features (audited: real, live-verified)
- Hallucination/uncertainty detection (detect_hallucination): per-token
  entropy/top-prob report; cold-prefill for bit-reproducible scores.
- Hook WRITE-BACK: per-layer callback returns Option<Tensor>; steering and
  memory-inject edit the live residual stream (inject gated OFF by default —
  engine.memory_inject_alpha, live A/B showed uncurated inject derails a 1B).
- Grammar-constrained decoding: regex mode (anchored byte-DFA token masking,
  EOS only at full match); ban: mode; json/bnf reject with 400.
- Honest surfaces: /v1/info reflects reality; unimplemented options fail loud;
  README/SUMMARY rewritten to match the code; dead scaffolds deleted.

### Code organization
- Binary now consumes the library crate (no dual compilation of the module
  tree). ZERO build warnings across default/vulkan/gpu configs.
- docs/TESTING.md manual playbook; BENCHMARKS.md head-to-head + addenda.


## v0.8.0 — 2026-06-16

Inference performance: CPU brought to parity with llama.cpp, and a new
from-scratch iGPU engine that beats the CPU path. Target box: AMD Ryzen
AI MAX+ 395 (Strix Halo), Radeon 8060S iGPU, Llama-3.2-1B-Instruct Q4_K_M.

### CPU decode — 50 → 63 tok/s (parity with llama.cpp CPU)

- Pin one rayon worker per **physical** core at startup (`main.rs`
  `configure_thread_pool`, new `core_affinity` dep). SMT-sibling
  oversubscription was a ~33% tax on this bandwidth-bound decode; pinning
  recovers it. Disable with `ZLLM_NO_PIN=1`.
- Measured baselines on this box: zllm CPU 50 → **63**; llama.cpp CPU 64;
  llama.cpp iGPU 208. Decode is memory-bandwidth bound (~55 GB/s CPU
  ceiling), so AVX-512 VNNI / wider SIMD / more threads do **not** help —
  confirmed by measurement and dropped.

### iGPU inference engine (new — `--features gpu`, off by default)

zllm's own GPU inference path via `wgpu` → Vulkan (pure-Rust WGSL through
naga; no Vulkan SDK / cmake needed). Lives in `src/backend/gpu`.

- Kernels, all validated **bit-exact** vs CPU/candle: coalesced Q4_K, Q6_K,
  and f32 matvecs (workgroup-per-row + shared-memory reduction); interleaved
  RoPE (`rope_i`); GQA decode SDPA (online/flash softmax); RMSNorm; residual
  add; fused FFN down-projection (SiLU·mul folded into `w2`); GPU argmax.
- `GpuModel` loads a real GGUF (mixed Q4_K/Q6_K) onto the GPU once, keeps the
  KV cache resident, and runs each decode token in a single command buffer.
- **Faithful greedy decode is bit-for-bit identical to the candle CPU forward
  (24/24 tokens) at ~80 tok/s — 27% over CPU.** 10 GPU tests, all bit-exact.
- Findings worth keeping: one-thread-per-row matvec is uncoalesced (the root
  cause of early slowness) — coalescing was decisive (Q4_K 192 GB/s, Q6_K
  120 GB/s in isolation, both beating llama.cpp's ~133 effective); residual
  fusion helped but further dispatch fusion plateaued; the per-token readback
  sync pattern (drain-then-read + a persistent `MAP_READ` buffer) was a
  bigger lever than fusion. Approaching llama.cpp's 208 would need raw
  `ash`+SPIR-V with mega-fused kernels and hand-managed barriers.

### iGPU prefill offload — TTFT (new)

The decode path is matvec (M=1, bandwidth-bound). Prefill is a batched GEMM
(M prompt rows reuse each weight load → compute-bound — the iGPU's strength).
`GpuModel::prefill_forward(prompt)` runs the whole prompt in one batched pass
(K-tiled Q4_K/Q6_K GEMMs, batched RMSNorm/RoPE/causal-GQA-SDPA), fills the
resident KV cache for positions 0..M, and returns the last-token logits;
decode then continues from position M.

- **Bit-faithful:** greedy output is identical to candle's batched CPU forward
  (12/12 tokens; last-token argmax matches, logit cosine 0.999). Supports
  1..=512 tokens (the GEMM's per-thread `acc[8]` caps M at 512).
- **Fast (averaged sweep, vs ~15 ms/token sequential decode):**

  | prompt M | prefill | tok/s | speedup vs M× decode |
  |---|---|---|---|
  | 12  | 48 ms  | 249 | 4.4× |
  | 32  | 61 ms  | 527 | 9.3× |
  | 128 | 256 ms | 500 | 8.8× |
  | 256 | 520 ms | 492 | 8.7× |

  Prefill wins at every length; throughput is a flat ~490–527 tok/s for M≥32,
  well above the 208 decode target.
- Two GEMM optimizations, both bit-exact, got there from an initial 2842 ms /
  481 ms-TTFT (M=256 / M=12):
  1. **8-wide unroll of the inner dot loop** (shared-mem weight row × x): the
     loop was ALU/latency-bound, not bandwidth-bound → **2.84×** (2842→1000 ms).
  2. **Shrinking the LDS weight-row tile 2048→256 floats** (8 KB→1 KB): an
     8 KB tile capped occupancy at ~8 workgroups/WGP on RDNA3.5, so weight-read
     latency stalled. 1 KB lifts that → **another ~2×** (1054→520 ms at M=256,
     TTFT 133→48 ms). Sweep at M=256: 2048→243, 512→313, **256→492**, 128→408
     tok/s. Note the read pattern was *already* coalesced (identical to the
     192 GB/s decode matvec) — occupancy, not coalescing, was the wall.

  Net: **2842→520 ms (5.5×) at M=256; 481→48 ms (10×) TTFT**.
- Bind groups + scratch buffers are cached (built once, lazily; only
  `m_rows`/embeddings/cos-sin are rewritten per call) — avoids a ~70 MB
  realloc + ~208 bind-group rebuilds per call (matters for repeated/server
  prefills). It did not by itself move single-call TTFT — the floor was GPU
  occupancy, fixed above.

### iGPU chat fast-lane — wired into the server (new)

The resident GPU engine is now reachable from real chat requests, not just
tests. Build with `--features gpu` and start with `ZLLM_GPU=1`: at startup the
server loads a `GpuModel` from the configured GGUF (logged "chat fast-lane
enabled") and stores it on `AppState`; it's reloaded on model swap.

- Both the blocking (`generate_blocking`) and **streaming (`chat_stream`,
  SSE)** chat paths gain a GPU fast-path that early-returns (same shape as the
  spec-decode redirect) when a request is on the **fast lane** — inspection off
  (`POST /v1/inspect/enabled {false}`), no spec-decode / PLD / early-exit /
  grammar, prompt 1..=512 tokens. The whole request runs on the iGPU:
  `prefill_forward` fills the resident KV cache, then GPU decode (greedy uses
  the 4-byte argmax readback; sampling reads full logits). Streaming emits the
  standard per-token SSE deltas + final chunk + `[DONE]`. The candle pool path
  is untouched for everything else.
- Verified end-to-end against the live server (Llama-3.2-1B, 8060S): coherent
  blocking + streaming completions, correct `finish_reason`/usage, proper SSE
  framing; **warm prefill 487 tok/s** in-process (cold 205 — the bind-group
  cache earning its keep across requests), decode ~42–54 tok/s under server
  load.
- Serialized through a `Mutex<Option<GpuModel>>` (one resident KV cache).
  Explicit fast-lane trades: no candle prefix-cache reuse (re-prefills each
  request — best for cold/long prompts, not cached multi-turn) and no
  inspection hooks. All feature-gated; default (CPU) build unchanged.

### Also folded in (in-progress backend work)

AVX-512 Q4_K `vec_dot` + 8×8 repack kernels (CPU, gated off by default),
custom CPU SDPA, prompt-lookup / speculative-decode scaffolding, and
REST / chat-UI / goal-manager / metrics updates.

## v0.7.0 — 2026-05-22

First release since v0.1.3. Six unreleased milestones fold in here.
Brief recap; see commit history for full detail.

### Major features

- **v0.2 — memory management overhaul**: `MemoryStore` gains byte budgets per category, TTL + pin flags, write quotas, per-category and per-tag indexes. `GoalManager` pins Goals and active Tasks so they survive `Context` write storms. Six new prometheus metrics on the `/metrics` endpoint (`zllm_memory_bytes_used`, `zllm_memory_entries{category}`, etc.). Lazy eviction, no background tasks.

- **v0.3 — installed-app stance cleanup**: drop SaaS vestiges. `tenant_id` removed from `MemoryMetadata` / `HookContext` / `InferenceRunner::generate` / `build_injection_vector*` / `query_by_tenant`. Delete `proto/admin.proto`, `src/control_plane/router.rs`, `src/control_plane/gateway.rs`, `src/memory/isolation.rs` (TenantMemoryPool). Drop unused CLI subcommands (`Tenants`, `Hooks`, `Bench`, `Metrics`). Net: 342 lines deleted.

- **v0.4 — REST-first**: drop gRPC entirely (no more `tonic` / `prost` / `tonic-build` / `protoc` / `build.rs` / `proto/`). New OpenAI-compatible REST surface on the existing axum server: `/v1/models`, `/v1/chat/completions` (with SSE streaming), `/v1/completions`. Goal CRUD over REST: `/v1/goal/{state,set,list,current,task,status}`. Embedded chat UI served at `/` (vanilla HTML/CSS/JS, sidebar shows current goal/tasks/status, localStorage history).

- **v0.5 — inference performance + mock-data audit**: 
  - Wire the project sampler into the chat path (was greedy-only — `temperature`/`top_p`/`top_k` were parsed and ignored).
  - `[profile.release]` tuning: `lto = "fat"`, `codegen-units = 1`, `strip = true`.
  - `.cargo/config.toml` with `target-cpu=native` for local builds; CI release artifacts override `RUSTFLAGS=""` to stay portable.
  - `LogitFSM` ban-mode wired into chat (`grammar = "ban:128001,128009"` etc.) — was completely unwired before.
  - `ConfidenceHead` real IPR-based implementation (was `false` always); `HookContext.current_confidence` updated per layer; `EarlyExitHook` now actually fires.
  - `DifficultyEstimator` real implementation; wired into runner's non-adaptive branch.
  - Fix download bug where Chrome saved the chat UI HTML instead of rendering it (`Content-Disposition: inline` + `X-Content-Type-Options: nosniff`).

- **v0.7 — Option E partial-fork**:
  - Vendor `candle_transformers::models::quantized_llama` into `src/backend/candle/quantized_llama_fork.rs` (MIT/Apache-2.0) to expose per-layer access that upstream keeps private.
  - New `ModelWeights::forward_with_callback` fires `(layer_idx, &Tensor)` after each transformer block.
  - New `CandleCpuBackend::forward_logits_with_observer` lets the chat path observe the residual stream mid-forward.
  - Chat-prefill capture wired: every chat completion now writes a mean-pooled final-layer hidden state into `MemoryStore` as a `Context` entry. `MemoryStore` is no longer goal-manager-write-only.

### Mock-data scorecard (vs v0.1.3)

| Was mocked | v0.7.0 status |
|---|---|
| `LogitFSM::apply_mask` / `advance` no-ops | **Fixed** (ban-mode real) |
| `ConfidenceHead::should_exit → false` | **Fixed** (IPR signal) |
| `runner.rs` fake confidence ramp | **Fixed** (real per-loop estimate) |
| `HookContext.current_confidence` static 0.0 | **Fixed** (Cell, updated per layer) |
| `DifficultyEstimator::estimate → 1` | **Fixed** (inverse-confidence buckets) |
| `tenant_id`-scoped APIs | **Deleted** (installed-app stance) |
| Per-layer hooks fire on chat | **Half-fixed** (capture wired; inject still mutation-only via forward; replacement is Phase 3) |
| `CandleCpuBackend::forward_layer → identity` | Still mocked — needs runner integration |
| `CandleCpuBackend::compute_logits → zeros` | Still mocked — same |
| `runner.rs vec![0.1f32; …]` placeholder | Still mocked — needs `Backend::embed_tokens` |
| `GoalManager` zero-vector storage | Still mocked — needs an encoder pass |

### Surface today

Single CLI: `zllm serve` (REST + chat UI) and `zllm generate` (one-shot CPU inference).

REST endpoints on `:8080`:
- `GET /`, `/health`, `/v1/info`, `/metrics`
- `GET /v1/models`
- `POST /v1/chat/completions` (stream + non-stream)
- `POST /v1/completions`
- Goal CRUD: `GET /v1/goal/state`, `GET /v1/goal/list`, `POST /v1/goal/{set,current,task,status}`, `PATCH /v1/goal/task/{id}`

### Tests

60 passing: 39 lib + 21 integration smoke. Three new test modules added since v0.1.3 (`LogitFSM`, `ConfidenceHead`, `DifficultyEstimator`); existing memory_store / goal_manager / sampler / inspection tests preserved.

### Build

- `cargo build --release` ≈ 2-4 min (LTO fat is expensive but worth it).
- No more `protoc` build dependency.
- `.cargo/config.toml` opts local builds into `target-cpu=native`; CI releases override.

## v0.1.3 — 2026-05-13

Workflow-only release. No code changes from v0.1.0.

- Linux CUDA: install `cuda-nvrtc-dev-12-5` alongside the existing cublas/curand dev libs. candle-kernels links `-lnvrtc` at build time (not just at runtime); the dev variant provides the `.so` symlink.
- Windows CUDA: expand `Jimver/cuda-toolkit` `sub-packages` to include `nvrtc`, `nvrtc_dev`, `cublas`, `cublas_dev`, `curand`, and `curand_dev`. On Windows, Jimver maps these symbolic names to the correct NVIDIA installer component IDs cleanly (unlike Linux, where CUDA 12's `libcublas-*`/`libcurand-*` rename broke the same symbolic mapping). The Windows installer didn't fail when these weren't requested — it simply omitted them, and the linker found `cuda.lib` / `cudart.lib` only.

## v0.1.2 — 2026-05-13

Workflow-only release. No code changes from v0.1.0.

- Release workflow: drop the `macos-13` (Intel) matrix entry. The runner pool for hosted `macos-13` has been exhausted on every run we've attempted (queue time > 1 hour both times), blocking the release job from starting. macOS aarch64 (Apple Silicon, Metal) coverage remains. Intel macOS support can return via cross-compilation in a later release if there is demand.
- Linux CUDA: install `libcublas-dev-12-5` and `libcurand-dev-12-5` via apt after the `Jimver/cuda-toolkit` step. CUDA 12 renamed these packages from `cuda-cublas-*`/`cuda-curand-*` to the `lib*-*` naming; the action's `sub-packages` parameter cannot express the new names. `candle-kernels` links `-lcublas -lcurand -lcublasLt` at build time, so the `-dev` packages are required (the runtime-only `lib*-12-5` packages are not enough).
- Windows CUDA: add `ilammy/msvc-dev-cmd@v1` before the build step so `cl.exe` is on PATH. `nvcc` shells out to MSVC for host-side compilation of `.cu` files; without it, every kernel `.cu` translation unit fails with "Cannot find compiler 'cl.exe' in PATH".

## v0.1.1 — 2026-05-13

Workflow-only release. No code changes from v0.1.0.

- Release workflow: trim `Jimver/cuda-toolkit` sub-packages to `nvcc` + `cudart` only. The renamed CUDA 12 apt packages (`libcublas-*`/`libcurand-*` in place of `cuda-cublas-*`/`cuda-curand-*`) were causing the Linux CUDA build to fail at apt-install time. cudarc dlopens cublas/curand at runtime, so the build does not need them.
- Release workflow: set `CUDA_COMPUTE_CAP=89` (Ada Lovelace) for both Linux and Windows CUDA builds. CI runners have no GPU, so `candle-kernels`' build script panics when it tries to detect compute capability via `nvidia-smi`. CUDA binaries produced by v0.1.x target sm_89 and rely on PTX JIT for forward compatibility.
- Release workflow: gate the release job on `!cancelled()` instead of implicit `needs: build` success — partial matrix success now ships partial artifacts instead of dropping everything.

## v0.1.0 — 2026-05-13

First tagged release of zllm.

### Engine

- 3-zone `InferenceRunner` (`src/engine/runner.rs`): zone 1 encode (layers 0..8) → zone 2 budgeted reasoning loops (layers 8..8+N) → zone 3 output (layers 8+N..32).
- White-box mid-layer hooks: steering, early-exit, memory-inject; ordered dispatch via `HookRegistry`.
- Adaptive `ReasoningBudget` with per-token importance scoring; configurable max loops and memory ceiling.
- FSM grammar-constrained decoding (`engine/logit_fsm.rs`): `json_schema`, `regex`, `bnf`.
- Sampler with temperature / top-k / top-p (`engine/sampler.rs`).

### Memory

- `MemoryStore` (`src/engine/memory_store.rs`): in-memory, tenant-scoped, with category / tag / similarity / tenant queries and per-request inspection traces.
- Categories: `Finding`, `Context`, `Pattern`, `Correction`, `Knowledge`, plus new `Goal`, `Task`, `Status` variants for agent-state continuity.
- Cross-request injection: `build_injection_vector` (legacy) and `build_injection_vector_by_categories` (new, category-aware).
- Zero-norm dilution guard: entries with empty vectors are skipped from latent inject so they don't dilute real memories.

### Control plane

- New: **`GoalService`** in `proto/control.proto` exposing `SetGoal / ListGoals / SetCurrentGoal / AddTask / UpdateTask / SetStatus / GetState`. Installed-app, single scope — no `tenant_id` on the new API surface.
- `GoalManager` (`src/control_plane/goal_manager.rs`): owns goal/task/status state, tag-encoded conventions, UUID-based IDs, prompt-prefix builder.
- `GetState` returns a `prompt_prefix` field — ready to prepend to user prompts once the inference path is wired up.

### Backend

- `CandleCpuBackend` works against Llama 3.2 1B Q4_K_M GGUF (and other GGUF models).
- Feature flags: `cuda` (Linux/Windows) and `metal` (macOS) for accelerated builds. CUDA binaries in this release are not runtime-tested in CI (GitHub-hosted runners have no GPU) — validate on a GPU box before relying on them.

### Server

- gRPC server (`tonic`, default port 50051) — `GoalService` is fully wired; `InferenceService.infer` is still a stub returning a placeholder response.
- REST server (`axum`, default port 8080) — `/health` and `/v1/info` available.

### CLI

- `zllm serve --config <toml>` — start REST + gRPC.
- `zllm generate --model <path> --prompt "..."` — one-shot CPU inference end-to-end (uses Candle's greedy argmax, not the project sampler yet).
- `zllm bench / hooks / tenants / metrics` — stub subcommands kept from earlier scaffolding.

### Tests

- 9 inference / hook / memory tests (`tests/test_real_inference.rs`) — model-gated; require `Llama-3.2-1B-Instruct-Q4_K_M.gguf` on disk.
- 9 unit tests for `GoalManager` and the new `MemoryStore` injection paths — all pass with `cargo test --lib`.

### Known limitations for v0.1.0

- `InferenceService.infer` does not yet call `InferenceRunner.generate` — the gRPC inference path returns a placeholder.
- `MemoryInjectHook` and the inline inject step at `runner.rs:91-105` are not yet collapsed (deferred to v0.2).
- `admin.proto` exists with no server impl (legacy, untouched in this release).
- No `zllm goal` CLI subcommand yet — interact via gRPC for now.
