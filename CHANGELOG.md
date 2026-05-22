# Changelog

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
