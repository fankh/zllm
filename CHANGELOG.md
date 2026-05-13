# Changelog

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
