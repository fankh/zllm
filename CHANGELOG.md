# Changelog

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
