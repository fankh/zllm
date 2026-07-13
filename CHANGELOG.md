# Changelog

## v0.12.0 — 2026-07-13 (V1_PLAN M3: "the model's context, not ours")

Exit criterion met live: Llama-3.2-1B retrieved a planted needle
(codename) from a **16,504-token** document with finish_reason=stop —
4× the old hardcoded ceiling, past the pre-scaling 8K boundary.

- **MAX_SEQ_LEN = 4096 is dead**: effective window = min(GGUF
  `{arch}.context_length`, config `model.max_seq_len`, `ZLLM_MAX_SEQ`),
  bounded deliberately — candle's KvCache preallocates the full window
  per layer per slot (32K on the 1B ≈ 2.1 GB/slot).
- **RoPE scaling from the GGUF**: linear `rope.scaling.factor` honored;
  Llama-3.1/3.2 "llama3" scaling applied via the `rope_freqs.weight`
  tensor (per-dim frequency divisors — required for coherence past 8K);
  YaRN approximated as linear with a LOUD load-time warning until a
  validation model lands.
- **Chunked prefill** (ZLLM_PREFILL_CHUNK, default 512): the single-shot
  forward materialized the full (n_head, seq, kv) attention matrix —
  tens of GB at 16K; found the hard way when the first long-context run
  hung 15 minutes. Chunks feed against the growing KV via the fork's
  rectangular masks (the prefix-cache machinery), and the fused-vs-
  manual parity test still reads max |Δlogit| = 0.
- **`context_length_exceeded` 400** on over-long prompts (AppState
  model_ctx, refreshed on swap) instead of a mid-forward failure.
- Deferred from the M3 list: **q8_0 KV quantization** — at the target
  hardware's RAM (Strix Halo 128 GB) f32 KV at 32K is affordable
  (~2 GB/slot on 1B, ~8 GB on 8B); revisit post-1.0 or with the VK-lane
  long-context work.

## v0.11.0 — 2026-07-13 (V1_PLAN M2: "any GGUF, one file")

Exit criterion met live: a bare single-file Qwen2.5-7B GGUF (hardlinked
into an empty directory, no tokenizer.json) served chat end-to-end —
embedded BPE vocab, the model's own embedded ChatML template via
minijinja, and the GGUF-declared `<|im_end|>` stop ("Paris.", 2 tokens,
finish_reason=stop; multi-turn and stop strings verified).

- **Qwen2 family** (`f6f994d`): ArchSpec block flags — `qkv_bias`
  (additive Q/K/V biases) and `rope_neox` (non-interleaved rotary). The
  dense fork detects `general.architecture` and consumes the flags; one
  forward, parameterized. Local Qwen2.5-7B: coherent greedy at
  8.8 tok/s CPU on first try.
- **GGUF-native chat templates**: `tokenizer.chat_template` read at
  load/swap, rendered with minijinja (raise_exception + strftime_now);
  ChatFamily heuristics demoted to fallback. Llama-3.2's own template
  (self-date-stamping) now drives the chat endpoint — conformance suite
  green through it.
- **Declared stop ids**: `tokenizer.ggml.{eos,eot,eom}_token_id` unioned
  with the vocab probe in every decode loop.
- **Single-file loading** (`gguf_vocab.rs`): HF tokenizer built from the
  GGUF-embedded byte-level BPE vocab (per-family split regexes:
  llama-bpe, qwen2; family-default BOS when quants omit add_bos_token).
  Sibling tokenizer.json preferred; embedded vocab is the fallback at
  startup, swap, and CLI. **Oracle-enforced fidelity**: token-identical
  to tokenizer.json over adversarial samples for both llama-bpe (BOS)
  and qwen2 (no BOS). SPM vocabs (Llama-2, Mistral v0.3) still require
  tokenizer.json — loud, documented error.

## v0.10.0 — 2026-07-13 (V1_PLAN M1: "no silent lies")

The OpenAI surface now honors what it accepts and rejects what it can't
honor. CI gates every push. Exit criterion met: a scripted conformance
run against a live spawned server passes end-to-end
(`tests/test_openai_conformance.rs`, model-gated).

- **CI** (`.github/workflows/ci.yml`): lib + smoke tests and full compile
  surface (default and gpu,vulkan) on ubuntu + windows for every push/PR.
- **DecodeControl** (`engine/decode_ctrl.rs`): one per-request owner of
  penalties, logit_bias, seeded RNG, and stop-string state, wired
  identically into every decode loop — candle blocking + streaming, GPU
  and VK fast lanes, PLD/spec commit paths.
- **`stop` strings** (string or array): matched on a re-decoded tail
  window (never per-token decodes — SPM-safe), trimmed from blocking
  output, checked before each streamed chunk so a completed stop never
  reaches the client.
- **Sampling**: `presence_penalty` / `frequency_penalty` (OpenAI),
  `repeat_penalty` (llama.cpp-style, alias `repetition_penalty`),
  `min_p`, `logit_bias` (±100 clamp), `seed` on all lanes (was CB-only).
  Fast paths stay honest: GPU argmax readback only when nothing adjusts
  the distribution; spec-decode/PLD gate off under penalties/bias
  (drafts verify against unadjusted rows); the CB lane rejects to the
  candle path rather than silently dropping parameters.
- **Loud 400s** for recognized-but-unsupported params: `tools` /
  `tool_choice`, `n > 1`, `best_of > 1`, non-text `response_format`.
- **New endpoints**: `/v1/embeddings` (mean-pooled, L2-normalized — same
  space as the GoalManager encoder), `/tokenize`, `/detokenize`
  (llama.cpp-compatible shapes).
- Live-verified on Llama-3.2-1B: stop strings cut at " four" mid-count;
  seed 42 reproduces byte-identical sampled output; banning the first
  answer token via logit_bias reroutes the greedy continuation.

## v0.9.3 — 2026-07-13

Dead-code purge driven by the project's own history (~500 lines net).

- **Deleted, zero callers:** `ModelWeights::from_ggml` (pre-GGUF GGML
  loader inherited from candle), `engine::layer_stepper`, the
  `create_backend` factory (its unknown-string arm silently served random
  DummyBackend logits), `MemoryStore::decay_relevance`, and the fake
  `Backend` trait surface (`alloc_kv_block`/`free_kv_block` counter no-ops,
  `read_hidden_state`/`write_hidden_state` zeros) plus `BlockId`.
- **Test-only public API removed:** `query_by_similarity`,
  `build_injection_vector_by_categories` (never-wired successor; the hook
  still uses the legacy builder), `EarlyExitHook` (prod early exit is the
  `ConfidenceHead` closure over `forward_logits_early_exit`; registry
  EarlyExit mechanics covered by a local test hook). `DeviceInfo` slimmed
  to `{name, backend}` — the fp8/fp4/memory fields were never read.
- **Vestiges:** `generate_token` returns a bare token id (the empty
  hidden-states Vec was reserved for a fork that shipped in v0.7);
  `QuantConfig` dropped from `load_model` (all impls ignored it); the CLI's
  silent HF-download fallback (gated meta-llama repo, 401 without a token)
  is now an explicit `--tokenizer` error; GoalManager's zero-vector
  fallback width follows the loaded model's real hidden size.
- **MoE cut from the fork:** Mixtral-class support (`MlpOrMoe::MoE`,
  expert-tensor loading) was inherited from candle but out of scope for
  the dense 1B–8B installed-app stance (GPU/VK engines are dense-only).
  `expert_count > 1` GGUFs now fail loudly at load instead of running on
  an untested CPU-only path. Hot-loop match indirection removed.

## v0.9.2 — 2026-07-13

Follow-up audit of the serving stack for residual mock/temp semantics; the
three findings that affect correctness are fixed.

- **Goal vectors re-encoded on model swap** (regression guard for v0.9.1):
  goal/task/status embeddings live in the loaded model's space and width, so
  the swap handler now calls `GoalManager::reencode_all()` after the pool +
  tokenizer are updated. Previously they were kept verbatim on the (now
  false) premise that they were zero-padded placeholders — a 1B→3B swap
  would silently zero every goal-similarity score.
- **Runner decode is real autoregression**: each generated token is fed back
  and the extended sequence re-forwarded through all layers before the next
  sample; previously every token was drawn from one frozen logit vector.
  EOS comes from `with_eos_tokens` (tokenizer-derived) instead of the
  hardcoded Llama-2 id 2. Pinned by
  `test_runner_decode_matches_greedy_continuation`: with reasoning_layers=0
  the runner's greedy output is token-identical to the fused path's
  stateless greedy continuation.
- **Stop tokens + chat template derived from the tokenizer, not hardcoded
  Llama-3 ids**: new `LlamaTokenizer::stop_token_ids()` (EOS + whichever of
  `<|eot_id|>`/`<|eom_id|>`/`<|im_end|>`/`<|end|>`/`<|endoftext|>` exist in
  the vocab) replaces the `128009`/`unwrap_or(128001)` pattern at ~20 call
  sites across rest.rs and the CLI; the CB server's two-scalar stop API now
  receives vocab-derived ids. `render_chat_prompt` detects the template
  family from the vocab (Llama-3 headers / ChatML / Llama-2 `[INST]`)
  instead of always emitting Llama-3 headers — llama-arch ChatML finetunes
  (Hermes, TinyLlama-Chat) now get the right prompt format and actually
  stop at `<|im_end|>`.

## v0.9.1 — 2026-07-13

Mock-data cleanup: the four "Still mocked" rows from the v0.7 scorecard are
now real. No placeholder tensors remain on the `Backend` trait surface.

- `Backend::embed_tokens` (new): real token-embedding lookup. The runner's
  Zone 1 now seeds the residual stream with model embeddings instead of the
  `vec![0.1f32; …]` fill.
- `CandleCpuBackend::forward_layer`: real per-layer transformer forward via
  the fork's new `forward_one_layer` (causal mask from pos 0, layer-local KV
  reset before/after so standalone passes never pollute the decode cache).
  Was: identity. Out-of-range layer is now a hard error.
- `CandleCpuBackend::compute_logits`: real final-norm + LM-head projection via
  the fork's `logits_from_hidden`. Was: zeros.
- Parity test (`test_manual_layer_drive_matches_fused_forward`): manual
  `embed_tokens → forward_layer × n_layers → compute_logits` reproduces the
  fused `forward_logits` pass **bit-exact** (max |Δlogit| = 0) on
  Llama-3.2-1B Q4_K_M.
- `Backend::n_layers` (new): runner zone boundaries clamp to the loaded
  model's real depth instead of assuming 32 layers.
- `Backend::forward_layer` takes `&mut self` (real layers touch KV/mask
  state); `InferenceRunner::generate` returns `Result` and propagates backend
  errors instead of unwrapping.
- `GoalManager`: goal/task/status entries are embedded by a real encoder pass
  (tokenize → mean-pool token embeddings → L2 normalize) instead of storing
  `vec![0.0; d_model]`. Encoder is injected from `main.rs`, `try_lock`s across
  the backend pool so goal CRUD never queues behind a generation, and falls
  back to a zero vector (cosine 0 — honest "no embedding") when no model is
  loaded. Disk-restored entries are re-encoded on startup.
- `test_hooks_on_real_backend` modernized: steering asserts the
  `residual_delta` write-back contract (v0.9 design) instead of the removed
  mutate-on-`fire` behavior.

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
| `CandleCpuBackend::forward_layer → identity` | **Fixed in v0.9.1** (real block forward, bit-exact vs fused pass) |
| `CandleCpuBackend::compute_logits → zeros` | **Fixed in v0.9.1** (final norm + LM head) |
| `runner.rs vec![0.1f32; …]` placeholder | **Fixed in v0.9.1** (`Backend::embed_tokens`) |
| `GoalManager` zero-vector storage | **Fixed in v0.9.1** (mean-pooled embedding encoder) |

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
