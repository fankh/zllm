# zllm v1.0 Plan — from beta to a real major version

Status: draft (2026-07-13). Owner: fankh.

## What "1.0" means for zllm

zllm 1.0 is **not** a smaller llama.cpp. It is the best *white-box* local
inference server for dense llama-family-class models on one box, with an
OpenAI-compatible surface a real client can point at **without surprises**.

A release may call itself 1.0 when all of the following hold:

1. **API honesty** — every OpenAI parameter we accept is honored; every one
   we don't is rejected loudly (400), never silently ignored.
2. **Any dense GGUF in scope loads from a single file** — no sibling
   `tokenizer.json`, no wrong-family chat template, no silent metadata
   defaults (registry: done in v0.9.x line).
3. **The context ceiling is the model's, not ours** — no hardcoded 4096.
4. **Green CI gates every merge** — the "all suites green + bit-exact
   parity" quality story is enforced by a machine, not by discipline.
5. **A user can install and update it** without cloning the repo.
6. **Stability tiers are declared** — the OpenAI surface is semver-stable;
   the white-box surface is explicitly experimental.

Everything else — more arches, more hardware, more tools — is post-1.0.

---

## M1 — v0.10 "no silent lies" (API completeness + CI)

The Tier-1 gaps clients hit on day one, plus the machinery that protects
every later milestone. **CI lands first, before any feature.**

- [ ] `.github/workflows/ci.yml`: `cargo fmt --check`, `clippy -D warnings`,
      `cargo test` (default + `gpu,vulkan` check) on push/PR. Cache cargo.
      (The real-model suite stays local-only; CI runs lib + smoke.)
- [ ] `stop: [...]` **string** sequences on chat + completions + CB lanes
      (detokenized rolling-window match, truncate at match).
- [ ] Sampling: `repeat_penalty` / `presence_penalty` / `frequency_penalty`
      + `min_p` in `engine/sampler.rs`; plumb through every decode loop
      (candle, GPU, VK, CB, spec-decode verify path).
- [ ] `logit_bias`, `seed` on the candle path (CB already has seed).
- [ ] Unknown/unsupported OpenAI params → 400 with the param name
      (deny-list the big ones: `tools`, `n>1`, `response_format` until real).
- [ ] `/v1/embeddings` (mean-pooled, L2-normalized — encoder already exists).
- [ ] `/tokenize` + `/detokenize`.
- [ ] Exit: a stock OpenAI Python client run against zllm passes a scripted
      conformance check (new `tests/test_openai_conformance.rs` against a
      spawned server).

## M2 — v0.11 "any GGUF, one file" (multi-arch + vocab)

Foundation already in the working tree (arch registry + Mistral, live-
validated). This milestone commits and completes it.

- [ ] Commit the arch-registry + Mistral WIP (blocked on review hold).
- [ ] `backend/candle/gguf_vocab.rs`: build the tokenizer from GGUF-embedded
      `tokenizer.ggml.*` (BPE + SPM). Sibling `tokenizer.json` becomes a
      fallback, not a requirement.
- [ ] Chat templates via `minijinja` rendering GGUF `tokenizer.chat_template`;
      `ChatFamily` heuristics demoted to fallback for template-less GGUFs.
- [ ] Stop tokens from GGUF-declared ids (`tokenizer.ggml.eos_token_id`,
      `eot_token_id`), vocab probing as fallback.
- [ ] Tier-A arch entries (Mistral ✓ via llama arch; SmolLM/TinyLlama/OLMo-2
      verify-only) + **Qwen2.5** (qkv-bias flag on the dense fork — first
      real ArchSpec flag; validate vs llama.cpp on the local 7B).
- [ ] Exit: Llama-3.2, Mistral-7B, Qwen2.5-7B all serve correctly from a
      bare `.gguf`, template + stops from the file itself.

## M3 — v0.12 "the model's context, not ours" — ✅ SHIPPED (v0.12.0)

- [x] `{arch}.context_length` + `rope.scaling.*` in `HParams`; RoPE tables
      sized/scaled (linear exact; YaRN = loud linear approximation until a
      validation model lands; llama3 scaling via `rope_freqs.weight`).
      Candle path; VK/GPU lanes keep their own caps for now.
- [x] `MAX_SEQ_LEN = 4096` killed: window = min(model, `max_seq_len`,
      `ZLLM_MAX_SEQ`); KV allocation follows.
- [x] (Discovered) chunked prefill — single-shot 16K prefill materialized
      a tens-of-GB attention matrix; `ZLLM_PREFILL_CHUNK` (512) bounds it.
- [~] KV q8_0: DEFERRED post-1.0 — f32 KV at 32K is ~2 GB/slot (1B) on
      128 GB target hardware; not the binding constraint.
- [x] Exit met: needle retrieved from a 16,504-token document on
      Llama-3.2-1B (finish=stop); parity suite still bit-exact.

## M4 — v0.13 "doesn't fall over" (hardening + hygiene)

- [ ] Robustness suite: truncated/corrupt GGUF, wrong-arch GGUF, 32
      concurrent chat requests, client disconnect mid-SSE (cancellation
      frees the slot), max_tokens=0/absurd, empty messages.
- [ ] Request timeouts + graceful 503 when the pool is saturated.
- [ ] Security defaults: bind 127.0.0.1 unless `ZLLM_BIND` set; optional
      `ZLLM_API_KEY` bearer check; document the trust model in README.
- [ ] Split the two monoliths (`gpu/mod.rs` 320 KB, `vulkan/mod.rs` 292 KB)
      into ctx/model/kernels/server modules — no behavior change, gated by
      the (by then) full CI + parity suites.
- [ ] CLI: honor temperature/top-k/top-p (the last TODO from the mock
      audit); fix SentencePiece per-token decode spacing.

## M5 — v1.0-rc "installable"

- [ ] Packaging: `Dockerfile` (CPU baseline), Windows installer or winget
      manifest, `install.ps1`/`install.sh` fetching the GitHub release.
- [ ] API freeze + stability tiers documented: `/v1/{chat/completions,
      completions,models,embeddings}` = stable; `/v1/{goal,inspect,cb,
      debug,*enabled}` = experimental (may change in 1.x).
- [ ] `scripts/bench.ps1` reproducing the BENCHMARKS.md llama.cpp
      comparison on one command.
- [ ] Docs pass: README quickstart ≤ 10 lines to first token; config
      reference current; SUMMARY.md updated; explicit non-goals section.
- [ ] Version 1.0.0 after an RC soak week of daily local use.

---

## Explicit non-goals for 1.0 (post-1.0 or never)

Multi-GPU / tensor parallel; LoRA adapters; multimodal; MoE (revert path
kept, `4956b60`); DeepSeek-MLA / SSM / hybrid arches; distributed serving;
multi-model residency (swap stays); tool/function calling (needs a design
pass — candidate for 1.1); JSON-schema grammar compilation onto the regex
DFA (candidate for 1.1).

## Risks

- **Sampling penalties across five decode loops** is the sneaky-large item
  in M1 — the loops share no abstraction. Mitigation: extract a
  `DecodeState` (penalty history + stop matcher) used by all loops first.
- **minijinja template edge cases** (Jinja-isms in the wild). Mitigation:
  golden-file tests rendering each supported family's template against
  known-good llama.cpp output.
- **Long context × prefix cache × head-major KV** interactions on the VK
  lane. Mitigation: candle-first delivery; VK lane gates on parity tests
  at depth.
- **Scope creep via arch requests.** The registry makes additions cheap —
  the plan only *requires* llama + Mistral + Qwen2.5 for 1.0.

## Effort shape (single developer + agent)

M1 ≈ 1 week (CI half a day; penalties/stops are the bulk).
M2 ≈ 1 week (vocab + templates 2–3 days; Qwen2.5 fork 1–2 days + A/B).
M3 ≈ 1 week (candle path days; VK follow-on can slip past 1.0 as candle-only).
M4 ≈ 3–4 days + the monolith split as background work.
M5 ≈ 2–3 days.
Realistic calendar: **4–6 weeks** to v1.0.0.
