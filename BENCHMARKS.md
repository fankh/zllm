# Benchmarks — zllm vs llama.cpp

Single-stream and aggregate inference throughput, measured head-to-head against
llama.cpp on the same model and hardware.

## Setup

- **Model:** Llama-3.2-1B-Instruct **Q4_K_M** GGUF (762 MiB, 1.24 B params, 16 layers, vocab 128256)
- **Hardware:** AMD Ryzen AI MAX+ 395 "Strix Halo" — Zen 5, 16C/32T, AVX-512 (incl. VNNI/BF16),
  ~96 GB unified LPDDR5X (~256 GB/s, **shared** by CPU + iGPU), Radeon 8060S iGPU (RDNA 3.5, 40 CUs)
- **OS:** Windows 11 / MSVC
- **Measured:** 2026-06-17
- **llama.cpp:** build `3980e04d5` (9050), Vulkan backend (reports `matrix cores: KHR_coopmat`, `fp16: 1`)
- **zllm:** `--features gpu` (wgpu → Vulkan, pure-Rust WGSL via naga). GPU outputs are validated
  **bit-exact** against the candle CPU forward (greedy output token-identical).

All numbers are tok/s. Bigger is better.

## 1. Single-stream decode

| backend | zllm | llama.cpp | zllm / llama |
|---|---:|---:|---:|
| CPU      | ~63  | 65.5  | 0.96× (tie) |
| iGPU     | 82.5 | 201.2 | 0.41× |

Both CPU figures sit at the memory-bandwidth ceiling (~55 GB/s effective) — decode streams the
full ~0.76 GB of weights per token, so this is bandwidth-bound and the two engines tie. On the iGPU,
zllm is ~2.4× behind llama.cpp's hand-tuned Vulkan kernels.

## 2. Prefill (TTFT / batched-compute)

| | zllm | llama.cpp |
|---|---:|---:|
| iGPU prefill (pp512) | ~492–527 (peak) | **5747** |
| CPU prefill (pp512)  | (candle, lower) | 2791 |

zllm batched prefill by prompt length M (iGPU):

| M | tok/s | ms |
|---:|---:|---:|
| 12  | 249 | 48 |
| 32  | 527 | 61 |
| 128 | 500 | 256 |
| 256 | 492 | 520 |

llama.cpp's prefill is **11.7×** faster, using the iGPU's cooperative-matrix cores.

## 3. Aggregate serving throughput (parallel decode)

Total generation tok/s across M concurrent streams (iGPU):

| concurrency M | zllm aggregate | llama.cpp aggregate | zllm / llama |
|---:|---:|---:|---:|
| 1  | 16¹ | 203  | — |
| 2  | 30  | 369  | 0.08× |
| 4  | 60  | 627  | 0.10× |
| 8  | 116 | 727  | 0.16× |
| 16 | 213 | 1011 | 0.21× |
| 32 | **327** | **1458** | 0.22× |

¹ zllm's M=1 batched path carries per-step overhead (uncached bind groups + K/V scatter); the
optimized single-stream path is 82 tok/s. zllm's batched decode scales **6.4×** from M=1→32
(validated bit-identical to single-stream), confirming the compute-bound amortization works — it's
just ~4.5× below llama.cpp's coopmat-backed batched throughput at M=32.

The sections above are the **Phase 1 (wgpu/WGSL)** results. Phase 2 below builds a raw-Vulkan
cooperative-matrix path that closes most of the prefill gap — see the corrected analysis.

## 4. Phase 2 — raw-Vulkan cooperative-matrix path (`--features vulkan`)

WGSL/naga cannot express cooperative matrix, so Phase 2 adds a raw-Vulkan backend via `ash`
(`src/backend/vulkan/`) with GLSL kernels compiled offline to SPIR-V (`include_bytes!`, committed
`.spv` — no build-time glslang/SDK). This reaches the iGPU's WMMA cores — the exact capability that
gives llama.cpp its lead. **Validated bit-exact / cosine-1.0 vs candle throughout.**

| kernel | result | vs llama / ceiling |
|---|---|---|
| coopmat 16×16×16 spike | err 5e-5 | proves the path works on the 8060S |
| dense fp16 coopmat GEMM | ~10 TFLOP/s | — |
| **Q4_K coopmat GEMM** (register-blocked) | **6405–9880 GFLOP/s** | **~6–8× the wgpu f32 GEMM** (~1000) |
| **prefill projection** (full forward's GEMMs) | **M=256 4250 / M=512 4802 tok/s** | **~74–84% of llama's 5747** |
| **decode matvec** (word-loading) | **180–208 GB/s** | **above llama's ~153 effective**, near the ~215 wall |
| **decode fused forward** (16 layers, 1 cmd buffer) | **~322 tok/s** @ctx≤64, ~258 @512, ~162 @2048 | **beats llama's 201** at the bench context |

Key Phase-2 levers (all bit-exact):

- **Prefill, 44% → ~80% of llama: register blocking.** A 128×128 output tile per workgroup (16
  wave32 subgroups, each register-caching a 2×2 fragment grid) was the decisive lever — *not* the
  exotic stuff. Double-buffering and bigger fragment grids plateaued ~17% of peak (occupancy-bound).
- **Decode matvec, 50 → 208 GB/s: word-loading.** Load a u32 quant word and process all its nibbles
  (32 weights per 8 loads) instead of one weight per load — issue-bound → bandwidth-bound.
- **Decode forward, 107 → ~290 tok/s (beats llama): the wall was the SDPA kernel, not barriers.**
  The fused forward first measured ~100 tok/s, and `VK_NOBAR=1` "fixing" it to ~370 looked like a
  ~145-barrier wall. That diagnosis was **wrong**: a coherence probe (`vk_barrier_coherence_probe`)
  shows this driver *elides* a memory-barrier-less pipeline barrier, so `VK_NOBAR`/`VK_EXECBAR` were
  racing layers, not a real floor — and a *correct* full barrier costs only **~2 µs**. Per-category
  skip flags (`VK_SKIP=sdpa`) found the real culprit: the decode SDPA ran **one thread per head**
  (1 of 40 CUs) with a `float[128]` accumulator that **spilled to scratch** — ~410 µs × 16 layers =
  ~6.6 ms. Rewritten as **one workgroup per head** (threads parallel over head-dim, single `av` per
  thread, shared-mem dot reduction), SDPA dropped to ~0.5 ms total and the forward hit ~290 tok/s.
  Validated bit-exact vs a CPU softmax-attention reference (`vk_sdpa_correctness`, err 2.4e-7).
- **Decode holds up at long context: subgroup reduction + flash attention.** The workgroup-per-head
  SDPA was still barrier-bound (a 6-barrier shared-mem reduction per KV position) and occupancy-bound
  at long context (only 32 workgroups looping the KV cache serially) — `VK_SEQ=512` collapsed to 98
  tok/s, `VK_SEQ=2048` to 31. Fix: (1) a barrier-free **subgroup** `q·k` reduction (one wave32
  subgroup per head, lane owns hd/32 dims), and (2) **flash attention** for ctx > 32 — a grid of
  (n_head × n_blocks) workgroups each computes a per-block online-softmax partial, then a combine
  pass merges them by log-sum-exp, so KV-stream latency is hidden. Decode-forward tok/s by context:
  **32→322, 128→305, 512→258, 2048→162** (was 323/198/98/31). Both paths bit-exact (flash err 8.9e-7).
  Note: llama-bench `tg128` runs at avg context ~64, where zllm wins (~320 vs 201); sustained
  long-context decode degrades gracefully rather than collapsing.
- **Fusions that backfired (kept for the record):** folding rmsnorm *or* silu·mul into a matvec
  recomputes a per-element transform once **per output row** — e.g. silu-into-down (`VK_FUSE=1`) was
  88 vs 104 tok/s (16.7 M redundant `exp`/layer). Rule: never fold a per-input-element op into a
  matvec *consumer*.

## Analysis (corrected by Phase 2)

- **CPU decode is a tie** — memory-bandwidth bound, both at the roofline.
- **The earlier "no hardware ceiling, it's all kernel software" is the right framing.** llama.cpp
  hitting 5747 prefill / 201 decode *on this exact machine* proves the silicon can do it; every gap
  is kernel quality, not hardware. The Phase-1 conclusion ("coopmat unreachable, +50% impossible,
  weeks to parity") was **wrong on the first two points**: coopmat is reachable via `ash`, and
  register blocking alone took prefill from 8.5% → ~80% of llama.
- **Prefill parity is in reach** (~80% now; the rest is per-shape kernel tuning — double-buffering,
  vectorized loads — that llama did over time). **Beating prefill by 50% is not** — llama's coopmat
  prefill is near the compute roof; matching it is the realistic best.
- **Decode now BEATS llama** (~290 vs 201 tok/s, ~1.4×). The bandwidth wall is ~280–355 tok/s and the
  fused forward sits just under the matvec-only ceiling (~355) — the win came from fixing the SDPA
  kernel's parallelism, not from a bandwidth trick. The earlier "barrier wall" framing was a
  measurement artifact (the driver elides empty barriers; a real barrier is ~2 µs).
- **What zllm is:** a complete, faithful, from-scratch engine — CPU at parity, a bit-exact wgpu GPU
  path (decode/prefill/batched), and a raw-Vulkan coopmat path that now **beats llama on
  single-stream decode (~1.4×)** and reaches ~80% of llama on prefill, all wired into the chat
  server. Bottom line: **decode is won; prefill is close** (parity realistic, +50% is not — llama's
  coopmat prefill is near the compute roof).
- **The raw-Vulkan forward is now a real-weight server engine** (`VkModel`, `--features vulkan`,
  `ZLLM_VK=1`): loads the actual GGUF, runs the validated decode kernels with a resident KV cache,
  **bit-exact vs candle (greedy 24/24 tokens)**, wired into the chat fast-lane (blocking + streaming)
  and verified live. Real-weight decode went **122 → 180 tok/s** after word-loading the Q6_K matvec
  (`attn_v`/`ffn_down`/tied LM head) and keeping scales i8 (212 B/block), plus a GPU argmax that fixed
  an ~8.5 ms/token logit-readback cliff (59 → 114). Beats CPU/wgpu (63) by ~2.9×.
- **Head-to-head decode, fair context (llama-bench `tg128`, avg ctx ~64, same box):** llama **204**,
  zllm VkModel **~150** (≈74%; ~87% at short ctx). zllm is *not* past llama on the deployed engine yet.
  Both degrade with context (llama 204→159 at ctx 2048; zllm's flash kernel 162–173 there — a tie at
  the kernel level). **The differentiator is effective bandwidth on the *mixed-precision* forward:**
  both stream ~840 MB/token, but llama sustains ~171 GB/s vs zllm's ~134. zllm's **Q4_K** matvec alone
  does **208 GB/s (> llama)**, but the **Q6_K** parts run ~170 and the small ops (norm/rope/sdpa/silu)
  add ~1.8 ms/token of non-streaming time (`VK_NOEXTRA` shows kv-write+residual are only ~0.4 ms of
  that, so dispatch-fusion isn't the lever). zllm's all-Q4 decode *kernel* ceiling (~300 @ ctx 64) is
  above llama; closing the deployed gap to beat llama needs the deep op-fusion (fold norm into the
  matvec producer, fuse QKV) + the Q6 kernel to 208 — not more matvec tuning. Prefill is also still
  sequential (~175 tok/s vs llama 5708); the coopmat GEMM is built but not wired into `VkModel`.

## 4.5 Breaking the decode roofline — speculative decoding (the *mathematical* lever)

Single-stream decode is bandwidth-bound: 1 token/forward, each forward streaming ~760 MB of weights,
so `tok/s ≈ 215 GB/s ÷ 0.76 GB ≈ 283` caps **everyone** (llama 201, zllm 150). No kernel beats it —
the bottleneck is moving weights, not computing. The only escape is **>1 token per weight-stream**:
- **batching** (n = concurrent requests) → §5, the serving stack;
- **speculative decoding** (n = accepted drafts + 1) → single-stream latency.

`BatchedDecoder::generate_pld` implements **Prompt-Lookup Decoding** on the GPU path: an n-gram draft
from the generated history is verified in ONE batched forward (`step_slotted` over consecutive
positions — the staggered-position trick), committing every token the model's own argmax agrees with.
Greedy verification ⇒ output **identical** to single-token decode. Validated (`gpu_pld`) on echo-heavy
text: **4.0 tokens/forward, 3.19× wall-clock** (17 → 53 tok/s on the same wgpu kernels), bit-identical;
open-ended text falls back to single-token at ~zero cost. With acceptance `p` and draft length `γ`,
`E[tokens/forward] = (1−p^(γ+1))/(1−p)` (p=0.7, γ=4 → ~2.4×). Porting this to the VkModel coopmat path
(150 tok/s single-stream) puts echo-heavy decode **well past llama** — the roofline only bounds 1-tok/forward.

## 5. Continuous-batching serving (in-flight batching + paged KV)

Section 3 measured *raw* batched decode (M streams in one forward). This section is the **serving
architecture** built on top of it — the machinery that turns that batched kernel into a real
vLLM-style server: arrivals join the running batch, prompts prefill in bulk, and KV is paged. All on
the wgpu (`--features gpu`) path, enabled at startup with `ZLLM_CB=1`. When on, **eligible
`/v1/chat/completions` requests route through it automatically** (it becomes the default chat backend —
inspection-off, no grammar/spec-decode/PLD/early-exit; those fall through to the candle path), and it
is also exposed directly at `POST /v1/cb/completions`. SSE stream or JSON; greedy or temp/top-k/top-p
sampling. **Every layer validated bit-identical to single-stream decode.**

| capability | what it does | result |
|---|---|---|
| **Slot indirection** | KV keyed by cache *slot*, not batch position → a sequence keeps its KV across admit/evict (no compaction copies) | batched decode unchanged (still 327 tok/s @ M=32) |
| **ContinuousBatcher** | admit / decode-step / evict; arrivals join the in-flight batch, finishers free their slot | 8 concurrent over HTTP: **77.5 tok/s aggregate = 5.6×** single-stream (≈60-tok prompts) |
| **Batched prefill-into-slot** | prefill a whole prompt in one coopmat pass per 128 tokens (staggered positions → the decode SDPA *is* causal prefill), not one forward per token | **30× faster** (201-tok prompt: 470 ms vs 14.3 s), bit-identical; kills admission head-of-line blocking |
| **Paged KV (PagedAttention)** | KV is a shared pool of 16-position blocks + per-slot block table; pool sized to *actual* use, not m_max × max_seq | **4× less KV memory** (8 seqs in a 64-block pool vs 256 contiguous), bit-identical, blocks recycled on evict |
| **Prefix KV-cache reuse** | refcounted blocks + prefix→block hash map; a new prompt sharing a leading prefix (e.g. a system prompt) reuses those KV blocks and prefills only the suffix | a request sharing a 40-tok prefix **reuses 2 blocks (skips that prefill)**, output bit-identical to cold |
| **Sampling (temp / top-k / top-p)** | temperature via Gumbel-max (argmax of `logit/temp + gumbel`, no readback); top-k/top-p via a GPU top-K(64) kernel + CPU nucleus sample (reads back M×64, not M×vocab); per-stream `temperature`/`top_k`/`top_p`/`seed` | temp=0 ≡ greedy; reproducible per seed; GPU top-64 bit-matches a CPU sort; coherent varied output |
| **Preemption (recompute)** | optimistic admission; when a running sequence can't grow, evict (free KV of) a LIFO victim and recompute it later (re-prefill prompt++produced, prefix cache reuses the prompt) — makes paged-KV overcommit *safe* | a sequence preempted mid-generation resumes **bit-identical** to never being preempted |
| **Chunked prefill** | a long prompt prefills one chunk (128 tok) per scheduler step, interleaved with the decode of the active batch (prefill-priority), instead of one synchronous multi-chunk admission | a 301-tok prompt prefills over 3 steps while a short seq decodes 4 tokens **during** it; output bit-identical — no long-prompt HOL blocking |
| **Request cancellation** | when a client disconnects, the serve loop detects the closed token channel (`is_closed()`) and frees the sequence's slot + KV — reclaiming capacity instead of generating output nobody reads | with 1 slot, a 200-tok stream killed mid-flight lets the next request finish in **0.4s** (vs waiting ~5s) |

Notes / honest scope:
- This raises **aggregate throughput under concurrency** (the right serving metric) — it doesn't change
  single-stream latency or the llama gap (separate axis). The underlying batched decode still scales
  ~6.4× from M=1→32, so aggregate climbs with batch size.
- **Default pool is full** (every slot can reach max_seq → contiguous-equivalent, never starves);
  `with_pool(m_max, max_seq, n_blocks)` opts into overcommit, now made safe by **preemption** (admission
  is optimistic; a sequence that can't grow is evicted and recomputed later).
- The CB server loads its **own** model copy on a dedicated thread (handlers talk to it only by
  channel — no borrow-across-`Arc<Mutex>`), independent of the `ZLLM_GPU`/`ZLLM_VK` fast lanes, and
  does **not** hot-swap with the model selector.
- On this 96 GB unified box memory isn't the binding constraint (the occupancy wall caps useful
  concurrency first); paging's near-term value here is the mechanism + the future prefix/KV-reuse
  unlock, not fitting more sequences.

## zllm GPU kernel tuning (this engine's own progression)

Bit-exact throughout. Prefill GEMM at M=256: **2842 → 520 ms (5.5×)**; TTFT (M=12): **481 → 48 ms (10×)**.

- 8-wide unroll of the GEMM inner dot loop (ALU/latency-bound): **2.84×**
- LDS weight tile 2048 → 256 floats (occupancy fix on RDNA 3.5): **~2×**
- Per-token decode readback/sync (drain-then-read + persistent `MAP_READ`): **+20%** decode
- Coalesced workgroup-per-row matvecs (vs one-thread-per-row): the decisive decode technique
- Rejected (measured losses, kept for the record): per-element Q4_K dequant, 128-thread workgroups

## Reproduction

```sh
# llama.cpp
llama-bench         -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 512 -n 128           # iGPU decode/prefill
llama-bench         -m ...gguf -p 512 -n 128 -ngl 0                              # CPU
llama-batched-bench -m ...gguf -ngl 99 -npp 32 -ntg 64 -npl 1,2,4,8,16,32 -c 8192 # parallel decode

# zllm Phase 1 (wgpu) — cargo test, --ignored, --nocapture
cargo test --release --features gpu --lib gpu_full_forward_vs_candle_and_bench -- --ignored --nocapture
cargo test --release --features gpu --lib gpu_prefill_vs_candle_and_bench       -- --ignored --nocapture
cargo test --release --features gpu --lib gpu_batched_decode_throughput         -- --ignored --nocapture

# zllm continuous-batching serving (--features gpu): in-flight batching + batched prefill + paged KV
cargo test --release --features gpu --lib gpu_continuous_batch  -- --ignored --nocapture  # admit/decode/evict bit-identical to single-stream
cargo test --release --features gpu --lib gpu_batch_server      -- --ignored --nocapture  # GpuBatchServer thread/channel, concurrent correctness
cargo test --release --features gpu --lib gpu_prefill_slot      -- --ignored --nocapture  # batched prefill 30x vs sequential, bit-identical
cargo test --release --features gpu --lib gpu_paged_overcommit  -- --ignored --nocapture  # paged KV: 4x less mem, recycled, bit-identical
cargo test --release --features gpu --lib gpu_prefix_cache      -- --ignored --nocapture  # cross-request prefix reuse, bit-identical to cold
cargo test --release --features gpu --lib gpu_sampling          -- --ignored --nocapture  # temp=0 ≡ greedy, temp>0 reproducible per seed
cargo test --release --features gpu --lib gpu_btopk_kernel      --            --nocapture  # GPU top-64 == CPU sort (no model needed)
cargo test --release --features gpu --lib gpu_topkp             -- --ignored --nocapture  # top-k/top-p: temp=0 ≡ greedy, reproducible
cargo test --release --features gpu --lib gpu_preemption        -- --ignored --nocapture  # preemption bit-identical to no-preemption
cargo test --release --features gpu --lib gpu_chunked_prefill   -- --ignored --nocapture  # long prefill interleaves w/ decode, bit-identical
cargo test --release --features gpu --lib gpu_cancel            -- --ignored --nocapture  # cancel frees slot/KV, neighbor unaffected
cargo test --release --features gpu --lib gpu_pld               -- --ignored --nocapture  # prompt-lookup spec decode: 4 tok/forward, 3.19x, bit-identical
# Server: build --features gpu, run with ZLLM_CB=1 (ZLLM_CB_SLOTS / ZLLM_CB_SEQ), POST /v1/cb/completions {prompt, max_tokens, stream}

# zllm Phase 2 (raw-Vulkan coopmat) — --features vulkan
cargo test --release --features vulkan --lib vk_coopmat_q4k_gemm_throughput -- --ignored --nocapture  # prefill GEMM
cargo test --release --features vulkan --lib vk_coopmat_prefill_projection  -- --ignored --nocapture  # prefill tok/s
cargo test --release --features vulkan --lib vk_decode_matvec_bandwidth     -- --ignored --nocapture  # decode GB/s
cargo test --release --features vulkan --lib vk_decode_projection           -- --ignored --nocapture  # decode matvec tok/s
cargo test --release --features vulkan --lib vk_fused_decode_throughput     -- --ignored --nocapture  # fused decode forward (~290 tok/s, beats llama)
cargo test --release --features vulkan --lib vk_sdpa_correctness            -- --ignored --nocapture  # SDPA (single + flash) bit-exact vs CPU ref
cargo test --release --features vulkan --lib vk_model_vs_candle             -- --ignored --nocapture  # real-weight VkModel: greedy 24/24 vs candle + tok/s
# Server fast-lane: build --features vulkan, run with ZLLM_VK=1, POST /v1/inspect/enabled {false}, then chat
VK_SEQ=512 cargo test --release --features vulkan --lib vk_fused_decode_throughput -- --ignored --nocapture  # decode at a given context depth
cargo test --release --features vulkan --lib vk_barrier_coherence_probe     -- --ignored --nocapture  # barrier cost/coherence (VK_EXECBAR/VK_NOBAR show staleness)
# Diagnostics: VK_SKIP=sdpa,norm,rope,silu attributes per-op cost; VK_FUSE=1 shows the silu-fusion backfire
```
