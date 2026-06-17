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
| **decode fused forward** (16 layers, 1 cmd buffer) | **97 tok/s** | 1.2× wgpu (80), but **< llama's 201** |

Key Phase-2 levers (all bit-exact):

- **Prefill, 44% → ~80% of llama: register blocking.** A 128×128 output tile per workgroup (16
  wave32 subgroups, each register-caching a 2×2 fragment grid) was the decisive lever — *not* the
  exotic stuff. Double-buffering and bigger fragment grids plateaued ~17% of peak (occupancy-bound).
- **Decode matvec, 50 → 208 GB/s: word-loading.** Load a u32 quant word and process all its nibbles
  (32 weights per 8 loads) instead of one weight per load — issue-bound → bandwidth-bound.
- **Decode forward is barrier-bound.** `VK_NOBAR=1` (drop all barriers) → 94→373 tok/s, so the ~145
  per-token barriers cost ~8 ms (~55 µs each, full GPU drains on the AMD proprietary driver). The
  decode wall is the barrier *count* (9 sync points/layer; llama fuses to ~3), not the kernels.
  First fusion attempt (rmsnorm folded into each matvec) *backfired* — the 128k-row LM head's
  redundant per-workgroup reduction cost more than the barriers it saved.

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
- **Decode is where a lead is physically possible** (bandwidth wall ~280 tok/s; llama at 201 leaves
  headroom). The matvec already beats llama's effective bandwidth (208 vs 153 GB/s). But realizing
  it end-to-end is blocked by the **barrier wall** — reaching llama's ~3-sync-point/layer fusion is
  hard, multi-session kernel work, and the obvious fusions backfire. **zllm does not yet beat llama
  on decode (97 < 201).**
- **What zllm is:** a complete, faithful, from-scratch engine — CPU at parity, a bit-exact wgpu GPU
  path (decode/prefill/batched), a raw-Vulkan coopmat path (prefill ~80% of llama, decode matvec
  beating llama's bandwidth), wired into the chat server. Honest bottom line: **no iGPU metric beats
  llama yet**, but prefill is close and the decode path is understood (if not yet cracked).

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

# zllm Phase 2 (raw-Vulkan coopmat) — --features vulkan
cargo test --release --features vulkan --lib vk_coopmat_q4k_gemm_throughput -- --ignored --nocapture  # prefill GEMM
cargo test --release --features vulkan --lib vk_coopmat_prefill_projection  -- --ignored --nocapture  # prefill tok/s
cargo test --release --features vulkan --lib vk_decode_matvec_bandwidth     -- --ignored --nocapture  # decode GB/s
cargo test --release --features vulkan --lib vk_decode_projection           -- --ignored --nocapture  # decode matvec tok/s
cargo test --release --features vulkan --lib vk_fused_decode_throughput     -- --ignored --nocapture  # fused decode (VK_NOBAR=1 to isolate barriers)
```
