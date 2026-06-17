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

## Analysis

- **CPU decode is a tie** because it's memory-bandwidth bound and both engines are at the roofline.
- **Every iGPU compute-bound regime favors llama.cpp** (prefill 11.7×, batched decode 4.5×) for a
  single structural reason: llama.cpp's Vulkan backend uses the iGPU's **cooperative-matrix cores**
  (`KHR_coopmat`, fp16), while zllm's wgpu → WGSL path **has no access to cooperative matrix** and is
  confined to f32 ALU throughput.
- **+50% over llama.cpp is not reachable on this hardware** with the wgpu approach. Decode has no
  headroom above the bandwidth roofline llama already approaches; the compute-bound regimes need
  matrix-core hardware WGSL can't express. Matching llama there would require reimplementing the GPU
  path in raw Vulkan (`ash`) + hand-written SPIR-V with coopmat intrinsics — weeks of work to reach
  *parity*, not a lead.
- **What zllm is:** a complete, faithful, from-scratch engine — CPU at parity with llama.cpp, a
  bit-exact GPU path (decode, prefill, and batched serving), wired into the chat server. Its
  differentiation is the white-box layer (per-layer inspection, memory hooks, confidence/early-exit,
  goal manager), not raw GPU throughput.

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

# zllm (cargo test, --ignored, --nocapture)
cargo test --release --features gpu --lib gpu_full_forward_vs_candle_and_bench -- --ignored --nocapture
cargo test --release --features gpu --lib gpu_prefill_vs_candle_and_bench       -- --ignored --nocapture
cargo test --release --features gpu --lib gpu_batched_decode_throughput         -- --ignored --nocapture
```
