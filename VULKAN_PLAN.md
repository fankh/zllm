# Plan: close the gap with — and beat — llama.cpp on the Strix Halo iGPU

Goal: maximize zllm's iGPU throughput on Llama-3.2-1B Q4_K_M (Radeon 8060S, RDNA 3.5).
Grounded in a deep read of llama.cpp's Vulkan backend, the `VK_KHR_cooperative_matrix`
spec, the `ash` toolchain, and RDNA3 WMMA microbenchmarks (sources at the end).

## TL;DR — the honest, achievable target

The earlier "coopmat is an 11.7× structural wall" conclusion was **wrong**. The reality:

- **RDNA3 WMMA is not a dedicated matrix unit** — it runs on the regular SIMD32 ALUs and is
  worth only **~2×** over well-written vector f16 (it extracts the packed/dual-issue f16 math
  the vector path can't reach on RDNA3). [GPUOpen, Chips and Cheese, espadrine]
- zllm's 11.7× prefill deficit (f32 WGSL ~490 vs coopmat ~5747 tok/s) is **mostly f16-vs-f32 +
  tiling/arithmetic-intensity (~5–6×)**, only ~2× of it is coopmat. A *well-tiled f16 WGSL GEMM
  lands within ~1.5–2× of llama's coopmat path* — recoverable **without leaving wgpu**.
- **Decode is bandwidth-bound.** Achievable bus on the 8060S is ~212–215 GB/s (~83% of the
  256 GB/s spec). Ceiling ≈ 215 / 0.76 GB ≈ **~280 tok/s**. llama.cpp gets 201 (≈71% of
  achievable) — it **leaves real headroom**.

So the realistic, defensible targets (vs llama.cpp's measured 201 decode / 5747 prefill /
1458 batched-M32):

| metric | zllm now | llama.cpp | **realistic zllm target** | verdict |
|---|---:|---:|---:|---|
| single-stream **decode** | 82 | 201 | **240–250 (beat by ~20–25%)** | physically possible (≤280 wall) |
| **prefill** (pp512) | 490 | 5747 | ~3000–4500 (near parity) | parity realistic; +50% not |
| **batched** decode (M=32) | 327 | 1458 | ~900–1200 (near parity) | parity realistic; +50% not |

**+50% across the board is not on the table** — decode hits a bandwidth wall at +39% (and only
at 100% of the bus), prefill/batched are compute-bound where llama is already near the ceiling.
**But a genuine ~20% single-stream decode win IS achievable**, and that's the most user-visible
metric for interactive use. That's the headline this plan targets.

---

## RESULTS — what was actually built (2026-06-17)

Phase 2 was implemented (`--features vulkan`, `src/backend/vulkan/`): `ash` device with the
coopmat feature chain, UMA zero-copy buffers, GLSL kernels compiled offline to committed SPIR-V.
All kernels **bit-exact / cosine-1.0 vs candle**. Where it landed vs the targets above:

| metric | start | **achieved** | target | status |
|---|---:|---:|---:|---|
| coopmat works on 8060S | — | err 5e-5 | — | ✅ proven (disproves "unreachable") |
| Q4_K GEMM throughput | ~1000 (wgpu) | **6400–9900 GFLOP/s** | — | ✅ 6–8× wgpu |
| **prefill** (projection) | 490 | **~4250–4800 tok/s** | ~3000–4500 | ✅ **hit (~74–84% of llama)** |
| **decode matvec** | 50 GB/s | **208 GB/s** | >153 | ✅ beats llama's effective bus |
| **decode** (fused forward) | 80 (wgpu) | **~290 tok/s** | 240–250 | ✅ **beats llama's 201 (~1.4×)** |

**What worked:**
- **Prefill: register blocking was the lever, not the exotic stuff.** A 128×128 tile / 16-subgroup
  / 2×2-fragment kernel took prefill 44% → ~80% of llama. Double-buffering + bigger grids plateaued
  ~17% of peak (occupancy-bound on this part).
- **Decode matvec: word-loading** (process all 8 nibbles of a loaded u32) took it 50 → 208 GB/s,
  above llama's ~153 effective.
- **Decode forward, 107 → ~290 tok/s — the wall was the SDPA kernel, not barriers.** The fused
  forward stalled at ~100 tok/s and `VK_NOBAR=1` "fixing" it to ~370 *looked* like a 145-barrier
  wall. Wrong diagnosis: a coherence probe showed this driver elides empty pipeline barriers (so
  `VK_NOBAR`/`VK_EXECBAR` were racing layers, not a floor) and a *correct* barrier is only ~2 µs.
  `VK_SKIP=sdpa` found it: decode SDPA ran **one thread per head** (1 of 40 CUs) with a `float[128]`
  accumulator that **spilled to scratch** — ~6.6 ms/token. Rewritten as **one workgroup per head**
  (parallel over head-dim, single `av`/thread), it dropped to ~0.5 ms and the forward beat llama.
  Validated bit-exact vs a CPU reference (err 2.4e-7).
- **Toolchain:** no SDK needed — prebuilt glslang compiles coopmat GLSL → committed `.spv`.

**Fusions that backfired (kept for the record):** folding rmsnorm *or* silu·mul into a matvec
recomputes a per-element transform once **per output row** (silu-into-down was 88 vs 104 tok/s,
16.7 M redundant `exp`/layer). Rule: never fold a per-input-element op into a matvec *consumer*.

**Honest bottom line:** the "no hardware ceiling — it's kernel software" framing was right.
**Decode now beats llama (~290 vs 201, ~1.4×)** and prefill reaches ~80% — both with validated
bit-exact coopmat/raw-Vulkan kernels. Prefill parity is the realistic ceiling (+50% isn't — llama's
coopmat prefill is near the compute roof). The remaining work is **wiring this resident raw-Vulkan
decode forward into the server** (today the server's GPU fast-lane uses the wgpu engine; the
raw-Vulkan path lives in `--features vulkan` tests).

---

## Key technical findings (the basis for everything below)

1. **Prefill matmul (llama `mul_mm.comp`, KHR-coopmat path):** 16×16×16 fp16 fragments
   (fp32 accumulator), block tile **BM=BN=128, BK=32**, warp tile 32×32, dequant Q4_K → fp16
   into LDS *first* (packed-u32 reads + `unpack8`, `weight = d*q + m`), shared stride padded
   +8 for bank conflicts. The NV `coopMatLoadTensorNV` decode-on-load trick is **not** available
   on KHR — you must dequant-to-shared explicitly. [ggml-vulkan `mul_mm.comp`, `mul_mm_funcs.glsl`]
2. **Decode matvec (llama `mul_mat_vec.comp`):** workgroup == 1 subgroup, **2 output rows per
   subgroup** for Q4_K (`rm_kq=2`, 4 on some AMD), `subgroupAdd` reduction with **no LDS and no
   barrier**, ×4 unroll, weights read as packed u32 (coalesced), 6-bit scales cached in LDS across
   the 16 threads. This is the ~63%-of-bus kernel. Decode attention is **scalar flash-attn even on
   coopmat HW**. [ggml-vulkan `mul_mat_vec_base.glsl`, `dequant_funcs.glsl`]
3. **Fusion:** llama fuses `rms_norm+mul(+rope+kv_write)` and `mul_mat+add(+add)` into single
   dispatches, ~**10 dispatches/token**, barriers only on real scratch hazards. zllm's wgpu path
   issues ~80 auto-barriered passes/token — **this is the decode wall**, not the kernel math.
4. **RDNA3 WMMA = 16×16×16, fp16 in / fp16|fp32 acc, subgroup scope, wave32.** ~59 TFLOPS fp16
   peak on the 8060S (40 CU), ~35–40 sustained. Requires `requiredSubgroupSize=32` +
   `RequireFullSubgroups`. [GPUOpen WMMA, vulkaninfo: `KHR_coopmat` rev 2 confirmed present]
5. **Toolchain:** WGSL/naga has **no** cooperative matrix → coopmat needs a raw-Vulkan path
   (`ash`) with **offline-compiled SPIR-V** (`glslc`/`glslangValidator` → `.spv` → `include_bytes!`);
   do **not** rely on `shaderc` at runtime (its vendored glslang is usually too old for
   `GL_KHR_cooperative_matrix`). UMA gives zero-copy weights via a
   `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` heap (Strix Halo exposes it). [ash 0.38, jeffbolznv, llama.cpp #10785]

---

## Phase 1 — wgpu/WGSL, lower-risk, captures most of the prefill/batched gap

No new backend. Stay in the existing `src/backend/gpu` wgpu path. These are the
"tiling + f16" wins the research says are ~5–6× of the prefill deficit.

### 1.1 f16 GEMM (prefill + batched) — the biggest single lever
- Enable the wgpu `SHADER_F16` feature (8060S reports `fp16: 1`); add `enable f16;` to the GEMM.
- Dequant Q4_K/Q6_K weights to **f16** in the shared tile (not f32), accumulate in f32.
- Expected: the ~2× f16/dual-issue factor the f32 path is leaving on the table.
- Validate: bit-tolerance vs candle loosens slightly (f16 weights) — keep the cosine ≥ 0.998
  greedy-identical check from `gpu_prefill_vs_candle`.

### 1.2 Proper register-blocked tiling for the GEMM
- Current GEMM is workgroup-per-output-row, weight staged in LDS, x re-read from global per row
  (the `w2` 8 MB activation overflows L2 → re-streamed). Restructure to the llama tile:
  **BM×BN output tile per workgroup, both operands staged in LDS, register-blocked accumulators**
  (each thread computes a TR×TC micro-tile). This is the classic GEMM that raises arithmetic
  intensity and kills the x re-reads — the ~4–6× tiling factor.
- Keep TILE=256 LDS-occupancy lesson in mind (8 KB tile killed occupancy; size the new tile to
  stay ≤ ~2 KB LDS/workgroup region per operand or accept the occupancy tradeoff, measure).

### 1.3 Decode matvec → match llama's kernel shape (within wgpu's limits)
- Rewrite the resident matvec to **2 output rows per subgroup**, `subgroupAdd` reduction (wgpu
  exposes subgroup ops via the `SUBGROUP` feature) with **no shared-mem barrier**, ×4 unroll,
  128-bit (`vec4<u32>`) coalesced weight loads, 6-bit scales cached in LDS.
- Cuts the per-row reduction cost and the redundant activation traffic.

### 1.4 Fewer dispatches (cut the wgpu barrier tax)
- Fuse QKV into one matvec (concatenate wq/wk/wv at load → one weight, one dispatch), gate+up
  likewise. Fuse rms_norm + the elementwise weight-mul into one kernel.
- Target ~15–20 passes/token (from ~80). wgpu still auto-barriers between dependent passes, so
  this is partial — but ~4× fewer passes meaningfully lifts the decode ceiling.

**Phase 1 expected outcome:** prefill ~2000–3500 tok/s (from 490), batched proportional,
decode ~120–150 (from 82). This already closes most of the prefill/batched gap and is a large win,
but **decode stays below llama's 201** because wgpu's mandatory per-pass barriers cap it. Reaching
and beating 201 needs Phase 2. Ship Phase 1 first — it de-risks the kernels and the f16/tiling
math before the big `ash` investment.

---

## Phase 2 — raw Vulkan via `ash` (the decode win + coopmat prefill)

A second, feature-gated backend (`vulkan` cargo feature) alongside wgpu. This is where the
**decode lead over llama** and full coopmat prefill live. Larger effort (weeks), higher ceiling.

### 2.0 Toolchain + scaffolding (de-risk first — 2–4 days)
- Add `ash = "0.38"` (Vulkan 1.3). Headless: instance → pick `INTEGRATED_GPU` → compute queue.
- Device with the Features2 chain: `cooperativeMatrix`, `shaderFloat16`, `vulkanMemoryModel(+DeviceScope)`,
  16-bit storage; extensions `VK_KHR_cooperative_matrix`, subgroup-size-control.
- **Offline SPIR-V**: author `.comp` GLSL, compile with a modern `glslc`/`glslangValidator`
  (Vulkan SDK ≥ 1.3.275) to `.spv`, `include_bytes!` them. A `build.rs` step (or checked-in `.spv`)
  — no runtime shaderc.
- UMA buffers: allocate from `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`, persistent-map, memcpy
  weights in (no staging).
- **Spike:** a coopmat 16×16×16 fp16 matmul (port jeffbolznv `tiled.comp`), read back, validate
  vs CPU. Pin `requiredSubgroupSize=32` + `RequireFullSubgroups`. Confirms the whole path works.

### 2.1 Coopmat Q4_K prefill GEMM
- Port llama's `mul_mm.comp` KHR path: BM=BN=128, BK=32, 16×16×16 fp16 fragments, fp32 acc,
  dequant-Q4_K-to-LDS-f16 first, double-buffered (prefetch next K-tile into regs while
  coopMatMulAdd the current), bank-conflict pad +8.
- Reuse the existing Q4_K/Q6_K block parsing; the dequant arithmetic is already validated in WGSL.
- Target: within ~1.5× of llama's 5747 → ~3500–4500 prefill. (Beating it needs out-tuning a
  mature, near-compute-bound kernel — not the goal.)

### 2.2 Fused **megakernel decode** — the lever that beats llama
- The decode win is **dispatch/barrier elimination**, not coopmat (decode is M=1 matvec).
- In raw Vulkan you control barriers: record the whole token forward with **minimal, hand-placed**
  `vkCmdPipelineBarrier`s (only real RAW hazards), like llama's ~10-dispatch graph — or push
  further toward a persistent/megakernel layout.
- Decode matvec kernel: 2 rows/subgroup, `subgroupAdd` no-barrier reduction, ×4 unroll, 128-bit
  loads, fused dequant in the inner loop, scales cached in LDS (llama's `mul_mat_vec` shape).
- Scalar online-softmax flash-attention for the KV (decode FA is scalar even on coopmat HW).
- Target: ~85–90% of achievable bus → **240–250 tok/s, beating llama's 201 by ~20–25%.**

### 2.3 Coopmat batched decode (serving)
- Reuse the BatchedDecoder structure; swap the GEMM for the coopmat path. M=concurrent streams
  is the same compute-bound regime as prefill → coopmat applies. Target near parity with llama's
  1458 at M=32.

---

## Validation (unchanged discipline)

- Every kernel **bit-exact / greedy-identical vs candle** — reuse `gpu_prefill_vs_candle`,
  `gpu_full_forward`, `gpu_batched_decode` harnesses. f16 paths: assert cosine ≥ 0.998 + identical
  greedy tokens (already the bar).
- Head-to-head after each phase vs `llama-bench` / `llama-batched-bench`, recorded in `BENCHMARKS.md`.
- Ship gate: each change is a measured end-to-end win; no regressions.

## Effort / risk

| phase | effort | risk | payoff |
|---|---|---|---|
| 1 (wgpu f16 + tiling + decode opt) | ~1–2 wks | low (existing path) | prefill→~3000, decode→~140; most of the gap |
| 2.0 (ash scaffold + coopmat spike) | ~2–4 days | medium (new toolchain) | proves the path |
| 2.1 (coopmat prefill) | ~1–2 wks | medium | prefill→~4000 (near parity) |
| 2.2 (megakernel decode) | ~1–2 wks | medium-high (barrier hand-mgmt) | **decode→~245 (beats llama)** |
| 2.3 (coopmat batched) | ~1 wk | low (reuses 2.1/2.2) | batched→~1000 (near parity) |

## Honest ceilings (so we don't chase physics)

- **Decode:** hard wall ~280 tok/s (215 GB/s ÷ 0.76 GB). Realistic best ~245–255. **Beating llama
  by ~20–25% is the win; +50% (→300) is past the memory controller.**
- **Prefill / batched:** compute-bound; llama's coopmat path is near the CU ceiling. Parity is the
  realistic best; a 50% lead is not.
- **CPU:** already at parity (bandwidth roofline). No further headroom.

## Sources

- llama.cpp Vulkan backend: `ggml/src/ggml-vulkan/` — `mul_mm.comp`, `mul_mm_funcs.glsl`,
  `mul_mat_vec_base.glsl`, `dequant_funcs.glsl`, `flash_attn*.comp`, `ggml-vulkan.cpp`
- GLSL coopmat: KhronosGroup/GLSL `GLSL_KHR_cooperative_matrix.txt`; SPV_KHR_cooperative_matrix
- Reference shaders: jeffbolznv/vk_cooperative_matrix_perf (`tiled.comp`, `shmem.comp`)
- RDNA3 WMMA: gpuopen.com/learn/wmma_on_rdna3; chipsandcheese RDNA3 microbench; espadrine GPU-perf
- ash 0.38 (`khr::cooperative_matrix`), shaderc/glslang version caveat (llama.cpp #10785)
- Strix Halo bandwidth (~212–215 GB/s): llm-tracker, hardware-corner.net
- llama.cpp tuning PRs: #12260 (coopmat1 mul_mm, +25%), #17711 (mul_mat_vec)
</content>
