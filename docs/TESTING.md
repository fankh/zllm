# Manual testing playbook

Copy-paste commands to verify zllm's features and performance by hand.
Windows PowerShell: use `curl.exe` (not the `curl` alias). Linux/macOS: `curl`.
Server examples assume `configs/local-test.toml` (port **8090**; `default.toml`
uses 8080).

## 0. Build & start

```powershell
cargo build --release --features vulkan          # CPU + raw-Vulkan iGPU fast lane
$env:ZLLM_VK = "1"                               # enable the fast lane at startup
./target/release/zllm.exe serve --config configs/local-test.toml
```

Sanity:

```powershell
curl.exe -s http://127.0.0.1:8090/health          # {"engine":"zllm","status":"ok",...}
curl.exe -s http://127.0.0.1:8090/v1/info         # feature list
```

Or just open **http://127.0.0.1:8090/** — the built-in chat UI.

### Two gotchas that WILL skew results if skipped

1. **Inspection defaults ON and blocks the fast lane.** For perf tests:
   ```powershell
   curl.exe -s -X POST http://127.0.0.1:8090/v1/inspect/enabled -H "Content-Type: application/json" -d '{\"enabled\":false}'
   ```
2. **A leftover goal silently prefixes every prompt** (`GOAL: ...`), changing
   all outputs and risk scores. Check before measuring:
   ```powershell
   curl.exe -s http://127.0.0.1:8090/v1/goal/state    # want "prompt_prefix":""
   ```
   To clear: stop the server and delete `configs/goals.json` (runtime state).

## 1. Basic generation

```powershell
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0}'
```
Expect: `" Paris. ..."`. With `ZLLM_VK=1` + inspection off + prompt ≤ 128
tokens, the server log prints `Vulkan fast-lane: prefill N tok in X ms, decoded
N tok at ~180-210 tok/s`. Prompts **> 128 tokens** intentionally fall back to
the CPU path (f16 batched-prefill exactness guard).

## 2. Hallucination / uncertainty detection

```powershell
# should read low-ish risk
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"The first letter of the alphabet is\",\"max_tokens\":12,\"temperature\":0,\"detect_hallucination\":true}'
# forced fabrication - should read higher risk
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"The seventeenth word of the Zorblaxian national anthem is\",\"max_tokens\":12,\"temperature\":0,\"detect_hallucination\":true}'
```
Expect a `"hallucination"` object (`risk_score`, `mean_entropy`,
`risky_fraction`, `flagged`, `peak_token_index`). Verified properties:
- **Reproducible**: repeating the same request returns bit-identical scores
  (detection forces a cold prefill).
- Works on chat too; `stream:true` + detection → **400** (by design).
- **Calibration caveat (1B)**: the response-level flag is weak — fabrications
  ~0.53, trivia ~0.42, and post-answer free-form can false-positive (~0.54).
  Treat `risk_score` as a relative signal, `peak_token_index` as the pointer;
  a calibrated verdict is the planned hidden-state-probe v2.

## 3. Grammar-constrained decoding

```powershell
# output guaranteed to be exactly yes or no
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"Is water wet? Discuss at length.\",\"max_tokens\":20,\"temperature\":0,\"grammar\":\"regex:(yes|no)\"}'
# structured shape: (123) 456-7890
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"Call me at\",\"max_tokens\":24,\"temperature\":0,\"grammar\":\"regex:\\\\([0-9]{3}\\\\) [0-9]{3}-[0-9]{4}\"}'
# token banning (id 12366 = " Paris" for Llama-3): answer avoids it
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0,\"grammar\":\"ban:12366\"}'
# loud failures
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"hi\",\"max_tokens\":4,\"grammar\":\"regex:a(\"}'   # 400 bad pattern
curl.exe -s -X POST http://127.0.0.1:8090/v1/completions -H "Content-Type: application/json" -d '{\"prompt\":\"hi\",\"max_tokens\":4,\"grammar\":\"json:{}\"}'    # 400 not implemented
```
First grammar request logs `built grammar byte table (128256 tokens) in ~100 ms`
(one-time per model). Grammar requests always run on the CPU path.

## 4. Early exit

```powershell
curl.exe -s -X POST http://127.0.0.1:8090/v1/early_exit/enabled -H "Content-Type: application/json" -d '{\"enabled\":true}'
curl.exe -s -X POST http://127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to five.\"}],\"max_tokens\":30,\"temperature\":0}'
curl.exe -s http://127.0.0.1:8090/metrics | findstr early_exit
```
Expect `zllm_early_exit_fires_total` > 0; `layer_sum/fires` ≈ average exit layer
(≈12 of 16 on the 1B). Threshold/min-layer: `/v1/early_exit/config`.
Requires greedy (temp 0), no grammar/detection. Turn back off after.

## 5. Goals / memory / inspection

```powershell
curl.exe -s -X POST http://127.0.0.1:8090/v1/goal/set -H "Content-Type: application/json" -d '{\"text\":\"answer like a pirate\"}'
curl.exe -s http://127.0.0.1:8090/v1/goal/state     # prompt_prefix now "GOAL: ..."
# generation is now conditioned on the goal - REMEMBER TO CLEAR (see gotcha 2)
```
Inspection traces (per-layer confidence, top tokens): leave inspection ON,
run a chat request, then `curl.exe -s http://127.0.0.1:8090/v1/inspect`.
Memory-inject write-back stays OFF unless `engine.memory_inject_alpha > 0`
in the config — enabling it with an uncurated store degrades output (measured).

## 6. Toggles / model management

```powershell
curl.exe -s http://127.0.0.1:8090/v1/models                     # list GGUFs in model.dir
# POST /v1/models/select {"id": "<filename-stem>"} to hot-swap
# /v1/pld/enabled, /v1/spec_decode/enabled - same {"enabled":bool} shape
```

## 7. Performance (engine-level cargo harnesses; no server)

```powershell
# decode: bit-exactness vs candle + sustained tok/s (GPU-hot)
$env:ZLLM_NTIME="128"; $env:ZLLM_REPS="10"
cargo test --release --features vulkan --lib vk_model_vs_candle -- --ignored --nocapture
# expect: "agree on first 24/24 tokens" + tok/s (~175-210 depending on chassis temp)

# decode at depth / head-major long-context win (+5-12%)
$env:ZLLM_CTX="2048"
cargo test --release --features vulkan --lib vk_model_vs_candle -- --ignored --nocapture
$env:ZLLM_HEADMAJOR_KV="1"   # rerun; compare tok/s. Unset both after.

# prefill correctness probe
cargo test --release --features vulkan --lib vk_prefill_vs_candle -- --ignored --nocapture
```
**Thermal caveat**: absolute tok/s varies ±15% with chassis temperature
(10-rep runs can be *slower* than 3-rep = throttling). Compare A/B numbers
from the same session only. Reference numbers: ../BENCHMARKS.md.

Continuous batching (separate build): `cargo build --release --features gpu`,
run with `ZLLM_CB=1`, POST to `/v1/cb/completions`; aggregate throughput
scales ~5.6× at 8 concurrent (see ../BENCHMARKS.md §3).

## 8. Unit/integration suite

```powershell
cargo test --release --lib                        # ~88 tests, all green expected
cargo test --release --features vulkan --lib      # + vulkan units
```
