# CEIP — Cross-Engine Instrumentation Protocol (design & plan)

*(name is a placeholder)*

One instrumentation contract that runs the same plugin against multiple inference
engines — **zllm first as the reference adapter**, then vLLM and llama.cpp —
**capability-negotiated** so the deep hooks that don't portably exist are handled
honestly instead of silently no-op'ing.

## The two laws this design obeys

1. **Cross-engine ⟂ zero-copy shared memory.** vLLM (Python/CUDA), llama.cpp (C++),
   and zllm (Rust/Vulkan) share no runtime or address space, so cross-engine hooks
   go through a **protocol boundary**, never shared memory. That boundary is only
   affordable where data is small → the **portable core is the logit/sampling/
   telemetry layer**; mid-layer hidden states are **capability-gated** and, in
   practice, **native-transport-only** (the per-layer tensors are too big to copy
   over a process/WASM boundary at the bus budget).
2. **Depth is least portable; depth is zllm's moat.** The shallow hooks are portable
   but undifferentiated (prior art: Outlines/Guidance for constrained decoding,
   OpenTelemetry GenAI for tracing). zllm's value is the deep tier, which only zllm
   does fully (vLLM partially, llama.cpp barely). The protocol must **expose, not
   hide, this asymmetry** via capabilities.

## What each engine can actually expose

| tier | zllm | vLLM | llama.cpp |
|---|---|---|---|
| logits / sampling | native `on_logits` | logits processors (Py) | logit-bias + GBNF grammar |
| token / request telemetry | `metrics.rs` | stats hooks | timings cb |
| hidden states (observe) | native, zero-copy | PyTorch module fwd-hooks (Py, perf cost) | `cb_eval` tensor cb (observe-only, compiled-in) |
| hidden-state / steering (control) | native (wire the write-back) | hard (patch model) | no |
| KV / scheduling control | native (CB scheduler) | limited | no |

## Architecture

```
        ┌──────────────── CEIP spec (capabilities + hook contract) ────────────────┐
        │  data model: control msgs = protobuf/IDL ;  tensors = DLPack (C ABI)      │
        └───────────────────────────────────────────────────────────────────────────┘
 plugin (written once against the spec)
        │   declares required capabilities ──► loader rejects/degrades if unmet
        ├─ transport: NATIVE  (in-proc C-ABI/DLPack)   ── max perf, hot path, deep hooks
        ├─ transport: SIDECAR (gRPC / shm ring)        ── one binary any language, ~50–100µs/call
        └─ transport: WASM    (wasmtime + WIT, later)  ── one binary, in-proc sandbox, ~µs/call
 engine adapters (implement the spec for their engine)
        ├─ zllm   : NATIVE (= PLUGIN_PLAN.md ABI) + sidecar host + (later) WASM host  ◄ reference
        ├─ vLLM   : Python adapter — logits-processor + module fwd-hooks + caps
        └─ llama.cpp: C++ adapter — logit-bias/grammar + cb_eval(observe) + caps
```

**`PLUGIN_PLAN.md` is not superseded — it becomes zllm's NATIVE adapter.** That
in-process C-ABI is the zero-copy, full-depth transport; CEIP wraps it with the
capability descriptor + the portable (sidecar/WASM) transports so the same plugin
also runs elsewhere.

## Capability negotiation (the honesty mechanism)

At attach time the engine adapter returns a descriptor; the plugin declares needs;
the loader matches or refuses with a clear reason (never a silent no-op).

```protobuf
message Capabilities {
  uint32 protocol_version = 1;
  Engine engine = 2;                       // name, version
  Hook logits     = 3;                     // {supported, can_modify}
  Hook sampling   = 4;                     // {supported}  observe token+logprob
  Hook lifecycle  = 5;
  HiddenStates hidden = 6;                 // {mode: NONE|OBSERVE|CONTROL, layers, dtype, cost_hint}
  Hook attention  = 7;                     // optional
  Hook kv_cache   = 8;                     // observe slot/paged meta
  Hook scheduling = 9;                     // control priority/preempt
  repeated Transport transports = 10;      // NATIVE|SIDECAR|WASM
  Exec exec = 11;                          // batched, max_batch, streaming
}
// Example: plugin manifest `requires { hidden: OBSERVE }` → attaching to llama.cpp
// (hidden=NONE) is refused with "engine lacks hidden-state observation".
```

## Hook contract (engine-agnostic)

- `on_request_start(ctx)` / `on_request_end(ctx, stats)` — lifecycle. **portable**
- `on_logits(ctx, logits: TensorView) -> LogitAction` — per step. **portable**
  `LogitAction ∈ {None, Bias(sparse), Replace(tensor), AddVector, ForceToken, Stop}`
- `on_sample(ctx, token, logprob)` — observe chosen token. **portable**
- `on_hidden(ctx, layer, hidden: TensorView) -> HiddenAction` — **capability-gated**
  `HiddenAction ∈ {None, Replace(tensor), AddVector}`; observe-only engines ignore the return.
- `on_attention` / `on_kv` / `on_schedule` — **capability-gated**.
- `emit_metric(key, val)` (plugin→host) + `on_telemetry(event)` (host→plugin). **portable**

**Tensors cross as DLPack** (`DLManagedTensor`, the standard cross-framework C ABI):
zero-copy in-process (native transport), a single copy over sidecar/WASM. vLLM
(PyTorch) produces/consumes DLPack natively; Rust/C++ can wrap their buffers in it.

## Transport — the perf reality

| transport | per-call cost | tensors | one-binary plugin? | use for |
|---|---|---|---|---|
| **native** (per-engine SDK) | ~ns–µs | zero-copy (DLPack ptr) | no (Rust/Py/C++ per engine) | hot-path, **deep hooks** |
| **sidecar** (gRPC/shm) | ~50–100µs | copy | **yes**, any language | telemetry, cold-path, prove portability |
| **WASM** (wasmtime/WIT) | ~µs | copy (linear mem) | **yes** | logit-tier, sandboxed, later |

Per-token logit hooks are fine native; over sidecar they add ~1–2% to a ~5 ms decode
step (tolerable for monitoring, watch it at scale). **Hidden states are native-only in
practice** — copying per-layer residuals over any boundary blows the bus budget.

## Native adapters — the shared `zllm-probe` core

The native transport (the deep, zero-copy tier) is realized by ONE engine-agnostic
Rust crate, **`zllm-probe`**, embedded *in-process* into each engine. Write the probe
logic once against the hook contract + `TensorView`; each engine's thin adapter hands
it tensors via **DLPack** with no copy and no IPC.

```
 zllm-probe (Rust core)            ← hook contract + probe logic, written ONCE
   operates on TensorView (DLPack) + the capability/action types
        │
        ├─ zllm     : linked natively (it's Rust) — the PLUGIN_PLAN.md ABI IS this
        ├─ vLLM     : PyO3/maturin → a Python extension vLLM imports; a module
        │             fwd-hook calls to_dlpack(hidden) → core gets the CUDA device
        │             pointer (zero-copy), in vLLM's own process
        └─ llama.cpp: C-ABI cdylib; cb_eval hands tensor pointers (CPU/backend)
```

- **"zllm imported into vLLM's memory" = exactly this:** the core runs *inside vLLM's
  process*, shares its address space, and reads its tensors through DLPack handles —
  no copy, no socket. It is **not** zllm-the-engine (the Vulkan decode kernels are
  irrelevant on CUDA); it's only the instrumentation core.
- **GPU reality (decisive):** vLLM's hidden states are CUDA tensors in VRAM. The
  DLPack handle is zero-copy, but to *compute* on it without a copy the probe must run
  **GPU-side** — a CUDA kernel the core launches on vLLM's device/stream (cudarc), or
  a callback into torch ops. Pulling the tensor to CPU for Rust-side analysis
  re-introduces the per-layer GPU→CPU copy you're trying to avoid. So: zero-copy
  *handle* is free; zero-copy *analysis* means the probe math is GPU-side, not CPU-Rust.
- **Coupling cost:** the vLLM shim rides PyTorch module layout + CUDA context + version
  (pin-tested, fragile); an in-proc crash drops the server; vLLM's per-shard **worker
  processes** each load the extension.
- **Don't over-build:** a trivial probe (e.g. project hidden onto a learned direction)
  is a one-line torch op in a Python hook — no Rust core needed. The shared core earns
  its keep only when the logic is non-trivial / must be byte-identical across engines,
  or you also do CPU-side / shallow-tier work behind the same code.

This is the **native** row of the transport table made concrete; sidecar/WASM still
serve the shallow + latency-tolerant + cross-machine cases, with the *same* core behind them.

## Phases

- **Phase 0 — Spec + zllm native.** Write the CEIP spec (caps, hook contract, DLPack
  views, action enums). Implement zllm's **native** adapter = the `PLUGIN_PLAN.md`
  ABI, plus the capability descriptor (zllm reports the full set) and a DLPack view
  over its hidden-state buffers. **Wire the write-back that's discarded today.**
  Reference plugin: a logit-entropy **confidence/hallucination monitor** (portable
  tier). *Gate: plugin runs in-proc on zllm; capability negotiation + a refusal case work.*
- **Phase 1 — Portable transport on zllm.** Add the **sidecar (gRPC)** host so the
  SAME reference plugin runs out-of-process against zllm. *Gate: one plugin, two
  transports, identical output; per-token overhead measured.*
- **Phase 2 — Second engine (vLLM), native deep adapter.** PyO3/maturin shim embeds
  `zllm-probe` *in vLLM's process*; a module fwd-hook hands `to_dlpack(hidden)` device
  handles to the core (zero-copy `on_hidden`); caps + logits-processor cover the
  shallow tier. Probe math runs GPU-side (cudarc kernel or torch callback). *Gate:
  the SAME probe core on zllm AND vLLM; hidden read zero-copy in vLLM's process (no
  per-layer GPU→CPU copy on the hot path); richer signal where capable, graceful where not.*
- **Phase 3 — Third engine (llama.cpp) + capability floor.** C++ adapter: caps +
  logit-bias/grammar + `cb_eval` observe. *Gate: portable-tier plugin runs; a
  control-requiring plugin is cleanly **rejected** with a capability error.*
- **Later — WASM transport** for true one-binary in-process plugins at the logit tier.

## Reference plugin (proves the whole point)

A **confidence/hallucination monitor**: logit-entropy + output-distribution signal on
*every* engine (portable), and where `hidden=OBSERVE` is available (zllm, vLLM) it adds
an **activation-probe** score — same plugin, richer on capable engines, degraded (not
broken) on llama.cpp. This single plugin exercises caps, degradation, and both tiers.

## Risks & honest scoping

- **Adapter maintenance:** vLLM/llama.cpp adapters ride their internals (module layout,
  `cb_eval`) → version-fragile, ongoing cost. Pin tested versions.
- **Deep-hook transport:** hidden states are native-only; "cross-engine deep hooks"
  realistically means **zllm + vLLM, native/in-proc each** (the shared `zllm-probe`
  core embedded via native-link / PyO3 / C-ABI), not one binary everywhere — and on
  vLLM the probe math runs GPU-side (CUDA/torch), not CPU-Rust.
- **Latency at scale:** sidecar per-token is fine for monitoring, not for high-QPS
  logit rewriting — offer native there.
- **Scope:** spec + 3 adapters + transports is multi-month. **Prove value on zllm +
  one second engine (vLLM) before committing to llama.cpp.** zllm-native alone (Phase 0)
  already delivers the white-box plugin system; each later phase is independently useful.
- **Don't reinvent the shallow tier:** integrate/borrow from Outlines/Guidance
  (constrained decoding) and OTel GenAI (tracing) rather than re-spec them.

## Validation

Capability negotiation + a refusal case · one plugin runs native AND sidecar on zllm
with identical output · the SAME `zllm-probe` core runs on zllm AND vLLM, with vLLM
hidden-state reads zero-copy in-process (no per-layer GPU→CPU copy) · llama.cpp runs
the portable tier and cleanly rejects deep-hook plugins · per-transport per-token
overhead measured · DLPack round-trip bit-exact · zllm decode tok/s unchanged when no plugin attached.
