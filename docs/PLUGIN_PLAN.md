# zllm Plugin System — Design & Plan

In-process dynamic plugins with **zero-copy ("same memory") access** to engine
state, **observe + control**, **portable across Windows + Linux**, with **eBPF as a
Linux-only optional add-on**. (Decisions confirmed with the user.)

## TL;DR

zllm **already has an in-process hook system** — `Hook` trait (`on_layer →
HookAction`), `HookRegistry`, and three hooks (`steering`, `early_exit`,
`memory_inject`) in `src/engine/hooks/`. Two gaps make it not-yet-a-plugin-system:

1. Hooks must be **compiled into the binary** (`Box<dyn Hook>` registered in Rust).
2. Hook **write-back is discarded** — "read-only in v0.8: hook mutations to
   `&mut Tensor` are computed but discarded" (`runner_observer.rs`).

This plan closes both: a **stable C-ABI** so a `Hook` can live in a separately
compiled `.dll`/`.so`, dynamic loading via `libloading`, the **control/write-back
path** wired, and hook points extended beyond per-layer to **logits** and
**scheduling**. eBPF is a separate, optional, Linux-only transport for the
*observe* subset (kernel-side visibility), not the core.

## What already exists (extend, don't rebuild)

| piece | file | role |
|---|---|---|
| `Hook` trait (`on_layer`) + `HookAction` {Continue, EarlyExit, SkipRemaining} | `engine/hooks/traits.rs` | per-layer hook contract |
| `HookRegistry` (register/fire/clear) | `engine/hooks/registry.rs` | dispatch |
| `steering`, `early_exit`, `memory_inject` | `engine/hooks/*.rs` | concrete hooks (compiled-in) |
| `RunnerObserver` | `engine/runner_observer.rs` | bridges backend callback → registry |
| `forward_logits_with_observer(t, \|layer, &Tensor\|)` | `candle/backend.rs:173` | the per-layer callback (candle path) |
| `LogitFSM::apply_mask(&mut Tensor)` | `engine/logit_fsm.rs:53` | logit-level control precedent |
| `BatchScheduler` / `ContinuousBatcher` | `control_plane/scheduler.rs`, `backend/gpu` | scheduling/preemption surface |
| REST `/v1/inspect/enabled` | `server/rest.rs` | runtime enable/disable precedent |

## Hard constraints (these shape every choice)

1. **Rust has no stable ABI.** A `.dll`/`.so` boundary must be **C-ABI**
   (`extern "C"` + `#[repr(C)]`). Calling `Box<dyn Hook>` across a dynamic boundary
   directly is UB. → define a versioned C-ABI; offer an ergonomic Rust SDK on top.
2. **The LPDDR5X bus is the perf budget.** Per-layer hooks fire 16×/token on the hot
   path. The ABI must pass **pointers, not copies**, and let a plugin opt into *which
   layers* and *which slices* (small windows) — never force a full-tensor copy. Today
   `RunnerObserver` already mean-pools to one vector/layer to avoid per-token cost;
   keep that "small window" discipline.
3. **Mid-layer hooks are cheap on CPU/candle, costly on the Vulkan fast path.** The
   `ZLLM_VK=1` forward runs entirely in recorded GPU command buffers; mid-layer
   hidden states live on the GPU, so a layer hook there needs a **readback** (bus
   cost). → layer hooks target the candle path first; on Vulkan they are an explicit,
   opt-in readback (off by default so `ZLLM_VK=1` stays at its 1.12× decode). Logit
   and scheduling hooks are backend-agnostic and cheap everywhere.
4. **In-process = a bad plugin can crash the engine.** Accepted (user chose
   in-process). Mitigate: versioned ABI + layout checks, `catch_unwind` at every FFI
   boundary (a plugin panic must not unwind across C-ABI → abort-safe), and a
   capability manifest (observe vs control) for a permission model.

## Architecture

Three crates + two access modes + one optional transport.

```
 zllm (host)                         plugin author writes:
 ├─ PluginManager (libloading)       ┌─ my_plugin (cdylib)
 │   loads .dll/.so, negotiates ABI  │    use zllm-plugin (safe SDK)
 │   registers hook interest         │    impl LayerHook / LogitHook / SchedHook
 ├─ HookRegistry (existing, extended)│    #[zllm_plugin] export macro
 │   sync inline dispatch            └─ compiles to my_plugin.dll / .so
 └─ Snapshot publisher (lock-free)        ▲ links zllm-plugin-abi (#[repr(C)])
        │                                 │
        ├─ async observers (mmap or in-proc reader)
        └─ [Linux, feature=ebpf] BPF_F_MMAPABLE map ← kernel eBPF observer
```

- **`zllm-plugin-abi`** (new crate, no_std-friendly): the frozen `#[repr(C)]` structs,
  vtables, `ABI_VERSION`, hook-context layouts. The only thing both sides depend on.
- **`zllm-plugin`** (new crate): ergonomic Rust SDK — safe `LayerHook`/`LogitHook`/
  `SchedHook` traits, a `#[zllm_plugin]` macro that emits the `extern "C"` exports +
  the `catch_unwind` wrappers. Plugin authors never touch raw FFI.
- **host `PluginManager`** (in zllm, `engine/plugins/`): discovers `plugins/*.{dll,so}`
  + a `manifest.toml` each (name, abi_version, hooks, capabilities), loads, version-
  checks, and adapts each into the existing `HookRegistry`.

**Two access modes:**
- **Synchronous inline hooks** (control path): the plugin fn runs *inside* the engine
  thread at the hook point with raw pointers to live state — true zero-copy, can edit
  in place (steering) and return a `HookAction` (early-exit/continue). No concurrency
  hazard: the engine isn't touching that buffer during the call.
- **Async published snapshot** (lock-free observe): the engine publishes a small
  `#[repr(C)] EngineSnapshot` (active slots, phase, per-slot token counts, metrics)
  behind an `AtomicU64` version (seqlock) into a shared region. Async observers — and
  the eBPF map — read it without blocking the forward. Single-writer (engine) only.

## The C-ABI (sketch)

```rust
// zllm-plugin-abi — frozen, versioned
pub const ABI_VERSION: u32 = 1;

#[repr(C)] pub enum HookAction { Continue = 0, EarlyExit = 1, SkipRemaining = 2 }

#[repr(C)] pub struct LayerHookCtx {        // hot path — pointers, no copies
    pub layer_idx: u32, pub n_tokens: u32, pub hidden_dim: u32,
    pub hidden: *mut f32,                   // [n_tokens * hidden_dim], EDIT IN PLACE (steering)
    pub positions: *const u32, pub request_id: u64,
}
#[repr(C)] pub struct LogitHookCtx { pub vocab: u32, pub logits: *mut f32, pub step: u32, pub request_id: u64 }
#[repr(C)] pub struct SlotMeta { pub slot: u32, pub seq_len: u32, pub priority: i32, pub state: u8 }
#[repr(C)] pub struct SchedHookCtx { pub slots: *mut SlotMeta, pub n_slots: u32 }

// Control actions the plugin invokes back on the host (structural changes go through
// well-defined callbacks; raw &mut is only for the hidden-state/logits buffers above).
#[repr(C)] pub struct HostApi {
    pub set_priority: extern "C" fn(host: *mut c_void, slot: u32, prio: i32),
    pub request_preempt: extern "C" fn(host: *mut c_void, slot: u32),
    pub emit_metric: extern "C" fn(host: *mut c_void, key: *const u8, len: u32, val: f64),
    pub host: *mut c_void,
}

// Plugin exports (the only required symbols; the SDK macro generates them):
//   #[no_mangle] extern "C" fn zllm_plugin_abi_version() -> u32
//   #[no_mangle] extern "C" fn zllm_plugin_create(host: *const HostApi) -> *mut PluginVTable
//   #[no_mangle] extern "C" fn zllm_plugin_destroy(p: *mut PluginVTable)
#[repr(C)] pub struct PluginVTable {            // fn ptrs; null = "not interested"
    pub layer_mask: u64,                        // which layers fire on_layer (bus discipline)
    pub on_layer: Option<extern "C" fn(*mut c_void, *mut LayerHookCtx) -> HookAction>,
    pub on_logits: Option<extern "C" fn(*mut c_void, *mut LogitHookCtx) -> HookAction>,
    pub on_schedule: Option<extern "C" fn(*mut c_void, *mut SchedHookCtx)>,
    pub on_request_start: Option<extern "C" fn(*mut c_void, u64)>,
    pub on_request_end: Option<extern "C" fn(*mut c_void, u64)>,
    pub state: *mut c_void,
}
```

Negotiation: host calls `zllm_plugin_abi_version()`; if `!= ABI_VERSION`, refuse to
load (log + skip). Every host→plugin call is wrapped in `catch_unwind` on the plugin
side (SDK-generated) so a panic becomes a logged abort, never UB across FFI.

## Hook points → integration in zllm

| hook | where wired | backend | notes |
|---|---|---|---|
| `on_layer` (read+**write**) | extend `RunnerObserver` / `forward_logits_with_observer` | candle (cheap); Vulkan via opt-in readback | **wire the write-back that's discarded today** |
| `on_logits` | next to `LogitFSM::apply_mask` in the sample path | all (cheap) | FSM-style masking, early-exit, biasing |
| `on_schedule` | `BatchScheduler::schedule_step` / `ContinuousBatcher` | gpu/CB | priority + preempt via `HostApi` |
| lifecycle | `PluginManager` + request entry/exit in `server/rest.rs` | all | init/teardown, per-request state |

## Phases

- **Phase 0 — ABI + manager + echo plugin (no behavior change).** Create the two
  crates + `PluginManager`; load a no-op plugin from `plugins/`; ABI version check;
  `catch_unwind` boundary. **Gate: tok/s identical with no plugin loaded; bit-exact
  output with the echo plugin attached.**
- **Phase 1 — `on_layer` observe + control (candle).** Adapt dynamic plugins into
  `HookRegistry`; **wire the discarded write-back** so steering edits the live hidden
  state in place. Port the existing `steering`/`early_exit` hooks to the SDK as the
  reference plugins. **Gate: a steering plugin demonstrably shifts output; zero-copy
  (no per-layer full-tensor copy); CPU tok/s unchanged when plugin absent.**
- **Phase 2 — `on_logits` + `on_schedule` (control).** Logit masking/early-exit on
  the sample path (all backends); priority/preempt callbacks into the CB scheduler.
  **Gate: a grammar plugin constrains output; a priority plugin changes batch order.**
- **Phase 3 — async snapshot + eBPF (Linux, `--features ebpf`, optional).** Publish
  `EngineSnapshot` via seqlock; a sample async observer reads it; on Linux, also mirror
  it into a `BPF_F_MMAPABLE` Aya array map + a sample kernel-side eBPF reader. **Gate:
  snapshot read lock-free off-thread; eBPF reader sees live slot/phase data.**

## Risks & mitigations

- **ABI drift** → frozen `zllm-plugin-abi`, version negotiation, CI layout test.
- **Plugin crash/panic** → `catch_unwind` per call; capability manifest; (future)
  out-of-process or WASM sandbox for *untrusted* plugins (cold path only — WASM's
  separate linear memory breaks zero-copy, so not for hot-path control).
- **Bus blowup from eager plugins** → `layer_mask` + slice windows; a perf budget
  check in Phase 1's gate.
- **Vulkan readback cost** → layer hooks opt-in on `ZLLM_VK=1`, off by default.
- **eBPF only on Linux** → strictly Phase 3, feature-gated; the portable core
  (Phases 0–2) delivers the full plugin value without it.

## Validation

Zero-overhead-when-absent (tok/s vs baseline) · bit-exact with a no-op plugin ·
a real steering plugin shifts output · a logit plugin constrains grammar · a
scheduler plugin reorders/preempts · (Linux) eBPF reader sees the live snapshot.
