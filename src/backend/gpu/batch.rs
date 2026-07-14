//! GPU continuous-batching serving stack (paged-KV in-flight batching:
//! admit/prefill/decode scheduling, preemption + host-KV swap, and the
//! threaded GpuBatchServer), extracted from `mod.rs` (V1_PLAN split).
//! `use super::*` pulls in GpuModel/BatchedDecoder/BlockState and the
//! sampling helpers; the parent re-exports the public types.

use super::*;

/// An active sequence. `tokens` is the full history (prompt ++ every produced
/// token, including the not-yet-fed last one), so derived state is:
/// next = tokens.last(); KV write pos = tokens.len()-1; gen index = tokens.len()-prompt_len.
/// Keeping the full history lets a preempted sequence be recomputed on resume.
struct CbSeq { id: u64, slot: u32, tokens: Vec<u32>, prompt_len: usize, max_tokens: usize, eos: u32, params: SamplingParams, base_seed: u32 }
impl CbSeq {
    fn next(&self) -> u32 { *self.tokens.last().unwrap() }
    fn pos(&self) -> u32 { (self.tokens.len() - 1) as u32 }
    fn gen_index(&self) -> usize { self.tokens.len() - self.prompt_len }
}

/// A sequence's KV cache copied out to host RAM (swap-to-host preemption),
/// `nb = ceil(n_pos/block_size)` logical blocks per layer, K then V packed in
/// logical-block order `[layer*nb + lb][block_size*kv_dim]`.
pub(super) struct HostKv { pub(super) n_pos: usize, pub(super) nb: usize, pub(super) k: Vec<u32>, pub(super) v: Vec<u32> } // packed-f16 pool words (opaque round-trip)

/// A sequence evicted under KV-pool pressure. On reschedule it either:
/// - `kv = None` (recompute): re-prefills `tokens` (prompt ++ produced); the
///   prefix cache reuses the prompt's blocks. Or
/// - `kv = Some(blob)` (swap-to-host): restores the saved KV and runs one decode
///   step — no recompute.
/// Both are bit-identical to never preempting (KV at p depends only on
/// tokens[0..=p]; same seed index).
struct PreemptedSeq { id: u64, tokens: Vec<u32>, prompt_len: usize, max_tokens: usize, eos: u32, params: SamplingParams, base_seed: u32, kv: Option<HostKv> }

/// A sequence mid-prefill. Its prompt is prefilled one chunk per scheduler step
/// (interleaved with decode of the active batch, so a long prompt doesn't stall
/// in-flight requests). `prefill_pos` is the next position to prefill (starts at
/// the prefix cache's reused length); `cached_blocks` = blocks reused, registered
/// once prefill completes.
struct PrefillingSeq { id: u64, slot: u32, prompt: Vec<u32>, prompt_len: usize, prefill_pos: usize, cached_blocks: usize, max_tokens: usize, eos: u32, params: SamplingParams, base_seed: u32 }


/// Continuous (in-flight) batching scheduler over a `BatchedDecoder`. Sequences
/// are admitted at any time — their prompt is prefilled into a free KV slot —
/// and then all active sequences are decoded together each step regardless of
/// arrival time; finished sequences free their slot for new arrivals. This is
/// the single-device equivalent of datacenter in-flight batching: the GPU runs
/// a full batch instead of one request at a time.
pub struct ContinuousBatcher<'a> {
    dec: BatchedDecoder<'a>,
    free: Vec<u32>,
    active: Vec<CbSeq>,
    /// Sequences evicted under KV-pool pressure, awaiting reschedule (recompute).
    preempted: std::collections::VecDeque<PreemptedSeq>,
    /// Sequences mid-prefill (one chunk advanced per step, interleaved with decode).
    prefilling: std::collections::VecDeque<PrefillingSeq>,
    preemptions: u64,
    /// Preempt active sequences by SWAP-TO-HOST (copy KV out, restore on resume)
    /// instead of recompute. Set from `ZLLM_SWAP`; tests flip it directly. Default
    /// OFF: measured slower than recompute on this UMA box (see `gpu_swap_preemption`)
    /// — it's a discrete-GPU lever. Recompute is the right default here.
    pub(super) swap_mode: bool, // tests in the parent module toggle this
}

impl<'a> ContinuousBatcher<'a> {
    pub fn new(model: &'a GpuModel, m_max: usize, max_seq: usize) -> Self {
        Self { dec: model.batched_decoder(m_max, max_seq), free: (0..m_max as u32).rev().collect(), active: Vec::new(), preempted: Default::default(), prefilling: Default::default(), preemptions: 0, swap_mode: std::env::var("ZLLM_SWAP").is_ok() }
    }

    /// Continuous batcher over a paged KV pool of `n_blocks` blocks shared by all
    /// `m_max` slots. `n_blocks < m_max*ceil(max_seq/block)` overcommits memory:
    /// many short sequences fit where a contiguous max_seq-per-slot reservation
    /// could not. Admission is optimistic; if a running sequence can't grow, a
    /// victim is preempted (its KV freed) and recomputed later.
    pub fn with_pool(model: &'a GpuModel, m_max: usize, max_seq: usize, n_blocks: usize) -> Self {
        Self { dec: model.batched_decoder_paged(m_max, max_seq, n_blocks), free: (0..m_max as u32).rev().collect(), active: Vec::new(), preempted: Default::default(), prefilling: Default::default(), preemptions: 0, swap_mode: std::env::var("ZLLM_SWAP").is_ok() }
    }
    pub fn has_free(&self) -> bool { !self.free.is_empty() }
    pub fn active_len(&self) -> usize { self.active.len() }
    /// True if every active sequence is greedy (temp 0, no top-k) — the precondition
    /// for step_pld (whose argmax verify is bit-identical to greedy step()).
    pub fn all_greedy(&self) -> bool { self.active.iter().all(|s| !s.params.needs_topk() && s.params.temp == 0.0) }

    /// Cancel a sequence by id (e.g. its client disconnected): remove it from
    /// wherever it is — decoding, prefilling, or preempted — and free its KV slot
    /// + blocks. Returns true if it was found. Reclaims capacity immediately
    /// instead of generating output nobody will read.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(pos) = self.active.iter().position(|s| s.id == id) {
            let s = self.active.remove(pos);
            self.dec.free_slot(s.slot);
            self.free.push(s.slot);
            return true;
        }
        if let Some(pos) = self.prefilling.iter().position(|s| s.id == id) {
            let s = self.prefilling.remove(pos).unwrap();
            self.dec.free_slot(s.slot);
            self.free.push(s.slot);
            return true;
        }
        if let Some(pos) = self.preempted.iter().position(|s| s.id == id) {
            self.preempted.remove(pos); // preempted holds no slot/blocks
            return true;
        }
        false
    }

    /// Sequences currently mid-prefill (chunked).
    pub fn prefilling_len(&self) -> usize { self.prefilling.len() }
    /// Sequences currently preempted (evicted, awaiting recompute).
    pub fn preempted_len(&self) -> usize { self.preempted.len() }
    /// Total preemptions since construction (observability).
    pub fn preemption_count(&self) -> u64 { self.preemptions }
    /// (free, total) physical KV blocks — for observing pool pressure.
    pub fn block_pool(&self) -> (usize, usize) { (self.dec.free_blocks(), self.dec.n_blocks) }
    /// Prefix-cache stats: (reused blocks, freshly-prefilled blocks) since start.
    pub fn cache_stats(&self) -> (u64, u64) { self.dec.cache_stats() }

    /// Admit a sequence: batched-prefill its prompt into a free KV slot and join
    /// the decode batch. Returns (first_token, done) — `done` is true if that one
    /// token already completes the request (EOS or max_tokens<=1), in which case
    /// the slot + its blocks are returned immediately. Returns None if no slot is
    /// free OR the KV pool can't fit prompt+max_tokens (caller should retry later;
    /// with the default full pool this never trips).
    pub fn admit(&mut self, id: u64, prompt: &[u32], max_tokens: usize, eos: u32) -> Option<(u32, bool)> {
        self.admit_sampled(id, prompt, max_tokens, eos, 0.0, 0)
    }

    /// Admit with temperature sampling (`temp ≤ 0` = greedy), seeded by `seed`.
    pub fn admit_sampled(&mut self, id: u64, prompt: &[u32], max_tokens: usize, eos: u32, temp: f32, seed: u32) -> Option<(u32, bool)> {
        self.admit_params(id, prompt, max_tokens, eos, SamplingParams::temperature(temp), seed)
    }

    /// Admit with full sampling params (temperature, top-k, top-p), seeded by
    /// `seed` (reproducible per request). Note: the first (prefill) token uses
    /// temperature only; top-k/top-p apply from the first decode token on.
    pub fn admit_params(&mut self, id: u64, prompt: &[u32], max_tokens: usize, eos: u32, params: SamplingParams, seed: u32) -> Option<(u32, bool)> {
        // Optimistic admission: only the prompt's prefill must fit now; decode
        // growth is handled by preemption (make_room) if the pool fills later.
        if !self.dec.can_fit(prompt.len()) { return None; }
        let slot = self.free.pop()?;
        // First token = generation index 0 (temperature only).
        let samp = if params.temp > 0.0 { Some((params.temp, step_seed(seed, 0))) } else { None };
        let (g, _cached_len) = self.dec.prefill_slot_cached(prompt, slot, samp); // reuse shared-prefix KV; prefill the rest
        let done = max_tokens <= 1 || g == eos;
        if done {
            self.dec.free_slot(slot);
            self.free.push(slot);
        } else {
            let mut tokens = Vec::with_capacity(prompt.len() + max_tokens);
            tokens.extend_from_slice(prompt);
            tokens.push(g);
            self.active.push(CbSeq { id, slot, tokens, prompt_len: prompt.len(), max_tokens, eos, params, base_seed: seed });
        }
        Some((g, done))
    }

    /// Enqueue a sequence for CHUNKED prefill: assign a slot, reuse any cached
    /// prefix, then prefill one chunk per `step()` (interleaved with decode of the
    /// active batch, so a long prompt doesn't stall in-flight requests) until the
    /// prompt completes and it joins the decode batch. The first token is emitted
    /// by a later `step()`, not returned here. Returns false if no slot is free or
    /// the pool can't fit the prompt.
    pub fn enqueue_params(&mut self, id: u64, prompt: Vec<u32>, max_tokens: usize, eos: u32, params: SamplingParams, seed: u32) -> bool {
        if prompt.is_empty() || prompt.len() > self.dec.max_seq_len() { return false; }
        if self.free.is_empty() || !self.dec.can_fit(prompt.len()) { return false; }
        let slot = self.free.pop().unwrap();
        let prompt_len = prompt.len();
        let (cached_len, cached_blocks) = self.dec.prefill_prefix_reuse(&prompt, slot);
        self.prefilling.push_back(PrefillingSeq { id, slot, prompt, prompt_len, prefill_pos: cached_len, cached_blocks, max_tokens, eos, params, base_seed: seed });
        true
    }

    /// One scheduler step: decode the active batch, advance one chunk of the front
    /// prefilling sequence, and resume any preempted sequences that now fit.
    /// Returns (id, new_token, done) for every token produced this step.
    pub fn step(&mut self) -> Vec<(u64, u32, bool)> {
        let mut out = Vec::new();
        if !self.active.is_empty() {
            self.make_room(); // preempt victims so every active sequence can grow
            let toks: Vec<u32> = self.active.iter().map(|s| s.next()).collect();
            let pos: Vec<u32> = self.active.iter().map(|s| s.pos()).collect();
            let slots: Vec<u32> = self.active.iter().map(|s| s.slot).collect();
            let seeds: Vec<u32> = self.active.iter().map(|s| step_seed(s.base_seed, s.gen_index() as u32)).collect();
            // Cheapest path that satisfies every active stream:
            // top-k/top-p (CPU sample over GPU top-K) > temperature (Gumbel) > greedy.
            let nexts = if self.active.iter().any(|s| s.params.needs_topk()) {
                let params: Vec<SamplingParams> = self.active.iter().map(|s| s.params).collect();
                self.dec.step_slotted_topk(&toks, &pos, &slots, &params, &seeds)
            } else if self.active.iter().any(|s| s.params.temp > 0.0) {
                let temps: Vec<f32> = self.active.iter().map(|s| s.params.temp).collect();
                self.dec.step_slotted_sample(&toks, &pos, &slots, &temps, &seeds)
            } else {
                self.dec.step_slotted(&toks, &pos, &slots)
            };
            for (i, &nt) in nexts.iter().enumerate() {
                let s = &mut self.active[i];
                s.tokens.push(nt);
                let done = nt == s.eos || s.gen_index() >= s.max_tokens;
                out.push((s.id, nt, done));
            }
            let (free, dec) = (&mut self.free, &self.dec);
            self.active.retain(|s| {
                let done = s.next() == s.eos || s.gen_index() >= s.max_tokens;
                if done { dec.free_slot(s.slot); free.push(s.slot); }
                !done
            });
        }
        self.advance_prefill(&mut out); // one prefill chunk, interleaved with decode
        self.reschedule(&mut out); // resume preempted sequences that now fit
        out
    }

    /// Like `step()` but with prompt-lookup speculative decode IN the batch: every
    /// active sequence proposes an n-gram draft (from its own history), all drafts
    /// concatenate into ONE `step_slotted` forward (each row attends its sequence's
    /// slot at its position), and each sequence commits the tokens its own argmax
    /// agrees with. >1 token/stream/forward on echo-heavy workloads. GREEDY only —
    /// the verify is argmax, so output is bit-identical to `step()`; sequences with
    /// `temp>0` would diverge from sampling, so route those through `step()` instead.
    /// Assumes the full KV pool (a draft writes up to `draft_k+1` positions/step).
    pub fn step_pld(&mut self, lookup_len: usize, draft_k: usize) -> Vec<(u64, u32, bool)> {
        let mut out = Vec::new();
        if !self.active.is_empty() {
            self.make_room();
            let (mut rows, mut positions, mut slots) = (Vec::new(), Vec::new(), Vec::new());
            let mut layout: Vec<(usize, usize, Vec<u32>)> = Vec::new(); // (active_idx, start, draft)
            for (i, s) in self.active.iter().enumerate() {
                let k = draft_k.min(s.max_tokens.saturating_sub(s.gen_index()));
                let draft = if k >= 1 {
                    crate::engine::spec_decode::lookup_draft_best(&s.tokens, &s.tokens, lookup_len, k).unwrap_or_default()
                } else { Vec::new() };
                let start = rows.len();
                rows.push(s.next()); rows.extend_from_slice(&draft);
                let p0 = s.pos();
                for j in 0..=draft.len() { positions.push(p0 + j as u32); slots.push(s.slot); }
                layout.push((i, start, draft));
            }
            let outs = self.dec.step_slotted(&rows, &positions, &slots);
            for (i, start, draft) in &layout {
                let so = &outs[*start..*start + 1 + draft.len()];
                let mut acc = 0usize; while acc < draft.len() && so[acc] == draft[acc] { acc += 1; }
                let s = &mut self.active[*i];
                for &tok in so.iter().take(acc + 1) {
                    s.tokens.push(tok);
                    let done = tok == s.eos || s.gen_index() >= s.max_tokens;
                    out.push((s.id, tok, done));
                    if done { break; }
                }
            }
            let (free, dec) = (&mut self.free, &self.dec);
            self.active.retain(|s| {
                let done = s.next() == s.eos || s.gen_index() >= s.max_tokens;
                if done { dec.free_slot(s.slot); free.push(s.slot); }
                !done
            });
        }
        self.advance_prefill(&mut out);
        self.reschedule(&mut out);
        out
    }

    /// Preempt the most-recently-admitted active sequence (LIFO), queuing it for
    /// recompute. Returns false if there are no active sequences.
    fn preempt_last_active(&mut self) -> bool {
        let Some(victim) = self.active.pop() else { return false };
        self.preemptions += 1;
        // Swap-to-host: copy KV out (frees the blocks) so resume skips recompute.
        // Recompute: just free the blocks; `kv = None` re-prefills on resume.
        // Valid KV covers positions 0..tokens.len()-1 (the last token isn't fed yet).
        let kv = if self.swap_mode {
            Some(self.dec.swap_out(victim.slot, victim.tokens.len() - 1))
        } else {
            self.dec.free_slot(victim.slot);
            None
        };
        self.free.push(victim.slot);
        self.preempted.push_back(PreemptedSeq {
            id: victim.id, tokens: victim.tokens, prompt_len: victim.prompt_len,
            max_tokens: victim.max_tokens, eos: victim.eos, params: victim.params, base_seed: victim.base_seed, kv,
        });
        true
    }

    /// Preempt the most-recently-enqueued *prefilling* sequence (not the front).
    /// It has produced no tokens, so it re-prefills its prompt from scratch on
    /// reschedule. Returns false if there is at most the front prefilling sequence.
    fn preempt_last_prefill(&mut self) -> bool {
        if self.prefilling.len() < 2 { return false; }
        let pf = self.prefilling.pop_back().unwrap();
        self.dec.free_slot(pf.slot);
        self.free.push(pf.slot);
        self.preemptions += 1;
        self.preempted.push_back(PreemptedSeq {
            id: pf.id, tokens: pf.prompt, prompt_len: pf.prompt_len,
            max_tokens: pf.max_tokens, eos: pf.eos, params: pf.params, base_seed: pf.base_seed, kv: None,
        });
        true
    }

    /// Preempt (LIFO) active sequences until the pool can supply the blocks every
    /// remaining active sequence needs to grow this step. A single sequence always
    /// fits (pool ≥ one full sequence), so this terminates.
    fn make_room(&mut self) {
        let needed = |b: &BatchedDecoder, active: &[CbSeq]| -> usize {
            active.iter().map(|s| b.blocks_short(s.slot, s.tokens.len())).sum()
        };
        while self.active.len() > 1 && self.dec.free_blocks() < needed(&self.dec, &self.active) {
            self.preempt_last_active();
        }
    }

    /// Advance the front prefilling sequence by one chunk (prefill-priority: makes
    /// room by preempting active — then, if still short, other prefilling — so the
    /// front always progresses). On completion, registers its blocks and the
    /// sequence joins the decode batch, emitting its first token into `out`.
    fn advance_prefill(&mut self, out: &mut Vec<(u64, u32, bool)>) {
        if self.prefilling.is_empty() { return; }
        let (slot, start, chunk_end, temp, base_seed) = {
            let pf = &self.prefilling[0];
            let end = (pf.prefill_pos + self.dec.prefill_chunk_size()).min(pf.prompt_len);
            (pf.slot, pf.prefill_pos, end, pf.params.temp, pf.base_seed)
        };
        // Make the chunk's blocks fit (preempt active first, then other prefills).
        while self.dec.blocks_short(slot, chunk_end) > self.dec.free_blocks() {
            if !self.preempt_last_active() && !self.preempt_last_prefill() { return; } // can't fit yet; wait
        }
        let samp = if temp > 0.0 { Some((temp, step_seed(base_seed, 0))) } else { None };
        let (next, g0) = self.dec.prefill_chunk(&self.prefilling[0].prompt, slot, start, samp);
        self.prefilling[0].prefill_pos = next;
        let Some(g) = g0 else { return }; // chunk done but prompt not complete
        let pf = self.prefilling.pop_front().unwrap();
        self.dec.prefill_register(&pf.prompt, pf.slot, pf.cached_blocks);
        let done = pf.max_tokens <= 1 || g == pf.eos;
        out.push((pf.id, g, done));
        if done {
            self.dec.free_slot(pf.slot);
            self.free.push(pf.slot);
        } else {
            let mut tokens = pf.prompt;
            tokens.push(g);
            self.active.push(CbSeq { id: pf.id, slot: pf.slot, tokens, prompt_len: pf.prompt_len, max_tokens: pf.max_tokens, eos: pf.eos, params: pf.params, base_seed: pf.base_seed });
        }
    }

    /// One single-row decode step (`tok` at `pos` in `slot`), dispatching on the
    /// sequence's sampling params, returning the next token. Used to resume a
    /// swap-to-host sequence after its KV is restored.
    fn decode_one(&self, tok: u32, pos: u32, slot: u32, params: SamplingParams, seed: u32) -> u32 {
        let (t, p, s, sd) = ([tok], [pos], [slot], [seed]);
        if params.needs_topk() {
            self.dec.step_slotted_topk(&t, &p, &s, &[params], &sd)[0]
        } else if params.temp > 0.0 {
            self.dec.step_slotted_sample(&t, &p, &s, &[params.temp], &sd)[0]
        } else {
            self.dec.step_slotted(&t, &p, &s)[0]
        }
    }

    /// Resume preempted sequences (FIFO) while a slot is free and the pool can
    /// hold them. Swap-to-host: restore the saved KV + one decode step. Recompute
    /// (`kv = None`): re-prefill prompt ++ produced (prefix cache reuses the
    /// prompt). Either way, produces the exact next token they'd have produced.
    fn reschedule(&mut self, out: &mut Vec<(u64, u32, bool)>) {
        while !self.preempted.is_empty() && !self.free.is_empty() {
            let len = self.preempted.front().unwrap().tokens.len();
            if !self.dec.can_fit(len) { break; }
            let p = self.preempted.pop_front().unwrap();
            let slot = self.free.pop().unwrap();
            let gen_idx = (p.tokens.len() - p.prompt_len) as u32;
            let seed = step_seed(p.base_seed, gen_idx);
            let g = if let Some(kv) = &p.kv {
                self.dec.swap_in(slot, kv); // write KV back into fresh blocks…
                let next = *p.tokens.last().unwrap();
                self.decode_one(next, (p.tokens.len() - 1) as u32, slot, p.params, seed) // …then one step
            } else {
                let samp = if p.params.temp > 0.0 { Some((p.params.temp, seed)) } else { None };
                self.dec.prefill_slot_cached(&p.tokens, slot, samp).0
            };
            let mut tokens = p.tokens;
            tokens.push(g);
            let done = g == p.eos || (tokens.len() - p.prompt_len) >= p.max_tokens;
            out.push((p.id, g, done));
            if done {
                self.dec.free_slot(slot);
                self.free.push(slot);
            } else {
                self.active.push(CbSeq { id: p.id, slot, tokens, prompt_len: p.prompt_len, max_tokens: p.max_tokens, eos: p.eos, params: p.params, base_seed: p.base_seed });
            }
        }
    }
}

/// A generation request submitted to a [`GpuBatchServer`].
pub struct GenReq {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub eos: u32,
    /// Sampling knobs and per-request RNG seed (reproducible).
    pub params: SamplingParams,
    pub seed: u32,
    /// The server pushes `Some(token)` per produced token, then `None` at
    /// completion. Use a tokio unbounded channel so an async HTTP handler can
    /// stream from the receiver while the (sync) serving thread sends.
    pub tok_tx: tokio::sync::mpsc::UnboundedSender<Option<u32>>,
}

/// A GPU continuous-batching serving loop on its own OS thread. It OWNS the
/// `GpuModel` (and the `ContinuousBatcher` that borrows it), so there is no
/// borrow-across-`Arc<Mutex>` problem: handlers communicate only by channel.
/// `submit()` enqueues a prompt + a token channel; the loop admits it into a
/// free KV slot and decodes it together with every other in-flight request,
/// streaming tokens back and freeing the slot on completion.
/// A control message to the serving thread: a generation request, or a model
/// hot-swap (drop the batcher, load a new model, rebuild — acks when done).
enum CbMsg {
    Gen(GenReq),
    Swap { path: String, ack: std::sync::mpsc::Sender<bool> },
}

pub struct GpuBatchServer {
    tx: std::sync::mpsc::Sender<CbMsg>,
    m_max: usize,
}

impl GpuBatchServer {
    /// Spawn the serving thread. `model` is MOVED onto it (wgpu device/queue are
    /// Send). `m_max` = max concurrent sequences (KV slots), `max_seq` = max
    /// context length per slot.
    pub fn spawn(model: GpuModel, m_max: usize, max_seq: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<CbMsg>();
        std::thread::Builder::new()
            .name("gpu-batcher".into())
            .spawn(move || Self::serve(model, rx, m_max, max_seq))
            .expect("spawn gpu-batcher thread");
        Self { tx, m_max }
    }

    /// Max concurrent sequences the server can hold in flight.
    pub fn capacity(&self) -> usize { self.m_max }

    /// Submit a request. Returns a receiver yielding `Some(token)` per decode
    /// step then `None` at completion, or `Err` if the serving thread is gone.
    /// `params.temp ≤ 0` = greedy; `seed` makes sampling reproducible per request.
    pub fn submit(&self, prompt: Vec<u32>, max_tokens: usize, eos: u32, params: SamplingParams, seed: u32)
        -> Result<tokio::sync::mpsc::UnboundedReceiver<Option<u32>>, ()> {
        let (tok_tx, tok_rx) = tokio::sync::mpsc::unbounded_channel();
        self.tx.send(CbMsg::Gen(GenReq { prompt, max_tokens, eos, params, seed, tok_tx })).map_err(|_| ())?;
        Ok(tok_rx)
    }

    /// Hot-swap the served model (e.g. on `/v1/models/select`). In-flight
    /// sequences are aborted (their KV is on the old model). Blocks until the new
    /// model is loaded; returns false if the load failed or the thread is gone.
    pub fn swap_model(&self, path: String) -> bool {
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        if self.tx.send(CbMsg::Swap { path, ack: ack_tx }).is_err() { return false; }
        ack_rx.recv().unwrap_or(false)
    }

    fn serve(mut model: GpuModel, rx: std::sync::mpsc::Receiver<CbMsg>, m_max: usize, max_seq: usize) {
        use std::sync::mpsc::TryRecvError;
        let mut next_id: u64 = 0;
        let pld = std::env::var("ZLLM_CB_PLD").is_ok(); // batched spec-decode for greedy batches
        loop { // one iteration per loaded model
            let mut cb = ContinuousBatcher::new(&model, m_max, max_seq);
            let mut chans: std::collections::HashMap<u64, tokio::sync::mpsc::UnboundedSender<Option<u32>>> = Default::default();
            #[allow(unused_assignments)] // the None init is dead: every path to the consumer sets Some
            let mut reload: Option<(String, std::sync::mpsc::Sender<bool>)> = None;
            'serve: loop {
                // Idle (nothing active, prefilling, or preempted) → block for a message.
                let busy = cb.active_len() + cb.prefilling_len() + cb.preempted_len() > 0;
                if !busy {
                    match rx.recv() {
                        Ok(CbMsg::Gen(req)) => Self::admit_req(&mut cb, &mut chans, &mut next_id, req),
                        Ok(CbMsg::Swap { path, ack }) => { reload = Some((path, ack)); break 'serve; }
                        Err(_) => return, // every sender dropped → shut down
                    }
                }
                // Fill free slots with any waiting requests (non-blocking).
                while cb.has_free() {
                    match rx.try_recv() {
                        Ok(CbMsg::Gen(req)) => Self::admit_req(&mut cb, &mut chans, &mut next_id, req),
                        Ok(CbMsg::Swap { path, ack }) => { reload = Some((path, ack)); break 'serve; }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => { if cb.active_len() + cb.prefilling_len() + cb.preempted_len() == 0 { return; } break; }
                    }
                }
                if cb.active_len() + cb.prefilling_len() + cb.preempted_len() == 0 { continue; }
                // One scheduler step; stream + retire. ZLLM_CB_PLD turns on batched
                // prompt-lookup spec-decode when the whole active batch is greedy
                // (bit-identical to step(), fewer forwards on echo/RAG/code).
                let res = if pld && cb.all_greedy() { cb.step_pld(3, 7) } else { cb.step() };
                for (id, tok, done) in res {
                    if let Some(ch) = chans.get(&id) { let _ = ch.send(Some(tok)); }
                    if done { if let Some(ch) = chans.remove(&id) { let _ = ch.send(None); } }
                }
                // Reclaim sequences whose client disconnected (covers mid-prefill,
                // where no token has been sent yet to surface a send error).
                chans.retain(|id, ch| if ch.is_closed() { cb.cancel(*id); false } else { true });
            }
            // Swap requested: abort in-flight sequences (their KV is on the old model).
            for (_, ch) in chans.drain() { let _ = ch.send(None); }
            let Some((path, ack)) = reload else { return };
            drop(cb); // release the borrow of `model` before replacing it
            match GpuContext::new().and_then(|ctx| GpuModel::load(&path, ctx)) {
                Ok(m) => { model = m; let _ = ack.send(true); }
                Err(_) => { let _ = ack.send(false); return; } // failed load → shut down
            }
        }
    }

    fn admit_req(
        cb: &mut ContinuousBatcher,
        chans: &mut std::collections::HashMap<u64, tokio::sync::mpsc::UnboundedSender<Option<u32>>>,
        next_id: &mut u64,
        req: GenReq,
    ) {
        let id = *next_id; *next_id += 1;
        // Chunked prefill: enqueue; the first token is emitted by a later step().
        if cb.enqueue_params(id, req.prompt, req.max_tokens, req.eos, req.params, req.seed) {
            chans.insert(id, req.tok_tx);
        } else {
            let _ = req.tok_tx.send(None); // rejected (no slot / pool too small)
        }
    }
}
