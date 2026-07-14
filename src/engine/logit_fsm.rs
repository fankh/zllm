use crate::backend::traits::Tensor;
use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::util::primitives::StateID;
use regex_automata::{Anchored, Input, MatchKind};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Logit-level decoding constraints.
///
/// Supported modes:
///
/// - `grammar = ""`  → no constraint (default).
/// - `grammar = "ban:<id>,<id>,…"`  → set the listed token-id logits to
///   `-inf`, so they're never sampled. Stateless.
/// - `grammar = "regex:<pattern>"`  → the **entire generated text** must match
///   `<pattern>` (anchored). The pattern is compiled to a byte-level DFA
///   (`regex-automata`); each step masks every token whose surface bytes would
///   drive the DFA into a dead state, and EOS is only allowed when the text so
///   far is a complete match. Requires a [`TokenByteTable`] (tokenizer vocab →
///   surface bytes), so use [`LogitFSM::compile`] with a table.
/// - `grammar = "json_schema:<schema>"`  → the text must be a JSON document
///   satisfying `<schema>` (a JSON string). The schema is compiled to an
///   anchored regex by [`crate::engine::json_schema`] and enforced by the same
///   DFA engine as `regex:`. Also needs a [`TokenByteTable`].
/// - `grammar = "json:"`  → the text must be any bounded, well-formed JSON
///   value (no fixed shape). Also needs a [`TokenByteTable`].
///
/// Reserved-but-unimplemented mode (`bnf:`) is reported via
/// [`unsupported_mode`](LogitFSM::unsupported_mode) / [`compile`](LogitFSM::compile)
/// errors so handlers can reject it loudly instead of silently generating
/// unconstrained output.
pub struct LogitFSM {
    grammar: String,
    banned: Vec<u32>,
    regex: Option<RegexEngine>,
}

/// Tokenizer vocab → per-token surface bytes, for grammar masking. Build via
/// `LlamaTokenizer::token_bytes_table` (one decode per vocab entry — cache per
/// model). `None` entries (special tokens / non-UTF-8 partial pieces) are
/// disallowed whenever a grammar is active.
pub struct TokenByteTable {
    pub bytes: Vec<Option<Vec<u8>>>,
    pub eos: u32,
}

/// Anchored byte-DFA over the generated text. `state` is interior-mutable so
/// the generation loops can hold `&LogitFSM` (as they already do) and still
/// advance. Per-DFA-state allow-masks are memoized: computing one walks the
/// whole vocab (~128k tokens × avg ~5 bytes of table lookups, low-ms), but
/// generation revisits a small set of states, so the amortized cost per token
/// is ~a hash lookup. Single-threaded per request (RefCell, not Mutex).
struct RegexEngine {
    dfa: dense::DFA<Vec<u32>>,
    state: Cell<StateID>,
    table: Arc<TokenByteTable>,
    mask_cache: RefCell<HashMap<StateID, Arc<Vec<bool>>>>,
    live_cache: RefCell<HashMap<StateID, bool>>,
}

impl RegexEngine {
    fn compile(pattern: &str, table: Arc<TokenByteTable>) -> Result<Self, String> {
        // Anchored: the generated text must match from its first byte.
        // MatchKind::All: acceptance semantics (no leftmost-first priority) —
        // we're testing "does the whole text match", not searching.
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(StartKind::Anchored)
                    .match_kind(MatchKind::All)
                    .minimize(true),
            )
            .build(pattern)
            .map_err(|e| format!("regex compile failed: {e}"))?;
        let start = dfa
            .start_state_forward(&Input::new(&[] as &[u8]).anchored(Anchored::Yes))
            .map_err(|e| format!("regex start state: {e}"))?;
        Ok(Self {
            dfa,
            state: Cell::new(start),
            table,
            mask_cache: RefCell::new(HashMap::new()),
            live_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Walk `bytes` from `from`; `None` = hits the dead/quit state (not viable).
    fn walk(&self, from: StateID, bytes: &[u8]) -> Option<StateID> {
        let mut s = from;
        for &b in bytes {
            s = self.dfa.next_state(s, b);
            if self.dfa.is_dead_state(s) || self.dfa.is_quit_state(s) {
                return None;
            }
        }
        Some(s)
    }

    /// Is the text consumed so far a complete match? (match states are delayed
    /// by one transition in regex-automata, hence next_eoi_state.)
    fn is_accept(&self, s: StateID) -> bool {
        self.dfa.is_match_state(self.dfa.next_eoi_state(s))
    }

    /// Liveness: can ANY byte sequence from `s` (including the empty one) reach
    /// a match? regex-automata does NOT merge match-unreachable states into its
    /// canonical dead state (measured: after a full match of `ab*c`, further
    /// bytes lead to non-dead states), so dead-state checks alone under-prune.
    /// Computed by exploring the reachable subgraph (256 bytes/state; DFA
    /// subgraphs are small) + a reverse fixpoint, memoized for every state
    /// discovered along the way — so effectively one-time per engine.
    fn is_live(&self, s: StateID) -> bool {
        if let Some(&v) = self.live_cache.borrow().get(&s) {
            return v;
        }
        let mut stack = vec![s];
        let mut seen: HashSet<StateID> = HashSet::from([s]);
        let mut succ: HashMap<StateID, Vec<StateID>> = HashMap::new();
        while let Some(u) = stack.pop() {
            let mut outs = Vec::new();
            for b in 0..=255u8 {
                let v = self.dfa.next_state(u, b);
                if self.dfa.is_dead_state(v) || self.dfa.is_quit_state(v) {
                    continue;
                }
                if !outs.contains(&v) {
                    outs.push(v);
                }
                if seen.insert(v) {
                    stack.push(v);
                }
            }
            succ.insert(u, outs);
        }
        let mut live: HashMap<StateID, bool> =
            seen.iter().map(|&u| (u, self.is_accept(u))).collect();
        loop {
            let mut changed = false;
            for (&u, outs) in &succ {
                if !live[&u] && outs.iter().any(|v| live[v]) {
                    live.insert(u, true);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let result = live[&s];
        self.live_cache.borrow_mut().extend(live);
        result
    }

    /// allowed[token_id] for the current state (memoized per state). A token is
    /// allowed iff its bytes avoid dead states AND land somewhere a match is
    /// still reachable from.
    fn allowed_mask(&self, s: StateID) -> Arc<Vec<bool>> {
        if let Some(m) = self.mask_cache.borrow().get(&s) {
            return m.clone();
        }
        let mask: Vec<bool> = self
            .table
            .bytes
            .iter()
            .map(|tb| {
                tb.as_ref()
                    .is_some_and(|b| self.walk(s, b).is_some_and(|t| self.is_live(t)))
            })
            .collect();
        let mask = Arc::new(mask);
        self.mask_cache.borrow_mut().insert(s, mask.clone());
        mask
    }
}

impl LogitFSM {
    /// Compile a grammar. `table` is required for `regex:` mode (pass the
    /// cached per-model [`TokenByteTable`]); `ban:`/empty ignore it. Errors are
    /// user-facing strings (bad pattern, unimplemented mode, missing table) —
    /// handlers should surface them as 400s.
    pub fn compile(grammar: &str, table: Option<Arc<TokenByteTable>>) -> Result<Self, String> {
        let trimmed = grammar.trim();
        let mut banned: Vec<u32> = Vec::new();
        let mut regex = None;

        if let Some(rest) = trimmed.strip_prefix("ban:") {
            for piece in rest.split(',') {
                if let Ok(id) = piece.trim().parse::<u32>() {
                    banned.push(id);
                }
            }
        } else if let Some(pattern) = trimmed.strip_prefix("regex:") {
            let table = table.ok_or("regex grammar needs a tokenizer byte table")?;
            regex = Some(RegexEngine::compile(pattern, table)?);
        } else if let Some(schema_json) = trimmed.strip_prefix("json_schema:") {
            // Concrete schema → anchored regex (OpenAI structured-outputs strict
            // mode). The schema compiler owns the subset/limits; its errors are
            // user-facing 400 strings.
            let table = table.ok_or("json_schema grammar needs a tokenizer byte table")?;
            let schema: serde_json::Value = serde_json::from_str(schema_json.trim())
                .map_err(|e| format!("json_schema: invalid JSON schema: {e}"))?;
            let pattern = crate::engine::json_schema::schema_to_regex(&schema)?;
            regex = Some(RegexEngine::compile(&pattern, table)?);
        } else if trimmed.strip_prefix("json:").is_some() {
            // json_object mode: any bounded, well-formed JSON value (no fixed
            // shape). Any payload after `json:` is ignored — there's no schema.
            let table = table.ok_or("json grammar needs a tokenizer byte table")?;
            let pattern = crate::engine::json_schema::any_json_regex_default();
            regex = Some(RegexEngine::compile(&pattern, table)?);
        } else if !trimmed.is_empty() {
            let mode = Self::mode_name(trimmed).unwrap_or("unknown");
            return Err(format!(
                "grammar mode {mode:?} is not implemented yet; supported: \"ban:<id>,<id>,...\", \"regex:<pattern>\""
            ));
        }

        Ok(Self { grammar: trimmed.to_string(), banned, regex })
    }

    /// Lenient constructor for `ban:`/empty grammars (legacy call sites and
    /// tests). Unsupported/regex modes degrade to inactive with a warning —
    /// server handlers should use [`compile`](Self::compile) and 400 instead.
    pub fn new(grammar: &str) -> Self {
        Self::compile(grammar, None).unwrap_or_else(|e| {
            tracing::warn!("LogitFSM::new: {e}; constraint inactive");
            Self { grammar: grammar.trim().to_string(), banned: Vec::new(), regex: None }
        })
    }

    /// Mutate `logits` in-place to enforce the active constraint.
    ///
    /// `ban:` sets the listed ids to `-inf`. `regex:` masks every token whose
    /// bytes are not a viable continuation from the current DFA state; EOS is
    /// allowed iff the text so far is a complete match — except when *nothing*
    /// is viable (pattern dead-ended), where EOS is forced so generation stops
    /// instead of sampling from an all-`-inf` distribution.
    pub fn apply_mask(&self, logits: &mut Tensor) {
        for &id in &self.banned {
            if let Some(slot) = logits.get_mut(id as usize) {
                *slot = f32::NEG_INFINITY;
            }
        }
        if let Some(rx) = &self.regex {
            let s = rx.state.get();
            let allowed = rx.allowed_mask(s);
            let accept = rx.is_accept(s);
            let any_token_viable = allowed.iter().any(|&a| a);
            let eos = rx.table.eos as usize;
            for (id, slot) in logits.iter_mut().enumerate() {
                let ok = if id == eos {
                    accept || !any_token_viable // force-stop escape hatch
                } else {
                    allowed.get(id).copied().unwrap_or(false)
                };
                if !ok {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// State transition: feed the token that was actually sampled. No-op for
    /// stateless modes. Interior-mutable so the generation loops can call it
    /// through the `&LogitFSM` they already hold.
    pub fn advance(&self, token_id: u32) {
        if let Some(rx) = &self.regex {
            if token_id == rx.table.eos {
                return;
            }
            if let Some(bytes) = rx.table.bytes.get(token_id as usize).and_then(|b| b.as_ref()) {
                if let Some(next) = rx.walk(rx.state.get(), bytes) {
                    rx.state.set(next);
                } else {
                    // Shouldn't happen if apply_mask ran (masked tokens can't be
                    // sampled); log rather than corrupt the state.
                    tracing::warn!("LogitFSM::advance: token {token_id} not viable from current state");
                }
            }
        }
    }

    pub fn grammar(&self) -> &str {
        &self.grammar
    }

    /// True when the active constraint can actually affect sampling. Cheap
    /// way for the inference loop to skip `apply_mask` calls when there's
    /// nothing to mask.
    pub fn is_active(&self) -> bool {
        !self.banned.is_empty() || self.regex.is_some()
    }

    fn mode_name(grammar: &str) -> Option<&'static str> {
        for mode in ["regex", "json_schema", "json", "bnf"] {
            if grammar.strip_prefix(mode).is_some_and(|r| r.starts_with(':')) {
                return Some(mode);
            }
        }
        None
    }

    /// `Some(mode)` when the requested grammar names a mode that is NOT
    /// implemented (json / json_schema / bnf — recognized but stubbed).
    /// Handlers reject such requests with a 4xx instead of silently
    /// generating unconstrained output. (`regex` is implemented — construct
    /// via [`compile`](Self::compile) with a byte table.)
    pub fn unsupported_mode(&self) -> Option<&str> {
        match Self::mode_name(&self.grammar) {
            Some("regex") | Some("json_schema") | Some("json") => None,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny synthetic vocab: enough to exercise viability, multi-byte tokens,
    /// accept-state EOS, and the force-stop escape hatch without a tokenizer.
    fn tiny_table() -> Arc<TokenByteTable> {
        Arc::new(TokenByteTable {
            bytes: vec![
                Some(b"a".to_vec()),  // 0
                Some(b"b".to_vec()),  // 1
                Some(b"ab".to_vec()), // 2
                Some(b"c".to_vec()),  // 3
                None,                 // 4 = special (always banned under grammar)
            ],
            eos: 5,
        })
    }

    fn masked(l: &[f32]) -> Vec<bool> {
        l.iter().map(|x| x.is_infinite() && x.is_sign_negative()).collect()
    }

    #[test]
    fn regex_masks_to_viable_prefixes() {
        let f = LogitFSM::compile("regex:ab*c", Some(tiny_table())).unwrap();
        assert!(f.is_active());
        let mut logits = vec![0.0; 6];
        f.apply_mask(&mut logits);
        // From start: "a" viable, "ab" viable, "b"/"c" dead, special banned,
        // EOS banned (no match yet).
        assert_eq!(masked(&logits), vec![false, true, false, true, true, true]);
    }

    #[test]
    fn regex_advance_transitions_and_accepts() {
        let f = LogitFSM::compile("regex:ab*c", Some(tiny_table())).unwrap();
        f.advance(0); // "a"
        let mut logits = vec![0.0; 6];
        f.apply_mask(&mut logits);
        // After "a": "b" and "c" viable; "a"/"ab" dead; EOS still banned.
        assert_eq!(masked(&logits), vec![true, false, true, false, true, true]);
        f.advance(1); // "ab"
        f.advance(3); // "abc" — complete match
        let mut logits = vec![0.0; 6];
        f.apply_mask(&mut logits);
        // Complete: nothing extends "abc" (a,b,ab,c all dead) -> only EOS open.
        assert_eq!(masked(&logits), vec![true, true, true, true, true, false]);
    }

    #[test]
    fn regex_multibyte_token_walks_whole_surface() {
        // "ab" (token 2) consumes two DFA steps in one advance.
        let f = LogitFSM::compile("regex:abc", Some(tiny_table())).unwrap();
        f.advance(2);
        let mut logits = vec![0.0; 6];
        f.apply_mask(&mut logits);
        assert_eq!(masked(&logits), vec![true, true, true, false, true, true]); // only "c"
    }

    #[test]
    fn regex_optional_suffix_allows_eos_and_continuation() {
        let f = LogitFSM::compile("regex:ab?", Some(tiny_table())).unwrap();
        f.advance(0); // "a" is already a full match; "b" may extend it
        let mut logits = vec![0.0; 6];
        f.apply_mask(&mut logits);
        assert_eq!(masked(&logits), vec![true, false, true, true, true, false]); // "b" + EOS
    }

    #[test]
    fn regex_bad_pattern_is_a_compile_error() {
        assert!(LogitFSM::compile("regex:a(", Some(tiny_table())).is_err());
    }

    #[test]
    fn regex_without_table_is_a_compile_error() {
        assert!(LogitFSM::compile("regex:a+", None).is_err());
    }

    #[test]
    fn bnf_mode_still_unimplemented() {
        // bnf is the one remaining reserved-but-stubbed mode.
        let g = "bnf:root ::= x";
        assert!(LogitFSM::compile(g, None).is_err(), "{g} should be a compile error");
        let f = LogitFSM::new(g); // lenient path degrades to inactive
        assert_eq!(f.unsupported_mode(), Some("bnf"));
        assert!(!f.is_active());
    }

    #[test]
    fn json_and_json_schema_are_implemented() {
        // Need a byte table (they compile to regex); without one they error on
        // the missing table, not because the mode is unimplemented.
        assert!(LogitFSM::compile("json:", None).is_err());
        assert!(LogitFSM::compile("json_schema:{\"type\":\"boolean\"}", None).is_err());

        let j = LogitFSM::compile("json:", Some(tiny_table())).unwrap();
        assert!(j.is_active());
        assert_eq!(j.unsupported_mode(), None);

        let s = LogitFSM::compile("json_schema:{\"type\":\"boolean\"}", Some(tiny_table())).unwrap();
        assert!(s.is_active());
        assert_eq!(s.unsupported_mode(), None);

        // A malformed schema payload is a compile error even with a table.
        assert!(LogitFSM::compile("json_schema:{not json}", Some(tiny_table())).is_err());
        // An unsupported schema construct is a compile error.
        assert!(LogitFSM::compile("json_schema:{\"$ref\":\"#/x\"}", Some(tiny_table())).is_err());
    }

    #[test]
    fn supported_forms_not_flagged() {
        assert_eq!(LogitFSM::new("ban:1,2").unsupported_mode(), None);
        assert_eq!(LogitFSM::new("").unsupported_mode(), None);
        let f = LogitFSM::compile("regex:a", Some(tiny_table())).unwrap();
        assert_eq!(f.unsupported_mode(), None);
    }

    #[test]
    fn empty_grammar_is_inactive() {
        let f = LogitFSM::new("");
        assert!(!f.is_active());
        let mut logits = vec![1.0; 4];
        f.apply_mask(&mut logits);
        assert_eq!(logits, vec![1.0; 4]);
    }

    #[test]
    fn ban_mode_masks_listed_tokens() {
        let f = LogitFSM::new("ban:1,3");
        assert!(f.is_active());
        let mut logits = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        f.apply_mask(&mut logits);
        assert_eq!(logits[0], 0.5);
        assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
        assert_eq!(logits[2], 0.5);
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
        assert_eq!(logits[4], 0.5);
    }

    #[test]
    fn ban_mode_tolerates_whitespace_and_garbage() {
        let f = LogitFSM::new("ban:  2 , junk , 4  ");
        let mut logits = vec![1.0; 6];
        f.apply_mask(&mut logits);
        assert!(logits[2].is_infinite());
        assert!(logits[4].is_infinite());
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[5], 1.0);
    }

    #[test]
    fn ban_mode_out_of_range_id_ignored() {
        let f = LogitFSM::new("ban:99");
        let mut logits = vec![0.5; 4];
        f.apply_mask(&mut logits);
        assert_eq!(logits, vec![0.5; 4]);
    }

    #[test]
    fn grammar_string_round_trips() {
        let f = LogitFSM::new("ban:1,2");
        assert_eq!(f.grammar(), "ban:1,2");
    }

    /// End-to-end simulation: greedy "sampling" over rigged logits must be
    /// steered to produce exactly a string matching the pattern.
    #[test]
    fn regex_constrains_a_greedy_loop_end_to_end() {
        let f = LogitFSM::compile("regex:ab+c", Some(tiny_table())).unwrap();
        // Greedy sampler that always prefers token 0 ("a"), then 1, ... —
        // the mask must steer it through a, b, then keep b viable but we pick
        // lowest-index viable each time: a, b, b... to force termination make
        // "b" less preferred than "c" after two steps.
        let prefs: Vec<Vec<usize>> = vec![
            vec![3, 1, 0, 2, 4, 5], // wants "c" first -> mask must force a/ab
            vec![0, 2, 1, 3, 4, 5], // wants "a" again -> mask must force b
            vec![3, 1, 0, 2, 4, 5], // "c" now viable -> allowed through
            vec![0, 1, 2, 3, 4, 5], // wants to continue -> only EOS open
        ];
        let mut out: Vec<usize> = Vec::new();
        for pref in prefs {
            let mut logits = vec![0.0f32; 6];
            for (rank, &id) in pref.iter().enumerate() {
                logits[id] = 10.0 - rank as f32;
            }
            f.apply_mask(&mut logits);
            let pick = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap();
            if pick == 5 {
                break;
            }
            out.push(pick);
            f.advance(pick as u32);
        }
        // tokens: "ab"(2) would also be legal for step 1; greedy pref said "a"... the
        // mask decides viability, preference decides among viable: step1 viable {0,2},
        // pref order 3,1,0,2 -> picks 0 ("a"). step2 viable {1}, -> 1 ("b").
        // step3 viable {1,3}, pref -> 3 ("c"). step4 only EOS -> stop.
        assert_eq!(out, vec![0, 1, 3]);
        let text: String = out.iter().map(|&i| ["a", "b", "ab", "c"][i]).collect();
        assert_eq!(text, "abc");
    }
}
