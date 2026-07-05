use crate::backend::traits::Tensor;

/// Logit-level decoding constraints.
///
/// v0.5 supports one real mode and warns on everything else (instead of
/// silently no-opping the way the v0.4 stub did):
///
/// - `grammar = ""`  → no constraint (default).
/// - `grammar = "ban:<id>,<id>,…"`  → set the listed token-id logits to
///   `-inf`, so they're never sampled. Useful for blocking EOS until at
///   least N tokens are generated, blocking known-bad token IDs from the
///   tokenizer's special tokens, etc. Ignores whitespace; silently drops
///   non-numeric entries.
///
/// Reserved modes (`regex:`, `json:`, `json_schema:`, `bnf:`) emit a
/// `tracing::warn!` on construction and otherwise behave like no-grammar.
/// Wiring those up is v0.7 work — they need tokenizer-state integration
/// (decode each candidate token, check partial match against the
/// constraint) which `LogitFSM` doesn't currently have access to.
pub struct LogitFSM {
    grammar: String,
    banned: Vec<u32>,
}

impl LogitFSM {
    pub fn new(grammar: &str) -> Self {
        let trimmed = grammar.trim();
        let mut banned: Vec<u32> = Vec::new();

        if let Some(rest) = trimmed.strip_prefix("ban:") {
            for piece in rest.split(',') {
                if let Ok(id) = piece.trim().parse::<u32>() {
                    banned.push(id);
                }
            }
        } else if !trimmed.is_empty() {
            tracing::warn!(
                "LogitFSM: grammar {:?} is not yet implemented; logits will be unconstrained. \
                 Supported modes: \"\" (none), \"ban:<id>,<id>,…\".",
                trimmed
            );
        }

        Self {
            grammar: trimmed.to_string(),
            banned,
        }
    }

    /// Mutate `logits` in-place to enforce the active constraint. For
    /// `ban:` mode this sets each listed id's logit to `-inf`. For no /
    /// unsupported grammars, this is a no-op.
    pub fn apply_mask(&self, logits: &mut Tensor) {
        for &id in &self.banned {
            if let Some(slot) = logits.get_mut(id as usize) {
                *slot = f32::NEG_INFINITY;
            }
        }
    }

    /// State-transition hook for stateful grammars. `ban:` is stateless so
    /// this is a no-op; the signature exists so future regex / JSON-schema
    /// modes can advance their FSM on each sampled token.
    pub fn advance(&mut self, _token_id: u32) {}

    pub fn grammar(&self) -> &str {
        &self.grammar
    }

    /// True when the active constraint can actually affect sampling. Cheap
    /// way for the inference loop to skip `apply_mask` calls when there's
    /// nothing to mask.
    pub fn is_active(&self) -> bool {
        !self.banned.is_empty()
    }

    /// `Some(mode)` when the requested grammar names a mode that is NOT yet
    /// implemented (regex / json / json_schema / bnf — recognized but stubbed).
    /// Handlers should reject such requests with a 4xx instead of silently
    /// generating unconstrained output (which is what happens if this is
    /// ignored — `is_active()` stays false and `apply_mask` is a no-op).
    pub fn unsupported_mode(&self) -> Option<&str> {
        for mode in ["regex", "json_schema", "json", "bnf"] {
            if self.grammar.strip_prefix(mode).is_some_and(|r| r.starts_with(':')) {
                return Some(mode);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_modes_are_reported_not_silent() {
        for g in ["regex:a+b", "json:{}", "json_schema:{\"type\":\"object\"}", "bnf:root ::= x"] {
            let f = LogitFSM::new(g);
            assert!(f.unsupported_mode().is_some(), "{g} should report unsupported");
            assert!(!f.is_active());
        }
        // supported / no-constraint forms are NOT flagged
        assert_eq!(LogitFSM::new("ban:1,2").unsupported_mode(), None);
        assert_eq!(LogitFSM::new("").unsupported_mode(), None);
        // json_schema must not be mis-detected as json
        assert_eq!(LogitFSM::new("json_schema:{}").unsupported_mode(), Some("json_schema"));
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
    fn unsupported_grammar_is_inactive_but_does_not_panic() {
        let f = LogitFSM::new("regex:^foo$");
        assert!(!f.is_active());
        let mut logits = vec![1.0; 4];
        f.apply_mask(&mut logits);
        assert_eq!(logits, vec![1.0; 4]);
    }

    #[test]
    fn grammar_string_round_trips() {
        let f = LogitFSM::new("ban:1,2");
        assert_eq!(f.grammar(), "ban:1,2");
    }
}
