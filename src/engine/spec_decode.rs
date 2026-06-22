//! Prompt-Lookup Decoding (PLD).
//!
//! Zero-draft-model speculative decoding: when about to sample the
//! next token, we look back at the recent context for an n-gram match,
//! and propose the tokens that followed it as a "draft". The main
//! model then verifies all drafts in a single batched forward pass,
//! committing as many as it agrees with.
//!
//! Best on **echo-heavy** workloads (summarize, quote, RAG, code
//! refactoring) where the output recycles spans of the input.
//! Useless on open-ended generation — but adds zero cost in that
//! case (one cheap n-gram scan per token, no extra model work).

/// Find a draft proposal of up to `k` tokens for the next position
/// after `recent`. Scans `haystack` (the full token sequence
/// generated so far, including the prompt) for the last
/// `lookup_len` tokens of `recent` and returns the slice that
/// follows the match.
///
/// Returns `None` if no match is found, or the match position is at
/// the very end of `haystack` (no following tokens).
///
/// Tunables:
///   `lookup_len`: n-gram size. 2-3 is the sweet spot for
///     natural-language echo — longer = fewer matches but more
///     confident.
///   `k`: max draft length. Larger = bigger payoff on accept but
///     more wasted compute on reject. 5-10 is typical.
pub fn lookup_draft(
    haystack: &[u32],
    recent: &[u32],
    lookup_len: usize,
    k: usize,
) -> Option<Vec<u32>> {
    if recent.len() < lookup_len || haystack.len() < lookup_len + 1 || k == 0 {
        return None;
    }
    let key = &recent[recent.len() - lookup_len..];
    // Scan from most-recent backwards — later matches tend to predict
    // the immediate future better than ancient ones.
    let max_start = haystack.len().saturating_sub(lookup_len);
    for start in (0..max_start).rev() {
        if haystack[start..start + lookup_len] == *key {
            let after = start + lookup_len;
            let take = k.min(haystack.len() - after);
            if take == 0 {
                continue;
            }
            return Some(haystack[after..after + take].to_vec());
        }
    }
    None
}

/// Multi-length prompt lookup: try the longest n-gram first (most context
/// matched → most confident draft), falling back to shorter ones for coverage.
/// Tries `len` from `max_len` down to `min_len.max(2)` and returns the first
/// (longest) match's continuation. min<2 is avoided: a 1-gram match drafts off a
/// single repeated token — low confidence, so a wasted multi-token verify on
/// reject (the verify forward costs the same whether or not drafts are accepted).
///
/// Raises the accept rate vs a single fixed `lookup_len`: on realistic echo
/// (paraphrased summaries, edited code) a 3-gram often misses where a 2-gram
/// still pins the continuation. Multiplies tokens/forward wherever PLD runs.
pub fn lookup_draft_best(
    haystack: &[u32],
    recent: &[u32],
    max_len: usize,
    k: usize,
) -> Option<Vec<u32>> {
    let min_len = 2;
    for len in (min_len..=max_len.max(min_len)).rev() {
        if let Some(d) = lookup_draft(haystack, recent, len, k) {
            return Some(d);
        }
    }
    None
}

/// Given main-model logits at K+1 positions (one per input token)
/// and the K drafts that occupied positions 1..K+1, return:
///   - the number of drafts the main model agrees with (greedy
///     argmax verification),
///   - the corrected/bonus token to append after the accepted drafts.
///
/// `t0_argmax` is the main model's choice for position 0 (the first
/// input, which was the token we already committed). If it differs
/// from the actual `tokens[0]` we passed in, we don't bother with
/// drafts at all — caller should fall back to single-token gen.
pub struct VerifyResult {
    pub accepted: usize, // 0..=drafts.len()
    pub bonus: u32,
}

pub fn verify_drafts(
    drafts: &[u32],
    logits_per_position: &[Vec<f32>],
) -> VerifyResult {
    // logits_per_position[i] predicts position i+1 (in the input).
    // drafts[i] occupies position i+1.
    // So drafts[i] is accepted iff argmax(logits_per_position[i]) == drafts[i].
    let mut accepted = 0usize;
    for (i, d) in drafts.iter().enumerate() {
        if i >= logits_per_position.len() {
            break;
        }
        let arg = argmax(&logits_per_position[i]);
        if arg == *d {
            accepted += 1;
        } else {
            return VerifyResult { accepted, bonus: arg };
        }
    }
    // All drafts accepted. Bonus = prediction at the last position.
    let bonus = if let Some(last) = logits_per_position.last() {
        argmax(last)
    } else {
        0
    };
    VerifyResult { accepted, bonus }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i as u32
}

/// Run one "classic" speculative-decode iteration with a small draft
/// model proposing K tokens that the main model verifies in a single
/// batched forward.
///
/// **Inputs**:
///   `main` / `draft`: both backends, KV both currently at position P.
///   `last_main_logit`: main's prediction for position P (sampled from
///     in the previous step or right after prefill).
///   `last_draft_logit`: draft's prediction for position P (same
///     idea — must already be precomputed).
///   `k`: number of draft tokens to propose. 5 is a reasonable default.
///
/// **Returns** a `SpecIterResult` with the committed tokens, accepted
/// count, and fresh `last_main_logit` / `last_draft_logit` for the
/// next iteration (because the helper does the catch-up forward of
/// the final emitted token on both backends).
pub struct SpecIterResult {
    pub committed: Vec<u32>,
    pub accepted: usize,
    pub draft_proposed: usize,
    pub next_main_logit: Vec<f32>,
    pub next_draft_logit: Vec<f32>,
}

pub fn spec_iter(
    main: &mut crate::backend::candle::backend::CandleCpuBackend,
    draft: &mut crate::backend::candle::backend::CandleCpuBackend,
    last_main_logit: &[f32],
    last_draft_logit: &[f32],
    k: usize,
) -> crate::error::Result<SpecIterResult> {
    let t = argmax(last_main_logit);

    // Draft proposes K tokens, forwarding each one so the draft's KV
    // stays in sync. Note: we do K forwards (not K-1) so the post-
    // proposal draft KV ends at the same position as the main KV will
    // after its batched verify. Simplifies the catch-up math below.
    let mut drafts: Vec<u32> = Vec::with_capacity(k);
    let mut d_logit_buf: Vec<f32> = last_draft_logit.to_vec();
    for _ in 0..k {
        let d = argmax(&d_logit_buf);
        drafts.push(d);
        d_logit_buf = draft.forward_logits(&[d])?;
    }
    // Draft KV now at P + K. (last_draft_logit_at_end = d_logit_buf,
    // unused; we'll recompute below after catch-up.)

    // Main verifies: batched forward of all drafts.
    let main_rows = main.forward_all_logits(&drafts)?;
    // Main KV now at P + K.

    // Verify D_0 vs T, D_i (i>=1) vs argmax(main_rows[i-1]).
    let mut accepted = 0usize;
    if drafts[0] == t {
        accepted = 1;
        for i in 1..drafts.len() {
            let am = argmax(&main_rows[i - 1]);
            if am == drafts[i] {
                accepted += 1;
            } else {
                break;
            }
        }
    }

    // Pick the corrected (on mismatch) or bonus (all accepted) token.
    let corrected = if accepted == 0 {
        t
    } else if accepted < drafts.len() {
        argmax(&main_rows[accepted - 1])
    } else {
        argmax(&main_rows[drafts.len() - 1])
    };

    // Both KVs are currently at P + K. We accepted `accepted` drafts
    // → keep P + accepted entries in both, drop the rest. Then forward
    // [corrected] through both to commit + get fresh logits.
    let target = main.position() - (drafts.len() - accepted);
    main.truncate_to(target)?;
    draft.truncate_to(target)?;
    let next_main_logit = main.forward_logits(&[corrected])?;
    let next_draft_logit = draft.forward_logits(&[corrected])?;

    let mut committed: Vec<u32> = Vec::with_capacity(accepted + 1);
    committed.extend_from_slice(&drafts[..accepted]);
    committed.push(corrected);

    Ok(SpecIterResult {
        committed,
        accepted,
        draft_proposed: drafts.len(),
        next_main_logit,
        next_draft_logit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_finds_recent_match() {
        // "the quick brown fox jumps over the quick" — searching for "the quick"
        // should find both occurrences; we return the LATER one (start=6),
        // so the draft is empty (nothing follows). Try a longer haystack.
        let h: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 10, 20, 70, 80];
        let recent = vec![60, 10, 20];
        let draft = lookup_draft(&h, &recent, 2, 5).unwrap();
        // Most recent "10, 20" is at index 6; what follows is 70, 80.
        assert_eq!(draft, vec![70, 80]);
    }

    #[test]
    fn lookup_best_falls_back_to_shorter_ngram() {
        // recent [5,2,3]: the 3-gram [5,2,3] never appears in the haystack, but the
        // 2-gram [2,3] does (idx 2-3, preceded by 1 not 5) → followed by 70,80.
        let h: Vec<u32> = vec![99, 1, 2, 3, 70, 80];
        let recent = vec![5, 2, 3];
        // Fixed 3-gram [5,2,3] never recurs → no draft.
        assert!(lookup_draft(&h, &recent, 3, 5).is_none());
        // Best: tries 3 (miss) then 2 ([2,3] at idx 2 → follows 70,80).
        assert_eq!(lookup_draft_best(&h, &recent, 3, 5).unwrap(), vec![70, 80]);
    }

    #[test]
    fn lookup_best_prefers_longest_match() {
        // Both a 3-gram and 2-gram match; the 3-gram (more context) wins.
        let h: Vec<u32> = vec![10, 20, 30, 40, 7, 20, 30, 99, 10, 20, 30, 55];
        let recent = vec![10, 20, 30];
        // 3-gram [10,20,30] last at idx 8 → follows 55; 2-gram would also match but
        // best returns the longest (3-gram) result.
        assert_eq!(lookup_draft_best(&h, &recent, 3, 5).unwrap(), vec![55]);
    }

    #[test]
    fn lookup_returns_none_when_no_match() {
        let h = vec![1, 2, 3, 4, 5];
        let recent = vec![99, 100];
        assert!(lookup_draft(&h, &recent, 2, 5).is_none());
    }

    #[test]
    fn verify_all_accept_returns_bonus() {
        // Drafts [5, 7]. Logits agree at positions 0 (→5) and 1 (→7);
        // logits at position 2 say bonus = 9.
        let logits = vec![
            mk_logits(16, 5),
            mk_logits(16, 7),
            mk_logits(16, 9),
        ];
        let r = verify_drafts(&[5, 7], &logits);
        assert_eq!(r.accepted, 2);
        assert_eq!(r.bonus, 9);
    }

    #[test]
    fn verify_first_mismatch_returns_corrected() {
        // Draft says [5, 99] but model wanted [5, 7].
        let logits = vec![mk_logits(8, 5), mk_logits(8, 7)];
        let r = verify_drafts(&[5, 99], &logits);
        assert_eq!(r.accepted, 1);
        assert_eq!(r.bonus, 7);
    }

    fn mk_logits(vocab: usize, peak: u32) -> Vec<f32> {
        let mut v = vec![0.0; vocab];
        v[peak as usize] = 10.0;
        v
    }
}
