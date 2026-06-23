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

// ---------------------------------------------------------------------------
// Tree drafting (SpecInfer/Medusa-style token tree)
//
// A linear draft bets the whole row budget on ONE continuation: it accepts only
// up to the first token where the model disagrees with the top-1 n-gram guess.
// A token TREE spends the same budget across several branches — each node's
// children are the top-`branch` n-gram continuations — so the model's true
// continuation is recovered whenever it sits in the top-B at each step, not just
// the top-1. Verified in ONE forward with a tree-attention mask (each node
// attends only its ancestor path). Raises tokens/forward → shifts the spec-decode
// win threshold above the verify matvec floor. This module builds + scores the
// tree; the GPU tree-mask SDPA is the separate kernel a win needs.
// ---------------------------------------------------------------------------

/// One node of a flattened speculative token tree. `node[0]` is the root (the
/// already-committed token); every other node is a candidate continuation.
/// `parent` is the attention ancestry (a node attends 0..=pos AND its ancestor
/// chain) and `depth` is the position offset from the root.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeNode {
    pub token: u32,
    pub parent: usize,
    pub depth: usize,
}

/// A flattened draft tree ready for tree-mask verification.
#[derive(Clone, Debug, Default)]
pub struct DraftTree {
    pub nodes: Vec<TreeNode>,
}

impl DraftTree {
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Tokens on the path root→node, excluding the root, in depth order.
    fn path_after_root(&self, mut idx: usize) -> Vec<u32> {
        let mut t = Vec::new();
        while idx != 0 {
            t.push(self.nodes[idx].token);
            idx = self.nodes[idx].parent;
        }
        t.reverse();
        t
    }
}

/// Distinct tokens that followed `key` in `haystack`, ranked by match frequency
/// (then recency), capped at `branch`. Returns `(token, count)`.
fn ngram_continuations(haystack: &[u32], key: &[u32], branch: usize) -> Vec<(u32, usize)> {
    let l = key.len();
    if l == 0 || haystack.len() <= l || branch == 0 {
        return Vec::new();
    }
    let mut count: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut last_seen: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for start in 0..=haystack.len() - l - 1 {
        if haystack[start..start + l] == *key {
            let nxt = haystack[start + l];
            *count.entry(nxt).or_default() += 1;
            last_seen.insert(nxt, start); // later overwrites → most-recent wins
        }
    }
    let mut v: Vec<(u32, usize)> = count.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(last_seen[&b.0].cmp(&last_seen[&a.0])));
    v.truncate(branch);
    v
}

/// Best n-gram continuations for `recent`: try the longest key first (most
/// context = most confident), falling back to shorter for coverage. Mirrors
/// `lookup_draft_best`'s multi-length policy but returns the top-`branch` set.
fn best_continuations(haystack: &[u32], recent: &[u32], max_len: usize, branch: usize) -> Vec<(u32, usize)> {
    let min_len = 2;
    for len in (min_len..=max_len.max(min_len)).rev() {
        if recent.len() < len {
            continue;
        }
        let key = &recent[recent.len() - len..];
        let c = ngram_continuations(haystack, key, branch);
        if !c.is_empty() {
            return c;
        }
    }
    Vec::new()
}

/// Build a draft tree (≤ `max_nodes` nodes) rooted at the last `recent` token.
/// Expansion is best-first by cumulative path probability (product of per-step
/// frequency shares), so the node budget concentrates on the most probable
/// continuations — a deep "spine" along the confident path plus shallow branches
/// where the n-gram is least sure. `branch` children per node, n-gram length up
/// to `max_len`.
pub fn lookup_tree(haystack: &[u32], recent: &[u32], max_len: usize, branch: usize, max_nodes: usize) -> DraftTree {
    let Some(&root_tok) = recent.last() else { return DraftTree::default() };
    let mut tree = DraftTree { nodes: vec![TreeNode { token: root_tok, parent: 0, depth: 0 }] };
    if max_nodes <= 1 || branch == 0 {
        return tree;
    }
    // Frontier of expandable nodes with their cumulative path score.
    let mut frontier: Vec<(f64, usize)> = vec![(1.0, 0)];
    while tree.nodes.len() < max_nodes && !frontier.is_empty() {
        // Expand the highest-scoring frontier node (best-first / probable-tree).
        let bi = (0..frontier.len())
            .max_by(|&a, &b| frontier[a].0.partial_cmp(&frontier[b].0).unwrap())
            .unwrap();
        let (score, idx) = frontier.remove(bi);
        let mut eff: Vec<u32> = recent.to_vec();
        eff.extend(tree.path_after_root(idx));
        let conts = best_continuations(haystack, &eff, max_len, branch);
        let total: usize = conts.iter().map(|c| c.1).sum::<usize>().max(1);
        for (tok, cnt) in conts {
            if tree.nodes.len() >= max_nodes {
                break;
            }
            let child = tree.nodes.len();
            tree.nodes.push(TreeNode { token: tok, parent: idx, depth: tree.nodes[idx].depth + 1 });
            frontier.push((score * (cnt as f64 / total as f64), child));
        }
    }
    tree
}

/// Greedy tree acceptance: how many tokens of the model's true continuation
/// `target` (target[0] = the token after the root) follow a connected root path
/// in the tree. This is exactly what a greedy tree-verify commits minus the
/// bonus: the model's argmax chain is followed as deep as the tree contains it.
pub fn tree_accept(tree: &DraftTree, target: &[u32]) -> usize {
    if tree.nodes.is_empty() {
        return 0;
    }
    let mut cur = 0usize; // root
    let mut acc = 0usize;
    'next: for &want in target {
        for (i, n) in tree.nodes.iter().enumerate() {
            if i != 0 && n.parent == cur && n.token == want {
                cur = i;
                acc += 1;
                continue 'next;
            }
        }
        break;
    }
    acc
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

    // --- tree drafting ---

    #[test]
    fn ngram_continuations_ranked_by_frequency_then_recency() {
        // [1,2] is followed by 3 (twice) and 4 (once). 3 ranks first (frequency).
        let h = vec![1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 4, 9];
        let c = ngram_continuations(&h, &[1, 2], 4);
        assert_eq!(c, vec![(3, 2), (4, 1)]);
    }

    #[test]
    fn tree_recovers_what_linear_misses() {
        // [1,2] is followed by 4,5 (older) and by 3,6 (most recent). A LINEAR draft
        // bets on the most-recent continuation (3,…) and accepts 0 of the true
        // continuation [4,5]. The TREE keeps both 3 and 4 as depth-1 children, so it
        // follows 4 then 5 → accepts 2.
        let h = vec![1, 2, 4, 5, 8, 1, 2, 3, 6, 9];
        let recent = [1u32, 2];
        let target = [4u32, 5]; // the model's true continuation after the root (2)

        let linear = lookup_draft_best(&h, &recent, 2, 3).unwrap();
        assert_eq!(linear, vec![3, 6, 9]); // most-recent match
        let linear_accept = linear.iter().zip(&target).take_while(|(a, b)| a == b).count();
        assert_eq!(linear_accept, 0);

        let tree = lookup_tree(&h, &recent, 2, 2, 8);
        assert_eq!(tree.nodes[0].token, 2); // root = the committed token
        assert_eq!(tree_accept(&tree, &target), 2); // tree follows 4→5

        assert!(tree_accept(&tree, &target) > linear_accept, "tree must beat linear here");
    }

    #[test]
    fn tree_accept_follows_spine_and_stops_off_tree() {
        // Continuation 1→2→3→4: a deterministic spine. recent=[7,1] so the root
        // context ([7,1]) is present in the haystack and the spine expands.
        let h = vec![1, 2, 3, 4, 7, 1, 2, 3, 4, 8, 1];
        let tree = lookup_tree(&h, &[7u32, 1], 2, 2, 8); // root token = 1
        assert_eq!(tree_accept(&tree, &[2, 3, 4]), 3); // whole spine
        assert_eq!(tree_accept(&tree, &[2, 9]), 1);    // diverges after 2
        assert_eq!(tree_accept(&tree, &[5]), 0);       // not in tree at all
    }

    #[test]
    fn tree_respects_node_budget() {
        let h = vec![1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 3, 6, 1, 3, 7, 1];
        let tree = lookup_tree(&h, &[0u32, 1], 2, 3, 5);
        assert!(tree.len() <= 5, "tree exceeded node budget: {}", tree.len());
        assert_eq!(tree.nodes[0].depth, 0);
        // every non-root node points at a shallower parent
        for (i, n) in tree.nodes.iter().enumerate().skip(1) {
            assert!(n.parent < i && n.depth == tree.nodes[n.parent].depth + 1);
        }
    }
}
