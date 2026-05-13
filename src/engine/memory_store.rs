use crate::backend::traits::Tensor;
use crate::metrics;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub vector: Tensor,
    pub metadata: MemoryMetadata,
    pub created_at: u64,
    pub access_count: u32,
    pub relevance_score: f32,
    pub expires_at: Option<u64>,
    pub pinned: bool,
    pub byte_size: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryMetadata {
    pub source_request_id: String,
    pub layer_captured: usize,
    pub category: MemoryCategory,
    pub tags: Vec<String>,
    pub text_summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    Finding,        // Security finding, vulnerability detected
    Context,        // Background context about the project/codebase
    Pattern,        // Recurring pattern the model has seen
    Correction,     // User correction that should persist
    Knowledge,      // External knowledge injected (CVE, threat intel)
    Goal,           // Persistent agent goal (one current, optionally many archived)
    Task,           // Discrete task under a goal (active/done/blocked encoded in tags)
    Status,         // Rolling progress snapshot for the current goal
}

impl MemoryCategory {
    fn as_str(&self) -> &'static str {
        match self {
            MemoryCategory::Finding => "finding",
            MemoryCategory::Context => "context",
            MemoryCategory::Pattern => "pattern",
            MemoryCategory::Correction => "correction",
            MemoryCategory::Knowledge => "knowledge",
            MemoryCategory::Goal => "goal",
            MemoryCategory::Task => "task",
            MemoryCategory::Status => "status",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StoreOptions {
    /// Pinned entries are never evicted. Only the GoalManager and other
    /// privileged code paths should set this; hook-driven captures must leave
    /// it false.
    pub pinned: bool,
    /// Optional TTL — entries with `expires_at <= now` are skipped in queries
    /// and dropped first when the store is under pressure.
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StoreError {
    /// The entry's byte size exceeds the per-category budget on its own.
    /// No amount of eviction can make room — caller should not retry.
    Oversize,
    /// The store could not free enough space (everything else is pinned).
    Full,
}

#[derive(Debug, Clone)]
pub struct InspectionTrace {
    pub request_id: String,
    pub layers: Vec<LayerSnapshot>,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct LayerSnapshot {
    pub layer_idx: usize,
    pub loop_idx: usize,
    pub hidden_state_norm: f32,
    pub hidden_state_hash: u64,
    pub top_activations: Vec<(usize, f32)>,
    pub interpretation: String,
}

const DEFAULT_TOTAL_BUDGET_BYTES: usize = 256 * 1024 * 1024; // 256 MB

fn default_category_budgets() -> HashMap<MemoryCategory, usize> {
    use MemoryCategory::*;
    let mb = 1024 * 1024;
    [
        (Goal, 16 * mb),
        (Task, 32 * mb),
        (Status, 8 * mb),
        (Context, 128 * mb),
        (Finding, 16 * mb),
        (Pattern, 16 * mb),
        (Correction, 16 * mb),
        (Knowledge, 16 * mb),
    ]
    .into_iter()
    .collect()
}

pub struct MemoryStore {
    entries: HashMap<String, MemoryEntry>,
    traces: Vec<InspectionTrace>,
    max_entries: usize,
    max_traces: usize,
    epoch: Instant,

    // Byte accounting + budgets
    bytes_used: usize,
    byte_budget: usize,
    category_budgets: HashMap<MemoryCategory, usize>,
    category_bytes: HashMap<MemoryCategory, usize>,

    // Indexes (key sets) for fast filtered queries.
    by_category: HashMap<MemoryCategory, HashSet<String>>,
    by_tag: HashMap<String, HashSet<String>>,
}

impl MemoryStore {
    pub fn new(max_entries: usize, max_traces: usize) -> Self {
        Self::with_budget(
            max_entries,
            max_traces,
            DEFAULT_TOTAL_BUDGET_BYTES,
            default_category_budgets(),
        )
    }

    pub fn with_budget(
        max_entries: usize,
        max_traces: usize,
        byte_budget: usize,
        category_budgets: HashMap<MemoryCategory, usize>,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            traces: Vec::new(),
            max_entries,
            max_traces,
            epoch: Instant::now(),
            bytes_used: 0,
            byte_budget,
            category_budgets,
            category_bytes: HashMap::new(),
            by_category: HashMap::new(),
            by_tag: HashMap::new(),
        }
    }

    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    pub fn byte_budget(&self) -> usize {
        self.byte_budget
    }

    fn now(&self) -> u64 {
        self.epoch.elapsed().as_secs()
    }

    fn is_expired(entry: &MemoryEntry, now: u64) -> bool {
        matches!(entry.expires_at, Some(t) if t <= now)
    }

    // --- Persist: Store memory entries ---

    /// Legacy entry point. Stores with default options (not pinned, no TTL).
    /// Returns silently on failure for back-compat — callers wanting the
    /// error should use `store_with_options`.
    pub fn store(&mut self, key: String, vector: Tensor, metadata: MemoryMetadata) {
        let _ = self.store_with_options(key, vector, metadata, StoreOptions::default());
    }

    pub fn store_with_options(
        &mut self,
        key: String,
        vector: Tensor,
        metadata: MemoryMetadata,
        options: StoreOptions,
    ) -> Result<(), StoreError> {
        let byte_size = vector.len() * std::mem::size_of::<f32>();
        let cat = metadata.category.clone();
        let cat_budget = *self
            .category_budgets
            .get(&cat)
            .unwrap_or(&self.byte_budget);

        // Oversize-on-its-own — no eviction can save us.
        if byte_size > cat_budget || byte_size > self.byte_budget {
            metrics::memory_evictions_total()
                .with_label_values(&["oversize"])
                .inc();
            return Err(StoreError::Oversize);
        }

        // Replace-in-place: removing the old key first frees its bytes/indexes
        // before we account for the new entry.
        if self.entries.contains_key(&key) {
            self.remove_internal(&key);
        }

        let now = self.now();
        loop {
            let cat_used = *self.category_bytes.get(&cat).unwrap_or(&0);
            let fits_total = self.bytes_used + byte_size <= self.byte_budget;
            let fits_category = cat_used + byte_size <= cat_budget;
            let fits_count = self.entries.len() < self.max_entries;
            if fits_total && fits_category && fits_count {
                break;
            }

            // Lazy expiry: drop one expired entry to make room.
            if self.drop_one_expired(now) {
                continue;
            }

            // Score-based eviction, preferring the same category.
            if !self.evict_one(Some(&cat)) {
                metrics::memory_evictions_total()
                    .with_label_values(&["unfree"])
                    .inc();
                return Err(StoreError::Full);
            }
        }

        let expires_at = options.ttl_seconds.map(|t| now + t);
        let entry = MemoryEntry {
            key: key.clone(),
            vector,
            metadata,
            created_at: now,
            access_count: 0,
            relevance_score: 1.0,
            expires_at,
            pinned: options.pinned,
            byte_size,
        };

        self.by_category
            .entry(cat.clone())
            .or_default()
            .insert(key.clone());
        for tag in &entry.metadata.tags {
            self.by_tag
                .entry(tag.clone())
                .or_default()
                .insert(key.clone());
        }
        self.bytes_used += byte_size;
        *self.category_bytes.entry(cat).or_insert(0) += byte_size;
        self.entries.insert(key, entry);

        self.publish_gauges();
        Ok(())
    }

    pub fn get(&mut self, key: &str) -> Option<&MemoryEntry> {
        let now = self.now();
        if let Some(entry) = self.entries.get_mut(key) {
            if Self::is_expired(entry, now) {
                return None;
            }
            entry.access_count += 1;
            entry.relevance_score = (entry.relevance_score + 0.1).min(1.0);
        }
        self.entries.get(key).filter(|e| !Self::is_expired(e, now))
    }

    pub fn remove(&mut self, key: &str) -> bool {
        let removed = self.remove_internal(key);
        if removed {
            self.publish_gauges();
        }
        removed
    }

    fn remove_internal(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.bytes_used = self.bytes_used.saturating_sub(entry.byte_size);
            if let Some(b) = self.category_bytes.get_mut(&entry.metadata.category) {
                *b = b.saturating_sub(entry.byte_size);
            }
            if let Some(set) = self.by_category.get_mut(&entry.metadata.category) {
                set.remove(key);
            }
            for tag in &entry.metadata.tags {
                if let Some(set) = self.by_tag.get_mut(tag) {
                    set.remove(key);
                }
            }
            true
        } else {
            false
        }
    }

    // --- Query: Find relevant memories (expired entries are skipped) ---

    pub fn query_by_category(&self, category: &MemoryCategory) -> Vec<&MemoryEntry> {
        let now = self.now();
        let Some(keys) = self.by_category.get(category) else {
            return Vec::new();
        };
        keys.iter()
            .filter_map(|k| self.entries.get(k))
            .filter(|e| !Self::is_expired(e, now))
            .collect()
    }

    pub fn query_by_tag(&self, tag: &str) -> Vec<&MemoryEntry> {
        let now = self.now();
        let Some(keys) = self.by_tag.get(tag) else {
            return Vec::new();
        };
        keys.iter()
            .filter_map(|k| self.entries.get(k))
            .filter(|e| !Self::is_expired(e, now))
            .collect()
    }

    pub fn query_by_similarity(
        &self,
        query_vector: &Tensor,
        top_k: usize,
    ) -> Vec<(&MemoryEntry, f32)> {
        let now = self.now();
        let mut scored: Vec<(&MemoryEntry, f32)> = self
            .entries
            .values()
            .filter(|e| !Self::is_expired(e, now))
            .map(|entry| {
                let sim = cosine_similarity(&entry.vector, query_vector);
                (entry, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    // --- Inject: Build injection vector from relevant memories ---

    pub fn build_injection_vector(
        &self,
        query_vector: &Tensor,
        max_memories: usize,
        alpha: f32,
    ) -> Option<Tensor> {
        let now = self.now();
        let live_memories: Vec<&MemoryEntry> = self
            .entries
            .values()
            .filter(|e| !Self::is_expired(e, now))
            .collect();

        if live_memories.is_empty() {
            return None;
        }

        let mut scored: Vec<(&MemoryEntry, f32)> = live_memories
            .into_iter()
            .map(|entry| {
                let sim = cosine_similarity(&entry.vector, query_vector);
                let recency_boost =
                    1.0 / (1.0 + (now.saturating_sub(entry.created_at)) as f32 / 3600.0);
                let final_score = sim * 0.7 + entry.relevance_score * 0.2 + recency_boost * 0.1;
                (entry, final_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_memories);

        if scored.is_empty() {
            return None;
        }

        let d = scored[0].0.vector.len();
        let mut result = vec![0.0f32; d];
        let mut total_weight = 0.0f32;

        for (entry, score) in &scored {
            if entry.vector.len() == d && vector_norm(&entry.vector) > 1e-6 {
                for (r, v) in result.iter_mut().zip(entry.vector.iter()) {
                    *r += score * v;
                }
                total_weight += score;
            }
        }

        if total_weight > 0.0 {
            for r in result.iter_mut() {
                *r = (*r / total_weight) * alpha;
            }
            Some(result)
        } else {
            None
        }
    }

    /// Category-aware injection. Walks each (category, alpha, max) entry,
    /// builds a per-category injection vector via the per-category index
    /// (no full-table scan), and sums them.
    pub fn build_injection_vector_by_categories(
        &self,
        query_vector: &Tensor,
        weights: &[(MemoryCategory, f32, usize)],
    ) -> Option<Tensor> {
        let now = self.now();
        let mut result: Option<Vec<f32>> = None;

        for (category, alpha, max_per_category) in weights {
            let Some(keys) = self.by_category.get(category) else {
                continue;
            };
            let cat_memories: Vec<&MemoryEntry> = keys
                .iter()
                .filter_map(|k| self.entries.get(k))
                .filter(|e| !Self::is_expired(e, now))
                .collect();

            if cat_memories.is_empty() {
                continue;
            }

            let mut scored: Vec<(&MemoryEntry, f32)> = cat_memories
                .into_iter()
                .map(|entry| {
                    let sim = cosine_similarity(&entry.vector, query_vector);
                    let recency_boost =
                        1.0 / (1.0 + (now.saturating_sub(entry.created_at)) as f32 / 3600.0);
                    let final_score =
                        sim * 0.7 + entry.relevance_score * 0.2 + recency_boost * 0.1;
                    (entry, final_score)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(*max_per_category);

            let d = scored[0].0.vector.len();
            let mut sum = vec![0.0f32; d];
            let mut total_weight = 0.0f32;

            for (entry, score) in &scored {
                if entry.vector.len() == d && vector_norm(&entry.vector) > 1e-6 {
                    for (r, v) in sum.iter_mut().zip(entry.vector.iter()) {
                        *r += score * v;
                    }
                    total_weight += score;
                }
            }

            if total_weight > 0.0 {
                for r in sum.iter_mut() {
                    *r = (*r / total_weight) * alpha;
                }
                match result.as_mut() {
                    Some(acc) if acc.len() == sum.len() => {
                        for (a, s) in acc.iter_mut().zip(sum.iter()) {
                            *a += *s;
                        }
                    }
                    Some(_) => {
                        // Mismatched dimensions — skip; shouldn't happen in practice
                    }
                    None => {
                        result = Some(sum);
                    }
                }
            }
        }

        result
    }

    // --- Inspect: Record and retrieve traces ---

    pub fn record_trace(&mut self, trace: InspectionTrace) {
        if self.traces.len() >= self.max_traces {
            self.traces.remove(0);
        }
        self.traces.push(trace);
    }

    pub fn get_traces(&self, last_n: usize) -> &[InspectionTrace] {
        let start = self.traces.len().saturating_sub(last_n);
        &self.traces[start..]
    }

    pub fn get_trace_by_request(&self, request_id: &str) -> Option<&InspectionTrace> {
        self.traces.iter().rev().find(|t| t.request_id == request_id)
    }

    // --- Maintenance ---

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    pub fn decay_relevance(&mut self, factor: f32) {
        for entry in self.entries.values_mut() {
            entry.relevance_score *= factor;
        }
    }

    /// Drop one expired entry if any exists. Returns true if one was dropped.
    fn drop_one_expired(&mut self, now: u64) -> bool {
        let victim = self
            .entries
            .iter()
            .find(|(_, e)| Self::is_expired(e, now))
            .map(|(k, _)| k.clone());
        if let Some(key) = victim {
            self.remove_internal(&key);
            metrics::memory_evictions_total()
                .with_label_values(&["expired"])
                .inc();
            metrics::memory_expired_drops().inc();
            true
        } else {
            false
        }
    }

    /// Evict one non-pinned entry, preferring the supplied category if any.
    /// Returns false only if every remaining entry is pinned.
    fn evict_one(&mut self, prefer_category: Option<&MemoryCategory>) -> bool {
        // First pass: prefer the same category.
        if let Some(cat) = prefer_category {
            if let Some(key) = self.lowest_score_key(Some(cat)) {
                self.remove_internal(&key);
                metrics::memory_evictions_total()
                    .with_label_values(&["budget"])
                    .inc();
                return true;
            }
        }
        // Second pass: any non-pinned entry.
        if let Some(key) = self.lowest_score_key(None) {
            self.remove_internal(&key);
            metrics::memory_evictions_total()
                .with_label_values(&["budget"])
                .inc();
            return true;
        }
        false
    }

    fn lowest_score_key(&self, only_category: Option<&MemoryCategory>) -> Option<String> {
        self.entries
            .iter()
            .filter(|(_, e)| !e.pinned)
            .filter(|(_, e)| match only_category {
                Some(cat) => &e.metadata.category == cat,
                None => true,
            })
            .min_by(|a, b| {
                a.1.relevance_score
                    .partial_cmp(&b.1.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
    }

    fn publish_gauges(&self) {
        metrics::memory_bytes_used().set(self.bytes_used as i64);
        // Per-category gauges
        for (cat, set) in &self.by_category {
            metrics::memory_entries()
                .with_label_values(&[cat.as_str()])
                .set(set.len() as i64);
        }
        for (cat, bytes) in &self.category_bytes {
            metrics::memory_bytes_by_category()
                .with_label_values(&[cat.as_str()])
                .set(*bytes as i64);
        }
    }
}

// --- Inspection helpers ---

impl LayerSnapshot {
    pub fn from_hidden_state(
        layer_idx: usize,
        loop_idx: usize,
        hidden_state: &Tensor,
    ) -> Self {
        let norm: f32 = hidden_state.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Find top-k activations (highest absolute values)
        let mut indexed: Vec<(usize, f32)> = hidden_state
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(10);

        // Simple hash for change detection
        let hash = hidden_state
            .iter()
            .take(64)
            .fold(0u64, |acc, &v| acc.wrapping_add((v * 1000.0) as u64));

        Self {
            layer_idx,
            loop_idx,
            hidden_state_norm: norm,
            hidden_state_hash: hash,
            top_activations: indexed,
            interpretation: String::new(),
        }
    }
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn vector_norm(v: &Tensor) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(category: MemoryCategory) -> MemoryMetadata {
        MemoryMetadata {
            source_request_id: "test".into(),
            layer_captured: 0,
            category,
            tags: vec![],
            text_summary: String::new(),
        }
    }

    #[test]
    fn zero_norm_entry_does_not_dilute_injection() {
        let mut s = MemoryStore::new(16, 4);
        s.store("real".into(), vec![1.0; 8], meta(MemoryCategory::Context));
        s.store("zero".into(), vec![0.0; 8], meta(MemoryCategory::Goal));

        let query = vec![1.0; 8];
        let with_both = s
            .build_injection_vector(&query, 8, 1.0)
            .expect("expected injection");

        let mut s_real_only = MemoryStore::new(16, 4);
        s_real_only.store("real".into(), vec![1.0; 8], meta(MemoryCategory::Context));
        let expected = s_real_only
            .build_injection_vector(&query, 8, 1.0)
            .expect("expected injection");

        assert_eq!(with_both, expected);
    }

    #[test]
    fn category_aware_injection_routes_by_category() {
        let mut s = MemoryStore::new(16, 4);
        s.store("g".into(), vec![1.0; 4], meta(MemoryCategory::Goal));
        s.store("c".into(), vec![0.5; 4], meta(MemoryCategory::Context));

        let q = vec![1.0; 4];
        let out = s
            .build_injection_vector_by_categories(
                &q,
                &[(MemoryCategory::Goal, 0.5, 1), (MemoryCategory::Context, 0.1, 1)],
            )
            .expect("expected vector");
        assert_eq!(out.len(), 4);
        for v in out {
            assert!((v - 0.55).abs() < 1e-4, "expected 0.55, got {v}");
        }
    }

    // --- v0.2 tests ---

    fn tiny_budget_store() -> MemoryStore {
        // Tiny budgets to make pressure easy in tests.
        let mut cat = HashMap::new();
        cat.insert(MemoryCategory::Goal, 128);
        cat.insert(MemoryCategory::Task, 128);
        cat.insert(MemoryCategory::Status, 128);
        cat.insert(MemoryCategory::Context, 256);
        cat.insert(MemoryCategory::Finding, 128);
        cat.insert(MemoryCategory::Pattern, 128);
        cat.insert(MemoryCategory::Correction, 128);
        cat.insert(MemoryCategory::Knowledge, 128);
        MemoryStore::with_budget(64, 4, 1024, cat)
    }

    #[test]
    fn byte_budget_evicts_when_full() {
        let mut s = tiny_budget_store();
        // 16 floats = 64 bytes per Context entry. Context cap = 256 bytes = 4 entries.
        for i in 0..6 {
            s.store(format!("c{i}"), vec![0.5; 16], meta(MemoryCategory::Context));
        }
        let count = s.by_category.get(&MemoryCategory::Context).map(|s| s.len()).unwrap_or(0);
        assert!(count <= 4, "category cap should keep <=4 entries, got {count}");
        assert!(s.bytes_used <= s.byte_budget);
    }

    #[test]
    fn oversize_entry_refused() {
        let mut s = tiny_budget_store();
        // Goal cap = 128 bytes. 64 floats = 256 bytes.
        let err = s.store_with_options(
            "big".into(),
            vec![1.0; 64],
            meta(MemoryCategory::Goal),
            StoreOptions::default(),
        );
        assert_eq!(err, Err(StoreError::Oversize));
        assert_eq!(s.entry_count(), 0);
    }

    #[test]
    fn pinned_entry_survives_pressure() {
        let mut s = tiny_budget_store();
        s.store_with_options(
            "goal".into(),
            vec![0.5; 8],
            meta(MemoryCategory::Goal),
            StoreOptions { pinned: true, ttl_seconds: None },
        )
        .expect("pinned store ok");
        // Fill goal category with unpinned entries (under cap)
        for i in 0..10 {
            let _ = s.store_with_options(
                format!("g{i}"),
                vec![0.5; 8],
                meta(MemoryCategory::Goal),
                StoreOptions::default(),
            );
        }
        assert!(s.entries.contains_key("goal"), "pinned goal must survive");
    }

    #[test]
    fn ttl_skipped_in_queries_after_expiry() {
        let mut s = tiny_budget_store();
        // ttl = 0 → already expired by next tick.
        s.store_with_options(
            "transient".into(),
            vec![0.5; 4],
            meta(MemoryCategory::Status),
            StoreOptions { pinned: false, ttl_seconds: Some(0) },
        )
        .expect("store ok");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let entries = s.query_by_category(&MemoryCategory::Status);
        assert!(entries.is_empty(), "expired entries must not appear in queries");
    }

    #[test]
    fn category_index_isolates_queries() {
        let mut s = tiny_budget_store();
        s.store("g".into(), vec![0.1; 4], meta(MemoryCategory::Goal));
        s.store("c".into(), vec![0.2; 4], meta(MemoryCategory::Context));
        let goals = s.query_by_category(&MemoryCategory::Goal);
        let ctx = s.query_by_category(&MemoryCategory::Context);
        assert_eq!(goals.len(), 1);
        assert_eq!(ctx.len(), 1);
        assert_eq!(goals[0].metadata.category, MemoryCategory::Goal);
        assert_eq!(ctx[0].metadata.category, MemoryCategory::Context);
    }

    #[test]
    fn tag_index_round_trips() {
        let mut s = tiny_budget_store();
        let mut m = meta(MemoryCategory::Task);
        m.tags = vec!["goal:abc".into(), "active".into()];
        s.store("t1".into(), vec![0.1; 4], m);
        let by_goal = s.query_by_tag("goal:abc");
        let by_status = s.query_by_tag("active");
        assert_eq!(by_goal.len(), 1);
        assert_eq!(by_status.len(), 1);
        assert_eq!(by_goal[0].key, "t1");
    }
}
