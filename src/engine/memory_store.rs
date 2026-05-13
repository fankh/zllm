use crate::backend::traits::Tensor;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub vector: Tensor,
    pub metadata: MemoryMetadata,
    pub created_at: u64,
    pub access_count: u32,
    pub relevance_score: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryMetadata {
    pub source_request_id: String,
    pub tenant_id: String,
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

pub struct MemoryStore {
    entries: HashMap<String, MemoryEntry>,
    traces: Vec<InspectionTrace>,
    max_entries: usize,
    max_traces: usize,
    epoch: Instant,
}

impl MemoryStore {
    pub fn new(max_entries: usize, max_traces: usize) -> Self {
        Self {
            entries: HashMap::new(),
            traces: Vec::new(),
            max_entries,
            max_traces,
            epoch: Instant::now(),
        }
    }

    // --- Persist: Store memory entries ---

    pub fn store(&mut self, key: String, vector: Tensor, metadata: MemoryMetadata) {
        if self.entries.len() >= self.max_entries {
            self.evict_least_relevant();
        }

        let entry = MemoryEntry {
            key: key.clone(),
            vector,
            metadata,
            created_at: self.epoch.elapsed().as_secs(),
            access_count: 0,
            relevance_score: 1.0,
        };

        self.entries.insert(key, entry);
    }

    pub fn get(&mut self, key: &str) -> Option<&MemoryEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.relevance_score = (entry.relevance_score + 0.1).min(1.0);
        }
        self.entries.get(key)
    }

    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    // --- Query: Find relevant memories ---

    pub fn query_by_category(&self, category: &MemoryCategory) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|e| &e.metadata.category == category)
            .collect()
    }

    pub fn query_by_tenant(&self, tenant_id: &str) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|e| e.metadata.tenant_id == tenant_id)
            .collect()
    }

    pub fn query_by_tag(&self, tag: &str) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|e| e.metadata.tags.contains(&tag.to_string()))
            .collect()
    }

    pub fn query_by_similarity(&self, query_vector: &Tensor, top_k: usize) -> Vec<(&MemoryEntry, f32)> {
        let mut scored: Vec<(&MemoryEntry, f32)> = self
            .entries
            .values()
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
        tenant_id: &str,
        max_memories: usize,
        alpha: f32,
    ) -> Option<Tensor> {
        let tenant_memories: Vec<&MemoryEntry> = self
            .entries
            .values()
            .filter(|e| e.metadata.tenant_id == tenant_id)
            .collect();

        if tenant_memories.is_empty() {
            return None;
        }

        // Score by similarity to current hidden state
        let mut scored: Vec<(&MemoryEntry, f32)> = tenant_memories
            .into_iter()
            .map(|entry| {
                let sim = cosine_similarity(&entry.vector, query_vector);
                let recency_boost = 1.0 / (1.0 + (self.epoch.elapsed().as_secs() - entry.created_at) as f32 / 3600.0);
                let final_score = sim * 0.7 + entry.relevance_score * 0.2 + recency_boost * 0.1;
                (entry, final_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_memories);

        if scored.is_empty() {
            return None;
        }

        // Weighted average of top-k memory vectors
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

    /// Category-aware injection: for each (category, alpha, max) tuple, build a
    /// per-category injection vector and sum them into a single tensor.
    ///
    /// Used by goal/task/status memory injection. The existing
    /// `build_injection_vector` is left unchanged for backward compatibility.
    pub fn build_injection_vector_by_categories(
        &self,
        query_vector: &Tensor,
        tenant_id: &str,
        weights: &[(MemoryCategory, f32, usize)],
    ) -> Option<Tensor> {
        let mut result: Option<Vec<f32>> = None;

        for (category, alpha, max_per_category) in weights {
            let cat_memories: Vec<&MemoryEntry> = self
                .entries
                .values()
                .filter(|e| e.metadata.tenant_id == tenant_id && &e.metadata.category == category)
                .collect();

            if cat_memories.is_empty() {
                continue;
            }

            let mut scored: Vec<(&MemoryEntry, f32)> = cat_memories
                .into_iter()
                .map(|entry| {
                    let sim = cosine_similarity(&entry.vector, query_vector);
                    let recency_boost = 1.0
                        / (1.0
                            + (self.epoch.elapsed().as_secs() - entry.created_at) as f32 / 3600.0);
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

    fn evict_least_relevant(&mut self) {
        if let Some(key) = self
            .entries
            .iter()
            .min_by(|a, b| {
                a.1.relevance_score
                    .partial_cmp(&b.1.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&key);
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
            tenant_id: "local".into(),
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
            .build_injection_vector(&query, "local", 8, 1.0)
            .expect("expected injection");

        // Build expected: result with only the real entry.
        let mut s_real_only = MemoryStore::new(16, 4);
        s_real_only.store("real".into(), vec![1.0; 8], meta(MemoryCategory::Context));
        let expected = s_real_only
            .build_injection_vector(&query, "local", 8, 1.0)
            .expect("expected injection");

        // Bit-for-bit equality after the dilution guard.
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
                "local",
                &[(MemoryCategory::Goal, 0.5, 1), (MemoryCategory::Context, 0.1, 1)],
            )
            .expect("expected vector");
        assert_eq!(out.len(), 4);
        // Goal contributes 0.5*1.0 and Context 0.1*0.5 → 0.55 per element.
        for v in out {
            assert!((v - 0.55).abs() < 1e-4, "expected 0.55, got {v}");
        }
    }
}
