use crate::engine::memory_store::{
    MemoryCategory, MemoryMetadata, MemoryStore, StoreOptions,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

const D_MODEL_PLACEHOLDER: usize = 4096;
const TAG_CURRENT: &str = "current";

/// Status entries decay after one hour — they're rolling snapshots, not
/// durable state. Goals and active tasks are pinned and have no TTL.
const STATUS_TTL_SECONDS: u64 = 3600;

/// Pinned store for Goals and active Tasks — never evictable until the
/// pin is cleared (e.g. `update_task` to `Done`).
fn pinned_no_ttl() -> StoreOptions {
    StoreOptions { pinned: true, ttl_seconds: None }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Active,
    Done,
    Blocked,
}

impl TaskStatus {
    fn tag(&self) -> &'static str {
        match self {
            TaskStatus::Active => "active",
            TaskStatus::Done => "done",
            TaskStatus::Blocked => "blocked",
        }
    }

    fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "active" => Some(TaskStatus::Active),
            "done" => Some(TaskStatus::Done),
            "blocked" => Some(TaskStatus::Blocked),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub goal_id: String,
    pub text: String,
    pub is_current: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub goal_id: String,
    pub text: String,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusEntry {
    pub text: String,
    pub goal_id: String,
}

#[derive(Debug, Clone, Default)]
pub struct GoalState {
    pub current_goal: Option<Goal>,
    pub active_tasks: Vec<Task>,
    pub latest_status: Option<StatusEntry>,
}

/// On-disk snapshot of the goal state. Lists ALL goals (not just current)
/// and ALL tasks (active + done + blocked), so a restart reproduces the
/// full picture, not just what's currently "live".
#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedState {
    goals: Vec<Goal>,
    tasks: Vec<Task>,
    #[serde(default)]
    status: Option<StatusEntry>,
}

pub struct GoalManager {
    store: Arc<RwLock<MemoryStore>>,
    d_model: usize,
    /// If set, the manager writes its state to this path after every
    /// mutation and rebuilds it from this path on startup. Atomic-write
    /// (write to .tmp + rename) so a crash mid-save can't corrupt the
    /// file. `None` disables persistence — useful for tests.
    save_path: Option<PathBuf>,
}

impl GoalManager {
    pub fn new(store: Arc<RwLock<MemoryStore>>) -> Self {
        Self {
            store,
            d_model: D_MODEL_PLACEHOLDER,
            save_path: None,
        }
    }

    pub fn with_save_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.save_path = Some(path.into());
        self
    }

    /// Restore goal/task/status entries from disk. Idempotent — safe to
    /// call on a manager that has never been saved (no-op if file missing).
    /// Logs and continues on parse errors (returns the same instance) so
    /// a corrupt file doesn't take down the server.
    pub fn load_from_disk(&self) {
        let Some(path) = &self.save_path else { return; };
        if !path.exists() { return; }
        let Ok(s) = std::fs::read_to_string(path) else {
            tracing::warn!("goal persistence: failed to read {}", path.display());
            return;
        };
        let Ok(state): Result<PersistedState, _> = serde_json::from_str(&s) else {
            tracing::warn!("goal persistence: failed to parse {} — starting fresh", path.display());
            return;
        };
        for g in state.goals { self.restore_goal(&g); }
        for t in state.tasks { self.restore_task(&t); }
        if let Some(st) = state.status { self.restore_status(&st); }
        tracing::info!("goal persistence: restored from {}", path.display());
    }

    fn save(&self) {
        let Some(path) = &self.save_path else { return; };
        let state = self.snapshot_for_save();
        let json = match serde_json::to_string_pretty(&state) {
            Ok(s) => s,
            Err(e) => { tracing::warn!("goal persistence: serialize failed: {e}"); return; }
        };
        // Atomic write: tmp file + rename. Cheap insurance against a
        // partial-write corruption if the process dies mid-write.
        let tmp = path.with_extension("json.tmp");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(&tmp, json) {
            tracing::warn!("goal persistence: tmp write failed: {e}");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            tracing::warn!("goal persistence: rename failed: {e}");
        }
    }

    fn snapshot_for_save(&self) -> PersistedState {
        let store = self.store.read().expect("memory store poisoned");
        let goals: Vec<Goal> = store
            .query_by_category(&MemoryCategory::Goal)
            .into_iter()
            .map(|e| Goal {
                goal_id: extract_id(&e.metadata.tags, "goal:").unwrap_or_default(),
                text: e.metadata.text_summary.clone(),
                is_current: e.metadata.tags.iter().any(|t| t == TAG_CURRENT),
            })
            .collect();
        let tasks: Vec<Task> = store
            .query_by_category(&MemoryCategory::Task)
            .into_iter()
            .map(|e| Task {
                task_id: extract_id(&e.metadata.tags, "task:").unwrap_or_default(),
                goal_id: extract_id(&e.metadata.tags, "goal:").unwrap_or_default(),
                text: e.metadata.text_summary.clone(),
                status: e
                    .metadata
                    .tags
                    .iter()
                    .find_map(|t| TaskStatus::from_tag(t))
                    .unwrap_or(TaskStatus::Active),
            })
            .collect();
        let status = store
            .query_by_category(&MemoryCategory::Status)
            .into_iter()
            .max_by_key(|e| e.created_at)
            .map(|e| StatusEntry {
                text: e.metadata.text_summary.clone(),
                goal_id: extract_id(&e.metadata.tags, "goal:").unwrap_or_default(),
            });
        PersistedState { goals, tasks, status }
    }

    fn restore_goal(&self, g: &Goal) {
        let mut store = self.store.write().expect("memory store poisoned");
        let mut tags = vec![format!("goal:{}", g.goal_id)];
        if g.is_current { tags.push(TAG_CURRENT.to_string()); }
        let metadata = MemoryMetadata {
            source_request_id: format!("goal:{}", g.goal_id),
            layer_captured: 0,
            category: MemoryCategory::Goal,
            tags,
            text_summary: g.text.clone(),
        };
        let _ = store.store_with_options(
            format!("goal:{}", g.goal_id),
            vec![0.0f32; self.d_model],
            metadata,
            pinned_no_ttl(),
        );
    }

    fn restore_task(&self, t: &Task) {
        let mut store = self.store.write().expect("memory store poisoned");
        let tags = vec![
            format!("goal:{}", t.goal_id),
            format!("task:{}", t.task_id),
            t.status.tag().to_string(),
        ];
        let metadata = MemoryMetadata {
            source_request_id: format!("task:{}", t.task_id),
            layer_captured: 0,
            category: MemoryCategory::Task,
            tags,
            text_summary: t.text.clone(),
        };
        let opts = match t.status {
            TaskStatus::Done => StoreOptions { pinned: false, ttl_seconds: None },
            _ => pinned_no_ttl(),
        };
        let _ = store.store_with_options(
            format!("task:{}", t.task_id),
            vec![0.0f32; self.d_model],
            metadata,
            opts,
        );
    }

    fn restore_status(&self, s: &StatusEntry) {
        let mut store = self.store.write().expect("memory store poisoned");
        let key = "status:current".to_string();
        let metadata = MemoryMetadata {
            source_request_id: key.clone(),
            layer_captured: 0,
            category: MemoryCategory::Status,
            tags: vec![format!("goal:{}", s.goal_id)],
            text_summary: s.text.clone(),
        };
        let _ = store.store_with_options(
            key,
            vec![0.0f32; self.d_model],
            metadata,
            StoreOptions { pinned: false, ttl_seconds: Some(STATUS_TTL_SECONDS) },
        );
    }

    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    // --- Goals ---

    pub fn set_goal(&self, text: &str) -> String {
        let goal_id = Uuid::new_v4().to_string();
        let mut store = self.store.write().expect("memory store poisoned");

        // Clear the "current" tag on any existing current goal.
        let to_clear: Vec<String> = store
            .query_by_category(&MemoryCategory::Goal)
            .into_iter()
            .filter(|e| e.metadata.tags.iter().any(|t| t == TAG_CURRENT))
            .map(|e| e.key.clone())
            .collect();

        for key in to_clear {
            if let Some(e) = store.get(&key) {
                let mut new_meta = e.metadata.clone();
                new_meta.tags.retain(|t| t != TAG_CURRENT);
                let new_vec = e.vector.clone();
                // Stays pinned — it's still a goal, just no longer current.
                let _ = store.store_with_options(key, new_vec, new_meta, pinned_no_ttl());
            }
        }

        let metadata = MemoryMetadata {
            source_request_id: format!("goal:{goal_id}"),
            layer_captured: 0,
            category: MemoryCategory::Goal,
            tags: vec![format!("goal:{goal_id}"), TAG_CURRENT.to_string()],
            text_summary: text.to_string(),
        };
        let _ = store.store_with_options(
            format!("goal:{goal_id}"),
            vec![0.0f32; self.d_model],
            metadata,
            pinned_no_ttl(),
        );
        drop(store);
        self.save();
        goal_id
    }

    pub fn list_goals(&self) -> Vec<Goal> {
        let store = self.store.read().expect("memory store poisoned");
        store
            .query_by_category(&MemoryCategory::Goal)
            .into_iter()
            .map(|e| Goal {
                goal_id: extract_id(&e.metadata.tags, "goal:").unwrap_or_default(),
                text: e.metadata.text_summary.clone(),
                is_current: e.metadata.tags.iter().any(|t| t == TAG_CURRENT),
            })
            .collect()
    }

    pub fn set_current(&self, goal_id: &str) -> bool {
        let mut store = self.store.write().expect("memory store poisoned");
        let goal_key = format!("goal:{goal_id}");

        // Verify target exists.
        if store.get(&goal_key).is_none() {
            return false;
        }

        // Collect keys that currently have the "current" tag.
        let to_clear: Vec<String> = store
            .query_by_category(&MemoryCategory::Goal)
            .into_iter()
            .filter(|e| e.metadata.tags.iter().any(|t| t == TAG_CURRENT))
            .map(|e| e.key.clone())
            .collect();

        for key in to_clear {
            if let Some(e) = store.get(&key) {
                let mut new_meta = e.metadata.clone();
                new_meta.tags.retain(|t| t != TAG_CURRENT);
                let new_vec = e.vector.clone();
                let _ = store.store_with_options(key, new_vec, new_meta, pinned_no_ttl());
            }
        }

        // Tag the target as current.
        let ok = if let Some(e) = store.get(&goal_key) {
            let mut new_meta = e.metadata.clone();
            if !new_meta.tags.iter().any(|t| t == TAG_CURRENT) {
                new_meta.tags.push(TAG_CURRENT.to_string());
            }
            let new_vec = e.vector.clone();
            let _ = store.store_with_options(goal_key, new_vec, new_meta, pinned_no_ttl());
            true
        } else {
            false
        };
        drop(store);
        if ok { self.save(); }
        ok
    }

    pub fn current_goal(&self) -> Option<Goal> {
        let store = self.store.read().expect("memory store poisoned");
        store
            .query_by_category(&MemoryCategory::Goal)
            .into_iter()
            .find(|e| e.metadata.tags.iter().any(|t| t == TAG_CURRENT))
            .map(|e| Goal {
                goal_id: extract_id(&e.metadata.tags, "goal:").unwrap_or_default(),
                text: e.metadata.text_summary.clone(),
                is_current: true,
            })
    }

    // --- Tasks ---

    pub fn add_task(&self, goal_id: &str, text: &str) -> String {
        let task_id = Uuid::new_v4().to_string();
        let mut store = self.store.write().expect("memory store poisoned");
        let metadata = MemoryMetadata {
            source_request_id: format!("task:{task_id}"),
            layer_captured: 0,
            category: MemoryCategory::Task,
            tags: vec![
                format!("goal:{goal_id}"),
                format!("task:{task_id}"),
                TaskStatus::Active.tag().to_string(),
            ],
            text_summary: text.to_string(),
        };
        // Active tasks are pinned; they unpin only when marked Done.
        let _ = store.store_with_options(
            format!("task:{task_id}"),
            vec![0.0f32; self.d_model],
            metadata,
            pinned_no_ttl(),
        );
        drop(store);
        self.save();
        task_id
    }

    pub fn update_task(&self, task_id: &str, status: TaskStatus) -> bool {
        let mut store = self.store.write().expect("memory store poisoned");
        let key = format!("task:{task_id}");
        let ok = if let Some(e) = store.get(&key) {
            let mut new_meta = e.metadata.clone();
            new_meta
                .tags
                .retain(|t| TaskStatus::from_tag(t).is_none());
            new_meta.tags.push(status.tag().to_string());
            let new_vec = e.vector.clone();
            // Done tasks unpin (historical, evictable). Active/Blocked stay
            // pinned as live work-in-progress.
            let opts = match status {
                TaskStatus::Done => StoreOptions { pinned: false, ttl_seconds: None },
                TaskStatus::Active | TaskStatus::Blocked => pinned_no_ttl(),
            };
            let _ = store.store_with_options(key, new_vec, new_meta, opts);
            true
        } else {
            false
        };
        drop(store);
        if ok { self.save(); }
        ok
    }

    pub fn list_tasks(&self, goal_id: &str) -> Vec<Task> {
        let store = self.store.read().expect("memory store poisoned");
        let tag = format!("goal:{goal_id}");
        store
            .query_by_tag(&tag)
            .into_iter()
            .filter(|e| e.metadata.category == MemoryCategory::Task)
            .map(|e| Task {
                task_id: extract_id(&e.metadata.tags, "task:").unwrap_or_default(),
                goal_id: goal_id.to_string(),
                text: e.metadata.text_summary.clone(),
                status: e
                    .metadata
                    .tags
                    .iter()
                    .find_map(|t| TaskStatus::from_tag(t))
                    .unwrap_or(TaskStatus::Active),
            })
            .collect()
    }

    // --- Status ---

    pub fn set_status(&self, text: &str) -> bool {
        let mut store = self.store.write().expect("memory store poisoned");

        // Find current goal id under the already-acquired write lock.
        let goal_id = store
            .query_by_category(&MemoryCategory::Goal)
            .into_iter()
            .find(|e| e.metadata.tags.iter().any(|t| t == TAG_CURRENT))
            .and_then(|e| extract_id(&e.metadata.tags, "goal:"))
            .unwrap_or_else(|| "none".to_string());

        let key = "status:current".to_string();
        let metadata = MemoryMetadata {
            source_request_id: key.clone(),
            layer_captured: 0,
            category: MemoryCategory::Status,
            tags: vec![format!("goal:{goal_id}")],
            text_summary: text.to_string(),
        };
        // Status is a rolling snapshot — not pinned, with a 1-hour TTL.
        // Eviction or expiry just means we lose the breadcrumb; the goal
        // and tasks survive.
        let _ = store.store_with_options(
            key,
            vec![0.0f32; self.d_model],
            metadata,
            StoreOptions { pinned: false, ttl_seconds: Some(STATUS_TTL_SECONDS) },
        );
        drop(store);
        self.save();
        true
    }

    pub fn latest_status(&self) -> Option<StatusEntry> {
        let store = self.store.read().expect("memory store poisoned");
        let entries = store.query_by_category(&MemoryCategory::Status);
        entries
            .into_iter()
            .max_by_key(|e| e.created_at)
            .map(|e| StatusEntry {
                text: e.metadata.text_summary.clone(),
                goal_id: extract_id(&e.metadata.tags, "goal:").unwrap_or_default(),
            })
    }

    // --- Composite view ---

    pub fn get_state(&self) -> GoalState {
        let current_goal = self.current_goal();
        let active_tasks = match &current_goal {
            Some(g) => self
                .list_tasks(&g.goal_id)
                .into_iter()
                .filter(|t| t.status == TaskStatus::Active)
                .collect(),
            None => Vec::new(),
        };
        let latest_status = self.latest_status();
        GoalState {
            current_goal,
            active_tasks,
            latest_status,
        }
    }

    /// Build the text block that should be prepended to user prompts so the
    /// model always sees the current goal/tasks/status. Returns "" when there
    /// is nothing to share.
    pub fn build_prompt_prefix(&self) -> String {
        let state = self.get_state();
        if state.current_goal.is_none() && state.active_tasks.is_empty() && state.latest_status.is_none() {
            return String::new();
        }

        let mut out = String::new();
        if let Some(g) = &state.current_goal {
            out.push_str(&format!("GOAL: {}\n", g.text));
        }
        if !state.active_tasks.is_empty() {
            out.push_str("ACTIVE TASKS:\n");
            for t in &state.active_tasks {
                out.push_str(&format!("  - {}\n", t.text));
            }
        }
        if let Some(s) = &state.latest_status {
            out.push_str(&format!("STATUS: {}\n", s.text));
        }
        out.push('\n');
        out
    }
}

fn extract_id(tags: &[String], prefix: &str) -> Option<String> {
    tags.iter()
        .find_map(|t| t.strip_prefix(prefix).map(|s| s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::memory_store::MemoryStore;

    fn make_manager() -> GoalManager {
        let store = Arc::new(RwLock::new(MemoryStore::new(1024, 256)));
        GoalManager::new(store).with_d_model(16)
    }

    #[test]
    fn set_goal_round_trip() {
        let m = make_manager();
        let id = m.set_goal("build the bughunter");
        let goals = m.list_goals();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].goal_id, id);
        assert_eq!(goals[0].text, "build the bughunter");
        assert!(goals[0].is_current);
    }

    #[test]
    fn second_set_goal_supersedes_current() {
        let m = make_manager();
        let _id1 = m.set_goal("first");
        let id2 = m.set_goal("second");
        let goals = m.list_goals();
        assert_eq!(goals.len(), 2);
        let current: Vec<_> = goals.iter().filter(|g| g.is_current).collect();
        assert_eq!(current.len(), 1, "exactly one goal must be current");
        assert_eq!(current[0].goal_id, id2);
    }

    #[test]
    fn set_current_flips_correctly() {
        let m = make_manager();
        let id1 = m.set_goal("first");
        let _id2 = m.set_goal("second");
        assert!(m.set_current(&id1));
        let goals = m.list_goals();
        let current: Vec<_> = goals.iter().filter(|g| g.is_current).collect();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].goal_id, id1);
    }

    #[test]
    fn add_task_and_update_status() {
        let m = make_manager();
        let goal_id = m.set_goal("ship feature X");
        let task_id = m.add_task(&goal_id, "write the migration");
        let tasks = m.list_tasks(&goal_id);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, TaskStatus::Active);
        assert!(m.update_task(&task_id, TaskStatus::Done));
        let tasks = m.list_tasks(&goal_id);
        assert_eq!(tasks[0].status, TaskStatus::Done);
    }

    #[test]
    fn build_prompt_prefix_empty_when_nothing_set() {
        let m = make_manager();
        assert_eq!(m.build_prompt_prefix(), "");
    }

    #[test]
    fn build_prompt_prefix_contains_goal_tasks_status() {
        let m = make_manager();
        let goal_id = m.set_goal("build the bughunter");
        m.add_task(&goal_id, "wire auto-fix");
        m.add_task(&goal_id, "deploy to dev");
        m.set_status("iteration 12 running");
        let prefix = m.build_prompt_prefix();
        assert!(prefix.contains("GOAL: build the bughunter"));
        assert!(prefix.contains("wire auto-fix"));
        assert!(prefix.contains("deploy to dev"));
        assert!(prefix.contains("STATUS: iteration 12 running"));
    }

    #[test]
    fn done_tasks_not_in_active_state() {
        let m = make_manager();
        let goal_id = m.set_goal("g");
        let t1 = m.add_task(&goal_id, "task one");
        let _t2 = m.add_task(&goal_id, "task two");
        m.update_task(&t1, TaskStatus::Done);
        let state = m.get_state();
        assert_eq!(state.active_tasks.len(), 1);
        assert_eq!(state.active_tasks[0].text, "task two");
    }

    // --- v0.2 pin/TTL tests ---

    use crate::engine::memory_store::MemoryCategory;
    use std::collections::HashMap;

    fn pressure_manager() -> GoalManager {
        // Tiny category budgets to provoke pressure quickly. d_model=8 → each
        // entry is 32 bytes. Context cap is 96 bytes → 3 unpinned Context
        // entries max.
        let mut budgets = HashMap::new();
        budgets.insert(MemoryCategory::Goal, 1024);
        budgets.insert(MemoryCategory::Task, 1024);
        budgets.insert(MemoryCategory::Status, 256);
        budgets.insert(MemoryCategory::Context, 96);
        budgets.insert(MemoryCategory::Finding, 256);
        budgets.insert(MemoryCategory::Pattern, 256);
        budgets.insert(MemoryCategory::Correction, 256);
        budgets.insert(MemoryCategory::Knowledge, 256);
        let store = Arc::new(RwLock::new(MemoryStore::with_budget(
            128, 16, 8192, budgets,
        )));
        GoalManager::new(store).with_d_model(8)
    }

    #[test]
    fn goal_survives_context_pressure() {
        let m = pressure_manager();
        let _goal_id = m.set_goal("the only goal that matters");
        // Flood Context — goal must survive because it's pinned.
        {
            let mut store = m.store.write().unwrap();
            for i in 0..50 {
                let _ = store.store_with_options(
                    format!("noise{i}"),
                    vec![0.5; 8],
                    MemoryMetadata {
                        source_request_id: "test".into(),
                        layer_captured: 0,
                        category: MemoryCategory::Context,
                        tags: vec![],
                        text_summary: String::new(),
                    },
                    StoreOptions::default(),
                );
            }
        }
        let goals = m.list_goals();
        assert_eq!(goals.len(), 1, "pinned goal must survive Context pressure");
        assert_eq!(goals[0].text, "the only goal that matters");
    }

    #[test]
    fn done_task_becomes_evictable() {
        let m = pressure_manager();
        let goal_id = m.set_goal("g");
        let t = m.add_task(&goal_id, "the task");
        // While active + pinned: check the entry exists and is pinned.
        {
            let store = m.store.read().unwrap();
            let key = format!("task:{t}");
            let entry = store.query_by_tag(&format!("task:{t}"));
            assert_eq!(entry.len(), 1);
            assert_eq!(entry[0].key, key);
            assert!(entry[0].pinned, "active task must be pinned");
        }
        // Mark done → unpinned.
        assert!(m.update_task(&t, TaskStatus::Done));
        {
            let store = m.store.read().unwrap();
            let entry = store.query_by_tag(&format!("task:{t}"));
            assert_eq!(entry.len(), 1);
            assert!(!entry[0].pinned, "done task must be unpinned");
        }
    }
}
