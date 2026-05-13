use crate::control_plane::goal_manager::{GoalManager, TaskStatus as GmTaskStatus};
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub mod inference_proto {
    tonic::include_proto!("zllm.inference");
}

pub mod control_proto {
    tonic::include_proto!("zllm.control");
}

use control_proto::goal_service_server::GoalService;
use control_proto::{
    AddTaskRequest, AddTaskResponse, GetStateRequest, GetStateResponse, Goal as ProtoGoal,
    ListGoalsRequest, ListGoalsResponse, SetCurrentGoalRequest, SetCurrentGoalResponse,
    SetGoalRequest, SetGoalResponse, SetStatusRequest, SetStatusResponse,
    Status as ProtoStatusMsg, Task as ProtoTask, TaskStatus as ProtoTaskStatus, UpdateTaskRequest,
    UpdateTaskResponse,
};
use inference_proto::inference_service_server::InferenceService;
use inference_proto::{InferMetrics, InferRequest, InferResponse, StreamChunk};

pub struct ZllmInferenceService;

#[tonic::async_trait]
impl InferenceService for ZllmInferenceService {
    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> std::result::Result<Response<InferResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("gRPC infer: tenant={}, prompt_len={}", req.tenant_id, req.prompt.len());

        // Stub: return dummy response
        let response = InferResponse {
            text: format!("Hello from ZLLM! Echo: {}", &req.prompt[..req.prompt.len().min(50)]),
            token_ids: vec![1, 2, 3],
            metrics: Some(InferMetrics {
                ttft_ms: 42.0,
                total_ms: 100.0,
                tokens_generated: 3,
                reasoning_loops: 1,
                early_exit: false,
                tokens_per_second: 30.0,
            }),
        };

        Ok(Response::new(response))
    }

    type InferStreamStream = tokio_stream::wrappers::ReceiverStream<std::result::Result<StreamChunk, Status>>;

    async fn infer_stream(
        &self,
        request: Request<InferRequest>,
    ) -> std::result::Result<Response<Self::InferStreamStream>, Status> {
        let req = request.into_inner();
        tracing::info!("gRPC infer_stream: tenant={}", req.tenant_id);

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::spawn(async move {
            // Stub: stream 3 dummy tokens
            for i in 0..3 {
                let chunk = StreamChunk {
                    token: format!("token_{i}"),
                    token_id: i as u32,
                    is_final: i == 2,
                    metrics: if i == 2 {
                        Some(InferMetrics {
                            ttft_ms: 42.0,
                            total_ms: 100.0,
                            tokens_generated: 3,
                            reasoning_loops: 1,
                            early_exit: false,
                            tokens_per_second: 30.0,
                        })
                    } else {
                        None
                    },
                };
                let _ = tx.send(Ok(chunk)).await;
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

// --- GoalService -----------------------------------------------------------

pub struct ZllmGoalService {
    manager: Arc<GoalManager>,
}

impl ZllmGoalService {
    pub fn new(manager: Arc<GoalManager>) -> Self {
        Self { manager }
    }
}

fn task_status_to_proto(s: GmTaskStatus) -> ProtoTaskStatus {
    match s {
        GmTaskStatus::Active => ProtoTaskStatus::Active,
        GmTaskStatus::Done => ProtoTaskStatus::Done,
        GmTaskStatus::Blocked => ProtoTaskStatus::Blocked,
    }
}

fn task_status_from_proto(s: ProtoTaskStatus) -> GmTaskStatus {
    match s {
        ProtoTaskStatus::Done => GmTaskStatus::Done,
        ProtoTaskStatus::Blocked => GmTaskStatus::Blocked,
        _ => GmTaskStatus::Active,
    }
}

#[tonic::async_trait]
impl GoalService for ZllmGoalService {
    async fn set_goal(
        &self,
        request: Request<SetGoalRequest>,
    ) -> std::result::Result<Response<SetGoalResponse>, Status> {
        let req = request.into_inner();
        if req.text.trim().is_empty() {
            return Err(Status::invalid_argument("text must not be empty"));
        }
        let goal_id = self.manager.set_goal(&req.text);
        Ok(Response::new(SetGoalResponse { goal_id }))
    }

    async fn list_goals(
        &self,
        _request: Request<ListGoalsRequest>,
    ) -> std::result::Result<Response<ListGoalsResponse>, Status> {
        let goals = self
            .manager
            .list_goals()
            .into_iter()
            .map(|g| ProtoGoal {
                goal_id: g.goal_id,
                text: g.text,
                is_current: g.is_current,
            })
            .collect();
        Ok(Response::new(ListGoalsResponse { goals }))
    }

    async fn set_current_goal(
        &self,
        request: Request<SetCurrentGoalRequest>,
    ) -> std::result::Result<Response<SetCurrentGoalResponse>, Status> {
        let req = request.into_inner();
        let success = self.manager.set_current(&req.goal_id);
        Ok(Response::new(SetCurrentGoalResponse { success }))
    }

    async fn add_task(
        &self,
        request: Request<AddTaskRequest>,
    ) -> std::result::Result<Response<AddTaskResponse>, Status> {
        let req = request.into_inner();
        if req.goal_id.trim().is_empty() {
            return Err(Status::invalid_argument("goal_id must not be empty"));
        }
        if req.text.trim().is_empty() {
            return Err(Status::invalid_argument("text must not be empty"));
        }
        let task_id = self.manager.add_task(&req.goal_id, &req.text);
        Ok(Response::new(AddTaskResponse { task_id }))
    }

    async fn update_task(
        &self,
        request: Request<UpdateTaskRequest>,
    ) -> std::result::Result<Response<UpdateTaskResponse>, Status> {
        let req = request.into_inner();
        let status_proto = ProtoTaskStatus::try_from(req.status)
            .map_err(|_| Status::invalid_argument("invalid task status"))?;
        let success = self
            .manager
            .update_task(&req.task_id, task_status_from_proto(status_proto));
        Ok(Response::new(UpdateTaskResponse { success }))
    }

    async fn set_status(
        &self,
        request: Request<SetStatusRequest>,
    ) -> std::result::Result<Response<SetStatusResponse>, Status> {
        let req = request.into_inner();
        let success = self.manager.set_status(&req.text);
        Ok(Response::new(SetStatusResponse { success }))
    }

    async fn get_state(
        &self,
        _request: Request<GetStateRequest>,
    ) -> std::result::Result<Response<GetStateResponse>, Status> {
        let state = self.manager.get_state();
        let prompt_prefix = self.manager.build_prompt_prefix();
        let current_goal = state.current_goal.map(|g| ProtoGoal {
            goal_id: g.goal_id,
            text: g.text,
            is_current: g.is_current,
        });
        let active_tasks = state
            .active_tasks
            .into_iter()
            .map(|t| ProtoTask {
                task_id: t.task_id,
                goal_id: t.goal_id,
                text: t.text,
                status: task_status_to_proto(t.status) as i32,
            })
            .collect();
        let latest_status = state.latest_status.map(|s| ProtoStatusMsg {
            text: s.text,
            goal_id: s.goal_id,
        });
        Ok(Response::new(GetStateResponse {
            current_goal,
            active_tasks,
            latest_status,
            prompt_prefix,
        }))
    }
}
