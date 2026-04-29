use tonic::{Request, Response, Status};

pub mod inference_proto {
    tonic::include_proto!("zllm.inference");
}

use inference_proto::inference_service_server::InferenceService;
use inference_proto::{InferRequest, InferResponse, InferMetrics, StreamChunk};

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
