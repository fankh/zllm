# ZLLM

White-box LLM inference engine with zero-copy latent intercept.

## Features

- Rust-native control plane (zero Python overhead)
- Zero-copy hidden state access on UMA (Apple Silicon, Grace Hopper)
- Mid-layer hook system: activation steering, early exit, hallucination detection
- Logit FSM for grammar-constrained decoding
- Paged KV cache with tenant isolation
- Adaptive Latent Reasoning (latent CoT without thinking tokens)

## Quick Start

```bash
# Build
cargo build --release

# Run server
cargo run -- serve --config configs/default.toml

# Health check
curl http://localhost:8080/health

# CLI help
cargo run -- --help
```

## Architecture

See [ai-inference-engine docs](https://github.com/fankh/new-research/tree/main/ai-inference-engine) for full architecture specification.
