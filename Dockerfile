# zllm CPU baseline image. GPU/Vulkan lanes are host-specific (Strix
# Halo coopmat kernels) and intentionally not containerized.
#
# Build:  docker build -t zllm .
# Run:    docker run -p 8080:8080 -v /path/to/models:/models zllm
#
# NOTE (Windows/macOS Docker Desktop): bind mounts stream GGUFs slowly —
# ~5 min per slot for a 1B on the smoke box. Copy models into a named
# volume (or set backend_pool_size = 1) for fast startup.
# The server binds 0.0.0.0 INSIDE the container (the container boundary
# is the isolation); publish the port to loopback (-p 127.0.0.1:8080:8080)
# or set ZLLM_API_KEY when exposing it wider.

FROM rust:1-slim AS build
# C toolchain for the native-code crates in the tree (onig for
# tokenizers, ring for TLS) — rust:*-slim ships none.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# CI parity: no target-cpu=native inside the image (portable artifact).
ENV RUSTFLAGS=""
RUN cargo build --release --locked

FROM debian:stable-slim
# ca-certificates: the /v1/models/download endpoint fetches over HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/zllm /usr/local/bin/zllm
COPY configs/default.toml /etc/zllm/config.toml
# Container-internal bind; isolation comes from port publishing.
ENV ZLLM_BIND=0.0.0.0
VOLUME ["/models"]
EXPOSE 8080
# Point the config's model path at the mounted volume, e.g.
#   path = "/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
# (single-file GGUFs work — the tokenizer is read from the file).
ENTRYPOINT ["zllm", "serve", "--config", "/etc/zllm/config.toml"]
