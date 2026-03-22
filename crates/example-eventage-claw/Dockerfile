# syntax=docker/dockerfile:1
# Builds a fully static musl binary — runs on any Linux, no GLIBC version requirement.

# ── Stage 1: install cargo-chef + musl toolchain ─────────────────────────────
FROM rust:1-slim AS chef
RUN apt-get update \
    && apt-get install -y --no-install-recommends musl-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl
RUN cargo install cargo-chef --locked
WORKDIR /build

# ── Stage 2: compute dependency recipe ────────────────────────────────────────
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ── Stage 3: pre-compile dependencies (cached layer) ─────────────────────────
FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
RUN cargo chef cook --release --target x86_64-unknown-linux-musl \
    -p example-eventage-claw --recipe-path recipe.json

# Build the actual binary (only re-runs when source changes)
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-musl -p example-eventage-claw

# ── Stage 4: minimal runtime image ────────────────────────────────────────────
FROM alpine:3 AS runtime
RUN apk add --no-cache ca-certificates chromium docker-cli

COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/claw /usr/local/bin/claw

# Persistent data directory: config, skills, memory, tasks, event log.
VOLUME ["/root/.claw"]

EXPOSE 3000

ENTRYPOINT ["claw"]
CMD ["--no-tui", "--http-port", "3000", "--log", "/root/.claw/events.jsonl"]
