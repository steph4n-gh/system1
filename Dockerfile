# Reflex Decision Engine — gRPC Sidecar Container
# Build:  docker build -t reflex-grpc .
# Run:    docker run -p 50051:50051 reflex-grpc

FROM python:3.12-slim AS base

LABEL maintainer="Reflex / System 1 Authors"
LABEL description="Reflex Decision Engine gRPC sidecar server"

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# No additional system dependencies needed beyond what slim provides.

# Copy only dependency metadata first for layer caching.
COPY pyproject.toml README.md ./

# Copy source code.
COPY src/ src/

# Install the package with grpc extras.
RUN pip install --no-cache-dir ".[grpc]"

# Expose the default gRPC port.
EXPOSE 50051

# Health-check using grpc_health_v1 or simple TCP probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost',50051)); s.close()" || exit 1

# Default command: start the gRPC sidecar server.
CMD ["reflex", "serve", "--grpc", "--port", "50051"]
