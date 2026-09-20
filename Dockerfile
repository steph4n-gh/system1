# Experimental local gRPC container; see deploy/README.md for boundaries.
# Build:  docker build -t system1-grpc:local .
# Run:    docker run --rm -p 127.0.0.1:50051:50051 system1-grpc:local system1 serve --grpc --host 0.0.0.0 --port 50051

FROM python:3.12-slim AS base

LABEL maintainer="System 1 Authors"
LABEL description="System 1 Decision Engine gRPC sidecar server"

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# No additional system dependencies needed beyond what slim provides.

# Copy only dependency metadata first for layer caching.
COPY pyproject.toml README.md LICENSE ./

# Copy source code.
COPY src/ src/

# Install the package with grpc extras.
RUN pip install --no-cache-dir ".[grpc]"

# Expose the default gRPC port.
EXPOSE 50051

# TCP liveness only; this does not verify authorization or service readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost',50051)); s.close()" || exit 1

# Default command: start the gRPC sidecar server.
CMD ["system1", "serve", "--grpc", "--port", "50051"]
