# Experimental container templates

The [Dockerfile](../Dockerfile) and [Kubernetes examples](kubernetes/) are local
starting points. No container image is published by this project. The CLI service
uses insecure gRPC and does not configure caller authentication, signing keys,
a durable ledger or deterministic permission policy. Follow the
[deployment boundaries](../docs/deployment.md#grpc) for an application deployment.
The library's Python/OS and wheel checks do not validate a Kubernetes rollout.

## Local container

From the repository root with Docker running:

```bash
docker build -t system1-grpc:local .
docker run --rm -p 127.0.0.1:50051:50051 system1-grpc:local system1 serve --grpc --host 0.0.0.0 --port 50051
```

The explicit host binds inside the container so port publishing can reach it;
the host-side mapping stays on loopback. Running the image without an override
retains the CLI's loopback default. The TCP health check only checks connectivity.
The image includes the license and installs the existing `grpc` extra.

## Kubernetes examples

[deployment.yaml](kubernetes/deployment.yaml) and [service.yaml](kubernetes/service.yaml)
use the legacy `reflex-grpc` resource labels consistently. Load your locally built
`system1-grpc:local` image into the cluster or replace the image reference with
your own. The standalone service listens on the pod interface and is reachable
by other cluster clients; an authenticated boundary is not supplied.

[sidecar-example.yaml](kubernetes/sidecar-example.yaml) instead keeps the runtime
on loopback within the shared pod namespace. Its exec probes check loopback too.
Replace `your-agent-image:latest` with your own agent image. `REFLEX_GRPC_ENDPOINT`
is an example application variable, not a runtime configuration option.

These templates do not provision a production service, authenticated identity,
policy, keys or storage. Use the Python `system1.grpc_server.serve(...)` API for
explicit application configuration and supply your own protected transport.
