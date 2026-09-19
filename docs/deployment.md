# Deployment boundaries and release limitations

System 1 is beta software. The supported release claims are the behavior demonstrated by the current source, tests, and [recorded quality run](../benchmarks/quality/results/launch_review.json). Earlier research manuscripts and graphics contain historical or illustrative measurements; they are not deployment guarantees.

## Deterministic tool authorization

Use `SystemOneGuard(enforcement_profile=True)` with explicit `PolicyEngine` grants, an Ed25519 signing key, and a persistent `ActionLedger`. No matching deterministic permission produces `DENY`. Rules compose with `DENY` taking precedence over `REQUIRE_APPROVAL`, then `ALLOW`.

A guard result evaluates the policy you configured. It does not authenticate the `tenant_id`, `principal_id`, or `scope` supplied to it. Derive those values from a trusted application session, keep policy and key material outside agent control, and constrain tool arguments as well as names. Treat paths, commands, and remote targets according to the actual executor's semantics. A rule allowing every invocation of a shell or file tool is a broad grant.

The application must check the outcome and execute only the authorized operation with the authorized arguments. The library is an in-process hook, not an OS sandbox; it cannot stop code that bypasses it. A receipt is evidence of a recorded decision, not a one-use execution capability. Add application-level replay prevention when it is required.

Default/diagnostic guard mode can grant permission through the statistical path and does not require durable storage or signing. Use it for experiments, not as a substitute for an enforcement configuration.

## Statistical limits

The seed projector uses local features and schema descriptions. The included security-triage benchmark achieves 50% raw accuracy, including 8 of 35 BLOCK examples classified as ALLOW. This is evidence that the seed classifier must not be used as a standalone security boundary. The benchmark does not measure the error rate of the separate deterministic guard.

Split-conformal coverage requires calibration examples and future examples to be exchangeable, with the scoring model fixed independently of the conformal calibration fold. Its finite-sample guarantee concerns marginal membership of the true label in a prediction set. It does not guarantee correctness conditional on accepting a singleton, simultaneous coverage across every field, or adversarial robustness. An empty set is an abstention signal, not proof of distribution shift. See [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

Use separate training, calibration, and evaluation data. Recalibrate after model updates and measure behavior on your deployment distribution. Neither the significance level `alpha` nor a confidence threshold implies a fixed escalation or local-retention rate. Optional fallback routing is application behavior; `System1Engine.decide()` itself returns structured values and uncertainty rather than executing a frontier model or tool.

## Audit evidence and privacy

Signed receipts allow a verifier with an independently trusted public key to detect changes to the signed payload. The SQLite ledger is a linear SHA-256 hash chain, not a Merkle tree or hardware attestation. It cannot prove that an external tool executed correctly or that a compromised host recorded every event. Preserve trusted ledger-head checkpoints outside the writable ledger if detecting truncation or rollback is required.

Keep signing keys persistent and protected; regenerating a key on every process start breaks a single-key trust chain. Maintain backups and an application-specific rotation strategy. Outcome records are linked to their prior authorization through the ledger; they are not independent hardware execution proofs. Explicit policy denials and approval requirements do not always produce receipts, so applications that need a complete denial audit must record those events separately.

Receipts and ledgers can contain prompts, tool arguments, and results. Local processing does not remove the need to control access to those files. The package does not provide regulatory certification or establish an organization's compliance by itself.

The core local decision path does not initiate external network calls. Optional teacher/fallback clients, observability exporters, user-provided projectors, and wrapped tools can do so. A `zero_egress` setting controls the relevant library fallback path, not the entire Python process or operating system.

## ASGI gateway

`SystemOneGatewayMiddleware` can generate a local HTTP response before the downstream app runs. Put authentication, authorization, request-size limits, and any required logging outside this middleware. FastAPI endpoint dependencies do not run on its local response path.

Configure the intended `route_paths`; matching uses path prefixes, and the `x-reflex-route: true` header can also opt a POST into classification. Treat this middleware as a classification endpoint rather than a proxy that preserves arbitrary endpoint semantics. A custom `is_fastpath_fn` takes responsibility for its own acceptance rules.

The middleware buffers matched request bodies. Apply request-size limits in your front proxy or outer middleware. On escalation it replays the body once and then forwards subsequent ASGI receive events, including disconnects.

## LangChain and MCP

Use one configured principal per authenticated application context. LangChain run IDs correlate tool runs; they are not principal identities. LangChain callbacks must be attached to actual tool invocations, with blocking exceptions allowed to propagate. Keep `raise_error=True` enabled on the guard callback.

The wrappers are adapters rather than a complete agent host or MCP authentication layer. Inspect their proposal contracts when writing argument policies. The gRPC Guard RPC's prompt/schema contract is distinct from an application-bound MCP or Python tool authorization proposal.

## gRPC

The CLI server binds to loopback by default using insecure gRPC transport. It has no built-in caller authentication or TLS, and the CLI does not configure a persistent signing key, ledger, or deterministic policy. Keep diagnostic instances local. A service deployment needs an authenticated transport boundary and explicit application configuration through `system1.grpc_server.serve(...)`.

No maintained Docker image, Kubernetes manifests, HSM integration, or hardware enclave attestation is included in this release.

## Performance and optional backends

Sub-millisecond timings in the recorded run are median decision timings for small seed-model workloads. They do not include durable authorization, complete tool execution, or network services. Measure whole-operation latency and throughput on your own hardware; the ledger's integrity checks can grow with ledger history.

The base package uses NumPy and cryptography. MLX acceleration and Game Boy emulation are optional and depend on platform support and user-provided assets. The `reflex` compatibility namespace shares a name with a separate web framework; avoid installing both distributions in the same environment.
