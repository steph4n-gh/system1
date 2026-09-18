# Reflex: Technical Brief & Architecture Guide
**The On-Metal Decision Firewall for Autonomous AI Agents**

---

## Executive Summary

Autonomous AI agents executing in enterprise workflows present a critical security vulnerability: agents possess authorization to invoke tools (database queries, terminal commands, cloud API invocations, file modifications), yet existing architectures evaluate tool safety through slow, non-deterministic cloud language models.

This architectural pattern creates three enterprise bottlenecks:
1. **Prohibitive Latency (300 ms – 2,000+ ms)**: Blocking tool invocations on cloud API roundtrips breaks interactive SLAs and real-time execution loops.
2. **Data Sovereignty Violations**: Tool execution parameters and context prompts traverse public WAN networks, exposing PII, PHI, and credentials.
3. **Uncalibrated Safety Failure**: Heuristic scalar confidence scores emitted by neural networks are uncalibrated under covariate shift, causing silent permissions failures.

**Reflex** is the machine-native decision firewall engineered to sit directly between autonomous agents and their execution environments. Running entirely in-process on host silicon (CPU SIMD or Apple Silicon Metal), Reflex evaluates structured decision policies in **< 1.0 ms P50** with **zero external network egress**, enforced by **finite-sample Split Conformal Prediction** and **tamper-evident Ed25519 digital witness receipts**.

```
                           Incoming Agent Tool Request
                                        │
                                        ▼
                     ┌───────────────────────────────────────┐
                     │   Tier 0: Exact-Match Cache (< 10µs)  │ ── Hit ──► Certified Execution
                     └──────────────────┬────────────────────┘
                                        │ Miss
                                        ▼
                     ┌───────────────────────────────────────┐
                     │    Reflex Decision Engine (< 1 ms)    │
                     │    • In-Process Metal / BLAS GEMM     │
                     │    • Zero External Network Sockets    │
                     └──────────────────┬────────────────────┘
                                        │ Probability Vector P(y|x)
                                        ▼
                     ┌───────────────────────────────────────┐
                     │   Tier 2: Conformal Safety Gate       │
                     │   • Finite-Sample Coverage (1 - α)    │
                     │   • Dominance Margin M(x) ≥ τ         │
                     └───────┬───────────────────────┬───────┘
                             │                       │
               Pass: Confident & Safe (95-99%)       │ Ambiguous / Edge Case (1-5%)
                             │                       │
                             ▼                       ▼
              ┌─────────────────────────────┐  ┌───────────────────────────────┐
              │ Execute Locally on Metal    │  │ Halt Fail-Closed & Escalate   │
              │ • SHA-256 Merkle Ledger     │  │ • Frontier Reasoning Governor │
              │ • Ed25519 Signed Receipt    │  │   (OpenAI, Anthropic, Google) │
              └─────────────────────────────┘  └───────────────┬───────────────┘
                                                               │
                                                               │ Ground-Truth Resolution y*
                                                               ▼ (Sherman-Morrison < 50µs)
                                               ┌───────────────────────────────┐
                                               │ Online Model Adaptation       │
                                               │ Rotates Boundary on Silicon   │
                                               └───────────────────────────────┘
```

---

## 1. System Topology & Memory Hierarchy

Reflex organizes decision evaluation across a 4-tier execution hierarchy:

| Tier | Component | Latency (P50) | Description |
|---|---|---|---|
| **Tier 0** | **Exact-Match Cache** | **9.8 µs** | SHA-256 in-memory hash table for recurring queries and operational pulses. |
| **Tier 1** | **Non-Autoregressive Metal Engine** | **0.98 ms** | Hybrid sparse-dense projection ($D=4,160$) evaluated via Apple Metal or NumPy BLAS. |
| **Tier 2** | **Conformal Safety Gate** | **4.0 µs** | Evaluates prediction set cardinality $|\mathcal{C}_{1-\alpha}|$ and dominance margin $M(\mathbf{x}) \ge \tau$. |
| **Tier 3** | **Deliberative Escalation & Adaptation** | **38.4 µs** | Halts fail-closed for human or frontier model review; rotates hyperplanes via Sherman-Morrison rank-1 update. |

---

## 2. Enterprise Integrations

### 2.1 Model Context Protocol (MCP) Safety Proxy

Reflex intercepts MCP `tools/call` JSON-RPC requests on host silicon before tool execution:

```python
from reflex.integrations import ReflexMCPProxy, wrap_mcp_tool

# Option A: Protect individual tool functions
@wrap_mcp_tool(tool_name="execute_terminal_command")
def run_command(command: str) -> str:
    return subprocess.check_output(command, shell=True).decode()

# Option B: Raw JSON-RPC proxy for MCP servers
proxy = ReflexMCPProxy(tenant_id="prod_cluster")
allowed, err_resp, result = proxy.intercept_jsonrpc(mcp_request_json)
if not allowed:
    return err_resp  # JSON-RPC 2.0 error (-32000) with cryptographic audit proof
```

### 2.2 FastAPI / Starlette Gateway Middleware

Route incoming requests at the gateway level in $<1$ ms:

```python
from fastapi import FastAPI
from reflex.integrations import add_reflex_gateway
from reflex import DecisionSchema, ChoiceField

app = FastAPI()

class IntentRouter(DecisionSchema):
    route = ChoiceField(
        options=["technical_support", "billing_inquiry", "escalate_to_frontier"],
        descriptions={
            "technical_support": "Software setup, installation, or debugging questions",
            "billing_inquiry": "Invoices, subscription plans, and refunds",
            "escalate_to_frontier": "Complex reasoning, novel queries, or ambiguous disputes",
        }
    )

add_reflex_gateway(app, schema=IntentRouter, fastpath_threshold=0.85)
```

### 2.3 LangChain Agent Guard

Fail-closed callback handler preventing unauthorized actions during multi-turn agent execution:

```python
from reflex.integrations import ReflexGuardCallbackHandler

guard_handler = ReflexGuardCallbackHandler()
agent_executor = create_react_agent(llm, tools, callbacks=[guard_handler])
```

### 2.4 Polyglot gRPC Sidecar

Deploy Reflex as a Kubernetes sidecar for non-Python agents (Go, TypeScript, Rust):

```protobuf
syntax = "proto3";
package reflex.v1;

service ReflexService {
  rpc Decide(DecideRequest) returns (DecideResponse);
  rpc Guard(GuardRequest) returns (GuardResponse);
  rpc VerifyReceipt(VerifyReceiptRequest) returns (VerifyReceiptResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}
```

```bash
# Launch gRPC server on port 50051
reflex serve --grpc --port 50051
```

---

## 3. Observability & Monitoring

Reflex includes native Prometheus and OpenTelemetry instrumentation:

* **Prometheus `/metrics`**:
  * `reflex_decisions_total` (counter, labels: `schema`, `outcome`, `cache_hit`)
  * `reflex_decision_latency_seconds` (histogram: 0.1ms to 1000ms buckets)
  * `reflex_escalations_total` (counter, labels: `schema`, `reason`)
  * `reflex_cache_hit_ratio` (gauge: rolling hit rate)
  * `reflex_conformal_set_size` (histogram: prediction set cardinality)
  * `reflex_ledger_entries_total` (gauge: audit trail depth)
* **Grafana Dashboard**: Pre-built dashboard template located in `docs/observability/grafana-dashboard.json`.

---

## 4. Compliance, Auditability, and Security Architecture

### 4.1 Tamper-Evident ActionLedger & Receipts

Every authorized or denied decision generates an immutable cryptographic proof:
1. **DecisionWitnessReceipt**: Formatted as canonical JSON and signed by an on-device Ed25519 private key.
2. **ActionLedger**: Stored in a local SQLite database utilizing Write-Ahead Logging (WAL) and SHA-256 rolling Merkle hash-chaining.
3. **Offline Verifiability**: Third-party auditors can verify receipts without access to the runtime or model weights:
   ```bash
   reflex verify-receipt path/to/receipt.json
   ```

### 4.2 Regulatory Compliance Alignment

* **HIPAA**: Protected Health Information (PHI) evaluated within host process boundaries; 0 bytes transmitted over external networks.
* **GDPR**: Data processing remains on-premise without international data transfers.
* **SOC 2 Type II**: Non-repudiable audit trails recorded locally with software Ed25519 cryptographic receipts and SQLite Merkle chaining.

---

## 5. Performance Benchmark Summary

Empirical SLA on Apple M3 Max and Ubuntu 24.04 LTS (10,000 trials):

| Metric | Reflex L1 Cache | Reflex Forward Pass | Cloud Fast API | Frontier LLM |
|---|---|---|---|---|
| **P50 Latency** | **9.8 µs** | **0.98 ms** | 220 ms | 850 ms |
| **P99 Latency** | **14 µs** | **1.34 ms** | 410 ms | 1,480 ms |
| **Throughput (1 host)** | > 50,000 QPS | > 2,200 QPS | Cloud-limited | Cloud-limited |
| **Network Egress** | 0 Bytes | 0 Bytes | Full Payload | Full Payload |
| **Direct Token Cost** | $0.00 | $0.00 | Metered | Metered |

---

## 6. Deployment Patterns

1. **In-Process Python**: Directly embedded via `pip install reflex`.
2. **Containerized Microservice**: Official Docker image based on `python:3.12-slim` (`Dockerfile`).
3. **Kubernetes Sidecar**: Co-located with agent pods via `deploy/kubernetes/sidecar-example.yaml`.
