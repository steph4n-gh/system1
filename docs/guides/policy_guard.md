# Add an explicit tool policy

A taught skill suggests an answer. If that answer would trigger a tool, the
application also needs an explicit permission decision. This is an optional step
after [teaching and checking your first skill](first_skill.md).

This executable example permits one configuration lookup for one application-supplied principal, records the actual lookup result, and verifies the ledger. Other keys or tools have no grant. It persists a local demonstration key across runs.

```python
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import (
    ActionLedger, ActionProposal, DecisionOutcome, PolicyEngine, PolicyRule,
    RiskLevel, SystemOneGuard, load_private_key, save_keypair,
)

key_path = Path(".system1/demo-identity/identity.key")
if not key_path.exists():
    save_keypair(Ed25519PrivateKey.generate(), key_path.parent)
signing_key = load_private_key(key_path)

policy = PolicyEngine(rules=[PolicyRule(
    rule_id="read_service_name",
    tools=["read_config"],
    allowed_principals=["agent-worker"],
    allowed_tenants=["demo"],
    allowed_scopes=["config:read"],
    argument_limits={"key": ["service_name"]},
    outcome=DecisionOutcome.ALLOW,
    risk=RiskLevel.READ_ONLY,
)])

with ActionLedger(".system1/demo-audit.sqlite", require_durable=True) as ledger:
    guard = SystemOneGuard(
        policy=policy, ledger=ledger, signing_key=signing_key,
        enforcement_profile=True,
    )
    proposal = ActionProposal.create(
        tenant_id="demo", principal_id="agent-worker", scope="config:read",
        tool="read_config", arguments={"key": "service_name"},
        canonical_target="config:service_name", purpose="Inspect service name",
    )
    auth = guard.evaluate_proposal(proposal)
    if auth.outcome != DecisionOutcome.ALLOW:
        raise PermissionError(auth.reason)

    # Execute exactly the authorized operation and arguments.
    result = {"service_name": "system1-demo"}[proposal.arguments["key"]]
    ledger.record_execution_outcome(
        action_id=proposal.action_id,
        receipt_digest=auth.receipt.compute_digest(),
        status="SUCCEEDED", result_payload={"value": result},
        tenant_id=proposal.tenant_id, principal_id=proposal.principal_id,
        scope=proposal.scope, trusted_public_key=signing_key.public_key(),
    )
    assert ledger.verify_integrity(trusted_public_key=signing_key.public_key())
    print(result)
```

The application must derive identity from its authenticated session, constrain tool arguments and targets, and keep agent code from bypassing the guard. Policy correctness is the operator's responsibility. The default guard without `enforcement_profile=True` can use statistical classification to allow actions. See [deployment boundaries](https://github.com/steph4n-gh/system1/blob/main/docs/deployment.md) before granting consequential permissions.
