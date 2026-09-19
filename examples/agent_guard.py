#!/usr/bin/env python3
"""Teach operation triage, then demonstrate a separate deterministic tool grant.

Run: python examples/agent_guard.py
Triage labels are suggestions for intake review, never tool permissions.
Only the explicitly permitted local configuration lookup below is executed.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from system1 import (
    ActionLedger, ActionProposal, ChoiceField, DecisionOutcome, DecisionSchema,
    PolicyEngine, PolicyRule, RiskLevel, SystemOneGuard, load_private_key, save_keypair,
)
from _teaching_demo import run_example


class OperationTriageSchema(DecisionSchema):
    operation = ChoiceField(
        options=['inspect', 'change', 'restricted'],
        descriptions={
            'inspect': 'Read ordinary workspace information or report local status',
            'change': 'Create or modify ordinary workspace files, tests, settings, or dependencies',
            'restricted': 'Credentials, external transfers, access controls, privileged changes, or destructive removal',
        },
    )


def demonstrate_policy(output):
    key_path = output / 'identity' / 'identity.key'
    if not key_path.exists():
        save_keypair(Ed25519PrivateKey.generate(), key_path.parent)
    key = load_private_key(key_path)
    policy = PolicyEngine(rules=[PolicyRule(
        rule_id='read_service_name', tools=['read_config'],
        allowed_principals=['demo-agent'], allowed_tenants=['demo'],
        allowed_scopes=['config:read'], argument_limits={'key': ['service_name']},
        outcome=DecisionOutcome.ALLOW, risk=RiskLevel.READ_ONLY,
    )])
    outcomes = {}
    with ActionLedger(output / 'audit.sqlite', require_durable=True) as ledger:
        guard = SystemOneGuard(policy=policy, ledger=ledger, signing_key=key, enforcement_profile=True)
        for config_key in ['service_name', 'private_key']:
            proposal = ActionProposal.create(
                tenant_id='demo', principal_id='demo-agent', scope='config:read',
                tool='read_config', arguments={'key': config_key},
                canonical_target=f'config:{config_key}', purpose='Demonstrate explicit permissions',
            )
            auth = guard.evaluate_proposal(proposal)
            outcomes[config_key] = auth.outcome.value
            if auth.outcome == DecisionOutcome.ALLOW:
                value = {'service_name': 'system1-demo'}[config_key]
                ledger.record_execution_outcome(
                    action_id=proposal.action_id, receipt_digest=auth.receipt.compute_digest(),
                    status='SUCCEEDED', result_payload={'value': value}, tenant_id='demo',
                    principal_id='demo-agent', scope='config:read', trusted_public_key=key.public_key(),
                )
        assert ledger.verify_integrity(trusted_public_key=key.public_key())
    print(f"\nExplicit policy outcomes: {outcomes}; signed ledger verified.")
    print('The taught triage skill is not used to grant these permissions.')
    return outcomes


def main(argv=None):
    report = run_example(OperationTriageSchema, 'agent_guard', argv)
    demonstrate_policy(Path(report['skill_path']).parent)
    return report


if __name__ == '__main__':
    main()
