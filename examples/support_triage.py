#!/usr/bin/env python3
"""Teach one skill: route a single-issue support ticket to its department.

Run: python examples/support_triage.py
Edit the explicit examples in examples/teaching/support_triage.json to teach
another routing policy. No account, network access, or LLM is needed.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from system1 import ChoiceField, DecisionSchema
from _teaching_demo import run_example


class SupportTicketSchema(DecisionSchema):
    department = ChoiceField(
        options=['billing', 'technical_support', 'account_security', 'sales'],
        descriptions={
            'billing': 'Existing payments, invoices, refunds, subscriptions, and billing records',
            'technical_support': 'Product defects, errors, integrations, performance, and installation',
            'account_security': 'Login credentials, identity verification, account access, and suspicious sign-ins',
            'sales': 'Prospective purchases, demos, enterprise quotes, and commercial contracts',
        },
    )


def main(argv=None):
    return run_example(SupportTicketSchema, 'support_triage', argv)


if __name__ == '__main__':
    main()
