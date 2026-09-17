#!/usr/bin/env python3
"""Reflex Standalone Example: Customer Support Ticket Triage.

Demonstrates real-time customer support routing and sentiment triage,
deciding routing department, urgency, human escalation, and customer frustration
with sub-2ms latency.
"""

import time
from reflex import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    ReflexEngine,
    ScoreField,
)


class SupportTicketSchema(DecisionSchema):
    """Schema for automated support ticket triage."""

    department = ChoiceField(
        options=["billing", "technical_support", "account_security", "sales"],
        descriptions={
            "billing": "Invoices, credit card charges, refund requests, pricing questions",
            "technical_support": "System bugs, 502 bad gateway, crash reports, API integration errors",
            "account_security": "Password reset, compromised account, two-factor authentication",
            "sales": "Enterprise contract inquiries, custom plans, volume quotes",
        },
    )
    priority = ChoiceField(
        options=["low", "medium", "high", "critical"],
        descriptions={
            "low": "Minor cosmetic question or non-urgent inquiry",
            "medium": "Standard question with workaround available",
            "high": "Major functionality degraded or financial discrepancy",
            "critical": "Production outage, data loss, active security breach",
        },
    )
    needs_human_escalation = BooleanField(
        threshold=0.5,
        true_description="Angry customer, high financial dispute, or critical outage requiring human intervention",
        false_description="Standard inquiry solvable by automated response",
    )
    frustration_index = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="Calm and cooperative customer tone",
        high_description="Extremely frustrated or furious customer demanding executive attention",
    )


def main():
    print("Initializing Reflex Support Ticket Triage Engine...")
    engine = ReflexEngine(SupportTicketSchema, backend="auto")

    incoming_tickets = [
        "I was charged $499 twice on my corporate Visa for invoice #88219!",
        "How do I reset my password? I forgot it.",
        "URGENT: Our production cluster is down and returning 502 Bad Gateway to all users!",
        "We want to purchase 500 enterprise seats for our engineering division next quarter.",
    ]

    print("\nProcessing Incoming Tickets:")
    print("=" * 72)

    for ticket in incoming_tickets:
        t0 = time.perf_counter()
        result = engine.decide(ticket, alpha=0.05)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[Ticket]: \"{ticket}\"")
        print(f"  Latency:            {elapsed_ms:.3f} ms")
        print(f"  Department:         {result.department} (Confidence: {result.confidences['department']:.1%})")
        print(f"  Priority:           {result.priority} (Confidence: {result.confidences['priority']:.1%})")
        print(f"  Human Escalation:   {result.needs_human_escalation}")
        print(f"  Frustration Index:  {result.frustration_index:.3f}")
        print(f"  Receipt ID:         {result.receipt.receipt_id}")


if __name__ == "__main__":
    main()
