#!/usr/bin/env python3
"""Reflex Standalone Example: Customer Support Ticket Triage.

Demonstrates real-time customer support routing and sentiment triage,
deciding routing department, urgency, human escalation, and customer frustration
with sub-2ms latency.

Showcases three integration paradigms:
1. Native Reflex Declarative Schema (DecisionSchema + ReflexEngine)
2. TypeSafe SDK Drop-in Import Swap (from system1.compat.typesafe import TypeSafeClient, Choice, Noul, Score)
3. Zero-Code-Change Monkey Patching (patch_typesafe() -> import typesafe_sdk)
4. Live Side-by-Side Comparison vs TypeSafe Cloud (or realistic WAN baseline fallback)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import system1
from system1 import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    ReflexEngine,
    ScoreField,
)
from system1.compat.typesafe import (
    Choice,
    Noul,
    Score,
    TypeSafeClient,
    patch_typesafe,
)


# ============================================================================
# 1. Native Reflex Schema
# ============================================================================

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


def run_native_mode(incoming_tickets: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 1: NATIVE REFLEX DECISION ENGINE (DecisionSchema + ReflexEngine)")
    print("=" * 76)

    engine = ReflexEngine(SupportTicketSchema, backend="auto")

    print("\nProcessing Incoming Tickets:")
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


# ============================================================================
# 2. TypeSafe SDK Drop-in Import Swap
# ============================================================================

def run_typesafe_sdk_dropin_mode(incoming_tickets: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 2: TYPESAFE SDK DROP-IN IMPORT SWAP")
    print("  (from system1.compat.typesafe import TypeSafeClient, Choice, Noul, Score)")
    print("=" * 76)

    questions = {
        "department": Choice(
            "Which department should handle this ticket?",
            criteria={
                "billing": "Invoices, credit card charges, refund requests, pricing questions",
                "technical_support": "System bugs, 502 bad gateway, crash reports, API integration errors",
                "account_security": "Password reset, compromised account, two-factor authentication",
                "sales": "Enterprise contract inquiries, custom plans, volume quotes",
            },
        ),
        "priority": Choice(
            "What is the priority level of this issue?",
            criteria=["low", "medium", "high", "critical"],
        ),
        "needs_human_escalation": Noul(
            "Does this ticket need human escalation?",
            criteria={"true": "Angry customer, high financial dispute, or critical outage", "false": "Standard inquiry"},
        ),
        "frustration_index": Score(
            "Rate customer frustration from 0 (calm) to 1 (furious)",
            min_value=0.0,
            max_value=1.0,
        ),
    }

    client = TypeSafeClient()

    print("\nProcessing via TypeSafeClient.systemone():")
    for ticket in incoming_tickets:
        t0 = time.perf_counter()
        resp = client.systemone(ticket, questions, alpha=0.05)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[Ticket]: \"{ticket}\"")
        print(f"  Latency:            {latency_ms:.3f} ms")
        print(f"  Department:         {resp.answers.department.choice} (conf: {resp.answers.department.confidence:.1%})")
        print(f"  Priority:           {resp.answers.priority.choice}")
        print(f"  Escalation:         {resp.answers.needs_human_escalation.value} (p={resp.answers.needs_human_escalation.noul:.2f})")
        print(f"  Frustration:        {resp.answers.frustration_index.score:.3f}")
        print(f"  Conformal Set (95%):{resp.answers.department.conformal_set}")
        print(f"  Receipt ID:         {resp.receipt.receipt_id}")


# ============================================================================
# 3. Monkey Patching Mode (Zero-Code-Change)
# ============================================================================

def run_monkey_patch_mode(incoming_tickets: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 3: ZERO-CODE-CHANGE MONKEY PATCHING (patch_typesafe())")
    print("=" * 76)

    with patch_typesafe():
        import typesafe_sdk
        from typesafe_sdk import Choice as TChoice, Noul as TNoul, TypeSafeClient as TSClient

        print(f"Imported typesafe_sdk successfully: {typesafe_sdk.__name__}")
        client = TSClient()

        q = {
            "dept": TChoice("Department", criteria=["billing", "technical_support", "sales"]),
            "urgent": TNoul("Is urgent?"),
        }

        for ticket in incoming_tickets[:2]:
            res = client.systemone(ticket, q)
            print(f"  • Ticket: \"{ticket[:45]}...\"")
            print(f"    -> Dept: {res.answers.dept.choice}, Urgent: {res.answers.urgent.value}, Latency: {res.latency_ms:.3f} ms")


# ============================================================================
# 4. Live Side-by-Side Comparison (Local vs TypeSafe Cloud / Baseline)
# ============================================================================

def run_side_by_side_comparison(incoming_tickets: list[str]):
    print("\n" + "=" * 76)
    print("  MODE 4: LIVE HEAD-TO-HEAD COMPARISON (Reflex System 1 vs TypeSafe Cloud)")
    print("=" * 76)

    api_key = os.environ.get("TYPESAFE_API_KEY", "") or os.environ.get("JEV_API_KEY", "")
    if api_key:
        print(f"Targeting real TypeSafe API with key: [***{api_key[-4:]}]")
    else:
        print("No TYPESAFE_API_KEY found. Measuring WAN socket RTT and realistic baseline profile.")

    client = TypeSafeClient(api_key=api_key)

    questions = {
        "department": Choice("Department", criteria=["billing", "technical_support", "account_security", "sales"]),
        "needs_escalation": Noul("Needs escalation?"),
        "frustration": Score("Frustration score", min_value=0.0, max_value=1.0),
    }

    print("\n" + "-" * 92)
    print(f"{'Ticket Summary':<42} | {'Local (ms)':<12} | {'Cloud (ms)':<12} | {'Speedup':<10} | {'Egress Diff'}")
    print("-" * 92)

    for ticket in incoming_tickets:
        comp = client.compare(ticket, questions)
        short_t = ticket[:40] + ".." if len(ticket) > 42 else ticket
        print(
            f"{short_t:<42} | "
            f"{comp.local_latency_ms:>8.3f} ms | "
            f"{comp.cloud_latency_ms:>8.2f} ms | "
            f"{comp.speedup_factor:>7.1f}x  | "
            f"0 B vs {comp.cloud_egress_bytes} B"
        )
    print("-" * 92)


def main():
    incoming_tickets = [
        "I was charged $499 twice on my corporate Visa for invoice #88219!",
        "How do I reset my password? I forgot it.",
        "URGENT: Our production cluster is down and returning 502 Bad Gateway to all users!",
        "We want to purchase 500 enterprise seats for our engineering division next quarter.",
    ]

    print("#" * 76)
    print("  REFLEX SYSTEM 1: CUSTOMER SUPPORT TICKET TRIAGE SHOWCASE")
    print("#" * 76)

    run_native_mode(incoming_tickets)
    run_typesafe_sdk_dropin_mode(incoming_tickets)
    run_monkey_patch_mode(incoming_tickets)
    run_side_by_side_comparison(incoming_tickets)

    print("\n" + "=" * 76)
    print("  SUPPORT TICKET TRIAGE DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
