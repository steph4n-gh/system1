#!/usr/bin/env python3
"""System 1 vs TypeSafe AI (Jev): Side-by-Side Killer Use Cases Live Benchmark.

Implements 5 canonical killer use cases highlighted by TypeSafe AI:
1. Real-Time Smart Home Assistant (Speculative Fan-Out)
2. Financial Crime & AML Alert Prioritization (FinTech KYC/SAR)
3. Semantic Code Linting & CI Security Gating
4. RAG Context Re-ranking & Passage Relevance Verification
5. Insurance Claims Triage & Straight-Through Processing (STP)

Runs live against both systems, capturing exact inputs, outputs, confidences,
conformal sets, latency, egress, and cryptographic receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
    SystemOneEngine,
    ScoreField,
)
from system1.ledger import ActionLedger
from system1.receipt import verify_decision_witness_receipt


# ============================================================================
# Schemas for Killer Use Cases
# ============================================================================

# 1. Smart Home Assistant (Speculative Fan-out)
class SmartHomeSchema(DecisionSchema):
    category = ChoiceField(
        options=["smarthome_command", "information_query", "system_settings"],
        descriptions={
            "smarthome_command": "Direct instruction to control physical smart home hardware or devices",
            "information_query": "General knowledge, weather, time, or conversational question",
            "system_settings": "Wi-Fi, volume, account, or assistant configuration",
        },
    )
    target_domain = ChoiceField(
        options=["living_room", "kitchen", "bedroom", "garage", "whole_house"],
        descriptions={
            "living_room": "Living room, television, couch area lights",
            "kitchen": "Kitchen lights, oven, coffee machine, refrigerator",
            "bedroom": "Master bedroom, guest bedroom, bedside lamps",
            "garage": "Garage door, exterior floodlights, workshop",
            "whole_house": "All rooms, entire residence, all devices",
        },
    )
    device_type = ChoiceField(
        options=["lights", "thermostat", "locks", "blinds", "music"],
        descriptions={
            "lights": "Light bulbs, lamps, ceiling fixtures, LED strips",
            "thermostat": "Temperature, heating, HVAC, AC, climate control",
            "locks": "Smart deadbolts, front door locks, garage entry",
            "blinds": "Window shades, roller blinds, curtains",
            "music": "Speakers, Spotify, volume, playlist",
        },
    )
    action = ChoiceField(
        options=["turn_on", "turn_off", "adjust_level", "lock", "unlock"],
        descriptions={
            "turn_on": "Power on, activate, illuminate, open",
            "turn_off": "Power off, shut down, extinguish, deactivate",
            "adjust_level": "Dim, brighten, raise temperature, change volume",
            "lock": "Engage lock, secure bolt",
            "unlock": "Disengage lock, unbolt door",
        },
    )
    is_compound_request = BooleanField(
        threshold=0.5,
        true_description="Request asks for multiple distinct actions requiring splitting",
        false_description="Single atomic action request",
    )


# 2. Financial Crime / AML Suspicious Activity
class FinancialCrimeSchema(DecisionSchema):
    risk_classification = ChoiceField(
        options=["normal", "structuring_suspect", "sanctions_nexus", "unusual_velocity", "high_risk_jurisdiction"],
        descriptions={
            "normal": "Standard payroll, retail purchases, or regular recurring business payments",
            "structuring_suspect": "Multiple sequential transactions just below $10,000 reporting threshold",
            "sanctions_nexus": "Counterparties, banks, or routing nodes associated with OFAC sanctioned entities",
            "unusual_velocity": "Rapid automated fund movement rapidly depositing and emptying new accounts",
            "high_risk_jurisdiction": "Cross-border transfers to non-cooperative jurisdictions or secrecy havens",
        },
    )
    action_required = ChoiceField(
        options=["clear", "monitor", "escalate_compliance", "freeze_account"],
        descriptions={
            "clear": "Transaction conforms to expected account profile",
            "monitor": "Flag for periodic 30-day account monitoring review",
            "escalate_compliance": "Assign to AML compliance analyst for mandatory investigation",
            "freeze_account": "Immediate automated hold pending law enforcement or SAR filing",
        },
    )
    file_sar = BooleanField(
        threshold=0.5,
        true_description="Sufficient suspicion of illicit origin requiring mandatory FinCEN Suspicious Activity Report (SAR)",
        false_description="Insufficient evidence for regulatory SAR filing",
    )
    risk_score = ScoreField(
        min_value=0.0,
        max_value=3.0,
        low_description="Zero financial crime risk",
        high_description="Definitive money laundering, terror financing, or sanctions evasion",
    )


# 3. Semantic Code Linting & CI Security Gating
class CodeSecurityLintSchema(DecisionSchema):
    security_verdict = ChoiceField(
        options=["pass", "warning", "critical_block"],
        descriptions={
            "pass": "Code meets modern secure coding guidelines",
            "warning": "Code contains minor code smell, deprecation, or non-critical inefficiency",
            "critical_block": "Code introduces critical remote code execution, SQL injection, or secret leakage",
        },
    )
    vulnerability_class = ChoiceField(
        options=["none", "sql_injection", "hardcoded_secret", "path_traversal", "unsafe_deserialization"],
        descriptions={
            "none": "No known security vulnerability present",
            "sql_injection": "Unsanitized user string concatenation directly inside database queries",
            "hardcoded_secret": "API keys, private tokens, passwords, or certificates stored in plain text",
            "path_traversal": "Arbitrary file reading via unvalidated relative directory paths ../",
            "unsafe_deserialization": "Loading untrusted pickle, yaml, or object payloads directly into runtime",
        },
    )
    block_ci_merge = BooleanField(
        threshold=0.5,
        true_description="Pull request must be immediately blocked from merging",
        false_description="Pull request can proceed with automated merge",
    )
    severity_score = ScoreField(
        min_value=0.0,
        max_value=3.0,
        low_description="Harmless or clean code diff",
        high_description="Critical severity CVSS 9.0+ vulnerability",
    )


# 4. RAG Search & Retrieval Reranker
class RAGRerankerSchema(DecisionSchema):
    relevance_tier = ChoiceField(
        options=["irrelevant", "contextually_related", "direct_answer"],
        descriptions={
            "irrelevant": "Passage has no bearing on the user query",
            "contextually_related": "Passage discusses the general subject area but does not answer the specific query",
            "direct_answer": "Passage provides the exact facts, figures, or explanation answering the query",
        },
    )
    sufficient_for_generation = BooleanField(
        threshold=0.5,
        true_description="Passage provides complete factual ground truth allowing LLM to synthesize response",
        false_description="Passage is incomplete or requires additional external search context",
    )
    fact_density_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="Fluff, conversational filler, or marketing copy with zero hard facts",
        high_description="Dense factual documentation, specifications, or numerical data",
    )


# 5. Insurance Claim Triage & Straight-Through Processing (STP)
class InsuranceClaimSchema(DecisionSchema):
    triage_queue = ChoiceField(
        options=["straight_through_approval", "desk_adjuster_review", "special_investigations_unit", "urgent_catastrophe"],
        descriptions={
            "straight_through_approval": "Low-value standard claim with complete documentation eligible for instant payout",
            "desk_adjuster_review": "Moderate damage requiring manual policy verification and repair quote validation",
            "special_investigations_unit": "High fraud risk, staged accident indicators, or falsified invoices",
            "urgent_catastrophe": "Total structural destruction, bodily injury, or emergency displacement",
        },
    )
    potential_fraud = BooleanField(
        threshold=0.5,
        true_description="Indicators of fraudulent claim, pre-existing damage, or inconsistent timelines",
        false_description="Genuine substantiated claim",
    )
    fraud_risk_score = ScoreField(
        min_value=0.0,
        max_value=3.0,
        low_description="Completely clean legitimate claimant record",
        high_description="Known fraud syndicate signature or blatant fabrication",
    )


# ============================================================================
# API Helper
# ============================================================================

def call_jev_api(prompt: str, questions: Dict[str, Any], api_key: str) -> Tuple[Optional[Dict[str, Any]], float, int]:
    payload = {"model": "jev-latest", "state": prompt, "questions": questions}
    raw_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=raw_bytes,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.perf_counter() - t0) * 1000.0
            return data, elapsed, len(raw_bytes)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"    [!] Warning: Jev API request failed ({type(e).__name__}: {e})")
        return None, elapsed, len(raw_bytes)


# ============================================================================
# Test Runner
# ============================================================================

def run_use_case_tests(api_key: str):
    signing_key = Ed25519PrivateKey.generate()
    ledger = ActionLedger(":memory:")

    scenarios = [
        {
            "category": "1. REAL-TIME SMART HOME ASSISTANT (Speculative Fan-out)",
            "schema_cls": SmartHomeSchema,
            "prompt": "Turn off all the lights in the house, we are going to sleep.",
            "jev_questions": {
                "category": {
                    "type": "choice",
                    "instructions": "What category of request is this?",
                    "criteria": {
                        "smarthome_command": "Direct instruction to control physical hardware",
                        "information_query": "General knowledge or weather question",
                        "system_settings": "Assistant settings or Wi-Fi",
                    },
                },
                "target_domain": {
                    "type": "choice",
                    "instructions": "What room or domain is targeted?",
                    "criteria": {
                        "living_room": "Living room",
                        "kitchen": "Kitchen",
                        "bedroom": "Bedroom",
                        "garage": "Garage",
                        "whole_house": "All rooms or entire residence",
                    },
                },
                "device_type": {
                    "type": "choice",
                    "instructions": "What type of device is targeted?",
                    "criteria": {
                        "lights": "Light bulbs and lamps",
                        "thermostat": "Temperature or HVAC",
                        "locks": "Door deadbolts",
                        "blinds": "Window shades",
                        "music": "Speakers and audio",
                    },
                },
                "action": {
                    "type": "choice",
                    "instructions": "What action should be taken?",
                    "criteria": {
                        "turn_on": "Power on or illuminate",
                        "turn_off": "Power off or extinguish",
                        "adjust_level": "Dim, brighten or change level",
                        "lock": "Lock door",
                        "unlock": "Unlock door",
                    },
                },
                "is_compound_request": {
                    "type": "noul",
                    "instructions": "Is this a compound request requiring splitting?",
                    "criteria": {"true": "Multiple actions", "false": "Single action"},
                },
            },
        },
        {
            "category": "2. FINANCIAL CRIME & AML ALERT TRIAGE (FinTech / Banking)",
            "schema_cls": FinancialCrimeSchema,
            "prompt": "Customer initiated 4 consecutive cash deposits of $9,850 within 48 hours across 3 branches, followed by immediate wire to an offshore entity.",
            "jev_questions": {
                "risk_classification": {
                    "type": "choice",
                    "instructions": "What financial crime typology is present?",
                    "criteria": {
                        "normal": "Normal legitimate business activity",
                        "structuring_suspect": "Multiple sequential transactions just below $10,000 threshold",
                        "sanctions_nexus": "Counterparties with sanctions nexus",
                        "unusual_velocity": "Rapid automated fund movement",
                        "high_risk_jurisdiction": "Transfers to secrecy havens",
                    },
                },
                "action_required": {
                    "type": "choice",
                    "instructions": "What compliance action is required?",
                    "criteria": {
                        "clear": "Clear transaction",
                        "monitor": "Flag for periodic monitoring",
                        "escalate_compliance": "Assign to AML investigator",
                        "freeze_account": "Immediate hold pending SAR filing",
                    },
                },
                "file_sar": {
                    "type": "noul",
                    "instructions": "Does this warrant filing a Suspicious Activity Report (SAR)?",
                    "criteria": {"true": "Mandatory SAR filing", "false": "No SAR needed"},
                },
                "risk_score": {
                    "type": "score",
                    "instructions": "Assess overall money laundering risk level",
                    "criteria": ["Zero risk", "Low risk", "High risk", "Critical AML violation"],
                },
            },
        },
        {
            "category": "3. SEMANTIC CODE LINTING & CI SECURITY GATING (DevSecOps)",
            "schema_cls": CodeSecurityLintSchema,
            "prompt": "query = f\"SELECT * FROM users WHERE email = '{user_input_email}' AND role = 'admin'\"\ndb.execute(query)",
            "jev_questions": {
                "security_verdict": {
                    "type": "choice",
                    "instructions": "What is the CI security gating verdict?",
                    "criteria": {
                        "pass": "Meets secure coding standards",
                        "warning": "Minor smell or inefficiency",
                        "critical_block": "Critical vulnerability requiring build failure",
                    },
                },
                "vulnerability_class": {
                    "type": "choice",
                    "instructions": "What vulnerability class is present?",
                    "criteria": {
                        "none": "No vulnerability",
                        "sql_injection": "Unsanitized user string in SQL query",
                        "hardcoded_secret": "Plaintext secret or key",
                        "path_traversal": "Path traversal ../ vulnerability",
                        "unsafe_deserialization": "Unsafe pickle or yaml loading",
                    },
                },
                "block_ci_merge": {
                    "type": "noul",
                    "instructions": "Should this PR merge be blocked immediately?",
                    "criteria": {"true": "Block merge", "false": "Allow merge"},
                },
                "severity_score": {
                    "type": "score",
                    "instructions": "Rate vulnerability severity",
                    "criteria": ["Informational", "Low", "Medium", "Critical CVSS 9+"],
                },
            },
        },
        {
            "category": "4. RAG SEARCH & RETRIEVAL RE-RANKING (Context Verification)",
            "schema_cls": RAGRerankerSchema,
            "prompt": "User Query: 'What is the maximum battery life of the Model X headphone?'\nRetrieved Passage: 'The Model X headphones offer up to 40 hours of continuous active playback on a single charge, or 28 hours with Active Noise Cancellation enabled.'",
            "jev_questions": {
                "relevance_tier": {
                    "type": "choice",
                    "instructions": "How relevant is this passage to the user query?",
                    "criteria": {
                        "irrelevant": "No bearing on user query",
                        "contextually_related": "Discusses topic generally but lacks answer",
                        "direct_answer": "Provides exact facts answering query",
                    },
                },
                "sufficient_for_generation": {
                    "type": "noul",
                    "instructions": "Is passage sufficient to answer without external search?",
                    "criteria": {"true": "Sufficient factual ground truth", "false": "Needs more context"},
                },
                "fact_density_score": {
                    "type": "score",
                    "instructions": "Rate factual density of the passage",
                    "criteria": ["Conversational filler", "General description", "Dense numerical facts"],
                },
            },
        },
        {
            "category": "5. INSURANCE CLAIM AUTOMATED TRIAGE (Straight-Through Processing)",
            "schema_cls": InsuranceClaimSchema,
            "prompt": "Claim #99104: Policyholder reports a cracked windshield caused by a stray pebble on Highway 101. Certified repair shop quote attached: $320. No injuries, police report not required, driver has 12-year clean history.",
            "jev_questions": {
                "triage_queue": {
                    "type": "choice",
                    "instructions": "Which claims triage queue should this be routed to?",
                    "criteria": {
                        "straight_through_approval": "Low-value substantiated claim eligible for instant payout",
                        "desk_adjuster_review": "Moderate claim requiring manual review",
                        "special_investigations_unit": "High fraud risk indicators",
                        "urgent_catastrophe": "Total structural destruction or bodily injury",
                    },
                },
                "potential_fraud": {
                    "type": "noul",
                    "instructions": "Are there indicators of insurance fraud?",
                    "criteria": {"true": "Fraud indicators present", "false": "Genuine claim"},
                },
                "fraud_risk_score": {
                    "type": "score",
                    "instructions": "Assess fraud risk score",
                    "criteria": ["Clean legitimate claimant", "Low suspicion", "Moderate suspicion", "Definitive fraud signature"],
                },
            },
        },
    ]

    print("\n" + "=" * 80)
    print("   TYPESAFE AI (JEV) vs. REFLEX SYSTEM 1: KILLER USE CASES SHOWCASE")
    print("=" * 80)
    print(f"Jev Model:      jev-latest (via api.typesafe.ai/v1/systemone)")
    print(f"System 1 Engine:  SystemOneModel (Apple Silicon Metal / NumPy BLAS)")
    print("=" * 80 + "\n")

    summary_records = []

    for i, scen in enumerate(scenarios, 1):
        cat_title = scen["category"]
        prompt_text = scen["prompt"]

        print(f"\n{'#' * 80}")
        print(f"USE CASE {i}: {cat_title}")
        print(f"{'#' * 80}")
        print(f"[INPUT STATE / PROMPT]:\n{prompt_text}\n")

        # 1. Execute on TypeSafe AI (Jev)
        jev_resp, jev_lat, egress_bytes = call_jev_api(prompt_text, scen["jev_questions"], api_key)
        if not jev_resp or not jev_resp.get("answers"):
            print("    Comparison unavailable: failed teacher response excluded.")
            continue

        # 2. Execute on System 1 System 1
        engine = SystemOneEngine(scen["schema_cls"], signing_key=signing_key, ledger=ledger, backend="auto")
        # Warmup
        _ = engine.decide("warmup", record_receipt=False)

        t0 = time.perf_counter()
        system1_res = engine.decide(prompt_text, alpha=0.05, record_receipt=True)
        system1_lat = (time.perf_counter() - t0) * 1000.0

        receipt_dict = system1_res.receipt.to_dict()
        receipt_valid = verify_decision_witness_receipt(receipt_dict, public_key=signing_key.public_key())
        speedup = jev_lat / system1_lat if system1_lat > 0 else 0.0

        # Print Side-by-Side Outputs
        print("-" * 80)
        print(f"  [A] TYPESAFE AI (JEV) OUTPUT  (Latency: {jev_lat:.2f} ms | Egress: {egress_bytes} B)")
        print("-" * 80)
        if jev_resp and "answers" in jev_resp:
            for q_name, ans in jev_resp["answers"].items():
                if ans.get("type") == "choice":
                    print(f"    • {q_name:<25}: {ans.get('choice')} (Confidence: {ans.get('confidence', 0):.1%})")
                elif ans.get("type") == "noul":
                    print(f"    • {q_name:<25}: p={ans.get('noul'):.2f}")
                elif ans.get("type") == "score":
                    print(f"    • {q_name:<25}: score={ans.get('score'):.2f} (Confidence: {ans.get('confidence', 0):.1%})")
            usage = jev_resp.get("usage", {})
            print(f"    -> Token Cost: {usage.get('input_tokens', 0)} input + {usage.get('output_tokens', 0)} output = {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} billed tokens")
            print(f"    -> Cryptographic Proof: NONE (ephemeral JSON)")
        else:
            print(f"    -> ERROR calling Jev API: {jev_resp}")

        print("\n" + "-" * 80)
        print(f"  [B] REFLEX SYSTEM 1 OUTPUT    (Latency: {system1_lat:.3f} ms | Egress: 0 B)")
        print("-" * 80)
        for field_name, field_val in system1_res.values.items():
            conf = system1_res.confidences.get(field_name, 0.0)
            cset = system1_res.conformal_sets.get(field_name, [])
            print(f"    • {field_name:<25}: {field_val} (Confidence: {conf:.1%}, Conformal Set: {cset})")
        print(f"    -> Token Cost: $0.00 (0 tokens)")
        print(f"    -> Ed25519 Receipt Digest: {system1_res.receipt.digest[:24]}...")
        print(f"    -> Cryptographic Non-Repudiation Verified: {receipt_valid}")
        print(f"    -> ActionLedger Entry: SHA-256 Hash Chained to SQLite")

        print(f"\n  >>> SPEEDUP ON THIS WORKLOAD: {speedup:.1f}x FASTER <<<")

        summary_records.append({
            "use_case": f"UC-{i}",
            "name": cat_title.split(":")[0],
            "jev_latency": jev_lat,
            "system1_latency": system1_lat,
            "speedup": speedup,
            "receipt_valid": receipt_valid,
        })

    if not summary_records:
        print("No successful paired API responses; no speedup can be reported.")
        return []

    # Summary Table
    print("\n" + "=" * 90)
    print("                     FINAL SIDE-BY-SIDE SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Use Case':<10} | {'Jev Cloud (ms)':<15} | {'System 1 Metal (ms)':<18} | {'Speedup':<12} | {'Ed25519 Verified'}")
    print("-" * 90)
    for r in summary_records:
        print(f"{r['use_case']:<10} | {r['jev_latency']:>10.2f} ms   | {r['system1_latency']:>12.3f} ms    | {r['speedup']:>8.1f}x   | {str(r['receipt_valid']):<16}")
    print("=" * 90)

    mean_j = np.mean([r["jev_latency"] for r in summary_records])
    mean_r = np.mean([r["system1_latency"] for r in summary_records])
    print(f"Average Jev Latency:     {mean_j:.2f} ms")
    print(f"Average System 1 Latency:  {mean_r:.3f} ms")
    print(f"Overall Average Speedup: {mean_j / mean_r:.1f}x FASTER\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TYPESAFE_API_KEY", ""),
    )
    args = parser.parse_args()
    run_use_case_tests(args.api_key)


if __name__ == "__main__":
    main()
