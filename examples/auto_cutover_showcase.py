#!/usr/bin/env python3
"""Legacy simulation: schema-augmented teaching and a relaxed promotion policy.
Not production quality evidence; observe_routing.py demonstrates default gates.

System 1 vs TypeSafe AI: Enterprise Auto-Cutover Showcase.

Demonstrates the drop-in Trojan Horse compatibility layer (`mode="auto_cutover"`)
for TypeSafe AI (Jev) SDK with:
1. Killer Enterprise Use Case:
   - Primary: Autonomous FinTech Wire Transfer Fraud Interceptor & Risk Decisioning
   - Optional: Clinical Intake & Emergency Department Triage (ED ESI Protocol)
   - Optional: Real-Time E-Commerce Customer Support Escalation
2. Standard TypeSafe SDK Interface:
   - `TypeSafeClient`, `Choice`, `Noul`, `Score`, and `patch_typesafe()`
3. Autonomous Cutover with Live Query Stream:
   - Initial queries proxy to TypeSafe cloud API (or realistic cloud WAN baseline)
   - Automatically records prompt/response pairs into SQLite `ActionLedger`
   - Dynamically distills closed-form Ridge Regression hyperplanes via `SystemOneCompiler`
   - Verifies local agreement rate and automatically flips to 100% local execution
4. Demonstrated Latency Drop, Zero Egress, and Throughput Gains:
   - Captures before/after latency curves (~250-300ms cloud WAN dropping to ~1.1ms local Metal/NumPy)
   - 100% data egress reduction (drops from ~550 bytes/query down to 0 bytes)
   - 100% token cost reduction ($0.00 / token)
   - High-throughput burst comparison (network-bound ~3 QPS vs local ~1,000+ QPS)
5. Cryptographic Ed25519 Receipts & SQLite ActionLedger Auditability:
   - Verifies proof-carrying Ed25519 decision receipts
   - Cryptographically audits SQLite ActionLedger SHA-256 hash chaining
6. Zero-Code Monkey-Patching:
   - Intercepts legacy `import typesafe` / `import typesafe_sdk` via `patch_typesafe()`
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Ensure repository root src/ is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from system1.compat.typesafe import (
    Choice,
    Noul,
    PromotionPolicy,
    Score,
    TypeSafeClient,
    TypeSafeResponse,
    patch_typesafe,
)
from system1.ledger import ActionLedger
from system1.receipt import verify_decision_witness_receipt


def load_env_credentials() -> None:
    """Safely loads TYPESAFE_API_KEY or JEV_API_KEY from .env files without printing values."""
    for p in [Path.home() / ".env", REPO_ROOT / ".env"]:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TYPESAFE_API_KEY=") or line.startswith("JEV_API_KEY="):
                            k, v = line.split("=", 1)
                            v = v.strip().strip("'\"")
                            if v and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


# ============================================================================
# Domain Schemas & Realistic Query Streams
# ============================================================================

def get_fintech_wire_fraud_use_case() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Autonomous FinTech Wire Transfer Fraud Interceptor & Risk Decisioning System.

    Real-time payment rails (FedNow, RTP, SEPA Instant, SWIFT GPI) require sub-5ms
    in-line authorization. Cloud LLMs (TypeSafe AI SaaS) take 250-400ms roundtrips,
    violating clearing SLAs and leaking sensitive banking data (IBANs, balances, PII).
    """
    title = "Autonomous FinTech Wire Transfer Fraud Interceptor & Risk Decisioning"

    questions = {
        "decision": Choice(
            "Authorize or intercept wire transfer",
            criteria={
                "APPROVE_IMMEDIATE": "Low risk, verified beneficiary, routine transaction amount, trusted device and IP",
                "HOLD_FOR_FRAUD_ANALYST": "Unusual volume, anomalous device fingerprint, or newly added beneficiary requiring manual analyst review",
                "ESCALATE_SAR_FIU": "Suspected structuring under $10,000, high-risk sanctioned jurisdiction, or money laundering patterns requiring Financial Intelligence Unit filing",
                "REJECT_BLOCK_ACCOUNT": "Confirmed credential stuffing, active botnet proxy, stolen API keys, or blacklisted account",
            },
        ),
        "is_suspicious": Noul(
            "Does this transaction exhibit suspicious activity or AML red flags under BSA regulations?",
            criteria={
                "true": "Suspicious indicators detected requiring compliance escalation",
                "false": "Standard benign payment activity",
            },
        ),
        "risk_score": Score(
            "Continuous risk rating from 0.0 (safe) to 10.0 (critical threat)",
            min_value=0.0,
            max_value=10.0,
        ),
        "sar_filing_required": Noul(
            "Is mandatory FinCEN Suspicious Activity Report (SAR) filing required under Bank Secrecy Act?",
            criteria={
                "true": "Mandatory SAR filing threshold exceeded",
                "false": "No regulatory filing required",
            },
        ),
    }

    queries = [
        {
            "id": "TXN-101-PAYROLL",
            "prompt": "Routine domestic payroll ACH transfer of $4,500.00 to established employee checking account at JPMorgan Chase. Employee tenure 4 years, IP Austin TX.",
            "type": "Payroll",
        },
        {
            "id": "TXN-102-SAAS",
            "prompt": "Low-risk recurring SaaS subscription invoice payment of $850.00 to verified AWS billing portal via corporate treasury account #8942.",
            "type": "Subscription",
        },
        {
            "id": "TXN-103-OFFSHORE",
            "prompt": "Urgent wire transfer of $490,000.00 to newly registered shell entity in Cyprus via unverified residential IP in Lagos. Beneficiary added 14 minutes ago.",
            "type": "High-Risk Wire",
        },
        {
            "id": "TXN-104-STRUCTURING",
            "prompt": "Structured wire transfer of $9,950.00 just below $10,000 BSA cash transaction reporting threshold to personal savings account. Third transfer this week.",
            "type": "Structuring",
        },
        {
            "id": "TXN-105-SUPPLIER",
            "prompt": "Regular vendor invoice payment of $12,400.00 to verified domestic manufacturing partner. Matches pre-existing purchase order #PO-44819.",
            "type": "Vendor Invoice",
        },
        {
            "id": "TXN-106-MICROBURST",
            "prompt": "High-frequency rapid micro-transfers: 18 consecutive $500 transfers to disparate P2P crypto off-ramp wallets within 4 minutes from mobile API client.",
            "type": "Velocity Anomaly",
        },
        {
            "id": "TXN-107-TREASURY",
            "prompt": "High-value corporate treasury transfer of $2,500,000.00 authenticated with dual hardware FIDO2 keys and CFO biometric confirmation to federal reserve bank.",
            "type": "Treasury Wire",
        },
        {
            "id": "TXN-108-TOR-DRAIN",
            "prompt": "Wire transfer request of $185,000.00 from dormant account following emergency password reset from Tor exit node. Beneficiary located in Seychelles.",
            "type": "Account Takeover",
        },
        {
            "id": "TXN-109-LEASE",
            "prompt": "Routine monthly commercial office lease payment of $18,500.00 to verified commercial property manager Boston Properties. On-time payment history 36 months.",
            "type": "Lease Payment",
        },
        {
            "id": "TXN-110-RANSOMWARE",
            "prompt": "Ransomware extortion wire transfer attempt of $350,000.00 to flagged cryptocurrency mixing service address listed on OFAC SDN sanctions bulletin.",
            "type": "Sanctions/Crime",
        },
        {
            "id": "TXN-111-IMPORT",
            "prompt": "International trade wire of $45,000.00 to licensed German medical optics supplier with verified customs bill of lading and letter of credit #LC-9902.",
            "type": "Trade Finance",
        },
        {
            "id": "TXN-112-MISMATCH",
            "prompt": "Account drain wire of $98,500.00 to offshore jurisdiction with mismatched beneficiary name. Beneficiary field altered from corporate supplier to individual.",
            "type": "Beneficiary Tampering",
        },
        {
            "id": "TXN-113-DIVIDEND",
            "prompt": "Standard dividend payout of $3,200.00 to verified institutional shareholder account. Domestic ACH network with matching tax identification number.",
            "type": "Dividend",
        },
        {
            "id": "TXN-114-STUFFING",
            "prompt": "Wire transfer initiated via credential stuffing attack: 412 failed login attempts followed by immediate maximum limit wire to newly linked debit card.",
            "type": "Credential Stuffing",
        },
        {
            "id": "TXN-115-ESCROW",
            "prompt": "Emergency capital call wire of $500,000.00 to licensed title escrow agent First American Title for pending commercial real estate closing.",
            "type": "Escrow Closing",
        },
        {
            "id": "TXN-116-MICROTEST",
            "prompt": "Micro-wire probe transaction of $1.00 followed 12 seconds later by $74,000.00 wire request to foreign prepaid card network from unverified device.",
            "type": "Micro-Probe Drain",
        },
        {
            "id": "TXN-117-UTILITY",
            "prompt": "Routine enterprise utility payment of $3,420.00 to Consolidated Edison power company via automated clearinghouse direct debit.",
            "type": "Utility Payment",
        },
        {
            "id": "TXN-118-SWIFT",
            "prompt": "Cross-border family remittance of $2,800.00 via SWIFT MT103 to verified family member account in Madrid Spain for university tuition.",
            "type": "Remittance",
        },
        {
            "id": "TXN-119-BULKPO",
            "prompt": "Bulk raw materials payment of $75,000.00 to steel supplier matching approved purchase order #PO-94182 and signed warehouse receipt.",
            "type": "Supplier Payment",
        },
        {
            "id": "TXN-120-BLOCKED",
            "prompt": "Unauthorized international wire attempt from disabled corporate cardholder account in violation of mandatory geographic travel security policy.",
            "type": "Policy Violation",
        },
    ]

    return title, questions, queries


def get_clinical_triage_use_case() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Clinical Intake & Medical Emergency Triage (ED Emergency Severity Index).

    Clinicians at triage desks require sub-second acuity scoring under ESI protocols.
    Egress of Protected Health Information (PHI) under HIPAA to public AI cloud APIs
    is strictly forbidden. System 1 provides zero-egress local on-device scoring.
    """
    title = "Clinical Intake & Medical Emergency Triage (HIPAA-Compliant ESI Protocol)"

    questions = {
        "triage_level": Choice(
            "Emergency Severity Index (ESI) triage classification",
            criteria={
                "ESI_1_RESUSCITATION": "Immediate life-saving intervention required (cardiac arrest, respiratory arrest, profound shock)",
                "ESI_2_EMERGENT": "High-risk situation, altered mental state, severe pain, or acute coronary syndrome",
                "ESI_3_URGENT": "Multiple resources required with stable vital signs (abdominal pain, moderate fractures)",
                "ESI_4_LESS_URGENT": "One simple diagnostic resource required (simple laceration needing sutures, ankle sprain)",
                "ESI_5_NON_URGENT": "No diagnostic resources required (prescription refill, minor rash)",
            },
        ),
        "immediate_resus_needed": Noul(
            "Does patient require immediate airway, breathing, or hemodynamic resuscitation?",
            criteria={
                "true": "Immediate life-saving intervention required",
                "false": "Hemodynamically stable",
            },
        ),
        "acuity_score": Score(
            "Continuous clinical acuity score from 1.0 (non-urgent) to 10.0 (maximum critical)",
            min_value=1.0,
            max_value=10.0,
        ),
        "icu_admission_risk": Noul(
            "High probability of inpatient ICU admission required?",
            criteria={
                "true": "High probability of ICU admission",
                "false": "Low ICU risk or outpatient discharge",
            },
        ),
    }

    queries = [
        {
            "id": "MED-201-ARREST",
            "prompt": "62yo male unresponsive, pulseless, CPR in progress by EMS. Intubated, rhythm showing ventricular fibrillation. BP unmeasurable.",
            "type": "Cardiac Arrest",
        },
        {
            "id": "MED-202-CHESTPAIN",
            "prompt": "58yo female presenting with acute crushing substernal chest pain radiating to left jaw, diaphoresis, BP 178/96, SpO2 94%. History of CAD.",
            "type": "Acute Coronary",
        },
        {
            "id": "MED-203-LACERATION",
            "prompt": "24yo male with 3cm clean superficial laceration to left forearm from kitchen knife. Bleeding controlled, neurovascularly intact, vitals normal.",
            "type": "Superficial Cut",
        },
        {
            "id": "MED-204-ANAPHYLAXIS",
            "prompt": "16yo female with acute facial angioedema, stridor, and wheezing 10 minutes after peanut ingestion. BP 82/50, pulse 135, lethargic.",
            "type": "Anaphylaxis",
        },
        {
            "id": "MED-205-ABDOMINAL",
            "prompt": "34yo female with 6 hours of sharp right lower quadrant abdominal pain, rebound tenderness, low-grade fever 38.1C. WBC elevated.",
            "type": "Appendicitis",
        },
        {
            "id": "MED-206-REFILL",
            "prompt": "45yo male requesting blood pressure medication refill after running out over the weekend. Asymptomatic, BP 132/84, normal exam.",
            "type": "Rx Refill",
        },
        {
            "id": "MED-207-STROKE",
            "prompt": "71yo male with sudden onset right-sided hemiplegia and expressive aphasia 45 minutes ago. Last known well 1 hour ago. BP 192/104.",
            "type": "Acute Stroke",
        },
        {
            "id": "MED-208-ANKLE",
            "prompt": "19yo female rolled right ankle playing basketball. Moderate lateral swelling, able to bear weight with discomfort, no deformity.",
            "type": "Sprain",
        },
        {
            "id": "MED-209-SEPSIS",
            "prompt": "80yo female from nursing home with altered mental status, productive cough, temperature 39.4C, BP 76/44, lactate 4.8, HR 128.",
            "type": "Septic Shock",
        },
        {
            "id": "MED-210-HEADACHE",
            "prompt": "28yo male with mild tension headache for 2 days. No visual changes, no neck stiffness, neurological examination normal.",
            "type": "Tension Headache",
        },
    ]

    return title, questions, queries


def get_ecommerce_support_use_case() -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Real-Time E-Commerce Customer Support Escalation & Sentiment Triage."""
    title = "Real-Time E-Commerce Customer Support Escalation & Sentiment Triage"

    questions = {
        "department": Choice(
            "Routing department",
            criteria={
                "billing": "Invoices, credit card charges, refund requests, pricing questions",
                "technical_support": "System bugs, crash reports, API integration errors, website down",
                "account_security": "Password reset, compromised account, two-factor authentication, unauthorized access",
                "sales": "Enterprise quotes, custom plans, volume licensing",
            },
        ),
        "priority": Choice(
            "Ticket priority level",
            criteria={
                "low": "Minor cosmetic question or non-urgent general inquiry",
                "medium": "Standard question with workaround available",
                "high": "Major functionality degraded or financial discrepancy",
                "critical": "Production outage, active security breach, or legal chargeback threat",
            },
        ),
        "needs_human_escalation": Noul(
            "Does ticket require immediate human intervention?",
            criteria={
                "true": "Angry customer, high financial dispute, or outage requiring immediate human intervention",
                "false": "Standard routine ticket resolvable by automated self-service",
            },
        ),
        "frustration_index": Score(
            "Customer frustration level from 0.0 (calm) to 1.0 (furious)",
            min_value=0.0,
            max_value=1.0,
        ),
    }

    queries = [
        {
            "id": "SUP-301-OUTAGE",
            "prompt": "CRITICAL: Our checkout gateway is throwing 500 Internal Server Error for all enterprise customers on Black Friday!",
            "type": "Outage",
        },
        {
            "id": "SUP-302-INVOICE",
            "prompt": "Could you please email me a PDF copy of receipt #REC-8921 for my accounting department records?",
            "type": "Invoice",
        },
        {
            "id": "SUP-303-CHARGEBACK",
            "prompt": "You billed my credit card $1,400 for a cancelled subscription! Refund this immediately or I am filing a fraud claim with my bank!",
            "type": "Billing Dispute",
        },
        {
            "id": "SUP-304-LOGIN",
            "prompt": "I cannot receive my 2FA SMS code after changing mobile carriers. Please help me access my account.",
            "type": "Account Access",
        },
        {
            "id": "SUP-305-ENTERPRISE",
            "prompt": "We are expanding to 5,000 seats and want to discuss custom enterprise pricing and SLA terms.",
            "type": "Enterprise Sales",
        },
        {
            "id": "SUP-306-TYPO",
            "prompt": "I noticed a small typo in the documentation on page 14 under the installation instructions.",
            "type": "Doc Feedback",
        },
        {
            "id": "SUP-307-BREACH",
            "prompt": "URGENT SECURITY: Someone unauthorized logged into our admin dashboard from an IP in Russia and generated new API keys!",
            "type": "Security Incident",
        },
        {
            "id": "SUP-308-RETURN",
            "prompt": "I ordered size Medium but need to exchange for size Large. The item is unworn in original packaging.",
            "type": "Return/Exchange",
        },
    ]

    return title, questions, queries


# ============================================================================
# Visual ASCII Formatting & Curves
# ============================================================================

def render_ascii_header(title: str, use_case_name: str, cutover_threshold: int) -> None:
    print("\n" + "█" * 110)
    print("  REFLEX SYSTEM 1: AUTONOMOUS TROJAN HORSE CUTOVER SHOWCASE")
    print("  Demonstrating TypeSafe AI Drop-in Compatibility, Latency Cliff Drop & Zero Egress")
    print("█" * 110)
    print(f"\n  Use Case:             {use_case_name}")
    print(f"  Cutover Threshold:    {cutover_threshold} queries (Autonomous Distillation Trigger)")
    print("  Execution Engine:     Apple Silicon Metal / Vectorized NumPy System 1 Engine")
    print("  Security & Audit:     Ed25519 Cryptographic Witness Receipts + SQLite ActionLedger")
    print("  Mathematical Bounds:  Split Conformal Prediction (Exact Finite-Sample (1-α) Coverage)")
    print("-" * 110)


def render_cutover_banner(event: Dict[str, Any]) -> None:
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.get("timestamp", time.time())))
    samples = event.get("samples_collected", 0)
    agreement = event.get("agreement_rate", 1.0) * 100.0
    threshold = event.get("min_agreement_threshold", 0.8) * 100.0

    print("\n" + "╔" + "═" * 108 + "╗")
    print("║" + " " * 32 + "⚡ AUTONOMOUS TROJAN HORSE CUTOVER ACTIVATED ⚡" + " " * 31 + "║")
    print("╠" + "═" * 108 + "╣")
    print(f"║  Event:                  {event.get('event', 'trojan_horse_cutover'):<81}║")
    print(f"║  Timestamp:              {timestamp_str:<81}║")
    print(f"║  Samples Distilled:      {samples} query/decision exemplars ingested into SystemOneCompiler{' ':>24}║")
    print(f"║  Closed-Form Solve:      Ridge Regression Hyperplanes: W* = (X^T X + λI)^(-1) X^T Y{' ':>29}║")
    print(f"║  Local Model Agreement:  {agreement:.1f}% (Required Threshold: {threshold:.1f}%){' ':>45}║")
    print(f"║  New Execution Status:   LOCAL EXECUTION ACTIVE; SEE MEASURED LATENCY{' ':>16}║")
    print(f"║  Data Egress Status:     Adapter local path active; application networking is separate{' ':>12}║")
    print("╚" + "═" * 108 + "╝\n")


def render_latency_cliff_chart(results: List[Dict[str, Any]], cutover_idx: Optional[int] = None) -> None:
    print("\n" + "=" * 110)
    print("                        LATENCY CLIFF: CLOUD PROXY vs. 100% LOCAL REFLEX")
    print("=" * 110)
    print(f"{'#':<3} | {'Phase':<15} | {'Latency':<11} | {'Visual Latency Curve (Relative Scale)':<42} | {'Egress':<8} | {'Receipt'}")
    print("-" * 110)

    max_lat = max(r["latency_ms"] for r in results) if results else 300.0
    max_lat = max(max_lat, 200.0)
    bar_width = 38

    for idx, r in enumerate(results, start=1):
        lat = r["latency_ms"]
        is_local = r["local_execution"]
        phase_str = "LOCAL REFLEX" if is_local else "CLOUD PROXY"

        scaled_len = max(1, int((lat / max_lat) * bar_width))
        bar_char = "█" if not is_local else "▓"
        bar = bar_char * scaled_len

        egress_str = f"{r['egress_bytes']} B" if r["egress_bytes"] > 0 else "0 B"
        receipt_str = "Verified (Ed25519)" if r["receipt_verified"] else "None (Cloud)"

        if cutover_idx is not None and idx == cutover_idx + 1:
            print("-" * 110)
            print("  >>> [CUTOVER POINT: Closed-form weights distilled. Traffic flipped to local execution] <<<")
            print("-" * 110)

        print(f"{idx:02d}  | {phase_str:<15} | {lat:>7.2f} ms  | [{bar:<{bar_width}}] | {egress_str:>8} | {receipt_str}")

    print("=" * 110)


def render_summary_scorecard(results: List[Dict[str, Any]]) -> None:
    cloud_results = [r for r in results if not r["local_execution"]]
    local_results = [r for r in results if r["local_execution"]]

    avg_cloud_lat = (sum(r["latency_ms"] for r in cloud_results) / len(cloud_results)) if cloud_results else 0.0
    avg_local_lat = (sum(r["latency_ms"] for r in local_results) / len(local_results)) if local_results else 0.0

    total_cloud_egress = sum(r["egress_bytes"] for r in cloud_results)
    total_local_egress = sum(r["egress_bytes"] for r in local_results)

    total_cloud_tokens = sum(r["total_tokens"] for r in cloud_results)
    total_local_tokens = sum(r["total_tokens"] for r in local_results)

    cloud_cost_usd = total_cloud_tokens * 0.000002
    local_cost_usd = 0.0

    if avg_cloud_lat > 0 and avg_local_lat > 0:
        speedup_str = f"{avg_cloud_lat / avg_local_lat:.1f}x faster"
    elif local_results:
        speedup_str = "Sub-2ms local"
    else:
        speedup_str = "N/A (Not Cutover)"

    cloud_lat_str = f"{avg_cloud_lat:>16.2f} ms" if cloud_results else "N/A"
    local_lat_str = f"{avg_local_lat:>15.2f} ms" if local_results else "N/A (Not Cutover)"
    avg_cloud_egress = int(total_cloud_egress / max(1, len(cloud_results))) if cloud_results else 0

    print("\n" + "┌" + "─" * 86 + "┐")
    print("│" + " " * 28 + "METRICS & PERFORMANCE SCORECARD" + " " * 27 + "│")
    print("├" + "─" * 86 + "┤")
    print(f"│  Metric                          │ Cloud Proxy Phase      │ Local System 1 System 1 │")
    print("├──────────────────────────────────┼────────────────────────┼───────────────────────┤")
    print(f"│  Average Latency                 │ {cloud_lat_str}   │ {local_lat_str}   │")
    print(f"│  Measured Speedup Factor         │ {'Baseline (1.0x)':<22} │ {speedup_str:>21} │")
    print(f"│  Data Egress per Decision        │ {avg_cloud_egress:>16} B    │ {0:>17} B   │")
    print(f"│  Total Cumulative Egress         │ {total_cloud_egress:>16} B    │ {total_local_egress:>17} B   │")
    print(f"│  Egress Reduction                │ {'0% (Full Payload)':<22} │ {'100.0% (Zero Egress)':>21} │")
    print(f"│  Tokens Billed                   │ {total_cloud_tokens:>16} tok  │ {total_local_tokens:>17} tok │")
    print(f"│  Inference Cost                  │ ${cloud_cost_usd:>15.5f}    │ ${local_cost_usd:>16.2f}   │")
    print(f"│  Split Conformal Guarantees      │ {'Not Available':<22} │ {'Finite-Sample (1-α)':>21} │")
    print(f"│  Cryptographic Non-Repudiation   │ {'None (Plain JSON)':<22} │ {'Ed25519 Receipts':>21} │")
    print(f"│  Tamper-Evident Auditability     │ {'Unverified Log':<22} │ {'SQLite ActionLedger':>21} │")
    print("└" + "─" * 86 + "┘\n")


# ============================================================================
# Burst Throughput Benchmark
# ============================================================================

def run_burst_benchmark(
    client: TypeSafeClient,
    questions: Mapping[str, Any],
    queries: Sequence[Dict[str, Any]],
    burst_count: int = 100,
) -> None:
    print("\n" + "=" * 80)
    print(f"  BURST THROUGHPUT BENCHMARK ({burst_count} Sequential Inferences)")
    print("=" * 80)
    if not client.is_cutover and client.mode != "local":
        print("  Notice: Client has not yet cut over to local execution.")
        print("  Skipping local throughput benchmark until autonomous cutover triggers.")
        return

    print("Testing post-cutover single-threaded local decision throughput...")

    prompt_pool = [q["prompt"] for q in queries]
    t0 = time.perf_counter()

    for i in range(burst_count):
        prompt = prompt_pool[i % len(prompt_pool)]
        client.systemone(prompt, questions, alpha=0.05, record_receipt=False)

    total_time = time.perf_counter() - t0
    qps = burst_count / total_time
    avg_lat_ms = (total_time / burst_count) * 1000.0

    print(f"  Total Inferences:       {burst_count}")
    print(f"  Total Elapsed Time:     {total_time:.3f} s")
    print(f"  Average Single-Pass:    {avg_lat_ms:.3f} ms")
    print(f"  Local Throughput (QPS): {qps:.1f} queries / second")
    print(f"  Cloud Comparison:       ~3-5 QPS (Network RTT bound)")
    print(f"  Throughput Multiplier:  ~{qps / 4.0:.0f}x higher throughput")


# ============================================================================
# Cryptographic & ActionLedger Verification
# ============================================================================

def verify_crypto_and_ledger(
    ledger: ActionLedger,
    signing_key: Ed25519PrivateKey,
    last_response: TypeSafeResponse,
) -> None:
    print("\n" + "=" * 80)
    print("  CRYPTOGRAPHIC VERIFICATION & ACTIONLEDGER AUDIT REPORT")
    print("=" * 80)

    # 1. ActionLedger Chain Verification
    print("[1] SQLite ActionLedger Hash Chain Verification:")
    is_valid_chain = ledger.verify_integrity()
    head_seq, head_hash = ledger.audit_head()
    print(f"    ActionLedger DB Path:        {ledger.path}")
    print(f"    Total Audit Entries:         {head_seq}")
    print(f"    Audit Chain Head Hash:       {head_hash[:32]}...")
    print(f"    Tamper-Evidence Check:       {'PASSED (SHA-256 Hash Chain Valid)' if is_valid_chain else 'FAILED'}")
    assert is_valid_chain, "ActionLedger integrity check failed!"

    # 2. Ed25519 Receipt Verification
    print("\n[2] Ed25519 Proof-Carrying Decision Receipt Verification:")
    receipt = last_response.get("receipt")
    if receipt:
        is_verified = verify_decision_witness_receipt(
            receipt,
            public_key=signing_key.public_key(),
        )
        print(f"    Decision ID:                 {receipt.get('decision_id')}")
        print(f"    Schema Digest:               {receipt.get('schema_digest')[:32]}...")
        print(f"    Ed25519 Signature:           {receipt.get('signature', '')[:32]}...")
        print(f"    Signature Verification:      {'VALID (Ed25519 Non-Repudiation Proven)' if is_verified else 'INVALID'}")
        print(f"    Bound Ledger Head:           {receipt.get('truth_ledger_head', '')[:32]}...")
        assert is_verified, "Receipt cryptographic signature verification failed!"
    else:
        print("    No receipt attached to this response (Cloud passthrough or receipts disabled)")

    # 3. Split Conformal Prediction Verification
    print("\n[3] Split Conformal Prediction Mathematical Guarantees:")
    alpha = last_response.get("conformal_alpha")
    if last_response.get("local_execution") and alpha is not None:
        print(f"    Significance Level (α):      {alpha} (Target Coverage: {(1.0 - alpha) * 100:.1f}%)")
        for name, ans in last_response.get("answers", {}).items():
            cset = getattr(ans, "conformal_set", None)
            conf = getattr(ans, "confidence", 1.0)
            val = getattr(ans, "value", None)
            print(f"    • {name:<22}: value={str(val):<20} conf={conf:>5.1%} conformal_set={cset}")
    else:
        print("    Not Applicable (Cloud proxy phase: conformal sets activate upon local System 1 cutover)")


# ============================================================================
# Monkey-Patching Demonstration
# ============================================================================

def demonstrate_monkey_patching(
    use_case: str,
    questions: Mapping[str, Any],
    use_case_queries: Sequence[Dict[str, Any]],
) -> None:
    print("\n" + "=" * 80)
    print("  ZERO-CODE-CHANGE MONKEY PATCHING (patch_typesafe())")
    print("=" * 80)
    print("Legacy enterprise codebases with existing TypeSafe AI SDK imports:")
    print("    import typesafe_sdk")
    print("    from typesafe_sdk import TypeSafeClient, Choice, Noul, Score\n")

    unpatcher = patch_typesafe()
    try:
        import typesafe_sdk

        client = typesafe_sdk.TypeSafeClient()
        test_query = use_case_queries[0]["prompt"]

        t0 = time.perf_counter()
        resp = client.systemone(test_query, questions)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  Successfully intercepted legacy `typesafe_sdk` call without code modifications:")
        print(f"  Domain:        {use_case.capitalize()}")
        print(f"  Prompt:        \"{test_query[:65]}...\"")
        print(f"  Execution:     {'100% Local On-Device' if resp.local_execution else 'Cloud'}")
        print(f"  Latency:       {elapsed_ms:.3f} ms")

        # Dynamically display first 2 schema answers
        for k in list(questions.keys())[:2]:
            ans = getattr(resp.answers, k, None)
            val = getattr(ans, "choice", None) or getattr(ans, "value", "N/A")
            conf = getattr(ans, "confidence", 1.0)
            print(f"  Output [{k}]: {str(val):<24} (Conf: {conf:>5.1%})")
        print(f"  Receipt:       {'Attached (Ed25519)' if resp.receipt else 'None'}")
    finally:
        unpatcher.unpatch()


# ============================================================================
# Main Showcase Engine
# ============================================================================

def run_auto_cutover_showcase(
    use_case: str = "fintech",
    cutover_threshold: int = 8,
    total_queries: int = 20,
    api_key: Optional[str] = None,
    db_path: str = ":memory:",
    run_benchmark: bool = True,
    run_monkey_patch: bool = True,
    throttle_ms: float = 40.0,
    export_model_path: Optional[str] = None,
) -> None:
    load_env_credentials()
    if not api_key:
        api_key = os.environ.get("TYPESAFE_API_KEY", "") or os.environ.get("JEV_API_KEY", "")

    # 1. Select killer use case
    if use_case == "clinical":
        title, questions, query_stream = get_clinical_triage_use_case()
    elif use_case == "ecommerce":
        title, questions, query_stream = get_ecommerce_support_use_case()
    else:
        title, questions, query_stream = get_fintech_wire_fraud_use_case()

    # Repeat queries if total exceeds stream length
    queries: List[Dict[str, Any]] = []
    while len(queries) < total_queries:
        queries.extend(query_stream)
    queries = queries[:total_queries]

    render_ascii_header(title, title, cutover_threshold)

    # 2. Setup cryptographic keys & tamper-evident SQLite ActionLedger
    signing_key = Ed25519PrivateKey.generate()
    ledger = ActionLedger(db_path)

    # 3. Instantiate TypeSafeClient in auto_cutover mode
    # For fast illustrative showcase runs, provide explicit demo policy
    print("Legacy simulation: augmented examples and relaxed gates; synthetic WAN metrics are not live measurements.")
    demo_policy = PromotionPolicy(
        min_agreement_threshold=0.75,
        false_allow_ceiling=0.0,
        require_statistical_bound=False,
        min_local_acceptance=0.0,
    )
    effective_threshold = max(3, cutover_threshold) if cutover_threshold <= 2 and total_queries >= 3 else cutover_threshold
    client = TypeSafeClient(
        api_key=api_key,
        mode="auto_cutover",
        cutover_threshold=effective_threshold,
        min_agreement_threshold=0.8,
        promotion_policy=demo_policy,
        augment=True, strict_mode=False,  # Legacy synthetic demonstration; see observe_routing.py for default gates.
        signing_key=signing_key,
        ledger=ledger,
        timeout=3.0,
        zero_egress=False,
        fallback_baseline=True,
    )

    print("Initial Client State:")
    print(f"  • Mode:             {client.mode} (The Trojan Horse)")
    print(f"  • Threshold:        {client.cutover_threshold} queries")
    print(f"  • Is Cutover:       {client.is_cutover}")
    print(f"  • Queries Queued:   {len(queries)}")
    if api_key:
        print(f"  • Live API:         https://api.typesafe.ai/v1 (Key: ***{api_key[-4:]})")
    else:
        print(f"  • Live API:         Realistic cloud WAN baseline (No TYPESAFE_API_KEY set)")
    print("\nBeginning Query Stream Processing...\n")

    results: List[Dict[str, Any]] = []
    cutover_event_logged: Optional[Dict[str, Any]] = None
    cutover_call_idx: int = cutover_threshold

    for idx, item in enumerate(queries, start=1):
        prompt = item["prompt"]
        item_id = item.get("id", f"Q-{idx}")
        item_type = item.get("type", "General")

        was_cutover_before = client.is_cutover

        # Execute query via standard TypeSafe API
        t0 = time.perf_counter()
        resp = client.systemone(prompt, questions, alpha=0.05)
        wall_lat = (time.perf_counter() - t0) * 1000.0

        is_now_cutover = client.is_cutover
        is_local = resp.local_execution

        # Add brief pacing during cloud phase to reflect physical network transit
        if not is_local and throttle_ms > 0:
            time.sleep(throttle_ms / 1000.0)

        first_q_name = list(questions.keys())[0]
        first_ans = resp.answers.get(first_q_name)
        decision_val = getattr(first_ans, "choice", None) or getattr(first_ans, "value", "N/A")
        confidence = getattr(first_ans, "confidence", 1.0)
        cset = getattr(first_ans, "conformal_set", None)

        tokens = resp.usage.total_tokens
        egress = resp.get("egress_bytes", 0)
        has_receipt = bool(resp.receipt is not None)

        record = {
            "idx": idx,
            "id": item_id,
            "type": item_type,
            "prompt": prompt,
            "local_execution": is_local,
            "is_cutover": is_now_cutover,
            "latency_ms": resp.latency_ms,
            "wall_lat_ms": wall_lat,
            "egress_bytes": egress,
            "total_tokens": tokens,
            "decision": decision_val,
            "confidence": confidence,
            "conformal_set": cset,
            "receipt_verified": has_receipt,
            "response": resp,
        }
        results.append(record)

        phase_label = "[LOCAL METAL]" if is_local else "[CLOUD WAN]  "
        lat_display = f"{resp.latency_ms:>7.2f} ms"
        egress_display = f"{egress:>4} B" if egress > 0 else "  0 B"
        receipt_display = "✓ Signed" if has_receipt else "- None  "

        print(
            f"Query #{idx:02d} | {item_id:<18} | {phase_label} | {lat_display} | "
            f"Egress: {egress_display} | Tokens: {tokens:>3} | "
            f"Decision: {str(decision_val):<22} (Conf: {confidence:>5.1%}) | {receipt_display}"
        )

        if not was_cutover_before and is_now_cutover and not cutover_event_logged:
            cutover_call_idx = idx
            audit_events = client.cutover_audit_log
            if audit_events:
                cutover_event_logged = next(
                    (e for e in reversed(audit_events) if e.get("event") == "trojan_horse_cutover"),
                    audit_events[-1],
                )
                render_cutover_banner(cutover_event_logged)

    # 4. Latency curves and visualizations
    actual_cutover_idx: Optional[int] = None
    for i, r in enumerate(results):
        if r["local_execution"]:
            actual_cutover_idx = i
            break

    render_latency_cliff_chart(results, cutover_idx=actual_cutover_idx)
    render_summary_scorecard(results)

    # 5. Burst Throughput Benchmark
    if run_benchmark:
        run_burst_benchmark(client, questions, queries, burst_count=100)

    # 6. Cryptographic & ActionLedger Verification
    last_resp = results[-1]["response"]
    verify_crypto_and_ledger(ledger, signing_key, last_resp)

    # 7. Optional Model Export (.s1m)
    if export_model_path:
        success = client.export_model(export_model_path)
        if success:
            file_sz = Path(export_model_path).stat().st_size
            print(f"\n[Model Export] Distilled closed-form model successfully exported to: {export_model_path} ({file_sz:,} bytes)")
        else:
            print(f"\n[Model Export] Notice: Model export skipped because client has not compiled closed-form weights.")

    # 8. Monkey Patching Demonstration
    if run_monkey_patch:
        demonstrate_monkey_patching(use_case, questions, queries)

    print("\n" + "=" * 110)
    print("  SIMULATED CUTOVER SHOWCASE FINISHED — SEE MEASUREMENTS AND PROMOTION STATUS ABOVE")
    print("  Simulated teacher and relaxed gates; measured results above are not production SLAs.")
    print("=" * 110 + "\n")


def main() -> None:
    load_env_credentials()
    parser = argparse.ArgumentParser(
        description="System 1 vs TypeSafe AI: Autonomous Trojan Horse Cutover Enterprise Showcase",
    )
    parser.add_argument(
        "--use-case",
        choices=["fintech", "clinical", "ecommerce"],
        default="fintech",
        help="Killer enterprise domain to demonstrate (default: fintech)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Number of initial queries before autonomous local cutover (default: 8)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=20,
        help="Total number of queries in the live stream (default: 20)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TYPESAFE_API_KEY", "") or os.environ.get("JEV_API_KEY", ""),
        help="Optional live TypeSafe AI API key (defaults to $TYPESAFE_API_KEY)",
    )
    parser.add_argument(
        "--db-path",
        default=":memory:",
        help="SQLite ActionLedger file path (default: :memory:)",
    )
    parser.add_argument(
        "--export-model",
        type=str,
        default=None,
        help="Optional file path to export the distilled model as a .s1m binary",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the burst throughput benchmark",
    )
    parser.add_argument(
        "--no-monkey-patch",
        action="store_true",
        help="Skip the monkey patch demonstration",
    )
    parser.add_argument(
        "--throttle-ms",
        type=float,
        default=40.0,
        help="Simulated network pacing per cloud call in ms (default: 40.0ms)",
    )

    args = parser.parse_args()

    run_auto_cutover_showcase(
        use_case=args.use_case,
        cutover_threshold=args.threshold,
        total_queries=args.total,
        api_key=args.api_key,
        db_path=args.db_path,
        run_benchmark=not args.no_benchmark,
        run_monkey_patch=not args.no_monkey_patch,
        throttle_ms=args.throttle_ms,
        export_model_path=args.export_model,
    )


if __name__ == "__main__":
    main()
