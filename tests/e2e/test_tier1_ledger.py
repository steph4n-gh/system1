"""Tier 1.6: E2E Requirement Tests for SQLite ActionLedger Integrity.

Authoritative Invariants:
1. Append-only ledger establishes a cryptographic SHA-256 hash chain rooted at genesis ("0"*64).
2. Sequential monotonic sequence numbers are assigned to every audit event.
3. Any mutation, payload alteration, or deletion of ledger rows is immediately detected by verify_integrity().
4. Read-only mode prevents mutations and allows secure auditing.
5. Concurrent multi-threaded writes preserve chain continuity without corrupting the audit tip.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import pytest

import reflex
from reflex import (
    ActionLedger,
    DefaultGuardDecisionSchema,
    IntegrityError,
    LedgerError,
)


def test_genesis_state_and_sequential_hash_chain(temp_ledger):
    """Verify genesis block initialization and verifiable SHA-256 hash chaining."""
    seq0, head0 = temp_ledger.audit_head()
    assert seq0 == 0
    assert head0 == "0" * 64
    assert temp_ledger.verify_integrity() is True

    # Record first decision
    res1 = reflex.decide("First audit transaction", schema=DefaultGuardDecisionSchema)
    hash1 = temp_ledger.record_decision_receipt(res1.receipt)
    seq1, head1 = temp_ledger.audit_head()
    assert seq1 == 1
    assert head1 == hash1
    assert len(hash1) == 64

    # Record second decision
    res2 = reflex.decide("Second audit transaction", schema=DefaultGuardDecisionSchema)
    hash2 = temp_ledger.record_decision_receipt(res2.receipt)
    seq2, head2 = temp_ledger.audit_head()
    assert seq2 == 2
    assert head2 == hash2
    assert hash2 != hash1

    assert temp_ledger.verify_integrity() is True


def test_tamper_detection_mutated_payload(temp_ledger):
    """Verify direct SQLite update to payload_json triggers integrity violation."""
    for i in range(3):
        res = reflex.decide(f"Transaction step {i}", schema=DefaultGuardDecisionSchema)
        temp_ledger.record_decision_receipt(res.receipt)

    assert temp_ledger.verify_integrity() is True

    # Direct raw database mutation (simulating malicious database compromise)
    temp_ledger._connection.execute(
        "UPDATE audit_entries SET payload_json = '{\"forged\": true}' WHERE sequence = 2"
    )
    temp_ledger._connection.commit()

    assert temp_ledger.verify_integrity() is False


def test_tamper_detection_broken_chain_link(temp_ledger):
    """Verify modifying previous_hash pointer breaks chain verification."""
    for i in range(3):
        res = reflex.decide(f"Transaction step {i}", schema=DefaultGuardDecisionSchema)
        temp_ledger.record_decision_receipt(res.receipt)

    # Corrupt chain linkage at sequence 2
    temp_ledger._connection.execute(
        "UPDATE audit_entries SET previous_hash = 'f' * 64 WHERE sequence = 2"
    )
    temp_ledger._connection.commit()

    assert temp_ledger.verify_integrity() is False


def test_tamper_detection_deleted_entry(temp_ledger):
    """Verify deleting a ledger row causes chain discontinuity."""
    for i in range(4):
        res = reflex.decide(f"Audit log item {i}", schema=DefaultGuardDecisionSchema)
        temp_ledger.record_decision_receipt(res.receipt)

    # Delete intermediate record
    temp_ledger._connection.execute("DELETE FROM audit_entries WHERE sequence = 2")
    temp_ledger._connection.commit()

    assert temp_ledger.verify_integrity() is False


def test_read_only_mode_and_querying(tmp_path: Path):
    """Verify read-only mode permits querying and rejects write attempts."""
    db_file = tmp_path / "durable_audit.db"
    
    # 1. Populate durable ledger
    with ActionLedger(db_file) as ledger:
        for i in range(3):
            res = reflex.decide(f"Durable step {i}", schema=DefaultGuardDecisionSchema)
            ledger.record_decision_receipt(res.receipt)
        assert ledger.verify_integrity() is True

    # 2. Open in read-only mode
    with ActionLedger(db_file, read_only=True) as ro_ledger:
        assert ro_ledger.read_only is True
        assert ro_ledger.verify_integrity() is True
        
        # Verify write rejection
        dummy_res = reflex.decide("Attempt write on read-only", schema=DefaultGuardDecisionSchema)
        with pytest.raises(LedgerError, match="read-only"):
            ro_ledger.record_decision_receipt(dummy_res.receipt)


def test_concurrent_multithread_ledger_appending(temp_ledger):
    """Verify multi-threaded concurrent writers append safely with intact hash chain."""
    num_threads = 4
    items_per_thread = 5

    def worker(worker_id: int):
        for i in range(items_per_thread):
            res = reflex.decide(f"Worker {worker_id} item {i}", schema=DefaultGuardDecisionSchema)
            temp_ledger.record_decision_receipt(res.receipt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, tid) for tid in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    total_expected = num_threads * items_per_thread
    final_seq, final_head = temp_ledger.audit_head()
    assert final_seq == total_expected
    assert len(final_head) == 64
    assert temp_ledger.verify_integrity() is True
