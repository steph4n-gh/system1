"""Reflex Lightweight Tamper-Evident SQLite Action Ledger.

Provides cryptographic audit logging and hash chaining for decision receipts,
ensuring non-repudiation, tamper detection, and historical chain verification.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Tuple

from reflex.receipt import (
    canonical_json,
    fingerprint,
    utc_now,
)

_ZERO_HASH = "0" * 64


class LedgerError(RuntimeError):
    """Base exception for ledger operations."""
    pass


class IntegrityError(LedgerError):
    """Raised when cryptographic chain validation fails."""
    pass


class ActionLedger:
    """Lightweight, tamper-evident SQLite audit ledger with hash chaining."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        read_only: bool = False,
    ) -> None:
        self.path = str(path)
        self.read_only = bool(read_only)
        if self.read_only and self.path == ":memory:":
            raise ValueError("Read-only ledger requires an existing durable database file")

        uri = False
        database = self.path
        if self.path != ":memory:":
            p = Path(self.path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(p)
            if self.read_only:
                database = p.as_uri() + "?mode=ro"
                uri = True
            else:
                database = str(p)

        self._connection = sqlite3.connect(
            database,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
            uri=uri,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        self._configure()
        if not self.read_only:
            self._initialize()

    def _configure(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode = WAL;")
            self._connection.execute("PRAGMA synchronous = NORMAL;")
            self._connection.execute("PRAGMA foreign_keys = ON;")

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_entries (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    action_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_entries(action_id, sequence);

                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO ledger_meta(key, value) VALUES
                    ('audit_head_hash', '0000000000000000000000000000000000000000000000000000000000000000'),
                    ('audit_head_sequence', '0');
                """
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise LedgerError("Cannot perform write transaction on read-only ledger")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> ActionLedger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def audit_head(self) -> Tuple[int, str]:
        """Returns (sequence, head_hash) of the latest audit record."""
        with self._lock:
            row = self._connection.execute(
                "SELECT sequence, entry_hash FROM audit_entries ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            actual = (
                (0, _ZERO_HASH) if row is None else (int(row["sequence"]), str(row["entry_hash"]))
            )
            meta = dict(self._connection.execute("SELECT key, value FROM ledger_meta").fetchall())
            persisted = (
                int(meta.get("audit_head_sequence", "-1")),
                str(meta.get("audit_head_hash", "")),
            )
            if persisted != actual:
                raise IntegrityError("Persisted audit head does not match the chain tip")
            return actual

    def head_hash(self) -> str:
        """Returns the current audit head hash string."""
        return self.audit_head()[1]

    def record_decision_receipt(
        self,
        receipt: Mapping[str, Any] | Any,
        *,
        tenant_id: str = "reflex",
        principal_id: str = "engine",
        scope: str = "decision",
    ) -> str:
        """Appends a Reflex decision receipt to the tamper-evident audit ledger."""
        with self._transaction() as connection:
            payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
            decision_id = payload.get("decision_id")
            receipt_digest = payload.get("receipt_digest") or (
                receipt.compute_digest() if hasattr(receipt, "compute_digest") else None
            )

            meta = dict(connection.execute("SELECT key, value FROM ledger_meta").fetchall())
            previous_hash = meta.get("audit_head_hash", _ZERO_HASH)
            previous_sequence = int(meta.get("audit_head_sequence", "0"))
            created_at = utc_now()
            event_type = "reflex_decision"
            event_id = f"audit_{fingerprint([previous_hash, event_type, created_at])[:32]}"

            entry_payload = {
                "decision_id": decision_id,
                "schema_name": payload.get("schema_name"),
                "schema_digest": payload.get("schema_digest"),
                "prompt_digest": payload.get("prompt_digest"),
                "receipt_digest": receipt_digest,
                "is_ambiguous": payload.get("is_ambiguous"),
            }

            body = {
                "event_id": event_id,
                "tenant_id": tenant_id,
                "principal_id": principal_id,
                "scope": scope,
                "action_id": decision_id,
                "event_type": event_type,
                "payload": entry_payload,
                "previous_hash": previous_hash,
                "created_at": created_at,
            }
            entry_hash = fingerprint(body)

            cursor = connection.execute(
                """
                INSERT INTO audit_entries(
                    event_id, tenant_id, principal_id, scope, action_id, event_type,
                    payload_json, previous_hash, entry_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    tenant_id,
                    principal_id,
                    scope,
                    decision_id,
                    event_type,
                    canonical_json(entry_payload),
                    previous_hash,
                    entry_hash,
                    created_at,
                ),
            )
            sequence = int(cursor.lastrowid)
            if sequence <= previous_sequence:
                raise IntegrityError("Audit sequence did not advance")

            connection.execute(
                "UPDATE ledger_meta SET value=? WHERE key='audit_head_hash'", (entry_hash,)
            )
            connection.execute(
                "UPDATE ledger_meta SET value=? WHERE key='audit_head_sequence'", (str(sequence),)
            )
            return entry_hash

    def verify_integrity(self, trusted_public_key: Optional[Any] = None) -> bool:
        """Verifies the complete cryptographic hash chain and meta head."""
        with self._lock:
            entries = self._connection.execute(
                "SELECT * FROM audit_entries ORDER BY sequence ASC"
            ).fetchall()
            previous = _ZERO_HASH
            last_sequence = 0

            for row in entries:
                if row["previous_hash"] != previous:
                    return False
                try:
                    payload_data = json.loads(row["payload_json"])
                    body = {
                        "event_id": row["event_id"],
                        "tenant_id": row["tenant_id"],
                        "principal_id": row["principal_id"],
                        "scope": row["scope"],
                        "action_id": row["action_id"],
                        "event_type": row["event_type"],
                        "payload": payload_data,
                        "previous_hash": row["previous_hash"],
                        "created_at": row["created_at"],
                    }
                except (TypeError, ValueError, json.JSONDecodeError):
                    return False

                if fingerprint(body) != row["entry_hash"]:
                    return False

                previous = row["entry_hash"]
                last_sequence = int(row["sequence"])

            meta = dict(self._connection.execute("SELECT key, value FROM ledger_meta").fetchall())
            if meta.get("audit_head_hash") != previous:
                return False
            if int(meta.get("audit_head_sequence", "-1")) != last_sequence:
                return False

            return True
