"""Shared fixtures and configuration for Reflex / System 1 E2E test suite."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import reflex
from reflex import (
    ActionLedger,
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
    save_keypair,
)


class BlockedNetworkCallError(RuntimeError):
    """Raised when an external network socket attempt is intercepted."""


@pytest.fixture
def enforce_zero_network(monkeypatch) -> None:
    """Hard-blocks any network socket creation or connection attempts."""
    orig_connect = socket.socket.connect

    def blocked_connect(self, address):
        raise BlockedNetworkCallError(f"Zero-network violation: blocked connection to {address}")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)


@pytest.fixture
def temp_ledger(tmp_path: Path) -> Generator[ActionLedger, None, None]:
    """Provides an isolated ActionLedger backed by a temporary SQLite file."""
    db_file = tmp_path / "test_ledger.db"
    ledger = ActionLedger(db_file)
    try:
        yield ledger
    finally:
        ledger.close()


@pytest.fixture
def temp_identity(tmp_path: Path) -> tuple[Path, Path]:
    """Generates and persists an Ed25519 keypair in a temp directory."""
    priv_path = tmp_path / "system1_ed25519.pem"
    pub_path = tmp_path / "system1_ed25519.pub"
    priv = Ed25519PrivateKey.generate()
    save_keypair(priv, priv_path, pub_path)
    return priv_path, pub_path


class E2ETriageSchema(DecisionSchema):
    """Canonical triage schema for requirement-driven E2E tests."""

    route = ChoiceField(
        options=["local_reflex", "cloud_planner", "human_escalation"],
        descriptions={
            "local_reflex": "deterministic safe fast read or cached calculation",
            "cloud_planner": "complex synthesis multi-step strategic planning",
            "human_escalation": "high-consequence irreversible credential or security action",
        },
    )
    is_safe = BooleanField(
        threshold=0.5,
        true_description="benign read only safe observation inspection",
        false_description="destructive wipe overwrite attack compromise",
    )
    confidence_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="vague uncertain incomplete input",
        high_description="clear explicit deterministic directive",
    )


@pytest.fixture
def triage_schema() -> type[DecisionSchema]:
    return E2ETriageSchema
