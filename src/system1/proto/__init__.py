"""System 1 Protocol Buffer definitions package.

Contains the .proto schema files and (when compiled) the generated
Python stubs for the SystemOneService gRPC API.
"""
from __future__ import annotations

__all__ = ["system1_proto_path"]


def system1_proto_path() -> str:
    """Return the filesystem path to reflex.proto."""
    from pathlib import Path

    return str(Path(__file__).parent / "reflex.proto")
