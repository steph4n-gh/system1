"""Reflex Protocol Buffer definitions package.

Contains the .proto schema files and (when compiled) the generated
Python stubs for the ReflexService gRPC API.
"""
from __future__ import annotations

__all__ = ["reflex_proto_path"]


def reflex_proto_path() -> str:
    """Return the filesystem path to reflex.proto."""
    from pathlib import Path

    return str(Path(__file__).parent / "reflex.proto")
