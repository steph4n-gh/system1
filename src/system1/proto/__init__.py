"""System 1 Protocol Buffer definitions package.

Contains the .proto schema files and (when compiled) the generated
Python stubs for the SystemOneService gRPC API.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["system1_proto_path", "system1_pb2", "system1_pb2_grpc"]


def system1_proto_path() -> str:
    """Return the filesystem path to system1.proto."""
    return str(Path(__file__).parent / "system1.proto")


try:
    from system1.proto import system1_pb2, system1_pb2_grpc
except ImportError:
    pass
