"""System 1 Proto Module (re-export from system1.proto)."""

from __future__ import annotations

from importlib import import_module

from system1.proto import system1_proto_path
from system1.proto import __all__ as _all

__all__ = list(_all)


def __getattr__(name):
    # Reading packaged .proto files must work without the optional gRPC extra.
    if name in ("system1_pb2", "system1_pb2_grpc"):
        return import_module(f"system1.proto.{name}")
    raise AttributeError(name)
