"""Reflex CLI Module (re-export from system1.cli)."""
from __future__ import annotations
from system1.cli import *
from system1.cli import __all__ as _all
__all__ = list(_all)

if __name__ == "__main__":
    import sys
    sys.exit(main())
