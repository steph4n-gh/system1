"""Compatibility modules for Reflex (re-export from system1.compat)."""

from __future__ import annotations

import sys
import system1.compat
import system1.compat.typesafe
from system1.compat import *
from system1.compat import __all__ as _all

# Ensure reflex.compat.typesafe maintains object identity with system1.compat.typesafe
sys.modules["reflex.compat.typesafe"] = system1.compat.typesafe
typesafe = system1.compat.typesafe

__all__ = list(_all)
