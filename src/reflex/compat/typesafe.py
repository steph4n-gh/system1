"""TypeSafe AI (Jev) Compatibility layer re-export for reflex namespace."""

from __future__ import annotations

import sys
import system1.compat.typesafe as _typesafe
from system1.compat.typesafe import *
from system1.compat.typesafe import __all__ as _compat_all

# Maintain strict object identity with system1.compat.typesafe
sys.modules[__name__] = _typesafe

__all__ = list(_compat_all)
