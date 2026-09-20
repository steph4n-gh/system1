#!/usr/bin/env python3
"""Launch scripted Pokémon milestones and damage-risk checks.

Optional PyBoy ticks an emulator; this does not verify a complete campaign,
a speed record or guaranteed wipe prevention."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root and examples are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLES_DIR = REPO_ROOT / "examples"
GAMING_DIR = EXAMPLES_DIR / "gaming"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(GAMING_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_DIR))

# Auto-switch to repo virtualenv if available and not already active
venv_python = REPO_ROOT / ".venv" / "bin" / "python"
if venv_python.is_file() and sys.executable != str(venv_python):
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from pokemon_kaizo_speedrun import main

if __name__ == "__main__":
    main()
