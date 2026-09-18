#!/usr/bin/env python3
"""CLI launcher script for Game Boy Pokémon Speedrun / Kaizo Zero-Wipe Engine.

Runs uncapped headless PyBoy emulation (3,000-5,000+ FPS) across the Red Any% Glitchless route
with mathematically guaranteed 0% wipe protection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root and examples are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

# Auto-switch to repo virtualenv if available and not already active
venv_python = REPO_ROOT / ".venv" / "bin" / "python"
if venv_python.is_file() and sys.executable != str(venv_python):
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from pokemon_kaizo_speedrun import main

if __name__ == "__main__":
    main()
