#!/usr/bin/env python3
"""Reflex vs TypeSafe AI: Enterprise Stress Showcase.

Alias entrypoint for `examples/autonomous_agent_firewall_showcase.py`
demonstrating the high-stakes Autonomous Agent Firewall scenario under high concurrency.
"""

import sys
from pathlib import Path

# Ensure src/ and examples/ are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
for p in (SRC_DIR, EXAMPLES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from autonomous_agent_firewall_showcase import (
    get_autonomous_agent_firewall_dataset,
    get_autonomous_agent_firewall_schema,
    get_interleaved_query_stream,
    main,
    run_autonomous_firewall_showcase,
    run_multithreaded_stress_test,
)

if __name__ == "__main__":
    main()
