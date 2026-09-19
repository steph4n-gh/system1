#!/usr/bin/env python3
"""System 1 Full Repository Audit Bundle Generator.

Consolidates all project code, documentation, specifications, benchmarks,
and test suites into a single structured, self-indexing text file for external
audit and security review.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "REFLEX_AUDIT_CODEBASE.txt"

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "roms",
    "scratch",
    "assets",
    ".system1",
}

# File extensions and names to include
ALLOWED_EXTENSIONS = {
    ".py",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".proto",
}

EXPLICIT_INCLUDE_FILES = {
    "Dockerfile",
    "LICENSE",
}

EXCLUDE_PATTERNS = {
    "uv.lock",
    "REFLEX_AUDIT_CODEBASE.txt",
    "test_eval.py",
    "test_monitor.py",
    "test_playwright.py",
}

# Filter out redundant historical benchmark results
EXCLUDE_PREFIXES = (
    "benchmarks/quality/results/quality_results_",
)


def categorize_file(rel_path: str) -> Tuple[int, str]:
    """Assigns an ordering rank and category name to a file."""
    if rel_path in ("README.md", "pyproject.toml", "PROJECT.md", "LICENSE", "Dockerfile") or rel_path.startswith("deploy/"):
        return (1, "1. CONFIGURATION & INFRASTRUCTURE")
    elif rel_path.startswith("docs/") or rel_path in ("DOCUMENT_REVIEW_REPORT.md", "TEST_INFRA.md", "TEST_READY.md", "ORIGINAL_REQUEST.md"):
        return (2, "2. SPECIFICATIONS, ARCHITECTURE & PAPERS")
    elif rel_path.startswith("src/"):
        return (3, "3. CORE SOURCE CODE (system1 & reflex)")
    elif rel_path.startswith("benchmarks/"):
        return (4, "4. BENCHMARKS & EVALUATION SUITE")
    elif rel_path.startswith("examples/"):
        return (5, "5. PRODUCTION EXAMPLES & DEMOS")
    elif rel_path.startswith("tests/"):
        return (6, "6. AUTOMATED TEST SUITE")
    elif rel_path.startswith("scripts/"):
        return (7, "7. UTILITY & RUNNER SCRIPTS")
    else:
        return (8, "8. ADDITIONAL FILES")


def collect_files() -> List[Tuple[int, str, str, Path]]:
    """Discovers and categorizes all eligible text files in the repository."""
    collected = []

    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        # In-place filter to avoid traversing ignored directories
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORE_DIRS and not d.startswith(".agents")
        ]

        for fname in sorted(filenames):
            if fname.startswith(".") or fname in EXCLUDE_PATTERNS:
                continue

            file_path = Path(dirpath) / fname
            rel_path = str(file_path.relative_to(REPO_ROOT))

            if any(rel_path.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
                continue

            if file_path.suffix in ALLOWED_EXTENSIONS or fname in EXPLICIT_INCLUDE_FILES:
                rank, category = categorize_file(rel_path)
                collected.append((rank, category, rel_path, file_path))

    # Sort primarily by section rank, secondarily by path
    collected.sort(key=lambda item: (item[0], item[2]))
    return collected


def build_bundle():
    """Builds the unified single-file audit bundle."""
    files = collect_files()
    print(f"Discovered {len(files)} files to bundle...")

    # First pass: gather file contents and line counts
    file_data = []
    total_bytes = 0
    total_lines = 0

    for rank, cat, rel_path, abs_path in files:
        try:
            content = abs_path.read_text(encoding="utf-8")
        except Exception as err:
            print(f"Warning: could not read {rel_path}: {err}", file=sys.stderr)
            continue

        lines = content.splitlines(keepends=True)
        num_lines = len(lines)
        size_bytes = len(content.encode("utf-8"))

        total_bytes += size_bytes
        total_lines += num_lines

        file_data.append({
            "rank": rank,
            "category": cat,
            "rel_path": rel_path,
            "abs_path": abs_path,
            "content": content,
            "lines": lines,
            "num_lines": num_lines,
            "size_bytes": size_bytes,
            "start_line": 0,  # Will calculate in pass 2
        })

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Create preliminary manifest buffer to measure exact header length
    def render_bundle(file_start_offsets: Optional[Dict[str, int]] = None) -> str:
        buf = []
        buf.append("=" * 80 + "\n")
        buf.append("REFLEX / SYSTEM 1 — UNIFIED CODEBASE & DOCUMENTATION AUDIT BUNDLE\n")
        buf.append("=" * 80 + "\n")
        buf.append(f"Generated Date (UTC): {now_iso}\n")
        buf.append(f"Repository Root:      {REPO_ROOT}\n")
        buf.append(f"Included Files:       {len(file_data)}\n")
        buf.append(f"Total Source Lines:   {total_lines:,}\n")
        buf.append(f"Total Unpacked Size:  {total_bytes / (1024*1024):.2f} MB ({total_bytes:,} bytes)\n")
        buf.append("=" * 80 + "\n\n")

        buf.append("TABLE OF CONTENTS / MANIFEST:\n")
        buf.append("-" * 80 + "\n")
        buf.append(f"{'#':<4} {'START LINE':<12} {'LINES':<8} {'SIZE (KB)':<10} {'FILE PATH'}\n")
        buf.append("-" * 80 + "\n")

        for idx, item in enumerate(file_data, 1):
            st_line = file_start_offsets.get(item["rel_path"], 0) if file_start_offsets else 0
            size_kb = item["size_bytes"] / 1024.0
            buf.append(f"{idx:<4} Line {st_line:<7} {item['num_lines']:<8} {size_kb:<9.1f} {item['rel_path']}\n")

        buf.append("-" * 80 + "\n\n")

        current_cat = None
        for item in file_data:
            if item["category"] != current_cat:
                current_cat = item["category"]
                buf.append("\n" + "#" * 80 + "\n")
                buf.append(f"### {current_cat}\n")
                buf.append("#" * 80 + "\n\n")

            buf.append("=" * 80 + "\n")
            buf.append(f"FILE:  {item['rel_path']}\n")
            buf.append(f"LINES: {item['num_lines']:,} | SIZE: {item['size_bytes']:,} bytes\n")
            buf.append("=" * 80 + "\n")
            buf.append(item["content"])
            if not item["content"].endswith("\n"):
                buf.append("\n")
            buf.append("=" * 80 + "\n")
            buf.append(f"END OF FILE: {item['rel_path']}\n")
            buf.append("=" * 80 + "\n\n")

        return "".join(buf)

    # Header banner format
    sep_line = "#" * 80 + "\n"
    # Dummy header for line offset calculation (4 lines)
    dummy_header = (
        sep_line +
        f"# BUNDLE SHA-256: {'0'*64}\n" +
        sep_line + "\n"
    )
    header_offset = len(dummy_header.splitlines())

    # Pass 1: Render with dummy start lines to calculate file line locations
    draft = render_bundle(None)
    draft_lines = draft.splitlines(keepends=True)

    # Find the line number of each "FILE:  <rel_path>" delimiter (offset by header lines)
    start_lines = {}
    current_line = 1 + header_offset
    for line in draft_lines:
        if line.startswith("FILE:  "):
            path = line.replace("FILE:  ", "").strip()
            start_lines[path] = current_line
        current_line += 1

    # Pass 2: Re-render with exact calculated line numbers
    final_output = render_bundle(start_lines)

    # Compute SHA-256 digest of the bundle content
    bundle_hash = hashlib.sha256(final_output.encode("utf-8")).hexdigest()

    header_with_hash = (
        sep_line +
        f"# BUNDLE SHA-256: {bundle_hash}\n" +
        sep_line + "\n"
    )
    final_output_with_hash = header_with_hash + final_output

    OUTPUT_FILE.write_text(final_output_with_hash, encoding="utf-8")
    actual_size = OUTPUT_FILE.stat().st_size
    actual_lines = len(final_output_with_hash.splitlines())

    print("\n" + "=" * 60)
    print(" AUDIT BUNDLE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output File:     {OUTPUT_FILE}")
    print(f"File Size:       {actual_size / (1024*1024):.2f} MB ({actual_size:,} bytes)")
    print(f"Total Lines:     {actual_lines:,}")
    print(f"Total Files:     {len(file_data)}")
    print(f"Bundle SHA-256:  {bundle_hash}")
    print("=" * 60)


if __name__ == "__main__":
    build_bundle()
