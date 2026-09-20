"""Check repository Markdown targets/anchors and current release labels, offline.

This intentionally handles the Markdown conventions used in this repository;
it is not a general Markdown parser or an external-site availability checker.
"""
from pathlib import Path
import re
import subprocess
import tomllib
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(text):
    """Exclude fenced examples from link and heading checks."""
    return re.sub(r"(?ms)^(```|~~~)[^\n]*\n.*?^\1[^\n]*(?:\n|$)", "", text)


def anchors(text):
    found, counts = set(), {}
    for title in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", prose(text)):
        title = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", title)
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        found.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    found.update(re.findall(r'(?:id|name)=[\'\"]([^\'\"]+)[\'\"]', text))
    return found


def main():
    # Include new documentation during development; ignore local environments/output.
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    docs = sorted({ROOT / p for p in paths if p.endswith(".md") and (ROOT / p).is_file()})
    errors, checked = [], 0
    for path in docs:
        text = prose(path.read_text())
        targets = re.findall(r"\[[^]\n]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", text)
        targets += re.findall(r"(?m)^\s*\[[^]]+\]:\s*(\S+)", text)
        targets += re.findall(r'(?:href|src)=[\'\"]([^\'\"]+)[\'\"]', text)
        for raw in targets:
            url = urlsplit(raw.strip("<>"))
            if url.scheme or url.netloc:
                continue
            checked += 1
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing {raw}")
            elif url.fragment and target.suffix == ".md" and unquote(url.fragment) not in anchors(target.read_text()):
                errors.append(f"{path.relative_to(ROOT)}: missing anchor {raw}")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    labels = {
        "README.md": r"\*\*Version ([\d.]+):",
        "docs/README.md": r"stable runtime: \*\*([\d.]+)\*\*",
        "docs/architecture/technical_specification.md": r"System 1 ([\d.]+)",
        "docs/paper/system1_whitepaper.md": r"System 1 ([\d.]+)",
        "docs/paper/system1_technical_brief.md": r"System 1 ([\d.]+)",
        "docs/paper/conformal_gating.md": r"System 1 ([\d.]+)",
        "docs/SPEEDRUN_SHOWDOWN_WORLD_RECORDS.md": r"System 1 ([\d.]+)",
        "src/system1/__init__.py": r'__version__ = "([\d.]+)"',
    }
    for name, pattern in labels.items():
        match = re.search(pattern, (ROOT / name).read_text())
        if not match or match[1] != version:
            errors.append(f"{name}: current version label must match {version}")
    for error in errors:
        print(error)
    print(f"Checked {len(docs)} Markdown files, {checked} local links and {len(labels)} version labels.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
