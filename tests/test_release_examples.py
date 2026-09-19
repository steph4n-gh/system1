"""Public quickstarts and benchmark failures must remain reproducible."""

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("example", [0, 1])
def test_readme_quickstarts_run_outside_checkout(tmp_path, example):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    for _ in range(2):  # The persistent signing identity must also work on a second run.
        result = subprocess.run(
            [sys.executable, "-c", blocks[example]], cwd=tmp_path,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "system1-demo" in result.stdout if example == 1 else "Needs review:" in result.stdout


def test_quality_benchmark_failure_sets_exit_status(tmp_path, monkeypatch):
    from benchmarks.quality import run_quality_benchmarks as runner

    def broken_benchmark():
        raise RuntimeError("sentinel benchmark failure")

    output = tmp_path / "nested" / "results.json"
    monkeypatch.setattr(runner, "BENCHMARKS", {"security": ("Security", broken_benchmark)})
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert json.loads(output.read_text())["benchmarks"]["security"]["error"] == "sentinel benchmark failure"


def test_teaching_example_saves_and_reuses_a_skill(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "teach_skill.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Suggested team: billing" in result.stdout
    assert "Needs review: True" in result.stdout
    assert (tmp_path / ".system1" / "support-route.s1m").is_file()
