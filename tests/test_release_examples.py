"""Public quickstarts and benchmark failures must remain reproducible."""

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_readme_policy_quickstart_runs_outside_checkout(tmp_path):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    for _ in range(2):  # The persistent signing identity must also work on a second run.
        result = subprocess.run(
            [sys.executable, "-c", blocks[0]], cwd=tmp_path,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "system1-demo" in result.stdout


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
    original = (tmp_path / ".system1" / "support-route.s1m").read_bytes()
    # Editing the answers, rather than an embedded rule, must teach the policy.
    lessons = json.loads((ROOT / 'examples/teaching/first_skill.json').read_text())
    for pair in lessons['team']:
        pair[1] = 'support' if pair[1] == 'billing' else 'billing'
    lesson_path = tmp_path / 'my-lessons.json'
    lesson_path.write_text(json.dumps(lessons))
    result = subprocess.run(
        [sys.executable, str(ROOT / 'examples/teach_skill.py'),
         '--lessons', str(lesson_path), '--prompt', 'Please refund this payment',
         '--output', 'candidate.s1m'],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'Suggested team: support' in result.stdout
    assert (tmp_path / 'candidate.s1m').is_file()
    assert (tmp_path / '.system1/support-route.s1m').read_bytes() == original


def test_proto_assets_do_not_require_optional_grpc_dependencies(tmp_path):
    script = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'grpc' or fullname == 'google' or fullname.startswith('google.'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
from importlib.resources import files
import reflex.proto, system1.proto
assert files('system1.proto').joinpath('system1.proto').is_file()
assert files('reflex.proto').joinpath('system1.proto').is_file()
assert reflex.proto.system1_proto_path is system1.proto.system1_proto_path
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
