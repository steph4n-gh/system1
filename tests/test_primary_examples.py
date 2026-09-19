"""Primary demos teach their own data, stay offline, and preserve uncertainty."""

import hashlib
import json
from pathlib import Path
import runpy
import socket

import pytest

from system1.compiler import CompiledSystemOneModel


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["support_triage", "model_routing", "agent_guard"])
def test_primary_teaching_demo_runs_offline_and_saves_its_evidence(tmp_path, monkeypatch, capsys, name):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))

    def no_network(*args, **kwargs):
        pytest.fail("Teaching examples must not contact external services")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    script = runpy.run_path(str(ROOT / "examples" / f"{name}.py"))
    report = script["main"](["--output-dir", str(tmp_path)])
    data_path = ROOT / "examples" / "teaching" / f"{name}.json"
    data = json.loads(data_path.read_text())
    artifact = CompiledSystemOneModel.load(tmp_path / name / "skill.s1m")
    assert artifact.metadata["example_dataset_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert report == json.loads((tmp_path / name / "report.json").read_text())
    assert len(report["predictions"]) == len(data["evaluate"])
    assert report["accuracy"] > report["starter_accuracy"]
    # Compare the published numbers to individual outcomes, including deferrals.
    rows = report["predictions"]
    assert report["accuracy"] == sum(row["prediction"] == row["label"] for row in rows) / len(rows)
    accepted = [row for row in rows if not row["needs_review"]]
    assert report["accepted_count"] == len(accepted)
    assert report["review_count"] == len(rows) - len(accepted)
    if not accepted:
        assert report["accepted_accuracy"] is None
    field = next(iter(artifact.heads))
    assert artifact.metadata["sample_counts"][field]["fit"] == len(data["teach"])
    assert artifact.metadata["sample_counts"][field]["generated"] == 0
    assert len(artifact.heads[field].calibration_scores) == len(data["calibration"]) // 2
    assert "Unseen accuracy:" in capsys.readouterr().out
    if name == "agent_guard":
        # A second run reuses the signing identity and verifies the existing ledger.
        outcomes = script["demonstrate_policy"](tmp_path / name)
        assert outcomes == {"service_name": "ALLOW", "private_key": "DENY"}


@pytest.mark.parametrize("overlap", ["prompt", "group"])
def test_case_loader_rejects_leakage_across_splits(tmp_path, monkeypatch, overlap):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from _teaching_demo import load_cases

    data = {
        "teach": [{"group": "refund-case", "prompt": "Refund my payment", "label": "billing"}],
        "calibration": [{"group": "invoice-case", "prompt": "Correct my invoice", "label": "billing"}],
        "evaluate": [{"group": "receipt-case", "prompt": "Find the receipt", "label": "billing"}],
    }
    if overlap == "prompt":
        data["evaluate"][0]["prompt"] = "  REFUND  MY PAYMENT "
    else:
        data["evaluate"][0]["group"] = "refund-case"
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disjoint"):
        load_cases(path)
