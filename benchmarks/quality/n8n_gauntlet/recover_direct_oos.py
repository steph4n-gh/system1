"""Recover the completed 2i measurement journal; never run or fit a classifier."""
import ast
import importlib.metadata
import json
import platform
import subprocess

import numpy as np

from direct_oos_teaching import COMPARISON, HERE, OUTPUT, compare_incumbents
from evaluate import deny_network_control, digest, gates, quality


def main():
    folder = OUTPUT / "direct-oos-development"
    source_path, journal_path = folder / "development.json", folder / "runtime.jsonl"
    log_path = OUTPUT / "direct-oos-runtime.log"
    source = json.loads(source_path.read_text())
    relative = str(HERE.relative_to(HERE.parents[2]) / "direct_oos_teaching.py")
    original_code = subprocess.check_output(["git", "show", f'{source["source_revision"]}:{relative}'], cwd=HERE.parents[2], text=True)
    # The reporting repair must not alter any fitting or decision method.
    old_ast, new_ast = (ast.parse(code) for code in (original_code, (HERE / "direct_oos_teaching.py").read_text()))
    for name in ("fit", "fit_one"):
        old, new = (next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for tree in (old_ast, new_ast))
        assert ast.dump(old) == ast.dump(new)
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    summaries = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith('{"dataset":')]
    assert len(journal) == 12380 and len(summaries) == len(source["results"]) == 4
    assert (folder / "runtime.json").read_bytes() == b""
    assert "FileNotFoundError" in log_path.read_text() and "banking77-system1-quadratic/manifest.json" in log_path.read_text()
    report = dict(scope="experiment 2i complete CLINC development adapters; recovered records, not qualification", qualified=False,
        source_report_sha256=digest(source_path), source_revision=source["source_revision"], protocol_sha256=source["protocol_sha256"],
        comparison_report_sha256=digest(COMPARISON), os_network_denial_errno=deny_network_control(),
        teacher_calls=0, new_api_cost=0, receipts=False, response_cache=False, process_import_preflight_ms=None,
        environment=dict(platform=platform.platform(), python=platform.python_version(), machine=platform.machine()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")},
        recovery=dict(reason="Published polynomial artifact name differs from historical result folder; failure after all measured rows",
            original_exit_code=1, original_journal_sha256=digest(journal_path), original_log_sha256=digest(log_path),
            original_empty_report_sha256=digest(folder / "runtime.json"), rerun_requests=0,
            lost_metadata=["original process import/preflight time", "original per-candidate load times"],
            environment_and_network_probe="recorded during recovery; original passed mandatory OS-denial check before inference",
            fitting_functions_unchanged=True), results=[])
    for selected, summary in zip(source["results"], summaries, strict=True):
        outcomes = [{k: v for k, v in row.items() if k != "folder"} for row in journal if row["folder"] == selected["folder"]]
        assert len(outcomes) == 3095
        mismatches = [row["group"] for row, expected in zip(outcomes, selected["outcomes"], strict=True)
                      if any(row[k] != expected[k] for k in ("group", "truth", "suggestion", "category", "needsReview"))]
        times = [row["latency_ms"] for row in outcomes]
        measured = quality(outcomes)
        latency = dict(requests=3095, warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)))
        assert measured == summary["metrics"] and latency == summary["latency"] and mismatches == summary["selection_runtime_mismatches"] == []
        assert all(selected[k] == summary[k] for k in ("dataset", "kind", "condition"))
        candidate_folder = folder / selected["folder"]
        assert digest(candidate_folder / "manifest.json") == selected["manifest_sha256"]
        errors = sum("error" in row for row in outcomes)
        result = {k: selected[k] for k in ("dataset", "kind", "condition", "folder", "manifest")}
        result.update(candidate_sha256=selected["manifest_sha256"], metrics=measured, latency=latency,
            development_gates=gates(measured, latency["p95_ms"], errors), qualified=False, errors=errors, load_ms=None,
            artifact_bytes=sum(p.stat().st_size for p in candidate_folder.iterdir() if p.is_file()),
            external_encoder_bytes=sum(selected["manifest"].get("encoder", {}).get(k, 0) for k in ("model_bytes", "tokenizer_bytes")),
            selection_runtime_mismatches=mismatches,
            accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]],
            per_intent={label: quality([r for r in outcomes if r["truth"] == label]) for label in sorted({r["truth"] for r in outcomes})},
            outcomes=outcomes)
        report["results"].append(result)
    compare_incumbents(report)
    with (folder / "recovered-runtime.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(recovered_requests=len(journal), rerun_requests=0, all_incumbents_retained=all(
        r["selected_development_candidate"] == r["incumbent"] for r in report["comparisons"]))))


if __name__ == "__main__":
    main()
