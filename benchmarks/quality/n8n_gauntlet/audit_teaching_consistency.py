"""Verify recorded fitting-fold consistency evidence without fitting or inference."""
import hashlib
import csv
import io
import json
from pathlib import Path
import re

import numpy as np
from threadpoolctl import threadpool_limits

from audit_observed_regression import digest, same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / ".system1/n8n-gauntlet"


def cached(rows, identity):
    encoder_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    key = hashlib.sha256(("single-request-v1:" + encoder_digest + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
    path = CACHE / f"features-{key}.npy"
    matrix = np.load(path, allow_pickle=False)
    assert matrix.shape == (len(rows), 384) and np.isfinite(matrix).all()
    return matrix, dict(path=str(path.relative_to(ROOT)), sha256=digest(path), rows=len(rows))


def verify_neighbors(actual, vector, matrix, source, allowed, assigned, suggested):
    indices = np.asarray(allowed, dtype=int)
    similarities = matrix[indices] @ vector
    ordered = np.argsort(-similarities, kind="stable")
    def expected(position):
        row = source[int(indices[position])]
        return dict(group=row["group"], prompt=row["prompt"], label=row["label"], similarity=float(similarities[position]))
    same(actual["top_three"], [expected(int(i)) for i in ordered[:3]])
    for key, label in (("nearest_assigned", assigned), ("nearest_suggested", suggested)):
        matching = [int(i) for i in ordered if source[int(indices[i])]["label"] == label]
        same(actual[key], expected(matching[0]) if matching else None)


def counts(rows):
    return dict(rows=len(rows), semantic_disagreements=sum(r["semantic_suggestion"] != r["label"] for r in rows),
        lexical_disagreements=sum(r["lexical_suggestion"] != r["label"] for r in rows),
        consensus_disagreements=sum(r["semantic_suggestion"] == r["lexical_suggestion"] and r["semantic_suggestion"] != r["label"] for r in rows))


def main():
    path = HERE / "results/teaching-consistency.json"
    report = json.loads(path.read_text())
    proposals_path = HERE / "results/teaching-correction-proposals.json"
    proposals = json.loads(proposals_path.read_text())
    assert proposals["status"] == "proposed-not-applied" and proposals["independent_annotations"] is False
    assert proposals["source_report_sha256"] == digest(path)
    assert len(proposals["proposals"]) == len({r["group"] for r in proposals["proposals"]}) == 7
    assert report["qualified"] is report["independent_annotations"] is False and "failure" not in report
    assert report["teacher_calls"] == report["new_api_cost"] == report["label_changes"] == report["final_head_changes"] == 0
    assert report["os_network_denial_errno"] == 1
    freeze = json.loads((HERE / "latest-regression-freeze.json").read_text())
    for name, expected in dict(freeze["files"], **report["source_files"]).items():
        assert digest(ROOT / name) == expected, name
    assert [r["dataset"] for r in report["results"]] == ["clinc150", "banking77"]
    verified, cache_evidence, lineage = [], [], []
    for result in report["results"]:
        dataset = result["dataset"]
        source = json.loads((HERE / "manifest.json").read_text())["datasets"][dataset]["splits"]
        data = {}
        for part in ("fit", "calibration", "development"):
            p = CACHE / f"{dataset}-{part}.json"
            assert result["data_sha256"][part] == digest(p) == source[part]["sha256"]
            data[part] = json.loads(p.read_text())
        # Select only permitted sections of CLINC's monolithic source. Test
        # sections are neither decoded into rows nor inspected by this audit.
        filename = "clinc.json" if dataset == "clinc150" else "banking-train.csv"
        raw = (ROOT / ".system1/workload-sources" / filename).read_bytes()
        expected_source = json.loads((HERE / "manifest.json").read_text())["sources"][filename]
        assert hashlib.sha256(raw).hexdigest() == expected_source["sha256"]
        if dataset == "clinc150":
            text, original_sections = raw.decode(), {}
            for key in ("train", "oos_train", "val", "oos_val"):
                matches = list(re.finditer(r'(?m)^[ \t]*"' + key + r'"[ \t]*:\s*', text))
                assert len(matches) == 1
                original_sections[key], _ = json.JSONDecoder().raw_decode(text, matches[0].end())
        else:
            original_sections = {"train": [(r["text"], r["category"]) for r in csv.DictReader(io.StringIO(raw.decode()))]}
        for part, entries in data.items():
            for entry in entries:
                prompt, label = original_sections[entry["source"]][entry["index"]]
                assert (prompt, label) == (entry["prompt"], entry["label"])
        lineage.append(dict(dataset=dataset, filename=filename, sha256=expected_source["sha256"],
            url=expected_source["url"], sections_decoded=list(original_sections),
            rows_verified=sum(len(entries) for entries in data.values()), import_label_mismatches=0))
        original, rows = data["fit"], result["outcomes"]
        assert len(original) == len(rows) == (12019 if dataset == "clinc150" else 6026)
        labels = {r["label"] for r in original}
        assert proposals["fit_sha256"][dataset] == result["data_sha256"]["fit"]
        for proposal in (r for r in proposals["proposals"] if r["dataset"] == dataset):
            original_row = next(r for r in original if r["group"] == proposal["group"])
            assert all(proposal[k] == original_row[k] for k in ("prompt", "source", "index"))
            assert proposal["original_label"] == original_row["label"]
            assert proposal["proposed_label"] in labels and proposal["proposed_label"] != proposal["original_label"] and proposal["reason"]
            diagnostic = next(r for r in rows if r["group"] == proposal["group"])
            assert diagnostic["semantic_suggestion"] == diagnostic["lexical_suggestion"] == proposal["proposed_label"]
        original_folds = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"][dataset]["fold_groups"]
        for kind in ("system1", "baseline"):
            evidence = result["preparation"][kind]
            assert evidence["fold_groups"] == original_folds and evidence["generated_examples_in_fold_heads"] == 0
            assert evidence["original_fit_rows"] == len(original)
        folds = {group: fold for fold, groups in enumerate(original_folds) for group in groups}
        for row, entry in zip(rows, original, strict=True):
            assert all(row[k] == entry[k] for k in ("group", "prompt", "label"))
            assert row["fold"] == folds[row["group"]]
            assert row["semantic_suggestion"] in labels and row["lexical_suggestion"] in labels
            assert 0 <= row["semantic_confidence"] <= 1 and 0 <= row["lexical_confidence"] <= 1
            assert row["consensus_disagreement"] == (row["semantic_suggestion"] == row["lexical_suggestion"] and row["semantic_suggestion"] != row["label"])
        summary = result["summary"]
        for name, selected in (("all", rows), ("supported", [r for r in rows if r["label"] != "oos"]), ("oos", [r for r in rows if r["label"] == "oos"])):
            assert summary[name] == counts(selected)
        edges = [0, .5, .9, .95, .99, 1]
        assert len(summary["semantic_probability_bins"]) == 5
        for i, entry in enumerate(summary["semantic_probability_bins"]):
            lower, upper = edges[i:i + 2]
            selected = [r for r in rows if lower <= r["semantic_confidence"] < upper or (i == 4 and r["semantic_confidence"] == 1)]
            assert entry == dict(lower_inclusive=lower, upper=upper, upper_inclusive=i == 4, **counts(selected))
        matrix, matrix_evidence = cached(original, result["encoder"])
        cache_evidence.append(matrix_evidence)
        by_group = {r["group"]: i for i, r in enumerate(original)}
        ordered = sorted([r for r in rows if r["consensus_disagreement"]], key=lambda r: (-r["semantic_confidence"], -r["lexical_confidence"], r["group"]))[:50]
        assert len(ordered) == len(result["prioritized_fitting_examples"])
        for row, expected in zip(result["prioritized_fitting_examples"], ordered, strict=True):
            assert {k: v for k, v in row.items() if k != "neighbors"} == expected
            allowed = [i for i, r in enumerate(original) if folds[r["group"]] != row["fold"]]
            verify_neighbors(row["neighbors"], matrix[by_group[row["group"]]], matrix, original, allowed, row["label"], row["semantic_suggestion"])
        dev = result["development"]
        report_path = ROOT / dev["source"]
        assert digest(report_path) == dev["sha256"] and dev["post_hoc"] is True
        candidates = json.loads(report_path.read_text())["results"]
        incumbent = next(r for r in candidates if r["candidate_sha256"] == dev["candidate_sha256"])
        errors = [r for r in incumbent["outcomes"] if not r["needsReview"] and r["category"] != r["truth"]]
        assert len(errors) == len(dev["examples"]) == dev["accepted_errors"]
        development = {r["group"]: r for r in data["development"]}
        vectors, vector_evidence = cached([development[r["group"]] for r in errors], result["encoder"])
        cache_evidence.append(vector_evidence)
        for row, expected, vector in zip(dev["examples"], errors, vectors, strict=True):
            assert row["group"] == expected["group"] and row["label"] == expected["truth"]
            assert row["prompt"] == development[row["group"]]["prompt"]
            assert row["suggestion"] == expected["suggestion"] and row["confidence"] == expected["confidence"] and row["review_score"] == expected["reliability"]
            verify_neighbors(row["neighbors"], vector, matrix, original, np.arange(len(original)), row["label"], row["suggestion"])
        verified.append(dict(dataset=dataset, fitting_outcomes=len(rows), prioritized_examples=len(ordered), development_errors=len(errors), summary=summary["all"]))
    print(json.dumps(dict(scope="teaching-consistency record and cached-neighbor audit; no fits or model inference", qualified=False,
        report_sha256=digest(path), frozen_files_unchanged=len(freeze["files"]), declared_sources_verified=len(report["source_files"]),
        cache_files=cache_evidence, source_lineage=lineage, results=verified, independent_annotations=False, labels_changed=0,
        correction_proposals=dict(sha256=digest(proposals_path), verified_fitting_sources=7, applied=False)), indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
