"""Fixed fitting-fold consistency diagnostic; never relabels or selects a head."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from develop import load_splits
from evaluate import deny_network_control
from latest_regression import frozen
from polynomial_review import prepare
from review_teaching import features
from teach_boundaries import HERE, ROOT, committed, digest

PROTOCOL = HERE / "TEACHING_CONSISTENCY_PROTOCOL.md"
BINS = [0, .5, .9, .95, .99, 1]


def nearest(vector, rows, matrix, indices, assigned, suggested):
    indices = np.asarray(indices, dtype=int)
    scores = matrix[indices] @ vector
    order = np.argsort(-scores, kind="stable")
    def item(position):
        row = rows[int(indices[position])]
        return dict(group=row["group"], prompt=row["prompt"], label=row["label"], similarity=float(scores[position]))
    matches = {}
    for key, label in (("assigned", assigned), ("suggested", suggested)):
        positions = [int(i) for i in order if rows[int(indices[i])]["label"] == label]
        matches[key] = item(positions[0]) if positions else None
    return dict(top_three=[item(int(i)) for i in order[:3]], nearest_assigned=matches["assigned"], nearest_suggested=matches["suggested"])


def summarize(outcomes):
    def counts(rows):
        return dict(rows=len(rows), semantic_disagreements=sum(r["semantic_suggestion"] != r["label"] for r in rows),
            lexical_disagreements=sum(r["lexical_suggestion"] != r["label"] for r in rows),
            consensus_disagreements=sum(r["consensus_disagreement"] for r in rows))
    result = dict(all=counts(outcomes), supported=counts([r for r in outcomes if r["label"] != "oos"]),
        oos=counts([r for r in outcomes if r["label"] == "oos"]), semantic_probability_bins=[])
    for i, (lower, upper) in enumerate(zip(BINS[:-1], BINS[1:], strict=True)):
        selected = [r for r in outcomes if lower <= r["semantic_confidence"] and
            (r["semantic_confidence"] < upper or (i == len(BINS) - 2 and r["semantic_confidence"] <= upper))]
        result["semantic_probability_bins"].append(dict(lower_inclusive=lower, upper=upper,
            upper_inclusive=i == len(BINS) - 2, **counts(selected)))
    return result


def audit_one(dataset):
    started = time.perf_counter()
    data = load_splits(dataset)
    original = data["fit"]
    assert len(original) == (12019 if dataset == "clinc150" else 6026)
    predictions, evidence = {}, {}
    for kind in ("system1", "baseline"):
        parent, labels, _, teaching, chosen, _, _, _, _, details = prepare(dataset, kind)
        predictions[kind] = dict(labels=labels, chosen=chosen[:len(original)].copy(),
            confidence=1 / (1 + np.exp(-teaching[:len(original), 0])))
        evidence[kind] = details
        if kind == "system1":
            encoder = parent.encoder
        del parent
        gc.collect()
    assert predictions["system1"]["labels"] == predictions["baseline"]["labels"]
    assert evidence["system1"]["fold_groups"] == evidence["baseline"]["fold_groups"]
    folds = evidence["system1"]["fold_groups"]
    fold_by_group = {group: fold for fold, groups in enumerate(folds) for group in groups}
    assert set(fold_by_group) == {r["group"] for r in original}
    labels = predictions["system1"]["labels"]
    outcomes = []
    for i, row in enumerate(original):
        s, b = (predictions[kind] for kind in ("system1", "baseline"))
        semantic, lexical = labels[s["chosen"][i]], labels[b["chosen"][i]]
        outcomes.append(dict(group=row["group"], prompt=row["prompt"], label=row["label"], fold=fold_by_group[row["group"]],
            semantic_suggestion=semantic, semantic_confidence=float(s["confidence"][i]),
            lexical_suggestion=lexical, lexical_confidence=float(b["confidence"][i]),
            consensus_disagreement=bool(semantic == lexical and semantic != row["label"])))
    extraction = []
    x = features(original, encoder, single=True, evidence=extraction)
    ordered = sorted([r for r in outcomes if r["consensus_disagreement"]],
        key=lambda r: (-r["semantic_confidence"], -r["lexical_confidence"], r["group"]))[:50]
    indices = {r["group"]: i for i, r in enumerate(original)}
    examples = []
    for row in ordered:
        training = [i for i, r in enumerate(original) if fold_by_group[r["group"]] != row["fold"]]
        examples.append(dict(row, neighbors=nearest(x[indices[row["group"]]], original, x, training, row["label"], row["semantic_suggestion"])))
    path = HERE / "results" / ("context-runtime.json" if dataset == "clinc150" else "polynomial-runtime.json")
    source = json.loads(path.read_text())
    incumbent = next(r for r in source["results"] if r["dataset"] == dataset and r["kind"] == "system1"
        and (dataset == "clinc150" or r["condition"] == "quadratic"))
    development = {r["group"]: r for r in data["development"]}
    error_rows = [r for r in incumbent["outcomes"] if not r["needsReview"] and r["category"] != r["truth"]]
    error_vectors = features([development[r["group"]] for r in error_rows], encoder, single=True, evidence=extraction)
    errors = []
    for row, vector in zip(error_rows, error_vectors, strict=True):
        original_row = development[row["group"]]
        assert row["truth"] == original_row["label"]
        errors.append(dict(group=row["group"], prompt=original_row["prompt"], label=row["truth"], suggestion=row["suggestion"],
            confidence=row["confidence"], review_score=row["reliability"],
            neighbors=nearest(vector, original, x, np.arange(len(original)), row["truth"], row["suggestion"])))
    return dict(dataset=dataset, encoder=encoder.identity, preparation=evidence, feature_extraction=extraction,
        data_sha256={part: digest(ROOT / ".system1/n8n-gauntlet" / f"{dataset}-{part}.json") for part in data},
        summary=summarize(outcomes), outcomes=outcomes, prioritized_fitting_examples=examples,
        development=dict(source=str(path.relative_to(ROOT)), sha256=digest(path), candidate_sha256=incumbent["candidate_sha256"],
            accepted_errors=len(errors), examples=errors, post_hoc=True),
        total_preparation_ms=(time.perf_counter() - started) * 1000)


def main(output):
    sources = [PROTOCOL, Path(__file__).resolve(), HERE / "polynomial_review.py", HERE / "review_teaching.py",
        HERE / "results/context-runtime.json", HERE / "results/polynomial-runtime.json", HERE / "results/human-only-runtime.json",
        HERE / "manifest.json", HERE / "review-data-manifest.json"]
    revision = committed(sources)
    frozen(False)
    report = dict(scope="experiment 2o fitting-label consistency and post-hoc development diagnosis; not qualification", qualified=False,
        source_revision=revision, source_files={str(p.relative_to(ROOT)): digest(p) for p in sources},
        os_network_denial_errno=deny_network_control(), teacher_calls=0, new_api_cost=0, label_changes=0, final_head_changes=0,
        independent_annotations=False, environment=dict(platform=platform.platform(), python=platform.python_version()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")}, results=[])
    with output.open("x") as stream, threadpool_limits(limits=1):
        try:
            for dataset in ("clinc150", "banking77"):
                result = audit_one(dataset)
                report["results"].append(result)
                stream.seek(0)
                stream.write(json.dumps(report, indent=2) + "\n")
                stream.truncate()
                stream.flush()
                print(json.dumps(dict(dataset=dataset, summary=result["summary"], development_errors=result["development"]["accepted_errors"])), flush=True)
                gc.collect()
        except Exception as exc:
            report["failure"] = dict(error=type(exc).__name__, message=str(exc))
            stream.seek(0)
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.truncate()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
