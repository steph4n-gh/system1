"""Experiment 2i: teach the existing OOS intent, holding review data fixed.

Reuse experiment-2e fitting and existing saved formats. All original failures
and source files remain immutable; this runner never opens original tests.
"""
import argparse
import gc
import json
from pathlib import Path
import time
STARTED = time.perf_counter()
import importlib.metadata
import platform

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline_review import meta_features as lexical_meta, midpoint
from boundary_development import BoundaryBaseline, compile_head, logistic_fit, negative_source
from crossfit_probe import probabilities
from develop import OUTPUT, frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from evaluate import deny_network_control, gates, quality
from latest_regression import CANDIDATES, frozen
from review_runtime import ReviewCandidate
from review_teaching import features, meta_features
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler
from teach_boundaries import HERE, REQUESTS, committed, digest

PROTOCOL = HERE / "DIRECT_OOS_PROTOCOL.md"
COMPARISON = HERE / "results/context-runtime.json"

TEACHING = HERE / "results/boundary-teaching"


def fit_one(dataset, kind, lessons, folder, report, condition):
    begin = time.perf_counter()
    bank, neural = dataset == "banking77", kind == "system1"
    data = load_splits(dataset)
    original = data["fit"]
    positive = [r for r in lessons if r["dataset"] == dataset and r["label"] != "oos"]
    new_negative = [r for r in lessons if r["dataset"] == dataset and r["label"] == "oos"]
    earlier = json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"] if bank else []
    assert dataset == "clinc150" and condition in ("positive-control", "direct-oos")
    augmented = original + positive + (new_negative if condition == "direct-oos" else [])
    assert len(positive) == 1696 and len(new_negative) == 674 and not earlier
    labels = sorted({r["label"] for r in original})
    yi = np.asarray([labels.index(r["label"]) for r in original])
    ya = np.asarray([labels.index(r["label"]) for r in augmented])
    yc = np.asarray([labels.index(r["label"]) for r in data["calibration"]])
    folds = np.zeros(len(original), dtype=int)
    for label in labels:
        ids = sorted([i for i, r in enumerate(original) if r["label"] == label], key=lambda i: original[i]["group"])
        for position, index in enumerate(ids):
            folds[index] = position % 3
    # Retain the exact original review-fold assignment. Generated text is never
    # used in these fold heads/prototypes because its prompts reference fitting rows.
    fold_groups = [[original[i]["group"] for i in np.flatnonzero(folds == fold)] for fold in range(3)]
    previous = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"][dataset]
    if fold_groups != previous["fold_groups"]:
        raise ValueError("Original review folds changed")
    wiki = negative_source()
    assert len(wiki) == 2000
    parts = dict(fit=original, augmented=augmented, calibration=data["calibration"], development=data["development"])
    if wiki:
        parts["wiki"] = wiki
    cf, cp = np.zeros((len(original), 7)), np.zeros(len(original), dtype=int)
    preparation = []
    if neural:
        encoder = Encoder("bge-small" if bank else "minilm", threads=4 if bank else 1)
        x = {name: features(rows, encoder, single=True, evidence=preparation) for name, rows in parts.items()}
        schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
        prepared = PreparedFeatures(augmented + data["calibration"], np.concatenate([x["augmented"], x["calibration"]]), encoder)
        compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.01 if bank else .1)
    for fold in range(3):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if neural:
            model = compile_head(compiler, [original[i] for i in train], data["calibration"])
            values = probabilities(model.heads["intent"], x["fit"][valid])
            cf[valid] = meta_features(values, x["fit"][valid], x["fit"][train], yi[train])
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            xf = vectorizer.fit_transform([original[i]["prompt"] for i in train])
            xv = vectorizer.transform([original[i]["prompt"] for i in valid])
            model = logistic_fit(xf, yi[train])
            assert np.array_equal(model.classes_, np.arange(len(labels)))
            values = model.predict_proba(xv)
            cf[valid] = lexical_meta(values, xv, xf, yi[train])
        cp[valid] = values.argmax(axis=1)
        print(f"{dataset}/{kind}: original-only review fold {fold + 1}/3", flush=True)
    intent_start = time.perf_counter()
    if neural:
        taught = compile_head(compiler, augmented, data["calibration"])
        model = CompiledSystemOneModel(taught.schema, taught.heads, dimension=384,
                                      projector=encoder, metadata=taught.metadata, use_cache=False)
        engine = System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)
        predict = lambda vectors: probabilities(model.heads["intent"], vectors)
        meta = meta_features
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        xf = vectorizer.fit_transform([r["prompt"] for r in augmented])
        x = {name: vectorizer.transform([r["prompt"] for r in rows]) for name, rows in parts.items() if name != "augmented"}
        x["augmented"] = xf
        model = logistic_fit(xf, ya)
        assert np.array_equal(model.classes_, np.arange(len(labels)))
        predict = model.predict_proba
        meta = lexical_meta
    intent_fit_ms = (time.perf_counter() - intent_start) * 1000
    values, review, guesses = {}, {}, {}
    for name in (key for key in parts if key not in ("fit", "augmented")):
        values[name] = predict(x[name])
        guesses[name] = values[name].argmax(axis=1)
        review[name] = meta(values[name], x[name], x["augmented"], ya)
    truth = [r["label"] for r in data["development"]]
    pred = np.asarray(labels)[guesses["development"]]
    alpha = .075 if bank else .05
    eligible = ([not engine.decide(row["prompt"], embedding=vector, alpha=alpha, record_receipt=False).is_ambiguous
                 for row, vector in zip(data["development"], x["development"], strict=True)] if neural else None)
    review_correct = np.concatenate([(cp == yi) & np.asarray([r["label"] != "oos" for r in original]),
        (guesses["calibration"] == yc) & np.asarray([r["label"] != "oos" for r in data["calibration"]])])
    contract_path = HERE / "artifacts" / ("banking-review" if bank else "clinc-review-consistent") / "manifest.json"
    contract = json.loads(contract_path.read_text())
    extra = ["wiki"]
    teaching = np.concatenate([cf, review["calibration"]] + [review[name] for name in extra])
    chosen = np.concatenate([cp, guesses["calibration"]] + [guesses[name] for name in extra])
    target = np.concatenate([review_correct] + [np.zeros(len(parts[name]), dtype=bool) for name in extra])
    scaler = StandardScaler().fit(teaching)
    train = np.column_stack([scaler.transform(teaching), np.eye(len(labels))[chosen]])
    development = np.column_stack([scaler.transform(review["development"]), np.eye(len(labels))[guesses["development"]]])
    c = .1 if neural or bank else 1
    start = time.perf_counter()
    gate = LogisticRegression(C=c, max_iter=1000).fit(train, target)
    if gate.n_iter_.max() >= 1000:
        raise RuntimeError("Review head did not converge")
    review_fit_ms = (time.perf_counter() - start) * 1000
    scores = gate.predict_proba(development)[:, 1]
    development_frontier = frontier(truth, pred, scores, eligible=eligible)
    threshold = midpoint(scores, development_frontier["best_development_coverage_at_quality_targets"]["threshold"])
    result_folder = folder / f"{dataset}-{kind}-{condition}"
    result_folder.mkdir()
    manifest = dict(status="unqualified-direct-oos-teaching-development", dataset=dataset, kind=kind,
        condition=condition, categories=contract["categories"], instructions=contract["instructions"],
        predicted_class_feature=True, review_C=c, protocol_sha256=digest(PROTOCOL),
        lessons_sha256=report["lessons_sha256"], fit_rows=len(augmented), review_rows=len(target))
    if neural:
        model.save(result_folder / "intent.s1m")
        order = np.argsort(ya, kind="stable")
        np.savez_compressed(result_folder / "scope.npz", prototypes=x["augmented"][order],
            class_offsets=np.concatenate([[0], np.cumsum(np.bincount(ya, minlength=len(labels)))]),
            mean=scaler.mean_, scale=scaler.scale_, weights=gate.coef_[0], bias=gate.intercept_[0])
        manifest.update(encoder=encoder.identity, guard="category-reliability", alpha=alpha,
            reliability_threshold=threshold, teaching_projection="single-request", regularization=.01 if bank else .1,
            files={name: digest(result_folder / name) for name in ("intent.s1m", "scope.npz")})
    else:
        (result_folder / "vocabulary.json").write_text(json.dumps(vectorizer.vocabulary_, sort_keys=True) + "\n")
        np.savez_compressed(result_folder / "weights.npz", weights=model.coef_, bias=model.intercept_,
            idf=vectorizer.idf_, labels=np.asarray(labels), mean=scaler.mean_, scale=scaler.scale_,
            gate_weights=gate.coef_, gate_bias=gate.intercept_, fit_labels=ya)
        sparse.save_npz(result_folder / "prototypes.npz", x["augmented"])
        manifest.update(C=10, policy="reliability", threshold=threshold, ngram_range=[1, 2], sublinear_tf=True,
            files={name: digest(result_folder / name) for name in ("weights.npz", "prototypes.npz", "vocabulary.json")})
    (result_folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    outcomes = [dict(group=row["group"], truth=row["label"], suggestion=str(pred[i]), score=float(scores[i]),
                    needsReview=bool(pred[i] == "oos" or scores[i] < threshold or (eligible is not None and not eligible[i])),
                    category=None if pred[i] == "oos" or scores[i] < threshold or (eligible is not None and not eligible[i]) else str(pred[i]))
                for i, row in enumerate(data["development"])]
    result = dict(dataset=dataset, kind=kind, condition=condition, folder=result_folder.name,
        manifest_sha256=digest(result_folder / "manifest.json"), manifest=manifest,
        intent_fit_ms=intent_fit_ms, review_fit_ms=review_fit_ms, preparation=preparation,
        development_elapsed_ms=(time.perf_counter() - begin) * 1000,
        original_fit_rows=len(original), earlier_generated_supported=len(earlier),
        new_supported=len(positive), new_unsupported_intent=len(new_negative) if condition == "direct-oos" else 0,
        synthetic_unsupported_review=0, wikipedia_review_rows=len(wiki),
        generated_rows_in_fold_heads=0, fold_groups=fold_groups,
        quality=development_frontier, metrics=quality(outcomes), outcomes=outcomes)
    if neural and condition == "positive-control":
        old = next(r for r in json.loads((HERE / "results/boundary-development.json").read_text())["results"]
                   if (r["dataset"], r["kind"], r["condition"]) == (dataset, kind, "positive-only"))
        assert result["fold_groups"] == old["fold_groups"]
        assert result["metrics"] == old["metrics"]
        delta = max(abs(a["score"] - b["score"]) for a, b in zip(outcomes, old["outcomes"], strict=True))
        assert delta < 1e-10
        assert all(all(a[k] == b[k] for k in ("group", "truth", "suggestion", "category", "needsReview"))
                   for a, b in zip(outcomes, old["outcomes"], strict=True))
        result["control_reconstruction"] = dict(original_routing_matches=True, max_score_difference=delta,
            reference_sha256=digest(HERE / "results/boundary-development.json"))
    report["results"].append(result)
    (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("dataset", "kind", "condition", "metrics", "intent_fit_ms", "review_fit_ms")}), flush=True)
    del model, predict
    if neural:
        del engine, taught, compiler, prepared, encoder
    gc.collect()


def fit(folder):
    paths = [PROTOCOL, Path(__file__).resolve(), REQUESTS, COMPARISON,
             HERE / "results/boundary-development.json", HERE / "review-data-manifest.json"] + [
             TEACHING / name for name in ("teacher.json", "lessons.json", "summary.json")]
    revision = committed(paths)
    frozen(False)
    denied = deny_network_control()
    summary = json.loads((TEACHING / "summary.json").read_text())
    if summary["unfinished_or_unattempted_slots"] or summary["request_manifest_sha256"] != digest(REQUESTS):
        raise ValueError("Teaching evidence incomplete or changed")
    lessons = json.loads((TEACHING / "lessons.json").read_text())["lessons"]
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(scope="experiment 2i CLINC development; banking incumbents unchanged; no tests or reserves scored", qualified=False,
        source_revision=revision, protocol_sha256=digest(PROTOCOL), lessons_sha256=digest(TEACHING / "lessons.json"),
        teacher_summary_sha256=digest(TEACHING / "summary.json"), os_network_denial_errno=denied,
        new_teacher_calls=0, new_api_cost=0, source_files={str(p.relative_to(HERE.parents[2])): digest(p) for p in paths},
        comparison_report_sha256=digest(COMPARISON), results=[])
    with threadpool_limits(limits=1):
        for kind in ("system1", "baseline"):
            for condition in ("positive-control", "direct-oos"):
                try:
                    fit_one("clinc150", kind, lessons, folder, report, condition)
                except Exception as exc:
                    report["failure"] = dict(kind=kind, condition=condition, error=type(exc).__name__, message=str(exc))
                    (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
                    raise


def compare_incumbents(report):
    """Use retained complete development runs; ties keep the prior artifact."""
    report["comparisons"] = []
    prior = json.loads(COMPARISON.read_text())
    for dataset in ("clinc150", "banking77"):
        for kind in ("system1", "baseline"):
            incumbent = next(r["selected_development_candidate"] for r in prior["comparisons"]
                             if (r["dataset"], r["kind"]) == (dataset, kind))
            # Historical reports use pre-publication folder names for polynomial
            # artifacts. The frozen selection maps them to their published paths.
            assert incumbent["candidate_sha256"] == digest(HERE / "artifacts" / CANDIDATES[dataset][kind] / "manifest.json")
            winner = incumbent
            for result in report["results"]:
                if (result["dataset"], result["kind"]) != (dataset, kind):
                    continue
                valid = all(value for key, value in result["development_gates"].items() if key != "coverage_at_least_80")
                if valid and not result["selection_runtime_mismatches"] and result["metrics"]["supported_coverage"]["numerator"] > winner["metrics"]["supported_coverage"]["numerator"]:
                    winner = {key: result[key] for key in ("folder", "candidate_sha256", "metrics", "latency")}
                    winner["source"] = result["condition"]
            report["comparisons"].append(dict(dataset=dataset, kind=kind, incumbent=incumbent,
                selected_development_candidate=winner, qualified=False,
                incumbent_timings_from_prior_run=True))


def measure(folder):
    frozen(False)
    denied = deny_network_control()
    source = json.loads((folder / "development.json").read_text())
    for name, expected in source["source_files"].items():
        if digest(HERE.parents[2] / name) != expected:
            raise ValueError(f"Declared source changed: {name}")
    path = folder / "runtime.json"
    with path.open("x") as stream, path.with_suffix(".jsonl").open("x") as journal, threadpool_limits(limits=1):
        report = dict(scope="experiment 2i complete CLINC development adapters; not qualification", qualified=False,
            source_report_sha256=digest(folder / "development.json"), source_revision=source["source_revision"],
            protocol_sha256=source["protocol_sha256"], comparison_report_sha256=source["comparison_report_sha256"],
            os_network_denial_errno=denied, teacher_calls=0, new_api_cost=0, receipts=False, response_cache=False,
            process_import_preflight_ms=(time.perf_counter() - STARTED) * 1000,
            environment=dict(platform=platform.platform(), python=platform.python_version(), machine=platform.machine()),
            packages={name: importlib.metadata.version(name) for name in
                      ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")}, results=[])
        assert len(source["results"]) == 4 and "failure" not in source
        for selected in source["results"]:
            candidate_folder = folder / selected["folder"]
            begin = time.perf_counter()
            candidate = (ReviewCandidate if selected["kind"] == "system1" else BoundaryBaseline)(candidate_folder)
            load_ms = (time.perf_counter() - begin) * 1000
            assert candidate.identity == selected["manifest_sha256"]
            rows = load_splits(selected["dataset"])["development"]
            payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
            for row in rows[:100]:
                candidate.classify(dict(payload, text=row["prompt"]))
            outcomes, times, mismatches = [], [], []
            for row, expected in zip(rows, selected["outcomes"], strict=True):
                begin = time.perf_counter()
                try:
                    value = candidate.classify(dict(payload, text=row["prompt"]))
                except Exception as exc:
                    value = dict(candidate=candidate.identity, needsReview=True, category=None,
                                 suggestion=None, teacherCalls=0, error=type(exc).__name__)
                elapsed = (time.perf_counter() - begin) * 1000
                if any(value[k] != expected[k] for k in ("needsReview", "suggestion", "category")):
                    mismatches.append(row["group"])
                times.append(elapsed)
                outcome = dict(group=row["group"], truth=row["label"], latency_ms=elapsed, **value)
                outcomes.append(outcome)
                journal.write(json.dumps(dict(folder=selected["folder"], **outcome)) + "\n")
                journal.flush()
            measured = quality(outcomes)
            errors = sum("error" in row for row in outcomes)
            result = dict(dataset=selected["dataset"], kind=selected["kind"], condition=selected["condition"],
                folder=selected["folder"], candidate_sha256=candidate.identity, manifest=candidate.manifest,
                metrics=measured, latency=dict(requests=len(rows), warmups=100,
                    p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))),
                development_gates=gates(measured, float(np.percentile(times, 95)), errors), qualified=False, errors=errors,
                load_ms=load_ms, artifact_bytes=sum(p.stat().st_size for p in candidate_folder.iterdir() if p.is_file()),
                external_encoder_bytes=sum(candidate.manifest.get("encoder", {}).get(k, 0) for k in ("model_bytes", "tokenizer_bytes")),
                selection_runtime_mismatches=mismatches,
                accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]],
                per_intent={label: quality([r for r in outcomes if r["truth"] == label]) for label in sorted(set(r["truth"] for r in outcomes))},
                outcomes=outcomes)
            report["results"].append(result)
            stream.seek(0)
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.truncate()
            stream.flush()
            print(json.dumps({k: result[k] for k in ("dataset", "kind", "condition", "metrics", "latency", "selection_runtime_mismatches")}), flush=True)
            del candidate
            gc.collect()
        compare_incumbents(report)
        stream.seek(0)
        stream.write(json.dumps(report, indent=2) + "\n")
        stream.truncate()
    if any(r["selection_runtime_mismatches"] for r in report["results"]):
        raise RuntimeError("Runtime/selection mismatch; failed results retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "measure"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.action == "fit" else measure)(args.folder)
