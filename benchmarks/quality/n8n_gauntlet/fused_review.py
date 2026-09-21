"""Experiment 2k: fixed fused head, original-fold review, complete adapter."""
import time
STARTED = time.perf_counter()

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline_review import midpoint
from boundary_development import compile_head, negative_source
from crossfit_probe import probabilities
from develop import frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from evaluate import deny_network_control, gates, quality
from fused_features import FusedFeatures, join_features
from fused_review_runtime import FusedReviewCandidate, semantic_component
from latest_regression import frozen
from review_teaching import features, meta_features
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler
from system1.core.text import TfidfProjector
from teach_boundaries import HERE, ROOT, committed, digest

PROTOCOL = HERE / "FUSED_REVIEW_PROTOCOL.md"
COMPARISON = HERE / "results/direct-oos-runtime.json"
HEAD = HERE / "artifacts/fused-clinc150-semantic-plus-words"


def combined(rows, semantic, lexical):
    return np.stack([join_features(vector, lexical.project(row["prompt"])) for row, vector in zip(rows, semantic, strict=True)])


def fit(folder):
    paths = [PROTOCOL, Path(__file__).resolve(), HERE / "fused_review_runtime.py", HERE / "fused_features.py", COMPARISON,
             HERE / "results/fused-head-development.json", HERE / "review-data-manifest.json"] + list(HEAD.iterdir())
    revision = committed(paths)
    frozen(False)
    denied = deny_network_control()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(scope="experiment 2k CLINC development review; banking unchanged; not qualification", qualified=False,
        source_revision=revision, source_files={str(p.relative_to(ROOT)): digest(p) for p in paths},
        protocol_sha256=digest(PROTOCOL), comparison_report_sha256=digest(COMPARISON),
        os_network_denial_errno=denied, teacher_calls=0, new_api_cost=0, results=[])
    with threadpool_limits(limits=1):
        try:
            fit_review(folder, report)
        except Exception as exc:
            report["failure"] = dict(error=type(exc).__name__, message=str(exc))
            (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
            raise


def fit_review(folder, report):
    begin = time.perf_counter()
    data = load_splits("clinc150")
    original, calibration, development = data["fit"], data["calibration"], data["development"]
    head_manifest = json.loads((HEAD / "manifest.json").read_text())
    for name, expected in head_manifest["files"].items():
        assert digest(HEAD / name) == expected
    encoder = Encoder("minilm", threads=1)
    assert encoder.identity == head_manifest["encoder"]
    extraction = []
    parts = dict(fit=original, calibration=calibration, development=development, negative=negative_source())
    semantic = {key: features(rows, encoder, single=True, evidence=extraction) for key, rows in parts.items()}
    labels = sorted({r["label"] for r in original})
    yi = np.asarray([labels.index(r["label"]) for r in original])
    yc = np.asarray([labels.index(r["label"]) for r in calibration])
    folds = np.zeros(len(original), dtype=int)
    for label in labels:
        ids = sorted([i for i, r in enumerate(original) if r["label"] == label], key=lambda i: original[i]["group"])
        for position, index in enumerate(ids):
            folds[index] = position % 3
    fold_groups = [[original[i]["group"] for i in np.flatnonzero(folds == fold)] for fold in range(3)]
    previous = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"]["clinc150"]
    assert fold_groups == previous["fold_groups"]
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    cross, guesses = np.zeros((len(original), 7)), np.zeros(len(original), dtype=int)
    fold_evidence = []
    for fold in range(3):
        started = time.perf_counter()
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        training_rows, valid_rows = [original[i] for i in train], [original[i] for i in valid]
        lexical = TfidfProjector.fit([r["prompt"] for r in training_rows], max_features=2048)
        projector = FusedFeatures(encoder, lexical)
        xt, xv, xc = (combined(rows, vectors, lexical) for rows, vectors in
                      ((training_rows, semantic["fit"][train]), (valid_rows, semantic["fit"][valid]), (calibration, semantic["calibration"])))
        prepared = PreparedFeatures(training_rows + calibration, np.concatenate([xt, xc]), projector)
        temporary = compile_head(SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.1), training_rows, calibration)
        values = probabilities(temporary.heads["intent"], xv)
        queries = np.stack([semantic_component(vector) for vector in xv])
        cross[valid] = meta_features(values, queries, semantic["fit"][train], yi[train])
        guesses[valid] = values.argmax(axis=1)
        fold_evidence.append(dict(fold=fold, fit_rows=len(train), held_out_rows=len(valid),
            lexical_fit_groups=[r["group"] for r in training_rows], lexical_digest=lexical.projector_digest(),
            elapsed_ms=(time.perf_counter() - started) * 1000))
        print(f"CLINC fused review: original-only fold {fold + 1}/3", flush=True)
        del temporary, prepared, xt, xv, xc
        gc.collect()
    lexical = TfidfProjector.from_config(json.loads((HEAD / "lexical.json").read_text()))
    projector = FusedFeatures(encoder, lexical)
    model = CompiledSystemOneModel.load(HEAD / "intent.s1m", projector=projector)
    model.use_cache = False
    engine = System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)
    assert list(model.heads["intent"].options) == labels
    x = {key: combined(rows, semantic[key], lexical) for key, rows in parts.items() if key != "fit"}
    values = {key: probabilities(model.heads["intent"], matrix) for key, matrix in x.items() if key != "development"}
    head_record = next(r for r in json.loads((HERE / "results/fused-head-development.json").read_text())["results"]
                       if r["dataset"] == "clinc150" and r["condition"] == "semantic-plus-words")
    assert digest(HEAD / "manifest.json") == head_record["candidate_sha256"]
    probabilities_dev, eligible, head_differences = [], [], []
    for row, vector, expected in zip(development, x["development"], head_record["outcomes"], strict=True):
        decision = engine.decide(row["prompt"], embedding=vector, alpha=.05, record_receipt=False)
        assert row["group"] == expected["group"] and row["label"] == expected["truth"]
        assert decision.values["intent"] == expected["suggestion"] and decision.conformal_sets["intent"] == expected["predictionSet"]
        assert (not decision.is_ambiguous) == expected["strict_eligible"]
        head_differences.append(abs(decision.probabilities["intent"][expected["suggestion"]] - expected["confidence"]))
        probabilities_dev.append([decision.probabilities["intent"][label] for label in labels])
        eligible.append(not decision.is_ambiguous)
    assert max(head_differences) < 1e-5
    values["development"] = np.asarray(probabilities_dev)
    predicted = {key: p.argmax(axis=1) for key, p in values.items()}
    meta = {key: meta_features(p, np.stack([semantic_component(vector) for vector in x[key]]), semantic["fit"], yi)
            for key, p in values.items()}
    teaching = np.concatenate([cross, meta["calibration"], meta["negative"]])
    categories = np.concatenate([guesses, predicted["calibration"], predicted["negative"]])
    targets = np.concatenate([(guesses == yi) & (np.asarray([r["label"] for r in original]) != "oos"),
        (predicted["calibration"] == yc) & (np.asarray([r["label"] for r in calibration]) != "oos"), np.zeros(len(parts["negative"]), dtype=bool)])
    preparation_ms = (time.perf_counter() - begin) * 1000
    scaler = StandardScaler().fit(teaching)
    train = np.column_stack([scaler.transform(teaching), np.eye(len(labels))[categories]])
    dev = np.column_stack([scaler.transform(meta["development"]), np.eye(len(labels))[predicted["development"]]])
    started = time.perf_counter()
    review = LogisticRegression(C=.1, max_iter=1000).fit(train, targets)
    assert review.n_iter_.max() < 1000
    fit_ms = (time.perf_counter() - started) * 1000
    scores = review.predict_proba(dev)[:, 1]
    suggestions = np.asarray(labels)[predicted["development"]]
    selection = frontier([r["label"] for r in development], suggestions, scores, eligible=eligible)
    threshold = midpoint(scores, selection["best_development_coverage_at_quality_targets"]["threshold"])
    saved = folder / "clinc150-fused-review"
    saved.mkdir()
    for name in ("intent.s1m", "lexical.json"):
        shutil.copyfile(HEAD / name, saved / name)
    order = np.argsort(yi, kind="stable")
    np.savez_compressed(saved / "scope.npz", prototypes=semantic["fit"][order],
        class_offsets=np.concatenate([[0], np.cumsum(np.bincount(yi, minlength=len(labels)))]),
        mean=scaler.mean_, scale=scaler.scale_, weights=review.coef_[0], bias=review.intercept_[0])
    contract = json.loads((HERE / "artifacts/clinc-review-consistent/manifest.json").read_text())
    manifest = dict(head_manifest, status="unqualified-fused-review-development", kind="system1", condition="fused-review",
        head_source_manifest_sha256=digest(HEAD / "manifest.json"), protocol_sha256=digest(PROTOCOL),
        categories=contract["categories"], instructions=contract["instructions"], guard="category-reliability",
        predicted_class_feature=True, review_C=.1, review_rows=len(targets), reliability_threshold=threshold,
        files={name: digest(saved / name) for name in ("intent.s1m", "lexical.json", "scope.npz")})
    (saved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    candidate = FusedReviewCandidate(saved)
    differences = [abs(candidate.reliability(values["development"][i], x["development"][i]) - scores[i]) for i in range(32)]
    assert max(differences) < 1e-4
    outcomes = []
    for i, row in enumerate(development):
        needs_review = bool(suggestions[i] == "oos" or not eligible[i] or scores[i] < threshold)
        outcomes.append(dict(group=row["group"], truth=row["label"], suggestion=str(suggestions[i]), score=float(scores[i]),
            needsReview=needs_review, category=None if needs_review else str(suggestions[i])))
    result = dict(dataset="clinc150", kind="system1", condition="fused-review", folder=saved.name,
        manifest=manifest, manifest_sha256=candidate.identity, review_preparation_ms=preparation_ms, review_fit_ms=fit_ms,
        semantic_preparation=extraction, original_fit_rows=len(original), calibration_rows=len(calibration), negative_rows=len(parts["negative"]),
        generated_examples_in_fold_heads=0, fold_groups=fold_groups, folds=fold_evidence,
        fixed_head_matches_2j=True, fixed_head_max_confidence_difference=max(head_differences),
        scalar_review_preflight_max_difference=max(differences), quality=selection, metrics=quality(outcomes), outcomes=outcomes)
    report["results"].append(result)
    (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(metrics=result["metrics"], review_fit_ms=fit_ms, scalar_preflight=max(differences))), flush=True)


def measure(folder):
    frozen(False)
    source = json.loads((folder / "development.json").read_text())
    for name, expected in source["source_files"].items():
        assert digest(ROOT / name) == expected, name
    assert len(source["results"]) == 1 and "failure" not in source
    report = dict(scope="experiment 2k complete CLINC development adapter; banking unchanged; not qualification", qualified=False,
        source_report_sha256=digest(folder / "development.json"), source_revision=source["source_revision"],
        protocol_sha256=digest(PROTOCOL), comparison_report_sha256=digest(COMPARISON), os_network_denial_errno=deny_network_control(),
        teacher_calls=0, new_api_cost=0, response_cache=False, receipts=False,
        process_import_and_preflight_ms=(time.perf_counter() - STARTED) * 1000,
        environment=dict(platform=platform.platform(), python=platform.python_version()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")}, results=[])
    with (folder / "runtime.json").open("x") as stream, (folder / "runtime.jsonl").open("x") as journal, threadpool_limits(limits=1):
        selected = source["results"][0]
        saved = folder / selected["folder"]
        started = time.perf_counter()
        candidate = FusedReviewCandidate(saved)
        load_ms = (time.perf_counter() - started) * 1000
        assert candidate.identity == selected["manifest_sha256"]
        rows = load_splits("clinc150")["development"]
        payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
        for row in rows[:100]:
            candidate.classify(dict(payload, text=row["prompt"]))
        outcomes, mismatches = [], []
        for row, expected in zip(rows, selected["outcomes"], strict=True):
            started = time.perf_counter()
            try:
                response = candidate.classify(dict(payload, text=row["prompt"]))
            except Exception as exc:
                response = dict(candidate=candidate.identity, needsReview=True, category=None, suggestion=None, teacherCalls=0, error=type(exc).__name__)
            elapsed = (time.perf_counter() - started) * 1000
            if any(response[key] != expected[key] for key in ("suggestion", "category", "needsReview")):
                mismatches.append(row["group"])
            outcome = dict(group=row["group"], truth=row["label"], latency_ms=elapsed, **response)
            outcomes.append(outcome)
            journal.write(json.dumps(outcome) + "\n")
            journal.flush()
        timings = [row["latency_ms"] for row in outcomes]
        metrics = quality(outcomes)
        errors = sum("error" in row for row in outcomes)
        result = dict(dataset="clinc150", kind="system1", condition="fused-review", folder=selected["folder"], manifest=candidate.manifest,
            candidate_sha256=candidate.identity, metrics=metrics, qualified=False, errors=errors, selection_runtime_mismatches=mismatches,
            load_ms=load_ms, artifact_bytes=sum(p.stat().st_size for p in saved.iterdir() if p.is_file()),
            external_encoder_bytes=sum(candidate.manifest["encoder"][key] for key in ("model_bytes", "tokenizer_bytes")),
            latency=dict(requests=len(rows), warmups=100, p50_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95))),
            development_gates=gates(metrics, float(np.percentile(timings, 95)), errors),
            accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]],
            per_intent={label: quality([r for r in outcomes if r["truth"] == label]) for label in sorted({r["truth"] for r in outcomes})}, outcomes=outcomes)
        report["results"].append(result)
        stream.write(json.dumps(report, indent=2) + "\n")
        stream.flush()
        report["comparisons"] = []
        for prior in json.loads(COMPARISON.read_text())["comparisons"]:
            incumbent = prior["selected_development_candidate"]
            winner = incumbent
            if prior["dataset"] == "clinc150" and prior["kind"] == "system1":
                valid = all(v for key, v in result["development_gates"].items() if key != "coverage_at_least_80")
                if valid and not mismatches and metrics["supported_coverage"]["numerator"] > incumbent["metrics"]["supported_coverage"]["numerator"]:
                    winner = {key: result[key] for key in ("folder", "candidate_sha256", "metrics", "latency")}
                    winner["source"] = result["condition"]
            report["comparisons"].append(dict(dataset=prior["dataset"], kind=prior["kind"], incumbent=incumbent,
                selected_development_candidate=winner, qualified=False, incumbent_timings_from_prior_run=True))
        stream.seek(0)
        stream.write(json.dumps(report, indent=2) + "\n")
        stream.truncate()
        print(json.dumps({key: result[key] for key in ("metrics", "latency", "development_gates", "selection_runtime_mismatches")}), flush=True)
    assert not mismatches, "Saved adapter differs from selection; failure retained"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "measure"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.action == "fit" else measure)(args.folder)
