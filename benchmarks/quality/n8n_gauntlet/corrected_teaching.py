"""Experiment 2p: seven fitting-label corrections, four unchanged method setups."""
import time
STARTED = time.perf_counter()

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline_review import meta_features as lexical_meta, midpoint
from boundary_development import compile_head, logistic_fit, negative_source
from correction_runtime import CorrectionCandidate
from crossfit_probe import probabilities
from develop import frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from evaluate import deny_network_control, gates, quality
from human_only import HumanOnlyBaseline
from latest_regression import CANDIDATES, REFERENCES, frozen
from polynomial_scores import expand
from review_teaching import features, meta_features
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler
from teach_boundaries import HERE, ROOT, committed, digest

PROTOCOL = HERE / "CORRECTION_PROTOCOL.md"
PROPOSALS = HERE / "results/teaching-correction-proposals.json"
COMPARISON = HERE / "results/human-only-runtime.json"


def corrected_rows(dataset, rows, apply):
    source = json.loads(PROPOSALS.read_text())
    assert source["source_report_sha256"] == digest(HERE / "results/teaching-consistency.json")
    assert source["fit_sha256"][dataset] == digest(ROOT / ".system1/n8n-gauntlet" / f"{dataset}-fit.json")
    proposals = {r["group"]: r for r in source["proposals"] if r["dataset"] == dataset}
    assert len(proposals) == (6 if dataset == "clinc150" else 1)
    changed, result = [], []
    for row in rows:
        proposal = proposals.get(row["group"])
        if proposal:
            assert all(proposal[k] == row[k] for k in ("prompt", "source", "index")) and proposal["original_label"] == row["label"]
        updated = dict(row, label=proposal["proposed_label"]) if proposal and apply else dict(row)
        result.append(updated)
        if updated["label"] != row["label"]:
            changed.append(proposal)
    assert len(changed) == (len(proposals) if apply else 0)
    return result, changed


def reference(dataset, kind):
    identity = digest(HERE / "artifacts" / CANDIDATES[dataset][kind] / "manifest.json")
    rows = json.loads((HERE / "results" / REFERENCES[dataset][kind]).read_text())["results"]
    return next(r for r in rows if r["candidate_sha256"] == identity)


def fit_one(dataset, kind, corrected, folder):
    begin = time.perf_counter()
    bank, neural = dataset == "banking77", kind == "system1"
    context, quadratic = neural and not bank, bank
    data = load_splits(dataset)
    original, calibration, development = data["fit"], data["calibration"], data["development"]
    effective, changes = corrected_rows(dataset, original, corrected)
    labels = sorted({r["label"] for r in original})
    assert set(labels) == {r["label"] for r in effective}
    extra, negative = [], []
    if bank:
        extra = json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"]
        assert len(extra) == 924
    elif neural:
        negative = negative_source()
        assert len(negative) == 2000
    else:
        lessons = json.loads((HERE / "results/boundary-teaching/lessons.json").read_text())["lessons"]
        extra = [r for r in lessons if r["dataset"] == dataset and r["label"] != "oos"]
        negative = [r for r in lessons if r["dataset"] == dataset and r["label"] == "oos"]
        assert len(extra) == 1696 and len(negative) == 674
    augmented = effective + extra
    yi = np.asarray([labels.index(r["label"]) for r in effective])
    ya = np.asarray([labels.index(r["label"]) for r in augmented])
    yc = np.asarray([labels.index(r["label"]) for r in calibration])
    fold_groups = json.loads((HERE / "results/review-teaching-development.json").read_text())["datasets"][dataset]["fold_groups"]
    fold_by_group = {group: fold for fold, groups in enumerate(fold_groups) for group in groups}
    assert set(fold_by_group) == {r["group"] for r in original}
    folds = np.asarray([fold_by_group[r["group"]] for r in original])
    parts = dict(fit=original, augmented=original + extra, calibration=calibration, development=development)
    if negative:
        parts["negative"] = negative
    extraction = []
    if neural:
        encoder = Encoder("bge-small" if bank else "minilm", threads=4 if bank else 1)
        # Labels do not alter text vectors; retain existing cache identities.
        x = {key: features(rows, encoder, single=True, evidence=extraction) for key, rows in parts.items()}
        schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
        prepared = PreparedFeatures(augmented + calibration, np.concatenate([x["augmented"], x["calibration"]]), encoder)
        compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.01 if bank else .1)
    cross, cross_guesses = np.zeros((len(original), 7)), np.zeros(len(original), dtype=int)
    for fold in range(3):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if neural:
            model = compile_head(compiler, [effective[i] for i in train], calibration)
            p = probabilities(model.heads["intent"], x["fit"][valid])
            cross[valid] = meta_features(p, x["fit"][valid], x["fit"][train], yi[train])
        else:
            lexical = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            xt = lexical.fit_transform([effective[i]["prompt"] for i in train])
            xv = lexical.transform([effective[i]["prompt"] for i in valid])
            model = logistic_fit(xt, yi[train])
            assert np.array_equal(model.classes_, np.arange(len(labels)))
            p = model.predict_proba(xv)
            cross[valid] = lexical_meta(p, xv, xt, yi[train])
        cross_guesses[valid] = p.argmax(axis=1)
    started = time.perf_counter()
    if neural:
        taught = compile_head(compiler, augmented, calibration)
        model = CompiledSystemOneModel(taught.schema, taught.heads, dimension=384, projector=encoder, metadata=taught.metadata, use_cache=False)
        engine = System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)
        predict, make_meta = lambda vectors: probabilities(model.heads["intent"], vectors), meta_features
    else:
        lexical = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        x = {"augmented": lexical.fit_transform([r["prompt"] for r in augmented])}
        x.update({key: lexical.transform([r["prompt"] for r in rows]) for key, rows in parts.items() if key != "augmented"})
        model = logistic_fit(x["augmented"], ya)
        assert np.array_equal(model.classes_, np.arange(len(labels)))
        predict, make_meta = model.predict_proba, lexical_meta
    intent_fit_ms = (time.perf_counter() - started) * 1000
    values = {key: predict(x[key]) for key in parts if key not in ("fit", "augmented")}
    meta = {key: make_meta(p, x[key], x["augmented"], ya) for key, p in values.items()}
    guesses = {key: p.argmax(axis=1) for key, p in values.items()}
    eligible, sets = None, None
    alpha = .075 if bank else .05
    if neural:
        decisions = [engine.decide(row["prompt"], embedding=vector, alpha=alpha, record_receipt=False)
            for row, vector in zip(development, x["development"], strict=True)]
        eligible = [not d.is_ambiguous for d in decisions]
        sets = [d.conformal_sets["intent"] for d in decisions]
    suffix = ["negative"] if negative else []
    teaching = np.concatenate([cross, meta["calibration"]] + [meta[k] for k in suffix])
    chosen = np.concatenate([cross_guesses, guesses["calibration"]] + [guesses[k] for k in suffix])
    target = np.concatenate([(cross_guesses == yi) & np.asarray([r["label"] != "oos" for r in effective]),
        (guesses["calibration"] == yc) & np.asarray([r["label"] != "oos" for r in calibration])] +
        ([np.zeros(len(negative), dtype=bool)] if negative else []))
    scaler = StandardScaler().fit(expand(teaching, quadratic))
    train = np.column_stack([scaler.transform(expand(teaching, quadratic)), np.eye(len(labels))[chosen]])
    dev = np.column_stack([scaler.transform(expand(meta["development"], quadratic)), np.eye(len(labels))[guesses["development"]]])
    if context:
        train = np.column_stack([train, np.concatenate([x["fit"], x["calibration"], x["negative"]])])
        dev = np.column_stack([dev, x["development"]])
    started = time.perf_counter()
    review_c = 1 if not neural and not bank else .1
    review = LogisticRegression(C=review_c, max_iter=1000).fit(train, target)
    assert review.n_iter_.max() < 1000
    review_fit_ms = (time.perf_counter() - started) * 1000
    scores = review.predict_proba(dev)[:, 1]
    suggestions = np.asarray(labels)[guesses["development"]]
    selection = frontier([r["label"] for r in development], suggestions, scores, eligible=eligible)
    threshold = midpoint(scores, selection["best_development_coverage_at_quality_targets"]["threshold"])
    outcomes = []
    for i, row in enumerate(development):
        needs_review = bool(suggestions[i] == "oos" or scores[i] < threshold or (eligible is not None and not eligible[i]))
        item = dict(group=row["group"], truth=row["label"], suggestion=str(suggestions[i]), score=float(scores[i]),
            confidence=float(values["development"][i].max()), needsReview=needs_review, category=None if needs_review else str(suggestions[i]))
        if neural:
            item.update(predictionSet=sets[i], strict_eligible=eligible[i])
        outcomes.append(item)
    previous = reference(dataset, kind)
    condition = "corrected" if corrected else "unchanged-control"
    result = dict(dataset=dataset, kind=kind, condition=condition, reference_sha256=previous["candidate_sha256"],
        fit_groups=[r["group"] for r in original], fold_groups=fold_groups, corrections_applied=changes,
        effective_fit_sha256=hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest(),
        original_fit_rows=len(original), generated_fit_rows=len(extra), negative_review_rows=len(negative),
        review_rows=len(target), review_negative_targets=int((~target).sum()), generated_rows_in_fold_heads=0,
        intent_fit_ms=intent_fit_ms, review_fit_ms=review_fit_ms, feature_extraction=extraction,
        quality=selection, metrics=quality(outcomes), outcomes=outcomes)
    if not corrected:
        differences, confidence_differences, mismatches = [], [], []
        for row, old in zip(outcomes, previous["outcomes"], strict=True):
            keys = ("group", "truth", "suggestion", "category", "needsReview") + (("predictionSet",) if neural else ())
            if any(row[k] != old[k] for k in keys):
                mismatches.append(row["group"])
            differences.append(abs(row["score"] - old["reliability" if neural else "acceptance_score"]))
            confidence_differences.append(abs(row["confidence"] - old["confidence"]))
        result["control_reproduction"] = dict(route_mismatches=mismatches, max_score_difference=max(differences),
            max_confidence_difference=max(confidence_differences), retimed=False)
    else:
        saved = folder / f"{dataset}-{kind}-corrected"
        saved.mkdir()
        manifest = dict(status="unqualified-corrected-teaching-development", dataset=dataset, kind=kind, condition=condition,
            categories=previous["manifest"]["categories"], instructions=previous["manifest"]["instructions"],
            protocol_sha256=digest(PROTOCOL), proposals_sha256=digest(PROPOSALS), corrected_fit_groups=[r["group"] for r in changes],
            fit_rows=len(augmented), review_rows=len(target), quadratic=quadratic, predicted_class_feature=True, review_C=review_c)
        core_width = (35 if quadratic else 7) + len(labels)
        if context:
            manifest.update(context_dimension=384, context_weights=review.coef_[0, core_width:].tolist())
        if neural:
            model.save(saved / "intent.s1m")
            order = np.argsort(ya, kind="stable")
            np.savez_compressed(saved / "scope.npz", prototypes=x["augmented"][order],
                class_offsets=np.concatenate([[0], np.cumsum(np.bincount(ya, minlength=len(labels)))]),
                mean=scaler.mean_, scale=scaler.scale_, weights=review.coef_[0, :core_width], bias=review.intercept_[0])
            manifest.update(encoder=encoder.identity, guard="category-reliability", alpha=alpha, regularization=.01 if bank else .1,
                reliability_threshold=threshold, teaching_projection="single-request", files={name: digest(saved / name) for name in ("intent.s1m", "scope.npz")})
        else:
            (saved / "vocabulary.json").write_text(json.dumps(lexical.vocabulary_, sort_keys=True) + "\n")
            np.savez_compressed(saved / "weights.npz", weights=model.coef_, bias=model.intercept_, idf=lexical.idf_, labels=np.asarray(labels),
                mean=scaler.mean_, scale=scaler.scale_, gate_weights=review.coef_, gate_bias=review.intercept_, fit_labels=ya)
            sparse.save_npz(saved / "prototypes.npz", x["augmented"])
            manifest.update(C=10, policy="reliability", threshold=threshold, ngram_range=[1, 2], sublinear_tf=True,
                files={name: digest(saved / name) for name in ("weights.npz", "prototypes.npz", "vocabulary.json")})
        (saved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        candidate = (CorrectionCandidate if neural else HumanOnlyBaseline)(saved)
        scalar = [candidate.reliability(values["development"][i], x["development"][i]) if neural
            else candidate.review_score(values["development"][i], x["development"][i]) for i in range(32)]
        difference = max(abs(p - scores[i]) for i, p in enumerate(scalar))
        assert difference < 1e-4
        result.update(folder=saved.name, manifest=manifest, manifest_sha256=candidate.identity, scalar_preflight_max_difference=float(difference))
    result["total_preparation_and_fit_ms"] = (time.perf_counter() - begin) * 1000
    return result


def fit(folder):
    sources = [PROTOCOL, PROPOSALS, Path(__file__).resolve(), HERE / "correction_runtime.py", COMPARISON,
        HERE / "human_only.py", HERE / "human_only_runtime.py", HERE / "results/teaching-consistency.json",
        HERE / "results/contrast-lessons.json", HERE / "results/boundary-teaching/lessons.json",
        HERE / "manifest.json", HERE / "review-data-manifest.json"] + [HERE / "results" / name for name in ("context-runtime.json", "polynomial-runtime.json", "boundary-runtime.json")]
    revision = committed(sources)
    frozen(False)
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(scope="experiment 2p fitting-only correction development; evaluation labels unchanged", qualified=False,
        source_revision=revision, source_files={str(p.relative_to(ROOT)): digest(p) for p in sources},
        protocol_sha256=digest(PROTOCOL), proposals_sha256=digest(PROPOSALS), comparison_report_sha256=digest(COMPARISON),
        os_network_denial_errno=deny_network_control(), teacher_calls=0, new_api_cost=0,
        independent_annotations=False, proposal_authoring_cost="Assistant-session cost not measured", results=[])
    with threadpool_limits(limits=1):
        try:
            for dataset in ("clinc150", "banking77"):
                for kind in ("system1", "baseline"):
                    for corrected in (False, True):
                        result = fit_one(dataset, kind, corrected, folder)
                        report["results"].append(result)
                        (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps({k: result[k] for k in ("dataset", "kind", "condition", "metrics")}), flush=True)
                        if not corrected:
                            check = result["control_reproduction"]
                            assert not check["route_mismatches"] and check["max_score_difference"] <= 1e-4 and check["max_confidence_difference"] <= 1e-4, check
                        gc.collect()
        except Exception as exc:
            report["failure"] = dict(error=type(exc).__name__, message=str(exc))
            (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")
            raise


def measure(folder):
    frozen(False)
    source = json.loads((folder / "development.json").read_text())
    assert len(source["results"]) == 8 and "failure" not in source
    for name, expected in source["source_files"].items():
        assert digest(ROOT / name) == expected, name
    report = dict(scope="experiment 2p complete corrected development adapters; not qualification", qualified=False,
        source_report_sha256=digest(folder / "development.json"), source_revision=source["source_revision"],
        protocol_sha256=digest(PROTOCOL), proposals_sha256=digest(PROPOSALS), comparison_report_sha256=digest(COMPARISON),
        os_network_denial_errno=deny_network_control(), teacher_calls=0, new_api_cost=0, response_cache=False, receipts=False,
        process_import_and_preflight_ms=(time.perf_counter() - STARTED) * 1000,
        environment=dict(platform=platform.platform(), python=platform.python_version()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "threadpoolctl")}, results=[])
    with (folder / "runtime.json").open("x") as stream, (folder / "runtime.jsonl").open("x") as journal, threadpool_limits(limits=1):
        for selected in (r for r in source["results"] if r["condition"] == "corrected"):
            saved = folder / selected["folder"]
            begin = time.perf_counter()
            candidate = (CorrectionCandidate if selected["kind"] == "system1" else HumanOnlyBaseline)(saved)
            load_ms = (time.perf_counter() - begin) * 1000
            assert candidate.identity == selected["manifest_sha256"]
            rows = load_splits(selected["dataset"])["development"]
            payload = {key: candidate.manifest[key] for key in ("categories", "instructions")}
            for row in rows[:100]:
                candidate.classify(dict(payload, text=row["prompt"]))
            outcomes, mismatches = [], []
            for row, expected in zip(rows, selected["outcomes"], strict=True):
                begin = time.perf_counter()
                try:
                    response = candidate.classify(dict(payload, text=row["prompt"]))
                except Exception as exc:
                    response = dict(candidate=candidate.identity, needsReview=True, category=None, suggestion=None, teacherCalls=0, error=type(exc).__name__)
                elapsed = (time.perf_counter() - begin) * 1000
                keys = ("suggestion", "category", "needsReview") + (("predictionSet",) if selected["kind"] == "system1" else ())
                if any(response.get(k) != expected[k] for k in keys):
                    mismatches.append(row["group"])
                outcome = dict(group=row["group"], truth=row["label"], latency_ms=elapsed, **response)
                outcomes.append(outcome)
                journal.write(json.dumps(dict(folder=saved.name, **outcome)) + "\n")
                journal.flush()
            times = [r["latency_ms"] for r in outcomes]
            metrics = quality(outcomes)
            errors = sum("error" in r for r in outcomes)
            result = dict(dataset=selected["dataset"], kind=selected["kind"], condition="corrected", folder=saved.name,
                manifest=candidate.manifest, candidate_sha256=candidate.identity, metrics=metrics, qualified=False,
                errors=errors, selection_runtime_mismatches=mismatches, load_ms=load_ms,
                artifact_bytes=sum(p.stat().st_size for p in saved.iterdir() if p.is_file()),
                external_encoder_bytes=sum(candidate.manifest["encoder"][k] for k in ("model_bytes", "tokenizer_bytes")) if selected["kind"] == "system1" else 0,
                latency=dict(requests=len(rows), warmups=100, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))),
                development_gates=gates(metrics, float(np.percentile(times, 95)), errors),
                accepted_errors=[r for r in outcomes if not r["needsReview"] and r["category"] != r["truth"]],
                per_intent={label: quality([r for r in outcomes if r["truth"] == label]) for label in sorted({r["truth"] for r in outcomes})}, outcomes=outcomes)
            report["results"].append(result)
            stream.seek(0)
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.truncate()
            stream.flush()
            print(json.dumps({k: result[k] for k in ("dataset", "kind", "metrics", "latency", "development_gates", "selection_runtime_mismatches")}), flush=True)
            del candidate
            gc.collect()
        report["comparisons"] = []
        for prior in json.loads(COMPARISON.read_text())["comparisons"]:
            incumbent = prior["selected_development_candidate"]
            winner = incumbent
            result = next(r for r in report["results"] if (r["dataset"], r["kind"]) == (prior["dataset"], prior["kind"]))
            valid = all(v for k, v in result["development_gates"].items() if k != "coverage_at_least_80")
            if valid and not result["selection_runtime_mismatches"] and result["metrics"]["supported_coverage"]["numerator"] > incumbent["metrics"]["supported_coverage"]["numerator"]:
                winner = {k: result[k] for k in ("folder", "candidate_sha256", "metrics", "latency")}
                winner["source"] = "corrected-teaching"
            report["comparisons"].append(dict(dataset=prior["dataset"], kind=prior["kind"], incumbent=incumbent,
                selected_development_candidate=winner, qualified=False, incumbent_timings_from_prior_run=True))
        stream.seek(0)
        stream.write(json.dumps(report, indent=2) + "\n")
        stream.truncate()
    assert all(not r["selection_runtime_mismatches"] for r in report["results"]), "Routing differs; failure retained"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "measure"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.action == "fit" else measure)(args.folder)
