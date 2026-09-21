"""Bounded experiment-2j head comparison; no test data or adapter timing."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from boundary_development import compile_head
from develop import frontier, load_splits
from encoder_probe import Encoder, PreparedFeatures
from evaluate import deny_network_control, quality
from fused_features import FusedFeatures, join_features
from latest_regression import frozen
from review_teaching import features
from system1 import ChoiceField, CompiledSystemOneModel, DecisionSchema, System1Engine, SystemOneCompiler
from system1.core.text import TfidfProjector
from teach_boundaries import HERE, ROOT, committed, digest

PROTOCOL = HERE / "FUSED_FEATURES_PROTOCOL.md"


def fit_one(dataset, folder, report):
    bank = dataset == "banking77"
    data = load_splits(dataset)
    earlier = json.loads((HERE / "results/contrast-lessons.json").read_text())["lessons"] if bank else []
    original, augmented = data["fit"], data["fit"] + earlier
    labels = sorted({r["label"] for r in original})
    schema = type("Intent", (DecisionSchema,), {"intent": ChoiceField(options=labels)})
    started = time.perf_counter()
    encoder = Encoder("bge-small" if bank else "minilm", threads=4 if bank else 1)
    encoder_load_ms = (time.perf_counter() - started) * 1000
    extraction = []
    parts = dict(fit=augmented, calibration=data["calibration"], development=data["development"])
    semantic = {key: features(rows, encoder, single=True, evidence=extraction) for key, rows in parts.items()}
    for condition in ("semantic-control", "semantic-plus-words"):
        started = time.perf_counter()
        lexical = None
        if condition == "semantic-plus-words":
            lexical = TfidfProjector.fit([r["prompt"] for r in augmented], max_features=2048)
            projector = FusedFeatures(encoder, lexical)
            x = {key: np.stack([join_features(vector, lexical.project(row["prompt"]))
                               for vector, row in zip(semantic[key], rows, strict=True)]) for key, rows in parts.items()}
        else:
            projector, x = encoder, semantic
        additional_preparation_ms = (time.perf_counter() - started) * 1000
        prepared = PreparedFeatures(augmented + data["calibration"], np.concatenate([x["fit"], x["calibration"]]), projector)
        compiler = SystemOneCompiler(schema, projector=prepared, choice_solver="logistic", regularization=.01 if bank else .1)
        started = time.perf_counter()
        taught = compile_head(compiler, augmented, data["calibration"])
        model = CompiledSystemOneModel(taught.schema, taught.heads, dimension=projector.dimension,
                                      projector=projector, metadata=taught.metadata, use_cache=False)
        fit_ms = (time.perf_counter() - started) * 1000
        candidate_folder = folder / f"{dataset}-{condition}"
        candidate_folder.mkdir()
        model.save(candidate_folder / "intent.s1m")
        config = lexical.to_config() if lexical else None
        (candidate_folder / "lexical.json").write_text(json.dumps(config, indent=2) + "\n")
        manifest = dict(status="unqualified-fused-feature-head-comparison", dataset=dataset, condition=condition,
            dimension=projector.dimension, encoder=encoder.identity, projector_digest=projector.projector_digest(),
            protocol_sha256=digest(PROTOCOL), regularization=.01 if bank else .1, alpha=.075 if bank else .05,
            fit_rows=len(augmented), original_fit_rows=len(original), earlier_generated_supported=len(earlier),
            files={name: digest(candidate_folder / name) for name in ("intent.s1m", "lexical.json")})
        (candidate_folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        restored_lexical = json.loads((candidate_folder / "lexical.json").read_text())
        restored_projector = FusedFeatures(encoder, TfidfProjector.from_config(restored_lexical)) if restored_lexical else encoder
        restored = CompiledSystemOneModel.load(candidate_folder / "intent.s1m", projector=restored_projector)
        restored.use_cache = False
        engine = System1Engine(restored.schema, model=restored, strict_mode=True, use_cache=False)
        outcomes = []
        for row, vector in zip(data["development"], x["development"], strict=True):
            decision = engine.decide(row["prompt"], embedding=vector, alpha=manifest["alpha"], record_receipt=False)
            suggestion = decision.values["intent"]
            ordered = sorted(decision.probabilities["intent"].values())
            review = decision.is_ambiguous or suggestion == "oos"
            outcomes.append(dict(group=row["group"], truth=row["label"], suggestion=suggestion,
                confidence=ordered[-1], margin=ordered[-1] - ordered[-2], strict_eligible=not decision.is_ambiguous,
                predictionSet=decision.conformal_sets["intent"], needsReview=bool(review),
                category=None if review else suggestion, teacherCalls=0))
        truth, pred = [r["truth"] for r in outcomes], [r["suggestion"] for r in outcomes]
        curves = {score: frontier(truth, pred, [r[score] for r in outcomes]) for score in ("confidence", "margin")}
        strict = {score: frontier(truth, pred, [r[score] for r in outcomes], eligible=[r["strict_eligible"] for r in outcomes])
                  for score in ("confidence", "margin")}
        replay_differences = []
        for row, expected in zip(data["development"][:16], outcomes[:16], strict=True):
            actual = engine.decide(row["prompt"], alpha=manifest["alpha"], record_receipt=False)
            assert actual.values["intent"] == expected["suggestion"]
            assert actual.conformal_sets["intent"] == expected["predictionSet"]
            assert (not actual.is_ambiguous) == expected["strict_eligible"]
            delta = abs(actual.probabilities["intent"][expected["suggestion"]] - expected["confidence"])
            assert delta < 1e-5
            replay_differences.append(delta)
        result = dict(dataset=dataset, condition=condition, manifest=manifest, folder=candidate_folder.name,
            candidate_sha256=digest(candidate_folder / "manifest.json"), encoder_load_ms=encoder_load_ms,
            semantic_preparation=extraction, additional_feature_preparation_ms=additional_preparation_ms, head_fit_ms=fit_ms,
            artifact_bytes=sum(p.stat().st_size for p in candidate_folder.iterdir() if p.is_file()),
            external_encoder_bytes=encoder.identity["model_bytes"] + encoder.identity["tokenizer_bytes"],
            strict_only_metrics=quality(outcomes), probability_margin_frontiers=curves, strict_probability_margin_frontiers=strict,
            raw_oos_predictions=sum(r["truth"] == "oos" and r["suggestion"] == "oos" for r in outcomes),
            supported_predicted_oos=sum(r["truth"] != "oos" and r["suggestion"] == "oos" for r in outcomes),
            saved_projection_replay=dict(requests=16, max_confidence_difference=max(replay_differences)), outcomes=outcomes)
        report["results"].append(result)
        (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(dataset=dataset, condition=condition, raw_supported=result["strict_only_metrics"]["raw_supported_accuracy"],
                             raw_oos_predictions=result["raw_oos_predictions"], head_fit_ms=fit_ms)), flush=True)
        del model, taught, restored, engine, compiler, prepared
        gc.collect()
    del encoder
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    folder = parser.parse_args().folder
    paths = [PROTOCOL, Path(__file__), HERE / "fused_features.py", ROOT / "src/system1/core/text.py", ROOT / "tests/test_fused_features.py",
             HERE / "results/contrast-lessons.json", HERE / "results/baseline-review-runtime.json", HERE / "results/direct-oos-runtime.json"]
    revision = committed(paths)
    frozen(False)
    denied = deny_network_control()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(scope="experiment 2j development head comparison only; no new review, adapter latency or qualification", qualified=False,
        source_revision=revision, protocol_sha256=digest(PROTOCOL), source_files={str(p.relative_to(ROOT)): digest(p) for p in paths},
        environment=dict(platform=platform.platform(), python=platform.python_version()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "onnxruntime", "tokenizers")},
        os_network_denial_errno=denied, teacher_calls=0, new_api_cost=0, results=[])
    with threadpool_limits(limits=1):
        for dataset in ("clinc150", "banking77"):
            try:
                fit_one(dataset, folder, report)
            except Exception as exc:
                report["failure"] = dict(dataset=dataset, error=type(exc).__name__, message=str(exc))
                (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
                raise
    report["advance_to_review_comparison"] = {}
    for dataset in ("clinc150", "banking77"):
        control, fused = [r for r in report["results"] if r["dataset"] == dataset]
        report["advance_to_review_comparison"][dataset] = (fused["strict_only_metrics"]["raw_supported_accuracy"]["numerator"] >
                                                         control["strict_only_metrics"]["raw_supported_accuracy"]["numerator"])
    (folder / "head-development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["advance_to_review_comparison"]), flush=True)


if __name__ == "__main__":
    main()
