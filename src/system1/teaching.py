"""Explicit corrections and checked replacement of one local text skill.

A session is a single-writer local workspace, not a concurrent service. Supplied
labels are the only lessons: predictions are never recorded automatically.
Evaluation examples are recurring development checks, not fresh statistical
qualification. Keep related documents in the same split; normalized text checks
cannot detect paraphrases or related customer conversations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from system1.compiler import CompiledSystemOneModel, SystemOneCompiler
from system1.core.schema import ChoiceField, DecisionSchema
from system1.core.text import MAX_TEXT_LENGTH, TfidfProjector
from system1.engine import SystemOneEngine


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _atomic_bytes(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def _text_key(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A lesson needs nonempty text")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text must contain at most {MAX_TEXT_LENGTH} characters")
    return " ".join(text.casefold().split())


class TeachingSession:
    """Teach, assess, and explicitly adopt one ChoiceField text classifier.

    ``record(text, label)`` replaces a correction to the same normalized input.
    Separate ``calibrate`` examples estimate uncertainty; ``evaluate`` examples
    compare candidate and current skill. ``assess`` never changes the current
    artifact. ``adopt`` requires a passing, unchanged assessment.

    Defaults are TF-IDF (at most 1024 features), ridge regularization .1, strict
    decisions at alpha .05, and no inference caches. These choices are confined
    to this helper and do not alter existing compiler or runtime defaults.
    """

    def __init__(self, directory, schema=None, *, regularization=None, max_features=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_path = self.directory / "session.json"
        self.current_path = self.directory / "current.s1m"
        self.candidate_path = self.directory / "candidate.s1m"
        self.report_path = self.directory / "assessment.json"
        self._engine = None
        self._engine_digest = None
        supplied = schema() if isinstance(schema, type) and issubclass(schema, DecisionSchema) else schema
        if self.session_path.exists():
            state = self._read_state()
            if supplied is not None and (not isinstance(supplied, DecisionSchema) or
                                        supplied.to_dict() != state["schema"]):
                raise ValueError("The schema differs from this teaching session")
            for name, value in (("regularization", regularization), ("max_features", max_features)):
                if value is not None and value != state[name]:
                    raise ValueError(f"{name} differs from this teaching session")
        else:
            if not isinstance(supplied, DecisionSchema):
                raise ValueError("A new teaching session needs a single ChoiceField schema")
            state = {"version": 1, "schema": supplied.to_dict(), "records": [],
                     "regularization": .1 if regularization is None else regularization,
                     "max_features": 1024 if max_features is None else max_features}
            self._validate_state(state)
            _write_json(self.session_path, state)
        self.schema = DecisionSchema.from_dict(state["schema"])
        self.field_name = next(iter(self.schema.fields))

    @staticmethod
    def _validate_state(state):
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported teaching session version")
        schema = DecisionSchema.from_dict(state["schema"])
        if len(schema.fields) != 1 or not isinstance(next(iter(schema.fields.values())), ChoiceField):
            raise ValueError("TeachingSession supports exactly one ChoiceField")
        field = next(iter(schema.fields.values()))
        if not field.escalate_on_ambiguity:
            raise ValueError("TeachingSession requires escalate_on_ambiguity=True so uncertain choices require review")
        regularization = state["regularization"]
        if type(regularization) not in (int, float) or not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if type(state["max_features"]) is not int or not 1 <= state["max_features"] <= 4096:
            raise ValueError("max_features must be an integer between 1 and 4096")
        if not isinstance(state["records"], list):
            raise ValueError("Session records must be a list")
        seen = set()
        for row in state["records"]:
            if not isinstance(row, dict) or set(row) != {"input", "label", "split", "source"}:
                raise ValueError("Invalid teaching record")
            key = _text_key(row["input"])
            if key in seen:
                raise ValueError("Teaching, calibration, and evaluation inputs must be disjoint and unique")
            seen.add(key)
            if row["split"] not in ("teach", "calibrate", "evaluate"):
                raise ValueError("split must be teach, calibrate, or evaluate")
            if not isinstance(row["source"], str) or not row["source"].strip():
                raise ValueError("A supplied label needs a nonempty source")
            if not isinstance(row["label"], str) or row["label"] not in field.options:
                raise ValueError("A supplied label must match a choice in the schema")
        return state

    def _read_state(self):
        state = self._validate_state(json.loads(self.session_path.read_text(encoding="utf-8")))
        if hasattr(self, "schema") and state["schema"] != self.schema.to_dict():
            raise ValueError("The schema differs from this teaching session")
        return state

    @property
    def records(self):
        return self._read_state()["records"]

    def record(self, text, label, *, split="teach", source="human"):
        """Add an explicit label, or replace a same-split correction in place."""
        row = {"input": text, "label": label, "split": split, "source": source}
        self.record_many([row])
        return row.copy()

    def record_many(self, rows):
        """Validate a batch of supplied lessons before writing any of them."""
        if not isinstance(rows, (list, tuple)):
            raise ValueError("Pass a list of labeled lesson records")
        state = self._read_state()
        for supplied in rows:
            required = {"input", "label", "split"}
            if (not isinstance(supplied, dict) or not required <= set(supplied)
                    or set(supplied) - (required | {"source"})):
                raise ValueError("A lesson record needs input, label, split, and optional source")
            row = {**supplied, "source": supplied.get("source", "human")}
            self._validate_state({**state, "records": [row]})
            key = _text_key(row["input"])
            for index, previous in enumerate(state["records"]):
                if _text_key(previous["input"]) == key:
                    if previous["split"] != row["split"]:
                        raise ValueError("Teaching, calibration, and evaluation inputs must be disjoint")
                    state["records"][index] = row
                    break
            else:
                state["records"].append(row)
        self._validate_state(state)
        _write_json(self.session_path, state)
        return self.records

    def remove(self, text):
        state = self._read_state()
        key = _text_key(text)
        remaining = [row for row in state["records"] if _text_key(row["input"]) != key]
        if len(remaining) == len(state["records"]):
            raise ValueError("No recorded lesson matches that input")
        state["records"] = remaining
        _write_json(self.session_path, state)

    def _load_engine(self, path):
        model = CompiledSystemOneModel.load(path)
        if model.schema.schema_digest() != self.schema.schema_digest():
            raise ValueError("Skill schema differs from this teaching session")
        model.use_cache = False
        return SystemOneEngine(model.schema, model=model, strict_mode=True, use_cache=False)

    @property
    def engine(self):
        digest = _file_digest(self.current_path)
        if digest != self._engine_digest:
            self._engine = self._load_engine(self.current_path) if digest else None
            self._engine_digest = digest
        return self._engine

    def predict(self, text):
        _text_key(text)
        engine = self.engine
        if engine is None:
            raise ValueError("Assess and adopt a skill before predicting")
        return engine.decide(text, alpha=.05, record_receipt=False)

    @property
    def report(self):
        return json.loads(self.report_path.read_text(encoding="utf-8")) if self.report_path.exists() else None

    def snapshot(self):
        state = self._read_state()
        report = self.report
        identity = self._identity(state)
        adopted = bool(report and identity["current"] == report["identity"]["candidate"])
        comparison = dict(identity)
        if adopted:
            comparison["current"] = report["identity"]["current"]
        return {"records": state["records"], "counts": self._counts(state),
                "current": identity["current"], "report": report, "adopted": adopted,
                "candidate_stale": bool(report and report["identity"] != comparison)}

    @staticmethod
    def _counts(state):
        return {split: sum(row["split"] == split for row in state["records"])
                for split in ("teach", "calibrate", "evaluate")}

    def _identity(self, state):
        return {"session": _digest(state), "candidate": _file_digest(self.candidate_path),
                "current": _file_digest(self.current_path)}

    @staticmethod
    def _thresholds(min_accuracy, min_coverage, max_accepted_errors, max_regressions):
        for name, value in (("min_accuracy", min_accuracy), ("min_coverage", min_coverage)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name, value in (("max_accepted_errors", max_accepted_errors), ("max_regressions", max_regressions)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        return dict(min_accuracy=min_accuracy, min_coverage=min_coverage,
                    max_accepted_errors=max_accepted_errors, max_regressions=max_regressions)

    def _measure(self, engine, checks):
        outcomes = []
        for row in checks:
            result = engine.decide(row["input"], alpha=.05, record_receipt=False)
            outcomes.append({"predicted": result.values[self.field_name], "review": bool(result.is_ambiguous),
                             "correct": result.values[self.field_name] == row["label"],
                             "prediction_set": list(result.conformal_sets[self.field_name])})
        count = len(outcomes)
        correct = sum(row["correct"] for row in outcomes)
        accepted = sum(not row["review"] for row in outcomes)
        accepted_correct = sum(row["correct"] and not row["review"] for row in outcomes)
        return {"count": count, "raw_correct": correct, "raw_accuracy": correct / count,
                "accepted": accepted, "accepted_correct": accepted_correct,
                "accepted_errors": accepted - accepted_correct, "review": count - accepted,
                "review_rate": (count - accepted) / count, "coverage": accepted / count}, outcomes

    def _compare(self, state, thresholds):
        checks = [row for row in state["records"] if row["split"] == "evaluate"]
        if not checks:
            raise ValueError("Supply separate evaluate examples before assessing a candidate")
        candidate_engine = self._load_engine(self.candidate_path)
        binding = {"session": _digest(state), "current": _file_digest(self.current_path), "thresholds": thresholds}
        if candidate_engine.model.metadata.get("teaching_session") != binding:
            raise ValueError("Candidate assessment settings changed; assess again before adoption")
        candidate, outcomes = self._measure(candidate_engine, checks)
        incumbent = None
        previous = [None] * len(checks)
        regressions = raw_regressions = 0
        if self.current_path.exists():
            incumbent, previous = self._measure(self._load_engine(self.current_path), checks)
            regressions = sum(old["correct"] and not old["review"] and (not new["correct"] or new["review"])
                              for old, new in zip(previous, outcomes))
            raw_regressions = sum(old["correct"] and not new["correct"] for old, new in zip(previous, outcomes))
        cases = [{"input": row["input"], "label": row["label"], "candidate": new, "incumbent": old,
                  "regression": bool(old and old["correct"] and not old["review"] and (not new["correct"] or new["review"])),
                  "raw_regression": bool(old and old["correct"] and not new["correct"])}
                 for row, new, old in zip(checks, outcomes, previous)]
        reasons = []
        if candidate["raw_accuracy"] < thresholds["min_accuracy"]:
            reasons.append("Raw accuracy is below the required minimum")
        if candidate["coverage"] < thresholds["min_coverage"]:
            reasons.append("Local acceptance coverage is below the required minimum")
        if candidate["accepted_errors"] > thresholds["max_accepted_errors"]:
            reasons.append("Accepted errors exceed the allowed count")
        if regressions > thresholds["max_regressions"]:
            reasons.append("Previously accepted correct checks became wrong or required review")
        diagnostics = []
        if candidate["raw_accuracy"] < thresholds["min_accuracy"]:
            diagnostics.append("The candidate chooses wrong labels before review gating; inspect examples and features. "
                               "More calibration alone does not repair the learned choices.")
        if candidate["review"]:
            diagnostics.append("Some checks require review. This measures withheld decisions, not a proven cause; "
                               "inspect the cases and independent calibration examples.")
        diagnostics.append("These are empirical results on recurring development checks, not a guarantee on new inputs "
                           "or fresh statistical qualification.")
        return {"version": 1, "passed": not reasons, "counts": self._counts(state),
                "candidate": candidate, "incumbent": incumbent, "regressions": regressions,
                "raw_regressions": raw_regressions, "thresholds": thresholds,
                "reasons": reasons, "diagnostics": diagnostics,
                "cases": cases,
                "calibration_counts": candidate_engine.model.metadata["sample_counts"][self.field_name],
                "evidence_scope": "recurring_development_checks", "identity": self._identity(state)}

    def assess(self, *, min_accuracy=.8, min_coverage=.5, max_accepted_errors=0, max_regressions=0):
        """Compile/save/reload a candidate; compare it without replacing the skill.

        ``min_accuracy`` is raw accuracy, before review gating. A regression is a
        previously accepted correct check becoming wrong *or* requiring review.
        Checks may be reused for development, but never enter fitting/calibration.
        """
        thresholds = self._thresholds(min_accuracy, min_coverage, max_accepted_errors, max_regressions)
        state = self._read_state()
        fitting = [row for row in state["records"] if row["split"] == "teach"]
        calibration = [row for row in state["records"] if row["split"] == "calibrate"]
        if not self._counts(state)["evaluate"]:
            raise ValueError("Supply separate evaluate examples before assessing a candidate")
        if len(calibration) < 2:
            raise ValueError("Supply at least two separate calibrate examples")
        if {row["label"] for row in fitting} != set(self.schema.fields[self.field_name].options):
            raise ValueError("Supply teaching examples for every choice before assessing")
        projector = TfidfProjector.fit([row["input"] for row in fitting], max_features=state["max_features"])
        compiler = SystemOneCompiler(self.schema, projector=projector, regularization=state["regularization"])
        model = compiler.compile({self.field_name: [(row["input"], row["label"]) for row in fitting]},
                                 augment=False, calibration_exemplars={self.field_name: [
                                     (row["input"], row["label"]) for row in calibration]})
        model.metadata["teaching_session"] = {
            "session": _digest(state), "current": _file_digest(self.current_path), "thresholds": thresholds}
        temporary = self.directory / ".candidate.s1m"
        try:
            model.save(temporary)
            self._load_engine(temporary)
            temporary.replace(self.candidate_path)
        finally:
            temporary.unlink(missing_ok=True)
        report = self._compare(state, thresholds)
        _write_json(self.report_path, report)
        return report

    def adopt(self):
        """Adopt the exact assessed bytes only while all assessment inputs match."""
        report = self.report
        if report is None:
            raise ValueError("Assess a candidate before adopting it")
        state = self._read_state()
        if report["identity"] != self._identity(state):
            raise ValueError("Candidate assessment is stale; assess the current lessons and skill again")
        thresholds = self._thresholds(**report["thresholds"])
        checked = self._compare(state, thresholds)
        if not checked["passed"] or not report["passed"]:
            raise ValueError("Candidate did not pass its adoption checks")
        if any(checked[key] != report[key] for key in checked):
            raise ValueError("Candidate assessment changed; assess again before adoption")
        artifact = self.candidate_path.read_bytes()
        if hashlib.sha256(artifact).hexdigest() != report["identity"]["candidate"]:
            raise ValueError("Candidate artifact changed; assess again before adoption")
        _atomic_bytes(self.current_path, artifact)
        # Keep the assessment intact: its incumbent digest describes the comparison,
        # and its candidate digest now identifies the approved skill exactly.
        self._engine = None
        self._engine_digest = None
        return report


__all__ = ["TeachingSession"]
