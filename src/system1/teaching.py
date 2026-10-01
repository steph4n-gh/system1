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
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

from system1.compiler import CompiledSystemOneModel, SystemOneCompiler
from system1.core.schema import ChoiceField, DecisionSchema
from system1.core.text import MAX_TEXT_LENGTH, TfidfProjector
from system1.engine import SystemOneEngine
from system1.compat.typesafe import _binomial_lower_bound


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

    def __init__(self, directory, schema=None, *, regularization=None, max_features=None, choice_solver=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_path = self.directory / "session.json"
        self.current_path = self.directory / "current.s1m"
        self.candidate_path = self.directory / "candidate.s1m"
        self.report_path = self.directory / "assessment.json"
        self.revisions_path = self.directory / "revisions"
        self.qualifications_path = self.directory / "qualifications"
        self._engine = None
        self._engine_digest = None
        supplied = schema() if isinstance(schema, type) and issubclass(schema, DecisionSchema) else schema
        if self.session_path.exists():
            state = self._read_state()
            if supplied is not None and (not isinstance(supplied, DecisionSchema) or
                                        supplied.to_dict() != state["schema"]):
                raise ValueError("The schema differs from this teaching session")
            for name, value in (("regularization", regularization), ("max_features", max_features),
                                ("choice_solver", choice_solver)):
                if value is not None and value != state.get(name, "ridge"):
                    raise ValueError(f"{name} differs from this teaching session")
        else:
            if not isinstance(supplied, DecisionSchema):
                raise ValueError("A new teaching session needs a single ChoiceField schema")
            state = {"version": 1, "schema": supplied.to_dict(), "records": [],
                     "regularization": .1 if regularization is None else regularization,
                     "max_features": 1024 if max_features is None else max_features,
                     "choice_solver": "ridge" if choice_solver is None else choice_solver}
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
        if state.get("choice_solver", "ridge") not in ("ridge", "logistic"):
            raise ValueError("choice_solver must be ridge or logistic")
        if not isinstance(state["records"], list):
            raise ValueError("Session records must be a list")
        seen, groups = set(), {}
        for row in state["records"]:
            if (not isinstance(row, dict) or not {"input", "label", "split", "source"} <= set(row)
                    or set(row) - {"input", "label", "split", "source", "group"}):
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
            if "group" in row:
                if not isinstance(row["group"], str) or not row["group"].strip():
                    raise ValueError("group must be a nonempty string")
                if row["group"] in groups and groups[row["group"]] != row["split"]:
                    raise ValueError("Related groups must remain disjoint across splits")
                groups[row["group"]] = row["split"]
        return state

    def _read_state(self):
        state = self._validate_state(json.loads(self.session_path.read_text(encoding="utf-8")))
        if hasattr(self, "schema") and state["schema"] != self.schema.to_dict():
            raise ValueError("The schema differs from this teaching session")
        return state

    @property
    def records(self):
        return self._read_state()["records"]

    def record(self, text, label, *, split="teach", source="human", group=None):
        """Add an explicit label, or replace a same-split correction in place."""
        row = {"input": text, "label": label, "split": split, "source": source}
        if group is not None:
            row["group"] = group
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
                    or set(supplied) - (required | {"source", "group"})):
                raise ValueError("A lesson record needs input, label, split, and optional source")
            row = {**supplied, "source": supplied.get("source", "human")}
            self._validate_state({**state, "records": [row]})
            key = _text_key(row["input"])
            for request in self.qualifications_path.glob("*.request.json"):
                reserved = json.loads(request.read_text(encoding="utf-8"))["rows"]
                if any(_text_key(item["input"]) == key or
                       (row.get("group") and row["group"] == item.get("group")) for item in reserved):
                    raise ValueError("Qualification examples must remain disjoint from session lessons")
            for index, previous in enumerate(state["records"]):
                if _text_key(previous["input"]) == key:
                    if "group" in previous and "group" not in row:
                        row["group"] = previous["group"]
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

    def decision_details(self, text):
        """Return the strict decision and observable reasons for requesting review."""
        result = self.predict(text)
        reasons = []
        engine = self.engine
        if not engine.model.projector.project(text).any():
            reasons.append("unknown_vocabulary")
        choices = list(result.conformal_sets[self.field_name])
        if len(choices) != 1:
            reasons.append("ambiguous_prediction_set")
        if result.is_ambiguous and not reasons:
            reasons.append("model_review")
        return {"values": result.values, "review": bool(result.is_ambiguous),
                "prediction_set": choices, "probabilities": result.probabilities,
                "review_reasons": reasons, "revision": _file_digest(self.current_path)}

    @property
    def report(self):
        return json.loads(self.report_path.read_text(encoding="utf-8")) if self.report_path.exists() else None

    @property
    def qualification(self):
        identity = self._identity(self._read_state())
        assessment = self.report
        if assessment and assessment.get("invalidated_by_rollback"):
            return None
        if (assessment and identity["current"] == identity["candidate"] == assessment["identity"]["candidate"]):
            identity["current"] = assessment["identity"]["current"]
        path = self.qualifications_path / f"{identity['candidate']}.json"
        if not path.exists():
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        return report if report["identity"] == identity else None

    @property
    def history(self):
        """Approved artifacts only; these files never enter fitting or calibration."""
        current = _file_digest(self.current_path)
        items = []
        for path in self.revisions_path.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            items.append({"revision": record["revision"], "approved_at": record["approved_at"],
                          "current": record["revision"] == current,
                          "qualification": bool(record.get("qualification", {}).get("passed"))})
        return sorted(items, key=lambda item: (item["approved_at"], item["revision"]))

    def _retain_revision(self, artifact, report, qualification=None):
        revision = hashlib.sha256(artifact).hexdigest()
        self.revisions_path.mkdir(exist_ok=True)
        path = self.revisions_path / f"{revision}.s1m"
        record_path = path.with_suffix(".json")
        if path.exists() and path.read_bytes() != artifact:
            raise ValueError("Stored approved revision integrity failed")
        if record_path.exists():
            self._approved_revision(revision)
            return
        _atomic_bytes(path, artifact)
        record = {"revision": revision, "approved_at": datetime.now(timezone.utc).isoformat(),
                  "schema": self.schema.schema_digest(), "assessment": report,
                  "qualification": qualification or {}}
        _write_json(record_path, record)

    def _approved_revision(self, revision):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{64}", revision):
            raise ValueError("Pass the full approved revision digest")
        path = self.revisions_path / f"{revision}.s1m"
        metadata = path.with_suffix(".json")
        if not path.exists() or not metadata.exists():
            raise ValueError("Approved revision is missing")
        record = json.loads(metadata.read_text(encoding="utf-8"))
        report = record.get("assessment", {})
        if (record.get("revision") != revision or record.get("schema") != self.schema.schema_digest()
                or not report.get("passed") or report.get("identity", {}).get("candidate") != revision
                or _file_digest(path) != revision):
            raise ValueError("Approved revision integrity failed")
        model = self._load_engine(path).model
        thresholds = model.metadata.get("teaching_session", {}).get("thresholds", {})
        if report.get("evidence_scope") != "legacy_approved_skill" and report.get("thresholds") != thresholds:
            raise ValueError("Approved revision assessment settings changed")
        qualification = record.get("qualification", {})
        if thresholds.get("require_qualification") or qualification:
            request_path = self.qualifications_path / f"{revision}.request.json"
            if not request_path.exists() or (thresholds.get("require_qualification") and not qualification.get("passed")):
                raise ValueError("Approved revision qualification evidence is missing")
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if (qualification.get("identity", {}).get("candidate") != revision or
                    self._qualification_report(request, artifact_path=path) != qualification):
                raise ValueError("Approved revision qualification evidence changed")
        return path.read_bytes(), record

    def rollback(self, revision):
        """Explicitly restore exact approved bytes without changing any lessons."""
        artifact, record = self._approved_revision(revision)
        _atomic_bytes(self.current_path, artifact)
        self._engine = None
        self._engine_digest = None
        report = self.report
        if report:
            _write_json(self.report_path, {**report, "invalidated_by_rollback": revision})
        return {"revision": revision, "approved_at": record["approved_at"], "current": True,
                "qualification": bool(record.get("qualification", {}).get("passed"))}

    def snapshot(self):
        state = self._read_state()
        report = self.report
        identity = self._identity(state)
        adopted = bool(report and not report.get("invalidated_by_rollback") and
                       identity["current"] == report["identity"]["candidate"])
        comparison = dict(identity)
        if adopted:
            comparison["current"] = report["identity"]["current"]
        qualification = self.qualification
        stale = bool(report and (report.get("invalidated_by_rollback") or report["identity"] != comparison))
        ready = bool(report and report["passed"] and not stale and not adopted and
                     (not report["thresholds"].get("require_qualification") or
                      qualification and qualification["passed"]))
        return {"records": state["records"], "counts": self._counts(state),
                "current": identity["current"], "report": report, "adopted": adopted,
                "candidate_stale": stale, "settings": {
                    "regularization": state["regularization"], "max_features": state["max_features"],
                    "choice_solver": state.get("choice_solver", "ridge")},
                "history": self.history, "qualification": qualification, "can_adopt": ready}

    @staticmethod
    def _counts(state):
        return {split: sum(row["split"] == split for row in state["records"])
                for split in ("teach", "calibrate", "evaluate")}

    def _identity(self, state):
        return {"session": _digest(state), "candidate": _file_digest(self.candidate_path),
                "current": _file_digest(self.current_path)}

    @staticmethod
    def _thresholds(min_accuracy, min_coverage, max_accepted_errors, max_regressions, require_qualification=False):
        for name, value in (("min_accuracy", min_accuracy), ("min_coverage", min_coverage)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name, value in (("max_accepted_errors", max_accepted_errors), ("max_regressions", max_regressions)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(require_qualification) is not bool:
            raise ValueError("require_qualification must be a boolean")
        return dict(min_accuracy=min_accuracy, min_coverage=min_coverage,
                    max_accepted_errors=max_accepted_errors, max_regressions=max_regressions,
                    require_qualification=require_qualification)

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
        per_class = {}
        for label in self.schema.fields[self.field_name].options:
            selected = [outcome for row, outcome in zip(checks, outcomes) if row["label"] == label]
            local = [outcome for outcome in selected if not outcome["review"]]
            per_class[label] = {"count": len(selected), "raw_correct": sum(item["correct"] for item in selected),
                                "accepted": len(local), "accepted_correct": sum(item["correct"] for item in local),
                                "accepted_errors": sum(not item["correct"] for item in local),
                                "review": len(selected) - len(local)}
        return {"count": count, "raw_correct": correct, "raw_accuracy": correct / count,
                "accepted": accepted, "accepted_correct": accepted_correct,
                "accepted_errors": accepted - accepted_correct, "review": count - accepted,
                "review_rate": (count - accepted) / count, "coverage": accepted / count,
                "accepted_accuracy": accepted_correct / accepted if accepted else None,
                "accepted_correctness_lower_bound": _binomial_lower_bound(accepted_correct, accepted, .025) if accepted else None,
                "coverage_lower_bound": _binomial_lower_bound(accepted, count, .025),
                "per_class": per_class}, outcomes

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

    def assess(self, *, min_accuracy=.8, min_coverage=.5, max_accepted_errors=0, max_regressions=0,
               require_qualification=False):
        """Compile/save/reload a candidate; compare it without replacing the skill.

        ``min_accuracy`` is raw accuracy, before review gating. A regression is a
        previously accepted correct check becoming wrong *or* requiring review.
        Checks may be reused for development, but never enter fitting/calibration.
        """
        thresholds = self._thresholds(min_accuracy, min_coverage, max_accepted_errors, max_regressions, require_qualification)
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
        compiler = SystemOneCompiler(self.schema, projector=projector, regularization=state["regularization"],
                                     choice_solver=state.get("choice_solver", "ridge"))
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

    def qualify(self, rows, *, min_accepted_accuracy=.95, min_coverage=.8, confidence=.95, source=None):
        """Score one supplied independent cohort on the exact assessed candidate.

        Independence and representative sampling remain the caller's responsibility.
        One row per supplied group is required. Both exact one-sided bounds share
        the error budget; repeated development checks are never qualification.
        Requests are retained before scoring, including an interrupted evaluation.
        """
        for name, value in (("min_accepted_accuracy", min_accepted_accuracy),
                            ("min_coverage", min_coverage), ("confidence", confidence)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between 0 and 1 exclusively")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Qualification requires an explicit nonempty source")
        state = self._read_state()
        identity = self._identity(state)
        assessment = self.report
        if not assessment or assessment.get("invalidated_by_rollback") or assessment["identity"] != identity:
            raise ValueError("Assess an unchanged candidate before qualification; assessment is missing or stale")
        if not assessment["passed"]:
            raise ValueError("Candidate did not pass its development assessment")
        checked = self._compare(state, self._thresholds(**assessment["thresholds"]))
        if not checked["passed"] or any(checked[key] != assessment[key] for key in checked):
            raise ValueError("Candidate development assessment changed; assess again before qualification")
        path = self.qualifications_path / f"{identity['candidate']}.json"
        request_path = path.with_suffix(".request.json")
        if request_path.exists():
            raise ValueError("This candidate already has a frozen qualification request; do not reuse or replace it")
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("Supply a nonempty qualification cohort")
        known = {_text_key(row["input"]) for row in state["records"]}
        known_groups = {row["group"] for row in state["records"] if "group" in row}
        requests = list(self.qualifications_path.glob("*.request.json"))
        for previous in requests:
            saved = json.loads(previous.read_text(encoding="utf-8"))
            if saved["thresholds"]["confidence"] != confidence:
                raise ValueError("Qualification confidence is fixed across this session's attempts")
            reserved = saved["rows"]
            known.update(_text_key(row["input"]) for row in reserved)
            known_groups.update(row["group"] for row in reserved if "group" in row)
        seen, groups, cohort = set(), set(), []
        for row in rows:
            if (not isinstance(row, dict) or not {"input", "label"} <= set(row)
                    or set(row) - {"input", "label", "group"}):
                raise ValueError("Qualification rows need input, label, and optional group")
            key = _text_key(row["input"])
            if key in known or key in seen:
                raise ValueError("Qualification inputs must be disjoint from session records and unique")
            if row["label"] not in self.schema.fields[self.field_name].options:
                raise ValueError("Qualification label must match a choice")
            group = row.get("group")
            if "group" in row:
                if not isinstance(group, str) or not group.strip():
                    raise ValueError("Qualification group must be nonempty")
                if group in known_groups or group in groups:
                    raise ValueError("Qualification groups must be disjoint, with one row per group")
                groups.add(group)
            seen.add(key)
            cohort.append(dict(row))
        thresholds = {"min_accepted_accuracy": min_accepted_accuracy, "min_coverage": min_coverage,
                      "confidence": confidence}
        request = {"identity": identity, "source": source, "thresholds": thresholds,
                   "attempt": len(requests) + 1, "rows": cohort}
        self.qualifications_path.mkdir(exist_ok=True)
        # Freeze the whole cohort and policy before the first prediction.
        _write_json(request_path, request)
        report = self._qualification_report(request)
        if identity != self._identity(self._read_state()):
            raise ValueError("Candidate or session changed during qualification")
        _write_json(path, report)
        return report

    def _qualification_report(self, request, *, artifact_path=None):
        thresholds = request["thresholds"]
        candidate, outcomes = self._measure(self._load_engine(artifact_path or self.candidate_path), request["rows"])
        attempt = request["attempt"]
        failure = (1 - thresholds["confidence"]) / (2 * attempt * (attempt + 1))
        candidate["accepted_correctness_lower_bound"] = (
            _binomial_lower_bound(candidate["accepted_correct"], candidate["accepted"], failure)
            if candidate["accepted"] else None)
        candidate["coverage_lower_bound"] = _binomial_lower_bound(candidate["accepted"], candidate["count"], failure)
        reasons = []
        if (candidate["accepted_correctness_lower_bound"] is None or
                candidate["accepted_correctness_lower_bound"] < thresholds["min_accepted_accuracy"]):
            reasons.append("Accepted correctness lower bound is below the required minimum")
        if candidate["coverage_lower_bound"] < thresholds["min_coverage"]:
            reasons.append("Coverage lower bound is below the required minimum")
        report = {"version": 1, "identity": request["identity"], "source": request["source"], "thresholds": thresholds,
                  "qualification_id": _digest(request), "candidate": candidate, "passed": not reasons,
                  "reasons": reasons, "evidence_scope": "supplied_independent_qualification",
                  "attempt": attempt, "confidence_scope": "All qualification attempts in this session",
                  "sampling_assumption": "Caller-supplied independent, representative groups; not verified by software",
                  "cases": [{**row, "candidate": outcome} for row, outcome in zip(request["rows"], outcomes)]}
        return report

    def adopt(self):
        """Adopt the exact assessed bytes only while all assessment inputs match."""
        report = self.report
        if report is None:
            raise ValueError("Assess a candidate before adopting it")
        if report.get("invalidated_by_rollback"):
            raise ValueError("Candidate assessment is stale after rollback; assess again before adoption")
        state = self._read_state()
        if report["identity"] != self._identity(state):
            raise ValueError("Candidate assessment is stale; assess the current lessons and skill again")
        thresholds = self._thresholds(**report["thresholds"])
        checked = self._compare(state, thresholds)
        if not checked["passed"] or not report["passed"]:
            raise ValueError("Candidate did not pass its adoption checks")
        if any(checked[key] != report[key] for key in checked):
            raise ValueError("Candidate assessment changed; assess again before adoption")
        qualification = self.qualification
        if thresholds["require_qualification"] and (not qualification or not qualification["passed"]):
            raise ValueError("A passing independent qualification is required before adoption")
        if qualification:
            request_path = self.qualifications_path / f"{report['identity']['candidate']}.request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if _digest(request) != qualification["qualification_id"]:
                raise ValueError("Qualification evidence changed")
            if self._qualification_report(request) != qualification:
                raise ValueError("Qualification evidence changed")
        artifact = self.candidate_path.read_bytes()
        if hashlib.sha256(artifact).hexdigest() != report["identity"]["candidate"]:
            raise ValueError("Candidate artifact changed; assess again before adoption")
        # Preserve an approved 1.1 skill even when its old assessment was overwritten.
        if self.current_path.exists() and not (self.revisions_path / f"{report['identity']['current']}.json").exists():
            legacy = {"passed": True, "identity": {"candidate": report["identity"]["current"]},
                      "evidence_scope": "legacy_approved_skill", "assessment_available": False}
            self._retain_revision(self.current_path.read_bytes(), legacy)
        self._retain_revision(artifact, report, qualification)
        _atomic_bytes(self.current_path, artifact)
        # Keep the assessment intact: its incumbent digest describes the comparison,
        # and its candidate digest now identifies the approved skill exactly.
        self._engine = None
        self._engine_digest = None
        return report


__all__ = ["TeachingSession"]
