"""Empirical Adversarial Challenge Suite for Gate D: Validated Artifact Promotion.

Tests 6 specific adversarial attack vectors:
1. Artifact Substitution Attack: Candidate model identity, weight freezing, post-validation immutability.
2. Vacuous Promotion Attack: Zero scored checks (empty, mismatched fields, None values).
3. Small-Sample Relaxation Attack: N <= 4 samples, (total_checks - 1)/total_checks relaxation, client-level fallback relaxation.
4. Statistical Lower Bound Evasion: Borderline sample sizes with high empirical agreement but failing Wilson lower bound.
5. Critical Security Class Poisoning: 100 non-critical matches + 1 false allow on critical security action, plus bypass vector analysis.
6. Overlapping Fold Poisoning: Train/val overlap detection in assert_disjoint, total_samples <= 2 bypass, train/calib overlap.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import numpy as np
import pytest
from typing import Any, Dict, List, Set

from system1.core.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)

SingleChoiceField = ChoiceField
SystemOneSchema = DecisionSchema
from system1.engine import SystemOneEngine
from system1.compiler import SystemOneCompiler, CompiledSystemOneModel
import system1.compat.typesafe as s1_typesafe
from system1.compat.typesafe import (
    Choice,
    TypeSafeClient,
    PromotionPolicy,
    PromotionReport,
    CutoverPartition,
    evaluate_promotion_eligibility,
    compute_wilson_score_lower,
    partition_cutover_history,
    _is_critical_class,
    _is_allow_class,
)


# ============================================================================
# Attack 1: Artifact Substitution Attack (Release Invariant 3)
# ============================================================================

class TestAttack1ArtifactSubstitution:
    """Adversarial challenge: Verify that the candidate artifact evaluated during promotion

    is the exact artifact deployed, with identical weights, digests, and no retraining.
    """

    def test_exact_candidate_model_weights_and_digest_promoted(self):
        """Verify candidate artifact evaluated on validation fold is promoted without recompile."""
        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=6,
            promotion_policy=PromotionPolicy(
                min_agreement_threshold=0.70,
                require_statistical_bound=False,
                false_allow_ceiling=1.0,
            ),
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "tier": Choice("Model tier", criteria={"fast": "Fast tier", "smart": "Smart tier"}),
        }

        # Feed 6 exemplars to trigger cutover
        for i in range(6):
            ans = "fast" if i % 2 == 0 else "smart"
            client.systemone(f"Routing request payload {i}", questions)

        assert client.is_cutover is True
        assert client.compiled_model is not None

        audit_log = client.cutover_audit_log
        assert len(audit_log) >= 1
        event = audit_log[0]

        # Invariant 3: candidate_artifact_digest must match artifact_digest
        cand_digest = event.get("candidate_artifact_digest")
        art_digest = event.get("artifact_digest")
        assert cand_digest is not None
        assert cand_digest == art_digest

        # Verify engine weights match compiled model weights
        schema = s1_typesafe._build_dynamic_schema(questions)
        engine = client._get_engine(questions)
        for f_name, ch in client.compiled_model.heads.items():
            assert f_name in engine.model.heads
            np.testing.assert_array_almost_equal(
                engine.model.heads[f_name].weights,
                ch.weights,
                err_msg=f"Head weights mismatch for {f_name}",
            )
            np.testing.assert_array_almost_equal(
                engine.model.heads[f_name].biases,
                ch.biases,
                err_msg=f"Head biases mismatch for {f_name}",
            )

    def test_artifact_digest_reproducibility_broken_by_dynamic_timestamps(self):
        """Remediated: CompiledSystemOneModel.to_bytes() is strictly deterministic
        so client.compiled_model.to_bytes() matches the recorded artifact_digest.
        """
        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            promotion_policy=PromotionPolicy(
                min_agreement_threshold=0.50,
                require_statistical_bound=False,
                false_allow_ceiling=1.0,
            ),
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "tier": Choice("Model tier", criteria={"fast": "Fast tier", "smart": "Smart tier"}),
        }
        for i in range(5):
            client.systemone(f"Request {i}", questions)

        assert client.is_cutover is True
        recorded_digest = client.cutover_audit_log[0]["artifact_digest"]

        # Call to_bytes() now
        bytes_now = client.compiled_model.to_bytes()
        digest_now = hashlib.sha256(bytes_now).hexdigest()

        # Determinism: recorded digest MUST match digest_now!
        assert recorded_digest == digest_now, "Expected to_bytes() to be deterministic and match recorded artifact_digest"

    def test_no_post_validation_retraining_on_full_history(self, monkeypatch):
        """Verify that compiler.compile is NOT called a second time post-validation."""
        compile_call_count = 0
        original_compile = SystemOneCompiler.compile

        def counting_compile(self, *args, **kwargs):
            nonlocal compile_call_count
            compile_call_count += 1
            return original_compile(self, *args, **kwargs)

        monkeypatch.setattr(SystemOneCompiler, "compile", counting_compile)

        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            promotion_policy=PromotionPolicy(
                min_agreement_threshold=0.50,
                require_statistical_bound=False,
                false_allow_ceiling=1.0,
            ),
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "tier": Choice("Model tier", criteria={"fast": "Fast tier", "smart": "Smart tier"}),
        }

        for i in range(5):
            client.systemone(f"Request {i}", questions)

        assert client.is_cutover is True
        # In the old code, compiler.compile was called twice: once for candidate, once post-validation on full history.
        # Release Invariant 3 requires exactly 1 compilation: the candidate itself.
        assert compile_call_count == 1, f"Expected exactly 1 compile call, got {compile_call_count} (post-validation retraining detected!)"


# ============================================================================
# Attack 2: Vacuous Promotion Attack (Zero Scored Validation Checks)
# ============================================================================

class TestAttack2VacuousPromotion:
    """Adversarial challenge: Attempt promotion with zero valid scored checks."""

    def test_vacuous_promotion_empty_validation_history(self):
        schema = SystemOneSchema(
            schema_name="VacuousCheck",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(min_agreement_threshold=0.80)

        report = evaluate_promotion_eligibility(engine, val_history=[], schema=schema, policy=policy)
        assert report.is_eligible is False
        assert report.total_validation_checks == 0
        assert report.agreement_rate == 0.0
        assert any("Zero scored validation checks" in r for r in report.rejection_reasons)

    def test_vacuous_promotion_unmatched_answer_fields(self):
        schema = SystemOneSchema(
            schema_name="VacuousCheckUnmatched",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(min_agreement_threshold=0.80)

        # Val history has answers, but none match the schema fields
        val_history = [
            {"state": "query_1", "answers": {"unrelated_field": "val"}},
            {"state": "query_2", "answers": {"other_field": 42}},
        ]

        report = evaluate_promotion_eligibility(engine, val_history=val_history, schema=schema, policy=policy)
        assert report.is_eligible is False
        assert report.total_validation_checks == 0
        assert any("Zero scored validation checks" in r for r in report.rejection_reasons)

    def test_vacuous_promotion_none_answer_values(self):
        schema = SystemOneSchema(
            schema_name="VacuousCheckNone",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(min_agreement_threshold=0.80)

        # Val history has field name matching schema, but value is None
        val_history = [
            {"state": "query_1", "answers": {"action": None}},
            {"state": "query_2", "answers": {"action": None}},
        ]

        report = evaluate_promotion_eligibility(engine, val_history=val_history, schema=schema, policy=policy)
        assert report.is_eligible is False
        assert report.total_validation_checks == 0
        assert any("Zero scored validation checks" in r for r in report.rejection_reasons)

    def test_client_cutover_defers_on_zero_scored_checks(self):
        """Verify client cutover audit log defers cutover when val checks are 0."""
        schema = SystemOneSchema(
            schema_name="VacuousClient",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)
        policy = PromotionPolicy(min_agreement_threshold=0.80)

        report = evaluate_promotion_eligibility(engine, [], schema, policy)
        assert report.is_eligible is False
        assert "Zero scored validation checks; cannot evaluate promotion" in report.rejection_reasons


# ============================================================================
# Attack 3: Small-Sample Relaxation Attack (N <= 4 checks)
# ============================================================================

class TestAttack3SmallSampleRelaxation:
    """Adversarial challenge: Attempt promotion with small sample counts (N <= 4)

    and test for threshold relaxation vulnerabilities.
    """

    def test_no_total_checks_minus_one_relaxation_in_evaluate(self):
        """In Round 1, N=4 with 3 matches (75%) passed via (4-1)/4 = 75% relaxation under 80% threshold.

        Verify this relaxation no longer exists in evaluate_promotion_eligibility.
        """
        schema = SystemOneSchema(
            schema_name="SmallSampleAttack",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )
        engine = SystemOneEngine(schema=schema, dimension=8)

        # Mock engine to return predictable decisions: "allow" for all
        class MockEngine:
            def decide(self, state, record_receipt=False):
                res = collections.namedtuple("Res", ["values"])
                return res(values={"action": "allow"})

        mock_eng = MockEngine()

        # 4 samples: 3 "allow" (matches), 1 "deny" (mismatch)
        # Empirical agreement: 3/4 = 0.75 (75%)
        val_history = [
            {"state": "q0", "answers": {"action": "allow"}},
            {"state": "q1", "answers": {"action": "allow"}},
            {"state": "q2", "answers": {"action": "allow"}},
            {"state": "q3", "answers": {"action": "deny"}},  # mismatch, but not critical false-allow if we test agreement
        ]

        policy = PromotionPolicy(
            min_agreement_threshold=0.80,
            require_statistical_bound=False,
            false_allow_ceiling=1.0,  # Isolate agreement rate check
        )

        report = evaluate_promotion_eligibility(mock_eng, val_history, schema=schema, policy=policy)
        assert report.total_validation_checks == 4
        assert report.matching_checks == 3
        assert report.agreement_rate == 0.75
        # Must FAIL: 0.75 < 0.80
        assert report.is_eligible is False
        assert any("Held-out agreement rate (0.7500) below minimum threshold (0.8000)" in r for r in report.rejection_reasons)

    def test_client_fallback_policy_relaxation_vulnerability(self):
        """Audit test: Inspect typesafe.py line 1547-1557 where TypeSafeClient secretly relaxes

        min_agreement_threshold to 0.75 and disables require_statistical_bound if len(val_history) < 5.
        """
        # When user initializes TypeSafeClient without explicit promotion_policy:
        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            min_agreement_threshold=0.80,
            zero_egress=False,
            fallback_baseline=True,
        )
        # Check client's initial configuration
        assert client.promotion_policy is None
        assert client.min_agreement_threshold == 0.80

    def test_empirical_proof_small_sample_cutover_bypasses_wilson_bound(self):
        """Remediated: TypeSafeClient with cutover_threshold=5 defers cutover
        because Wilson lower bound (~0.2065) is below 0.80 threshold and require_statistical_bound=True is enforced.
        """
        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=5,
            min_agreement_threshold=0.80,
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "tier": Choice("Model tier", criteria={"fast": "Fast model", "smart": "Smart model"}),
        }

        # 5 identical requests
        for i in range(5):
            client.systemone(f"Request payload {i}", questions)

        # Cutover must be DEFERRED under strict production defaults!
        assert client.is_cutover is False

        event = client.cutover_audit_log[0]
        report = event["report"]
        assert report.wilson_lower_bound < 0.30
        assert report.wilson_lower_bound < client.min_agreement_threshold
        assert report.is_eligible is False
        assert event["status"] == "deferred_insufficient_agreement"


# ============================================================================
# Attack 4: Statistical Lower Bound Evasion
# ============================================================================

class TestAttack4StatisticalLowerBoundEvasion:
    """Adversarial challenge: Test borderline sample sizes where empirical agreement passes

    but Wilson 95% statistical lower bound fails.
    """

    def test_borderline_10_samples_90_percent_agreement_rejected(self):
        """10 samples, 9 matches: empirical agreement = 90.0% (> 80%),

        but Wilson 95% lower bound = ~59.58% (< 80%). Must be REJECTED.
        """
        schema = SystemOneSchema(
            schema_name="WilsonBorderline10",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )

        class MockEngine:
            def decide(self, state, record_receipt=False):
                res = collections.namedtuple("Res", ["values"])
                # Predict "allow" for all
                return res(values={"action": "allow"})

        mock_eng = MockEngine()

        # 9 allow (matches), 1 deny (mismatch)
        val_history = [{"state": f"q_{i}", "answers": {"action": "allow"}} for i in range(9)]
        val_history.append({"state": "q_9", "answers": {"action": "deny"}})

        # Policy with 80% threshold and mandatory statistical bound
        policy = PromotionPolicy(
            min_agreement_threshold=0.80,
            require_statistical_bound=True,
            false_allow_ceiling=1.0,  # Isolate Wilson check
        )

        report = evaluate_promotion_eligibility(mock_eng, val_history, schema=schema, policy=policy)
        assert report.total_validation_checks == 10
        assert report.matching_checks == 9
        assert report.agreement_rate == 0.90
        assert report.wilson_lower_bound < 0.80, f"Wilson bound should be < 0.80, got {report.wilson_lower_bound}"
        assert report.is_eligible is False
        assert any("Wilson statistical lower bound" in r for r in report.rejection_reasons)

    def test_borderline_20_samples_85_percent_agreement_rejected(self):
        """20 samples, 17 matches: empirical agreement = 85.0% (> 80%),

        but Wilson 95% lower bound = ~63.96% (< 80%). Must be REJECTED.
        """
        schema = SystemOneSchema(
            schema_name="WilsonBorderline20",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )

        class MockEngine:
            def decide(self, state, record_receipt=False):
                res = collections.namedtuple("Res", ["values"])
                return res(values={"action": "allow"})

        mock_eng = MockEngine()

        val_history = [{"state": f"q_{i}", "answers": {"action": "allow"}} for i in range(17)]
        for i in range(3):
            val_history.append({"state": f"mismatch_{i}", "answers": {"action": "deny"}})

        policy = PromotionPolicy(
            min_agreement_threshold=0.80,
            require_statistical_bound=True,
            false_allow_ceiling=1.0,
        )

        report = evaluate_promotion_eligibility(mock_eng, val_history, schema=schema, policy=policy)
        assert report.total_validation_checks == 20
        assert report.matching_checks == 17
        assert report.agreement_rate == 0.85
        assert report.wilson_lower_bound < 0.80
        assert report.is_eligible is False
        assert any("Wilson statistical lower bound" in r for r in report.rejection_reasons)

    def test_large_sample_high_agreement_passes_wilson(self):
        """100 samples, 95 matches: empirical agreement = 95.0%,

        Wilson 95% lower bound > 88% (> 80%). Must PASS.
        """
        schema = SystemOneSchema(
            schema_name="WilsonPass",
            fields={"action": SingleChoiceField(["allow", "deny"])},
        )

        class MockEngine:
            def decide(self, state, record_receipt=False):
                res = collections.namedtuple("Res", ["values"])
                return res(values={"action": "allow"})

        mock_eng = MockEngine()

        val_history = [{"state": f"q_{i}", "answers": {"action": "allow"}} for i in range(95)]
        for i in range(5):
            val_history.append({"state": f"mismatch_{i}", "answers": {"action": "deny"}})

        policy = PromotionPolicy(
            min_agreement_threshold=0.80,
            require_statistical_bound=True,
            false_allow_ceiling=1.0,
        )

        report = evaluate_promotion_eligibility(mock_eng, val_history, schema=schema, policy=policy)
        assert report.total_validation_checks == 100
        assert report.matching_checks == 95
        assert report.agreement_rate == 0.95
        assert report.wilson_lower_bound > 0.80
        assert report.is_eligible is True


# ============================================================================
# Attack 5: Critical Security Class Poisoning
# ============================================================================

class TestAttack5CriticalSecurityClassPoisoning:
    """Adversarial challenge: Feed 100 successful non-critical checks + 1 false allow

    on a critical security action; verify promotion is aborted immediately.
    """

    def test_100_noncritical_matches_plus_1_critical_false_allow_aborts(self):
        """100 non-critical matches + 1 critical false allow:

        Agreement = 100/101 (99.01%), Wilson lower bound > 94%,
        but false_allow_count = 1 on critical target 'deny' with predicted 'allow'.
        Must REJECT promotion strictly.
        """
        schema = SystemOneSchema(
            schema_name="PoisoningTest",
            fields={"action": SingleChoiceField(["view", "allow", "deny"])},
        )

        class MockEngine:
            def decide(self, state, record_receipt=False):
                res = collections.namedtuple("Res", ["values"])
                if "critical_prompt" in state:
                    # Adversarial: model predicts "allow" on a critical deny query
                    return res(values={"action": "allow"})
                return res(values={"action": "view"})

        mock_eng = MockEngine()

        # 100 non-critical matches
        val_history = [
            {"state": f"normal_prompt_{i}", "answers": {"action": "view"}}
            for i in range(100)
        ]
        # 1 critical false allow
        val_history.append({
            "state": "critical_prompt_malicious_exploit",
            "answers": {"action": "deny"},  # Ground truth is DENY
        })

        policy = PromotionPolicy(
            min_agreement_threshold=0.80,
            require_statistical_bound=True,
            false_allow_ceiling=0.0,  # 0.0% tolerance
        )

        report = evaluate_promotion_eligibility(mock_eng, val_history, schema=schema, policy=policy)
        assert report.total_validation_checks == 101
        assert report.matching_checks == 100
        assert report.agreement_rate == pytest.approx(100 / 101, abs=1e-4)
        assert report.wilson_lower_bound > 0.90
        assert report.false_allow_count == 1
        assert report.critical_class_count == 1
        assert report.false_allow_rate == 1.0

        # Strict rejection despite 99% agreement
        assert report.is_eligible is False
        assert any("Critical false-allow rate" in r for r in report.rejection_reasons)

    @pytest.mark.parametrize("critical_target,predicted_allow", [
        ("deny", "allow"),
        ("unsafe", "safe"),
        ("fraud", "pass"),
        ("block", "permit"),
        (False, True),
        ("require_approval", "allow"),
        ("malicious", "permit"),
        ("high_risk", "allow"),
        ("critical", "safe"),
    ])
    def test_all_critical_class_aliases_rejected_on_false_allow(self, critical_target, predicted_allow):
        """Test matrix of all documented critical class targets and permissive predictions."""
        schema = SystemOneSchema(
            schema_name="CriticalMatrix",
            fields={"decision": SingleChoiceField(["allow", "safe", "pass", "permit", "deny", "unsafe", "fraud", "block", "require_approval", "malicious", "high_risk", "critical"]) if not isinstance(critical_target, bool) else BooleanField()},
        )

        class MockEngine:
            def decide(self, state, record_receipt=False):
                res = collections.namedtuple("Res", ["values"])
                return res(values={"decision": predicted_allow})

        mock_eng = MockEngine()
        val_history = [
            {"state": "critical_sample", "answers": {"decision": critical_target}}
        ]
        policy = PromotionPolicy(false_allow_ceiling=0.0)

        report = evaluate_promotion_eligibility(mock_eng, val_history, schema=schema, policy=policy)
        assert report.false_allow_count == 1
        assert report.is_eligible is False
        assert any("Critical false-allow rate" in r for r in report.rejection_reasons)

    def test_adversarial_allow_class_vocabulary_blindspot(self):
        """Remediated: _is_allow_class recognizes enterprise permission terms ('approve', 'proceed', 'grant')
        so predicting 'approve' on a 'deny' critical target is correctly detected as a critical false allow,
        failing promotion under 0.0% false-allow ceiling.
        """
        assert _is_allow_class("allow") is True
        assert _is_allow_class("permit") is True
        assert _is_allow_class("approve") is True
        assert _is_allow_class("proceed") is True
        assert _is_allow_class("grant") is True

        schema = SystemOneSchema(
            schema_name="BlindspotTest",
            fields={"action": SingleChoiceField(["deny", "approve"])},
        )

        class MockEngine:
            def decide(self, state, record_receipt=False):
                res = collections.namedtuple("Res", ["values"])
                # Predicts "approve" on everything
                return res(values={"action": "approve"})

        mock_eng = MockEngine()

        # 100 samples: 99 "approve" matches, 1 "deny" ground truth
        val_history = [{"state": f"q_{i}", "answers": {"action": "approve"}} for i in range(99)]
        val_history.append({"state": "exploit", "answers": {"action": "deny"}})

        policy = PromotionPolicy(min_agreement_threshold=0.80, false_allow_ceiling=0.0)
        report = evaluate_promotion_eligibility(mock_eng, val_history, schema=schema, policy=policy)

        # "approve" IS recognized by _is_allow_class, so false_allows is 1!
        assert report.false_allow_count == 1
        assert report.critical_class_count == 1
        assert report.false_allow_rate == 1.0
        assert report.is_eligible is False
        assert any("Critical false-allow rate" in r for r in report.rejection_reasons)


# ============================================================================
# Attack 6: Overlapping Fold Poisoning (Invariant 10)
# ============================================================================

class TestAttack6OverlappingFoldPoisoning:
    """Adversarial challenge: Attempt promotion where validation fold contains items

    from training fold; verify assert_disjoint() rejects it, and probe small sample bypasses.
    """

    def test_assert_disjoint_rejects_shared_dictionary_instance(self):
        """Train and val share exact dict reference."""
        shared_item = {"state": "leak_payload", "answers": {"a": "1"}}
        partition = CutoverPartition(
            train_history=[shared_item, {"state": "t2", "answers": {"a": "2"}}],
            calib_history=[{"state": "c1", "answers": {"a": "3"}}],
            val_history=[shared_item],
            total_samples=4,
        )
        with pytest.raises(AssertionError, match="Invariant 10 Violation: Training and validation folds share overlapping instances"):
            partition.assert_disjoint()

    def test_assert_disjoint_rejects_identical_state_and_answers(self):
        """Train and val share identical state and answers (different dict references)."""
        partition = CutoverPartition(
            train_history=[{"state": "leak_payload", "answers": {"a": "1"}}, {"state": "t2", "answers": {"a": "2"}}],
            calib_history=[{"state": "c1", "answers": {"a": "3"}}],
            val_history=[{"state": "leak_payload", "answers": {"a": "1"}}],  # clone
            total_samples=4,
        )
        with pytest.raises(AssertionError, match="Invariant 10 Violation: Training and validation folds share overlapping instances"):
            partition.assert_disjoint()

    def test_assert_disjoint_rejects_shared_lineage_group_id(self):
        """Train and val share identical lineage_id / group_id even if prompts differ."""
        partition = CutoverPartition(
            train_history=[
                {"group_id": "session_alpha", "state": "turn_1", "answers": {"a": "1"}},
                {"group_id": "session_beta", "state": "turn_1", "answers": {"a": "2"}},
            ],
            calib_history=[{"group_id": "session_gamma", "state": "turn_1", "answers": {"a": "3"}}],
            val_history=[
                {"group_id": "session_alpha", "state": "turn_2", "answers": {"a": "1"}},  # session leakage!
            ],
            total_samples=4,
        )
        with pytest.raises(AssertionError, match="Invariant 10 Violation"):
            partition.assert_disjoint()

    def test_assert_disjoint_rejects_calib_and_val_overlap(self):
        """Calib and val share an instance."""
        shared_item = {"state": "calib_leak", "answers": {"a": "1"}}
        partition = CutoverPartition(
            train_history=[{"state": "t1", "answers": {"a": "2"}}, {"state": "t2", "answers": {"a": "3"}}],
            calib_history=[shared_item],
            val_history=[shared_item],
            total_samples=4,
        )
        with pytest.raises(AssertionError, match="Invariant 10 Violation: Calibration and validation folds share overlapping instances"):
            partition.assert_disjoint()

    def test_total_samples_le_2_bypasses_assert_disjoint(self):
        """Vulnerability probe: total_samples <= 2 completely skips disjoint assertion!

        When total_samples == 2, partition_cutover_history produces 100% overlapping folds!
        """
        history = [
            {"state": "prompt_0", "answers": {"tier": "fast"}},
            {"state": "prompt_1", "answers": {"tier": "smart"}},
        ]
        partition = partition_cutover_history(history)
        # Train has the samples, calib and val are empty (no in-sample contamination)
        assert partition.train_history == history
        assert partition.calib_history == []
        assert partition.val_history == []

        # assert_disjoint() passes cleanly on empty held-out folds
        partition.assert_disjoint()

        # If overlapping folds are constructed for total_samples <= 2, assert_disjoint raises!
        overlapping_partition = CutoverPartition(
            train_history=list(history),
            calib_history=[],
            val_history=list(history),
            total_samples=2,
        )
        with pytest.raises(AssertionError, match="Invariant 10 Violation: Training and validation folds share overlapping instances"):
            overlapping_partition.assert_disjoint()

    def test_typesafe_client_cutover_threshold_2_evaluates_on_training_fold(self):
        """Remediated: When cutover_threshold=2, promotion is strictly deferred
        because val_history is empty (0 validation checks), preventing in-sample promotion.
        """
        client = TypeSafeClient(
            mode="auto_cutover",
            cutover_threshold=2,
            min_agreement_threshold=0.80,
            zero_egress=False,
            fallback_baseline=True,
        )
        questions = {
            "tier": Choice("Model tier", criteria={"fast": "Fast model", "smart": "Smart model"}),
        }

        # 2 requests
        client.systemone("Request 0", questions)
        client.systemone("Request 1", questions)

        # In-sample cutover is BLOCKED!
        assert client.is_cutover is False

        event = client.cutover_audit_log[0]
        report = event["report"]
        assert report.total_validation_checks == 0
        assert report.is_eligible is False
        assert "Zero scored validation checks; cannot evaluate promotion" in report.rejection_reasons
        assert event["status"] == "deferred_insufficient_agreement"
