"""Tests for Semantic Sharpness, Contrastive Whitening, and Multi-Exemplar Centroids."""

import numpy as np
import pytest

from system1.core.model import DecisionFieldHead, DeterministicSemanticProjector, SystemOneModel
from system1.core.schema import ChoiceField, DecisionSchema, MultiChoiceField


def test_contrastive_whitening_angular_separation():
    """Verify contrastive whitening widens angular distance between choice prototypes by >= 30%."""
    # ChoiceField with repetitive boilerplate text across all options
    field_boilerplate = ChoiceField(
        options=["billing", "technical", "shipping"],
        descriptions={
            "billing": "Support triage request category for customer inquiry regarding invoice",
            "technical": "Support triage request category for customer inquiry regarding bug error",
            "shipping": "Support triage request category for customer inquiry regarding tracking delivery",
        },
    )

    proj = DeterministicSemanticProjector(dimension=256)

    # 1. Initialize head WITH contrastive whitening
    head_whitened = DecisionFieldHead(
        field_boilerplate,
        dimension=256,
        projector=proj,
        contrastive_whitening=True,
    )

    # 2. Initialize head WITHOUT contrastive whitening
    head_unwhitened = DecisionFieldHead(
        field_boilerplate,
        dimension=256,
        projector=proj,
        contrastive_whitening=False,
    )

    # Compute pairwise cosine similarities: W @ W.T
    sim_whitened = head_whitened.weights @ head_whitened.weights.T
    sim_unwhitened = head_unwhitened.weights @ head_unwhitened.weights.T

    # Average off-diagonal similarity (overlap between distinct options)
    # Cosine distance = 1 - cosine similarity
    off_diag_unwhitened = [
        sim_unwhitened[i, j]
        for i in range(3) for j in range(3) if i != j
    ]
    off_diag_whitened = [
        sim_whitened[i, j]
        for i in range(3) for j in range(3) if i != j
    ]

    avg_dist_unwhitened = 1.0 - float(np.mean(off_diag_unwhitened))
    avg_dist_whitened = 1.0 - float(np.mean(off_diag_whitened))

    # Angular distance expansion ratio
    expansion = (avg_dist_whitened - avg_dist_unwhitened) / max(1e-6, avg_dist_unwhitened)
    assert expansion >= 0.30, (
        f"Expected >= 30% expansion in angular distance, got {expansion*100:.1f}% "
        f"(dist_unwhitened={avg_dist_unwhitened:.4f}, dist_whitened={avg_dist_whitened:.4f})"
    )


def test_contrastive_whitening_mean_cancellation():
    """Verify that centered weights have near-zero mean across options before re-normalization."""
    field = ChoiceField(
        options=["alpha", "beta", "gamma", "delta"],
        descriptions={
            "alpha": "Common prefix category alpha option",
            "beta": "Common prefix category beta option",
            "gamma": "Common prefix category gamma option",
            "delta": "Common prefix category delta option",
        },
    )

    head = DecisionFieldHead(field, dimension=128, contrastive_whitening=True)

    # Weights are unit-normalized after centering
    row_norms = np.linalg.norm(head.weights, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(4), atol=1e-5)

    # Un-normalized centered vectors sum to 0
    proj = head.projector
    raw_W = np.stack([
        proj.project(f"{field.name} option {opt} {opt}: {field.descriptions[opt]}")
        for opt in field.options
    ])
    mu = np.mean(raw_W, axis=0)
    centered = raw_W - mu
    np.testing.assert_allclose(np.sum(centered, axis=0), np.zeros(128), atol=1e-5)


def test_multi_exemplar_prototypical_centroids():
    """Verify that passing a Sequence[str] of exemplars averages into a normalized centroid."""
    field_single = ChoiceField(
        options=["refund", "upgrade"],
        descriptions={
            "refund": "Customer asks to get their money back",
            "upgrade": "Customer asks to move to an enterprise subscription tier",
        },
    )

    # Multi-exemplar variant
    exemplars_refund = [
        "Customer asks to get their money back",
        "I need a full reimbursement on my credit card",
        "Please cancel and issue a refund immediately",
    ]
    field_multi = ChoiceField(
        options=["refund", "upgrade"],
        descriptions={
            "refund": exemplars_refund,
            "upgrade": "Customer asks to move to an enterprise subscription tier",
        },
    )

    assert isinstance(field_multi.descriptions["refund"], tuple)
    assert len(field_multi.descriptions["refund"]) == 3

    proj = DeterministicSemanticProjector(dimension=128)
    head_multi = DecisionFieldHead(field_multi, dimension=128, projector=proj, contrastive_whitening=False)

    # Verify prototype vector matches the normalized mean of individual exemplar vectors
    ex_vecs = [proj.project(ex) for ex in exemplars_refund]
    ex_norms = [v / (np.linalg.norm(v) or 1.0) for v in ex_vecs]
    expected_centroid = np.mean(ex_norms, axis=0)
    expected_centroid /= np.linalg.norm(expected_centroid)

    np.testing.assert_allclose(head_multi.weights[0], expected_centroid, atol=1e-4)


def test_multi_choice_contrastive_whitening_and_exemplars():
    """Verify MultiChoiceField supports contrastive centering and multi-exemplars."""
    field = MultiChoiceField(
        options=["db", "api", "auth"],
        descriptions={
            "db": ["Database query failure", "SQLite lock contention error", "Postgres pool exhaustion"],
            "api": "HTTP 500 error on external endpoint",
            "auth": ["Invalid JWT signature", "Expired bearer token", "Authentication failed"],
        },
    )

    assert isinstance(field.descriptions["db"], tuple)
    assert isinstance(field.descriptions["auth"], tuple)
    assert isinstance(field.descriptions["api"], str)

    head = DecisionFieldHead(field, dimension=128, contrastive_whitening=True)
    assert head.weights.shape == (3, 128)
    norms = np.linalg.norm(head.weights, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_decision_field_head_set_weights():
    """Verify set_weights updates weights and biases dynamically."""
    field = ChoiceField(options=["low", "high"])
    head = DecisionFieldHead(field, dimension=64)

    new_w = np.ones((2, 64), dtype=np.float32) * 0.5
    new_b = np.array([0.1, -0.1], dtype=np.float32)

    head.set_weights(new_w, new_b)
    np.testing.assert_allclose(head.weights, new_w)
    np.testing.assert_allclose(head.biases, new_b)


def test_system_one_model_with_multi_exemplars_end_to_end():
    """Verify SystemOneModel evaluation works cleanly with multi-exemplar schema."""
    class TriageSchema(DecisionSchema):
        intent = ChoiceField(
            options=["greeting", "support", "billing"],
            descriptions={
                "greeting": ["Hello", "Hi there", "Good morning", "Hey assistant"],
                "support": ["Technical problem with my app", "Bug report", "Service is down"],
                "billing": ["Invoice dispute", "Charges on my card", "Subscription fee"],
            },
        )

    model = SystemOneModel(TriageSchema, dimension=128)
    res = model.forward_single("Hey assistant, good morning!")
    assert res.fields["intent"].selected_value == "greeting"
    assert res.fields["intent"].confidence > 0.5

    res_bill = model.forward_single("Why was I charged $50 on my credit card?")
    assert res_bill.fields["intent"].selected_value == "billing"
    assert res_bill.fields["intent"].confidence > 0.5
