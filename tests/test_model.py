"""Tests for Reflex Non-Autoregressive Decision Model."""

import time
import numpy as np
import pytest

from reflex.model import SystemOneModel
from reflex.schema import (
    BooleanField,
    ChoiceField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)


class BenchmarkSchema(DecisionSchema):
    route = ChoiceField(
        options=["file_io", "sql_query", "network_call", "shell_exec"],
        descriptions={
            "file_io": "Reading or writing local files on the filesystem",
            "sql_query": "SQL queries, SELECT, INSERT, database tables and rows",
            "network_call": "External HTTP API or WebSocket call",
            "shell_exec": "Executing bash commands or subprocesses in shell",
        },
    )
    is_dangerous = BooleanField(
        threshold=0.5,
        true_description="Dangerous, destructive, or unauthorized command",
        false_description="Benign, read-only, or harmless task",
    )
    risk_score = ScoreField(
        min_value=0.0,
        max_value=1.0,
        low_description="Zero risk benign operation",
        high_description="Catastrophic data loss or system compromise",
    )
    tags = MultiChoiceField(
        options=["audit", "privileged", "ephemeral"],
        descriptions={
            "audit": "Requires audit logging",
            "privileged": "Requires elevated administrative credentials",
            "ephemeral": "Temporary scratch operation",
        },
    )


def test_model_single_pass_evaluation():
    schema = BenchmarkSchema()
    model = SystemOneModel(schema)

    res = model.forward_single("Read local file from /src/reflex/cli.py")

    assert res.prompt == "Read local file from /src/reflex/cli.py"
    assert "route" in res.fields
    assert "is_dangerous" in res.fields
    assert "risk_score" in res.fields
    assert "tags" in res.fields

    # Check semantic routing accuracy
    assert res.fields["route"].selected_value == "file_io"
    assert res.fields["is_dangerous"].selected_value is False
    assert 0.0 <= res.fields["risk_score"].selected_value <= 1.0

    # Latency should be sub-20ms
    assert res.inference_latency_ms < 20.0


def test_model_semantic_discrimination():
    schema = BenchmarkSchema()
    model = SystemOneModel(schema)

    # Test distinct domain intents
    r_db = model.forward_single("SELECT * FROM action_ledger WHERE action_id = '123'")
    assert r_db.fields["route"].selected_value == "sql_query"

    r_net = model.forward_single("Send POST request to https://api.example.com/v1/telemetry")
    assert r_net.fields["route"].selected_value == "network_call"

    r_shell = model.forward_single("Execute bash shell script rm -rf /var/log/old")
    assert r_shell.fields["route"].selected_value == "shell_exec"
    assert r_shell.fields["is_dangerous"].selected_value is True
    assert r_shell.fields["risk_score"].selected_value > 0.5


def test_model_batch_forward_consistency():
    schema = BenchmarkSchema()
    model = SystemOneModel(schema)

    prompts = [
        "Read file config.yaml",
        "Query database users",
        "Fetch remote webhook",
    ]

    batch_res = model.forward_batch(prompts)
    assert len(batch_res) == 3

    for p, b_item in zip(prompts, batch_res):
        single_res = model.forward_single(p)
        assert b_item.fields["route"].selected_value == single_res.fields["route"].selected_value
        np.testing.assert_allclose(
            b_item.fields["route"].logits,
            single_res.fields["route"].logits,
            rtol=1e-5,
        )


def test_model_sub_20ms_latency_guarantee():
    schema = BenchmarkSchema()
    model = SystemOneModel(schema)

    prompt = "Inspect repository file tree and check branch status"
    for _ in range(5):
        model.forward_single(prompt)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.forward_single(prompt)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p95 = np.percentile(latencies, 95)
    p50 = np.median(latencies)

    assert p50 < 10.0, f"P50 latency {p50:.2f}ms exceeds 10ms"
    assert p95 < 20.0, f"P95 latency {p95:.2f}ms exceeds 20ms target"


def test_model_metal_mlx_execution():
    schema = BenchmarkSchema()
    model_mlx = SystemOneModel(schema, backend="mlx")
    model_np = SystemOneModel(schema, backend="numpy")

    prompt = "Execute bash shell script rm -rf /var/log/old"
    res_mlx = model_mlx.forward_single(prompt)
    res_np = model_np.forward_single(prompt)

    assert model_mlx.backend == "mlx"
    assert model_np.backend == "numpy"

    for field_name in schema.fields:
        f_mlx = res_mlx.fields[field_name]
        f_np = res_np.fields[field_name]
        if isinstance(f_mlx.selected_value, float):
            assert np.isclose(f_mlx.selected_value, f_np.selected_value, atol=1e-4)
        else:
            assert f_mlx.selected_value == f_np.selected_value
        np.testing.assert_allclose(f_mlx.logits, f_np.logits, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(f_mlx.confidence, f_np.confidence, rtol=1e-4, atol=1e-4)


def test_model_mlx_batch_consistency():
    schema = BenchmarkSchema()
    model_mlx = SystemOneModel(schema, backend="mlx")

    prompts = [
        "Read file /etc/hosts",
        "SELECT id, name FROM users WHERE active = 1",
        "curl -X POST https://api.stripe.com/v1/charges",
    ]

    batch_res = model_mlx.forward_batch(prompts)
    assert len(batch_res) == 3

    for p, b_item in zip(prompts, batch_res):
        single_res = model_mlx.forward_single(p)
        assert b_item.fields["route"].selected_value == single_res.fields["route"].selected_value
        np.testing.assert_allclose(
            b_item.fields["route"].logits,
            single_res.fields["route"].logits,
            rtol=1e-4,
            atol=1e-4,
        )


def test_model_numpy_batch_consistency():
    schema = BenchmarkSchema()
    model_np = SystemOneModel(schema, backend="numpy")

    prompts = [
        "Read file /etc/hosts",
        "SELECT id, name FROM users WHERE active = 1",
        "curl -X POST https://api.stripe.com/v1/charges",
    ]

    batch_res = model_np.forward_batch(prompts)
    assert len(batch_res) == 3

    for p, b_item in zip(prompts, batch_res):
        single_res = model_np.forward_single(p)
        assert b_item.fields["route"].selected_value == single_res.fields["route"].selected_value
        np.testing.assert_allclose(
            b_item.fields["route"].logits,
            single_res.fields["route"].logits,
            rtol=1e-5,
            atol=1e-5,
        )


def test_model_prompt_type_validation():
    schema = BenchmarkSchema()
    model = SystemOneModel(schema)

    with pytest.raises(TypeError, match="Prompt must be a string"):
        model.encode(None)  # type: ignore

    with pytest.raises(TypeError, match="Prompt must be a string"):
        model.forward_single(12345)  # type: ignore
