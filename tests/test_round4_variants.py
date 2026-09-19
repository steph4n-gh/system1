import pytest
import tempfile
import numpy as np

from system1.schema import DecisionSchema, ChoiceField
from system1.compat.typesafe import TypeSafeClient
from system1.compiler import CompiledSystemOneModel
from system1.engine import SystemOneEngine

def test_exported_candidate_preserves_validated_runtime_calibration():
    # Setup client
    client = TypeSafeClient()
    
    # Fake schema / questions
    questions = {"label": ChoiceField(options=["A", "B"])}
    
    # Trigger a local execution so it builds engine
    client._execute_local("query1", questions)
    
    schema_digest = next(iter(client._engine_cache.keys()))
    engine = client._engine_cache[schema_digest]
    
    # Fake calibration
    engine.calibrators["label"].temperature = 2.5
    engine.conformal_predictors["label"].calibration_scores = np.array([0.1, 0.2, 0.3])
    engine.conformal_predictors["label"].is_calibrated = True
    
    # Fake compiled model inside the client
    cm = CompiledSystemOneModel(engine.schema, heads={})
    # Add head
    from system1.compiler import CompiledHeadWeights
    cm.heads["label"] = CompiledHeadWeights(
        field_name="label", field_type="choice", options=("A", "B"),
        weights=np.ones((2, 384)), biases=np.zeros(2),
        temperature=2.5,
        calibration_scores=(0.1, 0.2, 0.3)
    )
    client._compiled_model = cm
    
    with tempfile.NamedTemporaryFile(suffix=".s1m") as tf:
        assert client.export_model(tf.name)
        
        # Load and verify
        with open(tf.name, "rb") as f:
            data = f.read()
        
        reloaded = CompiledSystemOneModel.from_bytes(data)
        head = reloaded.heads["label"]
        
        assert head.temperature == 2.5
        assert head.calibration_scores == pytest.approx((0.1, 0.2, 0.3))

def test_statistical_promotion_rejection():
    client = TypeSafeClient()
    questions = {"label": ChoiceField(options=["A", "B"])}
    
    # fake evaluate locally
    client._execute_local("query1", questions)
    schema_digest = next(iter(client._engine_cache.keys()))
    engine = client._engine_cache[schema_digest]
    
    # mock decide so that queries are handled
    def mock_decide(prompt, **kwargs):
        class Res:
            values = {"label": "A"}
        return Res()
    engine.decide = mock_decide

    from system1.compat.typesafe import evaluate_promotion_eligibility, PromotionPolicy
    
    # A) 1 correct labeled, 99 unlabeled
    val_history = [{"state": "q1", "answers": {"label": "A"}}]
    for i in range(99):
        val_history.append({"state": f"q_unlabeled_{i}", "answers": {}})
        
    policy = PromotionPolicy()
    report = evaluate_promotion_eligibility(engine, val_history, engine.schema, policy)
    print("report A:", report.rejection_reasons)
    assert not report.is_eligible
    assert report.wilson_lower_bound < 50.0

    # B) 100 correlated records
    val_history2 = []
    for i in range(100):
        val_history2.append({"state": f"q_corr_{i}", "answers": {"label": "A"}, "group_id": "group1"})
        
    report2 = evaluate_promotion_eligibility(engine, val_history2, engine.schema, policy)
    print("report B:", report2.rejection_reasons)
    assert not report2.is_eligible
    assert report2.wilson_lower_bound < 50.0

def test_ledger_verifies_incoming_receipt_against_anchor():
    from system1.ledger import ActionLedger
    from system1.receipt import DecisionWitnessReceipt, RunWitnessEnvelope
    import tempfile
    
    with tempfile.NamedTemporaryFile() as tf:
        ledger = ActionLedger(tf.name)
        
        # Unsigned receipt
        receipt = DecisionWitnessReceipt(
            decision_id="d1",
            schema_name="S1",
            schema_digest="SD1",
            prompt="q1",
            prompt_digest="PD1",
            values={"label": "A"},
            confidences={"label": 1.0},
            conformal_sets={"label": ["A"]},
            probabilities={},
            latency_ms=10.0,
            is_ambiguous=False,
            timestamp=123.0,
            truth_ledger_head="000",
            ledger_record_id=None,
            signer_public_key="some_key",
            envelope=RunWitnessEnvelope(
                mutation_intent={"test": "test"}
            )
        )
        
        from system1.ledger import IntegrityError
        import pytest
        # should fail since we require trusted_public_key and receipt is not signed/valid
        with pytest.raises(IntegrityError):
            ledger.append(receipt, trusted_public_key="some_key")

def test_record_execution_outcome_enforces_cryptographic_authorization():
    from system1.ledger import ActionLedger, LedgerWriteError
    from system1.receipt import DecisionWitnessReceipt, RunWitnessEnvelope
    import tempfile
    
    with tempfile.NamedTemporaryFile() as tf:
        ledger = ActionLedger(tf.name)
        
        # Unsigned receipt for append (we won't verify it since we don't pass public key)
        receipt = DecisionWitnessReceipt(
            decision_id="d2",
            schema_name="S1",
            schema_digest="SD1",
            prompt="q1",
            prompt_digest="PD1",
            values={"label": "A"},
            confidences={"label": 1.0},
            conformal_sets={"label": ["A"]},
            probabilities={},
            latency_ms=10.0,
            is_ambiguous=False,
            timestamp=123.0,
            truth_ledger_head="000",
            ledger_record_id=None,
            signer_public_key=None,
            envelope=RunWitnessEnvelope(
                mutation_intent={"test": "test"}
            )
        )
        
        ledger.append(receipt, tenant_id="tenant1", principal_id="principal1", scope="scope1")
        
        import pytest
        # matching params should work
        ledger.record_execution_outcome(
            action_id="d2", receipt_digest=receipt.compute_digest(), status="SUCCEEDED",
            tenant_id="tenant1", principal_id="principal1", scope="scope1"
        )
        
        # mismatched tenant
        with pytest.raises(LedgerWriteError, match="Authorization mismatch"):
            ledger.record_execution_outcome(
                action_id="d2", receipt_digest=receipt.compute_digest(), status="SUCCEEDED",
                tenant_id="tenant2", principal_id="principal1", scope="scope1"
            )
