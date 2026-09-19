import pytest
import numpy as np

from system1.engine import ReflexEngine
from system1.schema import DecisionSchema, ChoiceField

def test_learning_marks_previous_calibration_stale():
    schema = DecisionSchema(
        schema_name="TestSchema",
        fields={"label": ChoiceField(options=["A", "B"])}
    )
    engine = ReflexEngine(
        schema=schema,
        model="dummy_model",
        strict_mode=True
    )
    # mock the model
    class DummyModel:
        def __init__(self):
            self.model_version = 0
            self._field_heads = {}
        def forward_single(self, *args, **kwargs):
            class RawEval:
                logits = np.array([0.1, 0.9])
            class RawRes:
                fields = {"label": RawEval()}
                embedding = np.zeros(384, dtype=np.float32)
            return RawRes()
        def learn_from_tier2(self, *args, **kwargs):
            return {"updated_fields": ["label"]}
            
    engine.model = DummyModel()
    
    # Fake calibration
    engine.calibrators["label"].is_calibrated = True
    engine.calibrators["label"].metrics = {"fake": "yes"}
    engine.conformal_predictors["label"].calibration_scores = np.array([0.9, 0.95])
    engine.conformal_predictors["label"].is_calibrated = True
    
    # 1. verify it works strict
    res1 = engine.decide("test1", strict=True)
    # mock conformal predict set directly to depend on is_calibrated
    def mock_predict_set(*args, **kwargs):
        is_cal = engine.conformal_predictors["label"].is_calibrated
        return type("CSet", (), {"prediction_set": ["B"] if is_cal else ["A", "B"], "margin": 1.0, "is_ambiguous": not is_cal})()
    engine.conformal_predictors["label"].predict_set = mock_predict_set
    res1 = engine.decide("test1", strict=True)
    assert not res1.is_ambiguous
    
    # 2. learn_from_tier2 with strict=False
    engine.learn_from_tier2("test2", {"label": "B"})
    
    # 3. verify strict=True now abstains (is_ambiguous)
    res2 = engine.decide("test3", strict=True)
    assert res2.is_ambiguous

def test_explicit_embedding_does_not_poison_implicit_embedding():
    schema = DecisionSchema(
        schema_name="TestSchema2",
        fields={"label": ChoiceField(options=["north", "south"])}
    )
    engine = ReflexEngine(
        schema=schema,
        model="dummy_model",
        strict_mode=True
    )
    class DummyModel2:
        def __init__(self):
            self.model_version = 0
            self._field_heads = {}
        def forward_single(self, prompt, embedding=None, **kwargs):
            class RawEval:
                if embedding is not None and np.sum(embedding) > 100:
                    logits = np.array([-1.0, 1.0])
                else:
                    logits = np.array([1.0, -1.0])
            class RawRes:
                fields = {"label": RawEval()}
                def __init__(self, emb):
                    self.embedding = emb if emb is not None else np.zeros(384, dtype=np.float32)
            return RawRes(embedding)
    engine.model = DummyModel2()
    engine.calibrators["label"].is_calibrated = True
    engine.conformal_predictors["label"].predict_set = lambda *args, **kwargs: type("CSet", (), {"prediction_set": ["south"] if np.argmax(args[0]) == 1 else ["north"], "margin": 1.0, "is_ambiguous": False})()

    res_explicit = engine.decide("query text", embedding=np.ones(384, dtype=np.float32))
    assert res_explicit.values["label"] == "south"

    res_implicit = engine.decide("query text")
    assert res_implicit.values["label"] == "north"
    
