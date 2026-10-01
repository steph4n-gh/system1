"""Checks for the experiment's cache and annotation-isolation contracts."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run import CACHE,choose,engine,group,sha
from system1 import ChoiceField,DecisionSchema,SystemOneCompiler
from system1.core.text import TfidfProjector


def row(text,label,origin):
    return dict(input=text,id=sha(text.casefold()),label=label,origin=origin)


def test_grouping_keeps_untouched_role_and_drops_conflicts():
    rows=[row("same text", "yes", "train"),row("same text", "yes", "test"),
          row("conflicting text", "yes", "train"),row("conflicting text", "no", "test")]
    kept,audit=group(rows,{"train":0,"test":1})
    assert len(kept)==1 and kept[0]["origin"]=="test"
    assert audit["conflicting_components"]==1


class Features:
    def project_batch(self,texts):
        return np.eye(len(texts),dtype=np.float32)


class FakeEngine:
    model=SimpleNamespace(projector=Features())

    def decide(self,text,**kwargs):
        value=.5+int(text)/100
        return SimpleNamespace(probabilities={"label":{"a":value,"b":1-value}})


@pytest.mark.parametrize("strategy",["random","uncertainty","uncertainty_diversity"])
def test_acquisition_cannot_observe_pool_labels(strategy):
    inputs=[dict(input=str(i),label="secret annotation") for i in range(10)]
    permuted=[dict(input=r["input"],label=f"changed label {i}") for i,r in enumerate(inputs)]
    first=choose(FakeEngine(),inputs,5,strategy,11)
    assert first==choose(FakeEngine(),permuted,5,strategy,11)
    assert len(first)==len(set(first))==5


def test_standalone_scorer_disables_both_caches(tmp_path):
    class Fruit(DecisionSchema):
        label=ChoiceField(options=["apple","banana"])

    fitting=[(f"{label} fruit order {i}",label) for label in ["apple","banana"] for i in range(5)]
    calibration=[(f"{label} fruit delivery {i}",label) for label in ["apple","banana"] for i in range(4)]
    model=SystemOneCompiler(Fruit,projector=TfidfProjector.fit([text for text,_ in fitting])).compile(
        {"label":fitting},augment=False,calibration_exemplars={"label":calibration})
    path=tmp_path/"candidate.s1m"
    model.save(path)
    loaded=engine(path)
    assert loaded.use_cache is False and loaded.model.use_cache is False
    first=loaded.decide("apple fruit",record_receipt=False)
    loaded.decide("banana fruit",record_receipt=False)
    again=loaded.decide("apple fruit",record_receipt=False)
    assert first.values==again.values and first.probabilities==again.probabilities
