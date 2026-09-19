#!/usr/bin/env python3
"""Teach one skill: choose an execution tier under an explicit routing policy.

Run: python examples/model_routing.py
This classifies requests; it does not call a local or cloud language model.
The labeled cases live in examples/teaching/model_routing.json.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from system1 import ChoiceField, DecisionSchema
from _teaching_demo import run_example


class ModelRouterSchema(DecisionSchema):
    target_tier = ChoiceField(
        options=['local_small', 'standard_chat', 'frontier_reasoning'],
        descriptions={
            'local_small': 'Mechanical formatting, extraction, sorting, and syntax conversion',
            'standard_chat': 'Ordinary drafting, summarization, translation, and explanation',
            'frontier_reasoning': 'Formal proofs, correctness analysis, difficult diagnosis, and constrained architecture design',
        },
    )


def main(argv=None):
    return run_example(ModelRouterSchema, 'model_routing', argv)


if __name__ == '__main__':
    main()
