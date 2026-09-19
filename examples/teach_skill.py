"""Teach one support-routing skill, save it, and use it locally.

Run: python examples/teach_skill.py
These eight examples demonstrate the API, not production readiness.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler


class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])


EXAMPLES = {"team": [
    ("Refund my payment", "billing"),
    ("Correct the invoice", "billing"),
    ("Subscription charged twice", "billing"),
    ("Payment receipt requested", "billing"),
    ("Debug this exception", "support"),
    ("Fix a software crash", "support"),
    ("API request fails", "support"),
    ("Investigate error logs", "support"),
]}


def main():
    skill = SystemOneCompiler(SupportRoute).compile(EXAMPLES, augment=False)
    path = Path(".system1/support-route.s1m")
    skill.save(path)

    restored = CompiledSystemOneModel.load(path)
    engine = System1Engine(restored.schema, model=restored, strict_mode=True)
    decision = engine.decide("Please refund this payment")

    print(f"Saved skill: {path} ({path.stat().st_size:,} bytes)")
    print(f"Suggested team: {decision.values['team']}")
    print(f"Needs review: {decision.is_ambiguous}")
    print(f"Possible teams: {decision.conformal_sets['team']}")
    # The application decides what review means: ask a person or call System 2.
    # This tiny teaching set is deliberately insufficient to establish certainty.


if __name__ == "__main__":
    main()
