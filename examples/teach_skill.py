"""Teach one support-routing skill, save it, and use it locally.

Run: python examples/teach_skill.py
Start from eight editable message-and-answer examples. See docs/guides/first_skill.md.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler


class SupportRoute(DecisionSchema):
    team = ChoiceField(options=["billing", "support"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", type=Path, default=Path(__file__).parent / "teaching/first_skill.json",
                        help="JSON file pairing each message with the team that should receive it")
    parser.add_argument("--prompt", default="Please refund this payment", help="A new message to try")
    parser.add_argument("--output", type=Path, default=Path(".system1/support-route.s1m"),
                        help="Where to save the taught skill")
    args = parser.parse_args(argv)
    try:
        examples = json.loads(args.lessons.read_text())
        skill = SystemOneCompiler(SupportRoute).compile(examples, augment=False)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"Cannot teach from {args.lessons}: {exc}")
    path = args.output
    skill.save(path)

    restored = CompiledSystemOneModel.load(path)
    engine = System1Engine(restored.schema, model=restored, strict_mode=True)
    decision = engine.decide(args.prompt, record_receipt=False)

    print(f"Lessons: {args.lessons}")
    print(f"Message: {args.prompt}")
    print(f"Saved skill: {path} ({path.stat().st_size:,} bytes)")
    print(f"Suggested team: {decision.values['team']}")
    print(f"Needs review: {decision.is_ambiguous}")
    print(f"Possible teams: {decision.conformal_sets['team']}")
    if decision.is_ambiguous:
        print("Ask a person to choose the team. This skill has not made an unambiguous choice for this message.")
    print("To teach a correction, edit the message's answer in your lessons file, then rerun.")
    print("Check different messages before relying on the revised skill.")
    # The application decides what review means: ask a person or call System 2.
    # This tiny teaching set is deliberately insufficient to establish certainty.


if __name__ == "__main__":
    main()
