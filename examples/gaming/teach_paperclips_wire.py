"""Teach one Paperclips skill: restock wire or keep the cash.

This is a numeric teaching demonstration, not a complete game-playing agent.
No browser, API key, game simulation, or extra dependency is required.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from system1 import ChoiceField, DecisionSchema, System1Engine
from system1.compiler import CompiledSystemOneModel, SystemOneCompiler


QUESTION = "Should we restock wire?"


class WireDecision(DecisionSchema):
    action = ChoiceField(options=["buy_wire", "hold"])


def telemetry(wire):
    if not isinstance(wire, (int, float)) or not 0 <= wire < float("inf"):
        raise ValueError("Wire must be a finite nonnegative number")
    # Scale inches into spool-sized units. The constant reference preserves
    # magnitude when the existing telemetry projector normalizes the vector.
    return {"wire_spools": wire / 1000, "unit": 1.0}


def suggest(engine, *, wire, funds, wire_cost):
    if any(not isinstance(x, (int, float)) or not 0 <= x < float("inf")
           for x in (funds, wire_cost)):
        raise ValueError("Funds and wire cost must be finite nonnegative numbers")
    decision = engine.decide(QUESTION, telemetry=telemetry(wire),
                             alpha=.05, record_receipt=False)
    prediction = decision.values["action"]
    # A classifier does not make an unavailable purchase legal. Check the
    # current browser button too if an application executes this suggestion.
    action = "review" if decision.is_ambiguous else prediction
    if action == "buy_wire" and funds < wire_cost:
        action = "hold"
    return {"prediction": prediction, "action": action,
            "needs_review": decision.is_ambiguous,
            "prediction_set": list(decision.conformal_sets["action"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", type=Path,
                        default=Path(__file__).parents[1] / "teaching/paperclips_wire.json")
    parser.add_argument("--output-dir", type=Path, default=Path(".system1/paperclips"))
    parser.add_argument("--wire", type=float, help="Try a current game's wire supply")
    parser.add_argument("--funds", type=float, default=0)
    parser.add_argument("--wire-cost", type=float, default=20)
    args = parser.parse_args(argv)
    data = json.loads(args.lessons.read_text())
    seen = set()
    for split in ("teach", "calibration", "evaluate"):
        states = [row["wire"] for row in data[split]]
        if len(set(states)) != len(states) or seen.intersection(states):
            raise ValueError("The three splits must contain distinct wire states")
        seen.update(states)
    def examples(split):
        return {"action": [(QUESTION, row["label"], telemetry(row["wire"]))
                           for row in data[split]]}
    start = time.perf_counter()
    skill = SystemOneCompiler(WireDecision, dimension=64, regularization=.1).compile(
        examples("teach"), augment=False, calibration_exemplars=examples("calibration"))
    teaching_ms = (time.perf_counter() - start) * 1000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "wire.s1m"
    skill.save(path)
    restored = CompiledSystemOneModel.load(path)
    restored.use_cache = False
    engine = System1Engine(restored.schema, model=restored, strict_mode=True, use_cache=False)
    rows = []
    for row in data["evaluate"]:
        start = time.perf_counter()
        result = suggest(engine, wire=row["wire"], funds=100, wire_cost=20)
        rows.append({**row, **result, "latency_ms": (time.perf_counter() - start) * 1000})
    accepted = [row for row in rows if not row["needs_review"]]
    report = {"provenance": data["provenance"], "policy": data["policy"],
              "dataset_sha256": hashlib.sha256(args.lessons.read_bytes()).hexdigest(),
              "counts": {key: len(data[key]) for key in ("teach", "calibration", "evaluate")},
              "teaching_ms": teaching_ms, "skill_bytes": path.stat().st_size,
              "correct": sum(row["prediction"] == row["label"] for row in rows),
              "accepted": len(accepted),
              "accepted_correct": sum(row["prediction"] == row["label"] for row in accepted),
              "decision_median_ms": statistics.median(row["latency_ms"] for row in rows),
              "results": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Saved and reloaded: {path}")
    if args.wire is not None:
        print("Current-state suggestion:", json.dumps(suggest(
            engine, wire=args.wire, funds=args.funds, wire_cost=args.wire_cost)))
    return report


if __name__ == "__main__":
    main()
