#!/usr/bin/env python3
"""Teach one local email skill from editable examples and a pinned rule contract."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

from system1 import ChoiceField, DecisionSchema, SystemOneCompiler
from system1.integrations.inbox_zero import CATEGORIES, email_text, instruction_digest

HERE = Path(__file__).resolve().parent


def load_lessons(path):
    data = json.loads(Path(path).read_text())
    seen, groups = set(), set()
    for split in ("teach", "calibration"):
        rows = data[split]
        if {row["label"] for row in rows} != set(CATEGORIES):
            raise ValueError(f"{split} must cover all seven categories")
        split_groups = set()
        for row in rows:
            text = " ".join(email_text(row["email"]).casefold().split())
            if text in seen or row["group"] in groups:
                raise ValueError("Examples and source groups must not cross teaching/calibration splits")
            seen.add(text)
            split_groups.add(row["group"])
        groups.update(split_groups)
    return data


def teach(lessons_path, contract_path, output_path):
    data = load_lessons(lessons_path)
    contract = json.loads(Path(contract_path).read_text())
    if tuple(contract["categories"]) != CATEGORIES or set(contract["criteria"]) != set(CATEGORIES) | {"None"}:
        raise ValueError("Expected the seven standard Inbox Zero categories plus None")
    schema = type("InboxZeroEmail", (DecisionSchema,), {
        "category": ChoiceField(options=list(CATEGORIES), descriptions={
            category: contract["criteria"][category] for category in CATEGORIES
        })
    })
    compiler = SystemOneCompiler(schema, dimension=2048, regularization=.1, backend="numpy")
    samples = {split: [(email_text(row["email"]), row["label"]) for row in data[split]]
               for split in ("teach", "calibration")}
    start = time.perf_counter()
    skill = compiler.compile({"category": samples["teach"]}, augment=False,
                             calibration_exemplars={"category": samples["calibration"]})
    teaching_ms = (time.perf_counter() - start) * 1000
    skill.metadata["inbox_zero"] = {
        "rule_digests": {key: instruction_digest(value) for key, value in contract["criteria"].items()},
        "question_digest": instruction_digest(contract["question"]), "alpha": .05,
        "lessons_sha256": hashlib.sha256(Path(lessons_path).read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(Path(contract_path).read_bytes()).hexdigest(),
        "upstream": contract["upstream"],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    skill.save(output_path)
    return skill, {"teaching_ms": teaching_ms, "skill_bytes": Path(output_path).stat().st_size,
                   "counts": {split: len(data[split]) for split in samples}, "teacher_calls": 0}


def no_network(*args, **kwargs):
    raise AssertionError("Teaching/evaluation must not contact a network or teacher")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons", type=Path, default=HERE / "lessons.json")
    parser.add_argument("--contract", type=Path, default=Path(".system1/inbox-zero/sources/contract.json"))
    parser.add_argument("--output", type=Path, default=Path(".system1/inbox-zero/email.s1m"))
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "create_connection", no_network):
        _, report = teach(args.lessons, args.contract, args.output)
    print(json.dumps({**report, "saved": str(args.output),
                      "next": "Evaluate before enabling actions; teaching is not qualification."}, indent=2))


if __name__ == "__main__":
    main()
