"""Check expert-selected source provenance and overlap without loading a model."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from prepare import OUTPUT, ROOT, SOURCES, group

HERE = Path(__file__).resolve().parent
REVISION = "667f7ad4a0e24cb5895e3e76d03a8961813899bd"
HASHES = {
    3: "a204f6c17a581379b604e0a45f98f47d8f40b18176edb00f1833454d7546dcc3",
    5: "556949ca7794991990a783141d25e3da51c2ff2bf731a00cc4bb78bba3f30a8a",
}


def checked_csv(path, expected, url=None):
    if not path.exists() and url is not None:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Downloaded source checksum mismatch")
        path.write_bytes(raw)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch: {path.name}")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def main():
    original = {}
    labels = defaultdict(set)
    for split in ("train", "test"):
        name = f"banking-{split}.csv"
        rows = checked_csv(ROOT / ".system1/workload-sources" / name, SOURCES[name][1])
        original[split] = {group(row["text"]) for row in rows}
        for row in rows:
            labels[group(row["text"])].add(row["category"])
    known = set().union(*original.values())
    report = dict(
        scope="Source inventory only; no encoding, scoring, teaching or label changes",
        repository_revision=REVISION,
        paper="https://arxiv.org/pdf/2311.06102", paper_section="4.4",
        provenance="The paper describes an expert selecting three of ten existing examples per class. It does not describe newly authored requests or relabeling. The five-example file is not explained by that procedure.",
        files=[], qualifies_as_full_confirmation=False)
    source_groups = {}
    for count, expected in HASHES.items():
        name = f"top_{count}_representative_samples_per_class_in_banking77.csv"
        url = f"https://huggingface.co/datasets/helvia/banking77-representative-samples/resolve/{REVISION}/{name}"
        rows = checked_csv(OUTPUT / f"banking-expert-top{count}-source.csv", expected, url)
        groups = {group(row["text"]) for row in rows}
        source_groups[count] = groups
        report["files"].append(dict(
            url=url, sha256=expected, rows=len(rows), unique_groups=len(groups),
            supported_labels=len({row["label_name"] for row in rows}),
            rows_per_label=dict(sorted(Counter(row["label_name"] for row in rows).items())),
            original_overlap={split: len(groups & values) for split, values in original.items()},
            new_groups=len(groups - known),
            original_label_disagreements=[dict(group=group(row["text"]),
                source_label=row["label_name"], original_labels=sorted(labels[group(row["text"])]))
                for row in rows if group(row["text"]) in labels and row["label_name"] not in labels[group(row["text"])]]))
    report["groups_shared_between_files"] = len(source_groups[3] & source_groups[5])
    report["combined_unique_groups"] = len(source_groups[3] | source_groups[5])
    report["combined_new_groups"] = len((source_groups[3] | source_groups[5]) - known)
    report["reason"] = "Existing BANKING77 subsets cannot supply independent confirmation. No CLINC or unfamiliar cohort is supplied. These files have not been adopted for teaching."
    (HERE / "results/banking-expert-source-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "files": [{k: v for k, v in item.items() if k != "rows_per_label"} for item in report["files"]]}, indent=2))


if __name__ == "__main__":
    main()
