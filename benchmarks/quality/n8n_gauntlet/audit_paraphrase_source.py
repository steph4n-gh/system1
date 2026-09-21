"""Inventory a human paraphrase source without loading or evaluating a model."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from prepare import OUTPUT, group

HERE = Path(__file__).resolve().parent
REVISION = "8c62edda5206965a45b77d561eb8a9868552539d"
URL = (f"https://raw.githubusercontent.com/kinit-sk/Crowd-vs-GPT-intent-class/{REVISION}/"
       "ood_robustness_experiments/challenge_data/clinc150/full_larson_human.csv")
SHA256 = "06966b017684e7dce0fd3a360025f7c05526a15831a87c3ce92fccca09402cac"
MAP_URL = URL.replace("full_larson_human.csv", "full_larson_orig_train.csv")
MAP_SHA256 = "be35d2d95f9ccb70691cf47ebf3b5bceb3ab0a24e443e50b68017064bd682981"


def source(path, url, expected):
    if path.exists():
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Paraphrase source changed")
    path.write_bytes(raw)
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    known = set()
    clinc_labels = defaultdict(set)
    for dataset in ("clinc150", "banking77"):
        for split, expected in manifest["datasets"][dataset]["splits"].items():
            raw = (OUTPUT / f"{dataset}-{split}.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                raise ValueError("Original split changed")
            known.update(row["group"] for row in json.loads(raw))
            if dataset == "clinc150":
                for row in json.loads(raw):
                    clinc_labels[row["group"]].add(row["label"])
    rows = source(OUTPUT / "clinc-human-paraphrases-reserve.csv", URL, SHA256)
    mapping_rows = source(OUTPUT / "clinc-paraphrase-label-map-source.csv", MAP_URL, MAP_SHA256)
    matches = defaultdict(set)
    for row in mapping_rows:
        names = clinc_labels[group(row["text"])]
        if len(names) == 1:
            matches[row["label"]].update(names)
    raw_source_labels = sorted({row["label"] for row in rows})
    mapping = {key: next(iter(values)) for key, values in matches.items() if len(values) == 1}
    unresolved = {key: sorted(matches[key]) for key in raw_source_labels if key not in mapping}
    # Keep disagreements unresolved; an inventory must not silently relabel data.
    rows = [dict(row, label=mapping.get(row["label"], "unresolved_source_label_" + row["label"])) for row in rows]
    labels = set(json.loads((HERE / "artifacts/clinc-review-consistent/manifest.json").read_text())["categories"]) - {"oos"}
    groups = defaultdict(set)
    for row in rows:
        groups[group(row["text"])].add(row["label"])
    unseen = {key: values for key, values in groups.items() if key not in known}
    report = dict(scope="source inventory only; raw source reserved, never encoded/scored or used for teaching",
        source=dict(url=URL, revision=REVISION, sha256=SHA256,
            paper="https://aclanthology.org/2023.emnlp-main.117/", section="Appendices C and C.1"),
        label_mapping=dict(url=MAP_URL, sha256=MAP_SHA256, numeric_to_intent=mapping,
            unresolved=unresolved,
            method="Exact normalized-group matches to original CLINC labels; disagreements retained, no predictions"),
        raw_source_labels=raw_source_labels,
        rows=len(rows), unique_normalized_groups=len(groups),
        groups_overlapping_original_folds=len(set(groups) & known), new_groups=len(unseen),
        conflicting_new_groups=sum(len(values) != 1 for values in unseen.values()),
        source_labels=sorted({row["label"] for row in rows}),
        new_group_counts_by_label=dict(sorted(Counter(label for values in unseen.values() for label in values).items())),
        missing_supported_labels=sorted(labels - {row["label"] for row in rows}),
        unknown_source_labels=sorted({row["label"] for row in rows} - labels),
        qualifies_as_full_confirmation=False,
        reason="Only a subset of CLINC intents, with an unresolved label mapping; no full BANKING77 or unfamiliar cohort. Human paraphrases derive from original seeds, not independently collected user contexts.")
    (HERE / "results/human-paraphrase-source-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in
                     ("source_labels", "missing_supported_labels", "new_group_counts_by_label", "source", "label_mapping", "raw_source_labels")}))


if __name__ == "__main__":
    main()
