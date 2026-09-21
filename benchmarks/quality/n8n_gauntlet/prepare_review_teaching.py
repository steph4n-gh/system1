"""Pin extra negative teaching rows and reserve unused human OOS examples.

No model/encoder imports. Test text is used only for normalized-group exclusion.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

from prepare import OUTPUT, ROOT, group

HERE = Path(__file__).resolve().parent
SOURCES = {
    "data_oos_plus.json": "bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f",
    "binary_wiki_aug.json": "0e059889d4771dd50463f4332374d8de8389c5828cb4d85b754b4ce8339b8544",
}


def prepare():
    manifest = json.loads((HERE / "manifest.json").read_text())
    known = set()
    for dataset in ("clinc150", "banking77"):
        for split, expected in manifest["datasets"][dataset]["splits"].items():
            raw = (OUTPUT / f"{dataset}-{split}.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                raise ValueError("Original split changed")
            known.update(r["group"] for r in json.loads(raw))
    data, sources = {}, {}
    for name, expected in SOURCES.items():
        url = "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/" + name
        path = OUTPUT / name
        raw = path.read_bytes() if path.exists() else urllib.request.urlopen(url, timeout=30).read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Additional source changed")
        path.write_bytes(raw)
        data[name] = json.loads(raw)
        sources[name] = dict(url=url, sha256=expected)
    human = {group(text): dict(prompt=text, label="oos", group=group(text), source="clinc-oos-plus-reserve")
             for text, label in data["data_oos_plus.json"]["oos_train"]
             if label == "oos" and group(text) not in known}
    known.update(human)
    wiki = {}
    for text, label in data["binary_wiki_aug.json"]["train"]:
        key = group(text)
        if label == "oos" and key not in known and key not in wiki:
            wiki[key] = dict(prompt=text, label="oos", group=key, source="clinc-wikipedia-negative")
    if len(human) != 151 or len(wiki) < 2000:
        raise ValueError("Unexpected additional-data inventory")
    rows = {"negative-fit": [wiki[key] for key in sorted(wiki)[:2000]],
            "human-oos-reserve": [human[key] for key in sorted(human)]}
    result = dict(scope="review teaching development, reserve never scored", sources=sources, files={})
    for name, items in rows.items():
        path = OUTPUT / f"review-{name}.json"
        raw = (json.dumps(items, indent=2, ensure_ascii=False) + "\n").encode()
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Refuse to overwrite changed review data")
        path.write_bytes(raw)
        result["files"][name] = dict(rows=len(items), sha256=hashlib.sha256(raw).hexdigest())
    return result


if __name__ == "__main__":
    result = prepare()
    (HERE / "review-data-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
