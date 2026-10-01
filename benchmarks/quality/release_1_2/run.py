#!/usr/bin/env python3
"""Frozen, local research comparison. Raw data and fitted weights stay ignored."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
from urllib.request import urlopen
import zipfile

import numpy as np
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler, TeachingSession
from system1.compiler import CompiledSystemOneModel
from system1.core.text import TfidfProjector
from system1.engine import SystemOneEngine

HERE = Path(__file__).resolve().parent
CACHE = ROOT / ".system1/release-1.2-research"
REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SEEDS = [11, 29, 47]
STRATEGIES = ["random", "uncertainty", "uncertainty_diversity"]


class Emotion(DecisionSchema):
    label = ChoiceField(options=LABELS)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_text())


def normalize(text):
    return " ".join(text.casefold().split())


def group(rows, priority):
    """One representative per connected exact/near duplicate component."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(i)] = find(j)

    exact, inverted, sizes = {}, defaultdict(list), []
    edges = 0
    for i, row in enumerate(rows):
        if row["id"] in exact:
            union(i, exact[row["id"]])
        exact[row["id"]] = i
        words = re.findall(r"\w+", normalize(row["input"]))
        shingles = set(zip(words, words[1:], words[2:]))
        sizes.append(len(shingles))
        shared = Counter(j for s in shingles for j in inverted[s])
        for j, overlap in shared.items():
            if overlap >= 3 and overlap / (len(shingles) + sizes[j] - overlap) >= .8:
                union(i, j)
                edges += 1
        for shingle in shingles:
            inverted[shingle].append(i)
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    kept, conflicts, cross = [], 0, 0
    for component in components.values():
        if len({r["label"] for r in component}) > 1:
            conflicts += 1
            continue
        cross += len({r["origin"] for r in component}) > 1
        # Higher-priority untouched roles retain their representative; related
        # rows in lower-priority roles are dropped rather than moved into fit.
        origin = max((r["origin"] for r in component), key=priority.__getitem__)
        eligible = [r for r in component if r["origin"] == origin]
        kept.append(min(eligible, key=lambda r: r["id"]))
    kept.sort(key=lambda r: r["id"])
    return kept, dict(source_rows=len(rows), retained=len(kept),
                     exact_duplicates=len(rows) - len(exact), near_edges=edges,
                     conflicting_components=conflicts, cross_source_components=cross)


def prepare():
    """Fetch pinned bytes and audit grouping; never predict evaluation rows."""
    import pyarrow.parquet as pq
    CACHE.mkdir(parents=True, exist_ok=True)
    sources, rows = [], []
    for split in ("train", "validation", "test"):
        url = (f"https://huggingface.co/datasets/dair-ai/emotion/resolve/{REVISION}/"
               f"split/{split}-00000-of-00001.parquet")
        target = CACHE / f"emotion-{split}.parquet"
        if not target.exists():
            target.write_bytes(urlopen(url).read())
        raw = target.read_bytes()
        sources.append(dict(url=url, sha256=sha(raw), bytes=len(raw)))
        for row in pq.read_table(target).to_pylist():
            text = " ".join(row["text"].split())
            rows.append(dict(id=sha(normalize(text)), input=text,
                             label=LABELS[row["label"]], origin=split))
    kept, audit = group(rows, {"train": 0, "validation": 1, "test": 2})
    splits = {split: [r for r in kept if r["origin"] == split]
              for split in ("train", "validation", "test")}
    write(CACHE / "emotion.json", splits)
    roles = {name: [r["id"] for r in part] for name, part in splits.items()}
    manifest = dict(source="dair-ai/emotion", revision=REVISION,
                    license="educational and research purposes only",
                    sources=sources, grouping=audit, roles=roles,
                    roles_sha256=sha(json.dumps(roles, sort_keys=True)),
                    counts={name:len(part) for name, part in splits.items()})
    write(HERE / "source-manifest.json", manifest)
    print(json.dumps({k:manifest[k] for k in ("source", "grouping", "counts")}))

    url = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
    target = CACHE / "uci-sentiment.zip"
    if not target.exists():
        target.write_bytes(urlopen(url).read())
    rows = []
    with zipfile.ZipFile(target) as archive:
        for origin, name in [("amazon", "amazon_cells_labelled.txt"),
                             ("imdb", "imdb_labelled.txt"), ("yelp", "yelp_labelled.txt")]:
            raw = archive.read(f"sentiment labelled sentences/{name}").decode()
            for line in raw.splitlines():
                text, label = line.rsplit("\t", 1)
                rows.append(dict(id=sha(normalize(text)), input=text,
                                 label="positive" if label == "1" else "negative", origin=origin))
    kept, audit = group(rows, {"amazon":0,"imdb":0,"yelp":1})
    ordered = sorted((r for r in kept if r["origin"] != "yelp"),
                     key=lambda r:sha(f"20261001:{r['id']}"))
    splits = dict(fit=ordered[:1000], calibrate=ordered[1000:1200],
                  develop=ordered[1200:1400], corrections=ordered[1400:1800],
                  unused=ordered[1800:], qualify=[r for r in kept if r["origin"] == "yelp"])
    write(CACHE / "sentiment.json", splits)
    manifest = dict(source="UCI Sentiment Labelled Sentences", doi="10.24432/C57604",
                    license="CC BY 4.0", url=url, sha256=sha(target.read_bytes()),
                    bytes=target.stat().st_size, grouping=audit,
                    roles={name:[r["id"] for r in part] for name,part in splits.items()},
                    counts={name:len(part) for name,part in splits.items()})
    write(HERE / "lifecycle-source-manifest.json", manifest)
    print(json.dumps({k:manifest[k] for k in ("source", "grouping", "counts")}))


def role_rows(seed):
    splits = read(CACHE / "emotion.json")
    ordered = sorted(splits["train"], key=lambda r:sha(f"{seed}:{r['id']}"))
    return ordered[:40], ordered[40:100], ordered[100:], splits


def engine(path, projector=None):
    model = CompiledSystemOneModel.load(path, projector=projector)
    return SystemOneEngine(model.schema, model=model, strict_mode=True, use_cache=False)


def fit_ridge(fitting, calibration, development, folder):
    start = time.perf_counter()
    session = TeachingSession(folder, Emotion, regularization=.1, max_features=1024)
    rows = [{"input":r["input"],"label":r["label"],"split":role,"source":"benchmark annotation"}
            for role,part in [("teach",fitting),("calibrate",calibration),("evaluate",development)] for r in part]
    session.record_many(rows)
    # Deliberately never adopt: exploratory metrics do not grant approval.
    report = session.assess(min_accuracy=0,min_coverage=0,max_accepted_errors=len(development),max_regressions=len(development))
    elapsed = time.perf_counter()-start
    return engine(session.candidate_path), elapsed, session.candidate_path


def fit_logistic(fitting, calibration, folder):
    start = time.perf_counter()
    projector = TfidfProjector.fit([r["input"] for r in fitting],max_features=1024)
    model = SystemOneCompiler(Emotion,projector=projector,regularization=.1,
                              choice_solver="logistic").compile(
        {"label":[(r["input"],r["label"]) for r in fitting]},augment=False,
        calibration_exemplars={"label":[(r["input"],r["label"]) for r in calibration]})
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"candidate.s1m"
    model.save(path)
    return engine(path),time.perf_counter()-start,path


def choose(current, pool, count, strategy, seed):
    rng=np.random.default_rng(seed)
    if strategy == "random":
        return rng.permutation(len(pool))[:count].tolist()
    # The oracle's labels are removed at this boundary. No fit/calibration labels
    # or pool labels can influence ranking except through the assessed engine.
    texts=[r["input"] for r in pool]
    probabilities=np.asarray([list(current.decide(t,record_receipt=False).probabilities["label"].values())
                              for t in texts])
    ordered=np.argsort(probabilities,axis=1)
    margin=np.take_along_axis(probabilities,ordered[:,-1:],axis=1).ravel()-np.take_along_axis(probabilities,ordered[:,-2:-1],axis=1).ravel()
    ranking=np.lexsort((np.arange(len(pool)),margin)).tolist()
    if strategy == "uncertainty":
        return ranking[:count]
    audit=rng.permutation(len(pool))[:int(np.ceil(count*.2))].tolist()
    remaining=count-len(audit)
    shortlist=[i for i in ranking if i not in set(audit)][:5*remaining]
    features=current.model.projector.project_batch([texts[i] for i in shortlist])
    selected=[0]
    distance=1-features@features[0]
    distance[0]=-np.inf
    while len(selected)<remaining:
        next_index=int(np.argmax(distance))
        selected.append(next_index)
        distance=np.minimum(distance,1-features@features[next_index])
        distance[selected]=-np.inf
    return [shortlist[i] for i in selected]+audit


def outcomes(current, rows):
    cases,times=[],[]
    for row in rows:
        start=time.perf_counter_ns()
        result=current.decide(row["input"],alpha=.05,record_receipt=False)
        times.append((time.perf_counter_ns()-start)/1e6)
        cases.append(dict(id=row["id"],label=row["label"],predicted=result.values["label"],
                          review=bool(result.is_ambiguous),correct=result.values["label"]==row["label"]))
    return summarize(cases,times)


def summarize(cases,times):
    n=len(cases)
    accepted=sum(not r["review"] for r in cases)
    correct=sum(r["correct"] for r in cases)
    accepted_correct=sum(r["correct"] and not r["review"] for r in cases)
    error=accepted-accepted_correct
    bound=float(beta.ppf(.95,error+1,accepted-error)) if accepted and error<accepted else 1.
    metrics=dict(count=n,raw_accuracy=correct/n,accepted=accepted,accepted_correct=accepted_correct,
                 accepted_errors=error,accepted_accuracy=accepted_correct/accepted if accepted else None,
                 accepted_correct_yield=accepted_correct/n,coverage=accepted/n,review_rate=(n-accepted)/n,
                 accepted_error_upper_95=bound,screen_pass=bool(accepted and bound<=.1 and accepted_correct/accepted>=.9 and accepted/n>=.2),
                 median_latency_ms=float(np.median(times)),p95_latency_ms=float(np.percentile(times,95)))
    metrics["by_class"]={label:dict(count=sum(r["label"]==label for r in cases),
                                     accepted=sum(r["label"]==label and not r["review"] for r in cases),
                                     accepted_errors=sum(r["label"]==label and not r["review"] and not r["correct"] for r in cases))
                         for label in sorted({r["label"] for r in cases})}
    return metrics,[r for r in cases if r["review"] or not r["correct"]]


def conventional(fitting,rows):
    start=time.perf_counter()
    vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=1024)
    features=vectorizer.fit_transform([r["input"] for r in fitting])
    model=LogisticRegression(C=1,max_iter=1000).fit(features,[r["label"] for r in fitting])
    elapsed=time.perf_counter()-start
    cases,times=[],[]
    for row in rows:
        start=time.perf_counter_ns()
        probs=model.predict_proba(vectorizer.transform([row["input"]]))[0]
        predicted=str(model.classes_[int(probs.argmax())])
        times.append((time.perf_counter_ns()-start)/1e6)
        cases.append(dict(id=row["id"],label=row["label"],predicted=predicted,
                          review=bool(probs.max()<.9),correct=predicted==row["label"]))
    metrics,cases=summarize(cases,times)
    return metrics,cases,elapsed


def develop():
    results, frozen=[],[]
    for seed in SEEDS:
        calibration,initial,pool,splits=role_rows(seed)
        for strategy in STRATEGIES:
            fitting=list(initial)
            remaining=list(pool)
            for budget in (100,200,400):
                name=f"ridge-{strategy}-{seed}-{budget}"
                folder=CACHE/name
                selection_time=0.
                if budget>100:
                    start=time.perf_counter()
                    indices=choose(current,remaining,budget-previous,strategy,seed+budget)
                    selection_time=time.perf_counter()-start
                    selected=set(indices)
                    fitting += [remaining[i] for i in indices]
                    remaining=[r for i,r in enumerate(remaining) if i not in selected]
                missing=set(LABELS)-{r["label"] for r in fitting}
                if missing:
                    results.append(dict(name=name,seed=seed,budget=budget,strategy=strategy,
                                        status="infeasible",missing_classes=sorted(missing)))
                    # Acquisition remains uniform until a candidate can be fit;
                    # uncertainty has no calibrated engine for ranking.
                    raise RuntimeError(f"Initial labels miss classes: {name}: {missing}")
                current,fit_time,path=fit_ridge(fitting,calibration,splits["validation"],folder)
                metrics,cases=outcomes(current,splits["validation"])
                write(folder/"development-cases.json",cases)
                result=dict(name=name,seed=seed,budget=budget,strategy=strategy,kind="system1_ridge",
                            status="measured",fit_labels=len(fitting),calibration_labels=len(calibration),
                            inspected_total=len(fitting)+len(calibration),evaluation_labels=len(splits["validation"]),
                            fit_seconds=fit_time,selection_seconds=selection_time,model_bytes=path.stat().st_size,
                            model_sha256=sha(path.read_bytes()),selection_sha256=sha(json.dumps([r["id"] for r in fitting])),
                            metrics=metrics,cases_sha256=sha((folder/"development-cases.json").read_bytes()))
                results.append(result)
                write(folder/"selection.json",[r["id"] for r in fitting])
                frozen.append(dict(name=name,path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes()),
                                   selection_sha256=result["selection_sha256"]))
                print(json.dumps(dict(name=name,metrics=metrics)),flush=True)
                if strategy=="random":
                    logistic,logistic_time,logistic_path=fit_logistic(fitting,calibration,CACHE/name.replace("ridge-","logistic-"))
                    log_metrics,log_cases=outcomes(logistic,splits["validation"])
                    log_name=name.replace("ridge-","logistic-")
                    write(CACHE/log_name/"development-cases.json",log_cases)
                    results.append(dict(name=log_name,seed=seed,budget=budget,strategy="random",kind="system1_logistic",
                                        inspected_total=budget,fit_labels=len(fitting),calibration_labels=40,
                                        fit_seconds=logistic_time,model_bytes=logistic_path.stat().st_size,
                                        metrics=log_metrics,model_sha256=sha(logistic_path.read_bytes())))
                    frozen.append(dict(name=log_name,path=str(logistic_path.relative_to(ROOT)),sha256=sha(logistic_path.read_bytes())))
                    conv_metrics,conv_cases,conv_time=conventional(fitting,splits["validation"])
                    conv_name=name.replace("ridge-","sklearn-")
                    write(CACHE/conv_name/"development-cases.json",conv_cases)
                    results.append(dict(name=conv_name,seed=seed,budget=budget,kind="sklearn_logistic",strategy="random",
                                        inspected_total=budget,fit_labels=len(fitting),calibration_labels=40,
                                        calibration_labels_unused=40,fit_seconds=conv_time,metrics=conv_metrics))
                previous=budget
    report=dict(protocol_sha256=sha((HERE/"PROTOCOL.md").read_bytes()),
                manifest_sha256=sha((HERE/"source-manifest.json").read_bytes()),
                python=platform.python_version(),numpy=np.__version__,results=results)
    write(HERE/"development.json",report)
    # Only frozen candidates selected by a fixed rule can proceed. Always retain
    # random reference; no candidate configurations change after development.
    scores={strategy:np.mean([r["metrics"]["accepted_correct_yield"] for r in results
                             if r.get("kind")=="system1_ridge" and r["budget"]==400 and r["strategy"]==strategy])
            for strategy in STRATEGIES}
    proposed=max(STRATEGIES,key=lambda s:(scores[s],-STRATEGIES.index(s)))
    eligible=[r for r in frozen if r["name"].endswith("-400")]
    freeze=dict(protocol_sha256=report["protocol_sha256"],manifest_sha256=report["manifest_sha256"],
                development_sha256=sha((HERE/"development.json").read_bytes()),
                candidates=eligible,proposed_strategy=proposed,mean_yields=scores,
                confirmation_consumed=False,confirmation_scope="untouched same-source public test; not customer traffic")
    write(HERE/"FREEZE.json",freeze)


def confirm():
    freeze=read(HERE/"FREEZE.json")
    assert sha((HERE/"PROTOCOL.md").read_bytes())==freeze["protocol_sha256"]
    assert sha((HERE/"source-manifest.json").read_bytes())==freeze["manifest_sha256"]
    target=HERE/"confirmation.json"
    if target.exists():
        raise RuntimeError("Confirmation was already consumed; never repeat a selection search against it")
    rows=read(CACHE/"emotion.json")["test"]
    results=[]
    for candidate in freeze["candidates"]:
        path=ROOT/candidate["path"]
        assert sha(path.read_bytes())==candidate["sha256"]
        metrics,cases=outcomes(engine(path),rows)
        write(CACHE/candidate["name"]/"confirmation-cases.json",cases)
        results.append(dict(name=candidate["name"],metrics=metrics,
                            cases_sha256=sha((CACHE/candidate["name"]/"confirmation-cases.json").read_bytes())))
        print(json.dumps(dict(name=candidate["name"],metrics=metrics)),flush=True)
    write(target,dict(freeze_sha256=sha((HERE/"FREEZE.json").read_bytes()),results=results,
                      evaluation_labels_per_candidate=len(rows),scope=freeze["confirmation_scope"]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["prepare","develop","confirm"])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        {"prepare":prepare,"develop":develop,"confirm":confirm}[args.phase]()


if __name__=="__main__":
    main()
