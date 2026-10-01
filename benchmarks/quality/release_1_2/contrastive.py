#!/usr/bin/env python3
"""Bounded cached-encoder control and supervised contrastive projection trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from run import CACHE,HERE,ROOT,LABELS,SEEDS,Emotion,read,write,sha,role_rows,outcomes
from system1 import SystemOneCompiler
from system1.compiler import CompiledSystemOneModel
from system1.engine import SystemOneEngine

ENCODER_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FOLDER=Path.home()/".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"/ENCODER_REVISION


class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoModel,AutoTokenizer
        torch.set_num_threads(1)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_FOLDER,local_files_only=True)
        self.model=AutoModel.from_pretrained(MODEL_FOLDER,local_files_only=True).eval()
        self.identity=dict(name="sentence-transformers/all-MiniLM-L6-v2",revision=ENCODER_REVISION,
                           source="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",license="Apache-2.0",
                           runtime="torch CPU, one thread, max_length256, mean pooling normalized",
                           files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
                                  for p in MODEL_FOLDER.iterdir() if p.is_file()})

    def encode(self,texts):
        output=[]
        for start in range(0,len(texts),32):
            tokens=self.tokenizer(texts[start:start+32],padding=True,truncation=True,max_length=256,return_tensors="pt")
            with self.torch.inference_mode():
                states=self.model(**tokens).last_hidden_state
                mask=tokens["attention_mask"].unsqueeze(-1)
                pooled=(states*mask).sum(1)/mask.sum(1).clamp_min(1)
                output.append(self.torch.nn.functional.normalize(pooled,dim=1).numpy())
        return np.concatenate(output)


def fit_projection(features,labels,seed):
    import torch
    torch.manual_seed(seed)
    layer=torch.nn.Linear(384,64,bias=False)
    optimizer=torch.optim.Adam(layer.parameters(),lr=.001)
    x=torch.tensor(features,dtype=torch.float32)
    y=torch.tensor([LABELS.index(label) for label in labels])
    rng=np.random.default_rng(seed)
    losses=[]
    start=time.perf_counter()
    for epoch in range(20):
        order=rng.permutation(len(features))
        for index in range(0,len(order),64):
            indices=order[index:index+64]
            z=torch.nn.functional.normalize(layer(x[indices]),dim=1)
            logits=z@z.T/.1
            self_mask=torch.eye(len(indices),dtype=torch.bool)
            positive=y[indices,None].eq(y[indices][None,:]) & ~self_mask
            logits=logits.masked_fill(self_mask,-1e9)
            log_probability=logits-torch.logsumexp(logits,dim=1,keepdim=True)
            positive_count=positive.sum(1)
            valid=positive_count>0
            loss=-(log_probability*positive).sum(1)[valid]/positive_count[valid]
            if not len(loss):
                continue
            loss=loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return layer.weight.detach().numpy(),time.perf_counter()-start,losses


class Projector:
    def __init__(self,encoder,rows,vectors,weights=None):
        self.encoder=encoder
        self.weights=weights
        self.dimension=64 if weights is not None else 384
        self.features={r["input"]:v for r,v in zip(rows,vectors)}

    def project(self,text):
        vector=self.features.get(text)
        if vector is None:
            vector=self.encoder.encode([text])[0]
        if self.weights is not None:
            vector=self.weights@vector
            vector=vector/max(np.linalg.norm(vector),1e-9)
        return vector.astype(np.float32)

    def projector_digest(self):
        return sha(json.dumps(self.encoder.identity,sort_keys=True)+(sha(self.weights.tobytes()) if self.weights is not None else "frozen"))


def fit_candidate(fitting,calibration,projector,path):
    start=time.perf_counter()
    model=SystemOneCompiler(Emotion,projector=projector,regularization=.1,backend="numpy").compile(
        {"label":[(r["input"],r["label"]) for r in fitting]},augment=False,
        calibration_exemplars={"label":[(r["input"],r["label"]) for r in calibration]})
    model.save(path)
    restored=CompiledSystemOneModel.load(path,projector=projector)
    return SystemOneEngine(restored.schema,model=restored,strict_mode=True,use_cache=False),time.perf_counter()-start


def develop():
    encoder=Encoder()
    write(HERE/"encoder-source-manifest.json",encoder.identity)
    results,candidates=[],[]
    for seed in SEEDS:
        calibration,initial,pool,splits=role_rows(seed)
        ids=read(CACHE/f"ridge-random-{seed}-400/selection.json")
        by_id={r["id"]:r for r in initial+pool}
        fitting=[by_id[i] for i in ids]
        rows=fitting+calibration+splits["validation"]
        start=time.perf_counter()
        features=encoder.encode([r["input"] for r in rows])
        embedding_time=time.perf_counter()-start
        weights,projection_time,losses=fit_projection(features[:len(fitting)],[r["label"] for r in fitting],seed)
        for kind,projection in [("frozen_encoder",None),("contrastive_projection",weights)]:
            name=f"{kind}-{seed}-400"
            folder=CACHE/name
            folder.mkdir(parents=True,exist_ok=True)
            projector=Projector(encoder,rows,features,projection)
            path=folder/"candidate.s1m"
            if projection is not None:
                np.save(folder/"projection.npy",projection,allow_pickle=False)
            current,fit_time=fit_candidate(fitting,calibration,projector,path)
            metrics,cases=outcomes(current,splits["validation"])
            metrics["cached_features_median_latency_ms"]=metrics.pop("median_latency_ms")
            metrics["cached_features_p95_latency_ms"]=metrics.pop("p95_latency_ms")
            # Count actual uncached tokenization+encoder+projection+decision time.
            projector.features={}
            timings=[]
            for row in splits["validation"][:40]:
                start=time.perf_counter_ns()
                current.decide(row["input"],record_receipt=False)
                timings.append((time.perf_counter_ns()-start)/1e6)
            metrics["median_latency_ms"]=float(np.median(timings))
            metrics["p95_latency_ms"]=float(np.percentile(timings,95))
            write(folder/"development-cases.json",cases)
            results.append(dict(name=name,seed=seed,kind=kind,budget=400,fit_labels=360,calibration_labels=40,
                                inspected_total=400,metrics=metrics,embedding_seconds=embedding_time,
                                projection_seconds=projection_time if projection is not None else 0,
                                projection_epochs=20 if projection is not None else 0,
                                first_loss=losses[0] if projection is not None else None,
                                final_loss=losses[-1] if projection is not None else None,
                                fit_seconds=fit_time,model_sha256=sha(path.read_bytes()),
                                model_bytes=path.stat().st_size,
                                encoder_weights_bytes=(MODEL_FOLDER/"model.safetensors").stat().st_size,
                                external_runtime_required=True))
            candidates.append(dict(name=name,path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes()),
                                   projection_sha256=sha((folder/"projection.npy").read_bytes()) if projection is not None else None))
            print(json.dumps(dict(name=name,metrics=metrics)),flush=True)
    write(HERE/"contrastive-development.json",dict(results=results,
          protocol_sha256=sha((HERE/"PROTOCOL.md").read_bytes()),encoder_identity=encoder.identity))
    write(HERE/"contrastive-FREEZE.json",dict(candidates=candidates,
          development_sha256=sha((HERE/"contrastive-development.json").read_bytes()),
          protocol_sha256=sha((HERE/"PROTOCOL.md").read_bytes()),encoder_identity=encoder.identity,
          source_manifest_sha256=sha((HERE/"source-manifest.json").read_bytes())))


def confirm():
    freeze=read(HERE/"contrastive-FREEZE.json")
    assert freeze["protocol_sha256"]==sha((HERE/"PROTOCOL.md").read_bytes())
    assert freeze["source_manifest_sha256"]==sha((HERE/"source-manifest.json").read_bytes())
    target=HERE/"contrastive-confirmation.json"
    if target.exists():
        raise RuntimeError("Confirmation already consumed")
    encoder=Encoder()
    assert encoder.identity==freeze["encoder_identity"]
    rows=read(CACHE/"emotion.json")["test"]
    features=encoder.encode([r["input"] for r in rows])
    results=[]
    for candidate in freeze["candidates"]:
        path=ROOT/candidate["path"]
        assert sha(path.read_bytes())==candidate["sha256"]
        projection=None
        if candidate["projection_sha256"]:
            weights=path.parent/"projection.npy"
            assert sha(weights.read_bytes())==candidate["projection_sha256"]
            projection=np.load(weights,allow_pickle=False)
        projector=Projector(encoder,rows,features,projection)
        restored=CompiledSystemOneModel.load(path,projector=projector)
        current=SystemOneEngine(restored.schema,model=restored,strict_mode=True,use_cache=False)
        metrics,cases=outcomes(current,rows)
        metrics["cached_features_median_latency_ms"]=metrics.pop("median_latency_ms")
        metrics["cached_features_p95_latency_ms"]=metrics.pop("p95_latency_ms")
        write(path.parent/"confirmation-cases.json",cases)
        results.append(dict(name=candidate["name"],metrics=metrics))
        print(json.dumps(dict(name=candidate["name"],metrics=metrics)),flush=True)
    write(target,dict(results=results,freeze_sha256=sha((HERE/"contrastive-FREEZE.json").read_bytes()),
                     scope="untouched same-source public confirmation; not customer validation"))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["develop","confirm"])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        {"develop":develop,"confirm":confirm}[args.phase]()
