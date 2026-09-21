"""Research-only supervised adaptation of a small encoder on fitting rows.

This changes pretrained encoder weights. It is not instantaneous head teaching,
and its compute/files must be counted if a candidate is ever adopted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer
from transformers import AutoModel

from develop import OUTPUT, load_splits, summarize
from encoder_probe import Encoder


class Classifier(torch.nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(OUTPUT / "minilm", local_files_only=True, attn_implementation="eager")
        self.head = torch.nn.Linear(384, len(labels))

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(torch.nn.functional.normalize(pooled, dim=1))


def main(dataset):
    torch.set_num_threads(1)
    torch.manual_seed(20260921)
    rng = np.random.default_rng(20260921)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_splits(dataset)
    labels = sorted({r["label"] for r in data["fit"]})
    tokenizer = Tokenizer.from_file(str(OUTPUT / "minilm/tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    model = Classifier(labels)
    # Initialize from the measured frozen-encoder linear fit.
    base = Encoder("minilm")
    key = hashlib.sha256((base.projector_digest() + json.dumps(data["fit"], sort_keys=True)).encode()).hexdigest()
    x = np.load(OUTPUT / f"features-{key}.npy", allow_pickle=False)
    with threadpool_limits(limits=1):
        initial = LogisticRegression(C=10, max_iter=1000).fit(x, [r["label"] for r in data["fit"]])
    model.head.weight.data.copy_(torch.from_numpy(initial.coef_.astype(np.float32)))
    model.head.bias.data.copy_(torch.from_numpy(initial.intercept_.astype(np.float32)))
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=.01)
    epochs, batch_size = 4, 64
    total_steps = epochs * math.ceil(len(data["fit"]) / batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(.05, 1 - step / total_steps))
    config = dict(kind="adapted-minilm-classifier", encoder_lr=2e-5, head_lr=1e-3,
                  weight_decay=.01, epochs=epochs, batch_size=batch_size, seed=20260921,
                  max_tokens=256, device=device, initialization="frozen-encoder logistic C=10")
    report = dict(scope="development only; pretrained encoder weights updated using fitting rows only",
                  config=config, environment=dict(torch=torch.__version__), results=[])
    folder = OUTPUT / f"adapted-{dataset}"
    folder.mkdir(exist_ok=True)
    best_coverage = -1
    start = time.perf_counter()

    def batch(rows):
        encoded = tokenizer.encode_batch([r["prompt"] for r in rows])
        return {name: torch.tensor([getattr(t, attribute) for t in encoded], device=device, dtype=torch.long)
                for name, attribute in [("input_ids", "ids"), ("attention_mask", "attention_mask"), ("token_type_ids", "type_ids")]}

    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            losses = []
            order = rng.permutation(len(data["fit"]))
            for offset in range(0, len(order), batch_size):
                rows = [data["fit"][i] for i in order[offset:offset + batch_size]]
                optimizer.zero_grad(set_to_none=True)
                logits = model(**batch(rows))
                target = torch.tensor([labels.index(r["label"]) for r in rows], device=device)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
                if offset % (batch_size * 40) == 0:
                    print(json.dumps(dict(epoch=epoch, fit_rows=offset, total=len(order), loss=losses[-1])), flush=True)
            print(json.dumps(dict(epoch=epoch, mean_loss=float(np.mean(losses)))), flush=True)
        model.eval()
        predictions = []
        with torch.no_grad():
            for offset in range(0, len(data["development"]), batch_size):
                predictions.extend(model(**batch(data["development"][offset:offset + batch_size])).cpu().numpy())
        elapsed = (time.perf_counter() - start) * 1000
        result = summarize(dataset, {**config, "epoch": epoch}, data["development"], labels, np.asarray(predictions), elapsed)
        report["results"].append(result)
        coverage = max(result[k]["best_development_coverage_at_quality_targets"]["coverage"] for k in ("probability", "margin"))
        if coverage > best_coverage:
            best_coverage = coverage
            model.encoder.save_pretrained(folder / "encoder", safe_serialization=True)
            np.savez_compressed(folder / "head.npz", weights=model.head.weight.detach().cpu().numpy(),
                                biases=model.head.bias.detach().cpu().numpy())
            report["selected_epoch"] = epoch
            (folder / "labels.json").write_text(json.dumps(labels) + "\n")
        (folder / "development.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["clinc150", "banking77"])
    main(parser.parse_args().dataset)
