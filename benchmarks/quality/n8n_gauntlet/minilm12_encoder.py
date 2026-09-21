"""Pinned 12-layer encoder for this experiment; older encoder settings stay frozen."""
import hashlib
import json
from pathlib import Path

from develop import OUTPUT
from encoder_probe import Encoder


class MiniLM12Encoder(Encoder):
    SETTINGS = {"minilm12": (384, "mean", 256, 0, "model_qint8_arm64.onnx")}

    def __init__(self):
        assets = json.loads((Path(__file__).parent / "minilm12-assets.json").read_text())
        for name, expected in assets["files"].items():
            raw = (OUTPUT / "minilm12" / name).read_bytes()
            if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                raise ValueError(f"Changed MiniLM12 asset: {name}")
        super().__init__("minilm12", threads=2)
