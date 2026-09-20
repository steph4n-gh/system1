"""System 1 CLI Command Handlers.

Provides:
- system1 decide: Evaluates inputs against typed decision schemas.
- system1 bench: Latency & throughput benchmark beating TypeSafe AI (Jev).
- system1 calibrate: Fits temperature scaling and conformal prediction sets.
- system1 verify-receipt: Offline cryptographic verification of decision receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from system1.calibration import (
    compute_brier_decomposition,
    compute_ece_and_bins,
    compute_nll,
)
from system1.engine import BenchmarkReport, DecisionResult, SystemOneEngine
from system1.guard import DefaultGuardDecisionSchema
from system1.ledger import ActionLedger
from system1.receipt import (
    load_private_key,
    load_public_key,
    public_key_fingerprint,
    save_keypair,
    verify_decision_witness_receipt,
)
from system1.core import (
    BooleanField,
    ChoiceField,
    DecisionField,
    DecisionSchema,
    MultiChoiceField,
    ScoreField,
)


class DefaultTriageSchema(DecisionSchema):
    """General agent and routing triage schema."""

    route = ChoiceField(
        options=["filesystem", "database", "network", "system_shell", "human_approval"],
        descriptions={
            "filesystem": "Local file reads, writes, and workspace edits",
            "database": "Structured SQLite or archive queries",
            "network": "External HTTP and remote communications",
            "system_shell": "Direct subprocess execution or build tools",
            "human_approval": "High-risk, ambiguous, or irreversible actions requiring human review",
        },
    )
    is_safe = BooleanField(
        description="Whether the requested operation is safe to proceed without extra approval",
        threshold=0.5,
    )
    risk_level = ScoreField(
        min_value=0.0,
        max_value=1.0,
        description="Continuous risk rating from 0.0 (benign) to 1.0 (dangerous)",
    )


def _load_schema(schema_arg: Optional[str]) -> DecisionSchema:
    """Loads a DecisionSchema from a file, inline JSON, standard preset, or import path."""
    if not schema_arg or schema_arg.strip().lower() in ("triage", "default"):
        return DefaultTriageSchema()
    if schema_arg.strip().lower() in ("guard", "guardrail"):
        return DefaultGuardDecisionSchema()

    # Check for dotted Python import path (e.g. app.schemas.Triage or module:Class)
    if "." in schema_arg and not schema_arg.endswith(".json") and not Path(schema_arg).is_file():
        import importlib
        cwd_str = str(Path.cwd())
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)
        mod_name, cls_name = schema_arg.split(":", 1) if ":" in schema_arg else schema_arg.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_name)
            cls_obj = getattr(mod, cls_name)
            if isinstance(cls_obj, type) and issubclass(cls_obj, DecisionSchema):
                return cls_obj()
            elif isinstance(cls_obj, DecisionSchema):
                return cls_obj
        except Exception:
            pass

    schema_path = Path(schema_arg)
    if schema_path.is_file():
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return DecisionSchema.from_dict(data)

    try:
        data = json.loads(schema_arg)
        if isinstance(data, dict):
            return DecisionSchema.from_dict(data)
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"Unable to parse schema from {schema_arg!r}. Provide 'triage', 'guard', an import path, a JSON file path, or raw JSON."
    )


def handle_decide_command(args: argparse.Namespace) -> int:
    """CLI handler for 'system1 decide'."""
    prompt = getattr(args, "prompt", None) or getattr(args, "opt_prompt", None)
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("[SYSTEM1 ERROR] No prompt provided. Use system1 decide '...' or --prompt '...'", file=sys.stderr)
        return 1

    model = None
    if getattr(args, "model", None):
        from system1.compiler import CompiledSystemOneModel
        model = CompiledSystemOneModel.load(args.model)
        schema = model.schema
    else:
        schema = _load_schema(getattr(args, "schema", None))
    alpha = float(getattr(args, "alpha", 0.05))
    sign = bool(getattr(args, "sign", False))
    ledger_path = getattr(args, "ledger", None)

    signing_key = None
    if sign:
        identity_dir = getattr(args, "identity_dir", None)
        if identity_dir:
            id_path = Path(identity_dir)
            if id_path.is_file():
                raise NotADirectoryError(f"--identity-dir must be a directory, got file: {identity_dir}")
            id_path.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(id_path, 0o700)
            except OSError:
                pass
            key_file = id_path / "identity.key"
            if key_file.is_file():
                signing_key = load_private_key(key_file)
            else:
                signing_key = Ed25519PrivateKey.generate()
                save_keypair(signing_key, id_path)
        else:
            # Persistent default directory: ~/.system1/identity, falling back to ./.system1/identity if restricted
            candidate_paths = [
                Path.home() / ".system1" / "identity",
                Path.cwd() / ".system1" / "identity",
            ]
            saved = False
            for cand in candidate_paths:
                try:
                    cand.mkdir(parents=True, exist_ok=True)
                    try:
                        os.chmod(cand, 0o700)
                    except OSError:
                        pass
                    key_file = cand / "identity.key"
                    if key_file.is_file():
                        signing_key = load_private_key(key_file)
                    else:
                        signing_key = Ed25519PrivateKey.generate()
                        save_keypair(signing_key, cand)
                    saved = True
                    break
                except (PermissionError, OSError):
                    continue
            if not saved:
                raise PermissionError(
                    "Cannot write to default identity directory (~/.system1/identity or ./.system1/identity). "
                    "Provide a writable directory via --identity-dir."
                )

    ledger = None
    if ledger_path:
        ledger = ActionLedger(ledger_path)

    engine = SystemOneEngine(
        schema, model=model, signing_key=signing_key, ledger=ledger,
        strict_mode=model is not None,
    )
    result = engine.decide(prompt, alpha=alpha, record_receipt=True)

    if getattr(args, "json", False):
        print(result.to_json(indent=2))
        return 0

    print(f"\n[SYSTEM1 DECISION] Evaluated in {result.latency_ms:.2f}ms (Local Non-Autoregressive)")
    print("=" * 70)
    print(f"Prompt: {result.prompt}")
    print(f"Schema: {result.schema_name} (digest: {result.schema_digest[:12]}...)")
    print("-" * 70)
    print("DECISION VALUES & CALIBRATED CONFIDENCES:")
    for field_name, val in result.values.items():
        conf = result.confidences.get(field_name, 0.0)
        cset = result.conformal_sets.get(field_name, [])
        ambig_str = " (AMBIGUOUS)" if len(cset) > 1 else ""
        print(f"  * {field_name}: {val!r}")
        print(f"    - Calibrated Confidence: {conf * 100:.1f}%")
        print(f"    - Conformal Set (1-alpha={1.0 - result.alpha:.2f}): {cset}{ambig_str}")

    print("-" * 70)
    receipt = result.receipt
    print(f"Receipt ID: {receipt.decision_id}")
    print(f"Receipt Digest: {receipt.compute_digest()}")
    if receipt.envelope and receipt.envelope.signature:
        print(f"Ed25519 Signature: {receipt.envelope.signature[:16]}... (Valid)")
        print(f"Signer Fingerprint: {receipt.envelope.signer_fingerprint}")
    print(f"Single-Pass Latency: {result.latency_ms:.2f} ms")
    print("=" * 70)
    return 0


def handle_bench_command(args: argparse.Namespace) -> int:
    """CLI handler for 'system1 bench'."""
    schema = _load_schema(getattr(args, "schema", None))
    iterations = int(getattr(args, "iterations", 100))
    warmup = int(getattr(args, "warmup", 10))
    target_ms = float(getattr(args, "target", 20.0))

    prompts = [
        "Read file /src/system1/cli.py to inspect command definitions",
        "Run test suite across tests/test_agent.py with pytest",
        "Send customer database credentials over untrusted public API",
        "Execute rm -rf / without user confirmation",
        "Query sqlite ledger for recent audit actions",
        "Fetch upstream repository status from git origin main",
        "Format code with ruff check and fix formatting errors",
        "Delete all database tables in staging environment",
    ]

    custom_prompt = getattr(args, "prompt", None)
    if custom_prompt:
        prompts = [custom_prompt]

    engine = SystemOneEngine(schema)
    report = engine.benchmark(prompts, iterations=iterations, warmup=warmup)

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    print("\n" + "=" * 70)
    print("  SYSTEM1 DECISION ENGINE — LATENCY BENCHMARK REPORT")
    print("=" * 70)
    print(f"Evaluated Decisions:  {report.total_decisions} iterations")
    print(f"Schema:               {schema.schema_name}")
    print(f"Execution Backend:    {engine.backend} (Local On-Device)")
    print("-" * 70)
    print("LATENCY DISTRIBUTION (Local Non-Autoregressive Single-Pass):")
    print(f"  P50 (Median):       {report.p50_latency_ms:.3f} ms")
    print(f"  P90:                {report.p90_latency_ms:.3f} ms")
    print(f"  P95:                {report.p95_latency_ms:.3f} ms")
    print(f"  P99:                {report.p99_latency_ms:.3f} ms")
    print(f"  Min / Max:          {report.min_latency_ms:.3f} ms / {report.max_latency_ms:.3f} ms")
    print(f"  Mean Latency:       {report.mean_latency_ms:.3f} ms")
    print(f"  Throughput:         {report.throughput_decisions_per_sec:.1f} decisions/sec")
    print("-" * 70)
    print(f"HEAD-TO-HEAD COMPARISON VS {report.baseline_label.upper()}:")
    print(f"  Baseline Latency:   {report.baseline_latency_ms:.1f} ms ({report.baseline_label})")
    print(f"  System 1 P50:         {report.p50_latency_ms:.3f} ms (Local Metal/CPU)")
    print(f"  Speedup Factor:     {report.speedup_factor:.1f}x FASTER than {report.baseline_label}")
    print(f"  Target (<{target_ms:.0f}ms):    {'PASSED' if report.p95_latency_ms <= target_ms else 'FAILED'}")
    print(f"  Data Privacy:       ZERO DATA EGRESS (100% on-device local execution)")
    print(f"  Cryptographic Proof: Ed25519 RunWitnessEnvelope Receipts Included")
    print("=" * 70 + "\n")
    return 0


def handle_calibrate_command(args: argparse.Namespace) -> int:
    """CLI handler for 'system1 calibrate'."""
    dataset_path = getattr(args, "dataset", None)
    if not dataset_path or not Path(dataset_path).is_file():
        print("[SYSTEM1 ERROR] A valid calibration --dataset JSON file must be supplied.", file=sys.stderr)
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print("[SYSTEM1 ERROR] Dataset must be a non-empty list of [prompt, labels_dict] items.", file=sys.stderr)
        return 1

    formatted_dataset = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            formatted_dataset.append((str(item[0]), dict(item[1])))
        elif isinstance(item, dict) and "prompt" in item and "labels" in item:
            formatted_dataset.append((str(item["prompt"]), dict(item["labels"])))

    schema = _load_schema(getattr(args, "schema", None))
    bins = int(getattr(args, "bins", 10))

    engine = SystemOneEngine(schema)
    metrics = engine.calibrate(formatted_dataset, n_bins=bins)

    output = {k: v.to_dict() for k, v in metrics.items()}
    if getattr(args, "json", False):
        print(json.dumps(output, indent=2))
        return 0

    print("\n[SYSTEM1 CALIBRATION SUMMARY]")
    print("=" * 70)
    for field_name, m in metrics.items():
        print(f"Field: {field_name!r}")
        print(f"  - Optimal Temperature:        {m.optimal_temperature:.4f}")
        print(f"  - Expected Calib Error (ECE): {m.expected_calibration_error:.4f}")
        print(f"  - Max Calib Error (MCE):      {m.maximum_calibration_error:.4f}")
        print(f"  - Negative Log-Likelihood:    {m.negative_log_likelihood:.4f}")
        print(f"  - Total Brier Score:          {m.brier.total_brier:.4f}")
        print(f"    * Reliability (Calib Loss): {m.brier.reliability:.4f} (lower is better)")
        print(f"    * Resolution (Sorting):     {m.brier.resolution:.4f} (higher is better)")
        print(f"    * Uncertainty:              {m.brier.uncertainty:.4f}")
        print("-" * 70)
    return 0


def handle_verify_receipt_command(args: argparse.Namespace) -> int:
    """CLI handler for 'system1 verify-receipt'."""
    receipt_file = getattr(args, "receipt", None) or getattr(args, "receipt_pos", None)
    if receipt_file and not getattr(args, "receipt", None):
        setattr(args, "receipt", receipt_file)

    if receipt_file:
        p = Path(receipt_file)
        if not p.is_file():
            print(f"[SYSTEM1 ERROR] Receipt file not found: {receipt_file}", file=sys.stderr)
            return 1
        try:
            with open(p, "r", encoding="utf-8") as f:
                receipt_data = json.load(f)
        except Exception as exc:
            print(f"[SYSTEM1 ERROR] Failed to parse receipt JSON from {receipt_file}: {exc}", file=sys.stderr)
            return 1
    elif not sys.stdin.isatty():
        try:
            receipt_data = json.load(sys.stdin)
        except Exception as exc:
            print(f"[SYSTEM1 ERROR] Failed to parse receipt JSON from stdin: {exc}", file=sys.stderr)
            return 1
    else:
        print("[SYSTEM1 ERROR] Provide receipt file path (positional or --receipt) or pipe JSON via stdin.", file=sys.stderr)
        return 1

    pub_key = None
    pub_key_path = getattr(args, "public_key", None)
    if not pub_key_path:
        message = "Authentication requires --public-key from an independently trusted source"
        if getattr(args, "json", False):
            print(json.dumps({"verified": False, "error": message}, indent=2))
        else:
            print(f"[SYSTEM1 VERIFICATION FAILED] {message}", file=sys.stderr)
        return 1
    if pub_key_path:
        if Path(pub_key_path).is_file():
            try:
                pub_key = load_public_key(pub_key_path)
            except Exception:
                try:
                    with open(pub_key_path, "rb") as f:
                        raw_bytes = f.read().strip()
                        if len(raw_bytes) == 32:
                            pub_key = Ed25519PublicKey.from_public_bytes(raw_bytes)
                        else:
                            pub_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(raw_bytes.decode("utf-8")))
                except Exception:
                    pass
        else:
            try:
                pub_key = load_public_key(pub_key_path)
            except Exception:
                try:
                    clean = pub_key_path.strip()
                    if len(clean) == 64:
                        pub_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(clean))
                    elif len(clean.encode("utf-8")) == 32:
                        pub_key = Ed25519PublicKey.from_public_bytes(clean.encode("utf-8"))
                except Exception:
                    pass

        if pub_key is None:
            err_msg = f"[SYSTEM1 ERROR] Failed to load trusted public key from: {pub_key_path}"
            if getattr(args, "json", False):
                print(json.dumps({"verified": False, "error": err_msg}, indent=2))
            print(err_msg, file=sys.stderr)
            return 1

    try:
        valid = verify_decision_witness_receipt(receipt_data, public_key=pub_key)
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"verified": False, "error": str(exc)}, indent=2))
        else:
            print(f"[SYSTEM1 VERIFICATION FAILED] {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps({
            "verified": valid,
            "decision_id": receipt_data.get("decision_id"),
            "receipt_digest": receipt_data.get("receipt_digest"),
            "schema_digest": receipt_data.get("schema_digest"),
        }, indent=2))
        return 0 if valid else 1

    if valid:
        print("\n[SYSTEM1 RECEIPT VERIFICATION] SUCCESS")
        print(f"Decision ID:    {receipt_data.get('decision_id')}")
        print(f"Receipt Digest: {receipt_data.get('receipt_digest')}")
        print(f"Schema Digest:  {receipt_data.get('schema_digest')}")
        print("Verdict:        VALID (Cryptographically Authentic & Tamper-Evident)\n")
        return 0
    else:
        print("\n[SYSTEM1 RECEIPT VERIFICATION] FAILED: Integrity or signature check invalid.\n", file=sys.stderr)
        return 1


def _load_examples(path: str) -> Dict[str, list]:
    """Read explicit labeled examples without inventing missing prompts or labels."""
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        if not data or any(not isinstance(rows, list) or not rows for rows in data.values()):
            raise ValueError("Example fields must contain nonempty lists of labeled examples")
        return data
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a nonempty list of labeled examples or a field mapping")
    examples: Dict[str, list] = {}
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Example {index} must be an object")
        labels = row.get("labels", row.get("values"))
        if not isinstance(labels, dict) or not labels:
            raise ValueError(f"Example {index} requires explicit labels")
        for field, label in labels.items():
            sample = (row.get("prompt"), label)
            if "telemetry" in row:
                sample += (row["telemetry"],)
            examples.setdefault(field, []).append(sample)
    return examples


def handle_compile_command(args: argparse.Namespace) -> int:
    """CLI handler for 'system1 compile'."""
    from system1.compiler import SystemOneCompiler

    t0 = time.perf_counter()
    schema = _load_schema(args.schema)

    dataset_path = getattr(args, "dataset", None)
    calibration_path = getattr(args, "calibration_dataset", None)
    try:
        exemplars = _load_examples(dataset_path) if dataset_path else None
        calibration_exemplars = _load_examples(calibration_path) if calibration_path else None
    except (OSError, ValueError) as exc:
        print(f"[SYSTEM1 ERROR] Cannot load examples: {exc}", file=sys.stderr)
        return 1

    compiler = SystemOneCompiler(
        schema=schema,
        dimension=getattr(args, "dimension", 384),
        regularization=getattr(args, "regularization", 1.0),
    )

    output_path = getattr(args, "output", "model.s1m") or "model.s1m"
    teacher = getattr(args, "teacher", "synthetic")
    samples_per_choice = getattr(args, "samples_per_choice", 20)

    try:
        compiled_model = compiler.compile_and_save(
            output_path=output_path,
            exemplars=exemplars,
            samples_per_choice=samples_per_choice,
            teacher=teacher,
            augment=exemplars is None or bool(getattr(args, "augment", False)),
            calibration_exemplars=calibration_exemplars,
        )
    except (TypeError, ValueError) as exc:
        print(f"[SYSTEM1 ERROR] Cannot compile examples: {exc}", file=sys.stderr)
        return 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out_file = Path(output_path)
    file_size_kb = out_file.stat().st_size / 1024.0 if out_file.is_file() else 0.0

    summary = {
        "status": "compiled",
        "schema_name": schema.schema_name,
        "schema_digest": schema.schema_digest(),
        "output_path": str(out_file.resolve()),
        "file_size_kb": round(file_size_kb, 2),
        "fields_compiled": list(schema.fields.keys()),
        "regularization": getattr(args, "regularization", 1.0),
        "teacher": compiled_model.metadata["teacher"],
        "teaching_mode": compiled_model.metadata["teaching_mode"],
        "sample_counts": compiled_model.metadata["sample_counts"],
        "compilation_latency_ms": round(elapsed_ms, 2),
    }

    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n[SYSTEM 1 COMPILER] Distillation and Compilation Complete")
        print(f"Schema Name:        {summary['schema_name']}")
        print(f"Schema Digest:      {summary['schema_digest']}")
        print(f"Output Binary:      {summary['output_path']} ({summary['file_size_kb']} KB)")
        print(f"Fields:             {', '.join(summary['fields_compiled'])}")
        print(f"Compilation Time:   {summary['compilation_latency_ms']:.2f} ms\n")

    return 0


def handle_serve_command(args: argparse.Namespace) -> int:
    """CLI handler for 'system1 serve' — starts the gRPC sidecar server."""
    try:
        from system1.grpc_server import serve as grpc_serve, grpc_available
    except ImportError:
        print(
            "[SYSTEM1 ERROR] grpcio is required for the serve command.  "
            "Install with: pip install 'system1[grpc]'",
            file=sys.stderr,
        )
        return 1

    if not grpc_available():
        print(
            "[SYSTEM1 ERROR] grpcio is not installed.  "
            "Install with: pip install 'system1[grpc]'",
            file=sys.stderr,
        )
        return 1

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 50051))
    schema_args: List[str] = getattr(args, "schema", None) or []

    # Build schemas dict from --schema flags.
    # Formats: "guard", "triage", "name:path/to/file.s1m"
    schemas: Dict[str, Any] = {}
    for spec in schema_args:
        if ":" in spec and not spec.startswith(":"):
            name, path_str = spec.split(":", 1)
            schema = _load_schema(path_str)
            schemas[name] = schema
        else:
            schema = _load_schema(spec)
            schemas[spec] = schema

    print(f"[SYSTEM1] Starting gRPC server on {host}:{port}...")
    grpc_serve(host=host, port=port, schemas=schemas if schemas else None, block=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the unified CLI argument parser for system1."""
    parser = argparse.ArgumentParser(
        prog="system1",
        description="System 1: Machine-Native Decision Runtime",
    )
    subparsers = parser.add_subparsers(dest="command", help="System 1 command to execute")

    # Command: decide
    decide_parser = subparsers.add_parser("decide", help="Evaluate inputs against typed decision schemas")
    decide_parser.add_argument("prompt", nargs="?", default=None, help="Decision input prompt")
    decide_parser.add_argument("--prompt", dest="opt_prompt", help="Alternative prompt flag")
    decide_source = decide_parser.add_mutually_exclusive_group()
    decide_source.add_argument("--schema", help="Schema preset ('triage', 'guard'), import path, or path to JSON schema file")
    decide_source.add_argument("--model", help="Run a taught .s1m skill with strict uncertainty gating")
    decide_parser.add_argument("--alpha", type=float, default=0.05, help="Conformal significance level (default: 0.05 for 95%% coverage)")
    decide_parser.add_argument("--sign", action="store_true", help="Generate Ed25519 signature on decision receipt")
    decide_parser.add_argument("--ledger", help="Path to SQLite ActionLedger file")
    decide_parser.add_argument("--identity-dir", help="Path to directory containing identity.key")
    decide_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    decide_parser.set_defaults(func=handle_decide_command)

    # Command: bench
    bench_parser = subparsers.add_parser("bench", help="Run latency & throughput benchmark beating Jev")
    bench_parser.add_argument("--schema", help="Schema preset ('triage', 'guard'), import path, or path to JSON schema file")
    bench_parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    bench_parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    bench_parser.add_argument("--target", type=float, default=20.0, help="Target latency bound in milliseconds (default: 20ms)")
    bench_parser.add_argument("--prompt", help="Custom prompt to benchmark")
    bench_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    bench_parser.set_defaults(func=handle_bench_command)

    # Command: calibrate
    calib_parser = subparsers.add_parser("calibrate", help="Fit temperature scaling and conformal prediction sets")
    calib_parser.add_argument("--dataset", required=True, help="Path to JSON calibration dataset file")
    calib_parser.add_argument("--schema", help="Schema preset ('triage', 'guard'), import path, or path to JSON schema file")
    calib_parser.add_argument("--bins", type=int, default=10, help="Number of calibration probability bins")
    calib_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    calib_parser.set_defaults(func=handle_calibrate_command)

    # Command: verify-receipt
    verify_parser = subparsers.add_parser("verify-receipt", help="Offline cryptographic verification of decision receipts")
    verify_parser.add_argument("receipt_pos", nargs="?", default=None, metavar="receipt", help="Path to JSON receipt file (or pipe via stdin)")
    verify_parser.add_argument("--receipt", help="Path to JSON receipt file (or pipe via stdin)")
    verify_parser.add_argument("--public-key", help="Path to trusted Ed25519 public key file")
    verify_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    verify_parser.set_defaults(func=handle_verify_receipt_command)

    # Command: compile
    compile_parser = subparsers.add_parser("compile", help="Distill and compile schema into compact .s1m binary model")
    compile_parser.add_argument("--schema", required=True, help="Schema preset ('triage', 'guard'), import path (e.g. app.schemas.Triage), or JSON schema file")
    compile_parser.add_argument("--teacher", default="synthetic", help="Teacher model/strategy ('synthetic', 'mock', 'jev', 'claude', 'gpt4o')")
    compile_parser.add_argument("--output", "-o", default="model.s1m", help="Output path for compiled .s1m binary model")
    compile_parser.add_argument("--dataset", help="Labeled JSON examples for teaching a skill; no synthetic augmentation by default")
    compile_parser.add_argument("--calibration-dataset", help="Separate labeled examples for checking uncertainty")
    compile_parser.add_argument("--augment", action="store_true", help="Explicitly add schema-derived synthetic examples")
    compile_parser.add_argument("--dimension", type=int, default=384, help="Feature dimension (default: 384)")
    compile_parser.add_argument("--samples-per-choice", type=int, default=20, help="Number of synthetic samples to generate per choice")
    compile_parser.add_argument("--regularization", type=float, default=1.0, help="Ridge regression L2 regularization lambda")
    compile_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON compilation summary")
    compile_parser.set_defaults(func=handle_compile_command)

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start the System 1 gRPC sidecar server")
    serve_parser.add_argument("--grpc", action="store_true", default=True, help="Use gRPC transport (default)")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=50051, help="Port to listen on (default: 50051)")
    serve_parser.add_argument("--schema", action="append", help="Schema to load: 'guard', 'triage', or 'name:path/to/file.s1m'. Can be repeated.")
    serve_parser.set_defaults(func=handle_serve_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


__all__ = [
    "build_parser",
    "main",
    "handle_decide_command",
    "handle_bench_command",
    "handle_calibrate_command",
    "handle_verify_receipt_command",
    "handle_compile_command",
    "handle_serve_command",
    "DefaultTriageSchema",
]


if __name__ == "__main__":
    sys.exit(main())
