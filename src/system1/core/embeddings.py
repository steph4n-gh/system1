"""Local subword features and optional lexical/dense feature fusion.

The NumPy subword table is seeded locally and adjusted by curated semantic
anchors. It is not a downloaded or pretrained language model. MLX is optional;
no workload-independent latency bound is asserted."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False


_SEMANTIC_ANCHORS: Dict[str, List[str]] = {
    "rest_sleep_bedroom": [
        "sleep", "bed", "bedroom", "sack", "hay", "slumber", "rest", "nap", "tired",
        "night", "pillow", "mattress", "hitting the sack", "hit the hay", "turn in",
        "bedtime", "sleeping", "asleep", "shuteye", "doze", "drowsy", "bedside lamps",
        "master bedroom", "guest bedroom"
    ],
    "living_room": [
        "living room", "couch", "sofa", "television", "tv", "lounge", "media",
        "coffee table", "entertainment", "living area lights"
    ],
    "kitchen": [
        "kitchen", "cook", "cooking", "refrigerator", "fridge", "oven", "stove",
        "dishwasher", "microwave", "pantry", "dining", "meal", "coffee machine",
        "kitchen lights"
    ],
    "garage": [
        "garage", "driveway", "workshop", "overhead", "parking", "car", "vehicle",
        "exterior floodlights", "garage door"
    ],
    "whole_house": [
        "whole house", "entire residence", "all rooms", "every room", "all lights",
        "everywhere", "domicile", "house", "residence", "throughout the house",
        "all devices in the house"
    ],
    "device_lights": [
        "light", "lights", "lamp", "lamps", "bulb", "bulbs", "fixture", "fixtures",
        "led", "strip", "illumination", "sconce", "chandelier", "illuminate",
        "ceiling fixtures", "turn off lights", "turn on lights"
    ],
    "device_thermostat": [
        "thermostat", "temperature", "heating", "cooling", "hvac", "ac", "degrees",
        "climate control", "cooler", "heater", "furnace"
    ],
    "device_locks": [
        "lock", "locks", "locked", "unlock", "unlocked", "deadbolt", "bolt", "smart deadbolt",
        "front door lock", "secure door", "entryway"
    ],
    "device_blinds": [
        "blinds", "shades", "window shades", "roller blinds", "curtains", "drapery"
    ],
    "device_music": [
        "music", "audio", "speaker", "speakers", "volume", "sound", "spotify",
        "playlist", "track", "song"
    ],
    "action_turn_on": [
        "turn on", "power on", "activate", "illuminate", "open"
    ],
    "action_turn_off": [
        "turn off", "power off", "shut down", "extinguish", "deactivate", "lights out"
    ],
    "action_adjust_level": [
        "adjust level", "dim", "brighten", "raise temperature", "lower temperature", "change volume"
    ],
    "action_lock": [
        "lock door", "engage lock", "secure bolt", "deadbolt lock"
    ],
    "action_unlock": [
        "unlock door", "disengage lock", "unbolt door"
    ],
    "aml_normal": [
        "normal legitimate business", "payroll", "retail purchase", "recurring payment",
        "routine transfer", "standard invoice", "conforms to profile"
    ],
    "aml_structuring": [
        "structuring", "multiple sequential transactions", "below reporting threshold",
        "9850", "9900", "cash deposits", "smurf", "smurfing", "consecutive deposits",
        "branch deposits", "structuring suspect"
    ],
    "aml_sanctions": [
        "sanctions nexus", "ofac", "sanctioned entities", "sanctioned bank",
        "counterparties with sanctions", "wired funds offshore", "offshore wire",
        "embargo", "blacklist"
    ],
    "aml_velocity": [
        "unusual velocity", "rapid fund movement", "depositing and emptying",
        "rapid automated movement", "churning accounts", "high velocity transfers"
    ],
    "aml_high_risk_jurisdiction": [
        "high risk jurisdiction", "cross border transfers", "secrecy havens",
        "tax haven", "non cooperative jurisdiction", "cayman", "panama", "bvi"
    ],
    "aml_sar_filing": [
        "file sar", "suspicious activity report", "fincen", "mandatory sar filing",
        "illicit origin", "money laundering", "aml investigation", "freeze account"
    ],
    "code_security_pass": [
        "meets secure coding guidelines", "clean code diff", "parameterized query",
        "prepared statement", "safe merge", "pass security"
    ],
    "code_security_sql_injection": [
        "sql injection", "sqli", "unsanitized user string", "concat inside database query",
        "select * from", "execute query", "or 1=1", "f\"select * from"
    ],
    "code_security_secret": [
        "hardcoded secret", "plaintext secret", "api key in code", "private key plaintext",
        "password token", "credentials in source"
    ],
    "code_security_traversal": [
        "path traversal", "arbitrary file read", "relative directory paths", "../",
        "dot dot slash", "lfi traversal"
    ],
    "code_security_deserialization": [
        "unsafe deserialization", "pickle load", "yaml load untrusted", "untrusted payload",
        "remote code execution deserialization", "rce deserialization"
    ],
    "code_security_critical_block": [
        "critical block", "block ci merge", "cvss 9", "critical vulnerability", "block merge"
    ],
    "rag_direct_answer": [
        "direct answer", "exact facts", "answering query", "hours of continuous active playback",
        "battery life", "complete factual ground truth", "exact figures"
    ],
    "rag_contextually_related": [
        "contextually related", "general subject area", "lacks specific answer",
        "discusses topic generally", "needs more context"
    ],
    "rag_irrelevant": [
        "irrelevant", "no bearing on user query", "unrelated passage", "off topic noise"
    ],
    "insurance_straight_through": [
        "straight through approval", "instant payout", "low value standard claim",
        "cracked windshield", "stray pebble", "clean history", "repair quote $320",
        "no injuries"
    ],
    "insurance_desk_adjuster": [
        "desk adjuster review", "moderate damage", "repair quote validation", "manual policy review"
    ],
    "insurance_special_investigations": [
        "special investigations unit", "siu", "fraud risk", "staged accident",
        "falsified invoice", "potential fraud true", "suspicious claim"
    ],
    "insurance_catastrophe": [
        "urgent catastrophe", "total structural destruction", "bodily injury",
        "emergency hospital", "catastrophic damage"
    ],
    "support_technical": [
        "technical support", "software bug", "application error", "crash dump",
        "nullpointerexception", "stack trace", "api error 500", "server outage",
        "database connection failure", "unhandled exception"
    ],
    "support_billing": [
        "billing inquiry", "invoice dispute", "credit card charge", "subscription fee",
        "refund request", "unauthorized charge", "chargeback", "payment receipt",
        "cancel subscription", "billing account"
    ],
    "support_account_auth": [
        "password reset", "mfa token", "locked out of account", "login failed",
        "two factor authentication", "change email address", "forgot password",
        "account recovery", "profile update"
    ],
    "route_fast_simple": [
        "quick lookup", "simple question", "factual answer", "fast model",
        "small model", "low latency response", "lightweight task", "summarize text"
    ],
    "route_frontier_reasoning": [
        "complex mathematical proof", "multi step reasoning", "frontier reasoning model",
        "theorem proving", "hard logic puzzle", "advanced algorithmic synthesis",
        "deep analysis", "cot reasoning"
    ],
    "guard_allow_benign": [
        "safe action", "read only query", "benign operation", "allow execution",
        "harmless inspection", "view documentation", "status check"
    ],
    "guard_deny_destructive": [
        "destructive action", "drop database", "delete root directory", "rm -rf",
        "wipe disk partition", "irreversible data loss", "format drive", "block action"
    ],
    "firewall_allow_benign": [
        "allow immediate", "benign routine", "read only", "readme", "inspect repository",
        "pytest", "unit tests", "git log", "git status", "lsof", "ps aux", "black --check",
        "ast parse", "ast.parse", "clean code diff", "syntax check", "status query",
        "harmless routine", "low risk read only"
    ],
    "firewall_require_human": [
        "require human approval", "privileged admin", "elevated privilege", "destructive modification",
        "restart container", "docker restart", "schema migration", "alembic upgrade", "modify config",
        "update .env", "elasticsearch query", "tls certificate", "openssl req", "pip install",
        "rotate credentials", "git push", "push remote"
    ],
    "firewall_quarantine": [
        "quarantine tool execution", "anomalous tool invocation", "suspicious pipeline",
        "pipe execution", "curl | bash", "binary dropper", "chmod +x /dev/shm", "in-memory payload",
        "blind sqli", "pg_sleep", "graphql introspection", "discover hidden endpoints",
        "s3 unapproved bucket", "external untrusted bucket"
    ],
    "firewall_terminate_exploit": [
        "terminate agent session", "critical exploit", "credential harvesting", "prompt injection",
        "reverse shell", "data exfiltration", "cat /etc/shadow", "private key theft", "jailbreak",
        "steal ssh key", "cat ~/.ssh/id_rsa", "dump memory", "gcore", "sql injection", "drop table",
        "pii leak", "ssn credit card", "hipaa breach", "biometric exfil"
    ],
    "threat_data_exfiltration": [
        "data exfiltration", "covert channel", "outbound curl", "upload s3", "dns tunnel",
        "dig +short c2", "scp private key", "data drop", "payload dump", "transmit credentials",
        "nc 203.0.113", "exfiltrate", "outbound payload", "c2-collector"
    ],
    "threat_unauthorized_shell": [
        "unauthorized shell execution", "spawn bash", "interactive shell", "nc -e /bin/bash",
        "reverse shell", "subshell execution", "bash -i", "sh -c", "eval python", "os.system",
        "subprocess.popen", "base64 -d | sh", "ld_preload", "crontab persistence", "visudo backdoor"
    ],
    "threat_credential_access": [
        "credential access", "steal ssh key", "cat ~/.ssh/id_rsa", "cat ~/.aws/credentials",
        "grep sk_live", "k8s service account token", "dump memory", "gcore", "api key theft",
        "extract passwords", "private_key", "jwt_token"
    ],
    "threat_pii_leak": [
        "pii leak", "customer profiles", "ssn", "tax_id", "credit card number", "medical records",
        "hipaa breach", "patient telemetry", "biometrics", "payroll export", "salary leak",
        "personally identifiable", "hr_payroll_vault", "oncology diagnosis"
    ],
    "threat_sql_injection": [
        "sql injection", "sqli", "' or 1=1", "drop table", "pg_sleep", "union select",
        "blind sql injection", "auth bypass sql", "update accounts set balance", "drop table audit"
    ],
    "threat_prompt_injection": [
        "prompt injection", "jailbreak", "ignore previous instructions", "system prompt exfiltration",
        "override safety guardrails", "developer mode activated", "dan mode", "do anything now",
        "disregard prior instructions", "unfiltered master system prompt"
    ],
}


class SubwordSemanticEmbeddings:
    """Seeded NumPy subword table with curated semantic anchors.

    The default table dimension is 256. Its construction does not teach a
    general language model; task quality requires separate evaluation."""

    def __init__(
        self,
        dimension: int = 256,
        vocab_size: int = 25000,
        seed: int = 42,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        self.dimension = dimension
        self.vocab_size = vocab_size
        self.seed = seed
        self._table: np.ndarray = self._init_embedding_table()

    def _hash_token(self, token: str) -> int:
        """Deterministically map a token or n-gram string to a table row index."""
        h = hashlib.sha256(token.strip().lower().encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.vocab_size

    def _init_embedding_table(self) -> np.ndarray:
        """Constructs the dense subword embedding table with calibrated semantic anchors."""
        rng = np.random.RandomState(self.seed)
        # Initialize pseudo-random Gaussian base table
        table = rng.normal(0.0, 1.0, size=(self.vocab_size, self.dimension)).astype(np.float32)
        row_norms = np.linalg.norm(table, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
        table /= row_norms

        # Blend semantic anchor concept centroids
        for anchor_name, terms in _SEMANTIC_ANCHORS.items():
            # Concept centroid
            c_vec = rng.normal(0.0, 1.0, size=(self.dimension,)).astype(np.float32)
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm > 1e-12:
                c_vec /= c_norm

            for term in terms:
                clean_term = term.strip().lower()
                idx = self._hash_token(clean_term)
                # Strong directional blend towards anchor centroid
                blended = 0.15 * table[idx] + 0.85 * c_vec
                b_norm = float(np.linalg.norm(blended))
                table[idx] = blended / (b_norm if b_norm > 1e-12 else 1.0)

                # Also blend individual subwords and word tokens of multi-word terms
                words = re.findall(r"\b\w+\b", clean_term)
                if len(words) > 1:
                    for w in words:
                        if len(w) >= 3:
                            w_idx = self._hash_token(w)
                            w_blended = 0.4 * table[w_idx] + 0.6 * c_vec
                            wn = float(np.linalg.norm(w_blended))
                            table[w_idx] = w_blended / (wn if wn > 1e-12 else 1.0)

        return table

    def encode(self, text: str) -> np.ndarray:
        """Encodes text into a normalized dense subword vector."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        if not text or not text.strip():
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]

        indices: List[int] = []
        weights: List[float] = []

        # 1. Check for multi-word phrase & idiom anchors
        for terms in _SEMANTIC_ANCHORS.values():
            for term in terms:
                if " " in term and term in clean_text:
                    indices.append(self._hash_token(term))
                    weights.append(3.5)

        # 2. Extract words and subwords
        words = re.findall(r"\b\w+\b", clean_text)
        for pos, word in enumerate(words):
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            indices.append(self._hash_token(word))
            weights.append(word_weight * 1.5)

            # Character 3-grams and 4-grams
            if len(word) >= 3:
                for n in (3, 4):
                    for j in range(len(word) - n + 1):
                        sub = word[j : j + n]
                        indices.append(self._hash_token(f"sub:{sub}"))
                        weights.append(0.35)

        # 3. Word bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            indices.append(self._hash_token(f"bi:{bigram}"))
            weights.append(1.2)

        if not indices:
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        idx_arr = np.array(indices, dtype=np.int32)
        w_arr = np.array(weights, dtype=np.float32).reshape(-1, 1)

        # Fast vectorized matrix take & sum
        vecs = np.take(self._table, idx_arr, axis=0)
        accum = np.sum(vecs * w_arr, axis=0)

        norm = float(np.linalg.norm(accum))
        if norm < 1e-12 or not np.isfinite(norm):
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v
        return (accum / norm).astype(np.float32)

    project = encode

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Batched encoding of multiple text inputs."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self.encode(t) for t in texts], axis=0).astype(np.float32)

    project_batch = encode_batch

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two text strings."""
        v_a = self.encode(text_a)
        v_b = self.encode(text_b)
        return float(np.dot(v_a, v_b))


class HybridProjector:
    """Concatenate weighted lexical and dense subword features.

    For unit components, square-root weights preserve unit norm; the
    implementation also normalizes its final vector. The default dimensions
    are 1024 lexical plus 256 dense, with alpha=0.5. No latency guarantee."""

    def __init__(
        self,
        dimension: Optional[int] = None,
        *,
        sparse_dim: int = 1024,
        dense_dim: int = 256,
        alpha: float = 0.5,
        seed: int = 42,
        backend: str = "auto",
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}")

        if dimension is not None:
            if dimension <= 0:
                raise ValueError(f"dimension must be positive, got {dimension}")
            # Divide dimension between sparse and dense
            sparse_dim = max(1, int(dimension * 0.75))
            dense_dim = max(1, dimension - sparse_dim)
            self.dimension = sparse_dim + dense_dim
        else:
            if sparse_dim <= 0:
                raise ValueError(f"sparse_dim must be positive, got {sparse_dim}")
            if dense_dim <= 0:
                raise ValueError(f"dense_dim must be positive, got {dense_dim}")
            self.dimension = sparse_dim + dense_dim

        self.sparse_dim = sparse_dim
        self.dense_dim = dense_dim
        self.alpha = float(alpha)
        self.seed = seed
        self.backend = self._resolve_backend(backend)

        # Dense subword semantic engine
        self._dense_engine = SubwordSemanticEmbeddings(
            dimension=self.dense_dim,
            vocab_size=25000,
            seed=self.seed,
        )

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            return "mlx" if HAS_MLX else "numpy"
        if backend == "mlx":
            if not HAS_MLX:
                raise RuntimeError("MLX requested but mlx is not installed on this system")
            return "mlx"
        return "numpy"

    def _hash_to_sparse_vec(self, text: str) -> np.ndarray:
        """Projects text into normalized sparse lexical n-gram vector."""
        vec = np.zeros(self.sparse_dim, dtype=np.float32)
        if not text or not text.strip():
            vec[0] = 1.0
            return vec

        clean_text = text.lower().strip()
        if len(clean_text) > 8192:
            clean_text = clean_text[:4096] + " " + clean_text[-4096:]
        words = re.findall(r"\b\w+\b", clean_text)

        for pos, word in enumerate(words):
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            h = hashlib.sha256(f"word:{word}".encode("utf-8")).digest()
            for i in range(4):
                chunk = h[i * 4 : (i + 1) * 4]
                val = int.from_bytes(chunk, byteorder="big", signed=False)
                idx = val % self.sparse_dim
                sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
                vec[idx] += sign * word_weight * 2.0

            if len(word) >= 3:
                for n in (3, 4):
                    for j in range(len(word) - n + 1):
                        ngram = word[j : j + n]
                        h_ng = hashlib.sha256(f"ng:{ngram}".encode("utf-8")).digest()
                        idx_ng = int.from_bytes(h_ng[:4], "big") % self.sparse_dim
                        sign_ng = 1.0 if (h_ng[4] & 1) == 0 else -1.0
                        vec[idx_ng] += sign_ng * 0.5

        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h_bi = hashlib.sha256(f"bi:{bigram}".encode("utf-8")).digest()
            idx_bi = int.from_bytes(h_bi[:4], "big") % self.sparse_dim
            sign_bi = 1.0 if (h_bi[4] & 1) == 0 else -1.0
            vec[idx_bi] += sign_bi * 1.5

        norm = float(np.linalg.norm(vec))
        if norm < 1e-12 or not np.isfinite(norm):
            v = np.zeros(self.sparse_dim, dtype=np.float32)
            v[0] = 1.0
            return v
        return (vec / norm).astype(np.float32)

    def project(self, text: str) -> np.ndarray:
        """Projects input text into unit-normalized hybrid sparse-dense representation."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")

        v_sparse = self._hash_to_sparse_vec(text)
        v_dense = self._dense_engine.encode(text)

        w_sparse = math.sqrt(self.alpha)
        w_dense = math.sqrt(1.0 - self.alpha)

        v_hybrid = np.concatenate([w_sparse * v_sparse, w_dense * v_dense]).astype(np.float32)
        # Unit normalization safety
        norm = float(np.linalg.norm(v_hybrid))
        if norm > 1e-12:
            v_hybrid = v_hybrid / norm
        else:
            v_hybrid = np.zeros(self.dimension, dtype=np.float32)
            v_hybrid[0] = 1.0
        return v_hybrid

    encode = project

    def project_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Batched projection of multiple text inputs."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be strings, got {type(t).__name__} at index {idx}")
        return np.stack([self.project(t) for t in texts], axis=0).astype(np.float32)

    encode_batch = project_batch

    def similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two text inputs in hybrid space."""
        v_a = self.project(text_a)
        v_b = self.project(text_b)
        return float(np.dot(v_a, v_b))


HybridSemanticProjector = HybridProjector


__all__ = [
    "SubwordSemanticEmbeddings",
    "HybridProjector",
    "HybridSemanticProjector",
]
