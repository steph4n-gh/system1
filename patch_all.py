import re
import os

# 1. Fix engine.py
engine_file = 'src/system1/engine.py'
with open(engine_file, 'r') as f:
    engine_content = f.read()

target1 = """            if self.strict_mode:
                for f_name in updated_fields:
                    if f_name in self.calibrators:
                        self.calibrators[f_name].is_calibrated = False
                    if f_name in self.conformal_predictors:
                        self.conformal_predictors[f_name].is_calibrated = False"""

replacement1 = """            for f_name in updated_fields:
                if f_name in self.calibrators:
                    self.calibrators[f_name].is_calibrated = False
                if f_name in self.conformal_predictors:
                    self.conformal_predictors[f_name].is_calibrated = False
                if f_name in self.regression_conformal_predictors:
                    self.regression_conformal_predictors[f_name].is_calibrated = False"""

if target1 in engine_content:
    engine_content = engine_content.replace(target1, replacement1)
    with open(engine_file, 'w') as f:
        f.write(engine_content)
    print("Fixed engine.py")
else:
    print("Could not find target in engine.py")

# 2. Fix compiler.py
compiler_file = 'src/system1/compiler.py'
with open(compiler_file, 'r') as f:
    compiler_content = f.read()

target2 = """            if hasattr(ch, "calibration_scores") and len(ch.calibration_scores) > 0:
                arrays_to_save[f"{name}_calib_scores"] = np.asarray(ch.calibration_scores, dtype=np.float32)"""
replacement2 = """            if hasattr(ch, "calibration_scores") and len(ch.calibration_scores) > 0:
                arrays_to_save[f"{name}_calib_scores"] = np.asarray(ch.calibration_scores, dtype=np.float64)"""

if target2 in compiler_content:
    compiler_content = compiler_content.replace(target2, replacement2)
    with open(compiler_file, 'w') as f:
        f.write(compiler_content)
    print("Fixed compiler.py")
else:
    print("Could not find target in compiler.py")

# 3. Fix typesafe.py export_model & evaluate_promotion_eligibility
typesafe_file = 'src/system1/compat/typesafe.py'
with open(typesafe_file, 'r') as f:
    typesafe_content = f.read()

target3 = """    def export_model(self, path: Union[str, Path]) -> bool:
        \"\"\"Compiles and exports the validated System 1 model to disk.\"\"\"
        cm = self.compiled_model
        if cm is None:
            return False
            
        digest = cm.schema.schema_digest()
        with self._engine_lock:
            engine = self._engine_cache.get(digest)
            if engine is not None:
                for f_name, ch in cm.heads.items():
                    if f_name in engine.calibrators:
                        ch.temperature = engine.calibrators[f_name].temperature
                    if f_name in engine.conformal_predictors:
                        cp = engine.conformal_predictors[f_name]
                        ch.calibration_scores = tuple(cp.calibration_scores) if getattr(cp, "calibration_scores", None) is not None else ()
                    elif f_name in engine.regression_conformal_predictors:
                        rcp = engine.regression_conformal_predictors[f_name]
                        ch.calibration_scores = tuple(rcp.residuals) if getattr(rcp, "residuals", None) is not None else ()
                        
        cm.save(path)
        return True"""

replacement3 = """    def export_model(self, path: Union[str, Path]) -> bool:
        \"\"\"Compiles and exports the validated System 1 model to disk.\"\"\"
        cm = self.compiled_model
        if cm is None:
            return False
            
        cm.save(path)
        return True"""

if target3 in typesafe_content:
    typesafe_content = typesafe_content.replace(target3, replacement3)
    print("Fixed export_model in typesafe.py")
else:
    print("Could not find target3 in typesafe.py")

target4 = """    groups = collections.defaultdict(list)
    for item in val_history:
        g_key = item.get("group_id") or item.get("lineage_id") or item.get("request_id") or item.get("id") or item.get("nonce") or hash(item.get("state", ""))
        groups[str(g_key)].append(item)"""

replacement4 = """    adj = collections.defaultdict(set)
    for i, item in enumerate(val_history):
        keys = []
        for k in ["lineage_id", "group_id", "request_id", "id", "nonce"]:
            if item.get(k):
                keys.append(f"{k}:{item[k]}")
        state = item.get("state")
        if state:
            keys.append(f"state:{state}")
        keys.append(f"row:{i}")
        for k1 in keys:
            for k2 in keys:
                adj[k1].add(k2)

    visited = set()
    components = []
    for node in adj:
        if node not in visited:
            comp = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    stack.extend(adj[curr])
            components.append(comp)

    groups = collections.defaultdict(list)
    for i, item in enumerate(val_history):
        row_key = f"row:{i}"
        for c_idx, comp in enumerate(components):
            if row_key in comp:
                groups[str(c_idx)].append(item)
                break"""

if target4 in typesafe_content:
    typesafe_content = typesafe_content.replace(target4, replacement4)
    print("Fixed grouped validation components in typesafe.py")
else:
    print("Could not find target4 in typesafe.py")

target5 = """    if len(val_history) < policy.min_validation_samples:
        rejection_reasons.append(
            f"Validation sample count ({len(val_history)}) below minimum threshold ({policy.min_validation_samples})"
        )"""

replacement5 = """    if scored_units < policy.min_validation_samples:
        rejection_reasons.append(
            f"Validation sample count ({scored_units}) below minimum threshold ({policy.min_validation_samples})"
        )"""

if target5 in typesafe_content:
    typesafe_content = typesafe_content.replace(target5, replacement5)
    print("Fixed minimum samples check in typesafe.py")
else:
    print("Could not find target5 in typesafe.py")

with open(typesafe_file, 'w') as f:
    f.write(typesafe_content)

# 4. Fix ledger.py
ledger_file = 'src/system1/ledger.py'
with open(ledger_file, 'r') as f:
    ledger_content = f.read()

target6 = """    def _verify_integrity_locked(self, connection: sqlite3.Connection, trusted_public_key: Optional[Any] = None) -> bool:
        \"\"\"Internal helper to verify full cryptographic chain integrity under a coherent transaction/snapshot.\"\"\"
        entries = connection.execute(
            "SELECT * FROM audit_entries ORDER BY sequence ASC"
        ).fetchall()
        previous = _ZERO_HASH
        last_sequence = 0

        for row in entries:"""

replacement6 = """    def _verify_integrity_locked(self, connection: sqlite3.Connection, trusted_public_key: Optional[Any] = None) -> bool:
        \"\"\"Internal helper to verify full cryptographic chain integrity under a coherent transaction/snapshot.\"\"\"
        if not hasattr(self, "_verified_sequence"):
            self._verified_sequence = 0
            self._verified_hash = _ZERO_HASH
            
        if self._verified_sequence > 0:
            row = connection.execute(
                "SELECT entry_hash FROM audit_entries WHERE sequence = ?", (self._verified_sequence,)
            ).fetchone()
            if not row or row["entry_hash"] != self._verified_hash:
                self._verified_sequence = 0
                self._verified_hash = _ZERO_HASH
                entries = connection.execute("SELECT * FROM audit_entries ORDER BY sequence ASC").fetchall()
            else:
                entries = connection.execute("SELECT * FROM audit_entries WHERE sequence > ? ORDER BY sequence ASC", (self._verified_sequence,)).fetchall()
        else:
            entries = connection.execute("SELECT * FROM audit_entries ORDER BY sequence ASC").fetchall()

        previous = self._verified_hash
        last_sequence = self._verified_sequence

        for row in entries:"""

if target6 in ledger_content:
    ledger_content = ledger_content.replace(target6, replacement6)
    print("Fixed _verify_integrity_locked in ledger.py")
else:
    print("Could not find target6 in ledger.py")

target7 = """        meta = dict(connection.execute("SELECT key, value FROM ledger_meta").fetchall())
        if meta.get("audit_head_hash") != previous:
            return False
        if int(meta.get("audit_head_sequence", "-1")) != last_sequence:
            return False

        return True"""

replacement7 = """        meta = dict(connection.execute("SELECT key, value FROM ledger_meta").fetchall())
        if meta.get("audit_head_hash") != previous:
            return False
        if int(meta.get("audit_head_sequence", "-1")) != last_sequence:
            return False

        self._verified_sequence = last_sequence
        self._verified_hash = previous
        return True"""

if target7 in ledger_content:
    ledger_content = ledger_content.replace(target7, replacement7)
    print("Fixed _verify_integrity_locked cache updates in ledger.py")
else:
    print("Could not find target7 in ledger.py")

target8 = """            if found_digest and found_digest != receipt_digest:
                raise LedgerWriteError(f"Wrong action association: outcome receipt_digest {receipt_digest} does not match prior {found_digest}")"""

replacement8 = """            if not found_digest:
                raise ValueError("Prior authorization lacks a receipt digest")
            if found_digest != receipt_digest:
                raise LedgerWriteError(f"Wrong action association: outcome receipt_digest {receipt_digest} does not match prior {found_digest}")"""

if target8 in ledger_content:
    ledger_content = ledger_content.replace(target8, replacement8)
    print("Fixed record_execution_outcome in ledger.py")
else:
    print("Could not find target8 in ledger.py")

with open(ledger_file, 'w') as f:
    f.write(ledger_content)
