import re
import os

typesafe_file = 'src/system1/compat/typesafe.py'
with open(typesafe_file, 'r') as f:
    typesafe_content = f.read()

# 1. Fix _distill_and_cutover_locked to copy calibration
target1 = """        if calib_dataset:
            try:
                engine.calibrate(calib_dataset, n_bins=min(5, max(2, len(calib_dataset))))
            except Exception:
                pass"""

replacement1 = """        if calib_dataset:
            try:
                engine.calibrate(calib_dataset, n_bins=min(5, max(2, len(calib_dataset))))
                for f_name, ch in compiled_model.heads.items():
                    if f_name in engine.calibrators:
                        ch.temperature = engine.calibrators[f_name].temperature
                    if f_name in engine.conformal_predictors:
                        cp = engine.conformal_predictors[f_name]
                        ch.calibration_scores = tuple(cp.calibration_scores) if getattr(cp, "calibration_scores", None) is not None else ()
                    elif f_name in engine.regression_conformal_predictors:
                        rcp = engine.regression_conformal_predictors[f_name]
                        ch.calibration_scores = tuple(rcp.residuals) if getattr(rcp, "residuals", None) is not None else ()
            except Exception:
                pass"""

if target1 in typesafe_content:
    typesafe_content = typesafe_content.replace(target1, replacement1)
    print("Fixed calibration sync in typesafe.py")
else:
    print("Could not find calibration target in typesafe.py")

# 2. Fix evaluate_promotion_eligibility counting
target2 = """    for g_key, g_items in groups.items():
        unit_has_scored_evidence = False

        for item in g_items:
            state = item.get("state", "")
            answers = item.get("answers", {})
            if not answers:
                continue

            res = engine.decide(state, record_receipt=False)

            for f_name, f_def in schema.fields.items():
                if f_name not in answers or answers[f_name] is None:
                    continue

                target_v = answers[f_name]
                pred_v = res.values.get(f_name)

                total_checks += 1
                per_field_stats[f_name]["total"] += 1
                unit_has_scored_evidence = True

                is_target_critical = _is_critical_class(target_v, policy.critical_classes)
                if is_target_critical:
                    critical_targets += 1
                    per_field_stats[f_name]["critical_targets"] += 1

                # Check matching correctness
                is_match = False
                if (
                    isinstance(pred_v, (int, float))
                    and isinstance(target_v, (int, float))
                    and not isinstance(pred_v, bool)
                    and not isinstance(target_v, bool)
                ):
                    margin = max(0.5, 0.20 * (getattr(f_def, "max_value", 1.0) - getattr(f_def, "min_value", 0.0))) if f_def else 0.5
                    if abs(float(pred_v) - float(target_v)) <= margin:
                        is_match = True
                elif getattr(f_def, "__class__", None) is not None and getattr(f_def.__class__, "__name__", "") == "MultiChoiceField":
                    s_pred = set(pred_v) if isinstance(pred_v, (list, tuple, set)) else ({pred_v} if pred_v else set())
                    s_target = set(target_v) if isinstance(target_v, (list, tuple, set)) else ({target_v} if target_v else set())
                    if s_pred == s_target or (s_pred and s_target and len(s_pred & s_target) / len(s_pred | s_target) >= 0.5):
                        is_match = True
                elif isinstance(f_def, MultiChoiceField):
                    s_pred = set(pred_v) if isinstance(pred_v, (list, tuple, set)) else ({pred_v} if pred_v else set())
                    s_target = set(target_v) if isinstance(target_v, (list, tuple, set)) else ({target_v} if target_v else set())
                    if s_pred == s_target or (s_pred and s_target and len(s_pred & s_target) / len(s_pred | s_target) >= 0.5):
                        is_match = True
                elif pred_v == target_v:
                    is_match = True

                if is_match:
                    matching += 1
                    per_field_stats[f_name]["matching"] += 1
                else:
                    if is_target_critical and _is_allow_class(pred_v):
                        false_allows += 1
                        per_field_stats[f_name]["false_allows"] += 1

        if unit_has_scored_evidence:
            scored_units += 1"""

replacement2 = """    for g_key, g_items in groups.items():
        unit_has_scored_evidence = False
        unit_total = 0
        unit_matching = 0
        unit_false_allows = 0
        unit_critical = 0

        for item in g_items:
            state = item.get("state", "")
            answers = item.get("answers", {})
            if not answers:
                continue

            res = engine.decide(state, record_receipt=False)

            for f_name, f_def in schema.fields.items():
                if f_name not in answers or answers[f_name] is None:
                    continue

                target_v = answers[f_name]
                pred_v = res.values.get(f_name)

                unit_total += 1
                per_field_stats[f_name]["total"] += 1
                unit_has_scored_evidence = True

                is_target_critical = _is_critical_class(target_v, policy.critical_classes)
                if is_target_critical:
                    unit_critical += 1
                    per_field_stats[f_name]["critical_targets"] += 1

                # Check matching correctness
                is_match = False
                if (
                    isinstance(pred_v, (int, float))
                    and isinstance(target_v, (int, float))
                    and not isinstance(pred_v, bool)
                    and not isinstance(target_v, bool)
                ):
                    margin = max(0.5, 0.20 * (getattr(f_def, "max_value", 1.0) - getattr(f_def, "min_value", 0.0))) if f_def else 0.5
                    if abs(float(pred_v) - float(target_v)) <= margin:
                        is_match = True
                elif getattr(f_def, "__class__", None) is not None and getattr(f_def.__class__, "__name__", "") == "MultiChoiceField":
                    s_pred = set(pred_v) if isinstance(pred_v, (list, tuple, set)) else ({pred_v} if pred_v else set())
                    s_target = set(target_v) if isinstance(target_v, (list, tuple, set)) else ({target_v} if target_v else set())
                    if s_pred == s_target or (s_pred and s_target and len(s_pred & s_target) / len(s_pred | s_target) >= 0.5):
                        is_match = True
                elif isinstance(f_def, MultiChoiceField):
                    s_pred = set(pred_v) if isinstance(pred_v, (list, tuple, set)) else ({pred_v} if pred_v else set())
                    s_target = set(target_v) if isinstance(target_v, (list, tuple, set)) else ({target_v} if target_v else set())
                    if s_pred == s_target or (s_pred and s_target and len(s_pred & s_target) / len(s_pred | s_target) >= 0.5):
                        is_match = True
                elif pred_v == target_v:
                    is_match = True

                if is_match:
                    unit_matching += 1
                    per_field_stats[f_name]["matching"] += 1
                else:
                    if is_target_critical and _is_allow_class(pred_v):
                        unit_false_allows += 1
                        per_field_stats[f_name]["false_allows"] += 1

        if unit_has_scored_evidence:
            scored_units += 1
            total_checks += 1
            if unit_matching == unit_total:
                matching += 1
            false_allows += unit_false_allows
            critical_targets += unit_critical"""

if target2 in typesafe_content:
    typesafe_content = typesafe_content.replace(target2, replacement2)
    print("Fixed evaluate_promotion_eligibility loop in typesafe.py")
else:
    print("Could not find loop target in typesafe.py")
    
with open(typesafe_file, 'w') as f:
    f.write(typesafe_content)
