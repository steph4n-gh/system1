import os

typesafe_file = 'src/system1/compat/typesafe.py'
with open(typesafe_file, 'r') as f:
    typesafe_content = f.read()

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
            scored_units += 1

    if total_checks == 0:
        agreement_rate = 0.0
        rejection_reasons.append("Zero scored validation checks; cannot evaluate promotion")
    else:
        agreement_rate = matching / total_checks"""

replacement2 = """    for g_key, g_items in groups.items():
        unit_has_scored_evidence = False
        unit_total = 0
        unit_matching = 0

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
                unit_total += 1
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
                    unit_matching += 1
                    per_field_stats[f_name]["matching"] += 1
                else:
                    if is_target_critical and _is_allow_class(pred_v):
                        false_allows += 1
                        per_field_stats[f_name]["false_allows"] += 1

        if unit_has_scored_evidence:
            scored_units += 1
            if unit_matching == unit_total:
                unit_matches += 1

    if scored_units == 0:
        agreement_rate = 0.0
        rejection_reasons.append("Zero scored validation checks; cannot evaluate promotion")
    else:
        agreement_rate = unit_matches / scored_units"""

if target2 in typesafe_content:
    typesafe_content = typesafe_content.replace(target2, replacement2)
    with open(typesafe_file, 'w') as f:
        f.write(typesafe_content)
    print("Fixed evaluate_promotion_eligibility counting units in typesafe.py")
else:
    print("Could not find loop target in typesafe.py")
