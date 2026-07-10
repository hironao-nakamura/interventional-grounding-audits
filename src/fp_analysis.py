"""False Positive analysis for the main configuration (A consistent only).

For each FP certificate (verdict=GROUNDED but not in proof tree),
classify the cause into one of 5 types.
"""

import re
from collections import Counter, defaultdict


def classify_fp(cert, problem):
    """Classify a single false positive certificate.

    Uses the certificate's `matched_proof_step_id` (normalized-conclusion
    alignment) for proof-tree lookups; the raw `step_id` is a model CoT
    step number and must not be compared against proof-tree steps.
    """
    phi_orig = cert.get("phi_original", "") or ""
    phi_sem = cert.get("phi_semantic", "") or ""
    step_id = cert.get("matched_proof_step_id")
    prem_id = cert.get("premise_id", "P0")

    # --- Type 3: NORMALIZER_ERROR ---
    if phi_orig and phi_sem:
        orig_c = re.sub(r"\s+", "", phi_orig.lower())
        sem_c = re.sub(r"\s+", "", phi_sem.lower())
        if orig_c == sem_c:
            return "NORMALIZER_ERROR", (
                f"Normalized forms identical after cleanup: "
                f"'{phi_orig}' vs '{phi_sem}'"
            )

    # --- Type 1: STOCHASTIC (predicate change unrelated to substitution) ---
    if phi_orig and phi_sem:
        # Find the premise that was probed
        premise_text = ""
        for p in problem["premises"]:
            if p["id"] == prem_id:
                premise_text = p["text"]
                break

        # Extract predicates from orig and sem
        def extract_preds(form):
            m = re.match(r"(?:is|not_is|subtype|not_subtype)\((\w+),\s*(\w+)\)", form)
            if m:
                return m.group(1), m.group(2)
            return None, None

        subj_o, obj_o = extract_preds(phi_orig)
        subj_s, obj_s = extract_preds(phi_sem)

        if subj_o and subj_s:
            if subj_o == subj_s and obj_o != obj_s:
                # The object predicate changed
                changed_pred = obj_s
                # Does the changed predicate look like a proper substitution (zq prefix)?
                if changed_pred and changed_pred.startswith("zq"):
                    return "EXPECTED_SUBSTITUTION", (
                        f"Substituted predicate propagated correctly: "
                        f"'{obj_o}' -> '{changed_pred}'"
                    )
                else:
                    return "STOCHASTIC", (
                        f"Predicate changed from '{obj_o}' to '{obj_s}' "
                        f"without zq prefix — likely stochastic"
                    )
            elif subj_o != subj_s:
                return "STOCHASTIC", (
                    f"Subject changed from '{subj_o}' to '{subj_s}' — stochastic"
                )

    # --- Type 4: INDIRECT_DEPENDENCY (cascade from consistent substitution) ---
    # In ProntoQA, step N typically depends on premise N (chain) + entity premise.
    # If the FP is for a premise that is earlier in the chain, it could be
    # because the substitution propagated through the chain.
    proof_tree = problem.get("proof_tree", [])
    step_entry = None
    if step_id is not None:
        step_entry = next((e for e in proof_tree if e["step"] == step_id), None)
    if step_entry:
        direct_deps = [d for d in step_entry["depends_on"] if d.startswith("P")]
        if prem_id not in direct_deps:
            # This step doesn't directly depend on this premise
            # Check if ANY upstream step depends on this premise
            upstream_depends = False
            for e in proof_tree:
                if e["step"] < step_id:
                    if prem_id in [d for d in e["depends_on"] if d.startswith("P")]:
                        upstream_depends = True
                        break
            if upstream_depends:
                return "INDIRECT_DEPENDENCY", (
                    f"Step {step_id} doesn't directly depend on {prem_id}, "
                    f"but an upstream step does — substitution propagated through chain"
                )

    return "UNKNOWN", "Could not automatically classify"


def analyze_false_positives(certificates, problems, cot_steps_by_pid):
    """Analyze semantic-delta false-positive candidates under A consistent.

    Two counts are reported and must not be conflated in the paper:
      * semantic-delta candidates: evaluable certificates with
        semantic_delta=True whose aligned GT is False (the analysis set);
      * metric-counted FPs: the subset that the Table-1 A-consistent
        evaluator actually counts as FP (candidates whose surface probe is
        unparseable fall outside the Table-1 metric set).
    """
    from src.ground_truth import (
        POSITIVE_VERDICTS,
        PRIMARY_SKIP_VERDICTS,
        extract_ground_truth_dependencies_for_cot_steps,
    )

    prob_map = {p["problem_id"]: p for p in problems}

    gt_map = {}
    for p in problems:
        gt, _ = extract_ground_truth_dependencies_for_cot_steps(
            p, cot_steps_by_pid.get(p["problem_id"], []))
        for (sid, pid), dep in gt.items():
            gt_map[(p["problem_id"], sid, pid)] = dep

    def judge_consistent_only(cert):
        if not (cert.get("parse_ok_orig") and cert.get("parse_ok_sem")
                and cert.get("parse_ok_sur")):
            return "UNPARSEABLE"
        sem = cert.get("semantic_delta", False)
        sur = cert.get("surface_delta", False)
        if sem and not sur:
            return "GROUNDED"
        if sem and sur:
            return "INPUT-SENSITIVE"
        if not sem and sur:
            return "UNSTABLE"
        return "INSENSITIVE"

    candidates = []
    for cert in certificates:
        if cert.get("evaluation_excluded"):
            continue
        if not (cert.get("parse_ok_orig") and cert.get("parse_ok_sem")):
            continue
        if not cert.get("semantic_delta"):
            continue
        key = (cert["problem_id"], cert["step_id"], cert["premise_id"])
        gt_dep = gt_map.get(key)
        if gt_dep is None or gt_dep:
            continue
        candidates.append(cert)

    classifications = []
    n_metric_fp = 0
    for fp in candidates:
        pid = fp["problem_id"]
        problem = prob_map.get(pid)
        if not problem:
            continue
        verdict = judge_consistent_only(fp)
        metric_counted = (verdict in POSITIVE_VERDICTS
                          and verdict not in PRIMARY_SKIP_VERDICTS)
        if metric_counted:
            n_metric_fp += 1
        fp_type, reason = classify_fp(fp, problem)
        classifications.append({
            "problem_id": pid,
            "step_id": fp["step_id"],
            "matched_proof_step_id": fp.get("matched_proof_step_id"),
            "premise_id": fp["premise_id"],
            "type": fp_type,
            "reason": reason,
            "metric_counted_fp": metric_counted,
            "table1_verdict": verdict,
            "phi_original": fp.get("phi_original", ""),
            "phi_semantic": fp.get("phi_semantic", ""),
        })

    type_counts = Counter(c["type"] for c in classifications)
    metric_type_counts = Counter(
        c["type"] for c in classifications if c["metric_counted_fp"])
    return {
        "candidates": classifications,
        "n_semantic_delta_candidates": len(classifications),
        "n_metric_counted_fp": n_metric_fp,
        "n_outside_metric_set": len(classifications) - n_metric_fp,
        "type_counts": dict(type_counts),
        "metric_counted_type_counts": dict(metric_type_counts),
    }
