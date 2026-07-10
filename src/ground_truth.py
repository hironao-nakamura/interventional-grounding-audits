"""Phase 3: Evaluation — P/R/F1 vs ground-truth proof trees.

Two-stage evaluation (Option B formalized):
  F1_full: proof tree全依存を正解とする（全前提）
  F1_pred: 述語決定依存のみを正解とする（構造的前提を除外）

FN analysis distinguishes 'structural' from 'predicate-determining' premises
to show that most FN are by design (entity-intro), not bugs.
"""

import json
import os
import re
import sys
import warnings

from src.step_alignment import StepAlignment, align_cot_to_proof


def extract_ground_truth_dependencies_for_cot_steps(
    problem: dict,
    cot_steps: list[dict],
    *,
    pred_only: bool = False,
) -> tuple[dict[tuple[int, str], bool], StepAlignment]:
    """Return GT dependencies keyed by (cot_step_id, premise_id), not proof step ID.

    Unmatched CoT steps are excluded from the returned GT map.
    Ambiguous-alignment problems return an empty GT map; the alignment object
    flags them for problem-level exclusion from primary metrics.
    """
    alignment = align_cot_to_proof(problem, cot_steps)
    premise_ids = [p["id"] for p in problem["premises"]]
    proof_by_step = {e["step"]: e for e in problem.get("proof_tree", [])}

    gt: dict[tuple[int, str], bool] = {}
    if alignment.primary_metric_excluded_problem:
        # Whole problem excluded from primary metrics; no per-step GT.
        return gt, alignment

    for cot_id, proof_id in alignment.cot_to_proof.items():
        entry = proof_by_step[proof_id]
        # Only premise IDs count as direct premise dependencies; step
        # dependencies (S1, S2, ...) are not part of the premise metric.
        direct_premises = {d for d in entry["depends_on"] if d in set(premise_ids)}
        for pid in premise_ids:
            positive = pid in direct_premises
            if pred_only and positive:
                positive = classify_premise_type(problem, pid) == "predicate-determining"
            gt[(cot_id, pid)] = positive

    return gt, alignment



def _proof_keyed_gt(problem: dict, pred_only: bool = False) -> dict[tuple[int, str], bool]:
    """(proof_step_id, premise_id) -> bool ground truth, keyed by PROOF step id.

    Internal helper for callers that legitimately operate in proof-step space
    (e.g., the self-consistency baseline after content matching). Certificate
    evaluation must use extract_ground_truth_dependencies_for_cot_steps.
    """
    deps: dict[tuple[int, str], bool] = {}
    premise_ids = {p["id"] for p in problem["premises"]}
    for entry in problem["proof_tree"]:
        step_id = entry["step"]
        for pid in premise_ids:
            positive = pid in entry["depends_on"]
            if pred_only and positive:
                positive = classify_premise_type(problem, pid) == "predicate-determining"
            deps[(step_id, pid)] = positive
    return deps


def extract_ground_truth_dependencies(problem: dict) -> dict[tuple[int, str], bool]:
    """Extract (proof_step_id, premise_id) -> True/False from proof tree.

    True = step depends on this premise according to proof tree.

    .. deprecated::
        Keys are PROOF-TREE step IDs and must not be compared against
        certificate `step_id` values, which are model CoT step IDs.
        Use :func:`extract_ground_truth_dependencies_for_cot_steps` for
        certificate evaluation, or :func:`_proof_keyed_gt` for content-matched
        proof-step-space evaluation (self-consistency baseline).
    """
    warnings.warn(
        "extract_ground_truth_dependencies is proof-step-keyed; use "
        "extract_ground_truth_dependencies_for_cot_steps for certificate "
        "evaluation.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _proof_keyed_gt(problem, pred_only=False)


def classify_premise_type(problem: dict, premise_id: str) -> str:
    """Classify a premise as 'structural' or 'predicate-determining'.

    - Structural: introduces an entity into the chain (e.g., "Alex is a wumpus")
    - Predicate-determining: chain premise that determines step conclusions
    """
    premise = None
    for p in problem["premises"]:
        if p["id"] == premise_id:
            premise = p
            break

    if premise is None:
        return "unknown"

    text = premise["text"]

    # Entity premises: "X is a Y" where X is a proper noun
    if re.match(r"[A-Z][a-z]+\s+is\s+(?:a|an)\s+\w+", text):
        return "structural"

    return "predicate-determining"


def extract_pred_only_dependencies(problem: dict) -> dict[tuple[int, str], bool]:
    """Extract ground truth with structural premises excluded from positive set.

    For F1_pred: structural premises (entity-intro) are treated as non-dependencies
    because predicate substitution probes are not designed to detect them.

    .. deprecated::
        Keys are PROOF-TREE step IDs. For certificate evaluation use
        :func:`extract_ground_truth_dependencies_for_cot_steps` with
        ``pred_only=True``.
    """
    warnings.warn(
        "extract_pred_only_dependencies is proof-step-keyed; use "
        "extract_ground_truth_dependencies_for_cot_steps(pred_only=True) "
        "for certificate evaluation.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _proof_keyed_gt(problem, pred_only=True)


def _compute_prf1(tp, fp, fn):
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


# --- Fixed metric policy (shared by every evaluation script) ---
# GT alignment: normalized-conclusion match, CoT-keyed GT.
# Unmatched CoT steps: excluded from primary metrics (alignment coverage).
# Ambiguous alignments: whole problem excluded from primary metrics.
# UNPARSEABLE / UNSTABLE: excluded from the primary-metric denominator.
# GROUNDED / INPUT-SENSITIVE: predicted positive.
# INSENSITIVE: predicted negative.
# CASCADE: predicted negative in the direct-dependency metric.
PRIMARY_SKIP_VERDICTS = {"UNPARSEABLE", "UNSTABLE"}
POSITIVE_VERDICTS = {"GROUNDED", "INPUT-SENSITIVE"}
METRIC_POLICY = "aligned_cot_keyed_gt; skip UNPARSEABLE+UNSTABLE; GROUNDED+INPUT-SENSITIVE positive; CASCADE negative"


def evaluate_judged(
    certificates: list[dict],
    problems: list[dict],
    cot_steps_by_pid: dict[str, list[dict]],
    judge_fn,
    *,
    pred_only: bool = False,
) -> dict:
    """Aligned-GT evaluation with a pluggable per-certificate judge function.

    This is the single shared evaluation path used by Table 1/2 rows, the
    ablation, the cascade filter, bootstrap CIs, and analysis generators, so
    the metric policy cannot drift between scripts.

    ``judge_fn(cert) -> verdict string``. Certificates flagged
    ``evaluation_excluded`` (unmatched CoT steps, ambiguous-alignment
    problems) are skipped before judging.
    """
    gt_map: dict[tuple[str, int, str], bool] = {}
    alignments: dict[str, StepAlignment] = {}
    for problem in problems:
        pid = problem["problem_id"]
        gt, alignment = extract_ground_truth_dependencies_for_cot_steps(
            problem, cot_steps_by_pid.get(pid, []), pred_only=pred_only,
        )
        alignments[pid] = alignment
        for (cot_id, prem_id), dep in gt.items():
            gt_map[(pid, cot_id, prem_id)] = dep

    per_problem: dict[str, dict[str, int]] = {}
    tp = fp = fn = tn = 0
    for cert in certificates:
        if cert.get("evaluation_excluded"):
            continue  # unmatched CoT step or ambiguous-alignment problem
        verdict = judge_fn(cert)
        if verdict in PRIMARY_SKIP_VERDICTS:
            continue
        key = (cert["problem_id"], cert["step_id"], cert["premise_id"])
        gt_dep = gt_map.get(key)
        if gt_dep is None:
            continue
        predicted = verdict in POSITIVE_VERDICTS
        counts = per_problem.setdefault(
            cert["problem_id"], {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        if predicted and gt_dep:
            tp += 1
            counts["tp"] += 1
        elif predicted and not gt_dep:
            fp += 1
            counts["fp"] += 1
        elif not predicted and gt_dep:
            fn += 1
            counts["fn"] += 1
        else:
            tn += 1
            counts["tn"] += 1

    precision, recall, f1 = _compute_prf1(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_problem": per_problem,
        "alignments": alignments,
        "metric_policy": METRIC_POLICY,
        "pred_only": pred_only,
    }


def evaluate_verdicts(
    certificates: list[dict],
    problem: dict,
    cot_steps: list[dict],
) -> dict:
    """Evaluate audit verdicts against aligned ground truth for one problem.

    Computes both F1_full and F1_pred with CoT-keyed GT.
    """
    gt_full, alignment = extract_ground_truth_dependencies_for_cot_steps(
        problem, cot_steps, pred_only=False)
    gt_pred, _ = extract_ground_truth_dependencies_for_cot_steps(
        problem, cot_steps, pred_only=True)

    # Full evaluation
    tp_f, fp_f, tn_f, fn_f = 0, 0, 0, 0
    # Predicate-only evaluation
    tp_p, fp_p, tn_p, fn_p = 0, 0, 0, 0
    fn_details = []

    for cert in certificates:
        if cert.get("evaluation_excluded"):
            continue
        sid = cert["step_id"]
        pid = cert["premise_id"]
        verdict = cert["verdict"]

        if verdict in PRIMARY_SKIP_VERDICTS:
            continue

        predicted_dep = (verdict in POSITIVE_VERDICTS)

        # --- F1_full ---
        gt_dep_full = gt_full.get((sid, pid))
        if gt_dep_full is not None:
            if gt_dep_full and predicted_dep:
                tp_f += 1
            elif gt_dep_full and not predicted_dep:
                fn_f += 1
                fn_details.append({
                    "step_id": sid,
                    "matched_proof_step_id": cert.get("matched_proof_step_id"),
                    "premise_id": pid,
                    "verdict": verdict,
                    "verdict_consistent": cert.get("verdict_consistent", ""),
                    "premise_type": classify_premise_type(problem, pid),
                    "phi_original": cert.get("phi_original"),
                    "phi_semantic": cert.get("phi_semantic"),
                    "phi_local": cert.get("phi_local"),
                    "consistent_detected": bool(cert.get("semantic_delta", False)),
                    "local_detected": bool(cert.get("local_delta", False)),
                })
            elif not gt_dep_full and predicted_dep:
                fp_f += 1
            else:
                tn_f += 1

        # --- F1_pred (structural premises excluded from positives) ---
        gt_dep_pred = gt_pred.get((sid, pid))
        if gt_dep_pred is not None:
            if gt_dep_pred and predicted_dep:
                tp_p += 1
            elif gt_dep_pred and not predicted_dep:
                fn_p += 1
            elif not gt_dep_pred and predicted_dep:
                fp_p += 1
            else:
                tn_p += 1

    p_full, r_full, f1_full = _compute_prf1(tp_f, fp_f, fn_f)
    p_pred, r_pred, f1_pred = _compute_prf1(tp_p, fp_p, fn_p)

    return {
        "problem_id": problem["problem_id"],
        "alignment": {
            "unmatched_cot_steps": alignment.unmatched_cot_steps,
            "unmatched_proof_steps": alignment.unmatched_proof_steps,
            "ambiguous_cot_steps": alignment.ambiguous_cot_steps,
            "primary_metric_excluded_problem": alignment.primary_metric_excluded_problem,
        },
        "full": {"tp": tp_f, "fp": fp_f, "tn": tn_f, "fn": fn_f,
                 "precision": p_full, "recall": r_full, "f1": f1_full},
        "pred_only": {"tp": tp_p, "fp": fp_p, "tn": tn_p, "fn": fn_p,
                      "precision": p_pred, "recall": r_pred, "f1": f1_pred},
        "fn_details": fn_details,
    }


def evaluate_all(
    all_certificates: list[dict],
    problems: list[dict],
    cot_steps_by_pid: dict[str, list[dict]],
) -> dict:
    """Evaluate all verdicts across all problems with aligned CoT-keyed GT.

    Returns aggregate metrics for both F1_full and F1_pred + FN analysis.
    """
    certs_by_problem = {}
    for cert in all_certificates:
        pid = cert["problem_id"]
        certs_by_problem.setdefault(pid, []).append(cert)

    per_problem = []
    all_fn_details = []
    fn_structural = 0
    fn_predicate = 0

    for problem in problems:
        pid = problem["problem_id"]
        certs = certs_by_problem.get(pid, [])
        result = evaluate_verdicts(certs, problem, cot_steps_by_pid.get(pid, []))
        per_problem.append(result)

        for fnd in result["fn_details"]:
            all_fn_details.append(fnd)
            if fnd["premise_type"] == "structural":
                fn_structural += 1
            else:
                fn_predicate += 1

    # Recompute aggregate counts from per-problem results.
    t_tp_f = sum(r["full"]["tp"] for r in per_problem)
    t_fp_f = sum(r["full"]["fp"] for r in per_problem)
    t_tn_f = sum(r["full"]["tn"] for r in per_problem)
    t_fn_f = sum(r["full"]["fn"] for r in per_problem)
    t_tp_p = sum(r["pred_only"]["tp"] for r in per_problem)
    t_fp_p = sum(r["pred_only"]["fp"] for r in per_problem)
    t_tn_p = sum(r["pred_only"]["tn"] for r in per_problem)
    t_fn_p = sum(r["pred_only"]["fn"] for r in per_problem)

    p_full, r_full, f1_full = _compute_prf1(t_tp_f, t_fp_f, t_fn_f)
    p_pred, r_pred, f1_pred = _compute_prf1(t_tp_p, t_fp_p, t_fn_p)

    return {
        "aggregate": {
            "tp": t_tp_f, "fp": t_fp_f, "tn": t_tn_f, "fn": t_fn_f,
            "precision": round(p_full, 4),
            "recall": round(r_full, 4),
            "f1": round(f1_full, 4),
            "total_evaluated": t_tp_f + t_fp_f + t_tn_f + t_fn_f,
        },
        "aggregate_pred_only": {
            "tp": t_tp_p, "fp": t_fp_p, "tn": t_tn_p, "fn": t_fn_p,
            "precision": round(p_pred, 4),
            "recall": round(r_pred, 4),
            "f1": round(f1_pred, 4),
            "total_evaluated": t_tp_p + t_fp_p + t_tn_p + t_fn_p,
        },
        "fn_analysis": {
            "total_fn": t_fn_f,
            "fn_structural": fn_structural,
            "fn_predicate_determining": fn_predicate,
            "structural_ratio": round(fn_structural / t_fn_f, 4) if t_fn_f > 0 else 0,
            "explanation": (
                "Structural FN: entity-introduction premises (e.g., 'Alex is a wumpus'). "
                "With local substitution, most structural FNs are now detected. "
                "Remaining FN are true limitations of the probing approach."
            ),
        },
        "per_problem": per_problem,
        "fn_examples": all_fn_details[:20],
    }


def print_evaluation(eval_result: dict) -> None:
    """Print evaluation results with both F1_full and F1_pred."""
    agg = eval_result["aggregate"]
    agg_p = eval_result["aggregate_pred_only"]
    fna = eval_result["fn_analysis"]

    print("\n" + "=" * 70)
    print("  EVALUATION: Verdicts vs Ground-Truth Proof Trees")
    print("=" * 70)

    print(f"\n  {'Metric':<30} {'F1_full':>10} {'F1_pred':>10}")
    print("  " + "-" * 50)
    print(f"  {'Precision':<30} {agg['precision']:>10.4f} {agg_p['precision']:>10.4f}")
    print(f"  {'Recall':<30} {agg['recall']:>10.4f} {agg_p['recall']:>10.4f}")
    print(f"  {'F1':<30} {agg['f1']:>10.4f} {agg_p['f1']:>10.4f}")
    print(f"  {'TP':<30} {agg['tp']:>10} {agg_p['tp']:>10}")
    print(f"  {'FP':<30} {agg['fp']:>10} {agg_p['fp']:>10}")
    print(f"  {'FN':<30} {agg['fn']:>10} {agg_p['fn']:>10}")
    print(f"  {'Total evaluated':<30} {agg['total_evaluated']:>10} {agg_p['total_evaluated']:>10}")

    print(f"\n  F1_full : all proof-tree dependencies as ground truth")
    print(f"  F1_pred : predicate-determining dependencies only (structural excluded)")

    print(f"\n  --- FN Analysis ---")
    print(f"  Total FN (full): {fna['total_fn']}")
    print(f"    Structural (entity-intro):      {fna['fn_structural']} ({fna['structural_ratio']:.1%})")
    print(f"    Predicate-determining (chain):   {fna['fn_predicate_determining']}")

    if eval_result["fn_examples"]:
        print(f"\n  FN examples:")
        for ex in eval_result["fn_examples"][:5]:
            print(f"    [{ex.get('step_id','?')}x{ex.get('premise_id','?')}] "
                  f"verdict={ex['verdict']} type={ex['premise_type']} "
                  f"phi_orig={ex.get('phi_original')} "
                  f"phi_local={ex.get('phi_local')}")

    print("=" * 70)
