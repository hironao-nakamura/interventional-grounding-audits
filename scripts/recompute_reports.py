#!/usr/bin/env python3
"""Regenerate every report/analysis JSON from certificates (single source of truth).

Usage:
  python scripts/recompute_reports.py --model-dir evidence/gpt-4o
  python scripts/recompute_reports.py --model-dir evidence/claude-sonnet-4-5

Outputs (for --model-dir evidence/<model>):
  evidence/<model>/final_report.json
  evidence/<model>/ablation.json
  evidence/<model>/bootstrap_ci.json
  evidence/<model>/chain_length_analysis.json
  evidence/<model>/rawr_analysis.json
  evidence/<model>/misrepresentation_analysis.json
  evidence/<model>/fp_analysis.json
  evidence/<model>/step_alignment_summary.json
  baselines/self_consistency/<model>/baseline_report.json   (if samples exist)

All metrics use aligned CoT-keyed ground truth and the fixed metric policy
in src.ground_truth (skip UNPARSEABLE/UNSTABLE; GROUNDED+INPUT-SENSITIVE
positive; CASCADE negative). No API calls; deterministic.
"""

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.ablation import run_ablation  # noqa: E402
from src.baseline_self_consistency import evaluate_baseline  # noqa: E402
from src.bootstrap_ci import bootstrap_ci  # noqa: E402
from src.cascade_filter import evaluate_with_cascade_filter  # noqa: E402
from src.chain_length_analysis import analyze_chain_length  # noqa: E402
from src.evidence_loader import (  # noqa: E402
    load_baseline_samples,
    load_certificates,
    load_cot_steps_by_pid,
    load_original_raw,
    load_problems,
)
from src.fp_analysis import analyze_false_positives  # noqa: E402
from src.ground_truth import (  # noqa: E402
    METRIC_POLICY,
    classify_premise_type,
    evaluate_judged,
    extract_ground_truth_dependencies_for_cot_steps,
)
from src.step_alignment import align_cot_to_proof  # noqa: E402
from src.evidence_loader import parse_final_answer  # noqa: E402


def judge_consistent_only(cert: dict) -> str:
    """Table 1 'A consistent' evaluator (identical to cert verdict_consistent)."""
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


def judge_combined(cert: dict) -> str:
    return cert.get("verdict", "UNPARSEABLE")


def build_alignment_summary(problems, cot_steps_by_pid) -> dict:
    total_cot = total_proof = matched = unmatched_cot = 0
    unmatched_proof = ambiguous = 0
    p_uc, p_up, p_amb, p_excl = [], [], [], []
    per_problem = {}
    for problem in problems:
        pid = problem["problem_id"]
        cot = cot_steps_by_pid.get(pid, [])
        a = align_cot_to_proof(problem, cot)
        per_problem[pid] = {
            "cot_to_proof": {str(k): v for k, v in sorted(a.cot_to_proof.items())},
            "unmatched_cot_steps": a.unmatched_cot_steps,
            "unmatched_proof_steps": a.unmatched_proof_steps,
            "ambiguous_cot_steps": a.ambiguous_cot_steps,
            "primary_metric_excluded_problem": a.primary_metric_excluded_problem,
        }
        total_cot += len(cot)
        total_proof += len(problem["proof_tree"])
        matched += len(a.cot_to_proof) if not a.primary_metric_excluded_problem else 0
        unmatched_cot += len(a.unmatched_cot_steps)
        unmatched_proof += len(a.unmatched_proof_steps)
        ambiguous += len(a.ambiguous_cot_steps)
        if a.unmatched_cot_steps:
            p_uc.append(pid)
        if a.unmatched_proof_steps:
            p_up.append(pid)
        if a.ambiguous_cot_steps:
            p_amb.append(pid)
        if a.primary_metric_excluded_problem:
            p_excl.append(pid)

    return {
        "total_problems": len(problems),
        "total_cot_steps": total_cot,
        "total_proof_steps": total_proof,
        "matched_cot_steps": matched,
        "unmatched_cot_steps": unmatched_cot,
        "unmatched_proof_steps": unmatched_proof,
        "ambiguous_steps": ambiguous,
        "problems_with_unmatched_cot_steps": p_uc,
        "problems_with_unmatched_proof_steps": p_up,
        "problems_with_ambiguous_alignment": p_amb,
        "primary_metric_excluded_problems": p_excl,
        "primary_metric_policy": "exclude_unmatched_cot_steps_and_ambiguous_problems",
        "lower_bound_policy": "treat_unmatched_or_unparseable_gt_dependencies_as_false_negatives",
        "requires_alignment_disclosure": bool(p_excl),
        "requires_lower_bound_disclosure": bool(unmatched_proof or p_excl),
        "per_problem": per_problem,
    }


def compute_parse_and_coverage(certs) -> dict:
    n_total = len(certs)
    evaluable = [c for c in certs if not c.get("evaluation_excluded")]
    n_eval = len(evaluable)

    def rate(field):
        return round(sum(1 for c in certs if c.get(field)) / n_total, 4) if n_total else 0

    combined_unparseable = sum(1 for c in evaluable if c["verdict"] == "UNPARSEABLE")
    combined_unstable = sum(1 for c in evaluable if c["verdict"] == "UNSTABLE")
    a_unparseable = sum(
        1 for c in evaluable if judge_consistent_only(c) == "UNPARSEABLE")
    a_unstable = sum(
        1 for c in evaluable if judge_consistent_only(c) == "UNSTABLE")

    return {
        "n_total_certificates": n_total,
        "n_evaluable_certificates": n_eval,
        "n_excluded_certificates": n_total - n_eval,
        "parse_rate_orig": rate("parse_ok_orig"),
        "parse_rate_sem": rate("parse_ok_sem"),
        "parse_rate_local": rate("parse_ok_local"),
        "parse_rate_sur": rate("parse_ok_sur"),
        "combined_unparseable": combined_unparseable,
        "combined_unparseable_rate": round(combined_unparseable / n_eval, 4) if n_eval else 0,
        "combined_unstable": combined_unstable,
        "a_consistent_unparseable": a_unparseable,
        "a_consistent_unparseable_rate": round(a_unparseable / n_eval, 4) if n_eval else 0,
        "a_consistent_unstable": a_unstable,
    }


def compute_lower_bound(certs, problems, cot_steps_by_pid, primary) -> dict:
    """A-consistent lower bound: skipped-with-GT-True and never-audited GT
    dependencies (unmatched proof steps, ambiguous problems) count as FN."""
    gt_map = {}
    for p in problems:
        gt, _ = extract_ground_truth_dependencies_for_cot_steps(
            p, cot_steps_by_pid.get(p["problem_id"], []))
        for (sid, pid), dep in gt.items():
            gt_map[(p["problem_id"], sid, pid)] = dep

    extra_fn_skipped = 0
    for c in certs:
        if c.get("evaluation_excluded"):
            continue
        if judge_consistent_only(c) not in ("UNPARSEABLE", "UNSTABLE"):
            continue
        if gt_map.get((c["problem_id"], c["step_id"], c["premise_id"])):
            extra_fn_skipped += 1

    extra_fn_unscored = 0
    for p in problems:
        pid = p["problem_id"]
        a = align_cot_to_proof(p, cot_steps_by_pid.get(pid, []))
        premise_ids = {pr["id"] for pr in p["premises"]}
        if a.primary_metric_excluded_problem:
            unscored_steps = [e["step"] for e in p["proof_tree"]]
        else:
            unscored_steps = a.unmatched_proof_steps
        for e in p["proof_tree"]:
            if e["step"] in unscored_steps:
                extra_fn_unscored += sum(
                    1 for d in e["depends_on"] if d in premise_ids)

    tp, fp = primary["tp"], primary["fp"]
    fn = primary["fn"] + extra_fn_skipped + extra_fn_unscored
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "extra_fn_from_unparseable_or_unstable": extra_fn_skipped,
        "extra_fn_from_unscored_proof_dependencies": extra_fn_unscored,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "policy": "treat_unmatched_or_unparseable_gt_dependencies_as_false_negatives",
    }


def compute_rawr(certs, problems, cot_steps_by_pid, model, baseline_samples) -> dict:
    """RAWR under the A-consistent (full) evaluator with aligned GT."""
    prob_map = {p["problem_id"]: p for p in problems}
    certs_by_pid = {}
    for c in certs:
        certs_by_pid.setdefault(c["problem_id"], []).append(c)

    cases = []
    total_correct = 0
    total_insensitive_deps = 0
    for p in problems:
        pid = p["problem_id"]
        raw = load_original_raw(model, pid)["raw_response"]
        model_answer = parse_final_answer(raw)
        gt_answer = p.get("answer")
        answer_correct = (model_answer is not None and model_answer == gt_answer)
        if not answer_correct:
            continue
        total_correct += 1

        gt, alignment = extract_ground_truth_dependencies_for_cot_steps(
            p, cot_steps_by_pid.get(pid, []))
        if alignment.primary_metric_excluded_problem:
            continue  # ambiguous problems excluded from primary metrics

        insens = []
        for c in certs_by_pid.get(pid, []):
            if c.get("evaluation_excluded"):
                continue
            key = (c["step_id"], c["premise_id"])
            if not gt.get(key):
                continue
            if judge_consistent_only(c) != "INSENSITIVE":
                continue
            ptype = classify_premise_type(p, c["premise_id"])
            premise_text = next(
                pr["text"] for pr in p["premises"] if pr["id"] == c["premise_id"])
            insens.append({
                "step_id": c["step_id"],
                "matched_proof_step_id": c.get("matched_proof_step_id"),
                "premise_id": c["premise_id"],
                "premise_text": premise_text,
                "premise_type": ptype,
                "phi_original": c.get("phi_original"),
                "phi_semantic": c.get("phi_semantic"),
                "phi_local": c.get("phi_local"),
                "verdict_consistent": c.get("verdict_consistent"),
            })
        if not insens:
            continue

        n_structural = sum(1 for d in insens if d["premise_type"] == "structural")
        n_predicate = len(insens) - n_structural
        total_insensitive_deps += len(insens)

        sc_perfect = None
        if baseline_samples and pid in baseline_samples:
            svs = baseline_samples[pid]["step_verdicts"].values()
            sc_perfect = all(sv.get("agreement", 0) == 1.0 for sv in svs)

        cases.append({
            "problem_id": pid,
            "gt_answer": gt_answer,
            "model_answer": model_answer,
            "answer_correct": True,
            "n_insensitive_deps": len(insens),
            "n_structural": n_structural,
            "n_predicate": n_predicate,
            "self_consistency_perfect": sc_perfect,
            "insensitive_dependencies": insens,
        })

    structural_only = sum(1 for c in cases if c["n_predicate"] == 0)
    has_predicate = sum(1 for c in cases if c["n_predicate"] > 0)
    sc_perfect_cases = sum(1 for c in cases if c.get("self_consistency_perfect"))
    return {
        "total_correct": total_correct,
        "total_rawr": len(cases),
        "rawr_rate": round(len(cases) / total_correct, 4) if total_correct else 0,
        "structural_only": structural_only,
        "has_predicate_rawr": has_predicate,
        "total_insensitive_deps": total_insensitive_deps,
        "self_consistency_perfect_cases": sc_perfect_cases,
        "metric_policy": "A_consistent_full_aligned_gt",
        "cases": cases,
    }


def compute_misrepresentation(certs, problems, cot_steps_by_pid) -> dict:
    prob_map = {p["problem_id"]: p for p in problems}
    steps_map = {
        pid: {s["step_id"]: s for s in steps}
        for pid, steps in cot_steps_by_pid.items()
    }
    gt_map = {}
    for p in problems:
        gt, _ = extract_ground_truth_dependencies_for_cot_steps(
            p, cot_steps_by_pid.get(p["problem_id"], []))
        for (sid, pid_), dep in gt.items():
            gt_map[(p["problem_id"], sid, pid_)] = dep

    cases = []
    for c in certs:
        if c.get("evaluation_excluded"):
            continue
        if not c.get("citation_detected"):
            continue
        if judge_consistent_only(c) != "INSENSITIVE":
            continue
        key = (c["problem_id"], c["step_id"], c["premise_id"])
        if not gt_map.get(key):
            continue
        p = prob_map[c["problem_id"]]
        premise_text = next(
            pr["text"] for pr in p["premises"] if pr["id"] == c["premise_id"])
        step = steps_map.get(c["problem_id"], {}).get(c["step_id"], {})
        cases.append({
            "problem_id": c["problem_id"],
            "step_id": c["step_id"],
            "matched_proof_step_id": c.get("matched_proof_step_id"),
            "premise_id": c["premise_id"],
            "step_text": step.get("raw_text", ""),
            "premise_text": premise_text,
            "phi_original": c.get("phi_original"),
            "phi_semantic": c.get("phi_semantic"),
            "phi_local": c.get("phi_local"),
            "citation_type": c.get("citation_type"),
            "citation_evidence": c.get("citation_evidence"),
            "gt_should_depend": True,
            "verdict_consistent": c.get("verdict_consistent"),
        })

    return {
        "total": len(cases),
        "gt_should_depend": len(cases),
        "metric_policy": "A_consistent_full_aligned_gt + citation_detected",
        "cases": cases,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="e.g. evidence/gpt-4o")
    ap.add_argument("--out-dir", default=None,
                    help="Write reports here instead of evidence/<model> "
                         "(used by reproduce_all.py to verify shipped reports)")
    args = ap.parse_args()

    model = os.path.basename(args.model_dir.rstrip("/"))
    out_dir = args.out_dir or os.path.join(BASE, "evidence", model)
    os.makedirs(out_dir, exist_ok=True)

    problems = load_problems()
    certs = load_certificates(model)
    cot_steps = load_cot_steps_by_pid(model)
    try:
        baseline_samples = load_baseline_samples(model)
    except FileNotFoundError:
        baseline_samples = {}

    print(f"Recomputing reports for {model}: "
          f"{len(problems)} problems, {len(certs)} certificates", flush=True)

    # --- Alignment summary (Section 3.8 schema) ---
    summary = build_alignment_summary(problems, cot_steps)
    with open(os.path.join(out_dir, "step_alignment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Ablation (Table 2 rows, minus cascade which is added below) ---
    abl = run_ablation(certs, problems, cot_steps)
    cascade = evaluate_with_cascade_filter(certs, problems, cot_steps)
    abl_out = dict(abl)
    abl_out["A+A'_cascade_filter"] = {
        k: cascade["full"][k] for k in
        ("tp", "fp", "fn", "tn", "precision", "recall", "f1")}
    abl_out["A+A'_cascade_filter"]["n_cascaded"] = cascade["n_cascaded"]
    with open(os.path.join(out_dir, "ablation.json"), "w") as f:
        json.dump(abl_out, f, indent=2)

    # --- Table 1 rows via the shared evaluation path ---
    a_full = evaluate_judged(certs, problems, cot_steps, judge_consistent_only)
    a_pred = evaluate_judged(
        certs, problems, cot_steps, judge_consistent_only, pred_only=True)
    combined_full = evaluate_judged(certs, problems, cot_steps, judge_combined)

    baseline_report = None
    if baseline_samples:
        baseline_report = {
            "full": evaluate_baseline(baseline_samples, problems, gt_type="full"),
            "pred_only": evaluate_baseline(baseline_samples, problems, gt_type="pred"),
            "n_samples": next(iter(baseline_samples.values()))["n_samples"],
            "temperature": next(iter(baseline_samples.values()))["temperature"],
        }
        bl_dir = (out_dir if args.out_dir
                  else os.path.join(BASE, "baselines", "self_consistency", model))
        os.makedirs(bl_dir, exist_ok=True)
        with open(os.path.join(bl_dir, "baseline_report.json"), "w") as f:
            json.dump(baseline_report, f, indent=2)

    # --- Analyses ---
    rawr = compute_rawr(certs, problems, cot_steps, model, baseline_samples)
    with open(os.path.join(out_dir, "rawr_analysis.json"), "w") as f:
        json.dump(rawr, f, indent=2)

    misrep = compute_misrepresentation(certs, problems, cot_steps)
    with open(os.path.join(out_dir, "misrepresentation_analysis.json"), "w") as f:
        json.dump(misrep, f, indent=2)

    fp = analyze_false_positives(certs, problems, cot_steps)
    fp_out = {
        "config": "A_consistent semantic-delta candidates, aligned GT",
        "n_semantic_delta_candidates": fp["n_semantic_delta_candidates"],
        "n_metric_counted_fp": fp["n_metric_counted_fp"],
        "n_outside_metric_set": fp["n_outside_metric_set"],
        "type_counts": fp["type_counts"],
        "metric_counted_type_counts": fp["metric_counted_type_counts"],
        "classifications": fp["candidates"],
    }
    with open(os.path.join(out_dir, "fp_analysis.json"), "w") as f:
        json.dump(fp_out, f, indent=2)

    chain = analyze_chain_length(certs, problems, cot_steps)
    with open(os.path.join(out_dir, "chain_length_analysis.json"), "w") as f:
        json.dump(chain, f, indent=2)

    # --- Coverage / parse rates / lower bound ---
    coverage = compute_parse_and_coverage(certs)
    lower_bound = compute_lower_bound(certs, problems, cot_steps, a_full)

    # --- Bootstrap CIs (point estimates == Table 1 by construction) ---
    def ci_block(result):
        return {
            "point": {
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
            },
            "ci": bootstrap_ci(result["per_problem"]),
        }

    ci = {
        "A_consistent_full": ci_block(a_full),
        "A_consistent_pred": ci_block(a_pred),
        "A_Ap_combined_full": ci_block(combined_full),
        "A_Ap_cascade_full": ci_block(cascade["full"]),
    }
    if baseline_report:
        ci["self_consistency_full"] = ci_block(baseline_report["full"])
        ci["ci_overlap_A_vs_SC"] = {
            "a_ci_lower": ci["A_consistent_full"]["ci"]["f1"]["ci_lower"],
            "sc_ci_upper": ci["self_consistency_full"]["ci"]["f1"]["ci_upper"],
            "non_overlapping": (
                ci["A_consistent_full"]["ci"]["f1"]["ci_lower"]
                > ci["self_consistency_full"]["ci"]["f1"]["ci_upper"]),
            "gap": round(
                ci["A_consistent_full"]["point"]["f1"]
                - ci["self_consistency_full"]["point"]["f1"], 4),
        }
    with open(os.path.join(out_dir, "bootstrap_ci.json"), "w") as f:
        json.dump(ci, f, indent=2)

    # --- Canonical final report ---
    def row(result):
        return {k: result[k] for k in
                ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}

    final = {
        "model": model,
        "metric_policy": METRIC_POLICY,
        "main": {
            "self_consistency_full": (
                row(baseline_report["full"]) if baseline_report else None),
            "self_consistency_pred": (
                row(baseline_report["pred_only"]) if baseline_report else None),
            "string_diff_full": abl["string_diff_baseline"],
            "A_consistent_full": row(a_full),
            "A_consistent_pred": row(a_pred),
            "A_Ap_combined_full": row(combined_full),
            "A_Ap_cascade_full": {
                k: cascade["full"][k] for k in
                ("precision", "recall", "f1", "tp", "fp", "fn", "tn")},
            "A_Ap_cascade_pred": {
                k: cascade["pred_only"][k] for k in
                ("precision", "recall", "f1", "tp", "fp", "fn", "tn")},
            "n_cascaded": cascade["n_cascaded"],
        },
        "ablation": abl_out,
        "analysis": {
            "rawr": {k: rawr[k] for k in
                     ("total_correct", "total_rawr", "rawr_rate",
                      "structural_only", "has_predicate_rawr",
                      "total_insensitive_deps",
                      "self_consistency_perfect_cases", "metric_policy")},
            "misrepresentation": {
                "total": misrep["total"],
                "metric_policy": misrep["metric_policy"]},
            "fp": {k: fp_out[k] for k in
                   ("n_semantic_delta_candidates", "n_metric_counted_fp",
                    "n_outside_metric_set", "type_counts",
                    "metric_counted_type_counts")},
            "step_alignment": {k: summary[k] for k in summary if k != "per_problem"},
            "parse_rates_and_coverage": coverage,
            "lower_bound_A_consistent_full": lower_bound,
            "chain_length": chain,
        },
    }
    with open(os.path.join(out_dir, "final_report.json"), "w") as f:
        json.dump(final, f, indent=2)

    # final_report.json is canonical; remove the stale v2 file if present.
    v2 = os.path.join(out_dir, "final_report_v2.json")
    if os.path.exists(v2):
        os.remove(v2)
        print(f"  removed stale {v2}", flush=True)

    print(json.dumps({
        "A_consistent_full_f1": a_full["f1"],
        "A_consistent_pred_f1": a_pred["f1"],
        "A_Ap_combined_full_f1": combined_full["f1"],
        "A_Ap_cascade_full_f1": cascade["full"]["f1"],
        "self_consistency_full_f1": (
            baseline_report["full"]["f1"] if baseline_report else None),
        "rawr": f"{rawr['total_rawr']}/{rawr['total_correct']}",
        "misrepresentation": misrep["total"],
        "unmatched_cot_steps": summary["unmatched_cot_steps"],
        "unmatched_proof_steps": summary["unmatched_proof_steps"],
        "ambiguous_steps": summary["ambiguous_steps"],
        "lower_bound_f1": lower_bound["f1"],
    }, indent=2))
    print(f"Reports written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
