#!/usr/bin/env python3
"""Reproduce Table 1 (Main Results) from the evidence pack.

Requires NO API calls. Recomputes every row from saved certificates with
aligned CoT-keyed ground truth and compares against final_report.json
(the canonical source of truth) rather than hard-coded expected values.

Usage: python scripts/reproduce_table1.py
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.ablation import run_ablation
from src.baseline_self_consistency import evaluate_baseline
from src.cascade_filter import evaluate_with_cascade_filter
from src.evidence_loader import (
    load_baseline_samples,
    load_certificates,
    load_cot_steps_by_pid,
    load_problems,
)
from src.ground_truth import evaluate_judged


def judge_consistent_only(cert):
    """Re-judge using consistent substitution only (A alone)."""
    if not (cert.get("parse_ok_orig") and cert.get("parse_ok_sem")
            and cert.get("parse_ok_sur")):
        return "UNPARSEABLE"
    sem_delta = cert.get("semantic_delta", False)
    sur_delta = cert.get("surface_delta", False)
    if sem_delta and not sur_delta:
        return "GROUNDED"
    if sem_delta and sur_delta:
        return "INPUT-SENSITIVE"
    if not sem_delta and sur_delta:
        return "UNSTABLE"
    return "INSENSITIVE"


def main():
    print("=" * 65)
    print("  Reproducing Table 1: Main Results (GPT-4o, ProntoQA 50)")
    print("=" * 65)

    problems = load_problems()
    certs = load_certificates("gpt-4o")
    cot_steps = load_cot_steps_by_pid("gpt-4o")
    print(f"  Loaded {len(problems)} problems, {len(certs)} certificates")

    # --- Rows, all via the shared aligned evaluation path ---
    abl = run_ablation(certs, problems, cot_steps)
    a_full = evaluate_judged(certs, problems, cot_steps, judge_consistent_only)
    a_pred = evaluate_judged(
        certs, problems, cot_steps, judge_consistent_only, pred_only=True)
    cascade = evaluate_with_cascade_filter(certs, problems, cot_steps)

    bl_data = load_baseline_samples("gpt-4o")
    bl_full = evaluate_baseline(bl_data, problems, gt_type="full")
    sd = abl["string_diff_baseline"]

    # ================================================================
    # Print Table 1 (same rows and order as the paper)
    # ================================================================
    print(f"\n  Table 1: Main Results")
    print(f"  {'Method':<30} {'P':>7} {'R':>7} {'F1':>7}")
    print("  " + "-" * 51)
    print(f"  {'Self-Consistency':<30} {bl_full['precision']:>7.3f} "
          f"{bl_full['recall']:>7.3f} {bl_full['f1']:>7.3f}")
    print(f"  {'String-diff':<30} {sd['precision']:>7.3f} "
          f"{sd['recall']:>7.3f} {sd['f1']:>7.3f}")
    print(f"  {'A consistent (full)':<30} {a_full['precision']:>7.3f} "
          f"{a_full['recall']:>7.3f} {a_full['f1']:>7.3f}")
    label_cascade = "A+A'+cascade (full)"
    print(f"  {label_cascade:<30} {cascade['full']['precision']:>7.3f} "
          f"{cascade['full']['recall']:>7.3f} {cascade['full']['f1']:>7.3f}")
    print(f"  {'A consistent (pred)':<30} {a_pred['precision']:>7.3f} "
          f"{a_pred['recall']:>7.3f} {a_pred['f1']:>7.3f}")

    # ================================================================
    # Verify against the canonical report (single source of truth)
    # ================================================================
    report_path = os.path.join(BASE, "evidence", "gpt-4o", "final_report.json")
    report = json.load(open(report_path))["main"]
    checks = [
        ("Self-Consistency F1", bl_full["f1"], report["self_consistency_full"]["f1"]),
        ("A consistent (full) F1", a_full["f1"], report["A_consistent_full"]["f1"]),
        ("A consistent (pred) F1", a_pred["f1"], report["A_consistent_pred"]["f1"]),
        ("A+A'+cascade F1", cascade["full"]["f1"], report["A_Ap_cascade_full"]["f1"]),
    ]
    print("\n  --- Verification against evidence/gpt-4o/final_report.json ---")
    all_ok = True
    for name, actual, expected in checks:
        ok = abs(actual - expected) < 1e-9
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(f"    {name}: {actual:.4f} (report {expected:.4f}) [{status}]")

    if all_ok:
        print("\n  [DONE] Table 1 rows match the canonical report.")
        return 0
    print("\n  [ERROR] Table 1 rows do not match final_report.json. "
          "Rerun scripts/recompute_reports.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
