#!/usr/bin/env python3
"""Reproduce Table 2 (Ablation) from the evidence pack.

Requires NO API calls. Recomputes all six ablation rows from saved
certificates with aligned CoT-keyed ground truth and compares against
final_report.json (canonical) rather than hard-coded expected values.

Usage: python scripts/reproduce_table2.py
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.ablation import run_ablation
from src.cascade_filter import evaluate_with_cascade_filter
from src.evidence_loader import (
    load_certificates,
    load_cot_steps_by_pid,
    load_problems,
)

ROW_ORDER = [
    ("A_consistent_only", "A consistent only"),
    ("A'_local_only", "A' local only"),
    ("A+A'_combined", "A + A' combined"),
    ("A+A'_cascade_filter", "A + A' + cascade filter"),
    ("A+A'_no_surface_ctrl", "A + A' w/o surface ctrl"),
    ("string_diff_baseline", "String-diff baseline"),
]


def main():
    print("=" * 65)
    print("  Reproducing Table 2: Ablation (GPT-4o, ProntoQA 50)")
    print("=" * 65)

    problems = load_problems()
    certs = load_certificates("gpt-4o")
    cot_steps = load_cot_steps_by_pid("gpt-4o")
    print(f"  Loaded {len(problems)} problems, {len(certs)} certificates")

    results = run_ablation(certs, problems, cot_steps)
    cascade = evaluate_with_cascade_filter(certs, problems, cot_steps)
    results["A+A'_cascade_filter"] = {
        k: cascade["full"][k] for k in
        ("tp", "fp", "fn", "tn", "precision", "recall", "f1")}

    print(f"\n  {'Configuration':<28} {'P':>7} {'R':>7} {'F1':>7}")
    print("  " + "-" * 51)
    for key, label in ROW_ORDER:
        r = results[key]
        print(f"  {label:<28} {r['precision']:>7.3f} "
              f"{r['recall']:>7.3f} {r['f1']:>7.3f}")

    # --- Verification against the canonical report ---
    report_path = os.path.join(BASE, "evidence", "gpt-4o", "final_report.json")
    ablation_report = json.load(open(report_path))["ablation"]
    print("\n  --- Verification against evidence/gpt-4o/final_report.json ---")
    all_ok = True
    for key, label in ROW_ORDER:
        actual = results[key]["f1"]
        expected = ablation_report[key]["f1"]
        ok = abs(actual - expected) < 1e-9
        if not ok:
            all_ok = False
        print(f"    {label} F1: {actual:.4f} (report {expected:.4f}) "
              f"[{'OK' if ok else 'MISMATCH'}]")

    if all_ok:
        print("\n  [DONE] Table 2 rows match the canonical report.")
        return 0
    print("\n  [ERROR] Table 2 rows do not match final_report.json. "
          "Rerun scripts/recompute_reports.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
