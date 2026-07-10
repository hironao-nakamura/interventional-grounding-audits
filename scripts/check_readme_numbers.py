#!/usr/bin/env python3
"""Verify the README 'Key Numbers' table against the canonical reports.

Usage: python scripts/check_readme_numbers.py
Exit 0 + 'README NUMBERS: OK' when every value matches; 1 otherwise.
"""

import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    readme = open(os.path.join(BASE, "README.md")).read()
    fr = json.load(open(os.path.join(BASE, "evidence", "gpt-4o", "final_report.json")))
    ci = json.load(open(os.path.join(BASE, "evidence", "gpt-4o", "bootstrap_ci.json")))
    fr_c = json.load(open(os.path.join(
        BASE, "evidence", "claude-sonnet-4-5", "final_report.json")))
    m, an = fr["main"], fr["analysis"]
    cov = an["parse_rates_and_coverage"]
    sa = an["step_alignment"]
    ra = an["rawr"]
    sa_c = fr_c["analysis"]["step_alignment"]

    a_ci = ci["A_consistent_full"]["ci"]["f1"]
    sc_ci = ci["self_consistency_full"]["ci"]["f1"]

    expected = [
        ("A consistent F1 (full)", f"{m['A_consistent_full']['f1']:.3f}"),
        ("A consistent F1 (pred-determining)", f"{m['A_consistent_pred']['f1']:.3f}"),
        ("A consistent recall (pred-determining)",
         f"{100 * m['A_consistent_pred']['recall']:.1f}%"),
        ("A+A'+cascade F1 (full)", f"{m['A_Ap_cascade_full']['f1']:.3f}"),
        ("Self-consistency F1 (full)", f"{m['self_consistency_full']['f1']:.3f}"),
        ("String-diff F1 (full)", f"{m['string_diff_full']['f1']:.3f}"),
        ("95% CI, A vs self-consistency",
         f"non-overlapping ([{a_ci['ci_lower']:.3f}, {a_ci['ci_upper']:.3f}] "
         f"vs [{sc_ci['ci_lower']:.3f}, {sc_ci['ci_upper']:.3f}])"),
        ("Certificates (total / evaluable)",
         f"{cov['n_total_certificates']} / {cov['n_evaluable_certificates']}"),
        ("Unmatched CoT steps (GPT-4o)",
         f"{sa['unmatched_cot_steps']} "
         f"({len(sa['problems_with_unmatched_cot_steps'])} problems)"),
        ("RAWR cases (A consistent, full)",
         f"{ra['total_rawr']}/{ra['total_correct']} "
         f"({round(100 * ra['rawr_rate'])}%)"),
        ("RAWR breakdown",
         f"{ra['structural_only']} structural-only, "
         f"{ra['has_predicate_rawr']} predicate-determining"),
        ("Misrepresentation cases", str(an["misrepresentation"]["total"])),
        ("Semantic-delta FP candidates / metric-counted",
         f"{an['fp']['n_semantic_delta_candidates']} / "
         f"{an['fp']['n_metric_counted_fp']}"),
        ("Lower-bound F1 (A consistent, full)",
         f"{an['lower_bound_A_consistent_full']['f1']:.3f}"),
        ("Claude Sonnet 4.5 F1 (full, evaluable steps)",
         f"{fr_c['main']['A_consistent_full']['f1']:.3f}"),
        ("Claude Sonnet 4.5 lower-bound F1",
         f"{fr_c['analysis']['lower_bound_A_consistent_full']['f1']:.3f}"),
        ("Claude Sonnet 4.5 unmatched CoT steps",
         f"{sa_c['unmatched_cot_steps']} "
         f"({len(sa_c['problems_with_unmatched_cot_steps'])} problems)"),
    ]

    failures = []
    for label, value in expected:
        pattern = re.escape(f"| {label} | {value} |")
        if not re.search(pattern, readme):
            failures.append((label, value))

    # Rounding note: 0.790 must correspond to final_report 0.7897 etc.
    for label, want in [("0.806", m["A_consistent_full"]["f1"]),
                        ("0.885", m["A_consistent_pred"]["f1"]),
                        ("0.819", m["A_Ap_cascade_full"]["f1"]),
                        ("0.343", m["self_consistency_full"]["f1"])]:
        if f"{want:.3f}" != label:
            failures.append((f"rounding of {want}", label))

    if failures:
        print("README NUMBERS: MISMATCH")
        for label, value in failures:
            print(f"  missing/incorrect: | {label} | {value} |")
        return 1
    print(f"README NUMBERS: OK ({len(expected)} rows verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
