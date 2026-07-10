#!/usr/bin/env python3
"""Print every number the paper/README quotes, straight from the reports.

Usage:
  python scripts/export_paper_numbers.py --model-dir evidence/gpt-4o
  python scripts/export_paper_numbers.py --model-dir evidence/gpt-4o --format latex

This is the transcription aid used when editing paper.tex and README.md:
no number in either document should come from anywhere other than this
script's source JSONs (final_report.json, bootstrap_ci.json,
step_alignment_summary.json, rawr_analysis.json, baseline_report.json).
"""

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%" if LATEX else f"{100 * x:.{digits}f}%"


def f3(x):
    return f"{x:.3f}"


LATEX = False


def main() -> int:
    global LATEX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="evidence/gpt-4o")
    ap.add_argument("--format", choices=["text", "latex"], default="text")
    args = ap.parse_args()
    LATEX = args.format == "latex"

    model = os.path.basename(args.model_dir.rstrip("/"))
    ev = os.path.join(BASE, "evidence", model)
    fr = json.load(open(os.path.join(ev, "final_report.json")))
    ci = json.load(open(os.path.join(ev, "bootstrap_ci.json")))
    rawr = json.load(open(os.path.join(ev, "rawr_analysis.json")))
    summ = json.load(open(os.path.join(ev, "step_alignment_summary.json")))
    m = fr["main"]
    an = fr["analysis"]
    cov = an["parse_rates_and_coverage"]
    lb = an["lower_bound_A_consistent_full"]

    def cirange(key):
        c = ci[key]["ci"]["f1"]
        return f"[{c['ci_lower']:.3f}, {c['ci_upper']:.3f}]"

    print(f"=== {model}: canonical numbers (source: evidence/{model}/*.json) ===\n")

    print("--- Abstract / Table 1 ---")
    print(f"A consistent (full): P={f3(m['A_consistent_full']['precision'])} "
          f"R={f3(m['A_consistent_full']['recall'])} "
          f"F1={f3(m['A_consistent_full']['f1'])} CI {cirange('A_consistent_full')}")
    print(f"A consistent (pred): P={f3(m['A_consistent_pred']['precision'])} "
          f"R={f3(m['A_consistent_pred']['recall'])} "
          f"F1={f3(m['A_consistent_pred']['f1'])} CI {cirange('A_consistent_pred')}")
    print(f"A+A' combined (full): F1={f3(m['A_Ap_combined_full']['f1'])} "
          f"CI {cirange('A_Ap_combined_full')}")
    print(f"A+A' cascade (full): P={f3(m['A_Ap_cascade_full']['precision'])} "
          f"R={f3(m['A_Ap_cascade_full']['recall'])} "
          f"F1={f3(m['A_Ap_cascade_full']['f1'])} CI {cirange('A_Ap_cascade_full')} "
          f"(n_cascaded={m['n_cascaded']})")
    if m.get("self_consistency_full"):
        print(f"Self-consistency (full): P={f3(m['self_consistency_full']['precision'])} "
              f"R={f3(m['self_consistency_full']['recall'])} "
              f"F1={f3(m['self_consistency_full']['f1'])} "
              f"CI {cirange('self_consistency_full')}")
        ov = ci.get("ci_overlap_A_vs_SC", {})
        print(f"CI overlap A vs SC: non_overlapping={ov.get('non_overlapping')} "
              f"(A lower {ov.get('a_ci_lower')}, SC upper {ov.get('sc_ci_upper')})")
    print(f"String-diff (full): P={f3(m['string_diff_full']['precision'])} "
          f"R={f3(m['string_diff_full']['recall'])} F1={f3(m['string_diff_full']['f1'])}")

    print("\n--- Table 2 (ablation) ---")
    order = ["A_consistent_only", "A'_local_only", "A+A'_combined",
             "A+A'_cascade_filter", "A+A'_no_surface_ctrl", "string_diff_baseline"]
    for k in order:
        r = fr["ablation"][k]
        print(f"{k}: P={f3(r['precision'])} R={f3(r['recall'])} F1={f3(r['f1'])}")

    print("\n--- Section 3.1 (coverage & parse rates) ---")
    print(f"N_total certificates: {cov['n_total_certificates']}")
    print(f"N_evaluable (matched, non-ambiguous): {cov['n_evaluable_certificates']}")
    print(f"N_excluded by alignment: {cov['n_excluded_certificates']}")
    print(f"parse rate orig/sem/local/sur: {pct(cov['parse_rate_orig'])} / "
          f"{pct(cov['parse_rate_sem'])} / {pct(cov['parse_rate_local'])} / "
          f"{pct(cov['parse_rate_sur'])}")
    print(f"combined-protocol UNPARSEABLE: {cov['combined_unparseable']} "
          f"({pct(cov['combined_unparseable_rate'])} of evaluable)")
    print(f"A-consistent UNPARSEABLE: {cov['a_consistent_unparseable']} "
          f"({pct(cov['a_consistent_unparseable_rate'])} of evaluable); "
          f"UNSTABLE: {cov['a_consistent_unstable']}")
    sa = an["step_alignment"]
    print(f"CoT steps: {sa['total_cot_steps']}, proof steps: {sa['total_proof_steps']}, "
          f"matched: {sa['matched_cot_steps']}, unmatched CoT: {sa['unmatched_cot_steps']} "
          f"(problems: {len(sa['problems_with_unmatched_cot_steps'])}), "
          f"unmatched proof: {sa['unmatched_proof_steps']} "
          f"(problems: {len(sa['problems_with_unmatched_proof_steps'])}), "
          f"ambiguous: {sa['ambiguous_steps']}, "
          f"excluded problems: {len(sa['primary_metric_excluded_problems'])}")

    print("\n--- Section 3.4 (FP + RAWR + misrepresentation) ---")
    fp = an["fp"]
    print(f"semantic-delta FP candidates: {fp['n_semantic_delta_candidates']}")
    print(f"metric-counted FPs (Table-1 A consistent): {fp['n_metric_counted_fp']}")
    print(f"outside metric set (unparseable surface): {fp['n_outside_metric_set']}")
    print(f"candidate type counts: {fp['type_counts']}")
    print(f"metric-counted type counts: {fp['metric_counted_type_counts']}")
    ra = an["rawr"]
    print(f"RAWR: {ra['total_rawr']}/{ra['total_correct']} "
          f"({pct(ra['rawr_rate'])}) evaluator={ra['metric_policy']}")
    print(f"  structural-only: {ra['structural_only']}, "
          f"has-predicate: {ra['has_predicate_rawr']}, "
          f"insensitive deps total: {ra['total_insensitive_deps']}, "
          f"SC perfect on: {ra['self_consistency_perfect_cases']}")
    print(f"misrepresentation cases: {an['misrepresentation']['total']}")

    print("\n--- Section 5 (coverage & lower bound) ---")
    print(f"lower bound (A consistent full): P={f3(lb['precision'])} "
          f"R={f3(lb['recall'])} F1={f3(lb['f1'])}")
    print(f"  extra FN from UNPARSEABLE/UNSTABLE: "
          f"{lb['extra_fn_from_unparseable_or_unstable']}; "
          f"from unscored proof deps: {lb['extra_fn_from_unscored_proof_dependencies']}")

    print("\n--- Appendix C candidate (first RAWR case) ---")
    if rawr["cases"]:
        c = rawr["cases"][0]
        d = c["insensitive_dependencies"][0]
        print(f"problem {c['problem_id']}: CoT step {d['step_id']} "
              f"(proof step {d['matched_proof_step_id']}) premise {d['premise_id']} "
              f"({d['premise_type']}): '{d['premise_text']}'")
        print(f"  phi_original={d['phi_original']} phi_semantic={d['phi_semantic']} "
              f"phi_local={d['phi_local']}")
        print(f"  self_consistency_perfect={c['self_consistency_perfect']}")

    print("\n--- Chain length (A consistent full) ---")
    for row in an["chain_length"]:
        print(f"  length {row['chain_length']}: n={row['n_problems']} "
              f"P={f3(row['precision'])} R={f3(row['recall'])} F1={f3(row['f1'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
