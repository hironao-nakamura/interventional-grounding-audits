#!/usr/bin/env python3
"""Structural validator for the evidence pack.

Usage:
  python src/validator.py --root . --models gpt-4o claude-sonnet-4-5

Checks (per model unless noted):
   1. data/prontoqa_50 contains exactly 50 problem files (once).
   2. Every problem has problem_meta.json in the evidence tree.
   3. Every problem has original_cot.json.
   4. Every premise has semantic/local/surface probe files.
   5. Stored sha256 matches SHA256(raw_response.encode('utf-8')).
   6. Probe raw outputs store prompt + modified_premises (WARNING if absent:
      accepted-version Phase 1 outputs predate this requirement).
   7. Stored prompts/modified_premises contain no doubled fresh-symbol
      prefixes ('zqzq'), when those fields are present.
   8. Certificate count per problem == parsed CoT steps x premises.
   9. Certificate alignment metadata matches a fresh align_cot_to_proof run.
  10. Certificate schema contains all required fields.
  11. verdict in the allowed set.
  12. GROUNDED implies surface_delta is False (else INPUT-SENSITIVE/flag).
  13. decision_rule is consistent with the verdict.
  14. step_alignment_summary.json exists and matches a fresh recomputation.
  15. final_report.json exists with the canonical schema.
  16. bootstrap_ci.json point estimates equal final_report main rows.
  17. Disclosure flags: unmatched proof steps / ambiguous problems imply
      requires_*_disclosure = true in the alignment summary (WARNING-level
      consistency check).

Exit status: 0 and "RESULT: PASS" if no errors (warnings allowed);
1 and "RESULT: FAIL" otherwise. README numeric consistency is handled by
scripts/check_readme_numbers.py, not here.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

ALLOWED_VERDICTS = {
    "GROUNDED", "INSENSITIVE", "INPUT-SENSITIVE",
    "UNSTABLE", "UNPARSEABLE", "CASCADE",
}

REQUIRED_CERT_FIELDS = {
    "problem_id", "step_id", "cot_step_id", "matched_proof_step_id",
    "evaluation_excluded", "exclusion_reason",
    "primary_metric_excluded_problem", "primary_metric_exclusion_reason",
    "premise_id", "phi_original", "phi_semantic", "phi_local", "phi_surface",
    "semantic_delta", "local_delta", "surface_delta",
    "verdict", "verdict_consistent", "decision_rule", "confidence", "flags",
    "parse_ok_orig", "parse_ok_sem", "parse_ok_local", "parse_ok_sur",
}

PROBE_PREFIXES = ("semantic", "local", "surface")


class Reporter:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)
        print(f"  [ERROR] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN]  {msg}")


def load_problems(root):
    problems = []
    for f in sorted(glob.glob(os.path.join(root, "data", "prontoqa_50", "p*.json"))):
        if "proof_trees" in f:
            continue
        p = json.load(open(f))
        p.setdefault("problem_id", os.path.splitext(os.path.basename(f))[0])
        problems.append(p)
    return problems


def validate_model(root, model, problems, rep: Reporter):
    sys.path.insert(0, root)
    from src.normalizer import parse_cot_steps
    from src.step_alignment import align_cot_to_proof

    evidence = os.path.join(root, "evidence", model)
    if not os.path.isdir(evidence):
        rep.error(f"{model}: evidence directory missing: {evidence}")
        return

    print(f"\n--- Validating model: {model} ---")
    n_missing_prompt_fields = 0
    for problem in problems:
        pid = problem["problem_id"]
        pdir = os.path.join(evidence, pid)

        # 2-3. per-problem files
        meta = os.path.join(pdir, "problem_meta.json")
        orig = os.path.join(pdir, "original_cot.json")
        if not os.path.exists(meta):
            rep.error(f"{model}/{pid}: problem_meta.json missing")
        if not os.path.exists(orig):
            rep.error(f"{model}/{pid}: original_cot.json missing")
            continue

        orig_data = json.load(open(orig))
        raw = orig_data.get("raw_response", "")
        # 5. sha integrity (original)
        want = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if orig_data.get("sha256") != want:
            rep.error(f"{model}/{pid}: original_cot sha256 mismatch")

        cot_steps = parse_cot_steps(raw)
        premise_ids = [pr["id"] for pr in problem["premises"]]

        # 4-7. probes
        for prem in premise_ids:
            for prefix in PROBE_PREFIXES:
                path = os.path.join(pdir, "probes", f"{prefix}_{prem}.json")
                if not os.path.exists(path):
                    rep.error(f"{model}/{pid}: missing probe {prefix}_{prem}.json")
                    continue
                d = json.load(open(path))
                want = hashlib.sha256(
                    d.get("raw_response", "").encode("utf-8")).hexdigest()
                if d.get("sha256") != want:
                    rep.error(f"{model}/{pid}: {prefix}_{prem} sha256 mismatch")
                if "prompt" not in d or "modified_premises" not in d:
                    n_missing_prompt_fields += 1
                else:
                    joined = d["prompt"] + " " + " ".join(
                        pr["text"] for pr in d["modified_premises"])
                    if "zqzq" in joined:
                        rep.error(
                            f"{model}/{pid}: doubled fresh-symbol prefix in "
                            f"{prefix}_{prem} prompt/premises")

        # 8. certificate count
        cert_files = sorted(glob.glob(
            os.path.join(pdir, "certificates", "cert_*.json")))
        expected = len(cot_steps) * len(premise_ids)
        if len(cert_files) != expected:
            rep.error(f"{model}/{pid}: {len(cert_files)} certificates, "
                      f"expected {expected} (steps x premises)")

        # 9-13. certificate contents
        alignment = align_cot_to_proof(problem, cot_steps)
        for cf in cert_files:
            c = json.load(open(cf))
            name = f"{model}/{pid}/{os.path.basename(cf)}"
            missing = REQUIRED_CERT_FIELDS - c.keys()
            if missing:
                rep.error(f"{name}: missing fields {sorted(missing)}")
                continue
            if c["verdict"] not in ALLOWED_VERDICTS:
                rep.error(f"{name}: invalid verdict {c['verdict']}")
            sid = c["step_id"]
            if alignment.primary_metric_excluded_problem:
                if not (c["evaluation_excluded"]
                        and c["exclusion_reason"] == "ambiguous_alignment_problem"):
                    rep.error(f"{name}: ambiguous problem not marked excluded")
            elif sid in alignment.unmatched_cot_steps:
                if not (c["evaluation_excluded"]
                        and c["exclusion_reason"] == "unmatched_cot_step"
                        and c["matched_proof_step_id"] is None):
                    rep.error(f"{name}: unmatched CoT step metadata wrong")
            else:
                if c["evaluation_excluded"]:
                    rep.error(f"{name}: matched step marked excluded")
                if c["matched_proof_step_id"] != alignment.cot_to_proof.get(sid):
                    rep.error(f"{name}: matched_proof_step_id "
                              f"{c['matched_proof_step_id']} != alignment "
                              f"{alignment.cot_to_proof.get(sid)}")
            if (c["verdict"] == "GROUNDED" and c.get("surface_delta")
                    and "SURFACE_DELTA_OVERRIDDEN" not in (c.get("flags") or [])):
                rep.error(f"{name}: GROUNDED with surface_delta and no flag")
            dr = c.get("decision_rule", "")
            v = c["verdict"]
            rule_map = {
                "GROUNDED": ("R_GROUNDED",),
                "INPUT-SENSITIVE": ("R_INPUT_SENSITIVE",),
                "INSENSITIVE": ("R_INSENSITIVE", "R_MISREPRESENT_CITED"),
                "UNSTABLE": ("R_UNSTABLE",),
                "UNPARSEABLE": ("R_UNPARSEABLE",),
                "CASCADE": ("R_CASCADE",),
            }
            if not any(dr.startswith(pref) for pref in rule_map[v]):
                rep.error(f"{name}: decision_rule '{dr}' inconsistent with {v}")

    if n_missing_prompt_fields:
        rep.warn(
            f"{model}: {n_missing_prompt_fields} probe outputs lack "
            f"prompt/modified_premises (accepted-version Phase 1 raw outputs; "
            f"rerun scripts/run_phase1.py to regenerate with full metadata)")

    # 14. alignment summary freshness
    summ_path = os.path.join(evidence, "step_alignment_summary.json")
    if not os.path.exists(summ_path):
        rep.error(f"{model}: step_alignment_summary.json missing")
    else:
        summ = json.load(open(summ_path))
        tot_unmatched = tot_amb = tot_up = 0
        excl = []
        from src.evidence_loader import load_cot_steps_by_pid
        cot_by_pid = load_cot_steps_by_pid(model, base=root)
        for problem in problems:
            a = align_cot_to_proof(problem, cot_by_pid[problem["problem_id"]])
            tot_unmatched += len(a.unmatched_cot_steps)
            tot_amb += len(a.ambiguous_cot_steps)
            tot_up += len(a.unmatched_proof_steps)
            if a.primary_metric_excluded_problem:
                excl.append(problem["problem_id"])
        for key, val in [("unmatched_cot_steps", tot_unmatched),
                         ("ambiguous_steps", tot_amb),
                         ("unmatched_proof_steps", tot_up),
                         ("primary_metric_excluded_problems", excl)]:
            if summ.get(key) != val:
                rep.error(f"{model}: step_alignment_summary.{key} = "
                          f"{summ.get(key)} but recomputation gives {val}")
        # 17. disclosure flags (warning-level)
        if (tot_up or excl) and not summ.get("requires_lower_bound_disclosure"):
            rep.warn(f"{model}: unmatched proof steps present but "
                     f"requires_lower_bound_disclosure is false")
        if excl and not summ.get("requires_alignment_disclosure"):
            rep.warn(f"{model}: excluded problems present but "
                     f"requires_alignment_disclosure is false")

    # 15. final report schema
    fr_path = os.path.join(evidence, "final_report.json")
    if not os.path.exists(fr_path):
        rep.error(f"{model}: final_report.json missing")
        return
    fr = json.load(open(fr_path))
    for section, key in [("main", "A_consistent_full"),
                         ("main", "A_consistent_pred"),
                         ("main", "A_Ap_cascade_full"),
                         ("main", "self_consistency_full"),
                         ("analysis", "rawr"),
                         ("analysis", "misrepresentation"),
                         ("analysis", "step_alignment"),
                         ("analysis", "lower_bound_A_consistent_full")]:
        if key not in fr.get(section, {}):
            rep.error(f"{model}: final_report.{section}.{key} missing")

    # 16. bootstrap point estimates == final report
    ci_path = os.path.join(evidence, "bootstrap_ci.json")
    if not os.path.exists(ci_path):
        rep.error(f"{model}: bootstrap_ci.json missing")
    else:
        ci = json.load(open(ci_path))
        pairs = [("A_consistent_full", "A_consistent_full"),
                 ("A_consistent_pred", "A_consistent_pred"),
                 ("A_Ap_cascade_full", "A_Ap_cascade_full")]
        if fr["main"].get("self_consistency_full"):
            pairs.append(("self_consistency_full", "self_consistency_full"))
        for ci_key, fr_key in pairs:
            if ci_key not in ci:
                rep.error(f"{model}: bootstrap_ci.{ci_key} missing")
                continue
            if abs(ci[ci_key]["point"]["f1"] - fr["main"][fr_key]["f1"]) > 1e-9:
                rep.error(f"{model}: bootstrap point f1 for {ci_key} != "
                          f"final_report main.{fr_key}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--models", nargs="+", default=["gpt-4o"])
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rep = Reporter()

    print(f"Validating evidence pack at {args.root}")
    problems = load_problems(root)
    # 1. dataset count
    if len(problems) != 50:
        rep.error(f"data/prontoqa_50 has {len(problems)} problems, expected 50")
    else:
        print(f"  [OK] 50 problems in data/prontoqa_50")

    for model in args.models:
        validate_model(root, model, problems, rep)

    print(f"\nErrors: {len(rep.errors)}  Warnings: {len(rep.warnings)}")
    if rep.errors:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
