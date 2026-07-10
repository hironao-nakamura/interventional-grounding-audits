#!/usr/bin/env python3
"""Phase 2: deterministic certificate generation from saved Phase 1 raw outputs.

Usage:
  python scripts/run_phase2.py --model-dir evidence/gpt-4o --overwrite
  python scripts/run_phase2.py --model-dir evidence/claude-sonnet-4-5 --overwrite

For each problem:
  1. Parse the original CoT (src.normalizer.parse_cot_steps).
  2. Align CoT steps to proof-tree steps by normalized conclusion
     (src.step_alignment.align_cot_to_proof).
  3. For every premise Pj, parse semantic_Pj / local_Pj / surface_Pj outputs.
  4. Align original vs probed steps (src.aligner.align_steps_content_aware).
  5. Detect citations (src.citation_detector.detect_citation).
  6. Audit each (step, premise) pair (src.auditor.ReasoningAuditor.audit_step).
  7. Attach GT-alignment metadata (matched_proof_step_id, evaluation_excluded,
     exclusion_reason, primary_metric_excluded_problem, ...).
  8. Save evidence/<model>/pXXX/certificates/cert_Si_Pj.json.

No API calls; fully deterministic given the saved raw outputs.
"""

import argparse
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.aligner import align_steps_content_aware  # noqa: E402
from src.auditor import ReasoningAuditor  # noqa: E402
from src.citation_detector import detect_citation  # noqa: E402
from src.normalizer import parse_cot_steps  # noqa: E402
from src.step_alignment import align_cot_to_proof  # noqa: E402

PROBE_FILES = {
    "semantic": "semantic_{pid}.json",
    "local": "local_{pid}.json",
    "surface": "surface_{pid}.json",
}


def load_problems() -> list[dict]:
    problems = []
    for f in sorted(glob.glob(os.path.join(BASE, "data", "prontoqa_50", "p*.json"))):
        if "proof_trees" in f:
            continue
        problem = json.load(open(f))
        problem.setdefault("problem_id", os.path.splitext(os.path.basename(f))[0])
        problems.append(problem)
    return problems


def _aligned_probed_step(alignment_rows: list[dict], step_id: int) -> tuple[dict | None, bool]:
    """Return (probed step, aligned?) for a given original step id."""
    for row in alignment_rows:
        if row["original"] is not None and row["original"]["step_id"] == step_id:
            if row["alignment"] == "MATCHED" and row["probed"] is not None:
                return row["probed"], True
            return None, False
    return None, False


def process_problem(problem: dict, model_dir: str, overwrite: bool) -> int:
    pid = problem["problem_id"]
    prob_dir = os.path.join(model_dir, pid)
    cert_dir = os.path.join(prob_dir, "certificates")

    orig_path = os.path.join(prob_dir, "original_cot.json")
    raw = json.load(open(orig_path))["raw_response"]
    cot_steps = parse_cot_steps(raw)

    gt_alignment = align_cot_to_proof(problem, cot_steps)

    if os.path.isdir(cert_dir):
        existing = glob.glob(os.path.join(cert_dir, "cert_*.json"))
        if existing and not overwrite:
            raise FileExistsError(
                f"{cert_dir} already contains certificates; pass --overwrite.")
        for f in existing:
            os.remove(f)
    os.makedirs(cert_dir, exist_ok=True)

    # Parse and align every probe output once per premise.
    probe_steps: dict[str, dict[str, list[dict]]] = {}
    for premise in problem["premises"]:
        prem_id = premise["id"]
        per_type = {}
        for probe_type, tmpl in PROBE_FILES.items():
            path = os.path.join(prob_dir, "probes", tmpl.format(pid=prem_id))
            if os.path.exists(path):
                probed_raw = json.load(open(path))["raw_response"]
                probed = parse_cot_steps(probed_raw)
            else:
                probed = []
            per_type[probe_type] = align_steps_content_aware(cot_steps, probed)
        probe_steps[prem_id] = per_type

    auditor = ReasoningAuditor()
    n_written = 0
    for step in cot_steps:
        sid = step["step_id"]
        step_with_pid = dict(step)
        step_with_pid["problem_id"] = pid
        matched_proof = gt_alignment.cot_to_proof.get(sid)
        step_unmatched = sid in gt_alignment.unmatched_cot_steps
        step_ambiguous = sid in gt_alignment.ambiguous_cot_steps

        for premise in problem["premises"]:
            prem_id = premise["id"]
            per_type = probe_steps[prem_id]

            sem_step, aligned_sem = _aligned_probed_step(per_type["semantic"], sid)
            local_step, aligned_local = _aligned_probed_step(per_type["local"], sid)
            sur_step, aligned_sur = _aligned_probed_step(per_type["surface"], sid)

            citation = detect_citation(step["raw_text"], premise["text"], prem_id)

            cert = auditor.audit_step(
                step=step_with_pid,
                premise_id=prem_id,
                phi_sem=sem_step["normalized"] if sem_step else None,
                phi_local=local_step["normalized"] if local_step else None,
                phi_sur=sur_step["normalized"] if sur_step else None,
                parse_ok_sem=bool(sem_step and sem_step["parse_status"] == "OK"),
                parse_ok_local=bool(local_step and local_step["parse_status"] == "OK"),
                parse_ok_sur=bool(sur_step and sur_step["parse_status"] == "OK"),
                aligned_sem=aligned_sem,
                aligned_local=aligned_local,
                aligned_sur=aligned_sur,
                citation_detected=citation["cited"],
            )

            cert["citation_detected"] = citation["cited"]
            cert["citation_type"] = citation["citation_type"]
            cert["citation_evidence"] = citation["evidence"]

            # --- GT alignment metadata (Section 3.5 of the release fix spec) ---
            cert["cot_step_id"] = sid
            if gt_alignment.primary_metric_excluded_problem:
                cert["matched_proof_step_id"] = matched_proof
                cert["evaluation_excluded"] = True
                cert["exclusion_reason"] = "ambiguous_alignment_problem"
                cert["primary_metric_excluded_problem"] = True
                cert["primary_metric_exclusion_reason"] = "problem_has_ambiguous_alignment"
            elif step_unmatched or step_ambiguous or matched_proof is None:
                cert["matched_proof_step_id"] = None
                cert["evaluation_excluded"] = True
                cert["exclusion_reason"] = "unmatched_cot_step"
                cert["primary_metric_excluded_problem"] = False
                cert["primary_metric_exclusion_reason"] = None
            else:
                cert["matched_proof_step_id"] = matched_proof
                cert["evaluation_excluded"] = False
                cert["exclusion_reason"] = None
                cert["primary_metric_excluded_problem"] = False
                cert["primary_metric_exclusion_reason"] = None

            out = os.path.join(cert_dir, f"cert_S{sid}_{prem_id}.json")
            with open(out, "w") as f:
                json.dump(cert, f, indent=2)
            n_written += 1

    return n_written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="e.g. evidence/gpt-4o")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model_dir = args.model_dir if os.path.isabs(args.model_dir) \
        else os.path.join(BASE, args.model_dir)

    problems = load_problems()
    total = 0
    for problem in problems:
        n = process_problem(problem, model_dir, args.overwrite)
        print(f"  {problem['problem_id']}: {n} certificates", flush=True)
        total += n
    print(f"Phase 2 complete: {total} certificates written to {args.model_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
