"""Analyze how chain length (number of reasoning hops) affects
grounding detection accuracy.

Uses the A-consistent evaluator with aligned CoT-keyed ground truth,
matching the Table 1 "A consistent (full)" configuration.
"""

from collections import defaultdict

from src.ground_truth import (
    POSITIVE_VERDICTS,
    PRIMARY_SKIP_VERDICTS,
    _compute_prf1,
    extract_ground_truth_dependencies_for_cot_steps,
)


def _judge_consistent_only(cert: dict) -> str:
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


def analyze_chain_length(certificates, problems, cot_steps_by_pid):
    """Group by chain length, compute P/R/F1 per group (A consistent, full)."""
    gt_map = {}
    chain_lengths = {}
    for p in problems:
        pid = p["problem_id"]
        chain_lengths[pid] = len(p["proof_tree"])
        gt, _ = extract_ground_truth_dependencies_for_cot_steps(
            p, cot_steps_by_pid.get(pid, []))
        for (sid, prem), dep in gt.items():
            gt_map[(pid, sid, prem)] = dep

    certs_by_problem = defaultdict(list)
    for cert in certificates:
        certs_by_problem[cert["problem_id"]].append(cert)

    length_groups = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "pids": set()})

    for pid, certs in certs_by_problem.items():
        cl = chain_lengths.get(pid, 0)
        grp = length_groups[cl]
        grp["pids"].add(pid)

        for cert in certs:
            if cert.get("evaluation_excluded"):
                continue
            verdict = _judge_consistent_only(cert)
            if verdict in PRIMARY_SKIP_VERDICTS:
                continue
            gt_dep = gt_map.get((pid, cert["step_id"], cert["premise_id"]))
            if gt_dep is None:
                continue
            predicted = verdict in POSITIVE_VERDICTS
            if predicted and gt_dep:
                grp["tp"] += 1
            elif predicted and not gt_dep:
                grp["fp"] += 1
            elif not predicted and gt_dep:
                grp["fn"] += 1

    results = []
    for length in sorted(length_groups.keys()):
        g = length_groups[length]
        p, r, f1 = _compute_prf1(g["tp"], g["fp"], g["fn"])
        results.append({
            "chain_length": length,
            "n_problems": len(g["pids"]),
            "tp": g["tp"], "fp": g["fp"], "fn": g["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })

    return results
