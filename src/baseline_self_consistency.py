"""Self-consistency baseline (Wang+ 2023) adapted for grounding detection.

Key insight: self-consistency checks answer agreement across samples,
but CANNOT detect "right answer, wrong reasoning" — a step may
consistently produce the correct answer while not actually depending
on the stated premise.

We adapt it to step-level: for each step, check if the conclusion
is consistent across n samples. If consistent → predict "depends on all
cited premises". If inconsistent → predict "ungrounded".
"""

import json
import os
import sys
import time
from collections import Counter

# import config  # paths configured at runtime

# Phase 1 dependencies (LLM calls) — lazy imported inside run_self_consistency()
# to avoid requiring openai/anthropic for Phase 2 (evaluation-only) use.
# from src.llm_runner import run_llm, build_prompt
# from src.normalizer import parse_cot_steps

from src.ground_truth import _compute_prf1, _proof_keyed_gt


def run_self_consistency(problem, api_key, model="gpt-4o",
                         n_samples=5, temperature=0.7):
    """Run n_samples with temperature>0 on original premises.

    Returns per-step consistency verdicts.
    Requires openai/anthropic packages (Phase 1 only).
    """
    # Lazy imports — only needed for Phase 1 (LLM calls)
    from src.llm_runner import run_llm, build_prompt
    from src.normalizer import parse_cot_steps

    prompt = build_prompt(problem["premises"], problem["question"])

    samples = []
    for i in range(n_samples):
        result = run_llm(prompt, api_key, model=model, temperature=temperature)
        steps = parse_cot_steps(result["raw_response"])
        samples.append({"raw": result, "steps": steps})
        time.sleep(0.2)

    # Collect all step IDs across samples
    all_step_ids = set()
    for s in samples:
        for step in s["steps"]:
            all_step_ids.add(step["step_id"])

    step_verdicts = {}
    for sid in sorted(all_step_ids):
        conclusions = []
        for s in samples:
            matching = [st for st in s["steps"] if st["step_id"] == sid]
            if matching and matching[0]["parse_status"] == "OK":
                conclusions.append(matching[0]["normalized"])
            else:
                conclusions.append(None)

        valid = [c for c in conclusions if c is not None]
        if not valid:
            step_verdicts[sid] = {
                "consistent": False, "agreement": 0.0,
                "majority_conclusion": None, "n_valid": 0,
            }
            continue

        most_common, count = Counter(valid).most_common(1)[0]
        agreement = count / len(valid)
        step_verdicts[sid] = {
            "consistent": agreement >= 0.8,
            "agreement": round(agreement, 3),
            "majority_conclusion": most_common,
            "n_valid": len(valid),
            "n_unique": len(set(valid)),
        }

    return {
        "problem_id": problem["problem_id"],
        "n_samples": n_samples,
        "temperature": temperature,
        "step_verdicts": step_verdicts,
        "samples": samples,
    }


def evaluate_baseline(baseline_results: dict, problems: list[dict],
                      gt_type: str = "full") -> dict:
    """Convert self-consistency results to P/R/F1 with content-matched steps.

    Baseline logic (generous to baseline):
      - If a step is consistent -> predict ALL premises are dependencies
      - If a step is inconsistent -> predict NO premise is a dependency

    Sampled CoT step numbers are NOT proof-tree step numbers (models may
    prepend restatement steps), so each proof-tree step is matched to the
    baseline step whose majority conclusion equals that proof step's
    conclusion. Unmatched baseline steps and unmatched proof steps are
    excluded from the primary baseline metric and reported for coverage;
    conclusion collisions are treated as ambiguous and likewise excluded.

    Args:
        gt_type: "full" for all proof-tree deps, "pred" for predicate-only
    """
    tp = fp = fn = tn = 0
    per_problem: dict[str, dict[str, int]] = {}
    baseline_alignment_excluded = 0   # baseline steps matching no proof step
    unmatched_proof_steps = 0         # proof steps matching no baseline step
    ambiguous_matches = 0             # collisions (dup conclusions / many-to-one)

    for problem in problems:
        pid = problem["problem_id"]
        br = baseline_results.get(pid)
        if br is None:
            continue

        gt = _proof_keyed_gt(problem, pred_only=(gt_type == "pred"))
        premise_ids = [p["id"] for p in problem["premises"]]

        # proof conclusion -> [proof step ids]
        conclusion_to_proof: dict[str, list[int]] = {}
        for entry in problem["proof_tree"]:
            conclusion_to_proof.setdefault(entry["conclusion"], []).append(entry["step"])

        # match baseline steps to proof steps by majority conclusion
        proof_claimed_by: dict[int, list] = {}
        for raw_sid, sv in br["step_verdicts"].items():
            conclusion = sv.get("majority_conclusion")
            proof_ids = conclusion_to_proof.get(conclusion) if conclusion else None
            if not proof_ids:
                baseline_alignment_excluded += 1
                continue
            if len(proof_ids) > 1:
                ambiguous_matches += 1
                continue
            proof_claimed_by.setdefault(proof_ids[0], []).append(sv)

        matched: dict[int, dict] = {}
        for proof_id, svs in proof_claimed_by.items():
            if len(svs) > 1:
                ambiguous_matches += len(svs)
                continue
            matched[proof_id] = svs[0]

        for entry in problem["proof_tree"]:
            proof_id = entry["step"]
            sv = matched.get(proof_id)
            if sv is None:
                unmatched_proof_steps += 1
                continue  # excluded from primary baseline metric
            predicted_positive = bool(sv["consistent"])
            counts = per_problem.setdefault(pid, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            for prem_id in premise_ids:
                is_positive = gt.get((proof_id, prem_id), False)
                if predicted_positive and is_positive:
                    tp += 1
                    counts["tp"] += 1
                elif predicted_positive and not is_positive:
                    fp += 1
                    counts["fp"] += 1
                elif not predicted_positive and is_positive:
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
        "gt_type": gt_type,
        "baseline_alignment_excluded": baseline_alignment_excluded,
        "unmatched_proof_steps": unmatched_proof_steps,
        "ambiguous_matches": ambiguous_matches,
        "per_problem": per_problem,
        "alignment_policy": "match_majority_conclusion_to_proof_conclusion; exclude unmatched and ambiguous",
        "note": "Generous baseline: consistent steps predict ALL premises as deps",
    }


def run_and_evaluate(problems, api_key, model="gpt-4o",
                     n_samples=5, temperature=0.7):
    """Full pipeline: run baseline + evaluate. Requires openai/anthropic (Phase 1)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_results = {}
    baseline_dir = os.path.join(base, "baselines", "self_consistency", model)
    os.makedirs(baseline_dir, exist_ok=True)

    for i, problem in enumerate(problems):
        pid = problem["problem_id"]
        cache_file = os.path.join(baseline_dir, f"{pid}_samples.json")

        # Cache check
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                br = json.load(f)
            baseline_results[pid] = br
            continue

        print(f"    baseline {pid} ({i+1}/{len(problems)})...")
        br = run_self_consistency(
            problem, api_key, model=model,
            n_samples=n_samples, temperature=temperature,
        )

        # Save (without raw samples to save space — keep step_verdicts)
        save_data = {
            "problem_id": pid,
            "n_samples": n_samples,
            "temperature": temperature,
            "step_verdicts": br["step_verdicts"],
        }
        with open(cache_file, "w") as f:
            json.dump(save_data, f, indent=2)

        baseline_results[pid] = br
        time.sleep(0.3)

    # Evaluate
    eval_full = evaluate_baseline(baseline_results, problems, gt_type="full")
    eval_pred = evaluate_baseline(baseline_results, problems, gt_type="pred")

    return {
        "results": baseline_results,
        "eval_full": eval_full,
        "eval_pred": eval_pred,
    }


def print_baseline_report(eval_full, eval_pred):
    """Print baseline comparison."""
    print("\n" + "=" * 70)
    print("  PASSIVE BASELINE: Self-Consistency (Wang+ 2023)")
    print("=" * 70)
    print(f"\n  {'Metric':<20} {'Full':>10} {'Pred-only':>10}")
    print("  " + "-" * 40)
    print(f"  {'Precision':<20} {eval_full['precision']:>10.4f} {eval_pred['precision']:>10.4f}")
    print(f"  {'Recall':<20} {eval_full['recall']:>10.4f} {eval_pred['recall']:>10.4f}")
    print(f"  {'F1':<20} {eval_full['f1']:>10.4f} {eval_pred['f1']:>10.4f}")
    print(f"  {'TP':<20} {eval_full['tp']:>10} {eval_pred['tp']:>10}")
    print(f"  {'FP':<20} {eval_full['fp']:>10} {eval_pred['fp']:>10}")
    print(f"  {'FN':<20} {eval_full['fn']:>10} {eval_pred['fn']:>10}")
    print(f"\n  Note: {eval_full['note']}")
    print("=" * 70)
