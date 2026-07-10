#!/usr/bin/env python3
"""Re-collect Self-Consistency baseline (5 samples x 50 problems).

Writes to baselines/self_consistency/<out-name>/ so evidence_loader
(which keys on the evidence model-dir basename, e.g. gpt-4o) finds them.
Records the actual API model id in each file for auditability.

Usage:
  OPENAI_API_KEY=... python scripts/run_sc_baseline.py \
      --model gpt-4o-2024-08-06 --out-name gpt-4o --workers 6
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys
import threading

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.baseline_self_consistency import (  # noqa: E402
    evaluate_baseline,
    run_self_consistency,
)

_print_lock = threading.Lock()


def load_problems() -> list[dict]:
    problems = []
    for f in sorted(glob.glob(os.path.join(BASE, "data", "prontoqa_50", "p*.json"))):
        if "proof_trees" in f:
            continue
        problem = json.load(open(f))
        problem.setdefault("problem_id", os.path.splitext(os.path.basename(f))[0])
        problems.append(problem)
    return problems


def process_one(problem: dict, model: str, out_dir: str, api_key: str,
                n_samples: int, temperature: float) -> str:
    pid = problem["problem_id"]
    br = run_self_consistency(
        problem, api_key, model=model,
        n_samples=n_samples, temperature=temperature,
    )
    # Match shipped schema; add model for audit (old files lacked it).
    save_data = {
        "problem_id": pid,
        "model": model,
        "n_samples": n_samples,
        "temperature": temperature,
        "step_verdicts": {str(k): v for k, v in br["step_verdicts"].items()},
    }
    path = os.path.join(out_dir, f"{pid}_samples.json")
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    with _print_lock:
        print(f"  [done] {pid}", flush=True)
    return pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--out-name", default="gpt-4o",
                    help="Directory name under baselines/self_consistency/")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    api_key = os.environ["OPENAI_API_KEY"]
    problems = load_problems()
    out_dir = os.path.join(BASE, "baselines", "self_consistency", args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    # Remove stale (old-pack) samples so we never mix subjects.
    for old in glob.glob(os.path.join(out_dir, "p*_samples.json")):
        os.remove(old)

    print(f"SC baseline: {len(problems)} problems x {args.n_samples} samples, "
          f"model={args.model}, out={out_dir}, workers={args.workers}", flush=True)

    errors = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_one, p, args.model, out_dir, api_key,
                      args.n_samples, args.temperature): p["problem_id"]
            for p in problems
        }
        for fut in cf.as_completed(futs):
            pid = futs[fut]
            try:
                fut.result()
            except Exception as e:
                errors.append((pid, str(e)))
                with _print_lock:
                    print(f"  [ERROR] {pid}: {e}", flush=True)

    if errors:
        print(f"FAILED: {len(errors)} problems")
        for pid, err in errors:
            print(f"  {pid}: {err}")
        return 1

    # Evaluate + write baseline_report.json
    samples = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "p*_samples.json"))):
        d = json.load(open(f))
        samples[d["problem_id"]] = d
    report = {
        "full": evaluate_baseline(samples, problems, gt_type="full"),
        "pred_only": evaluate_baseline(samples, problems, gt_type="pred"),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "model": args.model,
    }
    with open(os.path.join(out_dir, "baseline_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        "model": args.model,
        "n_files": len(samples),
        "full_f1": report["full"]["f1"],
        "pred_f1": report["pred_only"]["f1"],
        "full_prf": {
            "P": report["full"]["precision"],
            "R": report["full"]["recall"],
            "F1": report["full"]["f1"],
        },
    }, indent=2))
    print("SC baseline complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
