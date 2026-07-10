#!/usr/bin/env python3
"""Full Phase 1 rerun (all 50 problems) with a thread pool for speed.

Produces byte-identical file layout to scripts/run_phase1.py (same paths,
same JSON schema per file) so Phase 2 / recompute_reports.py work unmodified.
Concurrency only affects wall-clock time, not the saved data.

Usage:
  OPENAI_API_KEY=... python scripts/full_phase1_concurrent.py \
      --model gpt-4o-2024-08-06 --out evidence/gpt-4o --workers 6
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import sys
import threading

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.llm_runner import run_original, run_probed, save_llm_result  # noqa: E402
from src.probe_generator import generate_all_probes  # noqa: E402

PROBE_FILENAME_PREFIX = {
    "semantic": "semantic",
    "local_semantic": "local",
    "surface": "surface",
}

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


def check_sha256(result: dict) -> dict:
    result["sha256"] = hashlib.sha256(result["raw_response"].encode("utf-8")).hexdigest()
    return result


def process_problem(problem: dict, model: str, out: str, api_key: str) -> int:
    pid = problem["problem_id"]
    prob_dir = os.path.join(out, pid)
    probe_dir = os.path.join(prob_dir, "probes")
    os.makedirs(probe_dir, exist_ok=True)

    with open(os.path.join(prob_dir, "problem_meta.json"), "w") as f:
        json.dump(problem, f, indent=2)

    result = check_sha256(run_original(problem, api_key, model=model))
    save_llm_result(result, os.path.join(prob_dir, "original_cot.json"))

    n = 1
    probes = generate_all_probes(problem)
    for premise_id, probe_set in probes.items():
        for probe_type, prefix in PROBE_FILENAME_PREFIX.items():
            probe = probe_set.get(probe_type)
            if probe is None:
                continue
            path = os.path.join(probe_dir, f"{prefix}_{premise_id}.json")
            result = check_sha256(run_probed(problem, probe, api_key, model=model))
            save_llm_result(result, path)
            n += 1

    with _print_lock:
        print(f"  [done] {pid}: {n} calls", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    api_key = os.environ["ANTHROPIC_API_KEY"] if args.model.startswith("claude") \
        else os.environ["OPENAI_API_KEY"]

    problems = load_problems()
    print(f"Full Phase 1 (concurrent): {len(problems)} problems, "
          f"model={args.model}, out={args.out}, workers={args.workers}", flush=True)

    total = 0
    errors = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_problem, p, args.model, args.out, api_key): p["problem_id"]
                for p in problems}
        for fut in cf.as_completed(futs):
            pid = futs[fut]
            try:
                total += fut.result()
            except Exception as e:
                errors.append((pid, str(e)))
                with _print_lock:
                    print(f"  [ERROR] {pid}: {e}", flush=True)

    print(f"Phase 1 complete: {total} calls, {len(errors)} errors.", flush=True)
    if errors:
        for pid, err in errors:
            print(f"  FAILED {pid}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
