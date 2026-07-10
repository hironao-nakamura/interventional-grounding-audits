#!/usr/bin/env python3
"""Phase 1: run the LLM on original and probed premises, save raw outputs.

Usage:
  python scripts/run_phase1.py --model gpt-4o --out evidence/gpt-4o --overwrite
  python scripts/run_phase1.py --model claude-sonnet-4-5-20250929 --out evidence/claude-sonnet-4-5 --overwrite

Input:
  data/prontoqa_50/p*.json

Output per problem:
  evidence/<model-dir>/pXXX/problem_meta.json
  evidence/<model-dir>/pXXX/original_cot.json
  evidence/<model-dir>/pXXX/probes/semantic_Pj.json
  evidence/<model-dir>/pXXX/probes/local_Pj.json
  evidence/<model-dir>/pXXX/probes/surface_Pj.json

Every raw output stores `raw_response`, `sha256` (SHA256 of the UTF-8
encoded raw_response), `prompt`, and (for probes) `modified_premises`,
so probe integrity is verifiable from the evidence pack alone.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.llm_runner import run_original, run_probed, save_llm_result  # noqa: E402
from src.probe_generator import generate_all_probes  # noqa: E402

PROBE_FILENAME_PREFIX = {
    "semantic": "semantic",
    "local_semantic": "local",
    "surface": "surface",
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


def check_sha256(result: dict) -> dict:
    """(Re)compute sha256 of raw_response bytes and store it."""
    result["sha256"] = hashlib.sha256(result["raw_response"].encode("utf-8")).hexdigest()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="Output evidence directory")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.model.startswith("claude"):
        api_key = os.environ["ANTHROPIC_API_KEY"]
    else:
        api_key = os.environ["OPENAI_API_KEY"]

    problems = load_problems()
    print(f"Phase 1: {len(problems)} problems, model={args.model}, out={args.out}")

    for i, problem in enumerate(problems, 1):
        pid = problem["problem_id"]
        prob_dir = os.path.join(args.out, pid)
        probe_dir = os.path.join(prob_dir, "probes")
        orig_path = os.path.join(prob_dir, "original_cot.json")

        if os.path.exists(orig_path) and not args.overwrite:
            raise FileExistsError(
                f"{orig_path} exists; pass --overwrite to regenerate Phase 1 outputs.")

        os.makedirs(probe_dir, exist_ok=True)

        # Problem metadata (premises, question, proof tree) for standalone audit.
        with open(os.path.join(prob_dir, "problem_meta.json"), "w") as f:
            json.dump(problem, f, indent=2)

        print(f"  [{i}/{len(problems)}] {pid}: original", flush=True)
        result = check_sha256(run_original(problem, api_key, model=args.model))
        save_llm_result(result, orig_path)

        probes = generate_all_probes(problem)
        for premise_id, probe_set in probes.items():
            for probe_type, prefix in PROBE_FILENAME_PREFIX.items():
                probe = probe_set.get(probe_type)
                if probe is None:
                    continue
                path = os.path.join(probe_dir, f"{prefix}_{premise_id}.json")
                if os.path.exists(path) and not args.overwrite:
                    raise FileExistsError(
                        f"{path} exists; pass --overwrite to regenerate.")
                print(f"      probe {probe_type} {premise_id}", flush=True)
                result = check_sha256(run_probed(problem, probe, api_key, model=args.model))
                save_llm_result(result, path)

    print("Phase 1 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
