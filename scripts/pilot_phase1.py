#!/usr/bin/env python3
"""Pilot Phase 1 run on a small subset of problems, isolated from the main
evidence/ dirs, to smoke-test API keys + pipeline before a full 50-problem run.

Usage:
  OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python scripts/pilot_phase1.py --n 3
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


def load_problems(n: int) -> list[dict]:
    problems = []
    for f in sorted(glob.glob(os.path.join(BASE, "data", "prontoqa_50", "p*.json")))[:n]:
        if "proof_trees" in f:
            continue
        problem = json.load(open(f))
        problem.setdefault("problem_id", os.path.splitext(os.path.basename(f))[0])
        problems.append(problem)
    return problems


def check_sha256(result: dict) -> dict:
    result["sha256"] = hashlib.sha256(result["raw_response"].encode("utf-8")).hexdigest()
    return result


def run_one_model(model: str, out: str, problems: list[dict], api_key: str) -> None:
    print(f"\n=== Pilot Phase 1: model={model}, out={out}, n_problems={len(problems)} ===")
    for i, problem in enumerate(problems, 1):
        pid = problem["problem_id"]
        prob_dir = os.path.join(out, pid)
        probe_dir = os.path.join(prob_dir, "probes")
        os.makedirs(probe_dir, exist_ok=True)

        with open(os.path.join(prob_dir, "problem_meta.json"), "w") as f:
            json.dump(problem, f, indent=2)

        print(f"  [{i}/{len(problems)}] {pid}: original", flush=True)
        result = check_sha256(run_original(problem, api_key, model=model))
        save_llm_result(result, os.path.join(prob_dir, "original_cot.json"))
        print(f"      -> answer snippet: {result['raw_response'][-80:]!r}")

        probes = generate_all_probes(problem)
        n_probes = 0
        for premise_id, probe_set in probes.items():
            for probe_type, prefix in PROBE_FILENAME_PREFIX.items():
                probe = probe_set.get(probe_type)
                if probe is None:
                    continue
                path = os.path.join(probe_dir, f"{prefix}_{premise_id}.json")
                result = check_sha256(run_probed(problem, probe, api_key, model=model))
                save_llm_result(result, path)
                n_probes += 1
        print(f"      {n_probes} probe calls done")
    print(f"=== Pilot Phase 1 complete for {model}: {out} ===")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["gpt-4o-2024-08-06", "claude-sonnet-4-5-20250929"])
    args = ap.parse_args()

    problems = load_problems(args.n)
    print(f"Loaded {len(problems)} pilot problems: {[p['problem_id'] for p in problems]}")

    for model in args.models:
        if model.startswith("claude"):
            api_key = os.environ["ANTHROPIC_API_KEY"]
            out = os.path.join(BASE, "evidence", "claude-sonnet-4-5_pilot")
        else:
            api_key = os.environ["OPENAI_API_KEY"]
            out = os.path.join(BASE, "evidence", "gpt-4o_pilot")
        run_one_model(model, out, problems, api_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
