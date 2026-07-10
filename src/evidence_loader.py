"""Shared loaders for problems, certificates, and parsed original CoT steps.

Every evaluation script loads evidence through these helpers so the data
path and parsing behavior cannot drift between scripts.
"""

import glob
import json
import os
import re

from src.normalizer import parse_cot_steps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_problems(base: str = BASE) -> list[dict]:
    problems = []
    for f in sorted(glob.glob(os.path.join(base, "data", "prontoqa_50", "p*.json"))):
        if "proof_trees" in f:
            continue
        problem = json.load(open(f))
        problem.setdefault("problem_id", os.path.splitext(os.path.basename(f))[0])
        problems.append(problem)
    return problems


def load_certificates(model: str = "gpt-4o", base: str = BASE) -> list[dict]:
    certs = []
    evidence_dir = os.path.join(base, "evidence", model)
    for pid_dir in sorted(glob.glob(os.path.join(evidence_dir, "p*"))):
        for cf in sorted(glob.glob(os.path.join(pid_dir, "certificates", "cert_*.json"))):
            certs.append(json.load(open(cf)))
    return certs


def load_original_raw(model: str, problem_id: str, base: str = BASE) -> dict:
    path = os.path.join(base, "evidence", model, problem_id, "original_cot.json")
    return json.load(open(path))


def load_cot_steps_by_pid(model: str = "gpt-4o", base: str = BASE) -> dict[str, list[dict]]:
    """problem_id -> parsed original CoT steps for that model."""
    steps = {}
    for problem in load_problems(base):
        pid = problem["problem_id"]
        raw = load_original_raw(model, pid, base)["raw_response"]
        steps[pid] = parse_cot_steps(raw)
    return steps


def parse_final_answer(raw_response: str) -> bool | None:
    """Extract the model's final True/False answer from a raw response."""
    m = re.search(r"Answer:\s*(True|False)", raw_response, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower() == "true"


def load_baseline_samples(model: str = "gpt-4o", base: str = BASE) -> dict[str, dict]:
    bl_dir = os.path.join(base, "baselines", "self_consistency", model)
    results = {}
    for f in sorted(glob.glob(os.path.join(bl_dir, "p*_samples.json"))):
        d = json.load(open(f))
        results[d["problem_id"]] = d
    return results
