"""Align model CoT steps to proof-tree steps by normalized conclusion.

Certificates key their `step_id` by the model's CoT step number, which is
NOT the proof-tree step number: models sometimes prepend a premise
restatement step (e.g., "Step 1: Sam is a blonkus."), shifting every
subsequent step by one. Matching by raw integer step number therefore
compares each CoT step against the wrong proof-tree ground truth.

This module aligns CoT steps to proof-tree steps by exact normalized
conclusion instead of by integer step number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepAlignment:
    problem_id: str
    cot_to_proof: dict[int, int]
    proof_to_cot: dict[int, int]
    unmatched_cot_steps: list[int]
    unmatched_proof_steps: list[int]
    ambiguous_cot_steps: list[int]
    ambiguous_proof_steps: list[int]
    primary_metric_excluded_problem: bool
    primary_metric_exclusion_reason: str | None


def align_cot_to_proof(problem: dict[str, Any], cot_steps: list[dict[str, Any]]) -> StepAlignment:
    """Map CoT step IDs to proof-tree step IDs by exact normalized conclusion.

    Rules:
    - Match only steps whose `normalized` equals a proof-tree `conclusion`.
    - If a CoT step has no proof conclusion match, mark it unmatched.
    - If a proof conclusion appears multiple times or multiple CoT steps match the same proof step,
      mark ambiguity and do not silently choose.
    - Primary metrics exclude unmatched CoT steps.
    - If any ambiguity remains for a problem, exclude that entire problem from primary metrics,
      record it in the alignment summary, and disclose the count in Appendix A / Section 3.1.
    """
    problem_id = problem.get("problem_id", "")
    proof_tree = problem.get("proof_tree", [])

    # 1. conclusion -> [proof_step_id]
    conclusion_to_proof: dict[str, list[int]] = {}
    for entry in proof_tree:
        conclusion_to_proof.setdefault(entry["conclusion"], []).append(entry["step"])

    cot_to_proof: dict[int, int] = {}
    proof_to_cot: dict[int, int] = {}
    unmatched_cot: list[int] = []
    ambiguous_cot: list[int] = []
    ambiguous_proof: set[int] = set()

    # proof steps whose conclusion is duplicated inside the proof tree
    duplicated_proof_conclusions = {
        c: steps for c, steps in conclusion_to_proof.items() if len(steps) > 1
    }

    # Track which proof step each CoT step wants, to detect many-to-one collisions.
    proof_claimed_by: dict[int, list[int]] = {}

    for step in cot_steps:
        cid = step["step_id"]
        norm = step.get("normalized")
        # 3. normalized is None -> unmatched
        if norm is None:
            unmatched_cot.append(cid)
            continue
        # 4. conclusion absent from proof tree -> unmatched
        proof_ids = conclusion_to_proof.get(norm)
        if not proof_ids:
            unmatched_cot.append(cid)
            continue
        # 5. conclusion matches multiple proof steps -> ambiguous
        if len(proof_ids) > 1:
            ambiguous_cot.append(cid)
            ambiguous_proof.update(proof_ids)
            continue
        pid = proof_ids[0]
        proof_claimed_by.setdefault(pid, []).append(cid)

    # 6. multiple CoT steps mapping to the same proof step -> ambiguous
    for pid, cids in proof_claimed_by.items():
        if len(cids) > 1:
            ambiguous_cot.extend(cids)
            ambiguous_proof.add(pid)
        else:
            cot_to_proof[cids[0]] = pid
            proof_to_cot[pid] = cids[0]

    matched_proof = set(proof_to_cot)
    all_proof = {e["step"] for e in proof_tree}
    unmatched_proof = sorted(all_proof - matched_proof - ambiguous_proof)

    ambiguous_cot = sorted(set(ambiguous_cot))
    ambiguous_proof_list = sorted(ambiguous_proof)

    # 7. Any ambiguity -> exclude the whole problem from primary metrics.
    # Repeated identical normalized conclusions come from the model's own
    # output and are not something an operator can safely "fix" by hand.
    has_ambiguity = bool(ambiguous_cot or ambiguous_proof_list)
    return StepAlignment(
        problem_id=problem_id,
        cot_to_proof=cot_to_proof,
        proof_to_cot=proof_to_cot,
        unmatched_cot_steps=sorted(unmatched_cot),
        unmatched_proof_steps=unmatched_proof,
        ambiguous_cot_steps=ambiguous_cot,
        ambiguous_proof_steps=ambiguous_proof_list,
        primary_metric_excluded_problem=has_ambiguity,
        primary_metric_exclusion_reason=(
            "problem_has_ambiguous_alignment" if has_ambiguity else None
        ),
    )
