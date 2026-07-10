"""Regression tests for GT<->CoT step alignment.

These tests read the FROZEN accepted-version original CoTs from
tests/fixtures/, NOT from evidence/. Phase 1 reruns overwrite evidence/
non-deterministically, so regression tests must not depend on it.

The fixtures reproduce the off-by-one that existed in the accepted
workshop artifact: the model prepends a premise-restatement step, so
CoT step N corresponds to proof-tree step N-1.
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.normalizer import parse_cot_steps  # noqa: E402


def _load_fixture(pid: str):
    problem = json.load(open(os.path.join(BASE, "data", "prontoqa_50", f"{pid}.json")))
    raw = json.load(open(os.path.join(
        BASE, "tests", "fixtures", f"{pid}_accepted_original_cot.json")))["raw_response"]
    return problem, parse_cot_steps(raw)


def test_p026_restatement_step_is_unmatched_and_step2_maps_to_proof1():
    from src.step_alignment import align_cot_to_proof
    problem, cot_steps = _load_fixture("p026")
    a = align_cot_to_proof(problem, cot_steps)
    assert 1 in a.unmatched_cot_steps
    assert a.cot_to_proof[2] == 1
    assert a.cot_to_proof[3] == 2
    assert a.cot_to_proof[4] == 3
    assert a.cot_to_proof[5] == 4


def test_p026_gt_for_cot_step2_uses_p6_and_p1_not_p2():
    from src.ground_truth import extract_ground_truth_dependencies_for_cot_steps
    problem, cot_steps = _load_fixture("p026")
    gt, alignment = extract_ground_truth_dependencies_for_cot_steps(problem, cot_steps)
    assert gt[(2, "P6")] is True
    assert gt[(2, "P1")] is True
    assert gt[(2, "P2")] is False
    assert (1, "P6") not in gt


def test_accepted_restatement_fixtures_have_expected_alignment_shape():
    from src.step_alignment import align_cot_to_proof
    for pid in ["p009", "p018", "p026", "p027"]:
        problem, cot_steps = _load_fixture(pid)
        a = align_cot_to_proof(problem, cot_steps)
        assert a.unmatched_cot_steps, pid
        assert len(a.cot_to_proof) >= 1, pid


def test_unmatched_cot_steps_excluded_but_problem_not_primary_excluded():
    from src.step_alignment import align_cot_to_proof
    problem, cot_steps = _load_fixture("p009")
    a = align_cot_to_proof(problem, cot_steps)
    # A restatement step alone is coverage exclusion, not problem exclusion.
    assert a.primary_metric_excluded_problem is False
    assert a.primary_metric_exclusion_reason is None
    assert a.unmatched_proof_steps == []


def test_ambiguous_repeated_conclusion_excludes_problem():
    from src.step_alignment import align_cot_to_proof
    problem = {
        "problem_id": "psynthetic",
        "premises": [{"id": "P1", "text": "Aas are bbs"}],
        "proof_tree": [
            {"step": 1, "conclusion": "is(x, b)", "depends_on": ["P1"]},
            {"step": 2, "conclusion": "is(x, c)", "depends_on": ["S1"]},
        ],
    }
    cot_steps = [
        {"step_id": 1, "normalized": "is(x, b)", "parse_status": "OK"},
        {"step_id": 2, "normalized": "is(x, b)", "parse_status": "OK"},
        {"step_id": 3, "normalized": "is(x, c)", "parse_status": "OK"},
    ]
    a = align_cot_to_proof(problem, cot_steps)
    assert a.primary_metric_excluded_problem is True
    assert a.primary_metric_exclusion_reason == "problem_has_ambiguous_alignment"
    assert 1 in a.ambiguous_cot_steps and 2 in a.ambiguous_cot_steps
    # The unambiguous mapping is still reported for coverage analysis.
    assert a.cot_to_proof.get(3) == 2
