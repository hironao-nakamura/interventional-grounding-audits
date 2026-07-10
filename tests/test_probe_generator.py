"""Tests for probe_generator.py — consistent vs local substitution."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.probe_generator import (
    generate_semantic_probe,
    generate_local_semantic_probe,
    generate_surface_probe,
    generate_all_probes,
)

# Test problem fixture
TEST_PROBLEM = {
    "problem_id": "t001",
    "premises": [
        {"id": "P1", "text": "Wumpuses are tumpuses"},
        {"id": "P2", "text": "Tumpuses are dumpuses"},
        {"id": "P3", "text": "Dumpuses are zumpuses"},
        {"id": "P4", "text": "Alex is a wumpus"},
    ],
    "question": "Is Alex a zumpus?",
    "answer": True,
    "proof_tree": [
        {"step": 1, "depends_on": ["P1", "P4"]},
        {"step": 2, "depends_on": ["P2", "S1"]},
        {"step": 3, "depends_on": ["P3", "S2"]},
    ],
}


class TestSemanticProbe:
    """Consistent substitution: replaces predicate everywhere."""

    def test_consistent_substitution_p1(self):
        probe = generate_semantic_probe(TEST_PROBLEM, "P1")
        assert probe is not None
        assert probe["probe_type"] == "semantic"
        # P1 targets "tumpus" → substitute replaces in P1 and P2
        sub = list(probe["substitution"].values())[0]
        # P1 should have substitute
        p1_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P1")
        assert sub in p1_text.lower() or sub.capitalize() in p1_text
        # P2 should ALSO have substitute (consistent)
        p2_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P2")
        assert sub in p2_text.lower() or sub.capitalize() in p2_text
        # P3, P4 should be unchanged (different predicates)
        p4_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P4")
        assert p4_text == "Alex is a wumpus"

    def test_consistent_substitution_p4(self):
        probe = generate_semantic_probe(TEST_PROBLEM, "P4")
        assert probe is not None
        # P4 targets "wumpus" → substitute replaces in P4 and P1
        sub = list(probe["substitution"].values())[0]
        p4_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P4")
        assert sub in p4_text.lower() or sub.capitalize() in p4_text
        p1_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P1")
        assert sub in p1_text.lower() or sub.capitalize() in p1_text


class TestLocalSemanticProbe:
    """Local substitution: replaces predicate ONLY in target premise."""

    def test_local_substitution_p1(self):
        probe = generate_local_semantic_probe(TEST_PROBLEM, "P1")
        assert probe is not None
        assert probe["probe_type"] == "local_semantic"
        sub = list(probe["substitution"].values())[0]
        # P1 should have substitute
        p1_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P1")
        assert sub in p1_text.lower() or sub.capitalize() in p1_text
        # P2 should NOT have substitute (local = target only)
        p2_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P2")
        assert p2_text == "Tumpuses are dumpuses"
        # P4 should be unchanged
        p4_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P4")
        assert p4_text == "Alex is a wumpus"

    def test_local_substitution_p4(self):
        probe = generate_local_semantic_probe(TEST_PROBLEM, "P4")
        assert probe is not None
        sub = list(probe["substitution"].values())[0]
        # P4 should have substitute
        p4_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P4")
        assert sub in p4_text.lower() or sub.capitalize() in p4_text
        # P1 should NOT have substitute (local)
        p1_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P1")
        assert p1_text == "Wumpuses are tumpuses"

    def test_consistent_vs_local_difference(self):
        """Verify that consistent and local produce different probe sets."""
        sem = generate_semantic_probe(TEST_PROBLEM, "P1")
        loc = generate_local_semantic_probe(TEST_PROBLEM, "P1")
        # P2 should differ between consistent and local
        sem_p2 = next(p["text"] for p in sem["modified_premises"] if p["id"] == "P2")
        loc_p2 = next(p["text"] for p in loc["modified_premises"] if p["id"] == "P2")
        assert sem_p2 != loc_p2  # Consistent changes P2, local doesn't


class TestSurfaceProbe:
    def test_surface_rephrase(self):
        probe = generate_surface_probe(TEST_PROBLEM, "P1")
        assert probe is not None
        assert probe["probe_type"] == "surface"
        p1_text = next(p["text"] for p in probe["modified_premises"] if p["id"] == "P1")
        # Should be rephrased but meaning-preserving
        assert p1_text != "Wumpuses are tumpuses"
        assert "tumpus" in p1_text.lower()


class TestGenerateAllProbes:
    def test_all_probes_generated(self):
        probes = generate_all_probes(TEST_PROBLEM)
        assert len(probes) == 4  # 4 premises
        for pid, pair in probes.items():
            assert "semantic" in pair
            assert "local_semantic" in pair
            assert "surface" in pair


def test_dynamic_substitution_does_not_rewrite_inside_substitute():
    from src.probe_generator import generate_semantic_probe
    problem = {
        "problem_id": "ptest",
        "premises": [
            {"id": "P1", "text": "Bempuses are bimpuses"},
            {"id": "P2", "text": "Alex is a bempus"},
        ],
        "question": "Is Alex a bimpus?",
    }
    probe = generate_semantic_probe(problem, "P1")
    joined = "\n".join(p["text"] for p in probe["modified_premises"])
    assert "zqzq" not in joined.lower()
    assert "Bempuses are zqbimpuses" in joined


def test_consistent_substitution_preserves_same_symbol_for_entity_premise():
    from src.probe_generator import generate_semantic_probe
    problem = {
        "problem_id": "ptest",
        "premises": [
            {"id": "P1", "text": "Bralpuses are brelpuses"},
            {"id": "P2", "text": "Kai is a bralpus"},
        ],
        "question": "Is Kai a brelpus?",
    }
    probe = generate_semantic_probe(problem, "P2")
    texts = {p["id"]: p["text"] for p in probe["modified_premises"]}
    assert texts["P1"] == "Zqbralpuses are brelpuses"
    assert texts["P2"] == "Kai is a zqbralpus"
    assert "zqzq" not in "\n".join(texts.values()).lower()
