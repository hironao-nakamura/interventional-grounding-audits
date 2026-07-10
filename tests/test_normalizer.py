"""Tests for normalizer.py — canonical form extraction from CoT step text."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.normalizer import normalize, _depluralize, _clean_predicate, parse_cot_steps


class TestDepluralize:
    def test_uses_ending(self):
        assert _depluralize("wumpuses") == "wumpus"
        assert _depluralize("tumpuses") == "tumpus"

    def test_already_singular(self):
        assert _depluralize("wumpus") == "wumpus"
        assert _depluralize("tumpus") == "tumpus"

    def test_general_s(self):
        assert _depluralize("cats") == "cat"

    def test_general_es(self):
        # "es" not ending in "us"
        assert _depluralize("foxes") == "fox"


class TestCleanPredicate:
    def test_trailing_e(self):
        assert _clean_predicate("stelpuse") == "stelpus"
        assert _clean_predicate("trevuse") == "trevus"

    def test_no_change_needed(self):
        assert _clean_predicate("stelpus") == "stelpus"
        assert _clean_predicate("wumpus") == "wumpus"


class TestNormalize:
    """Core normalization tests."""

    def test_is_a(self):
        norm, status = normalize("Alex is a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    def test_is_not_a(self):
        norm, status = normalize("Alex is not a zumpus")
        assert status == "OK"
        assert norm == "not_is(alex, zumpus)"

    def test_plural_are(self):
        norm, status = normalize("Wumpuses are tumpuses")
        assert status == "OK"
        assert norm == "subtype(wumpus, tumpus)"

    def test_are_not(self):
        norm, status = normalize("Stompuses are not vompuses")
        assert status == "OK"
        assert norm == "not_subtype(stompus, vompus)"

    def test_every(self):
        norm, status = normalize("Every wumpus is a tumpus")
        assert status == "OK"
        assert norm == "subtype(wumpus, tumpus)"

    def test_all(self):
        norm, status = normalize("All wumpuses are tumpuses")
        assert status == "OK"
        assert norm == "subtype(wumpus, tumpus)"

    def test_no_x_is_a_y(self):
        norm, status = normalize("No stompus is a vompus")
        assert status == "OK"
        assert norm == "not_subtype(stompus, vompus)"

    def test_one_of_the(self):
        norm, status = normalize("Alex is one of the tumpuses")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    def test_kind_of(self):
        norm, status = normalize("Alex is a kind of wumpus")
        assert status == "OK"
        assert norm == "is(alex, wumpus)"

    def test_also(self):
        norm, status = normalize("Alex is also a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    # Prefix stripping
    def test_therefore_prefix(self):
        norm, status = normalize("Therefore, Alex is a dumpus")
        assert status == "OK"
        assert norm == "is(alex, dumpus)"

    def test_based_on_prefix(self):
        norm, status = normalize("Based on this, Alex is a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    def test_given_prefix(self):
        norm, status = normalize("Given the premises, Alex is a dumpus")
        assert status == "OK"
        assert norm == "is(alex, dumpus)"

    # Compound sentences
    def test_so_conclusion(self):
        norm, status = normalize("Since wumpuses are tumpuses, so Alex is a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    def test_since_conclusion(self):
        norm, status = normalize("Since Alex is a wumpus, Alex is a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    # Appositive constructions
    def test_appositive_being(self):
        norm, status = normalize("Rex, being a brimpus, is a stelpus")
        assert status == "OK"
        assert norm == "is(rex, stelpus)"

    # Relative clause
    def test_relative_clause(self):
        norm, status = normalize("Alex, who is a wumpus, is a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    # Modal verbs
    def test_must_be(self):
        norm, status = normalize("Alex must be a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    def test_would_be(self):
        norm, status = normalize("Alex would be a tumpus")
        assert status == "OK"
        assert norm == "is(alex, tumpus)"

    # Clean predicate (misspelling)
    def test_misspelled_predicate(self):
        norm, status = normalize("Rex is a stelpuse")
        assert status == "OK"
        assert norm == "is(rex, stelpus)"


class TestParseCotSteps:
    def test_basic_steps(self):
        response = """Step 1: Alex is a wumpus.
Step 2: Alex is a tumpus.
Step 3: Alex is a dumpus.
Answer: True"""
        steps = parse_cot_steps(response)
        assert len(steps) == 3
        assert steps[0]["step_id"] == 1
        assert steps[0]["normalized"] == "is(alex, wumpus)"
        assert steps[0]["parse_status"] == "OK"

    def test_complex_step(self):
        response = """Step 1: Since Alex is a wumpus and wumpuses are tumpuses, Alex is a tumpus.
Answer: True"""
        steps = parse_cot_steps(response)
        assert len(steps) == 1
        assert steps[0]["normalized"] == "is(alex, tumpus)"
