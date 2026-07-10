"""Tests for judge.py — 5-value verdict + combined verdict."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.judge import judge_verdict, judge_combined_verdict, build_certificate


class TestJudgeVerdict:
    """Test consistent-only 5-value verdicts."""

    def test_grounded(self):
        assert judge_verdict("is(a,b)", "is(a,c)", "is(a,b)", True, True, True) == "GROUNDED"

    def test_insensitive(self):
        assert judge_verdict("is(a,b)", "is(a,b)", "is(a,b)", True, True, True) == "INSENSITIVE"

    def test_input_sensitive(self):
        assert judge_verdict("is(a,b)", "is(a,c)", "is(a,d)", True, True, True) == "INPUT-SENSITIVE"

    def test_unstable(self):
        assert judge_verdict("is(a,b)", "is(a,b)", "is(a,d)", True, True, True) == "UNSTABLE"

    def test_unparseable(self):
        assert judge_verdict(None, "is(a,c)", "is(a,b)", False, True, True) == "UNPARSEABLE"
        assert judge_verdict("is(a,b)", None, "is(a,b)", True, False, True) == "UNPARSEABLE"
        assert judge_verdict("is(a,b)", "is(a,c)", None, True, True, False) == "UNPARSEABLE"


class TestCombinedVerdict:
    """Test combined (consistent OR local) verdict."""

    def test_grounded_from_consistent(self):
        result = judge_combined_verdict(
            "is(a,b)", "is(a,c)", "is(a,b)", "is(a,b)",
            True, True, True, True)
        assert result == "GROUNDED"

    def test_grounded_from_local_only(self):
        """Local delta only → still GROUNDED."""
        result = judge_combined_verdict(
            "is(a,b)", "is(a,b)", "is(a,c)", "is(a,b)",
            True, True, True, True)
        assert result == "GROUNDED"

    def test_grounded_from_both(self):
        result = judge_combined_verdict(
            "is(a,b)", "is(a,c)", "is(a,d)", "is(a,b)",
            True, True, True, True)
        assert result == "GROUNDED"

    def test_insensitive(self):
        result = judge_combined_verdict(
            "is(a,b)", "is(a,b)", "is(a,b)", "is(a,b)",
            True, True, True, True)
        assert result == "INSENSITIVE"

    def test_unparseable_local_but_consistent_ok(self):
        """If local is unparseable but consistent works → still functional."""
        result = judge_combined_verdict(
            "is(a,b)", "is(a,c)", None, "is(a,b)",
            True, True, False, True)
        assert result == "GROUNDED"


class TestBuildCertificate:
    def test_certificate_has_both_verdicts(self):
        cert = build_certificate(
            problem_id="p001", step_id=1, premise_id="P1",
            phi_orig="is(a,b)", phi_sem="is(a,c)", phi_sur="is(a,b)",
            parse_ok_orig=True, parse_ok_sem=True, parse_ok_sur=True,
            alignment_sem="MATCHED", alignment_sur="MATCHED",
            phi_local="is(a,d)",
            parse_ok_local=True, alignment_local="MATCHED",
        )
        assert cert["verdict"] == "GROUNDED"
        assert cert["verdict_consistent"] == "GROUNDED"
        assert cert["phi_local"] == "is(a,d)"
        assert cert["local_delta"] == True

    def test_certificate_local_grounded_consistent_insensitive(self):
        """Local detects cascade, consistent doesn't."""
        cert = build_certificate(
            problem_id="p001", step_id=1, premise_id="P4",
            phi_orig="is(a,b)", phi_sem="is(a,b)", phi_sur="is(a,b)",
            parse_ok_orig=True, parse_ok_sem=True, parse_ok_sur=True,
            alignment_sem="MATCHED", alignment_sur="MATCHED",
            phi_local="is(a,c)",
            parse_ok_local=True, alignment_local="MATCHED",
        )
        assert cert["verdict"] == "GROUNDED"  # Combined: local detected
        assert cert["verdict_consistent"] == "INSENSITIVE"  # Consistent only: no change
