"""Tests for auditor.py — observe→assess→decide→certificate pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.auditor import ReasoningAuditor, R_GROUNDED_PRED_CHANGE, R_GROUNDED_CASCADE, R_INSENSITIVE_NO_CHANGE


class TestReasoningAuditor:

    def test_grounded_via_consistent(self):
        auditor = ReasoningAuditor()
        step = {"step_id": 1, "normalized": "is(a,b)", "parse_status": "OK", "problem_id": "p001"}
        cert = auditor.audit_step(
            step=step, premise_id="P1",
            phi_sem="is(a,c)", phi_local="is(a,b)", phi_sur="is(a,b)",
            parse_ok_sem=True, parse_ok_local=True, parse_ok_sur=True,
            aligned_sem=True, aligned_local=True, aligned_sur=True,
        )
        assert cert["verdict"] == "GROUNDED"
        assert cert["decision_rule"] == R_GROUNDED_PRED_CHANGE
        assert cert["confidence"] == 1.0

    def test_grounded_via_local_cascade(self):
        auditor = ReasoningAuditor()
        step = {"step_id": 1, "normalized": "is(a,b)", "parse_status": "OK", "problem_id": "p001"}
        cert = auditor.audit_step(
            step=step, premise_id="P4",
            phi_sem="is(a,b)", phi_local="is(a,c)", phi_sur="is(a,b)",
            parse_ok_sem=True, parse_ok_local=True, parse_ok_sur=True,
            aligned_sem=True, aligned_local=True, aligned_sur=True,
        )
        assert cert["verdict"] == "GROUNDED"
        assert cert["decision_rule"] == R_GROUNDED_CASCADE

    def test_insensitive(self):
        auditor = ReasoningAuditor()
        step = {"step_id": 1, "normalized": "is(a,b)", "parse_status": "OK", "problem_id": "p001"}
        cert = auditor.audit_step(
            step=step, premise_id="P5",
            phi_sem="is(a,b)", phi_local="is(a,b)", phi_sur="is(a,b)",
            parse_ok_sem=True, parse_ok_local=True, parse_ok_sur=True,
            aligned_sem=True, aligned_local=True, aligned_sur=True,
        )
        assert cert["verdict"] == "INSENSITIVE"
        assert cert["decision_rule"] == R_INSENSITIVE_NO_CHANGE

    def test_cumulative_stats(self):
        auditor = ReasoningAuditor()
        step = {"step_id": 1, "normalized": "is(a,b)", "parse_status": "OK", "problem_id": "p001"}
        auditor.audit_step(
            step=step, premise_id="P1",
            phi_sem="is(a,c)", phi_local="is(a,b)", phi_sur="is(a,b)",
            parse_ok_sem=True, parse_ok_local=True, parse_ok_sur=True,
            aligned_sem=True, aligned_local=True, aligned_sur=True,
        )
        auditor.audit_step(
            step=step, premise_id="P2",
            phi_sem="is(a,b)", phi_local="is(a,b)", phi_sur="is(a,b)",
            parse_ok_sem=True, parse_ok_local=True, parse_ok_sur=True,
            aligned_sem=True, aligned_local=True, aligned_sur=True,
        )
        stats = auditor.get_cumulative_stats()
        assert stats["total_audited"] == 2
        assert R_GROUNDED_PRED_CHANGE in stats["rule_counts"]
        assert R_INSENSITIVE_NO_CHANGE in stats["rule_counts"]

    def test_certificate_has_flags(self):
        auditor = ReasoningAuditor()
        step = {"step_id": 1, "normalized": "is(a,b)", "parse_status": "OK", "problem_id": "p001"}
        cert = auditor.audit_step(
            step=step, premise_id="P1",
            phi_sem="is(a,c)", phi_local="is(a,b)", phi_sur="is(a,b)",
            parse_ok_sem=True, parse_ok_local=True, parse_ok_sur=True,
            aligned_sem=True, aligned_local=True, aligned_sur=True,
        )
        assert "flags" in cert
        assert "semantic_change" in cert["flags"]
        assert "parse_ok:orig" in cert["flags"]
