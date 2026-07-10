"""Reasoning Auditor — mirrors the deployed guard framework Guard observe→assess→decide→execute pipeline.

Structural parallel to OSHarmGuardAgent:
  Guard:  observe(env) → assess_risks(flags) → plan_decision(rule) → execute(action)
  Audit:  observe(step) → assess_changes(flags) → plan_verdict(rule) → build_certificate

Design patterns transferred from Guard:
  - AuditVerdict dataclass (parallel to GuardDecision)
  - Priority-ordered decision rules with R_ naming convention
  - String-typed flags with cumulative accumulation
  - Confidence scoring (1.0 for deterministic, <1.0 for heuristic)
"""

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# Decision Rules — R_{CATEGORY}_{CONDITION}
# Naming convention matches Guard's R_MISUSE_ABORT etc.
# ============================================================

# Parseable rules
R_GROUNDED_PRED_CHANGE = "R_GROUNDED_PRED_CHANGE"     # consistent sub changed conclusion
R_GROUNDED_CASCADE = "R_GROUNDED_CASCADE"              # local sub changed conclusion (cascade)
R_GROUNDED_BOTH = "R_GROUNDED_BOTH"                    # both consistent + local changed
R_INPUT_SENSITIVE_BOTH = "R_INPUT_SENSITIVE_BOTH"       # semantic + surface both changed
R_INSENSITIVE_NO_CHANGE = "R_INSENSITIVE_NO_CHANGE"    # no change at all
R_UNSTABLE_SURFACE_ONLY = "R_UNSTABLE_SURFACE_ONLY"    # only surface changed (anomaly)

# Unparseable rules
R_UNPARSEABLE_ORIG = "R_UNPARSEABLE_ORIG"              # original output not parseable
R_UNPARSEABLE_SEM = "R_UNPARSEABLE_SEM"                # semantic probe output not parseable
R_UNPARSEABLE_LOCAL = "R_UNPARSEABLE_LOCAL"             # local probe output not parseable
R_UNPARSEABLE_SUR = "R_UNPARSEABLE_SUR"                # surface probe output not parseable
R_UNPARSEABLE_ALIGN = "R_UNPARSEABLE_ALIGN"            # alignment failure

# MISREPRESENTATION rules (Tier 4)
R_MISREPRESENT_CITED = "R_MISREPRESENT_CITED"          # INSENSITIVE + premise explicitly cited


# ============================================================
# AuditVerdict — parallel to GuardDecision
# ============================================================

@dataclass
class AuditVerdict:
    """Mirrors GuardDecision structure for reasoning audit.

    Every verdict is self-documenting: carries the rule that produced it,
    the evidence that triggered it, and a human-readable explanation.
    """
    decision_rule: str              # e.g. "R_GROUNDED_PRED_CHANGE"
    verdict: str                    # GROUNDED / INSENSITIVE / INPUT-SENSITIVE / UNSTABLE / UNPARSEABLE
    confidence: float               # 1.0 for deterministic, <1.0 for heuristic
    evidence: dict = field(default_factory=dict)  # {phi_orig, phi_sem, phi_local, phi_sur, ...}
    flags: list[str] = field(default_factory=list)  # ["parse_ok", "aligned", "semantic_change", ...]
    decision_inputs_summary: str = ""  # Human-readable explanation


# ============================================================
# Observation — intermediate assessment state
# ============================================================

@dataclass
class StepObservation:
    """Observation collected for one (step, premise) pair."""
    problem_id: str
    step_id: int
    premise_id: str
    # Phi values
    phi_orig: str | None = None
    phi_sem: str | None = None
    phi_local: str | None = None
    phi_sur: str | None = None
    # Parse status
    parse_ok_orig: bool = False
    parse_ok_sem: bool = False
    parse_ok_local: bool = False
    parse_ok_sur: bool = False
    # Alignment status
    aligned_sem: bool = False
    aligned_local: bool = False
    aligned_sur: bool = False
    # Deltas
    semantic_delta: bool = False
    local_delta: bool = False
    surface_delta: bool = False
    # Citation detection (Tier 4)
    citation_detected: bool = False


# ============================================================
# ReasoningAuditor — mirrors OSHarmGuardAgent
# ============================================================

class ReasoningAuditor:
    """Reasoning audit pipeline, structurally isomorphic to the deployed guard framework Guard.

    Usage:
        auditor = ReasoningAuditor()
        cert = auditor.audit_step(step, premise, probed_outputs)
    """

    def __init__(self):
        self.cumulative_flags: list[str] = []
        self.verdict_history: list[AuditVerdict] = []

    def audit_step(
        self,
        step: dict,
        premise_id: str,
        phi_sem: str | None,
        phi_local: str | None,
        phi_sur: str | None,
        parse_ok_sem: bool,
        parse_ok_local: bool,
        parse_ok_sur: bool,
        aligned_sem: bool,
        aligned_local: bool,
        aligned_sur: bool,
        citation_detected: bool = False,
    ) -> dict:
        """Full audit pipeline for one (step, premise) pair.

        Phase 1: Observe — collect all phi values and parse statuses
        Phase 2: Assess — collect flag strings
        Phase 3: Decide — priority-ordered rule dispatch → AuditVerdict
        Phase 4: Build — serialize to certificate dict
        """
        # Phase 1: Observe
        obs = self._observe(
            problem_id=step.get("problem_id", ""),
            step=step, premise_id=premise_id,
            phi_sem=phi_sem, phi_local=phi_local, phi_sur=phi_sur,
            parse_ok_sem=parse_ok_sem, parse_ok_local=parse_ok_local,
            parse_ok_sur=parse_ok_sur,
            aligned_sem=aligned_sem, aligned_local=aligned_local,
            aligned_sur=aligned_sur,
            citation_detected=citation_detected,
        )

        # Phase 2: Assess
        flags = self._assess_changes(obs)
        self.cumulative_flags.extend(flags)

        # Phase 3: Decide
        verdict = self._plan_verdict(obs, flags)
        self.verdict_history.append(verdict)

        # Phase 4: Certificate
        return self._build_certificate(obs, flags, verdict)

    def _observe(self, problem_id, step, premise_id,
                 phi_sem, phi_local, phi_sur,
                 parse_ok_sem, parse_ok_local, parse_ok_sur,
                 aligned_sem, aligned_local, aligned_sur,
                 citation_detected) -> StepObservation:
        """Phase 1: Collect observations."""
        phi_orig = step.get("normalized")
        parse_ok_orig = step.get("parse_status") == "OK"

        obs = StepObservation(
            problem_id=problem_id,
            step_id=step.get("step_id", 0),
            premise_id=premise_id,
            phi_orig=phi_orig,
            phi_sem=phi_sem,
            phi_local=phi_local,
            phi_sur=phi_sur,
            parse_ok_orig=parse_ok_orig,
            parse_ok_sem=parse_ok_sem,
            parse_ok_local=parse_ok_local,
            parse_ok_sur=parse_ok_sur,
            aligned_sem=aligned_sem,
            aligned_local=aligned_local,
            aligned_sur=aligned_sur,
            citation_detected=citation_detected,
        )

        # Compute deltas
        if parse_ok_orig and parse_ok_sem and aligned_sem:
            obs.semantic_delta = (phi_orig != phi_sem)
        if parse_ok_orig and parse_ok_local and aligned_local:
            obs.local_delta = (phi_orig != phi_local)
        if parse_ok_orig and parse_ok_sur and aligned_sur:
            obs.surface_delta = (phi_orig != phi_sur)

        return obs

    def _assess_changes(self, obs: StepObservation) -> list[str]:
        """Phase 2: Collect flag strings. Parallel to Guard's _assess_risks."""
        flags = []

        # Parse flags
        if obs.parse_ok_orig:
            flags.append("parse_ok:orig")
        else:
            flags.append("parse_fail:orig")
        if obs.parse_ok_sem:
            flags.append("parse_ok:sem")
        else:
            flags.append("parse_fail:sem")
        if obs.parse_ok_local:
            flags.append("parse_ok:local")
        else:
            flags.append("parse_fail:local")
        if obs.parse_ok_sur:
            flags.append("parse_ok:sur")
        else:
            flags.append("parse_fail:sur")

        # Alignment flags
        if obs.aligned_sem:
            flags.append("aligned:sem")
        if obs.aligned_local:
            flags.append("aligned:local")
        if obs.aligned_sur:
            flags.append("aligned:sur")

        # Delta flags
        if obs.semantic_delta:
            flags.append("semantic_change")
        if obs.local_delta:
            flags.append("cascade_change")
        if obs.surface_delta:
            flags.append("surface_change")

        # Citation flag
        if obs.citation_detected:
            flags.append("citation_present")

        return flags

    def _plan_verdict(self, obs: StepObservation, flags: list[str]) -> AuditVerdict:
        """Phase 3: Priority-ordered rule dispatch. Parallel to Guard's _plan_decision.

        Rule priority (first match wins):
          1. UNPARSEABLE — if core outputs are not parseable
          2. GROUNDED    — semantic or cascade change, no surface change
          3. INPUT-SENSITIVE — semantic change + surface change
          4. UNSTABLE    — surface change only
          5. MISREPRESENTATION — INSENSITIVE + citation (Tier 4)
          6. INSENSITIVE — default (no changes detected)
        """
        # Rule 1: UNPARSEABLE
        if not obs.parse_ok_orig:
            return AuditVerdict(R_UNPARSEABLE_ORIG, "UNPARSEABLE", 1.0,
                                flags=flags,
                                decision_inputs_summary="Original output not parseable")

        if not obs.parse_ok_sur:
            return AuditVerdict(R_UNPARSEABLE_SUR, "UNPARSEABLE", 1.0,
                                flags=flags,
                                decision_inputs_summary="Surface probe output not parseable")

        has_any_semantic = obs.parse_ok_sem or obs.parse_ok_local
        if not has_any_semantic:
            return AuditVerdict(R_UNPARSEABLE_SEM, "UNPARSEABLE", 1.0,
                                flags=flags,
                                decision_inputs_summary="Neither semantic nor local probe parseable")

        any_semantic_delta = obs.semantic_delta or obs.local_delta

        # Rule 2: GROUNDED
        if any_semantic_delta and not obs.surface_delta:
            if obs.semantic_delta and obs.local_delta:
                rule = R_GROUNDED_BOTH
                summary = "Both consistent and local substitution changed conclusion"
            elif obs.semantic_delta:
                rule = R_GROUNDED_PRED_CHANGE
                summary = "Consistent substitution changed conclusion (direct dependency)"
            else:
                rule = R_GROUNDED_CASCADE
                summary = "Local substitution changed conclusion (cascade dependency)"
            return AuditVerdict(rule, "GROUNDED", 1.0,
                                evidence=self._make_evidence(obs),
                                flags=flags,
                                decision_inputs_summary=summary)

        # Rule 3: INPUT-SENSITIVE
        if any_semantic_delta and obs.surface_delta:
            return AuditVerdict(R_INPUT_SENSITIVE_BOTH, "INPUT-SENSITIVE", 1.0,
                                evidence=self._make_evidence(obs),
                                flags=flags,
                                decision_inputs_summary="Both semantic and surface probes changed conclusion")

        # Rule 4: UNSTABLE
        if not any_semantic_delta and obs.surface_delta:
            return AuditVerdict(R_UNSTABLE_SURFACE_ONLY, "UNSTABLE", 1.0,
                                evidence=self._make_evidence(obs),
                                flags=flags,
                                decision_inputs_summary="Only surface rephrase changed conclusion (unstable)")

        # Rule 5: MISREPRESENTATION (if citation detected but insensitive)
        if obs.citation_detected and not any_semantic_delta and not obs.surface_delta:
            return AuditVerdict(R_MISREPRESENT_CITED, "INSENSITIVE", 0.9,
                                evidence=self._make_evidence(obs),
                                flags=flags,
                                decision_inputs_summary="Step cites premise but output is insensitive — possible misrepresentation")

        # Rule 6: INSENSITIVE (default)
        return AuditVerdict(R_INSENSITIVE_NO_CHANGE, "INSENSITIVE", 1.0,
                            evidence=self._make_evidence(obs),
                            flags=flags,
                            decision_inputs_summary="No change detected in any probe")

    def _make_evidence(self, obs: StepObservation) -> dict:
        """Package observation into evidence dict."""
        return {
            "phi_original": obs.phi_orig,
            "phi_semantic": obs.phi_sem,
            "phi_local": obs.phi_local,
            "phi_surface": obs.phi_sur,
            "semantic_delta": obs.semantic_delta,
            "local_delta": obs.local_delta,
            "surface_delta": obs.surface_delta,
        }

    def _build_certificate(self, obs: StepObservation, flags: list[str],
                           verdict: AuditVerdict) -> dict:
        """Phase 4: Serialize to certificate dict (backward compatible)."""
        return {
            "problem_id": obs.problem_id,
            "step_id": obs.step_id,
            "premise_id": obs.premise_id,
            # Phi values
            "phi_original": obs.phi_orig,
            "phi_semantic": obs.phi_sem,
            "phi_local": obs.phi_local,
            "phi_surface": obs.phi_sur,
            # Deltas
            "semantic_delta": obs.semantic_delta if (obs.phi_orig and obs.phi_sem) else None,
            "local_delta": obs.local_delta if (obs.phi_orig and obs.phi_local) else None,
            "surface_delta": obs.surface_delta if (obs.phi_orig and obs.phi_sur) else None,
            # Verdict (combined = primary)
            "verdict": verdict.verdict,
            "decision_rule": verdict.decision_rule,
            "confidence": verdict.confidence,
            "decision_inputs_summary": verdict.decision_inputs_summary,
            # Backward-compatible consistent-only verdict
            "verdict_consistent": self._consistent_only_verdict(obs),
            # Alignment
            "alignment_sem": "MATCHED" if obs.aligned_sem else "MISSING_IN_PROBED",
            "alignment_local": "MATCHED" if obs.aligned_local else "MISSING_IN_PROBED",
            "alignment_sur": "MATCHED" if obs.aligned_sur else "MISSING_IN_PROBED",
            # Parse flags
            "parse_ok": obs.parse_ok_orig and obs.parse_ok_sur and (obs.parse_ok_sem or obs.parse_ok_local),
            "parse_ok_orig": obs.parse_ok_orig,
            "parse_ok_sem": obs.parse_ok_sem,
            "parse_ok_local": obs.parse_ok_local,
            "parse_ok_sur": obs.parse_ok_sur,
            # Audit metadata
            "flags": flags,
        }

    def _consistent_only_verdict(self, obs: StepObservation) -> str:
        """Compute consistent-only verdict for ablation comparison."""
        if not (obs.parse_ok_orig and obs.parse_ok_sem and obs.parse_ok_sur):
            return "UNPARSEABLE"
        if obs.semantic_delta and not obs.surface_delta:
            return "GROUNDED"
        if not obs.semantic_delta and not obs.surface_delta:
            return "INSENSITIVE"
        if obs.semantic_delta and obs.surface_delta:
            return "INPUT-SENSITIVE"
        if not obs.semantic_delta and obs.surface_delta:
            return "UNSTABLE"
        return "UNKNOWN"

    def get_cumulative_stats(self) -> dict:
        """Aggregate stats from cumulative flags. Parallel to Guard's get_trust_summary."""
        from collections import Counter
        # Normalize parameterized flags: "parse_ok:orig" → "parse_ok"
        base_flags = [f.split(":")[0] for f in self.cumulative_flags]
        counts = Counter(base_flags)
        rule_counts = Counter(v.decision_rule for v in self.verdict_history)
        return {
            "total_audited": len(self.verdict_history),
            "flag_counts": dict(counts),
            "rule_counts": dict(rule_counts),
        }
