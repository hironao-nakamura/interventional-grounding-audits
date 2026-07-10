"""Step 6b: Verdict judgment for each (step, premise) pair.

Supports three probe types:
- semantic (consistent substitution): detects direct predicate-determining dependencies
- local_semantic (local substitution): detects cascade dependencies (chain-breaking)
- surface (meaning-preserving rephrase): filters false positives
"""


def judge_verdict(
    phi_orig: str | None,
    phi_sem: str | None,
    phi_sur: str | None,
    parse_ok_orig: bool,
    parse_ok_sem: bool,
    parse_ok_sur: bool,
) -> str:
    """Determine verdict type based on consistent semantic + surface probes only.
    
    Returns one of: GROUNDED, INSENSITIVE, INPUT-SENSITIVE, UNSTABLE, UNPARSEABLE
    """
    if not (parse_ok_orig and parse_ok_sem and parse_ok_sur):
        return "UNPARSEABLE"
    
    semantic_delta = (phi_orig != phi_sem)
    surface_delta = (phi_orig != phi_sur)
    
    if semantic_delta and not surface_delta:
        return "GROUNDED"
    if not semantic_delta and not surface_delta:
        return "INSENSITIVE"
    if semantic_delta and surface_delta:
        return "INPUT-SENSITIVE"
    if not semantic_delta and surface_delta:
        return "UNSTABLE"
    
    return "UNKNOWN"


def judge_combined_verdict(
    phi_orig: str | None,
    phi_sem: str | None,
    phi_local: str | None,
    phi_sur: str | None,
    parse_ok_orig: bool,
    parse_ok_sem: bool,
    parse_ok_local: bool,
    parse_ok_sur: bool,
) -> str:
    """Combined verdict using consistent + local substitution + surface control.
    
    Logic:
      any_semantic_delta = (consistent_delta OR local_delta)
      surface_delta = (phi_orig != phi_sur)
    
    Returns one of:
      GROUNDED         — any_semantic_delta AND NOT surface_delta
      INSENSITIVE      — NOT any_semantic_delta AND NOT surface_delta
      INPUT-SENSITIVE  — any_semantic_delta AND surface_delta
      UNSTABLE         — NOT any_semantic_delta AND surface_delta
      UNPARSEABLE      — some parse failure
    """
    # Need at least original + surface + one of (consistent, local) to be parseable
    if not (parse_ok_orig and parse_ok_sur):
        return "UNPARSEABLE"
    
    # Compute individual deltas (only if parseable)
    consistent_delta = (phi_orig != phi_sem) if (parse_ok_sem and phi_sem is not None) else False
    local_delta = (phi_orig != phi_local) if (parse_ok_local and phi_local is not None) else False
    
    # At least one semantic probe must be parseable
    if not (parse_ok_sem or parse_ok_local):
        return "UNPARSEABLE"
    
    any_semantic_delta = consistent_delta or local_delta
    surface_delta = (phi_orig != phi_sur)
    
    if any_semantic_delta and not surface_delta:
        return "GROUNDED"
    if not any_semantic_delta and not surface_delta:
        return "INSENSITIVE"
    if any_semantic_delta and surface_delta:
        return "INPUT-SENSITIVE"
    if not any_semantic_delta and surface_delta:
        return "UNSTABLE"
    
    return "UNKNOWN"


def build_certificate(
    problem_id: str,
    step_id: int,
    premise_id: str,
    phi_orig: str | None,
    phi_sem: str | None,
    phi_sur: str | None,
    parse_ok_orig: bool,
    parse_ok_sem: bool,
    parse_ok_sur: bool,
    alignment_sem: str,
    alignment_sur: str,
    phi_local: str | None = None,
    parse_ok_local: bool = False,
    alignment_local: str = "NOT_RUN",
) -> dict:
    """Build a full audit certificate for one (step, premise) pair.
    
    Includes both the consistent-only verdict and the combined verdict.
    """
    # Consistent-only verdict (backward compatible)
    verdict_consistent = judge_verdict(
        phi_orig, phi_sem, phi_sur,
        parse_ok_orig, parse_ok_sem, parse_ok_sur,
    )
    
    # Combined verdict (consistent OR local, with surface control)
    verdict_combined = judge_combined_verdict(
        phi_orig, phi_sem, phi_local, phi_sur,
        parse_ok_orig, parse_ok_sem, parse_ok_local, parse_ok_sur,
    )
    
    consistent_delta = (phi_orig != phi_sem) if (phi_orig and phi_sem) else None
    local_delta = (phi_orig != phi_local) if (phi_orig and phi_local) else None
    surface_delta = (phi_orig != phi_sur) if (phi_orig and phi_sur) else None
    
    return {
        "problem_id": problem_id,
        "step_id": step_id,
        "premise_id": premise_id,
        "phi_original": phi_orig,
        "phi_semantic": phi_sem,
        "phi_local": phi_local,
        "phi_surface": phi_sur,
        "semantic_delta": consistent_delta,
        "local_delta": local_delta,
        "surface_delta": surface_delta,
        "verdict": verdict_combined,            # Primary: combined
        "verdict_consistent": verdict_consistent,  # For ablation
        "alignment_sem": alignment_sem,
        "alignment_local": alignment_local,
        "alignment_sur": alignment_sur,
        "parse_ok": parse_ok_orig and parse_ok_sur and (parse_ok_sem or parse_ok_local),
        "parse_ok_orig": parse_ok_orig,
        "parse_ok_sem": parse_ok_sem,
        "parse_ok_local": parse_ok_local,
        "parse_ok_sur": parse_ok_sur,
    }
