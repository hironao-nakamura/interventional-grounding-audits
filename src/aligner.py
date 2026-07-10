"""Step 6a: Align steps between original and probed CoT outputs."""


def align_steps(original_steps: list[dict], probed_steps: list[dict]) -> list[dict]:
    """Match steps by step_id. Return aligned pairs + missing/extra.
    
    Each element: {step_id, original, probed, alignment}
    alignment: MATCHED / MISSING_IN_PROBED / EXTRA_IN_PROBED
    """
    orig_map = {s["step_id"]: s for s in original_steps}
    prob_map = {s["step_id"]: s for s in probed_steps}
    all_ids = sorted(set(orig_map) | set(prob_map))
    
    aligned = []
    for sid in all_ids:
        o = orig_map.get(sid)
        p = prob_map.get(sid)
        if o and p:
            aligned.append({
                "step_id": sid,
                "original": o,
                "probed": p,
                "alignment": "MATCHED",
            })
        elif o and not p:
            aligned.append({
                "step_id": sid,
                "original": o,
                "probed": None,
                "alignment": "MISSING_IN_PROBED",
            })
        else:
            aligned.append({
                "step_id": sid,
                "original": None,
                "probed": p,
                "alignment": "EXTRA_IN_PROBED",
            })
    return aligned


def align_steps_content_aware(
    original_steps: list[dict],
    probed_steps: list[dict],
) -> list[dict]:
    """Content-aware alignment: first try Step ID match, then detect step-count shifts.
    
    If the probed output has one extra initial step (entity restatement), shift alignment
    so that probed[i+1] maps to original[i]. This handles cases where the LLM adds
    or removes an introductory step.
    """
    n_orig = len(original_steps)
    n_prob = len(probed_steps)
    
    # If same number of steps, use simple ID-based alignment
    if n_orig == n_prob:
        return align_steps(original_steps, probed_steps)
    
    # If probed has exactly 1 more step, check for initial offset
    if n_prob == n_orig + 1:
        # Check if probed steps [1..N+1] align well with original steps [0..N]
        # by comparing normalized propositions
        shifted_matches = 0
        for i in range(n_orig):
            orig_norm = original_steps[i].get("normalized")
            prob_norm = probed_steps[i + 1].get("normalized")  # shifted by 1
            if orig_norm and prob_norm and orig_norm == prob_norm:
                shifted_matches += 1
        
        # If most steps match with shift, use shifted alignment
        if shifted_matches >= n_orig * 0.5:
            aligned = []
            # The extra step at the beginning gets EXTRA_IN_PROBED
            aligned.append({
                "step_id": probed_steps[0]["step_id"],
                "original": None,
                "probed": probed_steps[0],
                "alignment": "EXTRA_IN_PROBED",
            })
            # Align shifted
            for i in range(n_orig):
                orig = original_steps[i]
                prob = probed_steps[i + 1]
                aligned.append({
                    "step_id": orig["step_id"],
                    "original": orig,
                    "probed": prob,
                    "alignment": "MATCHED",
                })
            return aligned
    
    # If probed has exactly 1 fewer step, check if original has an extra initial step
    if n_prob == n_orig - 1:
        shifted_matches = 0
        for i in range(n_prob):
            orig_norm = original_steps[i + 1].get("normalized")
            prob_norm = probed_steps[i].get("normalized")
            if orig_norm and prob_norm and orig_norm == prob_norm:
                shifted_matches += 1
        
        if shifted_matches >= n_prob * 0.5:
            aligned = []
            aligned.append({
                "step_id": original_steps[0]["step_id"],
                "original": original_steps[0],
                "probed": None,
                "alignment": "MISSING_IN_PROBED",
            })
            for i in range(n_prob):
                orig = original_steps[i + 1]
                prob = probed_steps[i]
                aligned.append({
                    "step_id": orig["step_id"],
                    "original": orig,
                    "probed": prob,
                    "alignment": "MATCHED",
                })
            return aligned
    
    # Fallback: simple ID-based alignment
    return align_steps(original_steps, probed_steps)
