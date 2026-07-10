"""Fingerprinting — parallel to Guard's fingerprint_action() + detect_fingerprint_loop().

Detects when an LLM produces identical outputs regardless of probing:
- If ALL probed outputs have the same fingerprint as original → model is insensitive
- If consecutive probes produce identical fingerprints → possible degenerate behavior
"""

import hashlib
import re


def fingerprint_cot(raw_response: str) -> str:
    """Hash the CoT output for dedup/loop detection.

    Normalizes whitespace before hashing for robust comparison.
    Returns first 16 chars of SHA256 for compact representation.
    """
    clean = re.sub(r'\s+', ' ', raw_response.strip())
    return hashlib.sha256(clean.encode()).hexdigest()[:16]


def detect_insensitive_model(original_fp: str, probe_fps: list[str]) -> bool:
    """Check if model is completely insensitive to all probing.

    If ALL probe outputs produce the same fingerprint as the original,
    the model is not responding to any intervention at all.
    """
    if not probe_fps:
        return False
    return all(fp == original_fp for fp in probe_fps)


def compute_fingerprint_stats(
    original_fp: str,
    semantic_fps: list[str],
    local_fps: list[str],
    surface_fps: list[str],
) -> dict:
    """Compute fingerprint-based statistics for a single problem.

    Returns:
        dict with dedup rates and sensitivity indicators.
    """
    all_fps = semantic_fps + local_fps + surface_fps
    n_total = len(all_fps)
    n_identical_to_orig = sum(1 for fp in all_fps if fp == original_fp)
    n_unique = len(set(all_fps + [original_fp]))

    # Per-probe-type sensitivity
    sem_changed = sum(1 for fp in semantic_fps if fp != original_fp)
    loc_changed = sum(1 for fp in local_fps if fp != original_fp)
    sur_changed = sum(1 for fp in surface_fps if fp != original_fp)

    return {
        "original_fingerprint": original_fp,
        "n_total_probes": n_total,
        "n_identical_to_original": n_identical_to_orig,
        "n_unique_outputs": n_unique,
        "identity_rate": n_identical_to_orig / n_total if n_total > 0 else 0,
        "semantic_change_rate": sem_changed / len(semantic_fps) if semantic_fps else 0,
        "local_change_rate": loc_changed / len(local_fps) if local_fps else 0,
        "surface_change_rate": sur_changed / len(surface_fps) if surface_fps else 0,
        "model_insensitive": n_identical_to_orig == n_total,
    }
