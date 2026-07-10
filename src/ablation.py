"""Ablation runner: 3-tier comparison aligned with the deployed guard framework cascade analysis.

Five configurations to demonstrate the contribution of each component:

Tier 1 — Single probes:
  A:  Consistent substitution only (normalized, with surface control)
      → Detects direct predicate-determining dependencies
  A': Local substitution only (normalized, with surface control)
      → Detects cascade dependencies (including structural premises)

Tier 2 — Combined probes:
  A+A': Consistent OR Local (normalized, with surface control)
      → Detects both direct + cascade dependencies

Tier 3 — Ablation of surface control:
  A+A' w/o surface: Consistent OR Local (normalized, NO surface control)
      → Shows value of surface control in reducing false positives

Tier 4 — Baseline:
  String-diff: Consistent substitution, raw string comparison (no normalization)
      → Shows value of normalized proposition equivalence
"""

import json
import os
import sys

# import config  # paths configured at runtime


def run_ablation(
    certificates: list[dict],
    problems: list[dict],
    cot_steps_by_pid: dict[str, list[dict]],
) -> dict:
    """Run all ablation configurations using saved certificates.

    Certificates contain phi_original, phi_semantic, phi_local, phi_surface
    and individual parse/alignment flags. We re-judge under each config,
    evaluated against aligned CoT-keyed ground truth via the shared
    evaluate_judged path (fixed metric policy).
    """
    from src.ground_truth import evaluate_judged

    configs = [
        ("A_consistent_only", _judge_consistent_only),
        ("A'_local_only", _judge_local_only),
        ("A+A'_combined", _judge_combined),
        ("A+A'_no_surface_ctrl", _judge_combined_no_surface),
        ("string_diff_baseline", _judge_string_diff),
    ]

    results = {}
    for config_name, judge_fn in configs:
        r = evaluate_judged(certificates, problems, cot_steps_by_pid, judge_fn)
        results[config_name] = {
            "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
        }

    return results


# ============================================================
# Judge functions for each ablation configuration
# ============================================================

def _judge_consistent_only(cert: dict) -> str:
    """A: Consistent substitution + normalization + surface control."""
    if not (cert.get("parse_ok_orig") and cert.get("parse_ok_sem") and cert.get("parse_ok_sur")):
        return "UNPARSEABLE"

    sem_delta = cert.get("semantic_delta", False)
    sur_delta = cert.get("surface_delta", False)

    if sem_delta and not sur_delta:
        return "GROUNDED"
    if sem_delta and sur_delta:
        return "INPUT-SENSITIVE"
    return "INSENSITIVE"


def _judge_local_only(cert: dict) -> str:
    """A': Local substitution + normalization + surface control."""
    if not (cert.get("parse_ok_orig") and cert.get("parse_ok_local") and cert.get("parse_ok_sur")):
        return "UNPARSEABLE"

    local_delta = cert.get("local_delta", False)
    sur_delta = cert.get("surface_delta", False)

    if local_delta and not sur_delta:
        return "GROUNDED"
    if local_delta and sur_delta:
        return "INPUT-SENSITIVE"
    return "INSENSITIVE"


def _judge_combined(cert: dict) -> str:
    """A+A': Consistent OR Local + normalization + surface control.
    This is the full method (same as the primary verdict).
    """
    return cert.get("verdict", "UNPARSEABLE")


def _judge_combined_no_surface(cert: dict) -> str:
    """A+A' without surface control: shows value of surface filtering."""
    if not cert.get("parse_ok_orig"):
        return "UNPARSEABLE"

    has_sem = cert.get("parse_ok_sem", False)
    has_local = cert.get("parse_ok_local", False)

    if not (has_sem or has_local):
        return "UNPARSEABLE"

    consistent_delta = cert.get("semantic_delta", False) if has_sem else False
    local_delta = cert.get("local_delta", False) if has_local else False

    if consistent_delta or local_delta:
        return "GROUNDED"
    return "INSENSITIVE"


def _judge_string_diff(cert: dict) -> str:
    """Baseline: Consistent substitution + raw string diff (no normalization).
    Approximated using phi values since raw text not stored in certificates.
    """
    phi_orig = cert.get("phi_original")
    phi_sem = cert.get("phi_semantic")

    if phi_orig is None or phi_sem is None:
        return "UNPARSEABLE"

    if phi_orig != phi_sem:
        return "GROUNDED"
    return "INSENSITIVE"


# ============================================================
# Pretty print
# ============================================================

def print_ablation(results: dict) -> None:
    """Print ablation comparison table."""
    print("\n" + "=" * 78)
    print("  ABLATION: Consistent vs Local vs Combined vs Surface Control")
    print("=" * 78)

    print(f"\n  {'Configuration':<35} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 72)

    for name, m in results.items():
        label = name.replace("_", " ")
        print(f"  {label:<35} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
              f" {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    print("=" * 78)

    # Highlight key comparisons
    if "A_consistent_only" in results and "A+A'_combined" in results:
        a = results["A_consistent_only"]
        aa = results["A+A'_combined"]
        r_gain = aa["recall"] - a["recall"]
        f1_gain = aa["f1"] - a["f1"]
        fn_reduction = a["fn"] - aa["fn"]
        print(f"\n  Key insight: Local substitution adds:")
        print(f"    Recall: {a['recall']:.3f} → {aa['recall']:.3f} (+{r_gain:.3f})")
        print(f"    F1:     {a['f1']:.3f} → {aa['f1']:.3f} (+{f1_gain:.3f})")
        print(f"    FN reduction: {a['fn']} → {aa['fn']} (-{fn_reduction})")
