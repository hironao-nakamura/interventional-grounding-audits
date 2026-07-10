"""Cascade filter for local substitution certificates.

When local_sub(Pj) causes Step Si to change, AND Step S_{i-1} also
changed w.r.t. the same Pj, then Si's change is CASCADE, not DIRECT.

This filters false positives from local substitution without needing
ground truth — it's a purely structural heuristic based on step ordering.
"""

from collections import defaultdict


def apply_cascade_filter(certificates: list[dict]) -> list[dict]:
    """Apply cascade filter to certificates.

    For each (problem, premise) group, if a step is GROUNDED via local_delta
    and the previous step was also GROUNDED for the same premise,
    reclassify as CASCADE.

    Only filters based on local_delta (cascade effect). Consistent-only
    GROUNDED verdicts are left untouched.

    Returns new list of certificates (copies, originals unchanged).
    """
    # Group by (problem_id, premise_id)
    groups = defaultdict(list)
    for cert in certificates:
        key = (cert["problem_id"], cert["premise_id"])
        groups[key].append(cert)

    filtered = []
    cascade_count = 0

    for key, certs in groups.items():
        certs_sorted = sorted(certs, key=lambda c: c["step_id"])

        # Track which steps are GROUNDED via local_delta
        local_grounded_steps = set()
        for cert in certs_sorted:
            local_delta = cert.get("local_delta", False)
            consistent_delta = cert.get("semantic_delta", False)
            verdict = cert.get("verdict", "")

            if verdict in ("GROUNDED", "INPUT-SENSITIVE") and local_delta:
                prev_step = cert["step_id"] - 1
                if prev_step in local_grounded_steps and not consistent_delta:
                    # Cascade: previous step also GROUNDED via local,
                    # and this step is NOT detected by consistent sub
                    # → likely propagated change, not direct dependency
                    cert_copy = dict(cert)
                    cert_copy["verdict_pre_filter"] = cert_copy["verdict"]
                    cert_copy["verdict"] = "CASCADE"
                    cert_copy["decision_rule"] = "R_CASCADE_FILTERED"
                    cert_copy["cascade_note"] = (
                        f"Step {prev_step} also GROUNDED(local) for {cert['premise_id']}, "
                        f"and consistent sub did not detect — likely cascade"
                    )
                    filtered.append(cert_copy)
                    cascade_count += 1
                    # Still add to grounded set so downstream steps
                    # are also filtered
                    local_grounded_steps.add(cert["step_id"])
                    continue

                # First GROUNDED step in chain — likely direct
                local_grounded_steps.add(cert["step_id"])

            filtered.append(dict(cert))

    return filtered, cascade_count


def evaluate_with_cascade_filter(certificates, problems, cot_steps_by_pid):
    """Apply cascade filter and re-evaluate P/R/F1 against aligned GT.

    This is the single shared function every script uses for the
    "A+A' + cascade" rows, so INPUT-SENSITIVE and CASCADE handling
    cannot drift between the paper tables, reports, and bootstrap CIs.
    CASCADE verdicts are predicted-negative in the direct-dependency metric.
    """
    from src.ground_truth import evaluate_judged

    filtered_certs, n_cascaded = apply_cascade_filter(certificates)

    judge = lambda cert: cert.get("verdict", "UNPARSEABLE")  # noqa: E731
    results = {}
    for gt_name, pred_only in [("full", False), ("pred_only", True)]:
        r = evaluate_judged(
            filtered_certs, problems, cot_steps_by_pid, judge, pred_only=pred_only)
        results[gt_name] = {
            "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
            "per_problem": r["per_problem"],
        }

    results["n_cascaded"] = n_cascaded
    return results


def print_cascade_report(results):
    """Print cascade filter results."""
    print("\n" + "=" * 70)
    print("  CASCADE FILTER RESULTS")
    print("=" * 70)
    print(f"\n  Certificates reclassified as CASCADE: {results['n_cascaded']}")
    print(f"\n  {'Metric':<20} {'Full':>10} {'Pred-only':>10}")
    print("  " + "-" * 40)
    for metric in ["precision", "recall", "f1"]:
        print(f"  {metric.capitalize():<20} "
              f"{results['full'][metric]:>10.4f} "
              f"{results['pred_only'][metric]:>10.4f}")
    print(f"  {'TP':<20} {results['full']['tp']:>10} {results['pred_only']['tp']:>10}")
    print(f"  {'FP':<20} {results['full']['fp']:>10} {results['pred_only']['fp']:>10}")
    print(f"  {'FN':<20} {results['full']['fn']:>10} {results['pred_only']['fn']:>10}")
    print("=" * 70)
