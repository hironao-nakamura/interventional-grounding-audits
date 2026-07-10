"""Bootstrap confidence intervals for all reported metrics.

Resamples at the PROBLEM level (not certificate level) to respect
the correlation structure within problems.
"""

import json
import numpy as np
from collections import defaultdict


def compute_f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": p, "recall": r, "f1": f1}


def bootstrap_ci(
    problem_metrics: dict,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
):
    """Bootstrap CI by resampling problem-level TP/FP/FN counts.

    Args:
        problem_metrics: dict mapping problem_id -> {"tp": N, "fp": N, "fn": N}
        n_bootstrap: iterations
        confidence: CI level
    """
    rng = np.random.RandomState(seed)
    pids = sorted(problem_metrics.keys())
    n = len(pids)

    tp_arr = np.array([problem_metrics[p]["tp"] for p in pids])
    fp_arr = np.array([problem_metrics[p]["fp"] for p in pids])
    fn_arr = np.array([problem_metrics[p]["fn"] for p in pids])

    boot = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        tp = int(tp_arr[idx].sum())
        fp = int(fp_arr[idx].sum())
        fn = int(fn_arr[idx].sum())
        m = compute_f1_from_counts(tp, fp, fn)
        for k in boot:
            boot[k].append(m[k])

    alpha = 1 - confidence
    results = {}
    for k, vals in boot.items():
        v = np.array(vals)
        results[k] = {
            "mean": round(float(np.mean(v)), 4),
            "std": round(float(np.std(v)), 4),
            "ci_lower": round(float(np.percentile(v, 100 * alpha / 2)), 4),
            "ci_upper": round(float(np.percentile(v, 100 * (1 - alpha / 2))), 4),
        }
    return results
