#!/usr/bin/env python3
"""Reproduce ALL tables and analysis reports from the evidence pack.

Requires NO API calls. Steps:
  1. Recompute Table 1 and Table 2 from certificates and compare against
     final_report.json (scripts/reproduce_table1.py / reproduce_table2.py).
  2. Recompute every report/analysis JSON into a temporary directory with
     scripts/recompute_reports.py --out-dir and byte-compare against the
     shipped reports (final_report, ablation, bootstrap_ci, rawr,
     misrepresentation, fp, chain_length, step_alignment_summary,
     baseline_report).

Exits non-zero if any reproduction fails or does not match.
Usage: python scripts/reproduce_all.py
"""

import filecmp
import json
import os
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT_FILES = [
    "final_report.json",
    "ablation.json",
    "bootstrap_ci.json",
    "step_alignment_summary.json",
    "rawr_analysis.json",
    "misrepresentation_analysis.json",
    "fp_analysis.json",
    "chain_length_analysis.json",
]

for script in ["scripts/reproduce_table1.py", "scripts/reproduce_table2.py"]:
    path = os.path.join(BASE, script)
    print(f"\n{'#' * 65}", flush=True)
    print(f"  Running: {script}", flush=True)
    print(f"{'#' * 65}\n", flush=True)
    subprocess.run([sys.executable, path], cwd=BASE, check=True)

print(f"\n{'#' * 65}", flush=True)
print("  Recomputing all analysis reports and comparing to shipped JSONs", flush=True)
print(f"{'#' * 65}\n", flush=True)

failures = []
for model in ["gpt-4o", "claude-sonnet-4-5"]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, os.path.join(BASE, "scripts", "recompute_reports.py"),
             "--model-dir", f"evidence/{model}", "--out-dir", tmp],
            cwd=BASE, check=True, stdout=subprocess.DEVNULL)
        for name in REPORT_FILES:
            shipped = os.path.join(BASE, "evidence", model, name)
            fresh = os.path.join(tmp, name)
            if not filecmp.cmp(shipped, fresh, shallow=False):
                failures.append(f"{model}/{name}")
                print(f"  [MISMATCH] evidence/{model}/{name}", flush=True)
            else:
                print(f"  [OK] evidence/{model}/{name}", flush=True)
        bl_shipped = os.path.join(
            BASE, "baselines", "self_consistency", model, "baseline_report.json")
        bl_fresh = os.path.join(tmp, "baseline_report.json")
        if os.path.exists(bl_shipped):
            if not (os.path.exists(bl_fresh)
                    and filecmp.cmp(bl_shipped, bl_fresh, shallow=False)):
                failures.append(f"{model}/baseline_report.json")
                print(f"  [MISMATCH] baselines/self_consistency/{model}/"
                      f"baseline_report.json", flush=True)
            else:
                print(f"  [OK] baselines/self_consistency/{model}/"
                      f"baseline_report.json", flush=True)

print(f"\n{'=' * 65}", flush=True)
if failures:
    print(f"  REPRODUCTION FAILED: {len(failures)} mismatched report(s)", flush=True)
    print(f"{'=' * 65}", flush=True)
    sys.exit(1)
print("  ALL REPRODUCTIONS COMPLETE (tables + analysis reports verified)", flush=True)
print(f"{'=' * 65}", flush=True)
