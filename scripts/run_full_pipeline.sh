#!/usr/bin/env bash
# Full pipeline: tests -> Phase 1 (API) -> Phase 2 (deterministic) -> reports -> reproduction.
#
# Usage:
#   OPENAI_API_KEY=... ./scripts/run_full_pipeline.sh gpt-4o evidence/gpt-4o
#   ANTHROPIC_API_KEY=... ./scripts/run_full_pipeline.sh claude-sonnet-4-5-20250929 evidence/claude-sonnet-4-5
#
# Phase 1 calls the model API and OVERWRITES evidence/<model>/ raw outputs.
# Phases 2+ are deterministic recomputations from the saved raw outputs.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: run_full_pipeline.sh <model> <evidence-dir>}"
OUT="${2:?usage: run_full_pipeline.sh <model> <evidence-dir>}"

if [[ "$MODEL" == claude* ]]; then
  : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set for Claude models}"
else
  : "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for OpenAI models}"
fi

echo "Step 0/5: alignment regression tests"
python3 -m pytest -q tests/test_step_alignment.py tests/test_probe_generator.py

echo "Phase 1 will OVERWRITE raw outputs in ${OUT}."
read -r -p "Continue? [y/N] " answer
if [[ "${answer}" != "y" && "${answer}" != "Y" ]]; then
  echo "Aborted."
  exit 1
fi

echo "Step 1/5: Phase 1 (LLM calls)"
python3 scripts/run_phase1.py --model "$MODEL" --out "$OUT" --overwrite

echo "Step 2/5: Phase 2 (deterministic certificates)"
python3 scripts/run_phase2.py --model-dir "$OUT" --overwrite

echo "Step 3/5: recompute reports"
python3 scripts/recompute_reports.py --model-dir "$OUT"

echo "Step 4/5: reproduce tables"
python3 scripts/reproduce_all.py

echo "Step 5/5: validate"
python3 src/validator.py --root . --models "$(basename "$OUT")"

echo "Pipeline complete."
