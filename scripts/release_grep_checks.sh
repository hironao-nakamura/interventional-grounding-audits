#!/usr/bin/env bash
set -euo pipefail

fail=0

check_zero () {
  label="$1"
  pattern="$2"
  shift 2
  echo "[check] $label"
  if grep -RInE \
      --exclude-dir=.venv \
      --exclude-dir=.git \
      --exclude-dir=_audit_logs \
      --exclude-dir=_release_staging \
      --exclude-dir=arxiv_bundle \
      --exclude='*.zip' \
      --exclude='release_grep_checks.sh' \
      "$pattern" "$@"; then
    echo "FAILED: $label"
    fail=1
  else
    echo "OK: $label"
  fi
}

python - <<'PY'
import glob, json, sys
bad = []
for f in glob.glob('evidence/*/p*/probes/*.json'):
    d = json.load(open(f))
    target = d.get('prompt', '') + '\n' + '\n'.join(p.get('text', '') for p in d.get('modified_premises', []))
    if 'zqzq' in target.lower():
        bad.append(f)
if bad:
    print('FAILED: recursive zq prefixes in prompts/modified_premises')
    print('\n'.join(bad[:20]))
    sys.exit(1)
print('OK: no recursive zq prefixes in prompts/modified_premises')
PY

paper_targets=(README.md evidence)
if [[ -f paper.tex ]]; then
  paper_targets+=(paper.tex)
fi
code_targets=(src scripts README.md)
if [[ -f paper.tex ]]; then
  code_targets+=(paper.tex)
fi

check_zero "missing Appendix B invented quote / deprecated wording" 'zqbrimpus|chain of reasoning breaks here|direct successor|Published as a conference' "${paper_targets[@]}"
check_zero "AI-ish comments" 'let me redo|the the|framework framework|adapt run_full_experiment|Claude 3\.5|TODO|FIXME|placeholder' "${code_targets[@]}"
# Key patterns require a key-like value after '=' so documented usage
# examples (OPENAI_API_KEY=... python scripts/run_phase1.py) do not trip.
check_zero "secrets" 'OPENAI_API_KEY=[A-Za-z0-9]|ANTHROPIC_API_KEY=[A-Za-z0-9]|sk-[A-Za-z0-9]|BEGIN PRIVATE KEY' .

exit "$fail"
