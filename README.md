# Supplementary Material

## Interventional Grounding Audits: Black-Box Premise-Dependency Tests for LLM Chain-of-Thought via Predicate Substitution

This evidence pack accompanies the paper. It contains **all raw data,
certificates, and reproduction scripts** needed to verify every number in
the paper, plus the exact pipeline code that produced them.

> **arXiv release note.** Relative to the accepted workshop version, this
> release (a) fixes a probe-generator bug that could stack fresh-symbol
> prefixes (`zqzq...`) in substituted premises, and (b) fixes a
> ground-truth alignment bug (model CoT step numbers were compared against
> proof-tree step numbers by raw integer, which is wrong whenever the model
> prepends a premise-restatement step). **All Phase 1 model outputs were
> re-collected with the corrected pipeline** (GPT-4o snapshot
> `gpt-4o-2024-08-06`; Claude Sonnet 4.5, `claude-sonnet-4-5-20250929`;
> temperature 0), and all certificates and reports were recomputed with the
> corrected alignment. Every substituted prompt is stored in the pack and
> verified free of stacked prefixes. The accepted version's cross-model
> subject (Claude Sonnet 4, `claude-sonnet-4-20250514`) was retired from
> the API before the rerun and is replaced by the closest available
> successor, chosen to minimize model drift relative to the accepted version.
> See Appendix A of the paper for the full correction history.

**Releases**

- Workshop-accepted artifact: [`workshop-accepted`](https://github.com/hironao-nakamura/interventional-grounding-audits/releases/tag/workshop-accepted)
- Corrected arXiv artifact (this tree): [`arxiv-v1.0`](https://github.com/hironao-nakamura/interventional-grounding-audits/releases/tag/arxiv-v1.0)

---

## Quick Start (5 minutes)

```bash
pip install -r requirements.txt   # numpy only
python scripts/reproduce_all.py   # Recompute Tables 1-2 and verify against final_report.json
python src/validator.py --root . --models gpt-4o claude-sonnet-4-5   # Structural validation

# Or verify individual tables:
python scripts/reproduce_table1.py   # Table 1 (Main Results)
python scripts/reproduce_table2.py   # Table 2 (Ablation)

# Print every number quoted in the paper, straight from the reports:
python scripts/export_paper_numbers.py --model-dir evidence/gpt-4o
python scripts/export_paper_numbers.py --model-dir evidence/claude-sonnet-4-5
```

**Requirements:** Python 3.10+, see `requirements.txt` (numpy only).
No `openai` or `anthropic` packages are needed for reproduction; they are
required only for re-running Phase 1 (see below).

---

## Directory Structure

```
evidence/                     All certificate data
  gpt-4o/                     Main model (Tables 1-2); gpt-4o-2024-08-06
    p001/                     Per-problem directory
      original_cot.json       Phase 1: original LLM output (with prompt + provenance)
      probes/                 Probe outputs (semantic, local, surface),
                              each storing the exact prompt and modified premises sent
      certificates/           Phase 2: deterministic verdicts (cert_Si_Pj.json)
      problem_meta.json       Premises, question, proof tree
    final_report.json         Canonical aggregate report (single source of truth)
    bootstrap_ci.json         10,000-iteration bootstrap CIs (points == Table 1)
    ablation.json             All ablation configurations (Table 2)
    step_alignment_summary.json  CoT<->proof alignment coverage + disclosure flags
    fp_analysis.json          FP cause classification
    chain_length_analysis.json
    rawr_analysis.json        Right-Answer-Wrong-Reasoning cases
    misrepresentation_analysis.json
  claude-sonnet-4-5/          Transfer model, Appendix B; claude-sonnet-4-5-20250929

baselines/self_consistency/gpt-4o/   5 samples x 50 problems (temperature 0.7,
                                     gpt-4o-2024-08-06) + baseline_report.json

data/prontoqa_50/             Input problems + proof trees

config/audit_policy.json      Externalized audit parameters

src/                          Pipeline + evaluation code
scripts/                      Phase 1/2, baseline, report generation, reproduction
tests/                        Test suite (incl. frozen accepted-version fixtures)
tests/fixtures/               Accepted-version original CoTs for alignment regression tests

checksums.sha256              SHA256 of all files
VALIDATION_LOG.txt            Output of src/validator.py
REPRODUCTION_LOG.txt          Output of scripts/reproduce_all.py
```

---

## Ground-Truth Alignment (important)

Model CoT step numbers are **not** proof-tree step numbers: models sometimes
prepend a premise-restatement step (e.g., "Step 1: Fae is a barpus"), shifting
every subsequent step by one. All evaluation in this pack therefore aligns
CoT steps to proof-tree steps **by normalized conclusion**
(`src/step_alignment.py`), never by raw integer.

Policy (recorded in `step_alignment_summary.json`):

- **Unmatched CoT steps** (restatements, unparseable steps) are excluded from
  primary metrics and reported as alignment coverage.
- **Ambiguous alignments** (duplicate conclusions, many-to-one matches)
  exclude the whole problem from primary metrics.
- **Lower-bound metrics** (paper Section 5) additionally treat unmatched or
  unparseable ground-truth dependencies as false negatives.

GPT-4o coverage: 179/196 CoT steps matched (17 unmatched across 8 problems;
9 proof steps unaudited in 4 problems; 0 ambiguous).
Claude Sonnet 4.5 coverage: 188/238 CoT steps matched (50 unmatched —
exactly one restatement step per problem; every proof-tree step audited;
0 ambiguous).

---

## Certificate Schema

Each `cert_Si_Pj.json` contains:

- `problem_id`, `step_id`, `premise_id` — identifiers. **`step_id` is the
  model's CoT step number, not a proof-tree step number.**
- `cot_step_id` — equals `step_id` (explicit alias).
- `matched_proof_step_id` — proof-tree step aligned by normalized conclusion
  (`null` for unmatched CoT steps).
- `evaluation_excluded`, `exclusion_reason` — `true` +
  `"unmatched_cot_step"` / `"ambiguous_alignment_problem"` for certificates
  excluded from primary metrics.
- `primary_metric_excluded_problem`, `primary_metric_exclusion_reason` —
  problem-level ambiguity exclusion flags.
- `verdict` — GROUNDED, INSENSITIVE, INPUT-SENSITIVE, UNSTABLE, or UNPARSEABLE
  (combined A+A' protocol).
- `verdict_consistent` — verdict under consistent substitution alone
  (the Table 1 "A consistent" evaluator).
- `decision_rule` — named rule that produced the verdict
  (e.g., `R_GROUNDED_PRED_CHANGE`); `confidence`; `flags`.
- `phi_original`, `phi_semantic`, `phi_local`, `phi_surface` — normalized
  step conclusions per condition.
- `semantic_delta`, `local_delta`, `surface_delta` — boolean change indicators.
- `parse_ok_orig`, `parse_ok_sem`, `parse_ok_local`, `parse_ok_sur` — parse
  success per output.
- `citation_detected`, `citation_type`, `citation_evidence` — premise-citation
  detection for the misrepresentation analysis.

Phase 1 raw-output JSONs additionally record `model`, `temperature`,
`timestamp`, and `usage` for full provenance, plus the exact `prompt` and
(for probes) `modified_premises` sent to the model.

---

## Phase 1 / Phase 2 Separation

- **Phase 1** (`evidence/*/p*/original_cot.json`, `evidence/*/p*/probes/`,
  `baselines/self_consistency/*/`): raw LLM outputs. Non-deterministic;
  requires an API key to re-run:

  ```bash
  OPENAI_API_KEY=... python scripts/run_phase1.py --model gpt-4o-2024-08-06 --out evidence/gpt-4o --overwrite
  ANTHROPIC_API_KEY=... python scripts/run_phase1.py --model claude-sonnet-4-5-20250929 --out evidence/claude-sonnet-4-5 --overwrite

  # Faster, thread-pooled variant (same file layout):
  OPENAI_API_KEY=... python scripts/full_phase1_concurrent.py --model gpt-4o-2024-08-06 --out evidence/gpt-4o --workers 6

  # Self-consistency baseline (5 samples x 50 problems, temperature 0.7):
  OPENAI_API_KEY=... python scripts/run_sc_baseline.py --model gpt-4o-2024-08-06 --out-name gpt-4o --workers 6
  ```

- **Phase 2** (`evidence/*/p*/certificates/`): deterministic verdicts
  computed from Phase 1 outputs — no API key needed:

  ```bash
  python scripts/run_phase2.py --model-dir evidence/gpt-4o --overwrite
  python scripts/recompute_reports.py --model-dir evidence/gpt-4o
  ```

- **Full pipeline** (Phase 1 + 2 + reports + reproduction):
  `scripts/run_full_pipeline.sh <model> <evidence-dir>` (asks for
  confirmation before overwriting Phase 1 outputs).

Readers only need to verify Phase 2. Phase 1 data is included for
completeness and auditability.

---

## Verification

1. **Checksums:** `sha256sum -c checksums.sha256` verifies no file was modified.
2. **Automated validation:** `python src/validator.py --root . --models gpt-4o claude-sonnet-4-5`
   checks dataset size, per-problem files, SHA256 integrity, stored prompts
   (including absence of stacked fresh-symbol prefixes), certificate
   counts (parsed CoT steps x premises), alignment metadata against a fresh
   `align_cot_to_proof` run, schema completeness, verdict/decision-rule
   consistency, and report/bootstrap agreement. See `VALIDATION_LOG.txt`.
3. **Reproduce numbers:** `python scripts/reproduce_all.py` recomputes
   Tables 1-2 from raw certificates and fails loudly on any mismatch with
   `final_report.json`. See `REPRODUCTION_LOG.txt`.
4. **README consistency:** `python scripts/check_readme_numbers.py` verifies
   the Key Numbers table below against the report JSONs.

---

## Key Numbers (Paper Reference)

All values below are read from `evidence/gpt-4o/final_report.json` /
`bootstrap_ci.json` (and `evidence/claude-sonnet-4-5/final_report.json` for
Appendix B); `scripts/check_readme_numbers.py` enforces agreement.

| Location | Metric | Value |
|----------|--------|-------|
| Table 1 | A consistent F1 (full) | 0.806 |
| Table 1 | A consistent F1 (pred-determining) | 0.885 |
| Table 1 | A consistent recall (pred-determining) | 100.0% |
| Table 1 | A+A'+cascade F1 (full) | 0.819 |
| Table 1 | Self-consistency F1 (full) | 0.343 |
| Table 1 | String-diff F1 (full) | 0.787 |
| Table 1 | 95% CI, A vs self-consistency | non-overlapping ([0.760, 0.852] vs [0.317, 0.371]) |
| Sec. 3.1 | Certificates (total / evaluable) | 1127 / 1031 |
| Sec. 3.1 | Unmatched CoT steps (GPT-4o) | 17 (8 problems) |
| Sec. 3.4 | RAWR cases (A consistent, full) | 33/50 (66%) |
| Sec. 3.4 | RAWR breakdown | 33 structural-only, 0 predicate-determining |
| Sec. 3.4 | Misrepresentation cases | 30 |
| Sec. 3.4 | Semantic-delta FP candidates / metric-counted | 40 / 39 |
| Sec. 5 | Lower-bound F1 (A consistent, full) | 0.703 |
| App. B | Claude Sonnet 4.5 F1 (full, evaluable steps) | 0.872 |
| App. B | Claude Sonnet 4.5 lower-bound F1 | 0.836 |
| App. B | Claude Sonnet 4.5 unmatched CoT steps | 50 (50 problems) |

Note on Appendix B: the accepted workshop version studied Claude Sonnet 4
(`claude-sonnet-4-20250514`; retired from the API in June 2026) and reported
F1 = 0.161 (base normalizer) and 0.320 (post-hoc adapted normalizer). Those
figures were computed under the misaligned integer-keyed ground truth and
from probes affected by the generator issue; they are superseded and are not
comparable with the Claude Sonnet 4.5 values above, which measure a
different (successor) model with the corrected pipeline.
