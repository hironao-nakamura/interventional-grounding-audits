# Interventional Grounding Audits for LLM Chain-of-Thought

Supplementary evidence pack for:

> **Interventional Grounding Audits for LLM Chain-of-Thought Faithfulness**
> Hironao Nakamura
> ICLR 2026 Workshop on Logical Reasoning of Large Language Models

## Quick Start

```bash
unzip supplementary.zip -d supplementary
cd supplementary
pip install -r requirements.txt   # numpy only
python scripts/reproduce_all.py   # Reproduces ALL paper tables (no API key needed)
```

## What's Inside

The ZIP contains all raw data, certificates, and reproduction scripts needed to verify every number in the paper:

- `evidence/` — Per-problem probe outputs and verdict certificates (GPT-4o + Claude Sonnet 4)
- `baselines/` — Self-consistency baseline data
- `data/` — ProntoQA input problems and proof trees
- `src/` — Full source code
- `scripts/` — One-command reproduction scripts
- `tests/` — Test suite
- `checksums.sha256` — SHA256 integrity verification

See `supplementary/README.md` inside the ZIP for full details.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), consistent with the paper.
