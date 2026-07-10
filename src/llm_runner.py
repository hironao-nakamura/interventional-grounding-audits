"""Step 2 & 4: Run LLM on original and probed premises, save outputs.

Supports OpenAI (GPT-4o) and Anthropic (Claude Sonnet 4.5).
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 1024

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def build_prompt(premises: list[dict], question: str) -> str:
    """Build the CoT prompt from premises and question."""
    premise_lines = []
    for i, p in enumerate(premises, 1):
        premise_lines.append(f"{i}. {p['text']}")
    premises_text = "\n".join(premise_lines)

    return f"""Given the following premises:
{premises_text}

Question: {question}

Solve step by step. For each step, write EXACTLY in this format:
Step 1: [Your conclusion in one sentence].
Step 2: [Your conclusion in one sentence].
...

After all steps, write exactly:
Answer: True
or
Answer: False"""


def run_llm(
    prompt: str,
    api_key: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> dict:
    """Call the LLM and return the result with metadata.

    Automatically routes to OpenAI or Anthropic based on model name.
    Retries on transient errors (rate limit, timeout, server errors).
    """
    model = model or DEFAULT_OPENAI_MODEL
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature

    last_error = None
    for attempt in range(max_retries):
        try:
            if model.startswith("claude"):
                return _run_anthropic(prompt, api_key, model, temperature, max_tokens)
            else:
                return _run_openai(prompt, api_key, model, temperature, max_tokens)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # Retry on rate limit, timeout, server errors
            is_retryable = any(kw in err_str for kw in [
                "rate_limit", "rate limit", "timeout", "server_error",
                "500", "502", "503", "529", "overloaded",
            ])
            if is_retryable and attempt < max_retries - 1:
                wait = backoff * (2 ** attempt)
                time.sleep(wait)
                continue
            raise  # Non-retryable or final attempt

    raise last_error  # Should not reach here


def _run_openai(prompt: str, api_key: str, model: str, temperature: float,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Call OpenAI API."""
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_response = response.choices[0].message.content
    sha256 = hashlib.sha256(raw_response.encode()).hexdigest()

    return {
        "raw_response": raw_response,
        "model": model,
        "temperature": temperature,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def _run_anthropic(prompt: str, api_key: str, model: str, temperature: float,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Call Anthropic API."""
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_response = response.content[0].text
    sha256 = hashlib.sha256(raw_response.encode()).hexdigest()

    return {
        "raw_response": raw_response,
        "model": model,
        "temperature": temperature,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def run_original(problem: dict, api_key: str, model: str = None) -> dict:
    """Run LLM on original premises. Returns result dict."""
    prompt = build_prompt(problem["premises"], problem["question"])
    result = run_llm(prompt, api_key, model=model)
    result["problem_id"] = problem["problem_id"]
    result["probe_type"] = "original"
    result["prompt"] = prompt
    result["premises"] = problem["premises"]
    result["question"] = problem["question"]
    return result


def run_probed(problem: dict, probe: dict, api_key: str, model: str = None) -> dict:
    """Run LLM on probed premises. Returns result dict."""
    prompt = build_prompt(probe["modified_premises"], problem["question"])
    result = run_llm(prompt, api_key, model=model)
    result["problem_id"] = problem["problem_id"]
    result["probe_type"] = probe["probe_type"]
    result["target_premise"] = probe["target_premise"]
    result["prompt"] = prompt
    result["modified_premises"] = probe["modified_premises"]
    result["substitution"] = probe.get("substitution")
    result["original_text"] = probe.get("original_text")
    result["rephrased_text"] = probe.get("rephrased_text")
    return result


def save_llm_result(result: dict, filepath: str) -> None:
    """Save LLM result to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
