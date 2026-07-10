"""Meta-reasoning detector for Claude-style probe responses.

When Claude encounters inconsistent premises from our probes,
it often explicitly detects and reports the inconsistency instead
of following the broken chain. Examples:

  "the chain breaks here because zqtumpuses are not mentioned..."
  "premise 2 only tells us about zqbimpuses, so this doesn't apply..."
  "we cannot conclude anything about Alex being a dumpus..."

These responses are FUNCTIONALLY GROUNDED: the premise change
caused the model's output to change. But the normalizer cannot
parse them into is(X, Y) form.

This detector identifies such cases and marks them GROUNDED_META.
"""

import re


# Patterns indicating the model detected the probe's inconsistency.
# Ordered from most specific (low FP risk) to least specific.
CHAIN_BREAK_PATTERNS = [
    # Explicit chain break language
    r"(?:the\s+)?chain\s+(?:is\s+)?break|(?:the\s+)?chain\s+(?:is\s+)?broken",
    r"(?:the\s+)?(?:reasoning|logical)\s+(?:chain|path)\s+(?:break|stop|end|fail)",
    r"cannot\s+(?:continue|complete|finish)\s+(?:the\s+)?(?:chain|reasoning)",

    # Premise doesn't apply / doesn't match
    r"(?:premise|rule|statement)\s+\d*\s*(?:doesn't|does not|no longer)\s+(?:apply|match|hold)",
    r"(?:this|that|the)\s+(?:premise|rule)\s+(?:doesn't|does not)\s+(?:apply|hold|work|help)",
    r"(?:doesn't|does not)\s+(?:directly\s+)?(?:apply|match)\s+(?:to|here|anymore)",

    # Cannot conclude / derive / establish
    r"(?:can't|cannot|unable to)\s+(?:conclude|derive|infer|determine|establish)\s+(?:that|whether|if|anything)",
    r"(?:can't|cannot|unable to)\s+(?:reach|arrive at|get to)\s+(?:a\s+)?(?:conclusion|answer)",
    r"(?:no|not)\s+(?:way|basis|ground|evidence)\s+to\s+(?:conclude|infer|derive)",
    r"(?:I|we)\s+cannot\s+(?:conclude|establish|determine)",

    # Explicit mismatch / inconsistency detection
    r"(?:there\s+(?:is|appears to be)\s+(?:a\s+)?)?mismatch",
    r"(?:premises?\s+(?:are|is)\s+)?(?:inconsistent|contradictory|conflicting)",

    # "Nothing tells us about X" / "No information about X"
    r"(?:nothing|no\s+(?:premise|rule|information|statement))\s+(?:tells?\s+us|says?|mentions?|connects?)",
    r"(?:we\s+)?(?:don't|do not)\s+(?:know|have)\s+(?:anything|information|a\s+rule)\s+about",

    # Substituted predicate explicitly noted as unknown/not mentioned
    r"zq\w+\S*\s+(?:is|are)\s+not\s+(?:mentioned|defined|covered|addressed)",
    r"(?:no|not\s+any)\s+(?:rule|premise|statement)\s+(?:about|for|mentioning|involving)\s+zq\w+",
    r"only\s+(?:tells?\s+us|applies?|works?)\s+(?:about|for|with)\s+(?:\"?zq)",

    # "which is different from" / "not the same as"
    r"which\s+is\s+(?:different|distinct)\s+from",
    r"(?:is|are)\s+(?:different|distinct)\s+from\s+\"?zq",
]


def detect_meta_reasoning(raw_response: str) -> dict:
    """Detect if a probe response contains meta-reasoning about chain breaks.

    Returns:
        {
            "detected": bool,
            "confidence": float,  # 0.0-1.0
            "matched_patterns": list[str],
            "evidence": str,
        }
    """
    if not raw_response:
        return {"detected": False, "confidence": 0.0,
                "matched_patterns": [], "evidence": ""}

    text = raw_response.lower()

    matched = []
    evidence_snippets = []

    for pattern in CHAIN_BREAK_PATTERNS:
        m = re.search(pattern, text)
        if m:
            matched.append(pattern)
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            evidence_snippets.append(text[start:end].strip())

    if not matched:
        return {"detected": False, "confidence": 0.0,
                "matched_patterns": [], "evidence": ""}

    # Confidence based on match count
    if len(matched) >= 3:
        confidence = 0.95
    elif len(matched) >= 2:
        confidence = 0.90
    else:
        confidence = 0.70

    return {
        "detected": True,
        "confidence": confidence,
        "matched_patterns": matched[:5],
        "evidence": evidence_snippets[0] if evidence_snippets else "",
    }


def reclassify_certificate(certificate: dict,
                           raw_semantic_response: str,
                           raw_local_response: str = "") -> dict:
    """Given an UNPARSEABLE certificate and raw probe responses,
    check if it should be reclassified as GROUNDED_META.

    Only reclassifies if:
    1. Current verdict is UNPARSEABLE
    2. Original was parseable (we need a valid baseline)
    3. Meta-reasoning detected in semantic or local probe response

    Returns updated certificate (copy).
    """
    cert = dict(certificate)

    if cert.get("verdict") != "UNPARSEABLE":
        return cert

    if not cert.get("parse_ok_orig", False):
        return cert

    # Check semantic probe response
    sem_det = detect_meta_reasoning(raw_semantic_response) if raw_semantic_response else \
        {"detected": False, "confidence": 0.0, "matched_patterns": [], "evidence": ""}

    # Check local probe response
    loc_det = detect_meta_reasoning(raw_local_response) if raw_local_response else \
        {"detected": False, "confidence": 0.0, "matched_patterns": [], "evidence": ""}

    # Use the detection with highest confidence
    best = sem_det if sem_det["confidence"] >= loc_det["confidence"] else loc_det
    source = "semantic" if sem_det["confidence"] >= loc_det["confidence"] else "local"

    if best["detected"] and best["confidence"] >= 0.7:
        cert["verdict"] = "GROUNDED_META"
        cert["decision_rule"] = "R_GROUNDED_META_REASONING"
        cert["meta_reasoning"] = {
            "confidence": best["confidence"],
            "matched_patterns": best["matched_patterns"],
            "evidence": best["evidence"],
            "probe_source": source,
            "original_verdict": "UNPARSEABLE",
        }

    return cert
