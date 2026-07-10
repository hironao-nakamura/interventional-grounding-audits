"""Step 5: Normalize CoT step text to canonical logical form (deterministic)."""

import re


def _depluralize(word: str) -> str:
    """Convert plural ProntoQA predicate to singular.
    
    'wumpuses' -> 'wumpus', 'tumpuses' -> 'tumpus',
    'cats' -> 'cat', but 'wumpus' -> 'wumpus' (already singular).
    """
    w = word.lower()
    # ProntoQA pattern: ends in "-uses" → strip "es" → "-us"
    if w.endswith("uses") and len(w) > 4:
        return w[:-2]
    # General "-es" plurals (but not "-us")
    if w.endswith("es") and not w.endswith("us") and len(w) > 2:
        return w[:-2]
    # General "-s" plurals (but not if already ends in "-us")
    if w.endswith("s") and not w.endswith("us") and len(w) > 1:
        return w[:-1]
    return w


def _clean_predicate(word: str) -> str:
    """Clean ProntoQA predicate: strip trailing 'e' from model misspellings.
    
    GPT-4o sometimes outputs 'stelpuse' instead of 'stelpus', 'trevuse' instead of 'trevus'.
    This strips a trailing 'e' when the result would end in '-us' (ProntoQA convention).
    """
    w = word.lower()
    # If the word ends in '-use' and stripping 'e' gives a valid '-us' ending
    if w.endswith("use") and len(w) > 3:
        # Don't strip from words like "use", "abuse" etc. — only ProntoQA-like words
        # ProntoQA predicates are nonsense words, so this is safe
        return w[:-1]
    if w.endswith("e") and w[:-1].endswith("us"):
        return w[:-1]
    return w


def normalize(step_text: str) -> tuple[str | None, str]:
    """Parse free-text CoT step into canonical form.
    
    Returns (normalized_form, parse_status).
    parse_status is "OK" or "UNPARSEABLE".
    """
    text = step_text.strip().rstrip(".")
    
    # --- Strip Claude-style parenthetical annotations ---
    # Claude often appends "(given in premise N)" or "(from premise N)" etc.
    text = re.sub(r"\s*\((?:given\s+in|from|by|via|per|using|see)\s+premise\s*\d*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(premise\s+\d+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(given\)", "", text, flags=re.IGNORECASE)
    text = text.strip().rstrip(".")
    
    # --- Extract final conclusion from compound sentences ---
    # GPT-4o generates several patterns:
    #   (a) "Since X and Y, Z"            → conclusion Z (after last comma)
    #   (b) "X is a Y, and Z, so W"       → conclusion W (after "so")
    #   (c) "X, and since Z, W"           → conclusion W (after last comma)
    #   (d) "Therefore, X is a Y"         → conclusion X is a Y
    
    # Strategy 1: "... so CONCLUSION" (highest priority — unambiguous)
    m = re.search(r",\s*so\s+(.+)$", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if re.match(r"\w+\s+(?:is|are)\s+", candidate, re.IGNORECASE):
            text = candidate
    else:
        # Strategy 2: "Since/Because ... , CONCLUSION"
        m = re.match(r"^(?:Since|Because|As|Given that)\s+.+,\s*(.+)$", text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if re.match(r"\w+\s+(?:is|are)\s+", candidate, re.IGNORECASE):
                text = candidate
        else:
            # Strategy 3: "X is/are ..., and since/because ..., CONCLUSION"
            m = re.search(r",\s*and\s+(?:since|because)\s+.+,\s*(.+)$", text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if re.match(r"\w+\s+(?:is|are)\s+", candidate, re.IGNORECASE):
                    text = candidate
            else:
                # Strategy 4: Generic — last clause after final comma if it looks like a conclusion
                parts = text.rsplit(",", 1)
                if len(parts) == 2:
                    candidate = parts[1].strip()
                    # Remove leading "and" if present
                    candidate = re.sub(r"^and\s+", "", candidate, flags=re.IGNORECASE)
                    if re.match(r"\w+\s+(?:is|are)\s+", candidate, re.IGNORECASE):
                        text = candidate
    
    # Remove simpler prefixes
    text = re.sub(r"^(?:Therefore|Thus|So|Hence|Consequently),?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:We know that|We can conclude that|It follows that|This means that|This tells us that)\s+",
                  "", text, flags=re.IGNORECASE)
    # "Based on this/the above/premise N, ..." / "Given that ..., "
    text = re.sub(r"^(?:Based on|Given)\s+(?:this|that|the above|the premises?|premise \d+|these facts?),?\s*",
                  "", text, flags=re.IGNORECASE)
    # "From the premises, ..." / "According to premise N, ..."
    text = re.sub(r"^(?:From|According to)\s+(?:the )?(?:premises?|the facts?|premise \d+),?\s*",
                  "", text, flags=re.IGNORECASE)
    text = text.strip()

    # --- Handle appositive constructions ---
    # "Rex, being a brimpus, is a stelpus" → "Rex is a stelpus"
    # "Stella, being a lorpus, is also a melpus" → "Stella is a melpus"
    m = re.match(r"(\w+),\s+being\s+(?:a|an)\s+\w+,\s+is\s+(?:also\s+)?(.+)$", text, re.IGNORECASE)
    if m:
        text = f"{m.group(1)} is {m.group(2)}"

    # --- Handle relative clause constructions ---
    # "Alex, who is a wumpus, is a tumpus" → "Alex is a tumpus"
    m = re.match(r"(\w+),\s+who\s+is\s+(?:a|an)\s+\w+,\s+is\s+(?:also\s+)?(.+)$", text, re.IGNORECASE)
    if m:
        text = f"{m.group(1)} is {m.group(2)}"

    # --- Handle "X must be / would be / has to be a Y" → "X is a Y" ---
    text = re.sub(r"\b(must|would|has to|needs to)\s+be\b", "is", text, flags=re.IGNORECASE)

    # Helper: apply _clean_predicate to the output of normalization
    def _norm(subj: str, obj: str, template: str) -> tuple[str, str]:
        return template.format(_clean_predicate(subj.lower()), _clean_predicate(obj.lower())), "OK"

    # --- Pattern: "X is not a/an Y" → not_is(x, y) ---
    m = re.match(r"(\w+)\s+is\s+not\s+(?:a|an)\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(m[1], m[2], "not_is({}, {})")

    # --- Pattern: "X is a/an Y" → is(x, y) ---
    m = re.match(r"(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(m[1], m[2], "is({}, {})")

    # --- Pattern: "X is one of the Ys" → is(x, y) ---
    m = re.match(r"(\w+)\s+is\s+one\s+of\s+the\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(m[1], _depluralize(m[2]), "is({}, {})")

    # --- Pattern: "X is a kind of Y(s)" → is(x, y) ---
    m = re.match(r"(\w+)\s+is\s+a\s+kind\s+of\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(m[1], _depluralize(m[2]), "is({}, {})")

    # --- Pattern: "X is also a/an Y" → is(x, y) ---
    m = re.match(r"(\w+)\s+is\s+also\s+(?:a|an)\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(m[1], m[2], "is({}, {})")

    # --- Pattern: "No X is a Y" → not_subtype(x, y) ---
    m = re.match(r"No\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(m[1], m[2], "not_subtype({}, {})")

    # --- Pattern: "Xs are not Ys" → not_subtype(x, y) ---
    m = re.match(r"(\w+)\s+are\s+not\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(_depluralize(m[1]), _depluralize(m[2]), "not_subtype({}, {})")

    # --- Pattern: "All/Every/Each X is a Y" → subtype(x, y) ---
    m = re.match(r"(?:All|Every|Each)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(_depluralize(m[1]), m[2].lower(), "subtype({}, {})")

    # --- Pattern: "All Xs are Ys" → subtype(x, y) ---
    m = re.match(r"(?:All|Every|Each)\s+(\w+)\s+are\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(_depluralize(m[1]), _depluralize(m[2]), "subtype({}, {})")

    # --- Pattern: "Xs are Ys" (bare) → subtype(x, y) ---
    m = re.match(r"(\w+)\s+are\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return _norm(_depluralize(m[1]), _depluralize(m[2]), "subtype({}, {})")

    return None, "UNPARSEABLE"


def parse_cot_steps(raw_response: str) -> list[dict]:
    """Parse raw LLM response into list of NormalizedStep dicts.
    
    Expects format: "Step N: [conclusion]."
    """
    steps = []
    # Match "Step N: ..." patterns (capture until next Step or Answer or end)
    pattern = re.compile(r"Step\s+(\d+)\s*:\s*(.+?)(?:\.\s*$|\.\s*(?=Step|\n|Answer))", re.MULTILINE)
    for m in pattern.finditer(raw_response):
        step_id = int(m.group(1))
        raw_text = m.group(2).strip()
        norm, status = normalize(raw_text)
        steps.append({
            "step_id": step_id,
            "raw_text": raw_text,
            "normalized": norm,
            "parse_status": status,
        })
    
    # Fallback: try simpler pattern if nothing matched
    if not steps:
        pattern2 = re.compile(r"Step\s+(\d+)\s*:\s*(.+?)$", re.MULTILINE)
        for m in pattern2.finditer(raw_response):
            step_id = int(m.group(1))
            raw_text = m.group(2).strip().rstrip(".")
            norm, status = normalize(raw_text)
            steps.append({
                "step_id": step_id,
                "raw_text": raw_text,
                "normalized": norm,
                "parse_status": status,
            })
    
    return steps


if __name__ == "__main__":
    # Quick unit tests
    tests = [
        ("Alex is a tumpus", "is(alex, tumpus)"),
        ("Alex is a wumpus", "is(alex, wumpus)"),
        ("Alex is one of the tumpuses", "is(alex, tumpus)"),
        ("Alex is not a zumpus", "not_is(alex, zumpus)"),
        ("Wumpuses are tumpuses", "subtype(wumpus, tumpus)"),
        ("Every wumpus is a tumpus", "subtype(wumpus, tumpus)"),
        ("Each wumpus is a tumpus", "subtype(wumpus, tumpus)"),
        ("All wumpuses are tumpuses", "subtype(wumpus, tumpus)"),
        ("Stompuses are not vompuses", "not_subtype(stompus, vompus)"),
        ("Therefore, Alex is a dumpus", "is(alex, dumpus)"),
        ("No stompus is a vompus", "not_subtype(stompus, vompus)"),
        ("Alex is a kind of wumpus", "is(alex, wumpus)"),
    ]
    print("=== Normalizer Unit Tests ===")
    passed = 0
    for text, expected in tests:
        norm, status = normalize(text)
        ok = (norm == expected)
        if not ok:
            print(f"  FAIL: '{text}' -> '{norm}' (expected '{expected}')")
        else:
            passed += 1
    print(f"  {passed}/{len(tests)} passed")
