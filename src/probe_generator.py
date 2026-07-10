"""Step 3: Generate semantic (predicate substitution) and surface (rephrasing) probes."""

import json
import os
import re
import sys

# import config  # paths configured at runtime


# --- Predicate substitution dictionary ---
# Each real predicate maps to a unique substitute that doesn't appear in any problem.
PREDICATE_SUBSTITUTES = {
    # p001
    "wumpus": "krumpus", "tumpus": "glumpus", "dumpus": "blumpus", "zumpus": "frumpus",
    # p002
    "yumpus": "plumpus", "numpus": "stumpus", "rompus": "grimpus",
    "jompus": "crivpus", "vimpus": "drolpus",
    # p003
    "brimpus": "skelpus", "stelpus": "krendus", "grondus": "flindus",
    "flimpus": "brevpus", "quelpus": "strolpus",
    # p004
    "lorpus": "grilpus", "melpus": "trovpus", "quorpus": "blenpus", "timpus": "drevpus",
    # p005
    "plempus": "snivpus", "snorpus": "glenpus", "blukus": "trendus",
    "trelpus": "frovpus", "grivus": "klendus", "hompus": "strevpus",
    # p006
    "klempus": "brovpus", "frelpus": "glindus", "stompus": "krevpus",
    "vompus": "drenpus", "drimpus": "frolpus",
    # p007
    "snelpus": "trovdus", "trevus": "blinkpus", "blinkus": "grondpus", "grompus": "flendus",
    # p008
    "chompus": "strivpus", "drelpus": "klompus", "frevus": "brondus",
    "glimpus": "trevdus", "stelvus": "grimpous",
    # p009
    "prelvus": "klondus", "glompus": "frendus", "trelvus": "brovdus",
    "strivus": "glenpous", "krempus": "drolpous", "flonpus": "strevdus",
    # p010
    "brinpus": "klempous", "clempus": "frondus", "drevus": "glimpous",
    "frilpus": "strendus", "nelvus": "grolpus",
}


def _singularize(word: str) -> str:
    """Convert plural ProntoQA predicate to singular: 'wumpuses' -> 'wumpus'."""
    w = word.lower()
    if w.endswith("uses"):
        return w[:-2]  # "wumpuses" -> "wumpus"
    if w.endswith("es"):
        return w[:-2]  # fallback for -es
    if w.endswith("s") and not w.endswith("us"):
        return w[:-1]
    return w


def _pluralize(word: str) -> str:
    """Convert singular to plural: 'wumpus' -> 'wumpuses'."""
    w = word.lower()
    if w.endswith("us"):
        return w + "es"
    if w.endswith("s"):
        return w + "es"
    return w + "s"


def _case_like(replacement: str, original: str) -> str:
    """Return replacement with capitalization matching original token."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement.lower()


def _replace_predicate_whole_word(text: str, target_pred: str, substitute: str) -> str:
    """Replace singular/plural predicate forms exactly once as whole words.

    This avoids recursive substitutions such as:
        bimpuses -> zqbimpuses -> zqzqbimpuses
    caused by sequential str.replace calls.
    """
    target_pred = target_pred.lower()
    substitute = substitute.lower()
    forms = {
        target_pred: substitute,
        _pluralize(target_pred): _pluralize(substitute),
    }
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    def repl(match: re.Match) -> str:
        original = match.group(0)
        replacement = forms[original.lower()]
        return _case_like(replacement, original)

    return pattern.sub(repl, text)


def _find_predicates_in_premise(premise_text: str) -> list[str]:
    """Extract predicates from a premise text. Returns singular forms."""
    words = re.findall(r"\b[a-z]+(?:us|uses)\b", premise_text.lower())
    predicates = set()
    for w in words:
        predicates.add(_singularize(w))
    # Also check for entity names (capitalized words that aren't common)
    return list(predicates)


def _get_target_predicate(premise: dict, all_predicates_in_problem: set[str]) -> str | None:
    """Determine the key predicate introduced by this premise for substitution.
    
    For "Xs are Ys", the target is Y (the conclusion category).
    For "X is a Y", the target is Y.
    For "Xs are not Ys", the target is Y.
    """
    text = premise["text"]
    
    # "Xs are (not) Ys" — both words are plural, depluralize the object
    m = re.match(r"(\w+)\s+are\s+(?:not\s+)?(\w+)$", text, re.IGNORECASE)
    if m:
        return _singularize(m.group(2))
    
    # "X is a Y" — Y is singular, use as-is
    m = re.match(r"(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.IGNORECASE)
    if m:
        return m.group(2).lower()
    
    return None


def generate_semantic_probe(problem: dict, target_premise_id: str) -> dict | None:
    """Generate a semantic probe by substituting the target predicate in the given premise.
    
    The substitution is applied consistently across ALL premises that mention the predicate.
    """
    target_premise = None
    for p in problem["premises"]:
        if p["id"] == target_premise_id:
            target_premise = p
            break
    
    if target_premise is None:
        return None
    
    # Find target predicate in this premise
    target_pred = _get_target_predicate(target_premise, set())
    if target_pred is None:
        return None
    
    substitute = PREDICATE_SUBSTITUTES.get(target_pred)
    if substitute is None:
        # Dynamic substitute: prefix "zq" — guaranteed unique (no real predicate starts with zq)
        substitute = "zq" + target_pred
    
    # Apply substitution consistently across ALL premises
    modified_premises = []
    for p in problem["premises"]:
        new_text = _replace_predicate_whole_word(p["text"], target_pred, substitute)
        modified_premises.append({"id": p["id"], "text": new_text})
    
    return {
        "problem_id": problem["problem_id"],
        "probe_type": "semantic",
        "target_premise": target_premise_id,
        "substitution": {target_pred: substitute},
        "modified_premises": modified_premises,
    }


def generate_local_semantic_probe(problem: dict, target_premise_id: str) -> dict | None:
    """Generate a LOCAL semantic probe — substitute ONLY the target premise.

    Unlike consistent substitution (which replaces the predicate everywhere),
    local substitution breaks the chain and exposes cascade dependencies.
    E.g., "Alex is a wumpus" → "Alex is a krumpus" (but P1 stays "Wumpuses are tumpuses")
    → chain breaks → step depends on this premise.
    """
    target_premise = None
    for p in problem["premises"]:
        if p["id"] == target_premise_id:
            target_premise = p
            break

    if target_premise is None:
        return None

    # Find target predicate in this premise
    target_pred = _get_target_predicate(target_premise, set())
    if target_pred is None:
        return None

    substitute = PREDICATE_SUBSTITUTES.get(target_pred)
    if substitute is None:
        substitute = "zq" + target_pred

    # Apply substitution ONLY to the target premise
    modified_premises = []
    for p in problem["premises"]:
        if p["id"] == target_premise_id:
            new_text = _replace_predicate_whole_word(p["text"], target_pred, substitute)
            modified_premises.append({"id": p["id"], "text": new_text})
        else:
            modified_premises.append({"id": p["id"], "text": p["text"]})

    return {
        "problem_id": problem["problem_id"],
        "probe_type": "local_semantic",
        "target_premise": target_premise_id,
        "substitution": {target_pred: substitute},
        "modified_premises": modified_premises,
    }


def generate_surface_probe(problem: dict, target_premise_id: str) -> dict | None:
    """Generate a surface probe by rephrasing the target premise (meaning-preserving)."""
    target_premise = None
    for p in problem["premises"]:
        if p["id"] == target_premise_id:
            target_premise = p
            break
    
    if target_premise is None:
        return None
    
    text = target_premise["text"]
    rephrased = _surface_rephrase(text)
    
    # Only modify the target premise; others stay the same
    modified_premises = []
    for p in problem["premises"]:
        if p["id"] == target_premise_id:
            modified_premises.append({"id": p["id"], "text": rephrased})
        else:
            modified_premises.append({"id": p["id"], "text": p["text"]})
    
    return {
        "problem_id": problem["problem_id"],
        "probe_type": "surface",
        "target_premise": target_premise_id,
        "original_text": text,
        "rephrased_text": rephrased,
        "modified_premises": modified_premises,
    }


def _surface_rephrase(text: str) -> str:
    """Deterministic surface rephrasing rules. Meaning-preserving."""
    
    # Pattern: "Xs are not Ys" → "No X is a Y"
    m = re.match(r"(\w+?)(?:es|s)\s+are\s+not\s+(\w+?)(?:es|s)$", text, re.IGNORECASE)
    if m:
        subj = m.group(1).lower()
        obj = m.group(2).lower()
        return f"No {subj} is a {obj}"
    
    # Pattern: "Xs are Ys" → "Every X is a Y"
    m = re.match(r"(\w+?)(?:es|s)\s+are\s+(\w+?)(?:es|s)$", text, re.IGNORECASE)
    if m:
        subj = m.group(1).lower()
        obj = m.group(2).lower()
        return f"Every {subj} is a {obj}"
    
    # Pattern: "X is a Y" → "X is one of the Ys"
    m = re.match(r"(\w+)\s+is\s+a\s+(\w+)$", text, re.IGNORECASE)
    if m:
        subj = m.group(1)  # Keep case for proper nouns
        obj = m.group(2).lower()
        return f"{subj} is one of the {_pluralize(obj)}"
    
    # Fallback: no change (shouldn't happen for ProntoQA)
    return text


def generate_all_probes(problem: dict) -> dict:
    """Generate all probes for a problem.
    
    Returns dict mapping premise_id -> {"semantic": probe, "local_semantic": probe, "surface": probe}.
    """
    probes = {}
    for premise in problem["premises"]:
        pid = premise["id"]
        sem = generate_semantic_probe(problem, pid)
        loc = generate_local_semantic_probe(problem, pid)
        sur = generate_surface_probe(problem, pid)
        probes[pid] = {"semantic": sem, "local_semantic": loc, "surface": sur}
    return probes


def save_probes(problem_id: str, probes: dict, output_dir: str) -> None:
    """Save probes to disk under output_dir/<problem_id>/probes/."""
    probe_dir = os.path.join(output_dir, problem_id, "probes")
    os.makedirs(probe_dir, exist_ok=True)
    for premise_id, probe_pair in probes.items():
        for probe_type, filename_prefix in [
            ("semantic", "semantic"),
            ("local_semantic", "local"),
            ("surface", "surface"),
        ]:
            probe = probe_pair[probe_type]
            if probe is not None:
                path = os.path.join(probe_dir, f"{filename_prefix}_{premise_id}.json")
                with open(path, "w") as f:
                    json.dump(probe, f, indent=2)


if __name__ == "__main__":
    from prontoqa_loader import generate_problems
    
    problems = generate_problems()
    p = problems[0]
    print(f"Problem: {p['problem_id']}")
    probes = generate_all_probes(p)
    for pid, pair in probes.items():
        sem = pair["semantic"]
        sur = pair["surface"]
        print(f"\n  {pid}:")
        if sem:
            print(f"    Semantic: {sem['substitution']}")
            for mp in sem["modified_premises"]:
                print(f"      {mp['id']}: {mp['text']}")
        if sur:
            print(f"    Surface: '{sur['original_text']}' -> '{sur['rephrased_text']}'")
