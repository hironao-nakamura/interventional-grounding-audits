"""Programmatic ProntoQA problem generator for 50+ problems.

Generates problems with:
- Chain length: 3-5 steps
- Premise count: 4-6 (chain + optional distractors)
- True/False: half and half
- Each problem uses unique predicate names
- Proof trees with ground-truth dependencies
"""

import json
import os
import random
import sys

# import config  # paths configured at runtime

# --- Predicate name generation ---

# Consonant onsets for predicate construction
_ONSETS = [
    "b", "bl", "br", "ch", "cl", "cr", "d", "dr", "f", "fl", "fr",
    "g", "gl", "gr", "h", "j", "k", "kl", "kr", "l", "m", "n",
    "p", "pl", "pr", "qu", "r", "s", "sk", "sl", "sn", "sp", "st",
    "str", "t", "tr", "v", "w", "z", "sc", "sh", "sw", "tw", "thr",
]

# Medial vowel+consonant clusters
_MIDDLES = [
    "emp", "imp", "omp", "ump", "alp", "elp", "olp", "ulp",
    "arp", "erp", "orp", "urp", "anp", "enp", "onp", "unp",
    "eln", "oln", "iln", "alv", "elv", "olv", "ilv", "ulv",
    "ink", "onk", "unk", "ank", "enk",
    "ond", "ind", "und", "and", "end",
    "elt", "olt", "ilt", "ult",
]

# Entity names
_ENTITY_NAMES = [
    "Alex", "Fae", "Rex", "Stella", "Max", "Zara", "Pip", "Luna",
    "Sam", "Eve", "Kai", "Nova", "Leo", "Iris", "Ash", "Wren",
    "Juno", "Bolt", "Cleo", "Drake", "Finn", "Gale", "Haze", "Ivy",
    "Jade", "Knox", "Lark", "Milo", "Nyx", "Opal", "Quinn", "Rune",
    "Sage", "Tarn", "Uma", "Vale", "Xara", "Yuki", "Zion", "Bree",
    "Clay", "Dove", "Elm", "Fox", "Glen", "Hart", "Ink", "Jet",
    "Kit", "Lynx",
]


def _generate_unique_predicates(n: int, used: set[str]) -> list[str]:
    """Generate n unique ProntoQA-style predicate names ending in -us."""
    predicates = []
    attempts = 0
    for onset in _ONSETS:
        for middle in _MIDDLES:
            name = onset + middle + "us"
            if name not in used and len(name) >= 5:
                predicates.append(name)
                used.add(name)
                if len(predicates) >= n:
                    return predicates
            attempts += 1
    # Fallback with numbered predicates
    while len(predicates) < n:
        name = f"pred{len(predicates):03d}us"
        if name not in used:
            predicates.append(name)
            used.add(name)
    return predicates


def _pluralize(pred: str) -> str:
    """Pluralize a ProntoQA predicate: 'wumpus' -> 'Wumpuses'."""
    return pred.capitalize() + "es"


def generate_problem(
    problem_id: str,
    chain_length: int,
    n_distractors: int,
    answer: bool,
    entity_name: str,
    predicates: list[str],
) -> dict:
    """Generate a single ProntoQA problem.

    Args:
        problem_id: e.g. "p001"
        chain_length: number of reasoning steps (3-5)
        n_distractors: number of extra unused premises (0-2)
        answer: True or False
        entity_name: e.g. "Alex"
        predicates: list of unique predicate names; needs at least
                    chain_length+1 (for chain) + n_distractors + (1 for False negation target)
    """
    # Chain predicates: entity_type → pred[0] → pred[1] → ... → pred[chain_length-1]
    # For True: question asks about pred[chain_length-1]
    # For False: we insert a "not" at the end of the chain

    entity_type = predicates[0]  # The entity IS this type
    chain_preds = predicates[1:chain_length + 1]  # Chain of subtypes

    premises = []
    proof_tree = []

    # Entity premise (always last in premise list, as per ProntoQA convention)
    entity_premise_id = f"P{chain_length + 1 + n_distractors}"

    # Build chain premises
    prev = entity_type
    for i, next_pred in enumerate(chain_preds):
        pid = f"P{i + 1}"
        if answer or i < chain_length - 1:
            # Normal subtype: "Xs are Ys"
            premises.append({"id": pid, "text": f"{_pluralize(prev)} are {prev}es".replace(f"{prev}es", f"{next_pred}es") if False else f"{_pluralize(prev)} are {_pluralize(next_pred).lower()}"})
        else:
            # Last step for False: "Xs are not Ys"
            premises.append({"id": pid, "text": f"{_pluralize(prev)} are not {_pluralize(next_pred).lower()}"})
        prev = next_pred

    # Distractor premises
    distractor_preds = predicates[chain_length + 1: chain_length + 1 + n_distractors * 2]
    for i in range(0, len(distractor_preds) - 1, 2):
        pid = f"P{chain_length + 1 + i // 2}"
        premises.append({"id": pid, "text": f"{_pluralize(distractor_preds[i])} are {_pluralize(distractor_preds[i+1]).lower()}"})

    # Entity premise
    premises.append({"id": entity_premise_id, "text": f"{entity_name} is a {entity_type}"})

    # Question
    if answer:
        question_pred = chain_preds[-1]
        question = f"Is {entity_name} a {question_pred}?"
    else:
        question_pred = chain_preds[-1]
        question = f"Is {entity_name} a {question_pred}?"

    # Build proof tree
    for step_idx in range(chain_length):
        step_num = step_idx + 1
        chain_premise_id = f"P{step_idx + 1}"

        if step_idx == 0:
            depends = [entity_premise_id, chain_premise_id]
        else:
            depends = [f"S{step_idx}", chain_premise_id]

        if answer or step_idx < chain_length - 1:
            conclusion = f"is({entity_name.lower()}, {chain_preds[step_idx]})"
        else:
            conclusion = f"not_is({entity_name.lower()}, {chain_preds[step_idx]})"

        proof_tree.append({
            "step": step_num,
            "conclusion": conclusion,
            "depends_on": depends,
        })

    return {
        "problem_id": problem_id,
        "premises": premises,
        "question": question,
        "answer": answer,
        "proof_tree": proof_tree,
    }


def generate_problem_set(n_problems: int = 50, seed: int = 42) -> list[dict]:
    """Generate a balanced set of ProntoQA problems.

    Args:
        n_problems: total number of problems (half True, half False)
        seed: random seed for reproducibility
    """
    rng = random.Random(seed)
    used_predicates: set[str] = set()
    problems = []

    n_true = n_problems // 2
    n_false = n_problems - n_true

    # Generate answer labels
    answers = [True] * n_true + [False] * n_false
    rng.shuffle(answers)

    for i, answer in enumerate(answers):
        pid = f"p{i + 1:03d}"

        # Vary chain length and distractor count
        chain_length = rng.choice([3, 3, 4, 4, 5])  # bias toward 3-4
        n_distractors = rng.choice([0, 0, 1, 1, 2])  # bias toward 0-1

        # How many predicates do we need?
        n_preds_needed = 1 + chain_length + n_distractors * 2 + 2  # +2 for safety margin
        predicates = _generate_unique_predicates(n_preds_needed, used_predicates)

        entity_name = _ENTITY_NAMES[i % len(_ENTITY_NAMES)]

        problem = generate_problem(
            problem_id=pid,
            chain_length=chain_length,
            n_distractors=n_distractors,
            answer=answer,
            entity_name=entity_name,
            predicates=predicates,
        )
        problems.append(problem)

    return problems


def save_problems(problems: list[dict]) -> None:
    """Save problems to individual JSON files."""
    # Clear existing problems
    for f in os.listdir(config.PROBLEMS_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(config.PROBLEMS_DIR, f))

    for p in problems:
        pid = p["problem_id"]
        path = os.path.join(config.PROBLEMS_DIR, f"{pid}.json")
        with open(path, "w") as f:
            json.dump(p, f, indent=2)
    print(f"Saved {len(problems)} problems to {config.PROBLEMS_DIR}")


if __name__ == "__main__":
    problems = generate_problem_set(50)
    # Stats
    n_true = sum(1 for p in problems if p["answer"])
    n_false = sum(1 for p in problems if not p["answer"])
    chain_lengths = [len(p["proof_tree"]) for p in problems]
    premise_counts = [len(p["premises"]) for p in problems]

    print(f"Generated {len(problems)} problems")
    print(f"  True: {n_true}, False: {n_false}")
    print(f"  Chain lengths: min={min(chain_lengths)}, max={max(chain_lengths)}, avg={sum(chain_lengths)/len(chain_lengths):.1f}")
    print(f"  Premise counts: min={min(premise_counts)}, max={max(premise_counts)}, avg={sum(premise_counts)/len(premise_counts):.1f}")

    # Verify a few
    for p in problems[:3]:
        print(f"\n  {p['problem_id']} (answer={p['answer']}):")
        for pr in p["premises"]:
            print(f"    {pr['id']}: {pr['text']}")
        print(f"    Q: {p['question']}")
        for s in p["proof_tree"]:
            print(f"    S{s['step']}: {s['conclusion']} <- {s['depends_on']}")
