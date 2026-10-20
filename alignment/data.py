"""Dataset loading and prompt formatting for MATH experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path

from alignment.drgrpo_grader import extract_answer

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MATH_DIR = REPO_ROOT / "data" / "MATH"
SFT_PATH = REPO_ROOT / "data" / "sft.jsonl"


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_prompt_template(name: str = "r1_zero") -> str:
    return (PROMPTS_DIR / f"{name}.prompt").read_text()


def format_prompt(question: str, template: str) -> str:
    return template.format(question=question)


def question_from_r1_zero_prompt(prompt: str) -> str:
    """Recover the raw question from an r1_zero-formatted prompt."""
    return prompt.split("User: ", 1)[1].rsplit("\nAssistant:", 1)[0].strip()


def load_math(split: str = "validation", prompt_name: str = "r1_zero", limit: int | None = None) -> list[dict]:
    """Load MATH examples as {"problem", "answer", "prompt", "level", "type"}.

    The data has no separate answer field, so the gold answer is the last \\boxed{} in the solution.
    """
    template = load_prompt_template(prompt_name)
    rows = []
    for ex in load_jsonl(MATH_DIR / f"{split}.jsonl"):
        answer = extract_answer(ex["solution"])
        if answer is None:
            continue
        rows.append(
            {
                "problem": ex["problem"],
                "answer": answer,
                "prompt": format_prompt(ex["problem"], template),
                "level": ex.get("level"),
                "type": ex.get("type"),
            }
        )
    return rows[:limit] if limit is not None else rows


def load_sft(
    path: str | Path = SFT_PATH,
    num_examples: int | None = None,
    seed: int = 0,
    exclude_questions: set[str] | None = None,
) -> list[dict]:
    """Load {"prompt", "response"} SFT examples, optionally dropping questions (e.g. ones that leak
    into the validation set) and subsampling `num_examples` unique examples."""
    rows = load_jsonl(path)
    if exclude_questions:
        rows = [r for r in rows if question_from_r1_zero_prompt(r["prompt"]) not in exclude_questions]
    if num_examples is not None and num_examples < len(rows):
        rows = random.Random(seed).sample(rows, num_examples)
    return rows


def validation_questions() -> set[str]:
    return {ex["problem"].strip() for ex in load_jsonl(MATH_DIR / "validation.jsonl")}
