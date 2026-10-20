"""Problem math_baseline (b): diagnose why zero-shot generations get format or answer reward 0.

Reads runs/<run>/generations.jsonl (from math_baseline.py) and writes runs/<run>/analysis.json with
failure categories, counts, and example indices. CPU only.

Format-0 categories (r1_zero_reward_fn needs the literal "</think> <answer>" and "</answer>"):
  whitespace_variant   tags present but not separated by exactly one space (parser strictness)
  no_think_close       never closes </think>
  no_answer_tag        closes </think> but never opens <answer>
  answer_not_closed    opens <answer> but hits the token limit or stops before </answer>
  other

Format-1 / answer-0 cases are regraded with the slow grader (fast=False, which adds math-verify /
sympy equivalence checks): if that marks them correct, the fast grader gave a false negative.

  uv run python scripts/analyze_baseline.py --run baseline_r1_zero
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import typer

from alignment.drgrpo_grader import extract_answer, grade

app = typer.Typer(add_completion=False)


def format_failure(response: str) -> str:
    if re.search(r"</think>\s*<answer>", response) and "</answer>" in response:
        return "whitespace_variant"
    if "</think>" not in response:
        return "no_think_close"
    if "<answer>" not in response:
        return "no_answer_tag"
    if "</answer>" not in response:
        return "answer_not_closed"
    return "other"


def lenient_answer(response: str) -> str | None:
    """Best-effort answer extraction ignoring the tag format: <answer> contents, else last \\boxed{}."""
    if "<answer>" in response:
        ans = response.split("<answer>")[-1].replace("</answer>", "").strip()
        return extract_answer(ans) if "\\boxed" in ans else ans
    if "\\boxed" in response:
        return extract_answer(response)
    return None


def answer_failure(response: str, gold: str) -> str:
    ans = response.split("<answer>")[-1].replace("</answer>", "").strip()
    if "\\boxed" in ans:
        ans = extract_answer(ans) or ""
    if not ans:
        return "empty_answer"
    if grade(ans, gold, fast=False):
        return "grader_false_negative"
    if len(re.findall(r"[A-Za-z]{4,}", ans)) >= 3:
        return "prose_in_answer"
    return "wrong_answer"


@app.command()
def main(run: str = "baseline_r1_zero", n_examples: int = 12):
    run_dir = Path("runs") / run
    rows = [json.loads(l) for l in (run_dir / "generations.jsonl").read_text().splitlines() if l.strip()]

    fmt0 = [(i, r) for i, r in enumerate(rows) if r["format_reward"] == 0]
    fmt1_ans0 = [(i, r) for i, r in enumerate(rows) if r["format_reward"] == 1 and r["answer_reward"] == 0]

    fmt0_cats = {i: format_failure(r["response"]) for i, r in fmt0}
    fmt0_lenient_correct = [
        i for i, r in fmt0 if (a := lenient_answer(r["response"])) is not None and grade(a, r["ground_truth"], fast=False)
    ]
    ans0_cats = {i: answer_failure(r["response"], r["ground_truth"]) for i, r in fmt1_ans0}

    def examples(cats: dict[int, str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, c in cats.items():
            out.setdefault(c, [])
            if len(out[c]) < n_examples:
                out[c].append(i)
        return out

    lengths = [len(r["response"]) for r in rows]
    analysis = {
        "n": len(rows),
        "format0": {
            "count": len(fmt0),
            "categories": dict(Counter(fmt0_cats.values()).most_common()),
            "would_be_correct_with_lenient_parsing": len(fmt0_lenient_correct),
            "examples": examples(fmt0_cats),
        },
        "format1_answer0": {
            "count": len(fmt1_ans0),
            "categories": dict(Counter(ans0_cats.values()).most_common()),
            "examples": examples(ans0_cats),
        },
        "avg_response_chars": sum(lengths) / max(len(lengths), 1),
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps({k: v for k, v in analysis.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    app()
