"""Problem math_baseline (§3.2): zero-shot Qwen2.5-Math-1.5B on the MATH validation set.

Writes to runs/<run_name>/:
  generations.jsonl   every example with prompt, response, gold, and rewards
  summary.json        accuracy, format rate, and the three category counts
  examples.json       up to 20 examples from each category, for the writeup

Example:
  uv run python scripts/math_baseline.py --model models/Qwen2.5-Math-1.5B
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

from alignment.data import load_math
from alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from alignment.vllm_utils import evaluate_vllm, init_vllm, make_sampling_params

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = "models/Qwen2.5-Math-1.5B",
    prompt: str = typer.Option("r1_zero", help="r1_zero or question_only"),
    run_name: str = "",
    limit: int = typer.Option(None, help="Evaluate only the first N validation examples"),
    max_tokens: int = 1024,
    device: str = "cuda",
    gpu_memory_utilization: float = 0.85,
    seed: int = 0,
):
    run_dir = Path("runs") / (run_name or f"baseline_{prompt}")
    run_dir.mkdir(parents=True, exist_ok=True)

    examples = load_math("validation", prompt_name=prompt, limit=limit)
    reward_fn = r1_zero_reward_fn if prompt == "r1_zero" else question_only_reward_fn
    llm = init_vllm(model, device, seed, gpu_memory_utilization)
    params = make_sampling_params(max_tokens=max_tokens, seed=seed)

    metrics, records = evaluate_vllm(
        llm,
        reward_fn,
        [e["prompt"] for e in examples],
        [e["answer"] for e in examples],
        params,
        output_path=run_dir / "generations.jsonl",
        extra_fields=[{"level": e["level"], "type": e["type"]} for e in examples],
    )

    buckets = {
        "format1_answer1": [r for r in records if r["format_reward"] == 1 and r["answer_reward"] == 1],
        "format1_answer0": [r for r in records if r["format_reward"] == 1 and r["answer_reward"] == 0],
        "format0_answer0": [r for r in records if r["format_reward"] == 0 and r["answer_reward"] == 0],
    }
    rng = random.Random(seed)
    sampled = {k: rng.sample(v, min(20, len(v))) for k, v in buckets.items()}
    (run_dir / "examples.json").write_text(json.dumps(sampled, indent=2))
    (run_dir / "summary.json").write_text(json.dumps({"model": model, "prompt": prompt, **metrics}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
