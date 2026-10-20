"""vLLM helpers: starting an inference engine, syncing policy weights into it, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest.mock import patch

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from vllm import LLM, SamplingParams


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> "LLM":
    """Start vLLM on `device`, separate from the policy (starter code from the handout)."""
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    # Monkeypatch from TRL so vLLM can live on an arbitrary device in a single process.
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: "PreTrainedModel", llm: "LLM") -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def make_sampling_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    min_tokens: int = 0,
    n: int = 1,
    seed: int | None = None,
) -> "SamplingParams":
    """Sampling params used throughout: stop at </answer> and keep it in the output."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        n=n,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def generate(llm: "LLM", prompts: list[str], sampling_params: "SamplingParams") -> list[list[str]]:
    """Return, for each prompt, the list of its `n` sampled completions."""
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return [[o.text for o in out.outputs] for out in outputs]


def score_responses(
    reward_fn: Callable[[str, str], dict[str, float]],
    responses: list[str],
    ground_truths: list[str],
) -> list[dict[str, float]]:
    return [reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]


def summarize_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    n = max(len(scores), 1)
    fmt = [s["format_reward"] for s in scores]
    ans = [s["answer_reward"] for s in scores]
    return {
        "n": len(scores),
        "accuracy": sum(ans) / n,
        "format_rate": sum(fmt) / n,
        "reward": sum(s["reward"] for s in scores) / n,
        "count_format1_answer1": sum(1 for f, a in zip(fmt, ans) if f == 1 and a == 1),
        "count_format1_answer0": sum(1 for f, a in zip(fmt, ans) if f == 1 and a == 0),
        "count_format0_answer0": sum(1 for f, a in zip(fmt, ans) if f == 0 and a == 0),
        "count_format0_answer1": sum(1 for f, a in zip(fmt, ans) if f == 0 and a == 1),
    }


def evaluate_vllm(
    vllm_model: "LLM",
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: "SamplingParams",
    output_path: str | Path | None = None,
    extra_fields: list[dict] | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Generate one response per prompt, score it, and optionally serialize everything to jsonl."""
    responses = [outs[0] for outs in generate(vllm_model, prompts, eval_sampling_params)]
    scores = score_responses(reward_fn, responses, ground_truths)
    records = []
    for i, (p, r, gt, s) in enumerate(zip(prompts, responses, ground_truths, scores)):
        rec = {"prompt": p, "response": r, "ground_truth": gt, **s}
        if extra_fields is not None:
            rec.update(extra_fields[i])
        records.append(rec)
    metrics = summarize_scores(scores)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return metrics, records
