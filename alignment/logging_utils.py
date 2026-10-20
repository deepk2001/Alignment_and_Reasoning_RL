"""In-the-loop generation logging (handout §4.2, `log_generations`) and metric logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from alignment.grpo_utils import masked_mean
from alignment.sft_utils import get_response_log_probs, tokenize_prompt_and_output
from alignment.vllm_utils import generate, score_responses

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from vllm import LLM, SamplingParams


@torch.no_grad()
def response_entropies(
    policy: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompts: list[str],
    responses: list[str],
    device: str,
    batch_size: int = 4,
) -> list[float]:
    """Mean per-token entropy of the policy over each response's tokens."""
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenize_prompt_and_output(prompts[i : i + batch_size], responses[i : i + batch_size], tokenizer)
        res = get_response_log_probs(
            policy, batch["input_ids"].to(device), batch["labels"].to(device), return_token_entropy=True
        )
        mask = batch["response_mask"].to(device)
        ent = masked_mean(res["token_entropy"], mask, dim=-1)
        out.extend(ent.float().cpu().tolist())
    return out


def log_generations(
    vllm_model: "LLM",
    policy: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: "SamplingParams",
    device: str,
    step: int,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Generate for `prompts`, then log per-example prompt / response / gold / rewards / entropy, and
    aggregate response-length statistics (overall, correct, incorrect)."""
    responses = [outs[0] for outs in generate(vllm_model, prompts, sampling_params)]
    scores = score_responses(reward_fn, responses, ground_truths)
    was_training = policy.training
    policy.eval()
    entropies = response_entropies(policy, tokenizer, prompts, responses, device)
    if was_training:
        policy.train()

    lengths = [len(tokenizer(r, add_special_tokens=False)["input_ids"]) for r in responses]
    correct = [s["answer_reward"] == 1 for s in scores]

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    records = [
        {
            "step": step,
            "prompt": p,
            "response": r,
            "ground_truth": gt,
            "format_reward": s["format_reward"],
            "answer_reward": s["answer_reward"],
            "reward": s["reward"],
            "mean_token_entropy": e,
            "response_length": n,
        }
        for p, r, gt, s, e, n in zip(prompts, responses, ground_truths, scores, entropies, lengths)
    ]
    summary = {
        "gen/reward": _avg([s["reward"] for s in scores]),
        "gen/format_reward": _avg([s["format_reward"] for s in scores]),
        "gen/answer_reward": _avg([s["answer_reward"] for s in scores]),
        "gen/mean_token_entropy": _avg(entropies),
        "gen/avg_response_length": _avg(lengths),
        "gen/avg_response_length_correct": _avg([n for n, c in zip(lengths, correct) if c]),
        "gen/avg_response_length_incorrect": _avg([n for n, c in zip(lengths, correct) if not c]),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return summary


class MetricsLogger:
    """Appends metric dicts to a local jsonl (used for plots) and mirrors them to wandb if enabled."""

    def __init__(self, path: str | Path, use_wandb: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

    def log(self, metrics: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self.use_wandb:
            import wandb

            wandb.log(metrics)


def truncate_jsonl(path: str | Path, keep: Callable[[dict], bool]) -> list[dict]:
    """Rewrite a jsonl log keeping only rows where keep(row) is true (drops rows logged after the last
    checkpoint when a run resumes). Returns the kept rows."""
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if keep(r)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def init_wandb(project: str | None, run_name: str, config: dict) -> bool:
    if not project:
        return False
    import wandb

    wandb.init(project=project, name=run_name, config=config)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return True
