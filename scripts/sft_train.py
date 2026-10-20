"""Problem sft_experiment (§4.3): SFT Qwen2.5-Math-1.5B on R1 reasoning traces.

The policy trains on --policy-device; vLLM sits on --vllm-device and receives the policy weights before
each validation pass. Outputs go to runs/<run_name>/ (metrics.jsonl, generations.jsonl, summary.json).

Examples:
  # dataset-size sweep
  for n in 128 256 512; do uv run python scripts/sft_train.py --num-examples $n --run-name sft_n$n; done
  uv run python scripts/sft_train.py --run-name sft_full
  # filtered data (run scripts/filter_sft.py first)
  uv run python scripts/sft_train.py --data data/sft_filtered.jsonl --run-name sft_filtered
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import typer

from alignment.data import SFT_PATH, load_math, load_sft, validation_questions
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.logging_utils import MetricsLogger, init_wandb, log_generations
from alignment.training import load_policy, make_optimizer, save_policy, sft_train
from alignment.vllm_utils import evaluate_vllm, init_vllm, load_policy_into_vllm_instance, make_sampling_params

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = "models/Qwen2.5-Math-1.5B",
    data: str = str(SFT_PATH),
    num_examples: int = typer.Option(None, help="Number of unique SFT examples (default: all)"),
    run_name: str = "sft_full",
    lr: float = 2e-5,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_epochs: float = 3.0,
    min_steps: int = typer.Option(30, help="Train at least this many optimizer steps (helps tiny datasets)"),
    max_seq_len: int = typer.Option(4096, help="Truncate prompt+response to this many tokens"),
    loss_normalization: str = "token_mean",
    gradient_checkpointing: bool = True,
    eval_every: int = 20,
    n_eval: int = typer.Option(1500, help="Validation examples per eval (1500 = all)"),
    eval_max_tokens: int = 1024,
    n_log_generations: int = 32,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:0",
    gpu_memory_utilization: float = typer.Option(0.3, help="vLLM share of the GPU it shares with the policy"),
    seed: int = 0,
    wandb_project: str = typer.Option(None, help="Set to enable wandb logging"),
    save: bool = False,
):
    config = dict(locals())
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    for f in ("metrics.jsonl", "generations.jsonl"):
        (run_dir / f).unlink(missing_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)

    examples = load_sft(data, num_examples=num_examples, seed=seed, exclude_questions=validation_questions())
    val = load_math("validation")[:n_eval]
    log_subset = random.Random(seed).sample(val, min(n_log_generations, len(val)))

    use_wandb = init_wandb(wandb_project, run_name, config)
    logger = MetricsLogger(run_dir / "metrics.jsonl", use_wandb)
    # Start vLLM before the policy: on a shared GPU it sizes its KV cache from the memory free at startup.
    llm = init_vllm(model, vllm_device, seed, gpu_memory_utilization)
    policy, tokenizer = load_policy(model, policy_device, gradient_checkpointing)
    optimizer = make_optimizer(policy, lr)
    eval_params = make_sampling_params(max_tokens=eval_max_tokens, seed=seed)

    batch_size = micro_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, len(examples) // batch_size)
    num_epochs = max(num_epochs, min_steps / steps_per_epoch)
    total_steps = max(1, round(steps_per_epoch * num_epochs))
    print(f"{len(examples)} examples, batch {batch_size}, {num_epochs:.2f} epochs = {total_steps} steps")

    history = []

    def run_eval(step: int) -> None:
        policy.eval()
        load_policy_into_vllm_instance(policy, llm)
        metrics, _ = evaluate_vllm(
            llm, r1_zero_reward_fn, [e["prompt"] for e in val], [e["answer"] for e in val], eval_params
        )
        gen = log_generations(
            llm, policy, tokenizer, [e["prompt"] for e in log_subset], [e["answer"] for e in log_subset],
            r1_zero_reward_fn, eval_params, policy_device, step, run_dir / "generations.jsonl",
        )
        row = {"eval_step": step, **{f"eval/{k}": v for k, v in metrics.items()}, **{f"eval/{k}": v for k, v in gen.items()}}
        logger.log(row)
        history.append(row)
        print(f"[eval] step {step}: accuracy {metrics['accuracy']:.4f} format {metrics['format_rate']:.4f}")
        policy.train()

    run_eval(0)
    sft_train(
        policy, tokenizer, optimizer, examples,
        device=policy_device,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        lr=lr,
        max_seq_len=max_seq_len,
        loss_normalization=loss_normalization,
        seed=seed,
        log_fn=logger.log,
        on_step=lambda s: run_eval(s) if (s % eval_every == 0 and s != total_steps) else None,
    )
    run_eval(total_steps)

    best = max(history, key=lambda r: r["eval/accuracy"])
    summary = {
        "run_name": run_name,
        "num_train_examples": len(examples),
        "data": data,
        "total_steps": total_steps,
        "final_accuracy": history[-1]["eval/accuracy"],
        "final_format_rate": history[-1]["eval/format_rate"],
        "best_accuracy": best["eval/accuracy"],
        "best_step": best["eval_step"],
        "lr": lr,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "eval_max_tokens": eval_max_tokens,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if save:
        save_policy(policy, tokenizer, run_dir / "model")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
