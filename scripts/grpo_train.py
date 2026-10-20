"""Problem grpo_train_loop (§7.2, Algorithm 3): GRPO on MATH train with the r1_zero reward.

Defaults are the handout's starter hyperparameters (on-policy, reinforce_with_baseline, std-normalized
advantages). For the off-policy setting set --epochs-per-rollout-batch > 1 and/or
--train-batch-size < --rollout-batch-size, and use --loss-type grpo_clip.

Outputs go to runs/<run_name>/: metrics.jsonl (train + eval), rollouts.jsonl (example rollouts over
time) and summary.json.

Example:
  uv run python scripts/grpo_train.py --run-name grpo_default
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Literal

import torch
import typer

from alignment.data import load_math
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.grpo_utils import compute_group_normalized_rewards, grpo_microbatch_train_step, masked_mean
from alignment.logging_utils import MetricsLogger, init_wandb, truncate_jsonl
from alignment.sft_utils import get_response_log_probs
from alignment.training import load_checkpoint_state, load_policy, save_checkpoint, save_policy, tokenize_batch
from alignment.vllm_utils import evaluate_vllm, generate, init_vllm, load_policy_into_vllm_instance, make_sampling_params

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = "models/Qwen2.5-Math-1.5B",
    run_name: str = "grpo_default",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = typer.Option(0.3, help="vLLM share of the GPU it shares with the policy"),
    loss_type: str = typer.Option("reinforce_with_baseline", help="no_baseline | reinforce_with_baseline | grpo_clip"),
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    length_normalization: str = typer.Option("masked_mean", help="masked_mean | masked_normalize"),
    prompt: str = typer.Option("r1_zero", help="r1_zero | question_only"),
    eval_every: int = 10,
    n_eval: int = 1024,
    n_example_rollouts: int = 4,
    gradient_checkpointing: bool = True,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:0",
    seed: int = 0,
    wandb_project: str = typer.Option(None, help="Set to enable wandb logging"),
    save: bool = False,
    checkpoint_every: int = typer.Option(10, help="Save a resumable checkpoint every N GRPO steps (0 = never)"),
    resume: bool = typer.Option(True, help="Resume from runs/<run_name>/checkpoint if it exists"),
):
    config = dict(locals())
    assert loss_type in ("no_baseline", "reinforce_with_baseline", "grpo_clip")
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    assert rollout_batch_size % train_batch_size == 0, "rollout_batch_size must be divisible by train_batch_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    off_policy = epochs_per_rollout_batch > 1 or train_batch_size < rollout_batch_size
    if loss_type == "grpo_clip":
        assert off_policy, "GRPO-Clip is only meaningful off-policy (it needs old log-probs)"

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    ckpt_dir = run_dir / "checkpoint"
    resuming = resume and (ckpt_dir / "trainer_state.pt").exists()
    if not resuming:
        for f in ("metrics.jsonl", "rollouts.jsonl"):
            (run_dir / f).unlink(missing_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    if prompt == "question_only":
        from alignment.drgrpo_grader import question_only_reward_fn as reward_fn
    else:
        reward_fn = r1_zero_reward_fn
    train = load_math("train", prompt_name=prompt)
    val = load_math("validation", prompt_name=prompt)[:n_eval]

    use_wandb = init_wandb(wandb_project, run_name, config)
    logger = MetricsLogger(run_dir / "metrics.jsonl", use_wandb)
    # Start vLLM before the policy: on a shared GPU it sizes its KV cache from the memory free at startup.
    llm = init_vllm(model, vllm_device, seed, gpu_memory_utilization)
    policy, tokenizer = load_policy(str(ckpt_dir) if resuming else model, policy_device, gradient_checkpointing)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    eval_params = make_sampling_params(max_tokens=sampling_max_tokens, seed=seed)
    length_constant = float(sampling_max_tokens) if length_normalization == "masked_normalize" else None

    history = []
    train_step = 0
    start_step = 0
    if resuming:
        state = load_checkpoint_state(ckpt_dir, optimizer)
        start_step, train_step = state["grpo_step"], state["train_step"]
        rng.setstate(state["rng"])
        rows = truncate_jsonl(
            run_dir / "metrics.jsonl",
            lambda r: r.get("eval_step", r.get("grpo_step", start_step)) < start_step,
        )
        truncate_jsonl(run_dir / "rollouts.jsonl", lambda r: r["step"] < start_step)
        history = [r for r in rows if "eval_step" in r]
        print(f"Resumed from {ckpt_dir} at GRPO step {start_step}")

    def run_eval(step: int) -> None:
        metrics, records = evaluate_vllm(
            llm, reward_fn, [e["prompt"] for e in val], [e["answer"] for e in val], eval_params
        )
        row = {"eval_step": step, **{f"eval/{k}": v for k, v in metrics.items()}}
        logger.log(row)
        history.append(row)
        with open(run_dir / "rollouts.jsonl", "a") as f:
            for rec in records[:n_example_rollouts]:
                f.write(json.dumps({"step": step, **rec}) + "\n")
        print(f"[eval] GRPO step {step}: reward {metrics['reward']:.4f} format {metrics['format_rate']:.4f}")

    for grpo_step in range(start_step, n_grpo_steps):
        policy.eval()
        load_policy_into_vllm_instance(policy, llm)
        if grpo_step % eval_every == 0:
            run_eval(grpo_step)

        # 1. Rollouts: G samples for each of n_prompts questions.
        questions = rng.sample(train, n_prompts_per_rollout_batch)
        rollout_params = make_sampling_params(
            temperature=sampling_temperature, max_tokens=sampling_max_tokens,
            min_tokens=sampling_min_tokens, n=group_size, seed=seed + grpo_step,
        )
        outs = generate(llm, [q["prompt"] for q in questions], rollout_params)
        prompts = [q["prompt"] for q in questions for _ in range(group_size)]
        golds = [q["answer"] for q in questions for _ in range(group_size)]
        responses = [o for group in outs for o in group]

        # 2. Rewards and group-normalized advantages.
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn, responses, golds, group_size, advantage_eps, use_std_normalization
        )
        advantages = advantages.unsqueeze(1).to(policy_device)
        raw_rewards = raw_rewards.unsqueeze(1).to(policy_device)

        # 3. Old log-probs, computed once per rollout batch and reused across epochs (off-policy only).
        old_log_probs_mbs = None
        if off_policy:
            old_log_probs_mbs = []
            with torch.no_grad():
                for i in range(n_microbatches_per_rollout_batch):
                    sl = slice(i * micro_train_batch_size, (i + 1) * micro_train_batch_size)
                    b = tokenize_batch(prompts[sl], responses[sl], tokenizer, policy_device)
                    old_log_probs_mbs.append(get_response_log_probs(policy, b["input_ids"], b["labels"])["log_probs"])

        # 4. Policy updates.
        policy.train()
        n_train_batches = rollout_batch_size // train_batch_size
        for epoch in range(epochs_per_rollout_batch):
            mb_order = list(range(n_microbatches_per_rollout_batch))
            rng.shuffle(mb_order)
            for tb in range(n_train_batches):
                stats = {"loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
                for mb_idx in mb_order[tb * gradient_accumulation_steps : (tb + 1) * gradient_accumulation_steps]:
                    sl = slice(mb_idx * micro_train_batch_size, (mb_idx + 1) * micro_train_batch_size)
                    b = tokenize_batch(prompts[sl], responses[sl], tokenizer, policy_device)
                    out = get_response_log_probs(policy, b["input_ids"], b["labels"], return_token_entropy=True)
                    old = old_log_probs_mbs[mb_idx] if old_log_probs_mbs is not None else None
                    loss, meta = grpo_microbatch_train_step(
                        out["log_probs"], b["response_mask"], gradient_accumulation_steps, loss_type,
                        raw_rewards=raw_rewards[sl], advantages=advantages[sl],
                        old_log_probs=old, cliprange=cliprange,
                        length_normalize_constant=length_constant,
                    )
                    stats["loss"] += loss.item()
                    stats["entropy"] += masked_mean(out["token_entropy"], b["response_mask"]).item() / gradient_accumulation_steps
                    if "clip_fraction" in meta:
                        stats["clip_fraction"] += meta["clip_fraction"].item() / gradient_accumulation_steps

                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1
                row = {
                    "train_step": train_step,
                    "grpo_step": grpo_step,
                    "train/loss": stats["loss"],
                    "train/grad_norm": float(grad_norm),
                    "train/token_entropy": stats["entropy"],
                    "train/reward": reward_meta["reward_mean"],
                    "train/format_reward": reward_meta["format_reward_mean"],
                    "train/answer_reward": reward_meta["answer_reward_mean"],
                    "train/frac_groups_all_same": reward_meta["frac_groups_all_same"],
                }
                if off_policy:
                    row["train/clip_fraction"] = stats["clip_fraction"]
                logger.log(row)
        print(f"[GRPO {grpo_step}] train reward {reward_meta['reward_mean']:.3f} entropy {stats['entropy']:.3f}")
        if checkpoint_every and (grpo_step + 1) % checkpoint_every == 0 and grpo_step + 1 < n_grpo_steps:
            save_checkpoint(ckpt_dir, policy, tokenizer, optimizer,
                            {"grpo_step": grpo_step + 1, "train_step": train_step, "rng": rng.getstate()})

    policy.eval()
    load_policy_into_vllm_instance(policy, llm)
    run_eval(n_grpo_steps)

    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "use_std_normalization": use_std_normalization,
        "initial_val_reward": history[0]["eval/reward"],
        "final_val_reward": history[-1]["eval/reward"],
        "best_val_reward": max(r["eval/reward"] for r in history),
        "final_format_rate": history[-1]["eval/format_rate"],
        "n_grpo_steps": n_grpo_steps,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # run finished; free the space (re-running starts fresh)
    if save:
        save_policy(policy, tokenizer, run_dir / "model")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
