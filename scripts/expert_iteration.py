"""Problem expert_iteration_experiment (§5, Algorithm 2) on MATH train.

Each EI step: sample |D_b| train questions, draw G rollouts per question from the current policy,
keep the rollouts with reward 1, then run SFT on them for --sft-epochs. Validation accuracy and
response entropy are logged after every EI step (step 0 = base model).

Examples:
  uv run python scripts/expert_iteration.py --batch-size 512  --group-size 4 --sft-epochs 1 --run-name ei_b512_g4_e1
  uv run python scripts/expert_iteration.py --batch-size 1024 --group-size 8 --sft-epochs 2 --run-name ei_b1024_g8_e2
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import torch
import typer

from alignment.data import load_math
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.logging_utils import MetricsLogger, init_wandb, log_generations, truncate_jsonl
from alignment.training import (
    load_checkpoint_state,
    load_policy,
    make_optimizer,
    save_checkpoint,
    save_policy,
    sft_train,
)
from alignment.vllm_utils import (
    evaluate_vllm,
    generate,
    init_vllm,
    load_policy_into_vllm_instance,
    make_sampling_params,
    summarize_scores,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = "models/Qwen2.5-Math-1.5B",
    run_name: str = "ei",
    n_ei_steps: int = 5,
    batch_size: int = typer.Option(512, help="|D_b|: questions sampled per EI step (512/1024/2048)"),
    group_size: int = typer.Option(4, help="G: rollouts per question"),
    sft_epochs: float = 1.0,
    lr: float = 2e-5,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 32,
    max_seq_len: int = 2048,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 1024,
    sampling_min_tokens: int = 4,
    n_eval: int = 1500,
    n_log_generations: int = 32,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:0",
    gpu_memory_utilization: float = typer.Option(0.3, help="vLLM share of the GPU it shares with the policy"),
    seed: int = 0,
    wandb_project: str = typer.Option(None, help="Set to enable wandb logging"),
    save: bool = False,
    resume: bool = typer.Option(True, help="Resume from runs/<run_name>/checkpoint (saved after every EI step)"),
):
    config = dict(locals())
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    ckpt_dir = run_dir / "checkpoint"
    resuming = resume and (ckpt_dir / "trainer_state.pt").exists()
    if not resuming:
        for f in ("metrics.jsonl", "generations.jsonl"):
            (run_dir / f).unlink(missing_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    train = load_math("train")
    val = load_math("validation")[:n_eval]
    log_subset = rng.sample(val, min(n_log_generations, len(val)))

    use_wandb = init_wandb(wandb_project, run_name, config)
    logger = MetricsLogger(run_dir / "metrics.jsonl", use_wandb)
    # Start vLLM before the policy: on a shared GPU it sizes its KV cache from the memory free at startup.
    llm = init_vllm(model, vllm_device, seed, gpu_memory_utilization)
    policy, tokenizer = load_policy(str(ckpt_dir) if resuming else model, policy_device, gradient_checkpointing=True)
    optimizer = make_optimizer(policy, lr)
    eval_params = make_sampling_params(max_tokens=sampling_max_tokens, seed=seed)

    history = []

    def run_eval(ei_step: int, train_step: int) -> None:
        policy.eval()
        load_policy_into_vllm_instance(policy, llm)
        metrics, _ = evaluate_vllm(
            llm, r1_zero_reward_fn, [e["prompt"] for e in val], [e["answer"] for e in val], eval_params
        )
        gen = log_generations(
            llm, policy, tokenizer, [e["prompt"] for e in log_subset], [e["answer"] for e in log_subset],
            r1_zero_reward_fn, eval_params, policy_device, ei_step, run_dir / "generations.jsonl",
        )
        row = {"eval_step": ei_step, "train_step": train_step,
               **{f"eval/{k}": v for k, v in metrics.items()}, **{f"eval/{k}": v for k, v in gen.items()}}
        logger.log(row)
        history.append(row)
        print(f"[eval] EI step {ei_step}: accuracy {metrics['accuracy']:.4f} entropy {gen['gen/mean_token_entropy']:.3f}")
        policy.train()

    global_step, done_steps = 0, 0
    if resuming:
        state = load_checkpoint_state(ckpt_dir, optimizer)
        done_steps, global_step = state["ei_step"], state["global_step"]
        rng.setstate(state["rng"])
        rows = truncate_jsonl(run_dir / "metrics.jsonl", lambda r: r.get("eval_step", r.get("ei_step", 0)) <= done_steps)
        truncate_jsonl(run_dir / "generations.jsonl", lambda r: r["step"] <= done_steps)
        history = [r for r in rows if "eval_step" in r]
        print(f"Resumed from {ckpt_dir} after EI step {done_steps}")
    else:
        run_eval(0, 0)

    for ei_step in range(done_steps + 1, n_ei_steps + 1):
        load_policy_into_vllm_instance(policy, llm)
        questions = rng.sample(train, batch_size)
        rollout_params = make_sampling_params(
            temperature=sampling_temperature, max_tokens=sampling_max_tokens,
            min_tokens=sampling_min_tokens, n=group_size, seed=seed + ei_step,
        )
        rollouts = generate(llm, [q["prompt"] for q in questions], rollout_params)

        sft_data, scores, n_solved = [], [], 0
        for q, outs in zip(questions, rollouts):
            solved = False
            for o in outs:
                s = r1_zero_reward_fn(o, q["answer"])
                scores.append(s)
                if s["reward"] == 1.0:
                    sft_data.append({"prompt": q["prompt"], "response": o})
                    solved = True
            n_solved += solved
        roll = summarize_scores(scores)
        logger.log({"ei_step": ei_step, "train_step": global_step,
                    "rollout/accuracy": roll["accuracy"], "rollout/format_rate": roll["format_rate"],
                    "rollout/n_correct": len(sft_data), "rollout/frac_questions_solved": n_solved / batch_size})
        print(f"[EI {ei_step}] {len(sft_data)} correct of {len(scores)} rollouts; {n_solved}/{batch_size} questions solved")

        if sft_data:
            global_step = sft_train(
                policy, tokenizer, optimizer, sft_data,
                device=policy_device,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_epochs=sft_epochs,
                lr=lr,
                max_seq_len=max_seq_len,
                seed=seed + ei_step,
                start_step=global_step,
                log_fn=lambda m: logger.log({**m, "ei_step": ei_step}),
            )
        run_eval(ei_step, global_step)
        if ei_step < n_ei_steps:
            save_checkpoint(ckpt_dir, policy, tokenizer, optimizer,
                            {"ei_step": ei_step, "global_step": global_step, "rng": rng.getstate()})

    best = max(history, key=lambda r: r["eval/accuracy"])
    summary = {
        "run_name": run_name,
        "batch_size": batch_size,
        "group_size": group_size,
        "sft_epochs": sft_epochs,
        "accuracy_per_ei_step": [r["eval/accuracy"] for r in history],
        "entropy_per_ei_step": [r["eval/gen/mean_token_entropy"] for r in history],
        "final_accuracy": history[-1]["eval/accuracy"],
        "best_accuracy": best["eval/accuracy"],
        "best_ei_step": best["eval_step"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # run finished; free the space (re-running starts fresh)
    if save:
        save_policy(policy, tokenizer, run_dir / "model")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
