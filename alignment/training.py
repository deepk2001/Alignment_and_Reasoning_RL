"""Model loading and the hand-written SFT loop (Algorithm 1), shared by SFT and expert iteration."""

from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from alignment.grpo_utils import masked_mean
from alignment.sft_utils import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output


def load_policy(
    model_id: str, device: str, gradient_checkpointing: bool = False
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model.to(device), tokenizer


def save_policy(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path) -> None:
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


def save_checkpoint(
    ckpt_dir: str | Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    state: dict,
) -> None:
    """Save policy, optimizer and loop state so a run can resume after a crash or disconnect.

    Writes to a temp dir and renames it into place, so an interrupted save never clobbers the last good one.
    """
    ckpt_dir = Path(ckpt_dir)
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_policy(model, tokenizer, tmp)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "state": state,
            "torch_rng": torch.get_rng_state(),
        },
        tmp / "trainer_state.pt",
    )
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    tmp.rename(ckpt_dir)


def load_checkpoint_state(ckpt_dir: str | Path, optimizer: torch.optim.Optimizer) -> dict:
    """Restore optimizer and torch RNG state from `ckpt_dir` and return the saved loop state (which should
    include the caller's own random.Random state). Load the policy itself via load_policy(ckpt_dir)."""
    blob = torch.load(Path(ckpt_dir) / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["torch_rng"])
    return blob["state"]


def make_optimizer(model: PreTrainedModel, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def cosine_lr(step: int, total_steps: int, max_lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def tokenize_batch(
    prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    max_seq_len: int | None = None,
) -> dict[str, torch.Tensor]:
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    if max_seq_len is not None:
        batch = {k: v[:, :max_seq_len] for k, v in batch.items()}
    return {k: v.to(device) for k, v in batch.items()}


def sft_train(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    examples: list[dict],
    *,
    device: str,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: float,
    lr: float,
    warmup_ratio: float = 0.05,
    max_seq_len: int | None = None,
    loss_normalization: str = "token_mean",
    seed: int = 0,
    start_step: int = 0,
    log_fn: Callable[[dict], None] | None = None,
    on_step: Callable[[int], None] | None = None,
) -> int:
    """Run SFT on {"prompt", "response"} examples for `num_epochs` (may be fractional).

    Each optimizer step consumes micro_batch_size * gradient_accumulation_steps examples. Gradients are
    clipped to norm 1.0. `loss_normalization` is "token_mean" (divide by #response tokens in the
    microbatch) or "sum" (handout default normalize_constant=1.0). Returns the global step reached.
    """
    rng = random.Random(seed)
    batch_size = micro_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, len(examples) // batch_size)
    total_steps = max(1, round(steps_per_epoch * num_epochs))
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    order: list[int] = []
    policy.train()
    global_step = start_step
    for local_step in range(total_steps):
        for g in optimizer.param_groups:
            g["lr"] = cosine_lr(local_step, total_steps, lr, warmup_steps)

        loss_sum, ent_sum, tok_sum = 0.0, 0.0, 0
        for _ in range(gradient_accumulation_steps):
            if len(order) < micro_batch_size:
                perm = list(range(len(examples)))
                rng.shuffle(perm)
                order.extend(perm)
            idx, order = order[:micro_batch_size], order[micro_batch_size:]
            mb = [examples[i] for i in idx]
            batch = tokenize_batch([e["prompt"] for e in mb], [e["response"] for e in mb], tokenizer, device, max_seq_len)
            out = get_response_log_probs(policy, batch["input_ids"], batch["labels"], return_token_entropy=True)
            mask = batch["response_mask"]
            n_tok = int(mask.sum().item())
            if loss_normalization == "token_mean":
                normalize_constant = max(n_tok, 1) / len(mb)
            else:
                normalize_constant = 1.0
            loss, _ = sft_microbatch_train_step(out["log_probs"], mask, gradient_accumulation_steps, normalize_constant)
            loss_sum += loss.item()
            ent_sum += masked_mean(out["token_entropy"], mask).item() * n_tok
            tok_sum += n_tok

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if log_fn is not None:
            log_fn(
                {
                    "train_step": global_step,
                    "train/loss": loss_sum,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/token_entropy": ent_sum / max(tok_sum, 1),
                    "train/response_tokens": tok_sum,
                }
            )
        if on_step is not None:
            on_step(global_step)
    return global_step
