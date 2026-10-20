"""Policy-gradient / GRPO primitives (handout §7.2)."""

from __future__ import annotations

from typing import Callable, Literal

import torch

LossType = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Score each rollout and normalize rewards within groups of `group_size` (Eq. 28 / Eq. 31)."""
    assert len(rollout_responses) == len(repeated_ground_truths)
    assert len(rollout_responses) % group_size == 0

    scores = [reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)
    format_rewards = torch.tensor([s["format_reward"] for s in scores], dtype=torch.float32)
    answer_rewards = torch.tensor([s["answer_reward"] for s in scores], dtype=torch.float32)

    grouped = raw_rewards.view(-1, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - group_mean
    group_std = grouped.std(dim=1, keepdim=True)  # unbiased, as torch default
    if normalize_by_std:
        advantages = advantages / (group_std + advantage_eps)
    advantages = advantages.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item() if raw_rewards.numel() > 1 else 0.0,
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "format_reward_mean": format_rewards.mean().item(),
        "answer_reward_mean": answer_rewards.mean().item(),
        "frac_groups_all_same": (group_std.squeeze(1) == 0).float().mean().item(),
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-token loss -A * log p, with A of shape (B, 1) broadcast over the sequence."""
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token GRPO-Clip loss (Eq. 33)."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    loss = -torch.minimum(unclipped, clipped)
    metadata = {
        # True where the clipped term was the (strictly) smaller one, i.e. clipping was active.
        "clipped": clipped < unclipped,
        "ratio": ratio.detach(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: LossType,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the requested policy-gradient loss."""
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "no_baseline requires raw_rewards"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        assert advantages is not None, "reinforce_with_baseline requires advantages"
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        assert advantages is not None, "grpo_clip requires advantages"
        assert old_log_probs is not None, "grpo_clip requires old_log_probs"
        assert cliprange is not None, "grpo_clip requires cliprange"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean of `tensor` over positions where mask == 1."""
    mask = mask.to(tensor.dtype)
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: LossType,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token PG loss -> per-example masked mean -> batch mean -> /grad_accum -> backward().

    If `length_normalize_constant` is given, the per-example reduction is a masked sum divided by
    that constant (Dr. GRPO-style) instead of a masked mean.
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    if length_normalize_constant is None:
        per_example = masked_mean(per_token_loss, response_mask, dim=-1)
    else:
        per_example = (per_token_loss * response_mask).sum(dim=-1) / length_normalize_constant
    loss = per_example.mean() / gradient_accumulation_steps
    loss.backward()

    if "clipped" in metadata:
        metadata["clip_fraction"] = masked_mean(metadata["clipped"].float(), response_mask)
    return loss.detach(), metadata
