"""Helpers shared by SFT, expert iteration and GRPO (handout §4.2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

ENTROPY_CHUNK_POSITIONS = 512


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and outputs separately, concatenate, and build a response mask.

    Returns input_ids / labels / response_mask, each of shape (B, max_len - 1), where
    labels are input_ids shifted left by one and response_mask marks response tokens in labels.
    """
    assert len(prompt_strs) == len(output_strs)
    prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompt_strs]
    output_ids = [tokenizer(o, add_special_tokens=False)["input_ids"] for o in output_strs]

    seqs = [p + o for p, o in zip(prompt_ids, output_ids)]
    max_len = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    full = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        full[i, : len(p) + len(o)] = torch.tensor(p + o, dtype=torch.long)
        mask[i, len(p) : len(p) + len(o)] = True

    return {
        "input_ids": full[:, :-1],
        "labels": full[:, 1:],
        "response_mask": mask[:, 1:],
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-token entropy over the vocab dimension, computed stably via logsumexp."""
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - log_z
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-token log p(label_t | input_ids_<=t), and optionally per-token entropy."""
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    out = {"log_probs": log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)}
    if return_token_entropy:
        # Entropy is a logging metric only, so keep it out of the autograd graph, and compute it a chunk of
        # positions at a time: each float32 (seq_len, vocab) temporary is ~2.3 GiB at 4096 tokens.
        with torch.no_grad():
            chunks = logits.detach().split(ENTROPY_CHUNK_POSITIONS, dim=1)
            out["token_entropy"] = torch.cat([compute_entropy(c.float()) for c in chunks], dim=1)
    return out


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum `tensor` where mask == 1 (over `dim`, or everything if None) and divide by a constant."""
    masked = tensor * mask.to(tensor.dtype)
    total = masked.sum() if dim is None else masked.sum(dim=dim)
    return total / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """NLL of response tokens (summed per example / normalize_constant, averaged over batch),
    scaled for gradient accumulation; calls backward()."""
    per_example_nll = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1)
    loss = per_example_nll.mean() / gradient_accumulation_steps
    loss.backward()
    metadata = {
        "nll": per_example_nll.mean().detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    return loss.detach(), metadata
