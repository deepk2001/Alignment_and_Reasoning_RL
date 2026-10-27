"""Optional instruction-tuning / evaluation / DPO helpers (non-mandatory parts of the handout).

Covers the adapters exercised by tests/test_data.py, tests/test_metrics.py and tests/test_dpo.py.
"""

from __future__ import annotations

import json
import os
import random
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from alignment.data import load_prompt_template


def alpaca_template() -> str:
    # The prompt file ends with a newline after {response}; documents end right at the response, then EOS.
    return load_prompt_template("alpaca_sft").removesuffix("\n")


class PackedSFTDataset(Dataset):
    """Alpaca-formatted {"prompt", "response"} documents, each followed by EOS, concatenated into one token
    stream and cut into non-overlapping windows of `seq_length` (labels are the stream shifted by one)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
        seed: int = 0,
    ):
        template = alpaca_template()
        with open(dataset_path) as f:
            docs = [json.loads(line) for line in f if line.strip()]
        if shuffle:
            random.Random(seed).shuffle(docs)

        stream: list[int] = []
        for doc in docs:
            text = template.format(instruction=doc["prompt"], response=doc["response"])
            stream.extend(tokenizer.encode(text))  # adds BOS for tokenizers that use one
            stream.append(tokenizer.eos_token_id)

        self.seq_length = seq_length
        self.tokens = torch.tensor(stream, dtype=torch.long)
        self.num_examples = (len(stream) - 1) // seq_length

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if not 0 <= i < self.num_examples:
            raise IndexError(i)
        start = i * self.seq_length
        return {
            "input_ids": self.tokens[start : start + self.seq_length],
            "labels": self.tokens[start + 1 : start + self.seq_length + 1],
        }


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """One epoch over `dataset` in batches of `batch_size` (the last batch may be smaller)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(mmlu_example: dict, model_output: str) -> str | None:
    """Pull the predicted option letter out of e.g. "The correct answer is B"; None if there isn't one."""
    match = re.search(r"answer is\s*\(?([ABCD])\b", model_output)
    return match.group(1) if match else None


def parse_gsm8k_response(model_output: str) -> str | None:
    """The last number in the output (commas dropped), or None if it has no numbers."""
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", model_output)
    return numbers[-1].replace(",", "") if numbers else None


def _sequence_log_prob(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids.unsqueeze(0)).logits[0, :-1]
    return F.log_softmax(logits.float(), dim=-1).gather(-1, input_ids[1:].unsqueeze(-1)).sum()


def per_instance_dpo_loss(
    lm: PreTrainedModel,
    lm_ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """DPO loss -log sigmoid(beta * (policy log-ratio - reference log-ratio)) for one preference pair.

    Each response is Alpaca-formatted with its prompt and followed by EOS. Log-probs are summed over the whole
    sequence: the prompt tokens are identical for chosen and rejected, so they cancel in the difference.
    """
    template = alpaca_template()

    def encode(response: str) -> torch.Tensor:
        ids = tokenizer.encode(template.format(instruction=prompt, response=response)) + [tokenizer.eos_token_id]
        return torch.tensor(ids, dtype=torch.long)

    chosen, rejected = encode(response_chosen), encode(response_rejected)
    with torch.no_grad():
        ref_ratio = _sequence_log_prob(lm_ref, chosen.to(lm_ref.device)) - _sequence_log_prob(
            lm_ref, rejected.to(lm_ref.device)
        )
    policy_ratio = _sequence_log_prob(lm, chosen.to(lm.device)) - _sequence_log_prob(lm, rejected.to(lm.device))
    return -F.logsigmoid(beta * (policy_ratio - ref_ratio.to(policy_ratio.device)))
