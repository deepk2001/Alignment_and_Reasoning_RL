# Assignment 2 Roadmap: Alignment & Reasoning RL

Model: **Qwen2.5-Math-1.5B (base)** · Task: **MATH** · Reward: `alignment.drgrpo_grader.r1_zero_reward_fn`

Legend: 🧪 = graded by unit tests · 💻 = code deliverable (goes in `code.zip`) · 📝 = goes in `writeup.pdf` · 📈 = plot/figure for writeup

---

## 0. Submission at a glance

| File | Where | Contents |
|---|---|---|
| `code.zip` | Gradescope | Everything in repo except `data/`, weights, `*.json`, `*.txt`, logs — built by `./test_and_make_submission.sh` |
| `writeup.pdf` | Canvas | Typeset answers to every 📝 item below |

Required tests: **only `tests/test_sft.py` and `tests/test_grpo.py`**. (`test_data.py`, `test_dpo.py`, `test_metrics.py` are for non-mandatory parts — skip.)

---

## 1. Gotchas found while reading the repo (fix/keep in mind early)

- [x] **No GPU on this Mac → experiments run on a single Colab A100** via [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb). Policy and vLLM share the GPU (vLLM `gpu_memory_utilization=0.3`, started before the policy). flash-attn is skipped (SDPA fallback). GRPO and expert iteration checkpoint to Drive and resume after disconnects.
- [ ] **`pyproject.toml` packages `cs336_alignment`, but the code lives in `alignment/`.** Change `include = ["alignment"]` (or imports from `alignment.*` may fail once installed as a package).
- [ ] **MATH data has no `answer` field** — only `problem`, `solution`, `level`, `type`. Ground truth = `extract_answer(solution)` (the `\boxed{}` extractor in `drgrpo_grader.py`).
- [ ] **Validation set is 1,500 examples** (`data/MATH/validation.jsonl`), not the "5K test" the PDF intro mentions. Train is 6,000. Report on validation.
- [ ] **`data/sft.jsonl` has 1,000 examples**, so the SFT size sweep `{128, 256, 512, 1024}` + "full" effectively collapses 1024 ≈ full. Run `{128, 256, 512, full=1000}` and note this in the writeup.
- [ ] Reward is strict on format: requires the literal `"</think> <answer>"` (one space) and `"</answer>"`. Matters when diagnosing format-0 cases.
- [ ] Generation settings everywhere: `temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True`; add `min_tokens=4` for EI/GRPO rollouts.
- [ ] No HF `Trainer` or other training utilities allowed — hand-written loops only. vLLM for generation, HF only for model loading + forward passes.

---

## 2. Proposed code layout

```
alignment/
  data.py          # load MATH / sft.jsonl, format r1_zero prompts, extract gold answers
  sft_utils.py     # tokenize_prompt_and_output, compute_entropy, get_response_log_probs,
                   # masked_normalize, sft_microbatch_train_step
  grpo_utils.py    # compute_group_normalized_rewards, naive PG loss, grpo_clip loss,
                   # compute_policy_gradient_loss, masked_mean, grpo_microbatch_train_step
  vllm_utils.py    # init_vllm, load_policy_into_vllm_instance, evaluate_vllm
  logging_utils.py # log_generations
scripts/
  math_baseline.py       # §3 zero-shot eval
  sft_train.py           # §4 SFT experiment (typer CLI)
  expert_iteration.py    # §5 EI experiment
  grpo_train.py          # §7 GRPO train loop
  plots.py               # make all writeup figures from saved logs/W&B exports
tests/adapters.py        # thin wrappers calling the above
```

---

## 3. Phased roadmap

### Phase 1 — Core primitives + passing tests (local, CPU) 🧪💻
Do these first: they're autograded and everything later depends on them.

| # | Problem | Function | Test command |
|---|---|---|---|
| 1 | `tokenize_prompt_and_output` | returns `input_ids`, `labels` (shifted), `response_mask`; shape `(B, max_len-1)` | `uv run pytest -k test_tokenize_prompt_and_output` |
| 2 | `compute_entropy` | per-token entropy via `logsumexp` (numerically stable) | `-k test_compute_entropy` |
| 3 | `get_response_log_probs` | `log_softmax` + `gather` on labels; optional `token_entropy` | `-k test_get_response_log_probs` |
| 4 | `masked_normalize` | `sum(tensor*mask, dim) / constant` | `-k test_masked_normalize` |
| 5 | `sft_microbatch_train_step` | `-masked_normalize(logp, mask, c) ` averaged over batch, `/ grad_accum`, `.backward()` | `-k test_sft_microbatch_train_step` |
| 6 | `compute_group_normalized_rewards` | reshape `(n_groups, G)`, subtract mean, optionally `/ (std + eps)` | `-k test_compute_group_normalized_rewards` |
| 7 | `compute_naive_policy_gradient_loss` | `-A * logp` broadcast over seq | `-k test_compute_naive_policy_gradient_loss` |
| 8 | `compute_grpo_clip_loss` | `-min(ratio*A, clip(ratio,1-ε,1+ε)*A)`; log clip mask | `-k test_compute_grpo_clip_loss` |
| 9 | `compute_policy_gradient_loss` | dispatcher + arg asserts | `-k test_compute_policy_gradient_loss` |
| 10 | `masked_mean` | `sum(t*mask, dim) / sum(mask, dim)` | `-k test_masked_mean` |
| 11 | `grpo_microbatch_train_step` | per-token loss → `masked_mean(dim=-1)` → batch mean → `/ grad_accum` → `.backward()` | `-k test_grpo_microbatch_train_step` |
| 12 | `log_generations` 💻 | logs prompt, response, gold, format/answer/total reward, mean token entropy, avg length (all / correct / incorrect) | — |

**Exit criterion:** `uv run pytest tests/test_sft.py tests/test_grpo.py` all green.

### Phase 2 — Shared infrastructure (GPU box)
- [ ] `data.py`: load MATH jsonl, `r1_zero.prompt` formatting, gold answer extraction.
- [ ] `vllm_utils.py`: `init_vllm` (with the TRL monkeypatches from the PDF), `load_policy_into_vllm_instance`, `evaluate_vllm(llm, reward_fn, prompts, golds, sampling_params)` that returns metrics and writes a jsonl of `{prompt, response, gold, rewards}`.
- [ ] Training-loop skeleton: AdamW, grad clip 1.0, grad accumulation, W&B with `train_step`/`eval_step` metrics, policy on `cuda:0`, vLLM on `cuda:1` (or shared GPU with lower `gpu_memory_utilization`).
- [ ] Save checkpoints with `model.save_pretrained` + `tokenizer.save_pretrained`.

### Phase 3 — Zero-shot baseline (`math_baseline`) §3.2
- [ ] (a) 💻 `scripts/math_baseline.py`: load validation → r1_zero prompts → generate → score → serialize to disk.
- [ ] (b) 📝 Count the three buckets: (1) format=1 & answer=1, (2) format=1 & answer=0, (3) format=0. Inspect **≥10 format=0** and **≥10 format=1/answer=0** cases; argue whether the base model or the parser is at fault. Include example generations from each bucket.
- [ ] (c) 📝 1–2 sentences: zero-shot accuracy (answer reward mean) + format rate.

### Phase 4 — SFT experiment (`sft_experiment`) §4.3
- [ ] 💻 `scripts/sft_train.py` (Algorithm 1) with periodic vLLM eval on validation and `log_generations`.
- [ ] Tune LR / batch size until **full dataset ≥ 15% val accuracy** (hard requirement).
- [ ] Run sizes `{128, 256, 512, 1024/full}`.
  - 📈📝 Validation-accuracy curves for each dataset size (one plot, multiple lines).
- [ ] Filter `sft.jsonl` to examples whose response is graded correct by `r1_zero_reward_fn` against the gold answer (match prompts back to MATH train problems to get golds). Run SFT on the filtered set.
  - 📝 Size of filtered dataset.
  - 📈📝 Validation-accuracy curve for filtered SFT.
  - 📝 Comparison against unfiltered full-dataset SFT.
- [ ] 📝 Note the hyperparameters used (LR, batch size, epochs/steps).

### Phase 5 — Expert Iteration (`expert_iteration_experiment`) §5
- [ ] 💻 `scripts/expert_iteration.py` (Algorithm 2): sample `D_b` from `data/MATH/train.jsonl`, generate G rollouts with `min_tokens=4`, keep reward=1 samples, run SFT on them, `n_ei_steps=5`.
- [ ] Sweep (not full grid — enough to draw conclusions):
  - batch size `|D_b| ∈ {512, 1024, 2048}`
  - **≥2 rollout counts G** (e.g. 4, 8)
  - **≥2 SFT epoch counts** (e.g. 1, 2)
- [ ] Log mean response entropy throughout training.
- Report items:
  - 📈📝 Validation-accuracy curves for the different rollout configurations.
  - 📝 State that a model reaches **≥ 15% val accuracy** (give the config + number).
  - 📝 2-sentence comparison: EI vs. SFT, and how accuracy changes across EI steps.
  - 📈📝 Plot of response entropy over training.

### Phase 6 — GRPO (`grpo_train_loop`) §7
- [ ] 💻 `scripts/grpo_train.py` (Algorithm 3) using the starter hyperparameters:
  `n_grpo_steps=200, lr=1e-5, advantage_eps=1e-6, rollout_batch_size=256, group_size=8, epochs_per_rollout_batch=1, train_batch_size=256, grad_accum=128, loss_type="reinforce_with_baseline", use_std_normalization=True`, AdamW `betas=(0.9,0.95), wd=0`.
- [ ] Include the PDF's sanity asserts (`train_batch_size % grad_accum == 0`, etc.).
- [ ] Off-policy support: compute `old_log_probs` once per rollout batch under `torch.no_grad()`, reuse across epochs; `grpo_clip` only when off-policy.
- [ ] Log per optimizer step: loss, grad norm, token entropy, clip fraction (off-policy), train rewards (total/format/answer).
- [ ] Eval on **≥1024 validation examples every 5–10 steps**.
- Report items:
  - 📈📝 Validation reward vs. GRPO step.
  - 📝 A few example rollouts at different points in training (e.g. step 0, ~100, 200), showing how they change.
  - 📝 Brief confirmation that validation reward improves and rollouts look sensible.

> The PDF mentions comparing against the `question_only` prompt, no-baseline vs. baseline, std-normalization, and `masked_normalize` vs. `masked_mean` "later in the assignment", but this version of the handout has **no problem statements** for those ablations. Treat them as optional unless the course says otherwise. The code should still support them (`loss_type`, `use_std_normalization` flags).

---

## 4. Writeup (`writeup.pdf`) checklist

| Section | Problem | Required content |
|---|---|---|
| 3.2 | `math_baseline` (a) | Short description of the eval script (code itself goes in zip) |
| 3.2 | `math_baseline` (b) | Bucket counts (1)/(2)/(3); ≥10 format-0 cases analysed; ≥10 format-1/answer-0 cases analysed; model-vs-parser verdict with reasoning; example generations per bucket |
| 3.2 | `math_baseline` (c) | 1–2 sentences with zero-shot val accuracy |
| 4.2 | `log_generations` | (code only; optionally show a sample log) |
| 4.3 | `sft_experiment` 1 | 📈 Val accuracy curves for 128 / 256 / 512 / full; hyperparameters; full ≥ 15% |
| 4.3 | `sft_experiment` 2 | Filtered dataset size; 📈 val accuracy curve; comparison to unfiltered |
| 5 | `expert_iteration_experiment` | 📈 Val accuracy curves (≥2 G values, ≥2 epoch values, batch sizes from {512,1024,2048}); model ≥ 15%; 2-sentence EI vs SFT + across-step discussion; 📈 entropy over training |
| 7.2 | `grpo_train_loop` | 📈 Val reward vs steps; example rollouts over time |

Code-only deliverables (no writeup text needed, but must be in `code.zip` and pass tests): `tokenize_prompt_and_output`, `compute_entropy`, `get_response_log_probs`, `masked_normalize`, `sft_microbatch_train_step`, `compute_group_normalized_rewards`, `compute_naive_policy_gradient_loss`, `compute_grpo_clip_loss`, `compute_policy_gradient_loss`, `masked_mean`, `grpo_microbatch_train_step`.

---

## 5. Suggested order & rough GPU budget

1. Phase 1 locally (no GPU) — get tests green.
2. Spin up GPU box; Phase 2 + Phase 3 baseline (~minutes: 1.5K generations).
3. Phase 4 SFT — 5 runs (4 sizes + filtered); short runs, eval every N steps.
4. Phase 5 EI — ~4–6 runs × 5 EI steps; generation dominates cost (`|D_b| × G` rollouts per step).
5. Phase 6 GRPO — 200 steps × 256 rollouts + evals; the most expensive single run. Start early in case of debugging.
6. Make plots from saved logs → write `writeup.pdf` → run `./test_and_make_submission.sh` → upload `code.zip` (Gradescope) and `writeup.pdf` (Canvas).
