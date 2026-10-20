"""Problem sft_experiment part 2: keep only the SFT examples whose response reaches the correct answer.

data/sft.jsonl carries no gold answers. Its 1,000 examples are exactly s1K-1.1 (DeepSeek-R1 traces),
so gold answers come from that dataset's `solution` field. A response is kept if r1_zero_reward_fn
grades it correct against that gold. Many s1K golds are proofs or long free-form text that a string
grader cannot match, so for those the dataset's own `deepseek_grade` (an LLM judge comparing the
attempt to the gold) is used instead. See --mode.

Also drops examples whose question appears in the MATH validation set (it would leak into eval).

Example:
  uv run python scripts/filter_sft.py --mode grader_or_judge
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

from alignment.data import SFT_PATH, load_jsonl, question_from_r1_zero_prompt, validation_questions, write_jsonl
from alignment.drgrpo_grader import r1_zero_reward_fn

app = typer.Typer(add_completion=False)

S1K_REPO = "simplescaling/s1K-1.1"
SHORT_GOLD_CHARS = 40  # longer golds are treated as free-form and left to the judge


def load_s1k() -> dict[str, dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(S1K_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")
    return {row["question"].strip(): row for row in pq.read_table(path).to_pylist()}


@app.command()
def main(
    mode: str = typer.Option("grader_or_judge", help="grader | judge | grader_or_judge"),
    output: str = "data/sft_filtered.jsonl",
    gold_output: str = "data/sft_gold.jsonl",
):
    rows = load_jsonl(SFT_PATH)
    s1k = load_s1k()
    val_qs = validation_questions()

    kept, stats = [], {"total": len(rows), "leaks_into_val": 0, "unmatched": 0, "short_gold": 0,
                       "grader_correct": 0, "judge_correct": 0, "kept": 0}
    gold_rows = []
    for row in rows:
        q = question_from_r1_zero_prompt(row["prompt"])
        if q in val_qs:
            stats["leaks_into_val"] += 1
            continue
        meta = s1k.get(q)
        if meta is None:
            stats["unmatched"] += 1
            continue
        gold = str(meta["solution"]).strip()
        short = len(gold) <= SHORT_GOLD_CHARS
        grader_ok = short and r1_zero_reward_fn(row["response"], gold)["reward"] == 1.0
        judge_ok = meta["deepseek_grade"] == "Yes"
        stats["short_gold"] += short
        stats["grader_correct"] += grader_ok
        stats["judge_correct"] += judge_ok
        gold_rows.append({"question": q, "gold": gold, "short_gold": short,
                          "grader_correct": grader_ok, "judge_correct": judge_ok})

        if mode == "grader":
            keep = grader_ok
        elif mode == "judge":
            keep = judge_ok
        else:  # grader where the gold is gradable, else fall back to the judge
            keep = grader_ok if short else judge_ok
        if keep:
            kept.append(row)

    stats["kept"] = len(kept)
    agreement = Counter((g["grader_correct"], g["judge_correct"]) for g in gold_rows if g["short_gold"])
    stats["short_gold_agreement"] = {f"grader={a},judge={b}": n for (a, b), n in sorted(agreement.items())}
    write_jsonl(output, kept)
    write_jsonl(gold_output, gold_rows)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/sft_filter_stats.json").write_text(json.dumps({"mode": mode, **stats}, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    app()
