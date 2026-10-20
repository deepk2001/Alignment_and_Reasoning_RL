"""Build every writeup figure from runs/*/metrics.jsonl into report/figures/*.png.

Figures whose runs don't exist yet are skipped, so this can be re-run as experiments finish.

  uv run --with matplotlib python scripts/make_plots.py   (matplotlib is kept out of uv.lock on purpose)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs")
FIG_DIR = Path("report/figures")

# Reference categorical palette, light mode, in fixed slot order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_metrics(run: str) -> list[dict]:
    path = RUNS / run / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def series(rows: list[dict], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    pts = [(r[x_key], r[y_key]) for r in rows if x_key in r and y_key in r and r[y_key] is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def line_figure(
    lines: list[tuple[str, list[float], list[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    out: str,
    percent: bool = True,
) -> None:
    lines = [l for l in lines if l[1]]
    if not lines:
        print(f"skip {out}: no data yet")
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax, title, xlabel, ylabel)
    for i, (label, xs, ys) in enumerate(lines):
        color = SERIES[i]
        ys_plot = [100 * y for y in ys] if percent else ys
        ax.plot(xs, ys_plot, color=color, linewidth=2, marker="o", markersize=4, label=label)
        if len(lines) <= 4:  # direct-label the line ends in text ink, not series color
            ax.annotate(f"{ys_plot[-1]:.1f}{'%' if percent else ''}", (xs[-1], ys_plot[-1]),
                        textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=TEXT_2)
    if len(lines) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    if percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG_DIR / out}")


def runs_matching(prefix: str) -> list[str]:
    return sorted(p.name for p in RUNS.glob(f"{prefix}*") if (p / "metrics.jsonl").exists())


def main() -> None:
    sft_runs = [("128 examples", "sft_n128"), ("256 examples", "sft_n256"),
                ("512 examples", "sft_n512"), ("Full dataset", "sft_full")]
    line_figure(
        [(label, *series(read_metrics(run), "eval_step", "eval/accuracy")) for label, run in sft_runs],
        "SFT: validation accuracy by dataset size", "Optimizer step", "MATH val accuracy", "sft_sizes.png",
    )
    line_figure(
        [(label, *series(read_metrics(run), "eval_step", "eval/accuracy"))
         for label, run in [("Full (unfiltered)", "sft_full"), ("Filtered (correct only)", "sft_filtered")]],
        "SFT: filtered vs. unfiltered data", "Optimizer step", "MATH val accuracy", "sft_filtered.png",
    )

    ei_runs = runs_matching("ei_")[: len(SERIES)]
    line_figure(
        [(run.removeprefix("ei_"), *series(read_metrics(run), "eval_step", "eval/accuracy")) for run in ei_runs],
        "Expert iteration: validation accuracy", "EI step", "MATH val accuracy", "ei_accuracy.png",
    )
    line_figure(
        [(run.removeprefix("ei_"), *series(read_metrics(run), "eval_step", "eval/gen/mean_token_entropy"))
         for run in ei_runs],
        "Expert iteration: mean response entropy", "EI step", "Mean token entropy (nats)", "ei_entropy.png",
        percent=False,
    )

    line_figure(
        [("GRPO", *series(read_metrics("grpo_default"), "eval_step", "eval/reward"))],
        "GRPO: validation reward", "GRPO step", "Mean val reward", "grpo_val_reward.png",
    )
    line_figure(
        [("Train reward", *series(read_metrics("grpo_default"), "grpo_step", "train/reward"))],
        "GRPO: train (rollout) reward", "GRPO step", "Mean rollout reward", "grpo_train_reward.png",
    )
    line_figure(
        [("Entropy", *series(read_metrics("grpo_default"), "train_step", "train/token_entropy"))],
        "GRPO: response token entropy", "Optimizer step", "Mean token entropy (nats)", "grpo_entropy.png",
        percent=False,
    )


if __name__ == "__main__":
    main()
