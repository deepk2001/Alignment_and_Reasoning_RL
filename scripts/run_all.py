"""Run every experiment for the writeup with one command, in order:

  tests -> baseline -> sft -> ei -> grpo -> plots

Each step runs its script as a subprocess (so vLLM frees the GPU between steps), tees output to
runs/logs/<step>.log, and is skipped if runs/<step>/summary.json already exists. Expert iteration and GRPO
also resume from their own checkpoints, so after a crash or a dropped SSH session just re-run the same
command. Use tmux or nohup so the run survives disconnects:

  tmux new -s a2 'uv run python scripts/run_all.py'
  nohup uv run python scripts/run_all.py > runs/run_all.log 2>&1 &

GPU layout is detected automatically: with 2+ GPUs vLLM gets cuda:1 to itself; with one GPU the policy and
vLLM share cuda:0 (vLLM capped at --gpu-memory-utilization, default 0.3).

Examples:
  uv run python scripts/run_all.py                         # everything
  uv run python scripts/run_all.py --only sft --only ei    # some stages
  uv run python scripts/run_all.py --skip tests --dry-run  # print the plan
  uv run python scripts/run_all.py --force sft_full        # re-run one step even if it finished
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "runs"
LOG_DIR = RUNS / "logs"
STAGES = ["tests", "baseline", "sft", "ei", "grpo", "plots"]
TEST_MODEL_PATH = Path("/data/a5-alignment/models/Qwen2.5-Math-1.5B")  # hard-coded in tests/conftest.py

# (run name, extra args) for each experiment; see the per-script docstrings.
SFT_RUNS = [
    ("sft_n128", ["--num-examples", "128"]),
    ("sft_n256", ["--num-examples", "256"]),
    ("sft_n512", ["--num-examples", "512"]),
    ("sft_full", []),
    ("sft_filtered", ["--data", "data/sft_filtered.jsonl"]),
]
EI_RUNS = [
    ("ei_b512_g4_e1", ["--batch-size", "512", "--group-size", "4", "--sft-epochs", "1"]),
    ("ei_b512_g8_e1", ["--batch-size", "512", "--group-size", "8", "--sft-epochs", "1"]),
    ("ei_b512_g4_e2", ["--batch-size", "512", "--group-size", "4", "--sft-epochs", "2"]),
    ("ei_b1024_g4_e1", ["--batch-size", "1024", "--group-size", "4", "--sft-epochs", "1"]),
]
GRPO_RUNS = [("grpo_default", [])]

app = typer.Typer(add_completion=False)


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts" / script), *args]


def run_step(name: str, cmd: list[str], dry_run: bool, cwd: Path = REPO_ROOT) -> bool:
    print(f"\n=== {name}: {' '.join(cmd)}" + (f"  (in {cwd})" if cwd != REPO_ROOT else ""), flush=True)
    if dry_run:
        return True
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    with open(LOG_DIR / f"{name}.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    mins = (time.time() - start) / 60
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"=== {name}: {status} in {mins:.1f} min (log: {LOG_DIR / f'{name}.log'})", flush=True)
    return proc.returncode == 0


def ensure_model(model: Path, dry_run: bool) -> None:
    if (model / "config.json").exists():
        return
    print(f"Downloading Qwen/Qwen2.5-Math-1.5B to {model}")
    if not dry_run:
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen2.5-Math-1.5B", local_dir=str(model))


def link_test_model(model: Path) -> bool:
    """The unit tests load the model from a fixed path; point it at our copy if we can."""
    if (TEST_MODEL_PATH / "config.json").exists():
        return True
    try:
        TEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if TEST_MODEL_PATH.is_symlink():
            TEST_MODEL_PATH.unlink()
        TEST_MODEL_PATH.symlink_to(model.resolve())
        return True
    except OSError as e:
        print(f"Could not link {TEST_MODEL_PATH} -> {model} ({e}); running a temp copy of tests/ that loads {model}.")
        return False


def copy_tests_with_model_path(model: Path) -> Path:
    """Copy tests/ to a temp dir with the conftest model path swapped for `model`, leaving the repo's tests
    untouched. Returns the dir to run pytest from (fixtures and snapshots resolve relative to the copy)."""
    root = Path(tempfile.mkdtemp(prefix="a2_tests_"))
    shutil.copytree(REPO_ROOT / "tests", root / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    conftest = root / "tests" / "conftest.py"
    text = conftest.read_text()
    assert str(TEST_MODEL_PATH) in text, "tests/conftest.py no longer uses the expected model path"
    conftest.write_text(text.replace(str(TEST_MODEL_PATH), str(model.resolve())))
    return root


def gpu_layout(policy_device: str | None, vllm_device: str | None, gpu_memory_utilization: float | None):
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("WARNING: no CUDA GPU visible; GPU stages will fail.")
    two = n_gpus >= 2
    policy = policy_device or "cuda:0"
    vllm = vllm_device or ("cuda:1" if two else "cuda:0")
    util = gpu_memory_utilization if gpu_memory_utilization is not None else (0.3 if vllm == policy else 0.85)
    print(f"{n_gpus} GPU(s): policy on {policy}, vLLM on {vllm} (gpu_memory_utilization={util})")
    return policy, vllm, util


def print_results() -> None:
    print("\n=== Results")
    keys = ("accuracy", "format_rate", "best_accuracy", "final_accuracy", "initial_val_reward",
            "final_val_reward", "best_val_reward")
    for path in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(path.read_text())
        print(f"{path.parent.name:<20}", {k: round(s[k], 4) for k in keys if k in s})


@app.command()
def main(
    model: str = "models/Qwen2.5-Math-1.5B",
    only: list[str] = typer.Option(None, help=f"Run only these stages: {', '.join(STAGES)}"),
    skip: list[str] = typer.Option(None, help="Skip these stages"),
    force: list[str] = typer.Option(None, help="Re-run these steps (run names) even if summary.json exists"),
    policy_device: str = typer.Option(None, help="Default cuda:0"),
    vllm_device: str = typer.Option(None, help="Default cuda:1 if 2+ GPUs, else cuda:0"),
    gpu_memory_utilization: float = typer.Option(None, help="vLLM memory share; default 0.3 shared, 0.85 own GPU"),
    wandb_project: str = typer.Option(None, help="Passed to the training scripts"),
    keep_going: bool = typer.Option(False, help="Continue with later steps after a failure"),
    dry_run: bool = typer.Option(False, help="Print the plan without running anything"),
):
    for s in (only or []) + (skip or []):
        if s not in STAGES:
            raise typer.BadParameter(f"unknown stage {s!r}; choose from {STAGES}")
    stages = [s for s in STAGES if (not only or s in only) and s not in (skip or [])]
    force = set(force or [])

    os.chdir(REPO_ROOT)
    assert (REPO_ROOT / "data/MATH/validation.jsonl").exists(), "data/MATH is missing; copy data/ onto this machine"
    model_path = Path(model) if Path(model).is_absolute() else REPO_ROOT / model
    ensure_model(model_path, dry_run)
    policy, vllm, util = gpu_layout(policy_device, vllm_device, gpu_memory_utilization)
    train_args = ["--model", str(model_path), "--policy-device", policy, "--vllm-device", vllm,
                  "--gpu-memory-utilization", str(util)]
    if wandb_project:
        train_args += ["--wandb-project", wandb_project]

    # (step name, command, output that marks the step as done or None to always run)
    plan: list[tuple[str, list[str], Path | None]] = []
    test_cwd = REPO_ROOT
    if "tests" in stages:
        if not dry_run and not link_test_model(model_path):
            test_cwd = copy_tests_with_model_path(model_path)
        plan.append(("tests", [sys.executable, "-m", "pytest", "tests/test_sft.py", "tests/test_grpo.py", "-q"], None))
    if "baseline" in stages:
        plan.append(("baseline_r1_zero", py("math_baseline.py", "--model", str(model_path), "--device", vllm),
                     RUNS / "baseline_r1_zero/summary.json"))
        plan.append(("baseline_analysis", py("analyze_baseline.py", "--run", "baseline_r1_zero"),
                     RUNS / "baseline_r1_zero/analysis.json"))
    if "sft" in stages:
        if not (REPO_ROOT / "data/sft_filtered.jsonl").exists():
            plan.append(("filter_sft", py("filter_sft.py"), REPO_ROOT / "data/sft_filtered.jsonl"))
        plan += [(n, py("sft_train.py", *train_args, "--run-name", n, *a), RUNS / n / "summary.json") for n, a in SFT_RUNS]
    if "ei" in stages:
        plan += [(n, py("expert_iteration.py", *train_args, "--run-name", n, *a), RUNS / n / "summary.json")
                 for n, a in EI_RUNS]
    if "grpo" in stages:
        plan += [(n, py("grpo_train.py", *train_args, "--run-name", n, *a), RUNS / n / "summary.json")
                 for n, a in GRPO_RUNS]
    if "plots" in stages:
        cmd = (["uv", "run", "--no-sync", "--with", "matplotlib", "python", "scripts/make_plots.py"]
               if shutil.which("uv") else py("make_plots.py"))
        plan.append(("plots", cmd, None))

    failed = []
    for name, cmd, done_marker in plan:
        if done_marker is not None and done_marker.exists() and name not in force:
            print(f"\n=== {name}: already done ({done_marker.relative_to(REPO_ROOT)}), skipping")
            continue
        ok = run_step(name, cmd, dry_run, test_cwd if name == "tests" else REPO_ROOT)
        if test_cwd != REPO_ROOT and name == "tests":
            shutil.rmtree(test_cwd, ignore_errors=True)
        if not ok:
            failed.append(name)
            if not keep_going:
                break

    if not dry_run:
        print_results()
    if failed:
        print(f"\nFailed: {', '.join(failed)}. Fix and re-run the same command; finished steps are skipped.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
