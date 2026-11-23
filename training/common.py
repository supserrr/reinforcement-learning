"""Shared pieces for every TorchRL training script: environment construction,
networks, evaluation on unseen seeds, logging and the sweep runner."""

from __future__ import annotations

import copy
import csv
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MODELS = ROOT / "models"

# Held-out seeds: training uses seeds below 10_000 and never sees these.
# Validation: every run is scored on these, and the best run of each sweep is
# picked by its score here, so these scores flatter the winner a little.
EVAL_SEEDS = list(range(10_000, 10_020))
# Checked every 5,000 steps during training: the learning curves, and the
# checkpoint each run keeps (its best score on these seeds).
QUICK_EVAL_SEEDS = list(range(20_000, 20_020))
# Test: used only once, after the best runs are picked, for the final comparison
# (training/analyze.py). Nothing is chosen or tuned on them.
TEST_SEEDS = list(range(30_000, 30_200))
EVAL_CONFIGS = ("default", "busy", "windy")


REWARD_SCALE = 0.1  # learners see reward / 10; logged returns stay in raw units


def make_env(seed: int = 0, config: str = "default", reward_scale: float = REWARD_SCALE):
    """The custom Gymnasium env wrapped for TorchRL: a step counter, an
    episode-return tracker on the raw reward, then reward scaling for the
    learner (the crash penalty is ~-120, which swamps value targets)."""
    from torchrl.envs import (DTypeCastTransform, GymWrapper, RewardScaling, RewardSum, StepCounter,
                              TransformedEnv)

    from environment import DroneDeliveryEnv

    base = GymWrapper(DroneDeliveryEnv(config), categorical_action_encoding=True)
    env = TransformedEnv(base)
    env.append_transform(DTypeCastTransform(torch.int8, torch.bool, in_keys=["action_mask"]))
    env.append_transform(StepCounter())
    env.append_transform(RewardSum())
    env.append_transform(RewardScaling(loc=0.0, scale=reward_scale))
    env.set_seed(seed)
    return env


def mlp(in_dim: int, out_dim: int, hidden: int, depth: int = 2, act=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def greedy_action(net: nn.Module, obs: dict) -> int:
    """Highest-scoring action the flight controller allows."""
    scores = net(torch.as_tensor(obs["observation"]).unsqueeze(0)).squeeze(0)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
    return int(torch.where(mask, scores, torch.finfo(scores.dtype).min).argmax().item())


def best_model_dir(algo: str) -> Path:
    return MODELS / ("dqn" if algo == "dqn" else "pg") / f"{algo}_best"


def load_best_net(algo: str) -> nn.Module:
    """The best sweep run of an algorithm (models/<dqn|pg>/<algo>_best) as a
    network mapping observation -> action scores."""
    from environment import N_ACTIONS, OBS_DIM

    d = best_model_dir(algo)
    hp = json.loads((d / "hparams.json").read_text())
    net = mlp(OBS_DIM, N_ACTIONS, hp["hidden"], act=nn.ReLU if algo == "dqn" else nn.Tanh)
    net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    net.eval()
    return net


@torch.no_grad()
def evaluate(net: nn.Module, seeds=EVAL_SEEDS, config: str = "default") -> dict:
    """Greedy rollouts of a network that maps observation -> action scores
    (Q-values or logits) on seeds the agent never trained on."""
    from environment import DroneDeliveryEnv

    returns, stats = [], []
    for s in seeds:
        env = DroneDeliveryEnv(config)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            a = greedy_action(net, obs)
            obs, _, te, tr, _ = env.step(a)
            done = te or tr
        returns.append(env.total_reward)
        stats.append(env.stats.as_dict())
    out = {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns))}
    for k in ("delivered_emergency", "delivered_routine", "missed_emergency", "missed_routine",
              "service_rate", "crashed", "stranded", "wasted_trips", "invalid_actions",
              "recharges", "km_flown", "mean_emergency_wait"):
        out[k] = float(np.mean([st[k] for st in stats]))
    return out


def evaluate_all(net: nn.Module) -> dict:
    return {cfg: evaluate(net, EVAL_SEEDS, cfg) for cfg in EVAL_CONFIGS}


class RunLogger:
    """Writes one metrics.csv per run plus a summary.json at the end, and
    keeps the run's best checkpoint on the QUICK_EVAL_SEEDS."""

    def __init__(self, algo: str, run_name: str, hparams: dict):
        self.dir = LOGS / algo / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.algo, self.run_name, self.hparams = algo, run_name, hparams
        self.rows: list[dict] = []
        self.best: tuple[float, int, dict] | None = None
        self.t0 = time.time()
        (self.dir / "hparams.json").write_text(json.dumps(hparams, indent=2))

    def log(self, **row):
        row["wall_s"] = round(time.time() - self.t0, 1)
        self.rows.append(row)

    def checkpoint(self, net: nn.Module, score: float, frames: int):
        """Remember the network if it scored its best so far on the QUICK_EVAL_SEEDS."""
        if self.best is None or score > self.best[0]:
            self.best = (score, frames, copy.deepcopy(net.state_dict()))

    def finish(self, net: nn.Module, extra: dict | None = None) -> dict:
        if self.best is not None:  # evaluate and save the best checkpoint, not the last weights
            net.load_state_dict(self.best[2])
            extra = {**(extra or {}), "checkpoint_frames": self.best[1], "checkpoint_return": self.best[0]}
        keys: list[str] = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        evals = evaluate_all(net)
        summary = {"algo": self.algo, "run": self.run_name, "hparams": self.hparams,
                   "wall_s": round(time.time() - self.t0, 1), "eval": evals, **(extra or {})}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        model_dir = MODELS / ("dqn" if self.algo == "dqn" else "pg") / self.algo / self.run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), model_dir / "model.pt")
        (model_dir / "hparams.json").write_text(json.dumps(self.hparams, indent=2))
        return summary


def episode_returns(data) -> list[float]:
    done = data["next", "done"].squeeze(-1)
    if done.any():
        return data["next", "episode_reward"][done].flatten().tolist()
    return []


def run_name(i: int, seed: int | None = None) -> str:
    """run07 for the sweep (training seed 0); seed1_run07 for a repeat of
    configuration 7 on another training seed, which the sweep table ignores."""
    return f"run{i:02d}" if seed is None else f"seed{seed}_run{i:02d}"


def run_sweep(train_fn, algo: str, configs: list[dict], jobs: int, frames: int | None, only: list[int] | None,
              seed: int | None = None):
    """Run every hyperparameter configuration, in parallel processes, then
    write the sweep table and promote the best run to models/<algo>/best.
    With a seed, rerun the chosen configurations on that training seed."""
    idx = only if only else list(range(len(configs)))
    todo = []
    for i in idx:
        hp = dict(configs[i])
        if frames:
            hp["total_frames"] = frames
        if seed is not None:
            hp["seed"] = seed
        todo.append((i, hp))
    results = []
    if jobs <= 1:
        for i, hp in todo:
            results.append(train_fn(hp, run_name(i, seed)))
            _print_result(results[-1])
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_worker, train_fn, hp, run_name(i, seed)): i for i, hp in todo}
            for f in as_completed(futs):
                results.append(f.result())
                _print_result(results[-1])
    write_sweep_table(algo)
    return results


def _worker(train_fn, hp, name):
    torch.set_num_threads(1)
    return train_fn(hp, name)


def _print_result(s: dict):
    e = s["eval"]["default"]
    print(f"[{s['algo']}] {s['run']}: eval return {e['return_mean']:7.1f} ± {e['return_std']:5.1f}  "
          f"service {e['service_rate']:.2f}  crashes {e['crashed']:.2f}  ({s['wall_s']:.0f}s)", flush=True)


def write_sweep_table(algo: str):
    """logs/<algo>/sweep_summary.csv: one row per run, hyperparameters and
    unseen-seed results side by side; copies the best model to best.pt."""
    rows = []
    for p in sorted((LOGS / algo).glob("run[0-9]*/summary.json")):
        s = json.loads(p.read_text())
        row = {"run": s["run"], **s["hparams"]}
        for cfg, ev in s["eval"].items():
            row[f"{cfg}_return"] = round(ev["return_mean"], 2)
            row[f"{cfg}_return_std"] = round(ev["return_std"], 2)
        d = s["eval"]["default"]
        row["service_rate"] = round(d["service_rate"], 3)
        row["crash_rate"] = round(d["crashed"], 3)
        row["final_train_return"] = s.get("final_train_return")
        row["wall_s"] = s["wall_s"]
        rows.append(row)
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(LOGS / algo / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["default_return"])
    sub = "dqn" if algo == "dqn" else "pg"
    src = MODELS / sub / algo / best["run"]
    dst = MODELS / sub / f"{algo}_best"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "hparams.json"):
        if (src / name).exists():
            (dst / name).write_bytes((src / name).read_bytes())
    (dst / "source_run.txt").write_text(f"{best['run']}  default_return={best['default_return']}\n")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
