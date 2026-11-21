"""DQN on the drone delivery environment with TorchRL.

    uv run python -m training.dqn_training --sweep --jobs 3   # all 10 runs
    uv run python -m training.dqn_training --runs 0 4         # selected runs
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from training.common import (QUICK_EVAL_SEEDS, RunLogger, episode_returns, evaluate, make_env,
                             mlp, run_sweep, set_seed)

BASE = dict(
    total_frames=600_000,
    lr=5e-4,
    gamma=0.99,
    hidden=128,
    buffer_size=50_000,
    batch_size=128,
    frames_per_batch=64,
    updates_per_batch=4,
    learning_starts=2_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_frames=60_000,
    target_update="hard",
    target_interval=500,  # gradient steps between hard target copies
    tau=0.005,  # used when target_update == "soft"
    double_dqn=False,
    loss="l2",
    seed=0,
)

# Ten runs, each changing one or two DQN-specific controls from the baseline.
SWEEP = [
    dict(BASE, note="baseline"),
    dict(BASE, lr=1e-4, note="lower learning rate"),
    dict(BASE, lr=2e-3, note="higher learning rate"),
    dict(BASE, gamma=0.95, note="shorter horizon (gamma 0.95)"),
    dict(BASE, gamma=0.999, note="longer horizon (gamma 0.999)"),
    dict(BASE, buffer_size=5_000, batch_size=32, note="small replay, small batch"),
    dict(BASE, eps_frames=150_000, eps_end=0.1, note="slow exploration decay"),
    dict(BASE, eps_frames=15_000, eps_end=0.01, note="fast exploration decay"),
    dict(BASE, target_update="soft", tau=0.01, double_dqn=True, note="double DQN + soft target"),
    dict(BASE, target_interval=50, loss="smooth_l1", hidden=256, note="frequent target copy, Huber, wider net"),
]


def train(hp: dict, run_name: str) -> dict:
    from tensordict.nn import TensorDictSequential
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.modules import EGreedyModule, QValueActor
    from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

    from environment import N_ACTIONS, OBS_DIM

    set_seed(hp["seed"])
    env = make_env(seed=hp["seed"])
    net = mlp(OBS_DIM, N_ACTIONS, hp["hidden"])
    qnet = QValueActor(net, in_keys=["observation"], spec=env.action_spec, action_mask_key="action_mask")
    explore = EGreedyModule(env.action_spec, eps_init=hp["eps_start"], eps_end=hp["eps_end"],
                            annealing_num_steps=hp["eps_frames"], action_mask_key="action_mask")
    policy = TensorDictSequential(qnet, explore)

    loss_fn = DQNLoss(qnet, action_space=env.action_spec, delay_value=True,
                      double_dqn=hp["double_dqn"], loss_function=hp["loss"])
    loss_fn.make_value_estimator(gamma=hp["gamma"])
    if hp["target_update"] == "soft":
        target = SoftUpdate(loss_fn, tau=hp["tau"])
    else:
        target = HardUpdate(loss_fn, value_network_update_interval=hp["target_interval"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=hp["lr"])

    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(hp["buffer_size"]), batch_size=hp["batch_size"])
    collector = Collector(env, policy, frames_per_batch=hp["frames_per_batch"],
                          total_frames=hp["total_frames"], init_random_frames=hp["learning_starts"])

    log = RunLogger("dqn", run_name, hp)
    frames, recent, losses, qvals = 0, [], [], []
    next_log = 5_000
    for data in collector:
        frames += data.numel()
        recent += episode_returns(data)
        rb.extend(data.reshape(-1))
        explore.step(data.numel())
        if len(rb) >= hp["learning_starts"]:
            for _ in range(hp["updates_per_batch"]):
                batch = rb.sample()
                out = loss_fn(batch)
                optim.zero_grad()
                out["loss"].backward()
                nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optim.step()
                target.step()
                losses.append(out["loss"].item())
                qvals.append(batch["chosen_action_value"].mean().item() if "chosen_action_value" in batch.keys() else float("nan"))
        if frames >= next_log:
            next_log += 5_000
            quick = evaluate(net, QUICK_EVAL_SEEDS)["return_mean"]
            log.log(frames=frames, episodes_done=len(recent),
                    train_return=float(sum(recent[-20:]) / max(len(recent[-20:]), 1)) if recent else None,
                    eval_return=quick, loss=float(sum(losses) / max(len(losses), 1)) if losses else None,
                    q_mean=float(sum(qvals) / max(len(qvals), 1)) if qvals else None,
                    epsilon=float(explore.eps))
            losses, qvals = [], []
    collector.shutdown()
    final = float(sum(recent[-50:]) / max(len(recent[-50:]), 1)) if recent else None
    return log.finish(net, {"final_train_return": final, "episodes": len(recent)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run all 10 configurations")
    ap.add_argument("--runs", type=int, nargs="*", help="run only these configuration indices")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--frames", type=int, default=None, help="override total frames (smoke tests)")
    a = ap.parse_args()
    run_sweep(train, "dqn", SWEEP, a.jobs, a.frames, None if a.sweep else (a.runs or [0]))


if __name__ == "__main__":
    main()
