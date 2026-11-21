"""Policy-gradient methods on the drone delivery environment with TorchRL:
REINFORCE, PPO (clipped) and A2C.

    uv run python -m training.pg_training --algo ppo --sweep --jobs 3
    uv run python -m training.pg_training --algo reinforce --runs 0 3
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from training.common import (QUICK_EVAL_SEEDS, RunLogger, episode_returns, evaluate, make_env,
                             mlp, run_sweep, set_seed)

# ---------------------------------------------------------------- sweeps
REINFORCE_BASE = dict(total_frames=600_000, lr=3e-4, gamma=0.99, hidden=128, episodes_per_update=8,
                      baseline="value", normalize=True, entropy_coef=0.02, critic_lr=1e-3, seed=0)
REINFORCE_SWEEP = [
    dict(REINFORCE_BASE, note="baseline (value baseline, normalised returns)"),
    dict(REINFORCE_BASE, lr=1e-4, note="lower learning rate"),
    dict(REINFORCE_BASE, lr=1e-3, note="higher learning rate"),
    dict(REINFORCE_BASE, baseline="none", note="vanilla REINFORCE, no baseline"),
    dict(REINFORCE_BASE, normalize=False, note="baseline, raw advantages"),
    dict(REINFORCE_BASE, episodes_per_update=2, note="2 episodes per update"),
    dict(REINFORCE_BASE, episodes_per_update=16, note="16 episodes per update"),
    dict(REINFORCE_BASE, entropy_coef=0.0, note="no entropy bonus"),
    dict(REINFORCE_BASE, entropy_coef=0.08, note="strong entropy bonus"),
    dict(REINFORCE_BASE, gamma=0.95, note="shorter horizon (gamma 0.95)"),
]

PPO_BASE = dict(total_frames=600_000, lr=3e-4, gamma=0.99, lmbda=0.95, clip_epsilon=0.2,
                entropy_coef=0.01, critic_coef=0.5, frames_per_batch=2048, epochs=4,
                minibatch=256, hidden=128, max_grad_norm=0.5, seed=0)
PPO_SWEEP = [
    dict(PPO_BASE, note="baseline"),
    dict(PPO_BASE, lr=1e-4, note="lower learning rate"),
    dict(PPO_BASE, lr=1e-3, note="higher learning rate"),
    dict(PPO_BASE, clip_epsilon=0.1, note="tight clip 0.1"),
    dict(PPO_BASE, clip_epsilon=0.3, note="loose clip 0.3"),
    dict(PPO_BASE, entropy_coef=0.0, note="no entropy bonus"),
    dict(PPO_BASE, entropy_coef=0.05, note="strong entropy bonus"),
    dict(PPO_BASE, frames_per_batch=512, minibatch=128, note="short rollouts (512)"),
    dict(PPO_BASE, epochs=10, note="10 epochs per batch"),
    dict(PPO_BASE, lmbda=0.8, gamma=0.995, note="GAE lambda 0.8, gamma 0.995"),
]

A2C_BASE = dict(total_frames=600_000, lr=7e-4, gamma=0.99, lmbda=0.95, entropy_coef=0.01,
                critic_coef=0.5, frames_per_batch=256, hidden=128, max_grad_norm=0.5,
                optimizer="adam", seed=0)
A2C_SWEEP = [
    dict(A2C_BASE, note="baseline"),
    dict(A2C_BASE, lr=2e-4, note="lower learning rate"),
    dict(A2C_BASE, lr=2e-3, note="higher learning rate"),
    dict(A2C_BASE, frames_per_batch=64, note="short rollouts (64 steps)"),
    dict(A2C_BASE, frames_per_batch=1024, note="long rollouts (1024 steps)"),
    dict(A2C_BASE, entropy_coef=0.0, note="no entropy bonus"),
    dict(A2C_BASE, entropy_coef=0.05, note="strong entropy bonus"),
    dict(A2C_BASE, lmbda=1.0, note="Monte Carlo advantage (lambda 1.0)"),
    dict(A2C_BASE, critic_coef=1.0, hidden=256, note="heavier critic, wider net"),
    dict(A2C_BASE, optimizer="rmsprop", gamma=0.95, note="RMSprop, gamma 0.95"),
]

SWEEPS = {"reinforce": REINFORCE_SWEEP, "ppo": PPO_SWEEP, "a2c": A2C_SWEEP}


# ---------------------------------------------------------------- models
def build_actor_critic(env, hidden: int):
    from tensordict.nn import TensorDictModule
    from torchrl.modules import MaskedCategorical, ProbabilisticActor, ValueOperator

    from environment import N_ACTIONS, OBS_DIM

    actor_net = mlp(OBS_DIM, N_ACTIONS, hidden, act=nn.Tanh)
    actor = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys={"logits": "logits", "mask": "action_mask"}, spec=env.action_spec,
        distribution_class=MaskedCategorical, return_log_prob=True,
    )
    critic_net = mlp(OBS_DIM, 1, hidden, act=nn.Tanh)
    critic = ValueOperator(critic_net, in_keys=["observation"])
    return actor_net, actor, critic


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


# ---------------------------------------------------------------- PPO / A2C
def train_ppo_a2c(hp: dict, run_name: str, algo: str) -> dict:
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
    from torchrl.objectives import A2CLoss, ClipPPOLoss
    from torchrl.objectives.value import GAE

    set_seed(hp["seed"])
    env = make_env(seed=hp["seed"])
    actor_net, actor, critic = build_actor_critic(env, hp["hidden"])
    adv = GAE(gamma=hp["gamma"], lmbda=hp["lmbda"], value_network=critic, average_gae=True)
    if algo == "ppo":
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=hp["clip_epsilon"], entropy_bonus=True,
                              entropy_coeff=hp["entropy_coef"], critic_coeff=hp["critic_coef"],
                              loss_critic_type="smooth_l1")
    else:
        loss_fn = A2CLoss(actor, critic, entropy_bonus=hp["entropy_coef"] > 0,
                          entropy_coeff=hp["entropy_coef"] if hp["entropy_coef"] > 0 else None,
                          critic_coeff=hp["critic_coef"], loss_critic_type="smooth_l1")
    if hp.get("optimizer") == "rmsprop":
        optim = torch.optim.RMSprop(loss_fn.parameters(), lr=hp["lr"], alpha=0.99, eps=1e-5)
    else:
        optim = torch.optim.Adam(loss_fn.parameters(), lr=hp["lr"])

    fpb = hp["frames_per_batch"]
    collector = Collector(env, actor, frames_per_batch=fpb, total_frames=hp["total_frames"])
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(fpb), sampler=SamplerWithoutReplacement(),
                                batch_size=hp.get("minibatch", fpb))

    log = RunLogger(algo, run_name, hp)
    frames, recent = 0, []
    acc: dict[str, list] = {}
    next_log = 5_000
    for data in collector:
        frames += data.numel()
        recent += episode_returns(data)
        epochs = hp.get("epochs", 1)
        for _ in range(epochs):
            with torch.no_grad():
                adv(data)
            if algo == "ppo":
                rb.extend(data.reshape(-1))
                batches = [rb.sample() for _ in range(max(fpb // hp["minibatch"], 1))]
            else:
                batches = [data.reshape(-1)]
            for b in batches:
                out = loss_fn(b)
                total = out["loss_objective"] + out["loss_critic"]
                if "loss_entropy" in out.keys():
                    total = total + out["loss_entropy"]
                optim.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(loss_fn.parameters(), hp["max_grad_norm"])
                optim.step()
                for k in ("loss_objective", "loss_critic", "entropy", "clip_fraction", "kl_approx"):
                    if k in out.keys():
                        acc.setdefault(k, []).append(out[k].detach().mean().item())
        if frames >= next_log:
            next_log += 5_000
            row = dict(frames=frames, episodes_done=len(recent), train_return=_mean(recent[-20:]),
                       eval_return=evaluate(actor_net, QUICK_EVAL_SEEDS)["return_mean"])
            row.update({k: _mean(v) for k, v in acc.items()})
            log.log(**row)
            acc = {}
    collector.shutdown()
    return log.finish(actor_net, {"final_train_return": _mean(recent[-50:]), "episodes": len(recent)})


# ---------------------------------------------------------------- REINFORCE
def train_reinforce(hp: dict, run_name: str) -> dict:
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from torchrl.objectives import ReinforceLoss

    set_seed(hp["seed"])
    env = make_env(seed=hp["seed"])
    actor_net, actor, critic = build_actor_critic(env, hp["hidden"])
    loss_fn = ReinforceLoss(actor, critic, loss_critic_type="smooth_l1")
    optim = torch.optim.Adam([
        {"params": list(actor_net.parameters()), "lr": hp["lr"]},
        {"params": list(critic.parameters()), "lr": hp["critic_lr"]},
    ])
    log = RunLogger("reinforce", run_name, hp)
    frames, recent = 0, []
    acc: dict[str, list] = {}
    next_log = 5_000
    while frames < hp["total_frames"]:
        episodes = []
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            for _ in range(hp["episodes_per_update"]):
                ep = env.rollout(max_steps=500, policy=actor, break_when_any_done=True)
                episodes.append(ep)
        for ep in episodes:
            r = ep["next", "reward"].squeeze(-1)
            g = torch.zeros_like(r)
            running = 0.0
            for t in range(len(r) - 1, -1, -1):  # Monte Carlo discounted return
                running = r[t] + hp["gamma"] * running
                g[t] = running
            ep["value_target"] = g.unsqueeze(-1)
            frames += ep.numel()
            recent.append(float(ep["next", "episode_reward"][-1].item()))
        td = torch.cat(episodes, 0)
        g = td["value_target"]
        if hp["baseline"] == "value":
            with torch.no_grad():
                a = g - critic(td.select("observation").clone())["state_value"]
        else:
            a = g.clone()
        if hp["normalize"] and a.numel() > 1:
            a = (a - a.mean()) / (a.std() + 1e-8)
        td["advantage"] = a
        out = loss_fn(td)
        dist = actor.get_dist(td)
        entropy = dist.entropy().mean()
        total = out["loss_actor"] - hp["entropy_coef"] * entropy
        if hp["baseline"] == "value":
            total = total + out["loss_value"]
        optim.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(list(actor_net.parameters()) + list(critic.parameters()), 1.0)
        optim.step()
        acc.setdefault("loss_objective", []).append(out["loss_actor"].item())
        acc.setdefault("loss_critic", []).append(out["loss_value"].item())
        acc.setdefault("entropy", []).append(entropy.item())
        if frames >= next_log:
            next_log += 5_000
            row = dict(frames=frames, episodes_done=len(recent), train_return=_mean(recent[-20:]),
                       eval_return=evaluate(actor_net, QUICK_EVAL_SEEDS)["return_mean"])
            row.update({k: _mean(v) for k, v in acc.items()})
            log.log(**row)
            acc = {}
    _ = TensorDict  # keep import used for type clarity
    return log.finish(actor_net, {"final_train_return": _mean(recent[-50:]), "episodes": len(recent)})


TRAINERS = {
    "reinforce": train_reinforce,
    "ppo": lambda hp, name: train_ppo_a2c(hp, name, "ppo"),
    "a2c": lambda hp, name: train_ppo_a2c(hp, name, "a2c"),
}


def _train_ppo(hp, name):
    return train_ppo_a2c(hp, name, "ppo")


def _train_a2c(hp, name):
    return train_ppo_a2c(hp, name, "a2c")


PICKLABLE = {"reinforce": train_reinforce, "ppo": _train_ppo, "a2c": _train_a2c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(SWEEPS), required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--runs", type=int, nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--frames", type=int, default=None)
    a = ap.parse_args()
    run_sweep(PICKLABLE[a.algo], a.algo, SWEEPS[a.algo], a.jobs, a.frames,
              None if a.sweep else (a.runs or [0]))


if __name__ == "__main__":
    main()
