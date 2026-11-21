"""Turn the sweep logs into the report's figures and tables.

    uv run python -m training.analyze

Writes assets/plots/*.png, assets/tables/*.md and logs/baselines.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment import DroneDeliveryEnv, heuristic_action  # noqa: E402
from training.common import EVAL_CONFIGS, EVAL_SEEDS, LOGS, ROOT  # noqa: E402

PLOTS = ROOT / "assets" / "plots"
TABLES = ROOT / "assets" / "tables"
ALGOS = ["dqn", "ppo", "a2c", "reinforce"]
NAME = {"dqn": "DQN", "ppo": "PPO", "a2c": "A2C", "reinforce": "REINFORCE"}
# categorical slots 1-4, fixed per algorithm across every figure (validated:
# adjacent CVD dE >= 9.1; aqua/yellow sit under 3:1 so lines are direct-labelled)
COLOR = {"dqn": "#2a78d6", "ppo": "#eb6834", "a2c": "#1baf7a", "reinforce": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "lines.linewidth": 2,
})


# ------------------------------------------------------------------ loading
def load(algo: str) -> dict[str, dict]:
    runs = {}
    for d in sorted((LOGS / algo).glob("run*")):
        if (d / "summary.json").exists() and (d / "metrics.csv").exists():
            try:
                m = pd.read_csv(d / "metrics.csv")
            except pd.errors.EmptyDataError:
                continue  # unfinished or smoke-test run
            if len(m) < 3:
                continue
            runs[d.name] = {"summary": json.loads((d / "summary.json").read_text()), "metrics": m}
    return runs


def best_run(runs: dict) -> str:
    return max(runs, key=lambda r: runs[r]["summary"]["eval"]["default"]["return_mean"])


def baselines() -> dict:
    path = LOGS / "baselines.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    rng = np.random.default_rng(0)
    for name, pol in {"heuristic": lambda e, o: heuristic_action(e),
                      "random": lambda e, o: int(rng.choice(np.flatnonzero(o["action_mask"]))),
                      "hold": lambda e, o: 0}.items():
        out[name] = {}
        for cfg in EVAL_CONFIGS:
            rets, served, crashed = [], [], []
            for s in EVAL_SEEDS:
                env = DroneDeliveryEnv(cfg)
                obs, _ = env.reset(seed=s)
                done = False
                while not done:
                    obs, _, te, tr, _ = env.step(pol(env, obs))
                    done = te or tr
                rets.append(env.total_reward)
                st = env.stats.as_dict()
                served.append(st["service_rate"])
                crashed.append(st["crashed"])
            out[name][cfg] = {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
                              "service_rate": float(np.mean(served)), "crashed": float(np.mean(crashed))}
    path.write_text(json.dumps(out, indent=2))
    return out


def smooth(y: pd.Series, k: int = 3) -> pd.Series:
    return y.rolling(k, min_periods=1, center=True).mean()


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK, fontweight="bold")
    ax.plot([x], [y], "o", ms=5, color=color, mec="white", mew=1.5, zorder=5)


# ------------------------------------------------------------------ figures
def fig_reward_curves(data, base):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, col, title in ((axes[0], "train_return", "Training episodes (exploring policy)"),
                           (axes[1], "eval_return", "Greedy policy on 5 held-out seeds")):
        ends = []
        for algo in ALGOS:
            if not data[algo]:
                continue
            b = best_run(data[algo])
            m = data[algo][b]["metrics"].dropna(subset=[col])
            x, y = m["frames"] / 1000, smooth(m[col], 5)
            ax.plot(x, y, color=COLOR[algo], label=f"{NAME[algo]} ({b})")
            ax.plot([x.iloc[-1]], [y.iloc[-1]], "o", ms=5, color=COLOR[algo], mec="white", mew=1.5, zorder=5)
            ends.append([y.iloc[-1], x.iloc[-1], algo])
        # direct labels, nudged apart so they never collide
        ends.sort(key=lambda e: e[0])
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.065
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        over = ends[-1][0] - (hi - gap * 0.5) if ends else 0
        for ly, lx, algo in ends:
            ax.annotate(NAME[algo], (lx, ly - max(over, 0)), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK, fontweight="bold", annotation_clip=False)
        h = base["heuristic"]["default"]["return_mean"]
        ax.axhline(h, color=INK2, lw=1.2, ls=(0, (4, 3)), label=f"hand-written dispatcher ({h:+.0f})")
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps (thousands)")
    axes[0].set_ylabel("shift return")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Return per shift, best run of each algorithm", x=0.06, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.84)
    fig.savefig(PLOTS / "reward_curves.png")
    plt.close(fig)


def fig_sweeps(data, base):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        runs = data[algo]
        if not runs:
            continue
        b = best_run(runs)
        for r, d in runs.items():
            m = d["metrics"]
            if r != b:
                ax.plot(m["frames"] / 1000, smooth(m["eval_return"]), color="#c9c8c2", lw=1.2)
        m = runs[b]["metrics"]
        ax.plot(m["frames"] / 1000, smooth(m["eval_return"]), color=COLOR[algo], lw=2.2)
        note = runs[b]["summary"]["hparams"].get("note", "")
        ax.set_title(f"{NAME[algo]}: 10 runs (best: {b}, {note})", loc="left", fontsize=10)
        ax.axhline(base["heuristic"]["default"]["return_mean"], color=INK2, lw=1, ls=(0, (4, 3)))
    for ax in axes[1]:
        ax.set_xlabel("environment steps (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("greedy return, held-out seeds")
    fig.suptitle("Every hyperparameter run (grey) with the best highlighted; dashed = hand-written dispatcher",
                 x=0.06, ha="left", fontweight="bold")
    fig.savefig(PLOTS / "sweep_curves.png")
    plt.close(fig)


def fig_dqn_objective(data):
    runs = data["dqn"]
    if not runs:
        return
    b = best_run(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for r, d in runs.items():
        m = d["metrics"].dropna(subset=["loss"])
        kw = dict(color=COLOR["dqn"], lw=2.2) if r == b else dict(color="#c9c8c2", lw=1)
        axes[0].plot(m["frames"] / 1000, m["loss"], **kw)
        axes[1].plot(m["frames"] / 1000, m["q_mean"], **kw)
    axes[0].set_yscale("log")
    axes[0].set_title("TD loss (log scale)", loc="left")
    axes[1].set_title("Mean Q-value of chosen actions (reward / 10 units)", loc="left")
    for ax in axes:
        ax.set_xlabel("environment steps (thousands)")
    fig.suptitle(f"DQN objective, best run {b} in blue, other 9 runs grey", x=0.06, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.82)
    fig.savefig(PLOTS / "dqn_objective.png")
    plt.close(fig)


def fig_pg_entropy(data):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, algo in zip(axes, ["ppo", "a2c", "reinforce"]):
        runs = data[algo]
        if not runs:
            continue
        b = best_run(runs)
        for r, d in runs.items():
            if "entropy" not in d["metrics"].columns:
                continue  # entropy_coef 0 in A2C turns TorchRL's entropy term (and its log) off
            m = d["metrics"].dropna(subset=["entropy"])
            kw = dict(color=COLOR[algo], lw=2.2) if r == b else dict(color="#c9c8c2", lw=1)
            ax.plot(m["frames"] / 1000, smooth(m["entropy"]), **kw)
        ax.set_title(f"{NAME[algo]} (best {b})", loc="left")
        ax.set_xlabel("environment steps (thousands)")
    axes[0].set_ylabel("policy entropy (nats)")
    fig.suptitle("Policy entropy: how quickly each method stops exploring", x=0.06, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.8)
    fig.savefig(PLOTS / "pg_entropy.png")
    plt.close(fig)


def frames_to_reach(m: pd.DataFrame, level: float) -> float | None:
    y = smooth(m["eval_return"])
    hit = m["frames"][y >= level]
    return float(hit.iloc[0]) if len(hit) else None


def fig_convergence(data, base):
    level = base["heuristic"]["default"]["return_mean"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    max_f = max((d["metrics"]["frames"].max() for a in ALGOS for d in data[a].values()), default=600_000)
    rows = []
    for i, algo in enumerate(ALGOS):
        for r, d in data[algo].items():
            f = frames_to_reach(d["metrics"], level)
            rows.append((algo, r, f))
            if f is None:
                ax.plot(max_f / 1000 * 1.04, i, "o", ms=8, mfc="white", mec=COLOR[algo], mew=2)
            else:
                ax.plot(f / 1000, i, "o", ms=8, color=COLOR[algo], mec="white", mew=1.5)
    ax.set_yticks(range(len(ALGOS)), [NAME[a] for a in ALGOS])
    ax.invert_yaxis()
    ax.axvline(max_f / 1000, color=MUTED, lw=1)
    ax.set_xlabel("environment steps (thousands) until the greedy policy first matches the dispatcher "
                  f"({level:+.0f})")
    ax.set_title("Convergence speed: one dot per hyperparameter run (hollow = never reached it in 600k steps)",
                 loc="left")
    fig.savefig(PLOTS / "convergence.png")
    plt.close(fig)
    return rows


def fig_generalisation(data, base):
    fig, ax = plt.subplots(figsize=(10, 4))
    groups = list(EVAL_CONFIGS)
    labels = {"default": "Normal day\n(training conditions)", "busy": "Busy day\n(+50% requests)",
              "windy": "Windy day\n(+60% wind)"}
    series = [(NAME[a], COLOR[a], lambda c, a=a: data[a][best_run(data[a])]["summary"]["eval"][c])
              for a in ALGOS if data[a]]
    series.append(("Hand-written dispatcher", "#8a8984", lambda c: base["heuristic"][c]))
    n = len(series)
    w = 0.8 / n
    for j, (name, color, get) in enumerate(series):
        xs = np.arange(len(groups)) + (j - (n - 1) / 2) * w
        ys = [get(c)["return_mean"] for c in groups]
        err = [get(c)["return_std"] / np.sqrt(len(EVAL_SEEDS)) for c in groups]
        ax.bar(xs, ys, width=w - 0.02, color=color, label=name, edgecolor="white", linewidth=1)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(groups)), [labels[g] for g in groups])
    ax.set_ylabel("mean shift return (20 unseen seeds)")
    ax.legend(ncols=n, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8.5)
    ax.set_title("Generalisation: best run of each method on seeds and conditions never trained on",
                 loc="left", pad=28)
    ax.grid(axis="x", visible=False)
    fig.savefig(PLOTS / "generalisation.png")
    plt.close(fig)


# ------------------------------------------------------------------ tables
DQN_COLS = ["lr", "gamma", "buffer_size", "batch_size", "eps_end", "eps_frames", "target_update",
            "target_interval", "double_dqn", "loss", "hidden"]
PPO_COLS = ["lr", "gamma", "lmbda", "clip_epsilon", "entropy_coef", "frames_per_batch", "epochs", "minibatch"]
A2C_COLS = ["lr", "gamma", "lmbda", "entropy_coef", "critic_coef", "frames_per_batch", "optimizer", "hidden"]
RF_COLS = ["lr", "gamma", "baseline", "normalize", "episodes_per_update", "entropy_coef"]
COLS = {"dqn": DQN_COLS, "ppo": PPO_COLS, "a2c": A2C_COLS, "reinforce": RF_COLS}


def tables(data, base, conv_rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    conv = {(a, r): f for a, r, f in conv_rows}
    level = base["heuristic"]["default"]["return_mean"]
    for algo in ALGOS:
        runs = data[algo]
        if not runs:
            continue
        b = best_run(runs)
        head = ["Run", "Change from baseline", *COLS[algo], "Return (unseen)", "Busy", "Windy",
                "Service", "Steps to dispatcher level"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r, d in runs.items():
            hp, ev = d["summary"]["hparams"], d["summary"]["eval"]
            f = conv.get((algo, r))
            cells = [("**" + r + "**") if r == b else r, hp.get("note", "")]
            cells += [str(hp.get(c, "")) for c in COLS[algo]]
            cells += [f"{ev['default']['return_mean']:+.1f} ± {ev['default']['return_std']:.1f}",
                      f"{ev['busy']['return_mean']:+.1f}", f"{ev['windy']['return_mean']:+.1f}",
                      f"{ev['default']['service_rate']:.0%}", f"{f / 1000:.0f}k" if f else "never"]
            lines.append("| " + " | ".join(cells) + " |")
        (TABLES / f"{algo}_sweep.md").write_text(
            f"# {NAME[algo]} sweep (10 runs, 600k steps each, same environment and training seed)\n\n"
            f"Dispatcher level = {level:+.1f}. Returns are means over 20 unseen seeds.\n\n" + "\n".join(lines) + "\n")

    head = ["Method", "Best run", "Normal day", "Busy day", "Windy day", "Service rate", "Crash rate",
            "Emergencies delivered", "Emergencies missed", "Mean emergency wait (min)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for algo in ALGOS:
        if not data[algo]:
            continue
        b = best_run(data[algo])
        ev = data[algo][b]["summary"]["eval"]
        d = ev["default"]
        lines.append("| " + " | ".join([
            NAME[algo], f"{b} ({data[algo][b]['summary']['hparams'].get('note', '')})",
            f"{d['return_mean']:+.1f} ± {d['return_std']:.1f}", f"{ev['busy']['return_mean']:+.1f}",
            f"{ev['windy']['return_mean']:+.1f}", f"{d['service_rate']:.0%}", f"{d['crashed']:.0%}",
            f"{d['delivered_emergency']:.1f}", f"{d['missed_emergency']:.1f}",
            f"{d['mean_emergency_wait']:.1f}"]) + " |")
    for name, key in (("Hand-written dispatcher", "heuristic"), ("Always hold", "hold"), ("Random (masked)", "random")):
        bb = base[key]
        lines.append(f"| {name} | n/a | {bb['default']['return_mean']:+.1f} ± {bb['default']['return_std']:.1f} | "
                     f"{bb['busy']['return_mean']:+.1f} | {bb['windy']['return_mean']:+.1f} | "
                     f"{bb['default']['service_rate']:.0%} | {bb['default']['crashed']:.0%} | | | |")
    (TABLES / "headline.md").write_text("# Best run of each method vs baselines (20 unseen seeds per condition)\n\n"
                                        + "\n".join(lines) + "\n")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    data = {a: load(a) for a in ALGOS}
    base = baselines()
    fig_reward_curves(data, base)
    fig_sweeps(data, base)
    fig_dqn_objective(data)
    fig_pg_entropy(data)
    conv = fig_convergence(data, base)
    fig_generalisation(data, base)
    tables(data, base, conv)
    for a in ALGOS:
        if data[a]:
            b = best_run(data[a])
            print(f"{NAME[a]:>10}: {len(data[a])} runs, best {b} "
                  f"{data[a][b]['summary']['eval']['default']['return_mean']:+.1f}")
    print(f"dispatcher {base['heuristic']['default']['return_mean']:+.1f}  random {base['random']['default']['return_mean']:+.1f}")
    print(f"wrote {PLOTS} and {TABLES}")


if __name__ == "__main__":
    main()
