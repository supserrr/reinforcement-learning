"""Turn the sweep logs into the report's figures and tables.

    uv run python -m training.analyze             # reuses logs/test_results.json if still current
    uv run python -m training.analyze --retest    # re-run the test evaluation

Writes assets/plots/*.png, assets/tables/*.md, logs/baselines.json and
logs/test_results.json.

The best run of each algorithm is picked on the 20 validation seeds, so its
validation score flatters it. The final comparison (headline.md and
generalisation.png) therefore uses 200 test seeds that neither training nor
run selection ever saw.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment import DroneDeliveryEnv, heuristic_action  # noqa: E402
from training.common import (EVAL_CONFIGS, EVAL_SEEDS, LOGS, ROOT, TEST_SEEDS,  # noqa: E402
                             best_model_dir, greedy_action, load_best_net)

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
    """The sweep runs (run00..run09, training seed 0); seed repeats are read by seed_check."""
    runs = {}
    for d in sorted((LOGS / algo).glob("run[0-9]*")):
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


def rollout(policy, seeds, config: str) -> dict:
    """Play one shift per seed with policy(env, obs) -> action and summarise."""
    rets, stats = [], []
    for s in seeds:
        env = DroneDeliveryEnv(config)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, _, te, tr, _ = env.step(policy(env, obs))
            done = te or tr
        rets.append(env.total_reward)
        stats.append(env.stats.as_dict())
    out = {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
           "return_sem": float(np.std(rets) / np.sqrt(len(rets)))}
    for k in ("service_rate", "crashed", "stranded", "delivered_emergency", "missed_emergency", "delivered_routine",
              "missed_routine", "wasted_trips", "recharges", "mean_emergency_wait"):
        out[k] = float(np.mean([st[k] for st in stats]))
    out["returns"] = [round(float(r), 2) for r in rets]
    return out


def baseline_policies() -> dict:
    rng = np.random.default_rng(0)
    return {"heuristic": lambda e, o: heuristic_action(e),
            "random": lambda e, o: int(rng.choice(np.flatnonzero(o["action_mask"]))),
            "hold": lambda e, o: 0}


def baselines() -> dict:
    """Non-learning baselines on the validation seeds (the level the sweep
    tables and learning curves compare against)."""
    path = LOGS / "baselines.json"
    if path.exists():
        return json.loads(path.read_text())
    keep = ("return_mean", "return_std", "service_rate", "crashed")
    out = {name: {cfg: {k: v for k, v in rollout(pol, EVAL_SEEDS, cfg).items() if k in keep}
                  for cfg in EVAL_CONFIGS}
           for name, pol in baseline_policies().items()}
    path.write_text(json.dumps(out, indent=2))
    return out


def test_results(retest: bool = False) -> dict:
    """The final comparison: the best run of each algorithm (picked on the
    validation seeds) and the baselines, each played on the same 200 test
    seeds in every condition. Cached in logs/test_results.json and recomputed
    when a best model changes."""
    path = LOGS / "test_results.json"
    picked = {a: (best_model_dir(a) / "source_run.txt").read_text().strip()
              for a in ALGOS if (best_model_dir(a) / "model.pt").exists()}
    if path.exists() and not retest:
        cached = json.loads(path.read_text())
        if cached.get("best_runs") == picked and cached.get("seeds") == [TEST_SEEDS[0], TEST_SEEDS[-1]]:
            return cached
    policies = {}
    for a in picked:
        net = load_best_net(a)
        policies[a] = lambda e, o, net=net: greedy_action(net, o)
    policies.update(baseline_policies())
    results = {name: {cfg: rollout(pol, TEST_SEEDS, cfg) for cfg in EVAL_CONFIGS}
               for name, pol in policies.items()}
    out = {"seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]], "best_runs": picked, "results": results}
    text = json.dumps(out, indent=2)
    # one line per list of per-shift returns
    text = re.sub(r"\[\s+([-\d.,\s]+?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    path.write_text(text + "\n")
    return out


def load_net(algo: str, d: Path):
    """The network saved in a run's model directory (observation -> action scores)."""
    import torch
    from torch import nn

    from environment import N_ACTIONS, OBS_DIM
    from training.common import mlp

    hp = json.loads((d / "hparams.json").read_text())
    net = mlp(OBS_DIM, N_ACTIONS, hp["hidden"], act=nn.ReLU if algo == "dqn" else nn.Tanh)
    net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    net.eval()
    return net


def seed_check(data, retest: bool = False) -> dict:
    """The best configuration of each method retrained on training seeds 1 and
    2 (seedN_runXX): validation and test returns per seed, to show how much of
    a sweep gap one training seed can explain. Cached in logs/seed_check.json."""
    path = LOGS / "seed_check.json"
    picked = {a: best_run(data[a]) for a in ALGOS if data[a]}
    if path.exists() and not retest:
        cached = json.loads(path.read_text())
        if cached.get("best_runs") == picked:
            return cached
    out = {"best_runs": picked, "results": {}}
    for a, b in picked.items():
        sub = "dqn" if a == "dqn" else "pg"
        names = [b] + sorted(p.name for p in (LOGS / a).glob(f"seed*_{b}") if (p / "summary.json").exists())
        rows = []
        for name in names:
            s = json.loads((LOGS / a / name / "summary.json").read_text())
            net = load_net(a, ROOT / "models" / sub / a / name)
            t = rollout(lambda e, o, net=net: greedy_action(net, o), TEST_SEEDS, "default")
            rows.append({"run": name, "seed": s["hparams"]["seed"],
                         "validation": s["eval"]["default"]["return_mean"], "test": t["return_mean"]})
        out["results"][a] = rows
    path.write_text(json.dumps(out, indent=2) + "\n")
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
                           (axes[1], "eval_return", "Greedy policy on 20 checkpoint seeds")):
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
        ax.axhline(h, color=INK2, lw=1.2, ls=(0, (4, 3)), label=f"hand-written dispatcher ({h:+.1f})")
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps (thousands)")
    axes[0].set_ylabel("shift return")
    handles, labels = axes[0].get_legend_handles_labels()  # below the plots, clear of every line
    fig.legend(handles, labels, loc="lower center", ncols=len(labels), fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
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
                  f"({level:+.1f})")
    ax.set_title("Convergence speed: one dot per hyperparameter run (hollow = never reached it in 600k steps)",
                 loc="left")
    fig.savefig(PLOTS / "convergence.png")
    plt.close(fig)
    return rows


def fig_generalisation(test):
    fig, ax = plt.subplots(figsize=(10, 4))
    groups = list(EVAL_CONFIGS)
    labels = {"default": "Normal day\n(training conditions)", "busy": "Busy day\n(+50% requests)",
              "windy": "Windy day\n(+60% wind)"}
    res = test["results"]
    series = [(NAME[a], COLOR[a], a) for a in ALGOS if a in res]
    series.append(("Hand-written dispatcher", "#8a8984", "heuristic"))
    n = len(series)
    w = 0.8 / n
    for j, (name, color, key) in enumerate(series):
        xs = np.arange(len(groups)) + (j - (n - 1) / 2) * w
        ys = [res[key][c]["return_mean"] for c in groups]
        err = [res[key][c]["return_sem"] for c in groups]
        ax.bar(xs, ys, width=w - 0.02, color=color, label=name, edgecolor="white", linewidth=1)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(groups)), [labels[g] for g in groups])
    ax.set_ylabel(f"mean shift return ({len(TEST_SEEDS)} test seeds)\nerror bars: ± 1 standard error")
    ax.legend(ncols=n, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8.5)
    ax.set_title("Generalisation: best run of each method on test shifts never used for training or selection",
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


def tables(data, base, conv_rows, test, seeds=None):
    TABLES.mkdir(parents=True, exist_ok=True)
    conv = {(a, r): f for a, r, f in conv_rows}
    level = base["heuristic"]["default"]["return_mean"]
    for algo in ALGOS:
        runs = data[algo]
        if not runs:
            continue
        b = best_run(runs)
        head = ["Run", "Change from baseline", *COLS[algo], "Return (validation)", "Busy", "Windy",
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
            f"Dispatcher level = {level:+.1f}. Returns are means over 20 validation seeds that no run trained on. "
            "The best run (bold) is picked on these scores, which flatters it, so the final comparison in "
            f"headline.md uses {len(TEST_SEEDS)} separate test seeds.\n\n" + "\n".join(lines) + "\n")
    headline(data, base, test, seeds)


def seed_lines(seeds) -> str:
    if not seeds or not seeds.get("results"):
        return ""
    head = ["Method", "Configuration", "Validation by training seed", "Test by training seed",
            "Test across seeds (mean ± sd)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for a, rows in seeds["results"].items():
        tests = np.array([r["test"] for r in rows])
        lines.append("| " + " | ".join([
            NAME[a], seeds["best_runs"][a],
            " / ".join(f"{r['validation']:+.1f}" for r in rows),
            " / ".join(f"{r['test']:+.1f}" for r in rows),
            f"{tests.mean():+.1f} ± {tests.std(ddof=1) if len(tests) > 1 else 0:.1f}"]) + " |")
    return ("\n\n## Seed check\n\nThe best configuration of each method, retrained on training seeds 0, 1 and 2 "
            f"and played on the same {len(TEST_SEEDS)} test shifts (normal day). Seed 0 is the sweep run.\n\n"
            + "\n".join(lines) + "\n")


def headline(data, base, test, seeds=None):
    """Final comparison on the test seeds, plus paired differences (every
    method plays the same test shifts, so their returns can be compared shift
    by shift)."""
    res, n = test["results"], len(TEST_SEEDS)
    head = ["Method", "Best run (picked on validation)", "Validation return", "Test: normal day", "Busy day",
            "Windy day", "Service rate", "Crash rate (normal / busy)", "Ends away or in the air", "Emergencies delivered", "Emergencies missed",
            "Mean emergency wait (min)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = [(NAME[a], a) for a in ALGOS if a in res and data[a]]
    rows += [("Hand-written dispatcher", "heuristic"), ("Always hold", "hold"), ("Random (masked)", "random")]
    for name, key in rows:
        r = res[key]
        d = r["default"]
        if key in ALGOS:
            b = best_run(data[key])
            picked = f"{b} ({data[key][b]['summary']['hparams'].get('note', '')})"
            val = f"{data[key][b]['summary']['eval']['default']['return_mean']:+.1f}"
            emerg = [f"{d['delivered_emergency']:.1f}", f"{d['missed_emergency']:.1f}",
                     f"{d['mean_emergency_wait']:.1f}"]
        else:
            picked, val = "n/a", f"{base[key]['default']['return_mean']:+.1f}"
            emerg = ["", "", ""]
        lines.append("| " + " | ".join([
            name, picked, val, f"{d['return_mean']:+.1f} ± {d['return_std']:.1f}",
            f"{r['busy']['return_mean']:+.1f}", f"{r['windy']['return_mean']:+.1f}", f"{d['service_rate']:.0%}",
            f"{d['crashed']:.0%} / {r['busy']['crashed']:.0%}", f"{d.get('stranded', float('nan')):.0%}", *emerg]) + " |")

    learned = [a for a in ALGOS if a in res]
    lead = max(learned, key=lambda a: res[a]["default"]["return_mean"])
    phead = ["Method", *(f"{c.capitalize() if c != 'default' else 'Normal'} day" for c in EVAL_CONFIGS)]
    plines = ["| " + " | ".join(phead) + " |", "|" + "---|" * len(phead)]
    for key in [a for a in learned if a != lead] + ["heuristic"]:
        cells = []
        for c in EVAL_CONFIGS:
            diff = np.array(res[key][c]["returns"]) - np.array(res[lead][c]["returns"])
            cells.append(f"{diff.mean():+.1f} ± {diff.std() / np.sqrt(n):.1f} ({np.mean(diff > 0):.0%})")
        plines.append(f"| {NAME.get(key, 'Hand-written dispatcher')} | " + " | ".join(cells) + " |")

    (TABLES / "headline.md").write_text(
        f"# Final comparison on {n} test shifts\n\n"
        f"Each algorithm's best run was picked on 20 validation seeds (see the sweep tables). These are {n} "
        f"different seeds ({TEST_SEEDS[0]}–{TEST_SEEDS[-1]}) that neither training nor run selection used, "
        "and every method plays the same shifts. Test returns are mean ± standard deviation over the shifts "
        f"(the standard error of a mean is about a {np.sqrt(n):.0f}th of that). \"Validation return\" is the "
        "normal-day score the run was picked on.\n\n"
        + "\n".join(lines)
        + f"\n\n## Paired differences from {NAME[lead]}\n\n"
        f"{NAME[lead]} scored highest on normal test days. Each cell is the method's return minus "
        f"{NAME[lead]}'s on the same shift: mean ± standard error, and the share of shifts where the method "
        "scored higher. A difference within about two standard errors of zero is not a real gap.\n\n"
        + "\n".join(plines) + "\n" + seed_lines(seeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retest", action="store_true",
                    help="recompute the baselines, test evaluation and seed check even if they are cached")
    args = ap.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    data = {a: load(a) for a in ALGOS}
    if args.retest:
        (LOGS / "baselines.json").unlink(missing_ok=True)
    base = baselines()
    test = test_results(args.retest)
    fig_reward_curves(data, base)
    fig_sweeps(data, base)
    fig_dqn_objective(data)
    fig_pg_entropy(data)
    conv = fig_convergence(data, base)
    fig_generalisation(test)
    seeds = seed_check(data, args.retest)
    tables(data, base, conv, test, seeds)
    res = test["results"]
    print(f"{'':>23}validation  test ({len(TEST_SEEDS)} shifts)")
    for a in ALGOS:
        if data[a] and a in res:
            b = best_run(data[a])
            t = res[a]["default"]
            print(f"{NAME[a]:>10} ({len(data[a])} runs, best {b}) "
                  f"{data[a][b]['summary']['eval']['default']['return_mean']:+6.1f}   "
                  f"{t['return_mean']:+6.1f} ± {t['return_sem']:.1f}")
    h = res["heuristic"]["default"]
    print(f"{'dispatcher':>23} {base['heuristic']['default']['return_mean']:+6.1f}   "
          f"{h['return_mean']:+6.1f} ± {h['return_sem']:.1f}")
    print(f"wrote {PLOTS}, {TABLES} and {LOGS / 'test_results.json'}")


if __name__ == "__main__":
    main()
