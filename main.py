"""Watch a trained agent run a delivery shift in the 3D viewer.

    uv run main.py                       # best agent overall, 3D viewer + terminal log
    uv run main.py --algo ppo            # best PPO run (dqn, ppo, a2c, reinforce)
    uv run main.py --algo heuristic      # hand-written dispatcher, for comparison
    uv run main.py --no-viewer --fast    # terminal only, no waiting
    uv run main.py --config busy         # a busier day than training ever saw
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from environment import ACTION_NAMES, SITE_NAMES, DroneDeliveryEnv, heuristic_action
from environment.rendering import Viewer, scene_description, step_message

ROOT = Path(__file__).resolve().parent
ALGOS = ("dqn", "ppo", "a2c", "reinforce")
LABELS = {"dqn": "DQN Agent", "ppo": "PPO Agent", "a2c": "A2C Agent", "reinforce": "REINFORCE Agent",
          "heuristic": "Hand-Written Dispatcher", "random": "Random Actions"}
NOTES = {"heuristic": "Rule-based baseline, no learning", "random": "Any action the controller allows"}


def best_dir(algo: str) -> Path:
    return ROOT / "models" / ("dqn" if algo == "dqn" else "pg") / f"{algo}_best"


def best_algo() -> str:
    """The algorithm with the highest normal-day score in the final test
    comparison (logs/test_results.json), or, before that exists, the one whose
    best sweep run scored highest on the validation seeds."""
    test = ROOT / "logs" / "test_results.json"
    if test.exists():
        res = json.loads(test.read_text())["results"]
        scores = {a: res[a]["default"]["return_mean"] for a in ALGOS
                  if a in res and (best_dir(a) / "model.pt").exists()}
        if scores:
            return max(scores, key=scores.get)
    scores = {}
    for algo in ALGOS:
        table = ROOT / "logs" / algo / "sweep_summary.csv"
        if table.exists() and (best_dir(algo) / "model.pt").exists():
            with open(table) as f:
                scores[algo] = max(float(r["default_return"]) for r in csv.DictReader(f))
    if not scores:
        return "heuristic"
    return max(scores, key=scores.get)


def load_policy(algo: str):
    if algo == "heuristic":
        return (lambda env, obs: heuristic_action(env)), LABELS[algo], NOTES[algo]
    if algo == "random":
        rng = np.random.default_rng(0)
        return (lambda env, obs: int(rng.choice(np.flatnonzero(obs["action_mask"])))), LABELS[algo], NOTES[algo]

    from training.common import greedy_action, load_best_net

    d = best_dir(algo)
    net = load_best_net(algo)
    run = (d / "source_run.txt").read_text().split()[0] if (d / "source_run.txt").exists() else ""
    note = f"Best of 10 tuning runs ({run.replace('run', 'run ')})" if run else "Best tuning run"
    return (lambda env, obs: greedy_action(net, obs)), LABELS[algo], note


def fmt_requests(snap) -> str:
    out = []
    for i, r in enumerate(snap["requests"]):
        if r:
            out.append(f"{SITE_NAMES[i].split()[0]}({'E' if r['emergency'] else 'R'} {r['minutes_left']:.0f}m)")
    return " ".join(out) or "-"


def clock(t: float) -> str:
    m = 8 * 60 + int(round(t))
    return f"{m // 60:02d}:{m % 60:02d}"


def main():
    sys.stdout.reconfigure(line_buffering=True)  # the decision log appears live even when piped
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="best", choices=("best", *ALGOS, "heuristic", "random"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10_000, help="first evaluation seed (never used in training)")
    ap.add_argument("--config", default="default", choices=("default", "busy", "windy"))
    ap.add_argument("--ms-per-minute", type=float, default=110.0, help="pace with no browser open: ms per simulated minute")
    ap.add_argument("--no-viewer", action="store_true", help="terminal output only")
    ap.add_argument("--fast", action="store_true", help="do not wait for animations")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="start the viewer without opening a browser tab")
    args = ap.parse_args()

    algo = best_algo() if args.algo == "best" else args.algo
    policy, label, note = load_policy(algo)
    viewer = None
    if not args.no_viewer:
        viewer = Viewer(port=args.port, open_browser=not args.no_browser,
                        policy_fn=None if algo in ("heuristic", "random") else
                        (lambda obs: policy(None, obs)))
        print(f"3D viewer: {viewer.url}  (waiting for the browser to connect)")
        viewer.wait_for_client(30)
        time.sleep(1.5)  # let the page build the terrain

    log_dir = ROOT / "logs" / "demo"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAgent: {label}, {note.lower()}   map: Muhanga District, 6 health centres   day: {args.config}\n")
    summary = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        env = DroneDeliveryEnv(args.config)
        obs, _ = env.reset(seed=seed)
        if viewer:
            viewer.set_scene(scene_description(env, label, note))
            viewer.publish("episode_start", {"episode": ep + 1, "seed": seed, "snapshot": env.snapshot()})
            time.sleep(1.0 if not args.fast else 0)
        print(f"=== Shift {ep + 1}/{args.episodes} · seed {seed} · battery {env.battery:.0f}% · "
              f"open: {fmt_requests(env.snapshot())}")
        jsonl = open(log_dir / f"{algo}_seed{seed}.jsonl", "w")
        done, step = False, 0
        while not done:
            t0, b0 = env.t, env.battery
            action = policy(env, obs)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            step += 1
            minutes = max(env.t - t0, 0.5)
            anim_ms = 0 if args.fast else int(minutes * args.ms_per_minute)
            msg = step_message(env, episode=ep + 1, step=step, action=action, reward=reward,
                               anim_ms=anim_ms, agent=label)
            jsonl.write(json.dumps(msg) + "\n")
            snap = msg["snapshot"]
            notes = [e["type"] for e in snap["events"] if e["type"] in
                     ("delivered", "missed", "wasted", "crash", "stranded", "request")]
            print(f"[{clock(t0)}] {ACTION_NAMES[action]:<26} batt {b0:5.1f}%→{env.battery:5.1f}%  "
                  f"cargo {env.cargo}  r {reward:+6.1f}  total {env.total_reward:+7.1f}  "
                  f"open: {fmt_requests(snap)}" + (f"  [{', '.join(notes)}]" if notes else ""))
            if viewer:
                viewer.publish("step", msg)
                if not args.fast:  # the page sets the pace; it says when the step has played
                    viewer.wait_for_playback(ep + 1, step, anim_ms / 1000)
        jsonl.close()
        st = env.stats.as_dict()
        summary.append((seed, env.total_reward, st))
        print(f"--- shift return {env.total_reward:+.1f} · emergencies {st['delivered_emergency']} delivered / "
              f"{st['missed_emergency']} missed · routine {st['delivered_routine']} / {st['missed_routine']} · "
              f"service rate {st['service_rate']:.0%} · crashed {st['crashed']} · back at hub {not st['stranded']}\n")
        if viewer and not args.fast:
            time.sleep(2.5)

    rets = [s[1] for s in summary]
    print(f"Mean return over {len(rets)} unseen shifts: {np.mean(rets):+.1f} (± {np.std(rets):.1f})")
    if viewer:
        print(f"Viewer still running at {viewer.url}. Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            viewer.close()


if __name__ == "__main__":
    main()
