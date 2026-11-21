# The Drone Delivery RL Architecture

<https://github.com/supserrr/shima_serein_rl_summative>

## Introduction

This repository trains reinforcement learning agents to run a medical delivery drone's 8-hour shift in Muhanga District, Rwanda. A custom Gymnasium environment simulates one drone, a hub and six health centres whose requests for blood and medicine arrive at random with deadlines. Four TorchRL agents (DQN, REINFORCE, PPO, A2C) learn to dispatch it, each over a 10-run hyperparameter sweep, and are compared with a hand-written dispatcher on seeds they never trained on.

The environment is plain Python and NumPy with no rendering code. Training wraps it in TorchRL and writes logs, weights and a sweep table per algorithm. `main.py` loads the best weights, runs shifts, prints a decision log to the terminal and streams every step as JSON to a three.js page that draws the hills, the sites and the drone.

## What This Project is NOT

- It is not a flight controller. Flights are modelled as distance, climb, payload and wind costs on a fixed map, not as aircraft dynamics.
- It is not a survey of Muhanga. Site positions, elevations and demand rates are stylised.
- It does not train in the browser. The viewer only plays back what the Python process decides.

## Major Components

### Environment [PYTHON · GYMNASIUM]

#### [environment/custom_env.py](environment/custom_env.py)

`DroneDeliveryEnv`: the event-driven shift (one step is one decision, the clock moves by its duration), a 38-value observation plus an `action_mask`, 10 actions, rewards and the crash penalty. `DroneConfig` holds every number, with `CONFIGS` for the normal, busy and windy days. `Request` and `EpisodeStats` are its state and counters, `snapshot()` its JSON view, and `heuristic_action` the hand-written baseline.

#### [environment/\_\_init\_\_.py](environment/__init__.py)

Registers `DroneDelivery-v0` and exports the constants the trainers and the viewer share (`OBS_DIM`, `N_ACTIONS`, `ACTION_NAMES`, `SITE_NAMES`).

#### [environment/rendering.py](environment/rendering.py)

`Viewer`: a standard-library HTTP server for the web page, with `/scene`, `/state`, a Server-Sent Events stream at `/events` and `POST /api/act` for the loaded policy. `scene_description` and `step_message` define the JSON every consumer reads.

### Training [PYTHON · TORCHRL]

#### [training/common.py](training/common.py)

`make_env` (GymWrapper, mask cast to bool, `StepCounter`, `RewardSum`, then `RewardScaling`), `mlp`, `greedy_action`, `evaluate` on held-out seeds, `RunLogger`, and `run_sweep`/`write_sweep_table`, which run configurations in parallel processes and copy the best run to `models/*/<algo>_best/`.

#### [training/dqn_training.py](training/dqn_training.py)

DQN with `QValueActor`, `EGreedyModule`, `DQNLoss`, `HardUpdate` or `SoftUpdate` and a `TensorDictReplayBuffer`. `SWEEP` lists the 10 configurations.

#### [training/pg_training.py](training/pg_training.py)

`build_actor_critic` (a `MaskedCategorical` actor and a value head), `train_ppo_a2c` (`ClipPPOLoss` or `A2CLoss` with `GAE`) and `train_reinforce` (`ReinforceLoss` on Monte Carlo returns, with an optional value baseline). `SWEEPS` holds 10 configurations per method.

#### [training/analyze.py](training/analyze.py)

Reads every run's logs and writes the figures in `assets/plots/`, the tables in `assets/tables/` and `logs/baselines.json`.

### Viewer [JAVASCRIPT · THREE.JS]

#### [environment/web](environment/web)

`index.html` and `style.css` hold the shift card, the decision log and toasts, set in Space Grotesk. `app.js` reads the event stream, queues steps and animates each one. `world.js` (`createWorld`) builds the scene: sky and sun from the shift clock, terraced terrain shader, water, vegetation, villages, roads, mist, clouds, traffic, wind effects, the sites, the drone and post-processing.

### Entry Points [PYTHON]

#### [main.py](main.py) and [play.py](play.py)

Load the best agent (or `--algo`), run shifts on unseen seeds, print each decision and publish it to the viewer. `play.py` calls `main.main`.

#### [tests](tests)

`test_env.py`: spaces, determinism, the action mask, flight energy, deliveries, shift length, the crash penalty and the baseline ordering.

## External Pieces

- **TorchRL and TensorDict** for environments, collectors, losses and replay buffers; **PyTorch** for the networks; **Gymnasium** for the environment API.
- **three.js 0.170** (MIT) vendored in `environment/web/vendor/` with the add-ons it imports (OrbitControls, CSS2DRenderer, Sky, ImprovedNoise, BufferGeometryUtils, EffectComposer, GTAOPass, UnrealBloomPass, OutputPass), so the viewer works offline.
- **Space Grotesk** (SIL OFL 1.1) in `environment/web/fonts/`, with its licence in `OFL.txt`.
- **matplotlib** and **pandas** for the figures and tables.

## Tooling

- `uv` manages Python and dependencies (`pyproject.toml`, `uv.lock`, `.python-version`).
- `docs/video_script.md` is the plan for the 5-minute demonstration video.

# Workflows

## What does a shift look like?

`reset` puts the drone at the hub with 70–100% battery and up to two open requests. Each `step` applies one action, advances the clock by its duration, expires requests whose deadlines passed, spawns new ones (Poisson, busier at midday), drifts the wind, then pays for any delivery. The episode ends at 480 minutes or at a crash.

## What does a training run look like?

`run_sweep` gives each configuration to a worker. The worker builds the env with `make_env`, collects frames with a `Collector` (or full episodes with `env.rollout` for REINFORCE), updates the networks, logs every 5,000 frames with a quick greedy evaluation, and finally evaluates on 20 unseen seeds for each day type. `RunLogger.finish` writes `logs/<algo>/runXX/` and `models/.../runXX/model.pt`; `write_sweep_table` updates `logs/<algo>/sweep_summary.csv` and the `_best` copy.

## What does watching an agent look like?

`main.py` starts `Viewer` and opens the browser. For every decision it steps the env, prints a line, appends to `logs/demo/<agent>_seed<seed>.jsonl` and publishes a `step_message`, then waits for the animation. `app.js` turns each message into a flight, hold, recharge or load in `world.js` and updates the card.

## Development

```bash
uv sync
uv run pytest
uv run python -m training.dqn_training --sweep --jobs 2
uv run python -m training.pg_training --algo ppo --sweep --jobs 2
uv run python -m training.analyze
uv run main.py
```

## Release

The submission is this repository. A marker clones it and runs `uv sync` then `uv run main.py`, which needs the committed weights in `models/` and no training.

# Invariants

- The observation layout (`OBS_DIM` = 38, in the order `_features` writes it) and the action order (`ACTION_NAMES`) are what every saved model was trained on. Changing either invalidates `models/`.
- Hold is always allowed by `action_mask`, so a masked policy always has a legal action.
- Evaluation seeds (10000–10019 and 20000–20004) are never used for training; training uses seed 0.
- `RewardSum` comes before `RewardScaling` in `make_env`, so logged returns are in the environment's units while learners see reward × 0.1.
- A crash costs more than idling the whole shift (`_crash_penalty` adds every request the grounded service will miss); `test_crash_is_worse_than_doing_nothing` guards it.
- `wind_dir` is the heading the wind blows towards in env axes (x east, y north). `world.js` maps env (x, y) to scene (x, −y), and `app.js` names the direction the wind comes from.
- `main.py` rebuilds networks from each model's `hparams.json`, so `hidden` must stay in it.

## License

No licence has been chosen for this repository's own code yet. The vendored three.js is MIT and Space Grotesk is SIL OFL 1.1.
