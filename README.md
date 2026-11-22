# Reinforcement Learning: Medical Drone Delivery

Summative assignment, Machine Learning Techniques. Shima Serein.

A drone delivers blood and emergency medicine from a hub in Muhanga District, Rwanda, to six rural health centres over an 8-hour shift. Requests arrive at random with deadlines. DQN, REINFORCE, PPO and A2C are trained on it with TorchRL.

## Run

```bash
uv sync
uv run main.py
```

This plays the best agent in a three.js viewer in the browser and prints every decision in the terminal. `uv run play.py` does the same. In the viewer, `F` follows the drone and `[` / `]` change the speed (½× to 4×).

| Option | Effect |
|---|---|
| `--algo dqn` | another agent: `dqn`, `ppo`, `a2c`, `reinforce` or `heuristic` (hand-written baseline) |
| `--episodes 3` | number of shifts |
| `--config busy` | 50% more requests (`windy`: 60% stronger wind) |
| `--no-viewer --fast` | terminal only |

## Environment

`environment/custom_env.py`, a Gymnasium environment registered as `DroneDelivery-v0`. Each step is one decision, and the clock moves on by the time it takes. Battery, cargo (up to 2 packages), position and open requests carry over between steps. Flight energy depends on distance, wind, payload and the climb over the real terrain.

- **Actions (10):** Hold, Return to Hub, Deliver to one of 6 centres, Recharge, Load Package. An action mask rules out commands the drone cannot carry out.
- **Observation (38 values in [−1, 1]):** location, battery, cargo, time, wind, each centre's request, emergency flag and time left, and the battery needed to reach each site and get back.
- **Start:** at the hub with 70–100% battery, no cargo, up to 2 open requests and a random wind.
- **End:** the shift reaches 480 minutes, or the drone crashes.

| Event | Reward |
|---|---|
| Emergency delivered within 45 min | +10 |
| Routine delivered within 120 min | +4 |
| Emergency missed | −10 |
| Routine missed | −3 |
| Trip to a centre with no request | −1 |
| Shift ends away from the hub | −15 |
| Crash (battery runs out in flight) | −40, plus the requests it will now miss |

**Visualisation:** `environment/rendering.py` streams each step to a three.js scene in `environment/web/`: satellite imagery over real elevation, the hub and health centres, and a fixed-wing drone that launches, drops packages by parachute and lands on a recovery wire. The map and 3D models are baked into the repo by `scripts/`, so the viewer runs offline.

Imagery: Sentinel-2 cloudless 2025 by EOX (modified Copernicus Sentinel data), CC BY-NC-SA 4.0. Elevation: SRTM via AWS Terrain Tiles. Buildings: © OpenStreetMap contributors, ODbL.

## Training

| File | Algorithms |
|---|---|
| `training/dqn_training.py` | DQN (`DQNLoss`) |
| `training/pg_training.py` | REINFORCE (`ReinforceLoss`), PPO (`ClipPPOLoss`), A2C (`A2CLoss`) |

Each algorithm has 10 hyperparameter runs of 600,000 steps. The best run of each is picked on 20 validation seeds, then the four best are compared on 200 test seeds used for nothing else.

```bash
uv run python -m training.dqn_training --sweep --jobs 2
uv run python -m training.pg_training --algo ppo --sweep --jobs 2    # also: reinforce, a2c
uv run python -m training.analyze                                    # test evaluation, plots, tables
```

The same sweeps run on Google Colab with [`training/colab.ipynb`](https://colab.research.google.com/github/supserrr/reinforcement-learning/blob/main/training/colab.ipynb) (about 2–2.5 hours on a CPU runtime).

## Results

Mean return per shift; test scores are over 200 shifts, ± standard error.

| Agent | Validation | Test: normal day | Busy day | Windy day |
|---|---|---|---|---|
| DQN | +37.4 | +31.6 ± 1.5 | +19.3 | +31.6 |
| PPO | +43.0 | +31.5 ± 1.8 | +12.6 | +30.6 |
| A2C | +29.1 | +23.0 ± 1.3 | +6.4 | +23.3 |
| REINFORCE | +5.1 | +1.9 ± 1.9 | −9.2 | +2.3 |
| Hand-written dispatcher | +11.8 | +9.4 ± 1.7 | −11.6 | +8.8 |

DQN and PPO tie on normal and windy days; DQN is ahead on busy days. Validation scores are higher because the best runs were picked on them. More detail: `assets/tables/headline.md` (test comparison), `logs/<algo>/sweep_summary.csv` (all 40 runs) and `assets/plots/`.

## Structure

```text
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
├── play.py
├── environment/
│   ├── __init__.py
│   ├── custom_env.py
│   ├── rendering.py
│   └── web/              three.js viewer, baked map and 3D models
├── scripts/              bakes the map, buildings, route climbs and Blender models
├── training/
│   ├── __init__.py
│   ├── dqn_training.py
│   ├── pg_training.py
│   ├── common.py         shared environment, evaluation and logging
│   ├── analyze.py        test evaluation, plots and tables
│   └── colab.ipynb
├── models/
│   ├── dqn/
│   └── pg/
├── logs/
├── assets/
└── tests/
```
