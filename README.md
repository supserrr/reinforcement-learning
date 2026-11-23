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

- **Actions (10):** Hold, Return to Hub, Deliver to one of 6 centres, Recharge, Load Package. An action mask applies the return-to-base rule: a flight is accepted only if the battery covers it and the flight home after it (with a 2% reserve) and both land before the shift ends.
- **Observation (45 values in [−1, 1]):** location, battery, cargo, time, wind, each centre's request, emergency flag and time left, the battery needed to reach each site and get back, and the shift time left after each of those trips.
- **Start:** at the hub with 70–100% battery, no cargo, up to 2 open requests and a random wind.
- **End:** the shift reaches 480 minutes, or the drone crashes. Under the mask every shift ends with the drone at the hub; the last two rewards below only apply to a caller that bypasses it.

| Event | Reward |
|---|---|
| Emergency delivered within 45 min | +10 |
| Routine delivered within 120 min | +4 |
| Emergency missed | −10 |
| Routine missed | −3 |
| Trip to a centre with no request | −1 |
| Shift ends away from the hub | −25 |
| Crash (battery runs out in flight) | −40, plus the requests it will now miss |

**Visualisation:** `environment/rendering.py` streams each step to a three.js scene in `environment/web/`: satellite imagery over real elevation, the hub and health centres, and a fixed-wing drone that launches, drops packages by parachute and lands on a recovery wire. The map and 3D models are baked into the repo by `scripts/`, so the viewer runs offline.

Imagery: Sentinel-2 cloudless 2025 by EOX (modified Copernicus Sentinel data), CC BY-NC-SA 4.0. Elevation: SRTM via AWS Terrain Tiles. Buildings: © OpenStreetMap contributors, ODbL.

## Training

| File | Algorithms |
|---|---|
| `training/dqn_training.py` | DQN (`DQNLoss`) |
| `training/pg_training.py` | REINFORCE (`ReinforceLoss`), PPO (`ClipPPOLoss`), A2C (`A2CLoss`) |

Each algorithm has 10 hyperparameter runs of 600,000 steps. Every run keeps its best checkpoint on 20 checkpoint seeds, the best run of each algorithm is picked on 20 validation seeds, and the four best are compared on 200 test seeds used for nothing else. The best configuration of each algorithm is also retrained on training seeds 1 and 2 to show how much one seed can move a score.

```bash
uv run python -m training.dqn_training --sweep --jobs 2
uv run python -m training.pg_training --algo ppo --sweep --jobs 2    # also: reinforce, a2c
uv run python -m training.dqn_training --runs 7 --seed 1             # seed repeat of one configuration
uv run python -m training.analyze --retest                           # test evaluation, seed check, plots, tables
```

The same sweeps run on Google Colab with [`training/colab.ipynb`](https://colab.research.google.com/github/supserrr/shima_serein_rl_summative/blob/main/training/colab.ipynb) (about 2–2.5 hours on a CPU runtime).

## Results

Mean return per shift; test scores are over 200 shifts, ± standard error.

| Agent | Validation | Test: normal day | Busy day | Windy day | Test, 3 training seeds |
|---|---|---|---|---|---|
| DQN | +48.9 | +44.8 ± 1.0 | +33.4 | +43.8 | +43.5 ± 2.0 |
| PPO | +49.9 | +35.7 ± 1.3 | +21.9 | +35.5 | +33.8 ± 2.0 |
| A2C | +42.0 | +38.1 ± 1.5 | +19.3 | +38.5 | +35.6 ± 2.2 |
| REINFORCE | +37.0 | +37.2 ± 1.3 | +20.2 | +36.9 | +35.5 ± 2.4 |
| Hand-written dispatcher | +11.6 | +9.3 ± 1.7 | −12.1 | +8.7 | |

DQN leads in every condition, by 7–9 points on normal days and 11–14 on busy ones, and keeps that lead across training seeds (last column: mean ± standard deviation of the best configuration retrained on seeds 0, 1 and 2). No agent crashes or ends a test shift away from the hub. Validation scores are higher because the best runs were picked on them. More detail: `assets/tables/headline.md` (test comparison and seed check), `logs/<algo>/sweep_summary.csv` (all 40 runs) and `assets/plots/`.

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
