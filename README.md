# Mission-Based Reinforcement Learning: Medical Drone Delivery

Summative assignment, Machine Learning Techniques. Shima Serein.

A custom environment in which a delivery drone serves six rural health centres from a hub in Muhanga District, Rwanda, during an 8-hour shift. Requests for blood and emergency medicine arrive at random with deadlines. Four agents are trained on it with TorchRL: DQN, REINFORCE, PPO and A2C.

## Run

```bash
uv sync
uv run main.py
```

`main.py` loads the best trained agent (PPO), opens the three.js visualisation in the browser and prints every decision in the terminal. `uv run play.py` does the same.

| Option | Effect |
|---|---|
| `--algo dqn` | play another agent: `dqn`, `ppo`, `a2c`, `reinforce`, or `heuristic` for the hand-written baseline |
| `--episodes 3` | number of shifts to play |
| `--config busy` | a day with 50% more requests (`windy`: 60% stronger wind) |
| `--no-viewer --fast` | terminal only, no animation |

## Environment

`environment/custom_env.py`, Gymnasium API, registered as `DroneDelivery-v0`. Each step is one decision; the clock moves on by the time that decision takes.

**Actions (10):** Hold, Return to Hub, Deliver to one of the 6 health centres, Recharge, Load Package. An action mask in the observation removes commands the drone cannot carry out (flights the battery cannot finish, deliveries with no package, charging or loading away from the hub).

**Observation (38 values in [−1, 1]):** drone location (one-hot), battery, cargo, shift time, wind speed and direction, each centre's open request, emergency flag and time left, and the battery needed to reach each site and return to the hub.

**Rewards:**

| Event | Reward |
|---|---|
| Emergency delivered before its 45-minute deadline | +10 |
| Routine delivery before its 120-minute deadline | +4 |
| Emergency deadline missed | −10 |
| Routine deadline missed | −3 |
| Trip to a centre with no open request | −1 |
| Shift ends away from the hub | −15 |
| Battery runs out in flight (crash) | −40, plus the requests the grounded service will miss |

**Start state:** the drone at the hub with 70–100% battery, no cargo, up to 2 open requests and a random wind.
**Terminal conditions:** the shift reaches 480 minutes, or the drone crashes.

**Visualisation:** `environment/rendering.py` serves a three.js scene (`environment/web/`) updated over Server-Sent Events as the agent acts: real Sentinel-2 imagery of Muhanga draped over SRTM elevation (×4.5 vertical), the hub and health centres as map pins at their stylised positions, and a fixed-wing delivery drone that parachutes each package. The map is baked into `environment/web/map/` by `scripts/bake_map.py`, so the viewer runs offline. Imagery: Sentinel-2 cloudless 2025 by EOX (contains modified Copernicus Sentinel data), CC BY-NC-SA 4.0; elevation: SRTM via AWS Terrain Tiles.

## Training

| File | Algorithms |
|---|---|
| `training/dqn_training.py` | DQN (`DQNLoss`, replay buffer, target network, epsilon-greedy) |
| `training/pg_training.py` | REINFORCE (`ReinforceLoss`), PPO (`ClipPPOLoss` + GAE), A2C (`A2CLoss` + GAE) |

Each algorithm has 10 hyperparameter configurations, trained for 600,000 steps each on the same environment and evaluated on 20 seeds not used in training.

```bash
uv run python -m training.dqn_training --sweep --jobs 2
uv run python -m training.pg_training --algo ppo --sweep --jobs 2    # also: reinforce, a2c
uv run python -m training.analyze                                    # plots in assets/plots
```

## Results

| Agent | Mean return (20 unseen shifts) |
|---|---|
| PPO | +55.6 |
| DQN | +51.3 |
| A2C | +49.4 |
| REINFORCE | +36.8 |
| Hand-written dispatcher | +41.4 |

All 40 runs, with their hyperparameters and results, are in `logs/<algo>/sweep_summary.csv`; the plots are in `assets/plots/`.

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
│   └── web/                 three.js visualisation; map/ holds the baked imagery and elevation
├── scripts/
│   └── bake_map.py          downloads and bakes the viewer's map
├── training/
│   ├── __init__.py
│   ├── dqn_training.py
│   ├── pg_training.py
│   ├── common.py            shared environment, evaluation and logging code
│   └── analyze.py           plots and tables
├── models/
│   ├── dqn/
│   └── pg/
├── logs/
├── assets/
└── tests/
```
