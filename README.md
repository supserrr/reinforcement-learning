# Medical Drone Delivery in Rwanda's Hills: Mission-Based RL

One delivery drone works an 8-hour shift from a distribution hub in Muhanga District. Six rural health centres send requests for blood and emergency medicine at random times, each with a deadline. The agent is the hub's flight controller: every decision it makes (fly, hold, recharge, load) changes what it can do next, because battery, cargo, open requests and the drone's position all carry over.

Four agents are trained on the same environment with **TorchRL**: DQN (value-based), REINFORCE, PPO and A2C (policy gradient). Each gets a 10-run hyperparameter sweep at the same 600k-step budget, and all are compared with a hand-written dispatcher on seeds and conditions they never trained on. A **three.js** viewer shows the agent flying the shift in 3D.

## Run it

```bash
uv sync
uv run main.py            # best agent, 3D viewer in the browser + verbose terminal log
```

| Command | What it does |
|---|---|
| `uv run main.py --algo ppo` | best run of one algorithm (`dqn`, `ppo`, `a2c`, `reinforce`) |
| `uv run main.py --algo heuristic` | the hand-written dispatcher, for comparison |
| `uv run main.py --config busy` | a busier day (+50% requests) than training ever saw; `windy` = +60% wind |
| `uv run main.py --no-viewer --fast` | terminal only, no animation delays |
| `uv run play.py` | same as `main.py` |
| `uv run python -m training.dqn_training --sweep --jobs 2` | retrain the 10 DQN runs |
| `uv run python -m training.pg_training --algo ppo --sweep --jobs 2` | retrain a policy-gradient sweep (`reinforce`, `ppo`, `a2c`) |
| `uv run python -m training.analyze` | rebuild every figure and table in `assets/` |
| `uv run pytest` | environment tests |

## The environment (`environment/custom_env.py`)

Gymnasium API, registered as `DroneDelivery-v0`, wrapped for TorchRL with `GymWrapper`. It is event-driven: one step is one operational decision, and the simulated clock advances by however long that decision takes (a 10-minute flight and a 3-minute load are each one step). A shift takes 40–80 decisions.

### Map

A hub and six health centres, 20–25 km from the hub and 13–40 km from each other, at 1,750–2,300 m elevation. The layout is stylised, not surveyed.

### Actions (10, discrete)

| # | Action | Real-world command | Time |
|---|---|---|---|
| 0 | Hold | wait at the hub, or loiter over a centre (costs 0.35%/min battery away from the hub) | 5 min |
| 1 | Return to hub | fly back to base | distance ÷ 120 km/h |
| 2–7 | Deliver to centre 1–6 | fly to that centre and parachute one package | distance ÷ 120 km/h |
| 8 | Recharge | charge on the hub pad, +20% | 10 min |
| 9 | Load package | load one blood/medicine package (holds 2) | 3 min |

The observation also carries an **action mask**: the commands the flight controller would accept now. It refuses flights the battery cannot finish, deliveries with an empty hold, and charging or loading away from the hub. It does **not** stop the drone from flying somewhere it cannot get home from; avoiding that is the agent's job.

### Observation (38 values, all scaled to [-1, 1])

| Index | Value | Encoding and range | Where it comes from (simulation rule → real source) | Updated |
|---|---|---|---|---|
| 0–6 | Drone location | one-hot over hub + 6 centres | position after the last flight → GPS / ops system | every decision |
| 7 | Battery | % ÷ 100, [0, 1] | minus flight energy, plus 20 per recharge → battery management system | every decision |
| 8 | Cargo on board | packages ÷ 2, {0, 0.5, 1} | +1 per load, −1 per drop → cargo bay sensor | every decision |
| 9 | Shift clock | minutes ÷ 480, [0, 1] | advances by each action's duration → hub clock | every decision |
| 10 | Wind speed | km/h ÷ 64, [0, 1] | random walk (σ = 2% of max per √min) → hub anemometer | drifts continuously, sampled each decision |
| 11–12 | Wind direction | sin, cos of heading, [−1, 1] | random walk (σ = 0.03 rad per √min) → hub wind vane | as above |
| 13–30 | Per centre (×6): open request, emergency flag, time left | {0,1}, {0,1}, minutes ÷ 120 in [0, 1] | Poisson arrivals, 0.3–0.65 per hour per centre, peaking at midday; 30% emergencies (45-min deadline), else routine (120 min) → the centres' ordering system (SMS/web orders) | arrivals and expiries during every action |
| 31–37 | Battery needed to reach each site and get home | % ÷ 100, [0, 1] | distance × 0.9%/km × wind (headwind up to +55%) × payload (+6% per package) + 0.004%/m climbed → flight-planning software | every decision |

The agent receives indices 0–37 as the `observation` key and the mask as `action_mask` (TorchRL `GymWrapper` → MLP input; the mask feeds `QValueActor`, `EGreedyModule` and `MaskedCategorical`).

### Rewards

| Event | Reward |
|---|---|
| Emergency package delivered before its deadline | +10 |
| Routine package delivered before its deadline | +4 |
| Emergency deadline missed (the centre has to send a patient by road) | −10 |
| Routine deadline missed | −3 |
| Trip to a centre with no open request | −1 |
| Shift ends with the drone away from the hub | −15 |
| Battery runs out in the air (crash) | −40, plus every open request and every request still expected this shift (the service is grounded) |

Rewards are given unscaled by the environment; the learners see them ×0.1 (TorchRL `RewardScaling`) and every return reported here is in the original units.

### Start state and terminal conditions

- **Start:** drone at the hub, battery 70–100%, no cargo, 0–2 requests already open, random wind.
- **Terminal:** the shift reaches 480 minutes, or the drone crashes. A safety cap of 400 decisions truncates runaway episodes (never reached in practice).

### Why this needs RL

The right action depends on what happens next, and the environment only says so later. Leaving to serve a routine request costs nothing now, but an emergency that arrives while the drone is away can be missed 45 minutes later. Recharging before it is needed wastes time; not recharging can strand the drone. Loading two packages lets one trip serve two centres, but only if a second request turns up on the way. Demand and wind are random, so there is no fixed route to precompute: a policy has to weigh expected future requests against battery and time. A greedy rule (the hand-written dispatcher in `heuristic_action`) is a strong baseline, and the learned agents are judged against it.

## 3D viewer and integration (`environment/rendering.py`, `environment/web/`)

`main.py` starts a local server and opens a three.js page of Muhanga's hills:

- A physically based sky whose sun follows the shift clock (Muhanga's latitude, late November), soft shadows, sky reflections, ambient occlusion and haze.
- Terraced hillsides with crop rows, eucalyptus woodlots, banana groves, villages, red dirt roads with traffic, a river and lakes.
- Wind you can see: trees sway, gusts ripple across the crops, cloud shadows and smoke drift, the windsock turns, and the drone crabs into crosswinds.
- The hub, six health centres with request beacons (red = emergency, amber = routine), parachute drops and charging.
- A card with the clock, battery, cargo, wind, return and deliveries, and a decision log.

Press **F** to follow the drone and **O** to orbit. Add `?quality=low` to the address on slower machines.

The server exposes the same data a real deployment would use:

| Endpoint | Purpose |
|---|---|
| `GET /scene` | static map: sites, coordinates, elevations, action names |
| `GET /state` | the latest step as JSON |
| `GET /events` | live stream of steps (Server-Sent Events) |
| `POST /api/act` | `{"observation": [...38], "action_mask": [...10]}` → `{"action": 3, "name": "deliver to Kiyumba HC"}` from the loaded policy |

Each step is also written to `logs/demo/<agent>_seed<seed>.jsonl` in the same format.

## Training (`training/`)

| File | Contents |
|---|---|
| `common.py` | TorchRL env construction, networks, held-out evaluation, logging, parallel sweep runner |
| `dqn_training.py` | DQN: `QValueActor`, `EGreedyModule`, `DQNLoss`, `HardUpdate`/`SoftUpdate`, `TensorDictReplayBuffer`, 10 configurations |
| `pg_training.py` | REINFORCE (`ReinforceLoss`, Monte Carlo returns with optional value baseline), PPO (`ClipPPOLoss` + `GAE`), A2C (`A2CLoss` + `GAE`), 10 configurations each |
| `analyze.py` | figures (`assets/plots/`) and sweep tables (`assets/tables/`) |

Every run trains for 600k environment steps with training seed 0 and is evaluated greedily on 20 seeds (10000–10019) that training never uses, under three conditions: a normal day, a busy day (+50% requests) and a windy day (+60% wind). Per-run logs are in `logs/<algo>/runXX/` (`hparams.json`, `metrics.csv`, `summary.json`), weights in `models/dqn/` and `models/pg/`, and the best run of each algorithm is copied to `models/*/<algo>_best/`.

## Results

Best run of each algorithm, greedy policy, mean shift return over 20 seeds never used in training (± standard deviation):

| Method | Best run | Normal day | Busy day (+50% requests) | Windy day (+60% wind) | Requests served | Crashes |
|---|---|---|---|---|---|---|
| **PPO** | run09, GAE λ 0.8, γ 0.995 | **+55.6** ± 19.1 | **+46.7** | +53.4 | 68% | 0 of 20 |
| DQN | run08, double DQN + soft target | +51.3 ± 19.5 | +36.4 | **+56.8** | 67% | 1 of 20 |
| A2C | run08, heavier critic, wider net | +49.4 ± 26.9 | +37.7 | +50.8 | 67% | 1 of 20 |
| REINFORCE | run02, higher learning rate | +36.8 ± 22.2 | +9.3 | +39.0 | 64% | 1 of 20 |
| Hand-written dispatcher | | +41.4 ± 26.8 | +5.4 | +37.8 | 69% | 0 of 20 |
| Always hold | | −83.4 | −118.4 | −83.4 | 0% | 0 of 20 |
| Random (allowed actions) | | −157.9 | −205.2 | −156.6 | 23% | 20 of 20 |

What the 40 runs show:

- **PPO is the best and the most reliable.** All 10 PPO runs learned a working policy (the worst scored +27.5), and 9 of 10 reached the dispatcher's level, typically within 150–230k steps.
- **DQN learns fastest and never collapsed.** All 10 runs reached dispatcher level, the quickest by about 105k steps, but its returns stay noisier from run to run than PPO's.
- **A2C is fragile.** Four of ten runs collapsed to "always hold" (−83.4): a higher learning rate, no entropy bonus, Monte Carlo advantages (λ = 1) and RMSprop with γ 0.95. Their policy entropy drops to zero within the first 50k steps (`assets/plots/pg_entropy.png`).
- **REINFORCE is the weakest.** Two runs collapsed and only one of ten reached dispatcher level, after about 455k steps. Without a critic to cut the variance of whole-shift returns, it needs far more experience.
- **Learning pays off when conditions change.** On a busy day the dispatcher's fixed rules fall to +5.4, while PPO keeps +46.7. PPO does not copy the dispatcher's habits: it leaves the hub with a single package about half the time (the dispatcher always waits to load two), and it recharges earlier, at 54% battery on average against the dispatcher's 41%.

Every run is in `assets/tables/<algo>_sweep.md`. The figures are in `assets/plots/`:

| File | What it shows |
|---|---|
| `reward_curves.png` | Training return and greedy return over time, best run of each algorithm |
| `sweep_curves.png` | All 10 runs of each algorithm, best one highlighted |
| `dqn_objective.png` | DQN's TD loss and Q-values for every run |
| `pg_entropy.png` | Policy entropy for PPO, A2C and REINFORCE |
| `convergence.png` | Steps each run took to match the dispatcher |
| `generalisation.png` | Normal, busy and windy days on unseen seeds |

## Repository layout

```
├── main.py / play.py        run the best agent in the 3D viewer
├── environment/
│   ├── custom_env.py        the Gymnasium environment and the hand-written dispatcher
│   ├── rendering.py         viewer server, JSON serialisation, policy API
│   └── web/                 three.js viewer (three.js 0.170 vendored, works offline)
├── training/                TorchRL training, sweeps and analysis
├── models/dqn/, models/pg/  trained weights
├── logs/                    per-run metrics, sweep tables, demo episode logs
├── assets/                  figures and tables for the report
└── tests/                   environment tests
```
