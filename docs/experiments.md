# Hyperparameter experiments

Each algorithm ran 10 experiments of 600,000 steps with training seed 0, on the same environment. Every experiment starts from that algorithm's baseline and changes one or two of its own controls, so each result shows mostly the effect of those controls. All 40 runs are evaluated the same way: the greedy policy on 20 seeds training never used, on a normal, a busy and a windy day.

| Algorithm | Results and observations |
|---|---|
| DQN | [DQN_hyperparameter_experiment_results.md](../results/DQN_hyperparameter_experiment_results.md) |
| PPO | [PPO_hyperparameter_experiment_results.md](../results/PPO_hyperparameter_experiment_results.md) |
| A2C | [A2C_hyperparameter_experiment_results.md](../results/A2C_hyperparameter_experiment_results.md) |
| REINFORCE | [REINFORCE_hyperparameter_experiment_results.md](../results/REINFORCE_hyperparameter_experiment_results.md) |

## Values tested

**DQN.** Baseline: learning rate 0.0005, gamma 0.99, buffer 50,000, batch 128, epsilon 1.0 to 0.05 over 60,000 steps, hard target copy every 500 updates, squared loss, 128 units.

| Hyperparameter | Values |
|---|---|
| Learning rate | 0.0001, 0.0005, 0.002 |
| Gamma | 0.95, 0.99, 0.999 |
| Replay buffer, batch | 5,000 and 32; 50,000 and 128 |
| Epsilon end, decay steps | 0.01 over 15,000; 0.05 over 60,000; 0.10 over 150,000 |
| Target update | hard every 50 or 500 updates; soft, tau 0.01 |
| Double DQN | off, on |
| Loss, hidden units | squared loss and 128; Huber loss and 256 |

**PPO.** Baseline: learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01, critic 0.5, 2,048 steps per rollout, 4 epochs, minibatch 256.

| Hyperparameter | Values |
|---|---|
| Learning rate | 0.0001, 0.0003, 0.001 |
| Clip range | 0.1, 0.2, 0.3 |
| Entropy coefficient | 0, 0.01, 0.05 |
| Steps per rollout, minibatch | 512 and 128; 2,048 and 256 |
| Epochs per rollout | 4, 10 |
| GAE lambda, gamma | 0.95 and 0.99; 0.8 and 0.995 |

**A2C.** Baseline: learning rate 0.0007, gamma 0.99, GAE lambda 0.95, entropy 0.01, critic 0.5, 256 steps per update, Adam, 128 units.

| Hyperparameter | Values |
|---|---|
| Learning rate | 0.0002, 0.0007, 0.002 |
| Steps per update | 64, 256, 1,024 |
| Entropy coefficient | 0, 0.01, 0.05 |
| GAE lambda | 0.95, 1.0 |
| Critic coefficient, hidden units | 0.5 and 128; 1.0 and 256 |
| Optimiser, gamma | Adam and 0.99; RMSprop and 0.95 |

**REINFORCE.** Baseline: learning rate 0.0003, gamma 0.99, value baseline, normalised advantages, 8 shifts per update, entropy 0.02.

| Hyperparameter | Values |
|---|---|
| Learning rate | 0.0001, 0.0003, 0.001 |
| Baseline | none, learned value |
| Advantage normalisation | off, on |
| Shifts per update | 2, 8, 16 |
| Entropy coefficient | 0, 0.02, 0.08 |
| Gamma | 0.95, 0.99 |

The exact configurations are `SWEEP` in [`training/dqn_training.py`](../training/dqn_training.py) and `SWEEPS` in [`training/pg_training.py`](../training/pg_training.py).

## Outputs

Each run saves to `logs/<algo>/runNN/` and `models/<dqn|pg>/<algo>/runNN/`:

| File | Content |
|---|---|
| `hparams.json` | the hyperparameters used |
| `metrics.csv` | every 5,000 steps: training return, greedy return on 5 held-out seeds, losses, entropy or epsilon |
| `summary.json` | the final evaluation on 20 unseen seeds for each day type |
| `model.pt` | the trained network's weights |

`logs/<algo>/sweep_summary.csv` and `assets/tables/<algo>_sweep.md` are generated tables of every run's numbers, and the best run is copied to `models/<dqn|pg>/<algo>_best/`.

## Running them

```bash
uv run python -m training.dqn_training --sweep --jobs 2                # all 10 DQN runs, two at a time
uv run python -m training.pg_training --algo ppo --sweep --jobs 2      # all 10 PPO runs (also a2c, reinforce)
uv run python -m training.pg_training --algo a2c --runs 6 8            # selected runs
uv run python -m training.analyze                                      # rebuild the figures and tables
```

On an M1 Pro each run took 5 to 14 minutes; all 40 ran in a little over an hour, eight at a time.
