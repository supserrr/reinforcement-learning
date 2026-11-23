# Final comparison on 200 test shifts

Each algorithm's best run was picked on 20 validation seeds (see the sweep tables). These are 200 different seeds (30000–30199) that neither training nor run selection used, and every method plays the same shifts. Test returns are mean ± standard deviation over the shifts (the standard error of a mean is about a 14th of that). "Validation return" is the normal-day score the run was picked on.

| Method | Best run (picked on validation) | Validation return | Test: normal day | Busy day | Windy day | Service rate | Crash rate (normal / busy) | Emergencies delivered | Emergencies missed | Mean emergency wait (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| DQN | run04 (longer horizon (gamma 0.999)) | +38.0 | +27.7 ± 20.4 | +17.0 | +27.3 | 55% | 2% / 0% | 5.0 | 2.1 | 21.7 |
| PPO | run02 (higher learning rate) | +31.5 | +25.0 ± 23.1 | +6.1 | +22.8 | 51% | 1% / 4% | 5.0 | 1.9 | 21.7 |
| A2C | run06 (strong entropy bonus) | +38.4 | +22.8 ± 31.2 | +2.5 | +22.4 | 56% | 4% / 10% | 4.8 | 2.0 | 22.3 |
| REINFORCE | run02 (higher learning rate) | -6.1 | -12.0 ± 27.3 | -30.5 | -12.6 | 48% | 6% / 15% | 2.9 | 3.8 | 19.1 |
| Hand-written dispatcher | n/a | +11.6 | +9.3 ± 23.7 | -12.1 | +8.7 | 51% | 0% / 0% |  |  |  |
| Always hold | n/a | -83.4 | -85.4 ± 19.5 | -112.4 | -85.4 | 0% | 0% / 0% |  |  |  |
| Random (masked) | n/a | -154.1 | -154.9 ± 14.5 | -206.6 | -156.2 | 24% | 100% / 100% |  |  |  |

## Paired differences from DQN

DQN scored highest on normal test days. Each cell is the method's return minus DQN's on the same shift: mean ± standard error, and the share of shifts where the method scored higher. A difference within about two standard errors of zero is not a real gap.

| Method | Normal day | Busy day | Windy day |
|---|---|---|---|
| PPO | -2.6 ± 1.8 (48%) | -10.9 ± 2.5 (42%) | -4.5 ± 1.9 (44%) |
| A2C | -4.9 ± 2.5 (47%) | -14.6 ± 3.2 (40%) | -4.9 ± 2.6 (44%) |
| REINFORCE | -39.7 ± 2.3 (6%) | -47.5 ± 3.0 (10%) | -39.9 ± 2.3 (7%) |
| Hand-written dispatcher | -18.3 ± 2.0 (26%) | -29.1 ± 2.1 (16%) | -18.6 ± 1.9 (22%) |


## Seed check

The best configuration of each method, retrained on training seeds 0, 1 and 2 and played on the same 200 test shifts (normal day). Seed 0 is the sweep run.

| Method | Configuration | Validation by training seed | Test by training seed | Test across seeds (mean ± sd) |
|---|---|---|---|---|
| DQN | run04 | +38.0 / +25.4 / +25.6 | +27.7 / +22.8 / +24.9 | +25.1 ± 2.5 |
| PPO | run02 | +31.5 / +6.8 / +28.3 | +25.0 / +11.7 / +19.5 | +18.7 ± 6.7 |
| A2C | run06 | +38.4 / +16.0 / +13.1 | +22.8 / +11.7 / +17.0 | +17.2 ± 5.6 |
| REINFORCE | run02 | -6.1 / +7.2 / -10.9 | -12.0 / +3.5 / -7.7 | -5.4 ± 8.0 |
