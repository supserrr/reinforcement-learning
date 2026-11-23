# Final comparison on 200 test shifts

Each algorithm's best run was picked on 20 validation seeds (see the sweep tables). These are 200 different seeds (30000–30199) that neither training nor run selection used, and every method plays the same shifts. Test returns are mean ± standard deviation over the shifts (the standard error of a mean is about a 14th of that). "Validation return" is the normal-day score the run was picked on.

| Method | Best run (picked on validation) | Validation return | Test: normal day | Busy day | Windy day | Service rate | Crash rate (normal / busy) | Ends away or in the air | Emergencies delivered | Emergencies missed | Mean emergency wait (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DQN | run07 (fast exploration decay) | +48.9 | +44.8 ± 14.4 | +33.4 | +43.8 | 63% | 0% / 0% | 0% | 5.1 | 1.7 | 23.6 |
| PPO | run07 (short rollouts (512)) | +49.9 | +35.7 ± 18.5 | +21.9 | +35.5 | 56% | 0% / 0% | 0% | 5.0 | 1.8 | 21.9 |
| A2C | run02 (higher learning rate) | +42.0 | +38.1 ± 20.9 | +19.3 | +38.5 | 58% | 0% / 0% | 0% | 5.0 | 1.9 | 21.3 |
| REINFORCE | run02 (higher learning rate) | +37.0 | +37.2 ± 18.9 | +20.2 | +36.9 | 58% | 0% / 0% | 0% | 5.0 | 2.0 | 22.0 |
| Hand-written dispatcher | n/a | +11.6 | +9.3 ± 23.7 | -12.1 | +8.7 | 51% | 0% / 0% | 0% |  |  |  |
| Always hold | n/a | -83.4 | -85.4 ± 19.5 | -112.4 | -85.4 | 0% | 0% / 0% | 0% |  |  |  |
| Random (masked) | n/a | -54.9 | -56.9 ± 24.2 | -75.8 | -52.3 | 22% | 0% / 0% | 0% |  |  |  |

## Paired differences from DQN

DQN scored highest on normal test days. Each cell is the method's return minus DQN's on the same shift: mean ± standard error, and the share of shifts where the method scored higher. A difference within about two standard errors of zero is not a real gap.

| Method | Normal day | Busy day | Windy day |
|---|---|---|---|
| PPO | -9.1 ± 1.6 (36%) | -11.5 ± 2.1 (38%) | -8.2 ± 1.6 (32%) |
| A2C | -6.6 ± 1.8 (42%) | -14.1 ± 2.1 (33%) | -5.2 ± 1.7 (42%) |
| REINFORCE | -7.6 ± 1.6 (34%) | -13.1 ± 1.9 (34%) | -6.9 ± 1.6 (36%) |
| Hand-written dispatcher | -35.4 ± 1.9 (6%) | -45.4 ± 2.2 (6%) | -35.1 ± 1.9 (6%) |


## Seed check

The best configuration of each method, retrained on training seeds 0, 1 and 2 and played on the same 200 test shifts (normal day). Seed 0 is the sweep run.

| Method | Configuration | Validation by training seed | Test by training seed | Test across seeds (mean ± sd) |
|---|---|---|---|---|
| DQN | run07 | +48.9 / +46.1 / +49.1 | +44.8 / +44.5 / +41.2 | +43.5 ± 2.0 |
| PPO | run07 | +49.9 / +39.5 / +31.6 | +35.7 / +34.0 / +31.7 | +33.8 ± 2.0 |
| A2C | run02 | +42.0 / +36.7 / +37.9 | +38.1 / +34.1 / +34.7 | +35.6 ± 2.2 |
| REINFORCE | run02 | +37.0 / +35.0 / +40.6 | +37.2 / +32.8 / +36.5 | +35.5 ± 2.4 |
