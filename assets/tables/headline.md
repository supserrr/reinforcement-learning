# Final comparison on 200 test shifts

Each algorithm's best run was picked on 20 validation seeds (see the sweep tables). These are 200 different seeds (30000–30199) that neither training nor run selection used, and every method plays the same shifts. Test returns are mean ± standard deviation over the shifts (the standard error of a mean is about a 14th of that). "Validation return" is the normal-day score the run was picked on.

| Method | Best run (picked on validation) | Validation return | Test: normal day | Busy day | Windy day | Service rate | Crash rate (normal / busy) | Emergencies delivered | Emergencies missed | Mean emergency wait (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| DQN | run07 (fast exploration decay) | +37.4 | +31.6 ± 21.1 | +19.3 | +31.6 | 53% | 1% / 2% | 5.3 | 1.9 | 21.0 |
| PPO | run06 (strong entropy bonus) | +43.0 | +31.5 ± 25.2 | +12.6 | +30.6 | 59% | 1% / 5% | 4.9 | 2.0 | 22.7 |
| A2C | run06 (strong entropy bonus) | +29.1 | +23.0 ± 18.9 | +6.4 | +23.3 | 56% | 0% / 4% | 4.4 | 2.6 | 21.1 |
| REINFORCE | run02 (higher learning rate) | +5.1 | +1.9 ± 26.6 | -9.2 | +2.3 | 48% | 4% / 2% | 3.8 | 3.2 | 23.1 |
| Hand-written dispatcher | n/a | +11.8 | +9.4 ± 24.1 | -11.6 | +8.8 | 54% | 0% / 0% |  |  |  |
| Always hold | n/a | -83.4 | -85.4 ± 19.5 | -112.4 | -85.4 | 0% | 0% / 0% |  |  |  |
| Random (masked) | n/a | -154.1 | -154.9 ± 14.5 | -206.6 | -156.2 | 24% | 100% / 100% |  |  |  |

## Paired differences from DQN

DQN scored highest on normal test days. Each cell is the method's return minus DQN's on the same shift: mean ± standard error, and the share of shifts where the method scored higher. A difference within about two standard errors of zero is not a real gap.

| Method | Normal day | Busy day | Windy day |
|---|---|---|---|
| PPO | -0.1 ± 1.9 (54%) | -6.7 ± 2.8 (45%) | -1.0 ± 1.9 (48%) |
| A2C | -8.6 ± 1.8 (33%) | -12.9 ± 2.9 (36%) | -8.4 ± 1.9 (39%) |
| REINFORCE | -29.7 ± 2.3 (14%) | -28.5 ± 2.9 (20%) | -29.4 ± 2.2 (14%) |
| Hand-written dispatcher | -22.2 ± 2.0 (20%) | -30.8 ± 2.5 (18%) | -22.8 ± 2.0 (22%) |
