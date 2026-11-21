# Best run of each method vs baselines (20 unseen seeds per condition)

| Method | Best run | Normal day | Busy day | Windy day | Service rate | Crash rate | Emergencies delivered | Emergencies missed | Mean emergency wait (min) |
|---|---|---|---|---|---|---|---|---|---|
| DQN | run07 (fast exploration decay) | +37.4 ± 22.9 | +23.1 | +30.3 | 56% | 0% | 5.8 | 1.7 | 19.9 |
| PPO | run06 (strong entropy bonus) | +43.0 ± 29.1 | +9.1 | +34.3 | 63% | 0% | 5.5 | 1.6 | 22.1 |
| A2C | run06 (strong entropy bonus) | +29.1 ± 21.6 | +15.1 | +27.0 | 58% | 0% | 4.9 | 2.5 | 21.4 |
| REINFORCE | run02 (higher learning rate) | +5.1 ± 40.4 | -19.5 | +1.9 | 50% | 10% | 4.0 | 2.5 | 22.5 |
| Hand-written dispatcher | n/a | +11.8 ± 24.9 | +0.9 | +12.2 | 56% | 0% | | | |
| Always hold | n/a | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | 0% | | | |
| Random (masked) | n/a | -154.1 ± 11.3 | -209.4 | -152.3 | 16% | 100% | | | |
