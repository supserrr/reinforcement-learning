# Best run of each method vs baselines (20 unseen seeds per condition)

| Method | Best run | Normal day | Busy day | Windy day | Service rate | Crash rate | Emergencies delivered | Emergencies missed | Mean emergency wait (min) |
|---|---|---|---|---|---|---|---|---|---|
| DQN | run08 (double DQN + soft target) | +51.3 ± 19.5 | +36.4 | +56.8 | 67% | 5% | 6.2 | 1.4 | 19.0 |
| PPO | run09 (GAE lambda 0.8, gamma 0.995) | +55.6 ± 19.1 | +46.7 | +53.4 | 68% | 0% | 6.2 | 1.4 | 20.8 |
| A2C | run08 (heavier critic, wider net) | +49.4 ± 26.9 | +37.7 | +50.8 | 67% | 5% | 5.9 | 1.2 | 23.8 |
| REINFORCE | run02 (higher learning rate) | +36.8 ± 22.2 | +9.3 | +39.0 | 64% | 5% | 5.0 | 2.0 | 18.5 |
| Hand-written dispatcher | n/a | +41.4 ± 26.8 | +5.4 | +37.8 | 69% | 0% | | | |
| Always hold | n/a | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | 0% | | | |
| Random (masked) | n/a | -157.9 ± 13.5 | -205.2 | -156.6 | 23% | 100% | | | |
