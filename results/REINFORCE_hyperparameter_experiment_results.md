# Hyperparameter experiments: REINFORCE

Environment: `DroneDelivery-v0`, an 8-hour delivery shift. Every run trains for 600,000 steps with training seed 0.
"Return" is the greedy policy's mean shift return over 20 seeds that training never used (± standard deviation). The hand-written dispatcher scores +41.4 on the same seeds, and doing nothing all shift scores −83.4.
"Reached dispatcher" is the first step at which the greedy policy, checked every 5,000 steps on 5 other held-out seeds, matched +41.4. "Entropy" is the policy's entropy in nats at the first check and at the end.

Each update uses whole shifts: the discounted Monte Carlo return of every step, minus a learned value baseline where one is used. Models and logs are in `models/pg/reinforce/runNN/` and `logs/reinforce/runNN/`.

Baseline: learning rate 0.0003, gamma 0.99, value baseline, normalised advantages, 8 shifts per update, entropy coefficient 0.02, two hidden layers of 128.

| Run | Change | Return | Reached dispatcher | Entropy | Observed behaviour |
|---|---|---|---|---|---|
| 0 | Baseline | +20.7 ± 15.8 | never | 0.90 → 0.55 | Learned steadily but slowly, peaking at +26. |
| 1 | Learning rate 0.0001 | −87.0 ± 16.3 | never | 0.95 → 0.09 | Drifted into holding at the hub and never recovered. With small steps it did not find deliveries before the policy settled on the safe choice. |
| 2 | Learning rate 0.001 | **+36.8 ± 22.2** | 455k | 0.58 → 0.50 | **Best**, and the only REINFORCE run to match the dispatcher, late in training. Larger steps let it make use of the few good shifts in each batch. |
| 3 | No baseline | −11.2 ± 16.3 | never | 0.90 → 0.55 | Without a baseline every action in a shift is pushed up or down by the same total return, so it learned little and wasted the most trips (2.0 per shift). |
| 4 | Raw advantages (not normalised) | −41.2 ± 21.4 | never | 0.96 → 0.63 | Update sizes followed the size of the returns, which vary a lot between shifts. It served only 24% of requests. |
| 5 | 2 shifts per update | +22.8 ± 24.9 | never | 0.92 → 0.45 | Four times as many, noisier updates did slightly better than the baseline. |
| 6 | 16 shifts per update | −10.2 ± 19.1 | never | 0.94 → 0.62 | Half as many updates as the baseline, too few to learn in 600,000 steps. |
| 7 | No entropy bonus | −83.4 ± 18.5 | never | 0.90 → 0.06 | **Collapsed** to holding, like the A2C runs without one. |
| 8 | Entropy coefficient 0.08 | +17.6 ± 17.6 | never | 0.92 → 1.01 | The policy became more random over training (entropy rose to 1.0), which held it back. |
| 9 | Gamma 0.95 | +12.2 ± 14.8 | never | 0.88 → 0.55 | Still improving at the end (+26 on its last checks), so a shorter horizon may simply need longer. |

## Summary

REINFORCE was the weakest method: only one run matched the dispatcher, and two collapsed into holding. A shift is 40 to 80 decisions long, so the Monte Carlo return behind each step carries the luck of everything that happened after it. The learned baseline and normalised advantages were both essential (without them returns fell to −11 and −41), the learning rate was the strongest lever, and the entropy bonus had to be in a narrow range: none collapsed the policy, too much kept it random.

Each run is a single seed, and with returns varying by 15 to 25 points from shift to shift, differences under about 5 points between runs are within noise.
