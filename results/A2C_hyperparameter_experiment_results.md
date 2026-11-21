# Hyperparameter experiments: A2C

Environment: `DroneDelivery-v0`, an 8-hour delivery shift. Every run trains for 600,000 steps with training seed 0.
"Return" is the greedy policy's mean shift return over 20 seeds that training never used (± standard deviation). The hand-written dispatcher scores +41.4 on the same seeds, and doing nothing all shift scores −83.4.
"Reached dispatcher" is the first step at which the greedy policy, checked every 5,000 steps on 5 other held-out seeds, matched +41.4. "Entropy" is the policy's entropy in nats at the first check and at the end.

Models and logs are in `models/pg/a2c/runNN/` and `logs/a2c/runNN/`.

Baseline: learning rate 0.0007, gamma 0.99, GAE lambda 0.95, entropy coefficient 0.01, critic coefficient 0.5, one update per 256-step rollout, Adam, two hidden layers of 128.

| Run | Change | Return | Reached dispatcher | Entropy | Observed behaviour |
|---|---|---|---|---|---|
| 0 | Baseline | +41.1 ± 20.5 | 350k | 0.56 → 0.35 | Matched the dispatcher on a normal day but fell apart on a busy one (+1.9). |
| 1 | Learning rate 0.0002 | +31.5 ± 22.2 | never | 0.94 → 0.53 | Too slow: it never passed +40, and lost on busy days (−6.3). |
| 2 | Learning rate 0.002 | −83.4 ± 18.5 | never | 0.27 → 0.01 | **Collapsed** to holding at the hub all shift. Its entropy was already 0.27 at the first check: the large steps made it certain before it had seen a delivery. |
| 3 | 64 steps per rollout | +17.9 ± 63.7 | 240k | 0.86 → 0.33 | Reached the dispatcher on the 5-seed check, but the final policy crashed in 6 of 20 shifts. Short rollouts give noisy advantage estimates, and battery management was the first thing to suffer. |
| 4 | 1,024 steps per rollout | +23.1 ± 16.3 | never | 0.92 → 0.42 | Only about 585 updates in 600,000 steps, too few to finish learning. |
| 5 | No entropy bonus | −83.4 ± 18.5 | never | not logged | **Collapsed** to holding. With no entropy term TorchRL does not log entropy, but the behaviour is the same as runs 2, 7 and 9. |
| 6 | Entropy coefficient 0.05 | +49.2 ± 33.1 | 160k | 0.69 → 0.65 | Second best and the fastest A2C run: the strong entropy bonus kept it exploring. One shift in 20 ended in a crash. |
| 7 | GAE lambda 1.0 (Monte Carlo) | −83.4 ± 18.5 | never | 0.63 → 0.00 | **Collapsed** to holding. Whole-shift returns are much noisier advantages than GAE's, and once the policy leaned towards holding it stopped trying anything else. |
| 8 | Critic coefficient 1.0, 256 units | **+49.4 ± 26.9** | 195k | 0.58 → 0.33 | **Best**. A heavier, wider critic gave better advantages, and it held up on busy days (+37.7). |
| 9 | RMSprop, gamma 0.95 | −83.4 ± 18.5 | never | 0.18 → 0.00 | **Collapsed** to holding almost at once (entropy 0.18 at the first check). |

## Summary

A2C was the most fragile method: four of ten runs collapsed into holding at the hub for the whole shift. Hold is always allowed and costs nothing at once, while early random flights often strand the drone or crash it, so a noisy single-step update easily makes holding look safest. Once the entropy reaches zero the agent never tries a delivery again, and holding scores −83.4 on every seed. Every change that made the updates noisier or larger (a high learning rate, no entropy bonus, Monte Carlo advantages, RMSprop) caused it, and the two changes that kept it exploring or gave it a better critic (runs 6 and 8) produced the best A2C agents.

Each run is a single seed, and with returns varying by 20 to 30 points from shift to shift, differences under about 5 points between runs are within noise.
