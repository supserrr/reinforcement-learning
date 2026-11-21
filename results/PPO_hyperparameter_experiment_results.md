# Hyperparameter experiments: PPO

Environment: `DroneDelivery-v0`, an 8-hour delivery shift. Every run trains for 600,000 steps with training seed 0.
"Return" is the greedy policy's mean shift return over 20 seeds that training never used (± standard deviation). The hand-written dispatcher scores +41.4 on the same seeds, and doing nothing all shift scores −83.4.
"Reached dispatcher" is the first step at which the greedy policy, checked every 5,000 steps on 5 other held-out seeds, matched +41.4. "Entropy" is the policy's entropy in nats at the end of training (about 2.3 is uniform over the 10 actions).

Models and logs are in `models/pg/ppo/runNN/` and `logs/ppo/runNN/`.

Baseline: learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01, critic coefficient 0.5, 2,048 steps per rollout, 4 epochs, minibatches of 256, two hidden layers of 128.

| Run | Change | Return | Reached dispatcher | Entropy | Observed behaviour |
|---|---|---|---|---|---|
| 0 | Baseline | +55.1 ± 28.8 | 221k | 0.43 | Second best on a normal day. The wide spread comes from one shift that ended in a crash; it was also weak on busy days (+31.0). |
| 1 | Learning rate 0.0001 | +27.5 ± 15.5 | never | 0.46 | The only PPO run that never reached the dispatcher. It learned in the right direction but too slowly for 600,000 steps, peaking at +35. |
| 2 | Learning rate 0.001 | +46.8 ± 17.9 | 161k | 0.34 | Learned fast and committed early (lower entropy), and held up well on busy days (+40.5). |
| 3 | Clip 0.1 | +47.2 ± 22.8 | 225k | 0.48 | The small trust region slowed learning (the latest to reach the dispatcher), and it ended 8 of 20 shifts away from the hub. |
| 4 | Clip 0.3 | +47.5 ± 24.1 | 176k | 0.38 | Larger policy steps learned faster but less carefully: the most wasted trips of any PPO run (1.1 per shift) and half its shifts ended away from the hub. |
| 5 | No entropy bonus | +50.9 ± 15.9 | 155k | 0.32 | Did well without any entropy bonus: the clipped updates kept the policy from collapsing, unlike A2C and REINFORCE without one. |
| 6 | Entropy coefficient 0.05 | +44.1 ± 16.8 | 200k | 0.75 | The policy stayed the most random (entropy 0.75). Its best checks were high (+62) but it was the worst PPO run on a busy day (+15.8), where random choices cost the most. |
| 7 | 512 steps per rollout | +54.5 ± 17.7 | 170k | 0.36 | Four times as many updates on shorter rollouts gave a strong normal-day policy, but a weak busy-day one (+18.0). |
| 8 | 10 epochs per batch | +49.6 ± 39.7 | 151k | 0.37 | Reused each batch the most and learned fastest of all PPO runs, with the best final checks (+61.1). The large spread is one crashed shift. |
| 9 | GAE lambda 0.8, gamma 0.995 | **+55.6 ± 19.1** | 161k | 0.30 | **Best**, and the best busy-day score of any agent (+46.7). The lower lambda gives less noisy advantages, the higher gamma values getting back to the hub at the end of the shift, and its training episodes scored the highest (+49.8). |

## Summary

Nine of the ten PPO runs reached the dispatcher's level, between 151,000 and 225,000 steps; only the smallest learning rate fell short. PPO was the least sensitive of the four methods: clipping kept every run's entropy between 0.3 and 0.5 without help, so even the run with no entropy bonus did well, while too much entropy made the policy too random on busy days. The advantage estimate mattered most at the top: lowering GAE lambda to 0.8 and raising gamma to 0.995 gave the best agent, which `main.py` plays by default.

Each run is a single seed, and with returns varying by 15 to 30 points from shift to shift, differences under about 5 points between runs are within noise.
