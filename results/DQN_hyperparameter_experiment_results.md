# Hyperparameter experiments: DQN

Environment: `DroneDelivery-v0`, an 8-hour delivery shift. Every run trains for 600,000 steps with training seed 0.
"Return" is the greedy policy's mean shift return over 20 seeds that training never used (± standard deviation). The hand-written dispatcher scores +41.4 on the same seeds, and doing nothing all shift scores −83.4.
"Reached dispatcher" is the first step at which the greedy policy, checked every 5,000 steps on 5 other held-out seeds, matched +41.4.

I started from a baseline and changed one or two DQN controls per run, so each result shows mostly the effect of that control. Models and logs are in `models/dqn/dqn/runNN/` and `logs/dqn/runNN/`.

Baseline: learning rate 0.0005, gamma 0.99, replay buffer 50,000, batch 128, epsilon 1.0 to 0.05 over 60,000 steps, hard target copy every 500 updates, squared loss, two hidden layers of 128.

| Run | Change | Return | Reached dispatcher | Observed behaviour |
|---|---|---|---|---|
| 0 | Baseline | +28.0 ± 18.9 | 115k | Learned quickly and peaked near +58 on the 5-seed check, then slid back to about +38 by the end. It ended 1 shift in 5 away from the hub, which costs −15 each time. |
| 1 | Learning rate 0.0001 | +39.6 ± 19.9 | 275k | Slowest of the runs that learned, but steady: its Q-values stayed the flattest of all (about 1.5) and it was still improving when training stopped. |
| 2 | Learning rate 0.002 | +44.2 ± 19.1 | 175k | Good final greedy score, but training episodes averaged −9.0 at the end: with large steps the Q-values kept moving, so the policy was strong at the moment it was saved rather than consistently. |
| 3 | Gamma 0.95 | +50.0 ± 16.3 | 120k | Second best, and +42.2 on a busy day. Deadlines are at most 120 minutes away (about ten decisions), so a short horizon is enough, and its targets were the easiest to fit (lowest TD loss, 0.058). |
| 4 | Gamma 0.999 | +49.9 ± 15.1 | 180k | Almost the same score with the steadiest results (± 15.1). It values the whole shift, which gave it the largest Q-values (2.6), but it was weaker on busy days (+32.2). |
| 5 | Replay 5,000, batch 32 | +48.3 ± 15.3 | 155k | Worked well even though the buffer holds only about 80 shifts. Its learning curve was noisier and never passed +48 on the 5-seed check. |
| 6 | Epsilon to 0.10 over 150,000 steps | +39.5 ± 23.9 | 455k | Slowest to learn. One action in ten stayed random until the end, and in this environment a random flight can strand the drone, so training episodes still averaged −24.4. |
| 7 | Epsilon to 0.01 over 15,000 steps | +50.5 ± 17.8 | 105k | Fastest to reach the dispatcher. The action mask already rules out impossible commands, so little random exploration is needed, and its training episodes scored the highest of all DQN runs (+41.8). |
| 8 | Double DQN, soft target (tau 0.01) | **+51.3 ± 19.5** | 260k | **Best**, and the best windy-day score of any agent (+56.8). Its Q-values overshot to 6.6 early on, then settled at 1.6, and its last checks were the steadiest (+51.6). One shift in 20 ended in a crash. |
| 9 | Target copy every 50 updates, Huber loss, 256 units | +43.3 ± 18.3 | 120k | The Huber loss gave the lowest TD loss (0.035), but copying the target so often let the Q-values chase themselves up to 7.2. It did worst of the good runs on a windy day (+34.4). |

## Summary

Every DQN run learned, and every one matched the dispatcher at some point. The exploration schedule changed the speed most: ending exploration early reached the dispatcher's level after 105,000 steps, keeping 10% random moves delayed it to 455,000. Both discount factors I tried, 0.95 and 0.999, ended above the baseline's 0.99, which suggests the baseline's weaker result is partly the luck of one seed. The stabilisers in run 8 (double Q-learning and a soft target) gave the best and steadiest final policy, and the Q-value plot shows why they matter: the runs without them overestimated more.

Each run is a single seed, and with returns varying by 15 to 20 points from shift to shift, differences under about 5 points between runs are within noise.
