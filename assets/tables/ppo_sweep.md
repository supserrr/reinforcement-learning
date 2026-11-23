# PPO sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | lmbda | clip_epsilon | entropy_coef | frames_per_batch | epochs | minibatch | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +45.5 ± 19.8 | +22.1 | +41.0 | 61% | 55k |
| run01 | lower learning rate | 0.0001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +29.4 ± 27.4 | +9.3 | +27.9 | 57% | 152k |
| run02 | higher learning rate | 0.001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +41.4 ± 20.9 | +23.4 | +39.6 | 60% | 37k |
| run03 | tight clip 0.1 | 0.0003 | 0.99 | 0.95 | 0.1 | 0.01 | 2048 | 4 | 256 | +36.2 ± 21.4 | +25.6 | +37.1 | 58% | 66k |
| run04 | loose clip 0.3 | 0.0003 | 0.99 | 0.95 | 0.3 | 0.01 | 2048 | 4 | 256 | +37.0 ± 14.8 | +22.4 | +38.3 | 57% | 90k |
| run05 | no entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.0 | 2048 | 4 | 256 | +45.1 ± 20.6 | +29.9 | +45.3 | 58% | 76k |
| run06 | strong entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.05 | 2048 | 4 | 256 | +38.6 ± 21.1 | +23.0 | +38.5 | 60% | 72k |
| **run07** | short rollouts (512) | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 512 | 4 | 128 | +49.9 ± 18.9 | +16.3 | +42.6 | 60% | 55k |
| run08 | 10 epochs per batch | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 10 | 256 | +43.9 ± 20.7 | +20.2 | +40.9 | 62% | 41k |
| run09 | GAE lambda 0.8, gamma 0.995 | 0.0003 | 0.995 | 0.8 | 0.2 | 0.01 | 2048 | 4 | 256 | +43.2 ± 17.6 | +23.9 | +42.8 | 58% | 55k |
