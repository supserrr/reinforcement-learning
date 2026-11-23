# PPO sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | lmbda | clip_epsilon | entropy_coef | frames_per_batch | epochs | minibatch | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +21.4 ± 28.9 | +10.3 | +24.0 | 54% | 281k |
| run01 | lower learning rate | 0.0001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| **run02** | higher learning rate | 0.001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +31.5 ± 18.1 | -2.2 | +29.2 | 53% | 172k |
| run03 | tight clip 0.1 | 0.0003 | 0.99 | 0.95 | 0.1 | 0.01 | 2048 | 4 | 256 | +13.9 ± 39.7 | +7.9 | +8.2 | 44% | 287k |
| run04 | loose clip 0.3 | 0.0003 | 0.99 | 0.95 | 0.3 | 0.01 | 2048 | 4 | 256 | +11.4 ± 21.4 | -3.2 | +15.0 | 53% | 451k |
| run05 | no entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.0 | 2048 | 4 | 256 | +18.9 ± 23.3 | -1.9 | +17.1 | 53% | 475k |
| run06 | strong entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.05 | 2048 | 4 | 256 | +19.9 ± 38.4 | +17.4 | +22.5 | 57% | 180k |
| run07 | short rollouts (512) | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 512 | 4 | 128 | +27.9 ± 29.3 | +2.0 | +30.7 | 56% | 360k |
| run08 | 10 epochs per batch | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 10 | 256 | +26.6 ± 39.0 | +2.0 | +24.9 | 59% | 152k |
| run09 | GAE lambda 0.8, gamma 0.995 | 0.0003 | 0.995 | 0.8 | 0.2 | 0.01 | 2048 | 4 | 256 | +15.5 ± 37.1 | +1.3 | +19.1 | 56% | 236k |
