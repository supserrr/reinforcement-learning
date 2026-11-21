# PPO sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +41.4. Returns are means over 20 unseen seeds.

| Run | Change from baseline | lr | gamma | lmbda | clip_epsilon | entropy_coef | frames_per_batch | epochs | minibatch | Return (unseen) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +55.1 ± 28.8 | +31.0 | +58.9 | 70% | 221k |
| run01 | lower learning rate | 0.0001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +27.5 ± 15.5 | +10.2 | +27.2 | 60% | never |
| run02 | higher learning rate | 0.001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +46.8 ± 17.9 | +40.5 | +45.7 | 68% | 162k |
| run03 | tight clip 0.1 | 0.0003 | 0.99 | 0.95 | 0.1 | 0.01 | 2048 | 4 | 256 | +47.2 ± 22.8 | +30.7 | +49.6 | 67% | 225k |
| run04 | loose clip 0.3 | 0.0003 | 0.99 | 0.95 | 0.3 | 0.01 | 2048 | 4 | 256 | +47.5 ± 24.1 | +37.9 | +48.9 | 69% | 176k |
| run05 | no entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.0 | 2048 | 4 | 256 | +50.9 ± 15.9 | +35.7 | +47.6 | 68% | 156k |
| run06 | strong entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.05 | 2048 | 4 | 256 | +44.1 ± 16.8 | +15.8 | +48.5 | 64% | 201k |
| run07 | short rollouts (512) | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 512 | 4 | 128 | +54.5 ± 17.7 | +18.0 | +47.7 | 69% | 170k |
| run08 | 10 epochs per batch | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 10 | 256 | +49.6 ± 39.7 | +39.2 | +44.2 | 69% | 152k |
| **run09** | GAE lambda 0.8, gamma 0.995 | 0.0003 | 0.995 | 0.8 | 0.2 | 0.01 | 2048 | 4 | 256 | +55.6 ± 19.1 | +46.7 | +53.4 | 68% | 162k |
