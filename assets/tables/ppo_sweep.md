# PPO sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.8. Returns are means over 20 unseen seeds.

| Run | Change from baseline | lr | gamma | lmbda | clip_epsilon | entropy_coef | frames_per_batch | epochs | minibatch | Return (unseen) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +20.9 ± 37.3 | +16.3 | +23.7 | 56% | 236k |
| run01 | lower learning rate | 0.0001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +18.6 ± 12.7 | -5.5 | +17.1 | 52% | 371k |
| run02 | higher learning rate | 0.001 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 4 | 256 | +23.2 ± 37.7 | +11.5 | +24.7 | 55% | 72k |
| run03 | tight clip 0.1 | 0.0003 | 0.99 | 0.95 | 0.1 | 0.01 | 2048 | 4 | 256 | +22.5 ± 44.3 | +13.8 | +22.5 | 55% | 197k |
| run04 | loose clip 0.3 | 0.0003 | 0.99 | 0.95 | 0.3 | 0.01 | 2048 | 4 | 256 | +29.9 ± 23.2 | +14.2 | +25.4 | 57% | 252k |
| run05 | no entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.0 | 2048 | 4 | 256 | +31.5 ± 25.9 | +21.5 | +32.8 | 57% | 231k |
| **run06** | strong entropy bonus | 0.0003 | 0.99 | 0.95 | 0.2 | 0.05 | 2048 | 4 | 256 | +43.0 ± 29.1 | +9.1 | +34.3 | 63% | 141k |
| run07 | short rollouts (512) | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 512 | 4 | 128 | +12.1 ± 52.3 | +6.7 | +7.2 | 56% | 155k |
| run08 | 10 epochs per batch | 0.0003 | 0.99 | 0.95 | 0.2 | 0.01 | 2048 | 10 | 256 | +32.6 ± 37.4 | -4.9 | +41.6 | 60% | 111k |
| run09 | GAE lambda 0.8, gamma 0.995 | 0.0003 | 0.995 | 0.8 | 0.2 | 0.01 | 2048 | 4 | 256 | +40.0 ± 15.4 | +18.6 | +42.8 | 59% | 236k |
