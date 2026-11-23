# A2C sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | lmbda | entropy_coef | critic_coef | frames_per_batch | optimizer | hidden | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | +40.6 ± 17.7 | +13.1 | +41.0 | 59% | 120k |
| run01 | lower learning rate | 0.0002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | +28.8 ± 19.9 | +11.3 | +28.4 | 58% | 345k |
| **run02** | higher learning rate | 0.002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | +42.0 ± 19.8 | +23.6 | +43.2 | 58% | 75k |
| run03 | short rollouts (64 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 64 | adam | 128 | +32.9 ± 21.8 | +12.8 | +31.1 | 56% | 85k |
| run04 | long rollouts (1024 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 1024 | adam | 128 | +33.2 ± 23.8 | +17.6 | +32.9 | 58% | 260k |
| run05 | no entropy bonus | 0.0007 | 0.99 | 0.95 | 0.0 | 0.5 | 256 | adam | 128 | +35.8 ± 23.1 | +25.4 | +33.9 | 56% | 125k |
| run06 | strong entropy bonus | 0.0007 | 0.99 | 0.95 | 0.05 | 0.5 | 256 | adam | 128 | +41.0 ± 22.0 | +12.3 | +44.1 | 59% | 105k |
| run07 | Monte Carlo advantage (lambda 1.0) | 0.0007 | 0.99 | 1.0 | 0.01 | 0.5 | 256 | adam | 128 | +35.0 ± 23.8 | +14.9 | +35.9 | 59% | 185k |
| run08 | heavier critic, wider net | 0.0007 | 0.99 | 0.95 | 0.01 | 1.0 | 256 | adam | 256 | +39.0 ± 23.6 | +11.3 | +36.6 | 61% | 75k |
| run09 | RMSprop, gamma 0.95 | 0.0007 | 0.95 | 0.95 | 0.01 | 0.5 | 256 | rmsprop | 128 | +37.0 ± 17.5 | +21.1 | +40.1 | 58% | 125k |
