# A2C sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | lmbda | entropy_coef | critic_coef | frames_per_batch | optimizer | hidden | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -117.9 | -83.4 | 0% | never |
| run01 | lower learning rate | 0.0002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | -23.7 ± 27.3 | -53.2 | -27.0 | 42% | never |
| run02 | higher learning rate | 0.002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | -84.2 ± 18.5 | -118.6 | -84.2 | 0% | never |
| run03 | short rollouts (64 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 64 | adam | 128 | -13.7 ± 45.7 | -40.7 | -5.9 | 50% | never |
| run04 | long rollouts (1024 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 1024 | adam | 128 | -20.9 ± 28.5 | -78.0 | -20.8 | 44% | never |
| run05 | no entropy bonus | 0.0007 | 0.99 | 0.95 | 0.0 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| **run06** | strong entropy bonus | 0.0007 | 0.99 | 0.95 | 0.05 | 0.5 | 256 | adam | 128 | +38.4 ± 19.0 | +4.3 | +36.6 | 58% | 375k |
| run07 | Monte Carlo advantage (lambda 1.0) | 0.0007 | 0.99 | 1.0 | 0.01 | 0.5 | 256 | adam | 128 | -12.4 ± 26.0 | -22.2 | -10.8 | 47% | never |
| run08 | heavier critic, wider net | 0.0007 | 0.99 | 0.95 | 0.01 | 1.0 | 256 | adam | 256 | -9.4 ± 54.0 | -19.8 | -12.8 | 52% | never |
| run09 | RMSprop, gamma 0.95 | 0.0007 | 0.95 | 0.95 | 0.01 | 0.5 | 256 | rmsprop | 128 | -51.4 ± 29.7 | -71.9 | -51.4 | 14% | never |
