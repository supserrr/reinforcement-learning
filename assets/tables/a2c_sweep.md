# A2C sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +41.4. Returns are means over 20 unseen seeds.

| Run | Change from baseline | lr | gamma | lmbda | entropy_coef | critic_coef | frames_per_batch | optimizer | hidden | Return (unseen) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | +41.1 ± 20.5 | +1.9 | +40.5 | 61% | 350k |
| run01 | lower learning rate | 0.0002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | +31.5 ± 22.2 | -6.3 | +31.0 | 60% | never |
| run02 | higher learning rate | 0.002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| run03 | short rollouts (64 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 64 | adam | 128 | +17.9 ± 63.7 | +5.1 | +16.1 | 66% | 240k |
| run04 | long rollouts (1024 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 1024 | adam | 128 | +23.1 ± 16.3 | +12.7 | +21.9 | 57% | never |
| run05 | no entropy bonus | 0.0007 | 0.99 | 0.95 | 0.0 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| run06 | strong entropy bonus | 0.0007 | 0.99 | 0.95 | 0.05 | 0.5 | 256 | adam | 128 | +49.2 ± 33.1 | +37.2 | +47.6 | 68% | 160k |
| run07 | Monte Carlo advantage (lambda 1.0) | 0.0007 | 0.99 | 1.0 | 0.01 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| **run08** | heavier critic, wider net | 0.0007 | 0.99 | 0.95 | 0.01 | 1.0 | 256 | adam | 256 | +49.4 ± 26.9 | +37.7 | +50.8 | 67% | 195k |
| run09 | RMSprop, gamma 0.95 | 0.0007 | 0.95 | 0.95 | 0.01 | 0.5 | 256 | rmsprop | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
