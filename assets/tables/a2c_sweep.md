# A2C sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.8. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | lmbda | entropy_coef | critic_coef | frames_per_batch | optimizer | hidden | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| run01 | lower learning rate | 0.0002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | -3.2 ± 33.1 | -57.8 | -0.7 | 47% | 495k |
| run02 | higher learning rate | 0.002 | 0.99 | 0.95 | 0.01 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| run03 | short rollouts (64 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 64 | adam | 128 | +11.4 ± 33.5 | -25.2 | +12.1 | 55% | 205k |
| run04 | long rollouts (1024 steps) | 0.0007 | 0.99 | 0.95 | 0.01 | 0.5 | 1024 | adam | 128 | +7.0 ± 17.0 | -11.3 | +9.2 | 46% | 541k |
| run05 | no entropy bonus | 0.0007 | 0.99 | 0.95 | 0.0 | 0.5 | 256 | adam | 128 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| **run06** | strong entropy bonus | 0.0007 | 0.99 | 0.95 | 0.05 | 0.5 | 256 | adam | 128 | +29.1 ± 21.6 | +15.1 | +27.0 | 58% | 210k |
| run07 | Monte Carlo advantage (lambda 1.0) | 0.0007 | 0.99 | 1.0 | 0.01 | 0.5 | 256 | adam | 128 | +8.5 ± 42.3 | -21.2 | +7.6 | 52% | 270k |
| run08 | heavier critic, wider net | 0.0007 | 0.99 | 0.95 | 0.01 | 1.0 | 256 | adam | 256 | +4.5 ± 43.8 | -23.9 | +1.8 | 57% | 255k |
| run09 | RMSprop, gamma 0.95 | 0.0007 | 0.95 | 0.95 | 0.01 | 0.5 | 256 | rmsprop | 128 | -73.5 ± 20.2 | -110.6 | -76.0 | 5% | never |
