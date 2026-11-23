# REINFORCE sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | baseline | normalize | episodes_per_update | entropy_coef | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline (value baseline, normalised returns) | 0.0003 | 0.99 | value | True | 8 | 0.02 | -31.4 ± 29.7 | -46.7 | -31.9 | 39% | never |
| run01 | lower learning rate | 0.0001 | 0.99 | value | True | 8 | 0.02 | -84.0 ± 18.4 | -117.4 | -84.0 | 0% | never |
| **run02** | higher learning rate | 0.001 | 0.99 | value | True | 8 | 0.02 | -6.1 ± 28.6 | -39.2 | -7.0 | 51% | never |
| run03 | vanilla REINFORCE, no baseline | 0.0003 | 0.99 | none | True | 8 | 0.02 | -86.6 ± 16.7 | -115.5 | -86.6 | 0% | never |
| run04 | baseline, raw advantages | 0.0003 | 0.99 | value | False | 8 | 0.02 | -87.1 ± 16.3 | -116.2 | -87.1 | 0% | never |
| run05 | 2 episodes per update | 0.0003 | 0.99 | value | True | 2 | 0.02 | -6.5 ± 22.9 | -24.0 | -6.5 | 41% | never |
| run06 | 16 episodes per update | 0.0003 | 0.99 | value | True | 16 | 0.02 | -57.1 ± 26.2 | -59.9 | -57.1 | 20% | never |
| run07 | no entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.0 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| run08 | strong entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.08 | -12.2 ± 20.2 | -21.8 | -6.8 | 39% | never |
| run09 | shorter horizon (gamma 0.95) | 0.0003 | 0.95 | value | True | 8 | 0.02 | -83.8 ± 19.2 | -118.2 | -83.2 | 0% | never |
