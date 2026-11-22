# REINFORCE sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.8. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | baseline | normalize | episodes_per_update | entropy_coef | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline (value baseline, normalised returns) | 0.0003 | 0.99 | value | True | 8 | 0.02 | -85.2 ± 17.1 | -115.9 | -85.2 | 0% | never |
| run01 | lower learning rate | 0.0001 | 0.99 | value | True | 8 | 0.02 | -87.0 ± 16.3 | -116.2 | -87.0 | 0% | never |
| **run02** | higher learning rate | 0.001 | 0.99 | value | True | 8 | 0.02 | +5.1 ± 40.4 | -19.5 | +1.9 | 50% | never |
| run03 | vanilla REINFORCE, no baseline | 0.0003 | 0.99 | none | True | 8 | 0.02 | -15.3 ± 21.9 | -33.2 | -17.7 | 42% | never |
| run04 | baseline, raw advantages | 0.0003 | 0.99 | value | False | 8 | 0.02 | -87.0 ± 16.3 | -116.2 | -87.0 | 0% | never |
| run05 | 2 episodes per update | 0.0003 | 0.99 | value | True | 2 | 0.02 | +1.9 ± 30.0 | -12.7 | +0.9 | 43% | 510k |
| run06 | 16 episodes per update | 0.0003 | 0.99 | value | True | 16 | 0.02 | -84.3 ± 20.1 | -116.4 | -84.2 | 0% | never |
| run07 | no entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.0 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| run08 | strong entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.08 | -17.0 ± 22.6 | -13.8 | -16.5 | 37% | never |
| run09 | shorter horizon (gamma 0.95) | 0.0003 | 0.95 | value | True | 8 | 0.02 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
