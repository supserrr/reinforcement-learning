# REINFORCE sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +41.4. Returns are means over 20 unseen seeds.

| Run | Change from baseline | lr | gamma | baseline | normalize | episodes_per_update | entropy_coef | Return (unseen) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline (value baseline, normalised returns) | 0.0003 | 0.99 | value | True | 8 | 0.02 | +20.7 ± 15.8 | +7.5 | +19.2 | 56% | never |
| run01 | lower learning rate | 0.0001 | 0.99 | value | True | 8 | 0.02 | -87.0 ± 16.3 | -116.2 | -87.0 | 0% | never |
| **run02** | higher learning rate | 0.001 | 0.99 | value | True | 8 | 0.02 | +36.8 ± 22.2 | +9.3 | +39.0 | 64% | 455k |
| run03 | vanilla REINFORCE, no baseline | 0.0003 | 0.99 | none | True | 8 | 0.02 | -11.2 ± 16.3 | -23.2 | -12.1 | 44% | never |
| run04 | baseline, raw advantages | 0.0003 | 0.99 | value | False | 8 | 0.02 | -41.2 ± 21.4 | -48.1 | -38.5 | 24% | never |
| run05 | 2 episodes per update | 0.0003 | 0.99 | value | True | 2 | 0.02 | +22.8 ± 24.9 | +14.9 | +22.5 | 59% | never |
| run06 | 16 episodes per update | 0.0003 | 0.99 | value | True | 16 | 0.02 | -10.2 ± 19.1 | -28.9 | -13.4 | 45% | never |
| run07 | no entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.0 | -83.4 ± 18.5 | -118.4 | -83.4 | 0% | never |
| run08 | strong entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.08 | +17.6 ± 17.6 | +5.5 | +15.6 | 51% | never |
| run09 | shorter horizon (gamma 0.95) | 0.0003 | 0.95 | value | True | 8 | 0.02 | +12.2 ± 14.8 | +5.0 | +14.8 | 55% | never |
