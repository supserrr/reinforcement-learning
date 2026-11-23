# REINFORCE sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | baseline | normalize | episodes_per_update | entropy_coef | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline (value baseline, normalised returns) | 0.0003 | 0.99 | value | True | 8 | 0.02 | +15.7 ± 22.2 | +15.4 | +16.6 | 52% | 490k |
| run01 | lower learning rate | 0.0001 | 0.99 | value | True | 8 | 0.02 | -14.4 ± 16.4 | -29.0 | -13.9 | 42% | never |
| **run02** | higher learning rate | 0.001 | 0.99 | value | True | 8 | 0.02 | +37.0 ± 21.7 | +16.6 | +37.1 | 58% | 145k |
| run03 | vanilla REINFORCE, no baseline | 0.0003 | 0.99 | none | True | 8 | 0.02 | +20.2 ± 21.6 | +10.8 | +19.9 | 53% | 455k |
| run04 | baseline, raw advantages | 0.0003 | 0.99 | value | False | 8 | 0.02 | +21.7 ± 19.7 | +9.7 | +22.9 | 54% | 365k |
| run05 | 2 episodes per update | 0.0003 | 0.99 | value | True | 2 | 0.02 | +36.5 ± 22.0 | +22.9 | +36.4 | 60% | 195k |
| run06 | 16 episodes per update | 0.0003 | 0.99 | value | True | 16 | 0.02 | +9.8 ± 25.6 | -3.4 | +9.2 | 52% | 555k |
| run07 | no entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.0 | +25.8 ± 17.8 | +3.1 | +25.8 | 55% | 425k |
| run08 | strong entropy bonus | 0.0003 | 0.99 | value | True | 8 | 0.08 | +30.7 ± 23.3 | +19.9 | +29.9 | 53% | 390k |
| run09 | shorter horizon (gamma 0.95) | 0.0003 | 0.95 | value | True | 8 | 0.02 | +33.2 ± 22.5 | +17.9 | +33.1 | 59% | 310k |
