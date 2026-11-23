# DQN sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | buffer_size | batch_size | eps_end | eps_frames | target_update | target_interval | double_dqn | loss | hidden | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +46.5 ± 15.2 | +45.6 | +44.0 | 64% | 35k |
| run01 | lower learning rate | 0.0001 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +41.4 ± 20.7 | +27.4 | +36.9 | 63% | 55k |
| run02 | higher learning rate | 0.002 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +45.2 ± 16.9 | +28.9 | +41.5 | 62% | 35k |
| run03 | shorter horizon (gamma 0.95) | 0.0005 | 0.95 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +48.7 ± 19.1 | +34.8 | +50.9 | 64% | 40k |
| run04 | longer horizon (gamma 0.999) | 0.0005 | 0.999 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +44.0 ± 13.5 | +29.9 | +45.5 | 58% | 35k |
| run05 | small replay, small batch | 0.0005 | 0.99 | 5000 | 32 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +42.7 ± 19.1 | +26.6 | +45.2 | 63% | 45k |
| run06 | slow exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.1 | 150000 | hard | 500 | False | l2 | 128 | +48.1 ± 15.4 | +37.7 | +45.3 | 60% | 35k |
| **run07** | fast exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.01 | 15000 | hard | 500 | False | l2 | 128 | +48.9 ± 14.6 | +40.9 | +43.9 | 64% | 30k |
| run08 | double DQN + soft target | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | soft | 500 | True | l2 | 128 | +45.2 ± 21.7 | +33.0 | +43.0 | 59% | 15k |
| run09 | frequent target copy, Huber, wider net | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 50 | False | smooth_l1 | 256 | +44.1 ± 13.9 | +34.4 | +43.3 | 64% | 20k |
