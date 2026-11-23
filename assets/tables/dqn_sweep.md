# DQN sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.6. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | buffer_size | batch_size | eps_end | eps_frames | target_update | target_interval | double_dqn | loss | hidden | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +26.4 ± 21.4 | +27.4 | +27.1 | 51% | 130k |
| run01 | lower learning rate | 0.0001 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +19.6 ± 18.4 | -4.9 | +20.1 | 55% | 265k |
| run02 | higher learning rate | 0.002 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +18.3 ± 29.5 | -2.7 | +13.8 | 48% | 335k |
| run03 | shorter horizon (gamma 0.95) | 0.0005 | 0.95 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +35.5 ± 20.1 | +15.0 | +27.6 | 55% | 115k |
| **run04** | longer horizon (gamma 0.999) | 0.0005 | 0.999 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +38.0 ± 17.4 | +14.5 | +39.8 | 56% | 135k |
| run05 | small replay, small batch | 0.0005 | 0.99 | 5000 | 32 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +33.8 ± 19.3 | +14.8 | +34.8 | 56% | 170k |
| run06 | slow exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.1 | 150000 | hard | 500 | False | l2 | 128 | +17.6 ± 24.9 | +3.6 | +21.8 | 53% | 255k |
| run07 | fast exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.01 | 15000 | hard | 500 | False | l2 | 128 | +34.3 ± 30.0 | +13.7 | +39.8 | 58% | 110k |
| run08 | double DQN + soft target | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | soft | 500 | True | l2 | 128 | +31.9 ± 27.5 | +15.6 | +27.6 | 58% | 230k |
| run09 | frequent target copy, Huber, wider net | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 50 | False | smooth_l1 | 256 | +23.4 ± 30.8 | -2.6 | +26.7 | 52% | 165k |
