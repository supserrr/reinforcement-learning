# DQN sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +11.8. Returns are means over 20 validation seeds that no run trained on. The best run (bold) is picked on these scores, which flatters it, so the final comparison in headline.md uses 200 separate test seeds.

| Run | Change from baseline | lr | gamma | buffer_size | batch_size | eps_end | eps_frames | target_update | target_interval | double_dqn | loss | hidden | Return (validation) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +18.8 ± 20.1 | +14.6 | +17.9 | 54% | 115k |
| run01 | lower learning rate | 0.0001 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +27.4 ± 25.1 | +16.4 | +28.7 | 58% | 190k |
| run02 | higher learning rate | 0.002 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +26.6 ± 12.7 | +9.6 | +24.0 | 50% | 125k |
| run03 | shorter horizon (gamma 0.95) | 0.0005 | 0.95 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +31.7 ± 41.6 | +15.1 | +34.3 | 60% | 110k |
| run04 | longer horizon (gamma 0.999) | 0.0005 | 0.999 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +22.9 ± 19.1 | +3.9 | +20.2 | 48% | 110k |
| run05 | small replay, small batch | 0.0005 | 0.99 | 5000 | 32 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +30.4 ± 13.0 | +15.7 | +30.4 | 52% | 220k |
| run06 | slow exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.1 | 150000 | hard | 500 | False | l2 | 128 | +26.8 ± 19.1 | +20.1 | +26.5 | 46% | 195k |
| **run07** | fast exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.01 | 15000 | hard | 500 | False | l2 | 128 | +37.4 ± 22.9 | +23.1 | +30.3 | 56% | 90k |
| run08 | double DQN + soft target | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | soft | 500 | True | l2 | 128 | +27.1 ± 18.9 | +6.7 | +20.2 | 54% | 165k |
| run09 | frequent target copy, Huber, wider net | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 50 | False | smooth_l1 | 256 | +29.6 ± 20.7 | +18.1 | +27.9 | 48% | 105k |
