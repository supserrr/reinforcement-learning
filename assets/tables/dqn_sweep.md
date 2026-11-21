# DQN sweep (10 runs, 600k steps each, same environment and training seed)

Dispatcher level = +41.4. Returns are means over 20 unseen seeds.

| Run | Change from baseline | lr | gamma | buffer_size | batch_size | eps_end | eps_frames | target_update | target_interval | double_dqn | loss | hidden | Return (unseen) | Busy | Windy | Service | Steps to dispatcher level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run00 | baseline | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +28.0 ± 18.9 | +24.4 | +34.0 | 53% | 115k |
| run01 | lower learning rate | 0.0001 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +39.6 ± 19.9 | +33.4 | +45.6 | 66% | 275k |
| run02 | higher learning rate | 0.002 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +44.2 ± 19.1 | +31.0 | +50.0 | 62% | 175k |
| run03 | shorter horizon (gamma 0.95) | 0.0005 | 0.95 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +50.0 ± 16.3 | +42.2 | +47.2 | 64% | 120k |
| run04 | longer horizon (gamma 0.999) | 0.0005 | 0.999 | 50000 | 128 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +49.9 ± 15.1 | +32.2 | +48.5 | 63% | 180k |
| run05 | small replay, small batch | 0.0005 | 0.99 | 5000 | 32 | 0.05 | 60000 | hard | 500 | False | l2 | 128 | +48.3 ± 15.3 | +28.0 | +42.0 | 64% | 155k |
| run06 | slow exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.1 | 150000 | hard | 500 | False | l2 | 128 | +39.5 ± 23.9 | +26.8 | +36.9 | 56% | 455k |
| run07 | fast exploration decay | 0.0005 | 0.99 | 50000 | 128 | 0.01 | 15000 | hard | 500 | False | l2 | 128 | +50.5 ± 17.8 | +41.4 | +47.4 | 60% | 105k |
| **run08** | double DQN + soft target | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | soft | 500 | True | l2 | 128 | +51.3 ± 19.5 | +36.4 | +56.8 | 67% | 260k |
| run09 | frequent target copy, Huber, wider net | 0.0005 | 0.99 | 50000 | 128 | 0.05 | 60000 | hard | 50 | False | smooth_l1 | 256 | +43.3 ± 18.3 | +39.3 | +34.4 | 60% | 120k |
