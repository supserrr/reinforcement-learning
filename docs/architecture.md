# Agent architecture

All four agents read the same 38 observation values and choose among the same 10 actions, and all use small multilayer perceptrons built with TorchRL. The action mask is applied after the network, so an agent never picks a command the flight controller would refuse.

## Networks

```mermaid
flowchart LR
    O["Observation<br/>38 values"] --> H1["Linear 128"] --> H2["Linear 128"] --> OUT["10 outputs"]
    OUT --> M{"Action mask<br/>10 bits"}
    M --> A["Action"]
```

| Agent | Output of the network | Activation | Second network | How the action is chosen |
|---|---|---|---|---|
| DQN | a Q-value for each action | ReLU | a target copy of the Q-network | highest masked Q-value (epsilon-greedy while training) |
| REINFORCE | a logit for each action | Tanh | value network (38 → 128 → 128 → 1) used as a baseline | sampled from `MaskedCategorical` while training, most likely action when playing |
| PPO | a logit for each action | Tanh | value network (critic) | as REINFORCE |
| A2C | a logit for each action | Tanh | value network (critic) | as REINFORCE |

DQN run 9 and A2C run 8 use 256 units instead of 128.

## Training loops

```mermaid
flowchart LR
    E["DroneDeliveryEnv"] --> W["GymWrapper<br/>mask to bool, StepCounter,<br/>RewardSum, RewardScaling x0.1"]
    W --> C["Collector"]
    C -->|DQN| RB["Replay buffer"] --> L1["DQNLoss<br/>+ target update"]
    C -->|PPO| G1["GAE"] --> L2["ClipPPOLoss<br/>several epochs of minibatches"]
    C -->|A2C| G2["GAE"] --> L3["A2CLoss<br/>one update per rollout"]
    W -->|REINFORCE: whole shifts with env.rollout| R["Monte Carlo returns<br/>minus value baseline"] --> L4["ReinforceLoss"]
    L1 & L2 & L3 & L4 --> OPT["Adam<br/>(RMSprop in A2C run 9)"]
```

- **DQN** stores every step in a replay buffer and learns from random batches with the target `r + gamma * max Q_target(s', a')`, masked to allowed actions. The target network is copied every 500 updates (hard) or blended in a little every update (soft). Double DQN chooses the next action with the online network and scores it with the target network.
- **PPO** collects 2,048 steps, computes advantages with generalised advantage estimation (GAE), then trains for 4 epochs on minibatches of 256 with the clipped objective, a critic loss and an entropy bonus.
- **A2C** collects 256 steps and makes a single update from GAE advantages, with a critic loss and an entropy bonus.
- **REINFORCE** plays 8 whole shifts, computes each step's discounted return to the end of the shift, subtracts the critic's estimate (in most runs) and normalises the result, then makes one update.

Every 5,000 steps the training scripts play the greedy policy on 5 held-out seeds, and at the end on 20 more under each day type. [experiments.md](experiments.md) lists the hyperparameters.

## TorchRL pieces

| Piece | Used for |
|---|---|
| `GymWrapper`, `TransformedEnv`, `DTypeCastTransform`, `StepCounter`, `RewardSum`, `RewardScaling` | wrapping the Gymnasium environment |
| `Collector` | collecting frames for DQN, PPO and A2C |
| `QValueActor`, `EGreedyModule` | DQN's policy and exploration, both mask-aware |
| `ProbabilisticActor`, `MaskedCategorical`, `ValueOperator` | the policy-gradient actor and critic |
| `TensorDictReplayBuffer`, `LazyTensorStorage`, `SamplerWithoutReplacement` | DQN's replay buffer and PPO's minibatches |
| `DQNLoss`, `HardUpdate`, `SoftUpdate` | DQN |
| `ClipPPOLoss`, `A2CLoss`, `ReinforceLoss`, `GAE` | the policy-gradient methods |
