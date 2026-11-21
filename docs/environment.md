# The environment

`DroneDeliveryEnv` in [`environment/custom_env.py`](../environment/custom_env.py) uses the Gymnasium API, is registered as `DroneDelivery-v0`, and is wrapped for TorchRL with `GymWrapper`.

It is event-driven: one step is one operational decision, and the simulated clock advances by however long that decision takes, so a 10-minute flight and a 3-minute load are each one step. A shift takes 40 to 80 decisions.

## Map

A hub and six health centres in Muhanga District, 20–25 km from the hub and 13–40 km from each other, at 1,750–2,300 m elevation. The layout is stylised, not surveyed.

| Site | Position from the hub (km) | Elevation |
|---|---|---|
| Muhanga Hub | 0, 0 | 1,850 m |
| Nyabinoni HC | 17 W, 13 N | 2,300 m |
| Rongi HC | 5 W, 21 N | 2,100 m |
| Kiyumba HC | 14 E, 15 N | 1,900 m |
| Shyogwe HC | 20 E, 4 S | 1,750 m |
| Nyarusange HC | 6 E, 19 S | 1,800 m |
| Kibangu HC | 20 W, 9 S | 2,200 m |

## Actions (10, discrete)

| # | Action | Real-world command | Time |
|---|---|---|---|
| 0 | Hold | wait at the hub, or loiter over a centre (0.35% battery a minute away from the hub) | 5 min |
| 1 | Return to Hub | fly back to base | distance ÷ 120 km/h |
| 2–7 | Deliver to centre 1–6 | fly to that centre and parachute one package | distance ÷ 120 km/h |
| 8 | Recharge | charge on the hub pad, +20% | 10 min |
| 9 | Load Package | load one blood or medicine package (the drone holds 2) | 3 min |

The observation also carries an **action mask**: the commands the flight controller would accept now. It refuses flights the battery cannot finish, deliveries with an empty hold, and charging or loading away from the hub. It does not stop the drone flying somewhere it cannot get home from; avoiding that is the agent's job.

## Observation (38 values, all scaled to [−1, 1])

| Index | Value | Encoding and range | Where it comes from (simulation rule, then real source) | Updated |
|---|---|---|---|---|
| 0–6 | Drone location | one-hot over the hub and 6 centres | position after the last flight; GPS / operations system | every decision |
| 7 | Battery | % ÷ 100, [0, 1] | minus flight energy, plus 20 per recharge; battery management system | every decision |
| 8 | Cargo on board | packages ÷ 2, {0, 0.5, 1} | +1 per load, −1 per drop; cargo bay sensor | every decision |
| 9 | Shift clock | minutes ÷ 480, [0, 1] | advances by each action's duration; hub clock | every decision |
| 10 | Wind speed | km/h ÷ 64, [0, 1] | random walk (σ = 2% of the maximum per √min); hub anemometer | drifts continuously, read each decision |
| 11–12 | Wind direction | sin and cos of its heading, [−1, 1] | random walk (σ = 0.03 rad per √min); hub wind vane | as above |
| 13–30 | Per centre (×6): open request, emergency, time left | {0, 1}, {0, 1}, minutes ÷ 120 in [0, 1] | Poisson arrivals, 0.3–0.65 an hour per centre, busiest at midday; 30% emergencies (45-minute deadline), the rest routine (120 minutes); the centres' ordering system | arrivals and expiries during every action |
| 31–37 | Battery needed to reach each site and get home | % ÷ 100, [0, 1] | distance × 0.9%/km × wind (a headwind adds up to 55%) × payload (+6% per package) + 0.004% per metre climbed; flight-planning software | every decision |

The agent receives indices 0–37 as the `observation` key and the mask as `action_mask`. The observation feeds the network, and the mask feeds `QValueActor` and `EGreedyModule` (DQN) or `MaskedCategorical` (policy gradient).

## Rewards

| Event | Reward |
|---|---|
| Emergency package delivered before its deadline | +10 |
| Routine package delivered before its deadline | +4 |
| Emergency deadline missed (the centre has to send a patient by road) | −10 |
| Routine deadline missed | −3 |
| Trip to a centre with no open request | −1 |
| Shift ends with the drone away from the hub | −15 |
| Battery runs out in the air (crash) | −40, plus every open request and every request still expected this shift, because the service is grounded |

The environment returns these rewards unscaled. The learners see them × 0.1 (TorchRL `RewardScaling`), and every return in this repository is in the original units.

## Start state and terminal conditions

- **Start:** the drone at the hub with 70–100% battery and no cargo, 0–2 requests already open, and a random wind.
- **Terminal:** the shift reaches 480 minutes, or the drone crashes. A safety cap of 400 decisions truncates a runaway episode; it is never reached in practice.

## Day types used for testing

| Day | Change | Used for |
|---|---|---|
| Normal | as above | training and testing |
| Busy | requests arrive 50% more often | testing only |
| Windy | wind 60% stronger | testing only |

## Why this needs reinforcement learning

The right action depends on what happens next, and the environment only says so later. Leaving to serve a routine request costs nothing now, but an emergency that arrives while the drone is away can be missed 45 minutes later. Recharging before it is needed wastes time; not recharging can strand the drone. Loading two packages lets one trip serve two centres, but only if a second request turns up. Demand and wind are random, so there is no fixed route to work out in advance: a policy has to weigh expected future requests against battery and time.

A greedy rule, the hand-written dispatcher in `heuristic_action`, is a strong baseline (+41.4 per shift on the test seeds), and the learned agents are judged against it.

## 3D viewer and integration

[`environment/rendering.py`](../environment/rendering.py) serves the three.js page in [`environment/web/`](../environment/web) and exposes the data a real deployment would use:

| Endpoint | Purpose |
|---|---|
| `GET /scene` | static map: sites, coordinates, elevations, action names |
| `GET /state` | the latest step as JSON |
| `GET /events` | live stream of steps (Server-Sent Events) |
| `POST /api/act` | `{"observation": [...38], "action_mask": [...10]}` returns `{"action": 3, "name": "Deliver to Kiyumba HC"}` from the loaded policy |

Each step `main.py` plays is also written to `logs/demo/<agent>_seed<seed>.jsonl` in the same format.
