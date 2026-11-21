import numpy as np
import pytest

from environment import N_ACTIONS, OBS_DIM, DroneDeliveryEnv, heuristic_action
from environment.custom_env import A_DELIVER_FIRST, A_HOLD, A_LOAD, A_RECHARGE, A_RETURN, HUB


def rollout(policy, seed, config="default"):
    env = DroneDeliveryEnv(config)
    obs, _ = env.reset(seed=seed)
    done, steps = False, 0
    while not done:
        obs, r, te, tr, _ = env.step(policy(env, obs))
        done = te or tr
        steps += 1
    return env, steps


def test_spaces_and_reset():
    env = DroneDeliveryEnv()
    obs, info = env.reset(seed=1)
    assert obs["observation"].shape == (OBS_DIM,)
    assert obs["action_mask"].shape == (N_ACTIONS,)
    assert env.observation_space.contains(obs)
    assert env.loc == HUB and env.cargo == 0 and env.t == 0


def test_same_seed_same_episode():
    a, _ = rollout(lambda e, o: heuristic_action(e), seed=5)
    b, _ = rollout(lambda e, o: heuristic_action(e), seed=5)
    assert a.total_reward == b.total_reward


def test_mask_at_hub_with_empty_hold():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.battery, env.cargo = 50.0, 0
    m = env.action_mask()
    assert m[A_HOLD] and m[A_LOAD] and m[A_RECHARGE]
    assert not m[A_RETURN]  # already at the hub
    assert not any(m[A_DELIVER_FIRST:A_DELIVER_FIRST + 6])  # nothing to deliver


def test_flight_costs_more_into_headwind_and_with_cargo():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.wind_speed = 0.0
    calm = env.flight_energy(HUB, 4, cargo=0)
    assert env.flight_energy(HUB, 4, cargo=2) > calm
    # wind blowing from site 4 towards the hub = headwind on the way out
    vec = -(np.array([20.0, -4.0]) / np.hypot(20.0, -4.0))
    env.wind_dir, env.wind_speed = float(np.arctan2(vec[1], vec[0])), 0.8
    assert env.flight_energy(HUB, 4, cargo=0) > calm


def test_delivery_pays_and_clears_request():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.requests = [None] * 7
    env._spawn_request(3)
    env.battery, env.cargo = 100.0, 1
    _, r, *_ = env.step(A_DELIVER_FIRST + 2)
    assert env.requests[3] is None and r > 0


def test_shift_ends_at_480_minutes():
    env, steps = rollout(lambda e, o: A_HOLD, seed=2)
    assert env.t == pytest.approx(480.0)
    assert steps == 96


def test_crash_is_worse_than_doing_nothing():
    env = DroneDeliveryEnv()
    env.reset(seed=3)
    assert env._crash_penalty() < -74  # a whole idle shift averages about -74


def test_heuristic_beats_random():
    rng = np.random.default_rng(0)
    rand = np.mean([rollout(lambda e, o: int(rng.choice(np.flatnonzero(o["action_mask"]))), s)[0].total_reward
                    for s in range(20)])
    heur = np.mean([rollout(lambda e, o: heuristic_action(e), s)[0].total_reward for s in range(20)])
    assert heur > rand + 50
