import numpy as np
import pytest

from environment import N_ACTIONS, OBS_DIM, DroneDeliveryEnv, heuristic_action
from environment.custom_env import A_DELIVER_FIRST, A_HOLD, A_LOAD, A_RECHARGE, A_RETURN, HUB, N_SITES, SLACK_SCALE_MIN


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


def test_flight_still_in_the_air_at_shift_end_is_stranded():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.requests = [None] * 7
    env.loc, env.battery, env.cargo = 3, 100.0, 0
    env.t = 480.0 - env.flight_minutes(3, HUB) / 2  # half the way home is left when the shift ends
    _, r, te, *_ = env.step(A_RETURN)
    assert te and env.stats.stranded and env.loc == 3
    assert env.t == pytest.approx(480.0)
    assert r == pytest.approx(env.cfg.r_stranded)


def test_return_that_lands_in_time_is_not_stranded():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.requests = [None] * 7
    env.loc, env.battery, env.cargo = 3, 100.0, 0
    env.t = 480.0 - env.flight_minutes(3, HUB) - 1.0
    env.step(A_RETURN)
    assert env.loc == HUB and not env.stats.stranded


def test_dispatcher_is_home_when_the_shift_ends():
    for s in range(20):
        env, _ = rollout(lambda e, o: heuristic_action(e), seed=s)
        assert not env.stats.stranded and not env.stats.crashed


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


def test_time_left_features_say_whether_a_trip_fits_in_the_shift():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    k = OBS_DIM - N_SITES  # the last N_SITES values
    o = env._features()
    assert np.all(o[k:] == 1.0)  # 08:00: every trip fits with hours to spare
    env.loc = 3
    trip_home = env.flight_minutes(3, HUB)
    env.t = 480.0 - trip_home - 10.0  # 10 minutes of slack if it flies home now
    o = env._features()
    assert o[k + HUB] == pytest.approx(10.0 / SLACK_SCALE_MIN, abs=1e-5)
    assert o[k + 3] == pytest.approx(o[k + HUB])  # already at site 3: only the flight home is left
    assert o[k + 1] < 0  # a trip to another centre and back can no longer finish


def test_mask_only_launches_flights_that_get_home_in_time():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.requests = [None] * 7
    env.battery, env.cargo = 100.0, 1
    out, back = env.flight_minutes(HUB, 4), env.flight_minutes(4, HUB)
    env.t = 480.0 - out - back + 1.0  # reaches centre 4 in time but cannot get back
    assert not env.action_mask()[A_DELIVER_FIRST + 3]
    env.t = 480.0 - out - back - 1.0
    assert env.action_mask()[A_DELIVER_FIRST + 3]


def test_mask_keeps_enough_battery_for_the_flight_home():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.cargo = 1
    there_and_back = env.flight_energy(HUB, 4) + env.flight_energy(4, HUB)
    env.battery = env.flight_energy(HUB, 4) + 1.0  # covers the way out only
    assert not env.action_mask()[A_DELIVER_FIRST + 3]
    env.battery = there_and_back + env.cfg.return_reserve + 0.1
    assert env.action_mask()[A_DELIVER_FIRST + 3]


def test_hold_over_a_centre_only_while_it_can_still_get_home():
    env = DroneDeliveryEnv()
    env.reset(seed=0)
    env.loc, env.battery, env.cargo = 3, 100.0, 0
    env.t = 480.0 - env.flight_minutes(3, HUB) - 2.0  # 5 more minutes would be too late
    m = env.action_mask()
    assert m[A_RETURN] and not m[A_HOLD]
    env.t = 100.0
    assert env.action_mask()[A_HOLD]


def test_masked_policies_never_strand_or_crash():
    rng = np.random.default_rng(1)
    for s in range(30):
        env, _ = rollout(lambda e, o: int(rng.choice(np.flatnonzero(o["action_mask"]))), seed=s)
        assert not env.stats.stranded and not env.stats.crashed


def test_ending_away_costs_more_than_a_last_emergency_is_worth():
    c = DroneDeliveryEnv().cfg
    # delivering a final emergency and ending away must be worse than missing it
    assert c.r_emergency + c.r_stranded < c.r_emergency_missed
