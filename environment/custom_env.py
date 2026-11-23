"""Medical drone delivery in the hills of Muhanga District, Rwanda.

One fixed-wing delivery drone works an 8-hour shift from a distribution hub.
Six rural health centres send requests for blood and emergency medicine at
random times. Each request has a deadline: if no package arrives in time, the
centre has to send a patient or a motorbike courier by road instead.

Every decision the drone makes changes what it can do next:

* Battery carries over. Flights cost energy that depends on distance, the
  climb over the real hills on the way, payload and the current wind, plus
  the catapult launch, the parachute drop and the wire recovery at the hub. Recharging at the hub restores it but uses
  shift time while requests wait. Running flat in the air is a crash.
* Cargo carries over. The drone holds two packages, so it can serve two
  centres on one trip, but only if it loaded both before leaving the hub.
* Requests carry over. A routine request ignored now can turn into a missed
  deadline an hour later, and emergencies arrive while the drone is away.
* Position carries over. The drone has to be back at the hub when the shift
  ends, and a flight still in the air at that moment has not landed.

The environment is event-driven: each step is one operational decision
(fly somewhere, hold, recharge, load), and the simulation clock moves forward
by however long that decision takes, so a short flight and a long one are each
one step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------
# The map: Muhanga distribution hub and six health centres (positions in km
# from the hub). The positions are stylised, not surveyed; the ground under
# them is real (SRTM), baked by scripts/bake_routes.py from the viewer's map.
# --------------------------------------------------------------------------
HUB = 0
SITE_NAMES = (
    "Muhanga Hub",
    "Nyabinoni HC",
    "Rongi HC",
    "Kiyumba HC",
    "Shyogwe HC",
    "Nyarusange HC",
    "Kibangu HC",
)
SITE_XY = np.array(
    [
        [0.0, 0.0],
        [-17.0, 13.0],
        [-5.0, 21.0],
        [14.0, 15.0],
        [20.0, -4.0],
        [6.0, -19.0],
        [-20.0, -9.0],
    ],
    dtype=np.float64,
)
SITE_ELEVATION_M = np.array([1838.0, 1941.0, 2131.0, 1643.0, 1447.0, 1601.0, 1541.0])
# Metres climbed flying from site i to site j: a terrain-following cruise about
# 120 m above the ground, out of the hub from the launcher or out of a centre
# from its 90 m loiter (scripts/bake_routes.py).
ROUTE_CLIMB_M = np.array([
    [0.0, 919.0, 1055.0, 502.0, 284.0, 293.0, 585.0],
    [726.0, 0.0, 695.0, 1155.0, 953.0, 704.0, 337.0],
    [671.0, 505.0, 0.0, 358.0, 885.0, 898.0, 666.0],
    [607.0, 1454.0, 846.0, 0.0, 227.0, 865.0, 1084.0],
    [584.0, 1446.0, 1568.0, 423.0, 0.0, 553.0, 1067.0],
    [439.0, 1044.0, 1428.0, 906.0, 399.0, 0.0, 510.0],
    [791.0, 737.0, 1256.0, 1186.0, 973.0, 570.0, 0.0],
])
N_SITES = len(SITE_NAMES)
N_CENTRES = N_SITES - 1

# --------------------------------------------------------------------------
# Actions. The list covers every operational command the hub's flight
# controller can give this drone during a shift.
# --------------------------------------------------------------------------
A_HOLD = 0  # wait 5 min where the drone is (hover if away from the hub)
A_RETURN = 1  # fly back to the hub
A_DELIVER_FIRST = 2  # actions 2..7: fly to centre 1..6 and drop one package
A_RECHARGE = 8  # at the hub: charge for 10 min
A_LOAD = 9  # at the hub: load one package (3 min)
N_ACTIONS = 10
ACTION_NAMES = (
    "Hold",
    "Return to Hub",
    *(f"Deliver to {n}" for n in SITE_NAMES[1:]),
    "Recharge",
    "Load Package",
)

OBS_DIM = N_SITES + 1 + 1 + 1 + 3 + 3 * N_CENTRES + N_SITES + N_SITES  # 45
# Minutes of slack that map to 1.0 in the time-left features: enough to tell
# a trip that fits in the shift from one that does not, near the end of it.
SLACK_SCALE_MIN = 60.0


@dataclass(frozen=True)
class DroneConfig:
    """Every number the simulation uses. Variants of this drive the
    generalisation tests (busier days, stronger wind)."""

    shift_minutes: float = 480.0
    speed_km_per_min: float = 2.0  # 120 km/h cruise
    battery_capacity: float = 100.0  # percent
    energy_per_km: float = 0.9  # % battery per km in still air, empty
    payload_energy_factor: float = 0.06  # +6% energy per package carried
    climb_energy_per_m: float = 0.004  # % per metre climbed
    wind_energy_factor: float = 0.55  # full headwind at max wind = +55%
    max_wind_kmh: float = 40.0
    wind_mult: float = 1.0  # scales wind strength (generalisation tests)
    hover_energy_per_min: float = 0.35  # loitering over a centre
    # return-to-base rule: a flight is only launched if the drone can still get
    # home before the shift ends with this much battery to spare (covers wind drift)
    return_reserve: float = 2.0  # percent
    # flight operations: catapult launch at the hub, a descent to 35 m over the
    # centre's drop zone for the parachute release, and the approach circuit
    # into the recovery wire back at the hub
    launch_minutes: float = 1.0
    launch_energy: float = 0.5
    drop_minutes: float = 0.5
    drop_energy: float = 0.3
    recovery_minutes: float = 2.0
    recovery_energy: float = 0.8
    recharge_amount: float = 20.0
    recharge_minutes: float = 10.0
    load_minutes: float = 3.0
    hold_minutes: float = 5.0
    invalid_minutes: float = 1.0
    cargo_capacity: int = 2
    # requests per hour at each centre (index 0 = hub, unused)
    request_rate_per_hour: tuple = (0.0, 0.45, 0.35, 0.55, 0.65, 0.5, 0.3)
    demand_mult: float = 1.0
    emergency_prob: float = 0.3
    emergency_deadline: float = 45.0
    routine_deadline: float = 120.0
    # rewards
    r_emergency: float = 10.0
    r_routine: float = 4.0
    r_emergency_missed: float = -10.0
    r_routine_missed: float = -3.0
    r_wasted_trip: float = -1.0
    r_invalid: float = -0.5
    r_crash: float = -40.0  # plus every request the grounded service will now miss
    r_stranded: float = -25.0
    # start state randomisation
    start_battery_range: tuple = (70.0, 100.0)
    start_orders_max: int = 2
    max_decisions: int = 400  # safety cap; a normal shift takes 40-80 decisions


EASY = DroneConfig()
BUSY_DAY = replace(EASY, demand_mult=1.5)
WINDY_DAY = replace(EASY, wind_mult=1.6)
CONFIGS = {"default": EASY, "busy": BUSY_DAY, "windy": WINDY_DAY}


@dataclass
class Request:
    emergency: bool
    deadline: float  # absolute shift minute


@dataclass
class EpisodeStats:
    delivered_emergency: int = 0
    delivered_routine: int = 0
    missed_emergency: int = 0
    missed_routine: int = 0
    requests: int = 0
    wasted_trips: int = 0
    invalid_actions: int = 0
    recharges: int = 0
    km_flown: float = 0.0
    crashed: bool = False
    stranded: bool = False
    emergency_wait_minutes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "emergency_wait_minutes"}
        waits = self.emergency_wait_minutes
        d["mean_emergency_wait"] = float(np.mean(waits)) if waits else 0.0
        served = self.delivered_emergency + self.delivered_routine
        d["service_rate"] = served / self.requests if self.requests else 1.0
        return d


class DroneDeliveryEnv(gym.Env):
    """Gymnasium environment. See module docstring and README for the full
    observation, action and reward tables."""

    metadata = {"render_modes": ["ansi", "json"], "render_fps": 4}

    def __init__(self, config: DroneConfig | str = "default", render_mode=None):
        super().__init__()
        self.cfg = CONFIGS[config] if isinstance(config, str) else config
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        # "action_mask" is what the flight controller would accept right now:
        # no flights the battery cannot finish, no deliveries with an empty
        # hold, recharge/load only at the hub. Risky flights stay allowed.
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(N_ACTIONS),
        })
        self._dist = np.linalg.norm(SITE_XY[:, None, :] - SITE_XY[None, :, :], axis=-1)
        self.last_events: list[dict] = []

    # ------------------------------------------------------------------ core
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        self.t = 0.0
        self.loc = HUB
        self.airborne = False  # only at the end of a shift that ended mid-flight
        self.battery = float(self.np_random.uniform(*c.start_battery_range))
        self.cargo = 0
        self.wind_speed = float(self.np_random.uniform(0.1, 0.7))  # fraction of max
        self.wind_dir = float(self.np_random.uniform(0, 2 * np.pi))  # blowing towards
        self.requests: list[Request | None] = [None] * N_SITES
        self.stats = EpisodeStats()
        self.total_reward = 0.0
        self.last_action = None
        self.last_events = []
        self.decisions = 0
        for _ in range(int(self.np_random.integers(0, c.start_orders_max + 1))):
            self._spawn_request(int(self.np_random.integers(1, N_SITES)))
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        c = self.cfg
        self.last_action = action
        events: list[dict] = []
        reward = 0.0
        terminated = False

        if action == A_HOLD:
            minutes = c.hold_minutes
            if self.loc != HUB:
                self.battery -= c.hover_energy_per_min * minutes
            events.append({"type": "hold", "at": self.loc, "minutes": minutes})
        elif action == A_RETURN or A_DELIVER_FIRST <= action < A_DELIVER_FIRST + N_CENTRES:
            dest = HUB if action == A_RETURN else action - A_DELIVER_FIRST + 1
            if dest == self.loc:
                # flying to where you already are is a no-op that wastes a minute
                minutes = c.invalid_minutes
                reward += c.r_invalid
                self.stats.invalid_actions += 1
                events.append({"type": "invalid", "reason": "already here", "at": self.loc})
            else:
                need = self.flight_energy(self.loc, dest)
                minutes = self.flight_minutes(self.loc, dest)
                t_left = c.shift_minutes - self.t
                crash_frac = max(self.battery, 0.0) / need if need > self.battery else None
                cut_frac = t_left / minutes if minutes > t_left else None
                if crash_frac is not None and (cut_frac is None or crash_frac <= cut_frac):
                    # battery runs out part-way: the drone goes down
                    frac = crash_frac
                    events.append({"type": "crash", "from": self.loc, "to": dest, "fraction": frac,
                                   "minutes": minutes * frac})
                    self.stats.km_flown += self._dist[self.loc, dest] * frac
                    self.battery = 0.0
                    self.stats.crashed = True
                    reward += self._crash_penalty()
                    terminated = True
                    minutes = minutes * frac
                elif cut_frac is not None:
                    # the shift ends before the drone lands: it is still in the air, so it
                    # has not reached the hub and cannot drop a package
                    events.append({"type": "flight", "from": self.loc, "to": dest, "minutes": t_left,
                                   "energy": need * cut_frac, "landed": False, "fraction": cut_frac})
                    self.battery -= need * cut_frac
                    self.stats.km_flown += self._dist[self.loc, dest] * cut_frac
                    self.airborne = True
                    minutes = t_left
                else:
                    events.append({"type": "flight", "from": self.loc, "to": dest, "minutes": minutes,
                                   "energy": need})
                    self.battery -= need
                    self.stats.km_flown += self._dist[self.loc, dest]
                    self.loc = dest
        elif action == A_RECHARGE:
            if self.loc != HUB or self.battery >= c.battery_capacity - 1e-6:
                minutes = c.invalid_minutes
                reward += c.r_invalid
                self.stats.invalid_actions += 1
                events.append({"type": "invalid", "reason": "cannot recharge", "at": self.loc})
            else:
                minutes = c.recharge_minutes
                self.battery = min(c.battery_capacity, self.battery + c.recharge_amount)
                self.stats.recharges += 1
                events.append({"type": "recharge", "minutes": minutes})
        elif action == A_LOAD:
            if self.loc != HUB or self.cargo >= c.cargo_capacity:
                minutes = c.invalid_minutes
                reward += c.r_invalid
                self.stats.invalid_actions += 1
                events.append({"type": "invalid", "reason": "cannot load", "at": self.loc})
            else:
                minutes = c.load_minutes
                self.cargo += 1
                events.append({"type": "load", "minutes": minutes, "cargo": self.cargo})
        else:
            raise ValueError(f"unknown action {action}")

        # Advance the clock. Requests that expire during the action are missed;
        # new requests arrive; wind drifts.
        t_end = min(self.t + minutes, c.shift_minutes) if not terminated else self.t + minutes
        reward += self._advance_clock(self.t, t_end, events)
        self.t = t_end

        # Drop a package on arrival at a centre (after the clock moved, so a
        # request that expired in flight is no longer there).
        if (not terminated and not self.airborne and A_DELIVER_FIRST <= action < A_DELIVER_FIRST + N_CENTRES
                and self.loc != HUB):
            req = self.requests[self.loc]
            if self.cargo == 0:
                reward += c.r_wasted_trip
                self.stats.wasted_trips += 1
                events.append({"type": "wasted", "at": self.loc, "reason": "no package on board"})
            elif req is None:
                reward += c.r_wasted_trip
                self.stats.wasted_trips += 1
                events.append({"type": "wasted", "at": self.loc, "reason": "no open request"})
            else:
                self.cargo -= 1
                if req.emergency:
                    reward += c.r_emergency
                    self.stats.delivered_emergency += 1
                    wait = c.emergency_deadline - (req.deadline - self.t)
                    self.stats.emergency_wait_minutes.append(wait)
                else:
                    reward += c.r_routine
                    self.stats.delivered_routine += 1
                self.requests[self.loc] = None
                events.append({"type": "delivered", "at": self.loc, "emergency": req.emergency})

        if not terminated and self.battery <= 0.0:
            # hovering drained it
            self.battery = 0.0
            self.stats.crashed = True
            reward += self._crash_penalty()
            terminated = True
            events.append({"type": "crash", "from": self.loc, "to": self.loc, "fraction": 0.0, "minutes": 0})

        self.decisions += 1
        truncated = False
        if not terminated and self.t >= c.shift_minutes:
            terminated = True
            if self.loc != HUB or self.airborne:
                reward += c.r_stranded
                self.stats.stranded = True
                events.append({"type": "stranded", "at": self.loc, "in_flight": self.airborne})
            events.append({"type": "shift_end"})
        elif not terminated and self.decisions >= c.max_decisions:
            truncated = True

        self.total_reward += reward
        self.last_events = events
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------ dynamics
    def flight_minutes(self, src: int, dst: int) -> float:
        """Shift minutes a flight takes: cruise plus launch, drop or recovery."""
        c = self.cfg
        m = self._dist[src, dst] / c.speed_km_per_min
        m += c.launch_minutes if src == HUB else 0.0
        m += c.recovery_minutes if dst == HUB else c.drop_minutes
        return m

    def flight_energy(self, src: int, dst: int, cargo: int | None = None) -> float:
        """Battery % a flight from src to dst costs right now."""
        if src == dst:
            return 0.0
        c = self.cfg
        cargo = self.cargo if cargo is None else cargo
        d = self._dist[src, dst]
        heading = (SITE_XY[dst] - SITE_XY[src]) / d
        wind_vec = np.array([np.cos(self.wind_dir), np.sin(self.wind_dir)])
        headwind = -float(heading @ wind_vec) * min(self.wind_speed * c.wind_mult, 1.6)
        mult = 1.0 + c.wind_energy_factor * headwind
        mult *= 1.0 + c.payload_energy_factor * cargo
        ops = (c.launch_energy if src == HUB else 0.0) + (c.recovery_energy if dst == HUB else c.drop_energy)
        return d * c.energy_per_km * mult + ROUTE_CLIMB_M[src, dst] * c.climb_energy_per_m + ops

    def _crash_penalty(self) -> float:
        """A crash loses the aircraft and grounds the service, so every open
        request and every request still expected this shift will be missed."""
        c = self.cfg
        miss = c.emergency_prob * c.r_emergency_missed + (1 - c.emergency_prob) * c.r_routine_missed
        x0 = min(self.t / c.shift_minutes, 1.0)
        # integral of the daily demand curve 0.7 + 0.6 sin(pi x) over [x0, 1]
        daily = 0.7 * (1 - x0) + 0.6 / np.pi * (np.cos(np.pi * x0) - np.cos(np.pi))
        expected_new = sum(c.request_rate_per_hour) * c.demand_mult * c.shift_minutes / 60.0 * daily
        open_pen = sum((c.r_emergency_missed if r.emergency else c.r_routine_missed)
                       for r in self.requests if r is not None)
        return c.r_crash + open_pen + expected_new * miss

    def _advance_clock(self, t0: float, t1: float, events: list) -> float:
        c = self.cfg
        reward = 0.0
        dt = max(t1 - t0, 0.0)
        # expiries
        for i in range(1, N_SITES):
            req = self.requests[i]
            if req is not None and req.deadline <= t1:
                self.requests[i] = None
                if req.emergency:
                    reward += c.r_emergency_missed
                    self.stats.missed_emergency += 1
                else:
                    reward += c.r_routine_missed
                    self.stats.missed_routine += 1
                events.append({"type": "missed", "at": i, "emergency": req.emergency, "t": req.deadline})
        # arrivals (Poisson, busier around midday)
        if dt > 0:
            mid = (t0 + t1) / 2 / c.shift_minutes
            daily = 0.7 + 0.6 * np.sin(np.pi * mid)  # 0.7 at start/end, 1.3 midday
            for i in range(1, N_SITES):
                lam = c.request_rate_per_hour[i] * c.demand_mult * daily * dt / 60.0
                for _ in range(int(self.np_random.poisson(lam))):
                    arrival = float(self.np_random.uniform(t0, t1))
                    self._spawn_request(i, now=arrival, events=events)
            # wind drifts
            self.wind_dir = float((self.wind_dir + self.np_random.normal(0, 0.03 * np.sqrt(dt))) % (2 * np.pi))
            self.wind_speed = float(np.clip(self.wind_speed + self.np_random.normal(0, 0.02 * np.sqrt(dt)), 0.0, 1.0))
        return reward

    def _spawn_request(self, i: int, now: float | None = None, events: list | None = None):
        c = self.cfg
        now = self.t if now is None else now
        emergency = bool(self.np_random.random() < c.emergency_prob)
        deadline = now + (c.emergency_deadline if emergency else c.routine_deadline)
        cur = self.requests[i]
        if cur is None:
            self.requests[i] = Request(emergency, deadline)
            self.stats.requests += 1
        elif emergency and not cur.emergency:
            # an emergency at a centre with a routine request open upgrades it
            self.requests[i] = Request(True, min(deadline, cur.deadline))
        else:
            return
        if events is not None:
            events.append({"type": "request", "at": i, "emergency": emergency, "t": now})

    # --------------------------------------------------------- observation
    def action_mask(self) -> np.ndarray:
        """Commands the flight controller accepts now. Return-to-base rule: a
        flight is allowed only if the battery covers it and the flight home after
        it, with a small reserve, and both land before the shift ends; holding
        over a centre only while the drone can still get home afterwards."""
        c = self.cfg
        m = np.zeros(N_ACTIONS, dtype=np.int8)
        t_left = c.shift_minutes - self.t

        def gets_home(dest: int) -> bool:
            energy = self.flight_energy(self.loc, dest) + self.flight_energy(dest, HUB)
            minutes = self.flight_minutes(self.loc, dest) + (self.flight_minutes(dest, HUB) if dest != HUB else 0.0)
            return energy + c.return_reserve <= self.battery and minutes <= t_left

        if self.loc != HUB and self.flight_energy(self.loc, HUB) <= self.battery \
                and self.flight_minutes(self.loc, HUB) <= t_left:
            m[A_RETURN] = 1  # going home is always allowed when it can land in time
        if self.cargo > 0:
            for i in range(1, N_SITES):
                if i != self.loc and gets_home(i):
                    m[A_DELIVER_FIRST + i - 1] = 1
        if self.loc == HUB:
            m[A_HOLD] = 1
        else:
            after = self.battery - c.hover_energy_per_min * c.hold_minutes
            can_wait = (after >= self.flight_energy(self.loc, HUB) + c.return_reserve
                        and c.hold_minutes + self.flight_minutes(self.loc, HUB) <= t_left)
            m[A_HOLD] = 1 if can_wait or not m.any() else 0  # never leave the controller without a command
        if self.loc == HUB and self.battery < c.battery_capacity - 1e-6:
            m[A_RECHARGE] = 1
        if self.loc == HUB and self.cargo < c.cargo_capacity:
            m[A_LOAD] = 1
        return m

    def _obs(self) -> dict:
        return {"observation": self._features(), "action_mask": self.action_mask()}

    def _features(self) -> np.ndarray:
        c = self.cfg
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[self.loc] = 1.0  # 0..6 location one-hot
        k = N_SITES
        o[k] = self.battery / c.battery_capacity
        o[k + 1] = self.cargo / c.cargo_capacity
        o[k + 2] = self.t / c.shift_minutes
        o[k + 3] = min(self.wind_speed * c.wind_mult, 1.6) / 1.6
        o[k + 4] = np.sin(self.wind_dir)
        o[k + 5] = np.cos(self.wind_dir)
        k += 6
        for i in range(1, N_SITES):
            req = self.requests[i]
            j = k + 3 * (i - 1)
            if req is not None:
                o[j] = 1.0
                o[j + 1] = 1.0 if req.emergency else 0.0
                o[j + 2] = np.clip((req.deadline - self.t) / c.routine_deadline, 0.0, 1.0)
        k += 3 * N_CENTRES
        # battery needed to fly to each site now and then get home (flight planner estimate)
        for i in range(N_SITES):
            if i == HUB:
                need = self.flight_energy(self.loc, HUB)
            else:
                drop = 1 if self.cargo > 0 else 0
                need = self.flight_energy(self.loc, i) + self.flight_energy(i, HUB, cargo=self.cargo - drop)
            o[k + i] = np.clip(need / c.battery_capacity, 0.0, 1.0)
        k += N_SITES
        # shift minutes left after flying to each site now and then home; below zero
        # the trip cannot finish before the shift ends (flight planner estimate)
        t_left = c.shift_minutes - self.t
        home = self.flight_minutes(self.loc, HUB) if self.loc != HUB else 0.0
        for i in range(N_SITES):
            if i in (HUB, self.loc):
                trip = home
            else:
                trip = self.flight_minutes(self.loc, i) + self.flight_minutes(i, HUB)
            o[k + i] = np.clip((t_left - trip) / SLACK_SCALE_MIN, -1.0, 1.0)
        return o

    def _info(self) -> dict:
        return {"t": self.t, "battery": self.battery, "cargo": self.cargo, "loc": self.loc,
                **{f"stat_{k}": v for k, v in self.stats.as_dict().items()}}

    # -------------------------------------------------------------- render
    def snapshot(self) -> dict:
        """JSON-serialisable state for the 3D viewer and logs."""
        return {
            "t": round(self.t, 2),
            "shift_minutes": self.cfg.shift_minutes,
            "loc": self.loc,
            "airborne": self.airborne,
            "battery": round(self.battery, 2),
            "cargo": self.cargo,
            "cargo_capacity": self.cfg.cargo_capacity,
            "wind": {"speed_kmh": round(min(self.wind_speed * self.cfg.wind_mult, 1.6) * self.cfg.max_wind_kmh, 1),
                     "dir_rad": round(self.wind_dir, 3)},
            "requests": [None if r is None else {"emergency": r.emergency,
                                                 "minutes_left": round(r.deadline - self.t, 1)}
                         for r in self.requests],
            "last_action": None if self.last_action is None else ACTION_NAMES[self.last_action],
            "events": self.last_events,
            "total_reward": round(self.total_reward, 2),
            "stats": self.stats.as_dict(),
        }

    def render(self):
        if self.render_mode == "json":
            return self.snapshot()
        s = self.snapshot()
        reqs = " ".join(
            f"{SITE_NAMES[i].split()[0]}:{'E' if r['emergency'] else 'R'}{r['minutes_left']:.0f}"
            for i, r in enumerate(s["requests"]) if r
        )
        return (f"t={s['t']:5.1f} at={SITE_NAMES[s['loc']]:<14} batt={s['battery']:5.1f}% "
                f"cargo={s['cargo']} wind={s['wind']['speed_kmh']:4.1f}km/h  open[{reqs}]")


def heuristic_action(env: DroneDeliveryEnv) -> int:
    a = _heuristic(env)
    return a if env.action_mask()[a] else A_HOLD


def _heuristic(env: DroneDeliveryEnv) -> int:
    """Hand-written dispatcher used as a non-learning baseline: serve the most
    urgent reachable request, keep a 15% battery reserve, recharge below 60%,
    and only start trips that are back at the hub before the shift ends."""
    c = env.cfg
    reserve = 15.0
    time_left = c.shift_minutes - env.t

    def round_trip(i: int) -> float:  # minutes to fly to centre i and back to the hub from here
        return env.flight_minutes(env.loc, i) + env.flight_minutes(i, HUB)

    open_reqs = [(i, r) for i, r in enumerate(env.requests) if r is not None]
    if env.loc == HUB:
        if env.cargo < c.cargo_capacity and any(c.load_minutes + round_trip(i) <= time_left for i, _ in open_reqs):
            return A_LOAD
        if env.battery < 60.0:
            return A_RECHARGE
    if env.cargo > 0 and open_reqs:
        ranked = sorted(open_reqs, key=lambda ir: (not ir[1].emergency, ir[1].deadline - env.t))
        for i, _ in ranked:
            need = env.flight_energy(env.loc, i) + env.flight_energy(i, HUB, cargo=env.cargo - 1)
            if need + reserve <= env.battery and i != env.loc and round_trip(i) <= time_left:
                return A_DELIVER_FIRST + i - 1
    if env.loc != HUB:
        return A_RETURN
    if env.battery < c.battery_capacity:
        return A_RECHARGE
    return A_HOLD
