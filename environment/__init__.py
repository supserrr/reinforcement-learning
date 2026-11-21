from gymnasium.envs.registration import register

from .custom_env import (
    ACTION_NAMES,
    CONFIGS,
    N_ACTIONS,
    OBS_DIM,
    SITE_NAMES,
    DroneConfig,
    DroneDeliveryEnv,
    heuristic_action,
)

register(id="DroneDelivery-v0", entry_point="environment.custom_env:DroneDeliveryEnv")

__all__ = [
    "ACTION_NAMES",
    "CONFIGS",
    "N_ACTIONS",
    "OBS_DIM",
    "SITE_NAMES",
    "DroneConfig",
    "DroneDeliveryEnv",
    "heuristic_action",
]
