import gymnasium as gym
from . import forest_evader, forest_chaser

gym.register(
    id="ForestEvader",
    entry_point="envs.forest_evader:ForestEvader",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": forest_evader.ForestEvaderCfg}
)

gym.register(
    id="ForestChaser",
    entry_point="envs.forest_chaser:ForestChaser",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": forest_chaser.ForestChaserCfg}
)