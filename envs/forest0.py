from collections import deque, OrderedDict

import torch
import gymnasium as gym
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.lab.envs import DirectRLEnvCfg, DirectRLEnv
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, ContactSensor, ContactSensorCfg
from omni.isaac.lab.sensors.ray_caster.patterns import LidarPatternCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass

from my_utils.controller import PositionController
from my_utils.math import quat_to_z_axis



SCALE = 1.0
LIDAR_RANGE = 5.0


@configclass
class MySceneCfg(InteractiveSceneCfg):

    num_envs = 1

    env_spacing = 8.0

    terrain_cfg = TerrainImporterCfg(
        prim_path = "/World/ground",
        terrain_type = "generator",
        terrain_generator = TerrainGeneratorCfg(
            size = (100.0, 100.0),
            num_rows = 1,
            num_cols = 1,
            use_cache = False,
            difficulty_range = (1.0, 1.0),
            sub_terrains = {
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    num_obstacles = 2000,
                    obstacle_height_mode = "fixed",
                    obstacle_width_range = (0.4, 0.8),
                    obstacle_height_range = (4.0, 4.0),
                    platform_width = 0.0
                )
            },
        ),
        max_init_terrain_level = 0,
        collision_group = -1
    )
    
    light = AssetBaseCfg(
        prim_path = "/World/DomeLight",
        spawn = sim_utils.DomeLightCfg(intensity=2000.0)
    )
    
    evader = ArticulationCfg(
        prim_path = "{ENV_REGEX_NS}/evader",
        spawn = sim_utils.UsdFileCfg(
            usd_path = "/home/sbw/MyUAV/assets/cf2x/cf2x_blue.usd",
            scale = (SCALE, SCALE, SCALE),
            semantic_tags = [("class", "cf2x")],
            activate_contact_sensors = True,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity = False,
                max_depenetration_velocity = 10.0,
                enable_gyroscopic_forces = True,
            ),
            articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions = False,
                solver_position_iteration_count = 4,
                solver_velocity_iteration_count = 0,
                sleep_threshold = 0.005,
                stabilization_threshold = 0.001,
            ),
            copy_from_source = False,
        ),
        init_state = ArticulationCfg.InitialStateCfg(
            pos = (0.0, 0.0, 2.0),
            rot = (1.0, 0.0, 0.0, 0.0)
        ),
        actuators = {
            "dummy": ImplicitActuatorCfg(
                joint_names_expr = [".*"],
                stiffness = 0.0,
                damping = 0.0,
            )
        }
    )

    lidar = RayCasterCfg(
        prim_path = "{ENV_REGEX_NS}/evader/body",
        mesh_prim_paths = ["/World/ground/terrain"],
        attach_yaw_only = True,
        max_distance = 10.0,
        history_length = 3,
        pattern_cfg = LidarPatternCfg(
            channels = 1,
            vertical_fov_range = (-0.0, 0.0),
            horizontal_fov_range = (-180.0, 180.0),
            horizontal_res = 22.5
        ),
        debug_vis = False,
        visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/RayCaster",
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                )
            }
        )
    )

    contact = ContactSensorCfg(
        prim_path = "{ENV_REGEX_NS}/evader/.*",
        debug_vis = False,
        visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ContactSensor",
            markers={
                "contact": sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    visible=True
                ),
                "no_contact": sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    visible=False
                )
            }
        )
    )


@configclass
class ForestEnvCfg(DirectRLEnvCfg):
    episode_length_s = 60
    decimation = 6
    num_actions = 3
    num_observations = 55 + num_actions
    sim = SimulationCfg(
        dt = 1.0 / 120.0,
        render_interval = decimation,
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx = PhysxCfg(
            gpu_found_lost_pairs_capacity = 2**23,
            gpu_total_aggregate_pairs_capacity = 2**23
        )
    )
    scene = MySceneCfg()


class ForestEnv(DirectRLEnv):

    def __init__(self, cfg: ForestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._zeros = torch.zeros(self.num_envs, device=self.device)
        self._ones = torch.ones(self.num_envs, device=self.device)
        self._eye3 = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        self._actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._force = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torque = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self._evader: Articulation = self.scene["evader"]
        self._evader_body_id = self._evader.find_bodies("body")[0]

        self._contact: ContactSensor = self.scene["contact"]
        self._lidar: RayCaster = self.scene["lidar"]
        self._history_distance = deque(maxlen=3)
        
        self._robot_mass = self._evader.root_physx_view.get_masses()[0].sum()
        self._gravity_vector = torch.tensor(self.sim.cfg.gravity, device=self.device)  # [0, 0, -9.81]
        self._robot_weight_vector = self._robot_mass * self._gravity_vector  # [0, 0, -mg]
        self._robot_weight = torch.abs(self._robot_weight_vector[2])
        self._controller = PositionController(num_envs=self.num_envs, device=self.device, scale=SCALE)

        self._collision = torch.tensor([False], device=self.device).repeat(self.num_envs)
        self._desired_velocity = torch.tensor([0.5, 0.0, 0.0], device=self.device)
        self._desired_height = 2.0
    
    # 每step调用1次，policy应在外环
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
    
    # 每substep调用一次，每step调用decimation次，控制器应在内环
    def _apply_action(self):
        
        # 控制世界坐标系下线加速度
        self._force[:, 0, 2], self._torque[:, 0, :] = self._controller.acc_yaw_cmd(
            self._evader.data.root_quat_w,
            self._evader.data.root_ang_vel_b,
            self._actions * 2.0,
            self._zeros
        )
        self._evader.set_external_force_and_torque(self._force, self._torque, body_ids=self._evader_body_id)

        # 统计decimation个substep中是否出现碰撞
        self._collision = self._collision | torch.any(torch.abs(self._contact.data.net_forces_w) > 1e-8, dim=(1, 2))
    
    # self.sim.step()
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_range = torch.logical_or(self._evader.data.root_pos_w[:, 2] > 4.0, self._evader.data.root_pos_w[:, 2] < 0.1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_range | self._collision, time_out
    
    def _get_rewards(self) -> torch.Tensor:
        
        # collision
        dist = torch.norm(self._lidar.data.ray_hits_w - self._lidar.data.pos_w.unsqueeze(1), p=2, dim=-1).clamp_max(LIDAR_RANGE)
        minDist, _ = torch.min(dist, dim=-1)
        prevDist = self._history_distance[-1]
        prevMinDist, _ = torch.min(prevDist, dim=-1)
        avoidance_reward = torch.tanh(minDist - prevMinDist)
        
        # height
        dz = torch.abs(self._evader.data.root_pos_w[:, 2] - self._desired_height)
        height_reward = torch.exp(-dz)
        
        # velocity
        velocity_weights = torch.tensor([10.0, 1.0, 10.0], device=self.device)
        dv = self._evader.data.root_lin_vel_w - self._desired_velocity
        velocity_reward = torch.exp(-torch.sum(velocity_weights * dv * dv, dim=-1))
        
        # action
        u = torch.sum(torch.square(self._actions), dim=-1)
        action_reward = -u
        
        reward = 0.7 * velocity_reward + 0.2 * height_reward + 0.1 * avoidance_reward + 0.01 * action_reward

        reward = torch.where(self._collision, -1.0, reward)

        return reward

    def _reset_idx(self, env_ids):

        # reset scene
        super()._reset_idx(env_ids)

        # reset evader
        joint_pos = self._evader.data.default_joint_pos[env_ids]
        joint_vel = self._evader.data.default_joint_vel[env_ids]
        default_root_state = self._evader.data.default_root_state[env_ids]
        default_root_state[:, 0] = -51.0
        default_root_state[:, 1] = torch.rand(env_ids.shape) * 80.0 - 40.0
        self._evader.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._evader.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._evader.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._actions[env_ids] = 0.0
        self._collision[env_ids] = False
        self._history_distance.clear()
    
    def _get_observations(self):

        distance = torch.norm(self._lidar.data.ray_hits_w - self._lidar.data.pos_w.unsqueeze(1), p=2, dim=-1).clamp_max(LIDAR_RANGE)
        self._history_distance.append(distance)
        while(len(self._history_distance) < 3):
            self._history_distance.append(distance)
        
        history_distances = torch.stack(list(self._history_distance), dim=-1).flatten(1,2)

        observations = {
            "policy": {
                "history_distances": history_distances,
                "evader_states": torch.concat([
                        self._evader.data.root_pos_w[:, 2].unsqueeze(1) - self._desired_height,
                        self._evader.data.root_lin_vel_w - self._desired_velocity,
                        quat_to_z_axis(self._evader.data.root_quat_w)
                    ], dim=-1
                ),
                "actions": self._actions
            }
        }
        return observations

    # override
    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""

        # action space
        self.num_actions = self.cfg.num_actions
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # observation space
        self.single_observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Dict(
                OrderedDict([
                    ("history_distances", gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,))),
                    ("evader_states", gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))),
                    ("actions", gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)))
                ])
            )
        })
        #assert self.cfg.num_observations == calculate_num_observations(self.single_observation_space["policy"]), "num_observations and observation space mismatch"
        self.num_observations = self.cfg.num_observations

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        self.num_states = self.cfg.num_states
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)