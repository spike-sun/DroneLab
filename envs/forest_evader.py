import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, ContactSensor, ContactSensorCfg
from omni.isaac.lab.sensors.ray_caster.patterns import LidarPatternCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass

from my_utils.controller import PositionController
from my_utils.math import quat_rotate_inv, quat_to_rotation_matrix, quat_to_z_axis
from my_utils.tensor_queue import TensorQueue
from .direct_rl_env import DirectRLEnvCfg, DirectRLEnv

SCALE = 1.0  # 修改无人机的尺寸需要同时修改控制器参数
LIDAR_RANGE = 5.0  # 激光雷达最大测距

@configclass
class SceneCfg(InteractiveSceneCfg):

    num_envs = 1

    env_spacing = 1.0

    terrain_cfg = TerrainImporterCfg(
        prim_path = '/World/ground',
        terrain_type = 'generator',
        terrain_generator = TerrainGeneratorCfg(
            seed = 1,
            size = (100.0, 100.0),
            num_rows = 1,
            num_cols = 1,
            use_cache = False,
            difficulty_range = (1.0, 1.0),
            sub_terrains = {
                'obstacles': HfDiscreteObstaclesTerrainCfg(
                    num_obstacles = 2000,
                    obstacle_height_mode = 'fixed',
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
        prim_path = '/World/DomeLight',
        spawn = sim_utils.DomeLightCfg(intensity=2000.0)
    )
    
    evader = ArticulationCfg(
        prim_path = '{ENV_REGEX_NS}/evader',
        spawn = sim_utils.UsdFileCfg(
            usd_path = '/home/sbw/MyUAV/assets/cf2x/cf2x_blue.usd',
            scale = (SCALE, SCALE, SCALE),
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
            'dummy': ImplicitActuatorCfg(
                joint_names_expr = ['.*'],
                stiffness = 0.0,
                damping = 0.0,
            )
        }
    )
    
    lidar_e = RayCasterCfg(
        prim_path = '{ENV_REGEX_NS}/evader/body',
        mesh_prim_paths = ['/World/ground/terrain'],
        attach_yaw_only = True,
        max_distance = LIDAR_RANGE,
        pattern_cfg = LidarPatternCfg(
            channels = 1,
            vertical_fov_range = (-0.0, 0.0),
            horizontal_fov_range = (-180.0, 180.0),
            horizontal_res = 22.5
        ),
        debug_vis = False,
        visualizer_cfg = VisualizationMarkersCfg(
            prim_path='/Visuals/RayCaster',
            markers={
                'hit': sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                )
            }
        )
    )

    contact_e = ContactSensorCfg(prim_path = '{ENV_REGEX_NS}/evader/.*')


@configclass
class ForestEvaderCfg(DirectRLEnvCfg):
    num_actions = 3
    num_observations = 58
    episode_length_s = 180
    decimation = 10
    sim = SimulationCfg(
        dt = 1.0 / 100.0,
        render_interval = decimation,
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx = PhysxCfg(
            gpu_found_lost_pairs_capacity = 2**23,
            gpu_total_aggregate_pairs_capacity = 2**23
        )
    )
    scene = SceneCfg()


class ForestEvader(DirectRLEnv):

    def __init__(self, cfg: ForestEvaderCfg, render_mode: str | None = None, **kwargs):
        self.lidar_history_length = 3
        super().__init__(cfg, render_mode, **kwargs)
        
        # assets
        self.evader: Articulation = self.scene['evader']
        self.lidar_e: RayCaster = self.scene['lidar_e']
        self.contact_e: ContactSensor = self.scene['contact_e']

        # buffers
        self.history_lidar_e = TensorQueue(self.device, self.num_envs, self.lidar_history_length, 16)
        self.actions_e = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self.force_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)
        self.torque_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)
        self.collision_e = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # constants
        self.evader_body_id = self.evader.find_bodies('body')[0]
        self.zeros = torch.zeros(self.num_envs, device=self.device)
        self.desired_velocity = torch.tensor([0.5, 0.0, 0.0], device=self.device)
        self.desired_height = 2.0
        self.robot_mass = self.evader.root_physx_view.get_masses()[0].sum()
        self.gravity_vector = torch.tensor(self.sim.cfg.gravity, device=self.device)  # [0, 0, -9.81]
        self.robot_weight_vector = self.robot_mass * self.gravity_vector  # [0, 0, -mg]
        self.robot_weight = torch.norm(self.robot_weight_vector)  # mg
        self.controller = PositionController(num_envs=self.num_envs, device=self.device, scale=SCALE)
    
    # 每step调用1次，policy应在外环
    def _pre_physics_step(self, actions):
        self.actions_e = actions.detach().clamp(-1.0, 1.0)
    
    # 每step调用decimation次，控制器应在内环
    def _apply_action(self):

        # 控制世界坐标系下线加速度
        self.force_e[:, 0, 2], self.torque_e[:, 0, :] = self.controller.acc_yaw_cmd(
            self.evader.data.root_quat_w,
            self.evader.data.root_ang_vel_b,
            self.actions_e * 2.0,
            self.zeros
        )
        self.evader.set_external_force_and_torque(self.force_e, self.torque_e, body_ids=self.evader_body_id)

        self.collision_e |= torch.any(torch.abs(self.contact_e.data.net_forces_w) > 1e-8, dim=(1, 2))
    
    # self.sim.step()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_range_e = torch.logical_or(self.evader.data.root_pos_w[:, 2] > 4.0, self.evader.data.root_pos_w[:, 2] < 0.1)
        terminated = out_range_e | self.collision_e
        timeout = self.episode_length_buf >= self.max_episode_length
        return terminated, timeout
    
    def _get_rewards(self) -> torch.Tensor:
        
        # collision
        dist = torch.norm(self.lidar_e.data.ray_hits_w - self.lidar_e.data.pos_w.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE)
        minDist, _ = torch.min(dist, dim=-1)
        prevDist = self.history_lidar_e.newest()
        prevMinDist, _ = torch.min(prevDist, dim=-1)
        avoidance_reward = torch.tanh(minDist - prevMinDist)
        
        # height
        dz = torch.abs(self.evader.data.root_pos_w[:, 2] - self.desired_height)
        height_reward = torch.exp(-dz)
        
        # velocity
        velocity_weights = torch.tensor([10.0, 1.0, 10.0], device=self.device)
        dv = self.evader.data.root_lin_vel_w - self.desired_velocity
        velocity_reward = torch.exp(-torch.sum(velocity_weights * dv * dv, dim=-1))
        
        # action
        u = torch.sum(torch.square(self.actions_e), dim=-1)
        action_reward = -u
        
        reward = 0.7 * velocity_reward + 0.2 * height_reward + 0.1 * avoidance_reward + 0.01 * action_reward

        reward = torch.where(self.collision_e, -1.0, reward)

        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        y = torch.rand(env_ids.shape) * 80.0 - 40.0
        root_state = self.evader.data.default_root_state[env_ids]
        root_state[:, 0] = -51.0
        root_state[:, 1] = y
        root_state[:, 7] = 0.5
        self.evader.write_root_state_to_sim(root_state, env_ids)
        self.evader.write_joint_state_to_sim(self.evader.data.default_joint_pos[env_ids], self.evader.data.default_joint_vel[env_ids], None, env_ids)

        self.actions_e[env_ids] = 0.0
        self.collision_e[env_ids] = False
        lidar_e = torch.norm(self.lidar_e.data.ray_hits_w - self.lidar_e.data.pos_w.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE) / LIDAR_RANGE
        self.history_lidar_e.init_ids(env_ids, lidar_e)
    
    def _get_observations(self):
        
        lidar_e = torch.norm(self.lidar_e.data.ray_hits_w - self.lidar_e.data.pos_w.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE) / LIDAR_RANGE
        self.history_lidar_e.append(lidar_e)

        observations = {
            'history_lidar': self.history_lidar_e.buffer,
            'height_error': self.evader.data.root_pos_w[:, 2].unsqueeze(1) - self.desired_height,
            'lin_vel_w_error': self.evader.data.root_lin_vel_w - self.desired_velocity,
            'z_axis': quat_to_z_axis(self.evader.data.root_quat_w),
            'last_action': self.actions_e
        }

        return observations

    def _configure_gym_env_spaces(self):

        self.action_space = Box(low=-1.0, high=1.0, shape=[3])
        self.observation_space = Dict(
            history_lidar = Box(low=-np.inf, high=np.inf, shape=[self.lidar_history_length, 16]),
            height_error = Box(low=-np.inf, high=np.inf, shape=[1]),
            lin_vel_w_error = Box(low=-np.inf, high=np.inf, shape=[3]),
            z_axis = Box(low=-np.inf, high=np.inf, shape=[3]),
            last_action = Box(low=-np.inf, high=np.inf, shape=[3])
        )


def to_numpy(x):
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x