import os
import torch
from torch import abs, max, atan2
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg, PinholeCameraCfg, PhysxCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from .direct_rl_env import DirectRLEnvCfg, DirectRLEnv
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, ContactSensor, ContactSensorCfg, Camera, CameraCfg
from omni.isaac.lab.sensors.ray_caster.patterns import LidarPatternCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass
 
from my_utils.controller import PositionController
from my_utils.math import quat_rotate_inv, quat_to_rotation_matrix, quat_to_z_axis
from my_utils.tensor_queue import TensorQueue

ENABLE_CAMERAS = int(os.getenv('ENABLE_CAMERAS', '0'))
SCALE = 1.0  # 修改无人机的尺寸需要同时修改控制器参数
MAX_ANGLE = 0.463647609  # FOV角
LIDAR_RANGE = 5.0  # 激光雷达最大测距

@configclass
class MySceneCfg(InteractiveSceneCfg):

    num_envs = 2048

    env_spacing = 8.0

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
            semantic_tags = [('class', 'cf2x')],
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
            pos = (0.25, 0.0, 2.0),
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
    
    chaser = ArticulationCfg(
        prim_path = '{ENV_REGEX_NS}/chaser',
        spawn = sim_utils.UsdFileCfg(
            usd_path = f'/home/sbw/MyUAV/assets/cf2x/cf2x_red.usd',
            scale = (SCALE, SCALE, SCALE),
            #semantic_tags = [('class', 'cf2x')],
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
            pos = (-0.25, 0.0, 2.0),
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
        max_distance = 10.0,
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

    lidar_c = RayCasterCfg(
        prim_path = '{ENV_REGEX_NS}/chaser/body',
        mesh_prim_paths = ['/World/ground/terrain'],
        attach_yaw_only = True,
        max_distance = 10.0,
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

    contact_c = ContactSensorCfg(prim_path = '{ENV_REGEX_NS}/chaser/.*')

    camera_c = CameraCfg(
        prim_path = '{ENV_REGEX_NS}/chaser/body/front_camera',
        offset = CameraCfg.OffsetCfg(convention='world'),
        spawn = PinholeCameraCfg(),
        data_types = ['depth', 'rgb'],
        width = 224,
        height = 224
    )


@configclass
class ForestEnvCfg(DirectRLEnvCfg):
    episode_length_s = 60
    decimation = 10
    num_actions = 1
    num_observations = 1
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
    scene = MySceneCfg()


class ForestEnv(DirectRLEnv):

    def __init__(self, cfg: ForestEnvCfg, render_mode: str | None = None, **kwargs):
        self.lidar_history_length = 3
        super().__init__(cfg, render_mode, **kwargs)

        # chaser
        self.chaser: Articulation = self.scene['chaser']
        self.chaser_body_id = self.chaser.find_bodies('body')[0]
        self.contact_c: ContactSensor = self.scene['contact_c']
        self.lidar_c: RayCaster = self.scene['lidar_c']
        self.history_lidar_c = TensorQueue(self.device, self.num_envs, self.lidar_history_length, 16)
        self.collision_c = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.actions_c = torch.zeros((self.num_envs, 4), device=self.device)
        self.force_c = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.torque_c = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.camera_c: Camera = self.scene['camera_c']
        intrinsic_matrix = torch.tensor(
            [
                [224, 0,   112],
                [0,   224, 112],
                [0,   0,   1  ]
            ], device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.camera_c.set_intrinsic_matrices(intrinsic_matrix)
        
        # evader
        self.evader: Articulation = self.scene['evader']
        self.evader_body_id = self.evader.find_bodies('body')[0]
        self.lidar_e: RayCaster = self.scene['lidar_e']
        self.history_lidar_e = TensorQueue(self.device, self.num_envs, self.lidar_history_length, 16)
        self.actions_e = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self.force_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)
        self.torque_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)

        # constants
        self.zeros = torch.zeros(self.num_envs, device=self.device)
        self.desired_velocity = torch.tensor([0.5, 0.0, 0.0], device=self.device)
        self.desired_height = 2.0
        self.robot_mass = self.chaser.root_physx_view.get_masses()[0].sum()
        self.gravity_vector = torch.tensor(self.sim.cfg.gravity, device=self.device)  # [0, 0, -9.81]
        self.robot_weight_vector = self.robot_mass * self.gravity_vector  # [0, 0, -mg]
        self.robot_weight = torch.norm(self.robot_weight_vector)  # mg
        self.controller = PositionController(num_envs=self.num_envs, device=self.device, scale=SCALE)

    def _setup_scene(self):
        if ENABLE_CAMERAS:
            import omni.replicator.core as rep
            prim = rep.get.prim_at_path('/World/ground')
            with prim:
                rep.modify.semantics([('class', 'obstacles')])
    
    # 每step调用1次，policy应在外环
    def _pre_physics_step(self, actions):
        self.actions_e = actions['evader'].detach().clamp(-1.0, 1.0)
        self.actions_c = actions['chaser'].detach().clamp(-1.0, 1.0)
    
    # 每step调用decimation次，控制器应在内环
    def _apply_action(self):

        # evader 控制世界坐标系下线加速度
        self.force_e[:, 0, 2], self.torque_e[:, 0, :] = self.controller.acc_yaw_cmd(
            self.evader.data.root_quat_w,
            self.evader.data.root_ang_vel_b,
            self.actions_e * 2.0,
            self.zeros
        )
        self.evader.set_external_force_and_torque(self.force_e, self.torque_e, body_ids=self.evader_body_id)

        # chaser 控制体坐标系下角速度和推力
        self.force_c[:, 0, 2] = self.robot_weight * (self.actions_c[:, 0] + 1.0)
        self.torque_c[:, 0, :] = self.controller.angvel_cmd(self.chaser.data.root_ang_vel_b, self.actions_c[:, 1:4] * 2.0)
        self.chaser.set_external_force_and_torque(self.force_c, self.torque_c, body_ids=self.chaser_body_id)
        
        # 统计碰撞
        self.collision_c |= torch.any(torch.abs(self.contact_c.data.net_forces_w) > 1e-8, dim=(1, 2))
    
    # self.sim.step()
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        out_range_e = torch.logical_or(self.evader.data.root_pos_w[:, 2] > 4.0, self.evader.data.root_pos_w[:, 2] < 0.1)
        if(torch.any(out_range_e)):
            print('WARNING: evader out of range')
        
        out_range_c = torch.logical_or(self.chaser.data.root_pos_w[:, 2] > 4.0, self.chaser.data.root_pos_w[:, 2] < 0.1)
        dist = torch.norm(self.evader.data.root_pos_w - self.chaser.data.root_pos_w, p=2, dim=-1)
        terminated = out_range_c | self.collision_c | (dist > 3.0)

        timeout = self.episode_length_buf >= self.max_episode_length

        return terminated, timeout
    
    def _get_rewards(self) -> torch.Tensor:
        
        # avoid collision
        dist = torch.norm(self.lidar_c.data.ray_hits_w - self.lidar_c.data.pos_w.unsqueeze(1), p=2, dim=-1).clamp_max(LIDAR_RANGE)
        minDist, _ = torch.min(dist, dim=1)
        prevDist = self.history_lidar_c.newest()
        prevMinDist, _ = torch.min(prevDist, dim=1)
        avoidance_reward = torch.tanh(minDist - prevMinDist)

        # tracking
        dp = quat_rotate_inv(self.chaser.data.root_quat_w, self.evader.data.root_pos_w - self.chaser.data.root_pos_w)
        rx = max(1.0 - (abs(dp[:, 0] - 1.0)), self.zeros)
        ry = max(1.0 - abs(atan2(dp[:, 1], dp[:, 0]) / MAX_ANGLE), self.zeros)
        rz = max(1.0 - abs(atan2(dp[:, 2], dp[:, 0]) / MAX_ANGLE), self.zeros)
        track_reward = torch.sqrt(rx * ry * rz)

        # height
        dz = torch.abs(self.chaser.data.root_pos_w[:, 2] - self.desired_height)
        height_reward = torch.exp(-dz)
        
        # action
        u = torch.sum(torch.square(self.actions_c), dim=-1)
        action_reward = -u
        
        reward = 0.7 * track_reward + 0.2 * height_reward + 0.1 * avoidance_reward + 0.01 * action_reward

        reward = torch.where(self.collision_c, -1.0, reward)

        return reward

    def _reset_idx(self, env_ids):

        # reset scene
        super()._reset_idx(env_ids)

        new_y = torch.rand(env_ids.shape) * 80.0 - 40.0  # (-40, 40)    

        # reset chaser
        joint_pos = self.chaser.data.default_joint_pos[env_ids]
        joint_vel = self.chaser.data.default_joint_vel[env_ids]
        default_root_state = self.chaser.data.default_root_state[env_ids]
        default_root_state[:, 0] = -52.0
        default_root_state[:, 1] = new_y
        default_root_state[:, 7] = 0.5
        self.chaser.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.chaser.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.chaser.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # reset evader
        joint_pos = self.evader.data.default_joint_pos[env_ids]
        joint_vel = self.evader.data.default_joint_vel[env_ids]
        default_root_state = self.evader.data.default_root_state[env_ids]
        default_root_state[:, 0] = -51.0
        default_root_state[:, 1] = new_y
        default_root_state[:, 7] = 0.5
        self.evader.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.evader.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.evader.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        with torch.inference_mode():
            self.actions_e[env_ids] = 0.0
            self.actions_c[env_ids] = 0.0

        self.collision_c[env_ids] = False
        lidar_e = torch.norm(self.lidar_e.data.ray_hits_w - self.lidar_e.data.pos_w.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE) / LIDAR_RANGE
        lidar_c = torch.norm(self.lidar_c.data.ray_hits_w - self.lidar_c.data.pos_w.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE) / LIDAR_RANGE
        self.history_lidar_e.init_ids(env_ids, lidar_e)
        self.history_lidar_c.init_ids(env_ids, lidar_c)
    
    def _get_observations(self):
        
        lidar_e = torch.norm(self.lidar_e.data.ray_hits_w - self.lidar_e.data.pos_w.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE) / LIDAR_RANGE
        lidar_c = torch.norm(self.lidar_c.data.ray_hits_w - self.lidar_c.data.pos_w.unsqueeze(1), dim=-1).clamp_max(LIDAR_RANGE) / LIDAR_RANGE
        self.history_lidar_e.append(lidar_e)
        self.history_lidar_c.append(lidar_c)

        observations = {
            'evader': {
                'history_lidar': self.history_lidar_e.buffer,
                'height_error': self.evader.data.root_pos_w[:, 2].unsqueeze(1) - self.desired_height,
                'lin_vel_w_error': self.evader.data.root_lin_vel_w - self.desired_velocity,
                'z_axis': quat_to_z_axis(self.evader.data.root_quat_w),
                'last_action': self.actions_e
            },
            'chaser': {
                'history_lidar': self.history_lidar_c.buffer,
                'relative_pos': quat_rotate_inv(self.chaser.data.root_quat_w, self.evader.data.root_pos_w - self.chaser.data.root_pos_w),
                'relative_vel': quat_rotate_inv(self.chaser.data.root_quat_w, self.evader.data.root_lin_vel_w - self.chaser.data.root_lin_vel_w),
                'lin_vel_b': self.chaser.data.root_lin_vel_b,
                'rotmat': quat_to_rotation_matrix(self.chaser.data.root_quat_w).view(-1,9),
                'ang_vel_b': self.chaser.data.root_ang_vel_b,
                'last_action': self.actions_c
            }
        }

        if ENABLE_CAMERAS:
            observations['chaser']['rgb'] = self.camera_c.data.output['rgb'][:,:,:,0:3]
            observations['chaser']['depth'] = self.camera_c.data.output['depth'][:,:,:,0].cpu()
        
        return observations
    
    # override
    def _configure_gym_env_spaces(self):

        self.action_space = Dict(
            evader = Box(low=-1.0, high=1.0, shape=[3]),
            chaser = Box(low=-1.0, high=1.0, shape=[4])
        )

        if ENABLE_CAMERAS:
            self.observation_space = Dict(
                evader = Dict(
                    history_lidar = Box(low=-np.inf, high=np.inf, shape=[16, self.lidar_history_length]),
                    height_error = Box(low=-np.inf, high=np.inf, shape=[1]),
                    lin_vel_w_error = Box(low=-np.inf, high=np.inf, shape=[3]),
                    z_axis = Box(low=-np.inf, high=np.inf, shape=[3]),
                    last_action = Box(low=-np.inf, high=np.inf, shape=[3])
                ),
                chaser = Dict(
                    history_lidar = Box(low=-np.inf, high=np.inf, shape=[16, self.lidar_history_length]),
                    relative_pos = Box(low=-np.inf, high=np.inf, shape=[3]),
                    relative_vel = Box(low=-np.inf, high=np.inf, shape=[3]),
                    lin_vel_b = Box(low=-np.inf, high=np.inf, shape=[3]),
                    rotmat = Box(low=-np.inf, high=np.inf, shape=[9]),
                    ang_vel_b = Box(low=-np.inf, high=np.inf, shape=[3]),
                    last_action = Box(low=-np.inf, high=np.inf, shape=[4]),
                    rgb = Box(low=-np.inf, high=np.inf, shape=[224, 224, 3]),
                    depth = Box(low=-np.inf, high=np.inf, shape=[224, 224])
                )
            )
        else:
            self.observation_space = Dict(
                evader = Dict(
                    history_lidar = Box(low=-np.inf, high=np.inf, shape=[16, self.lidar_history_length]),
                    height_error = Box(low=-np.inf, high=np.inf, shape=[1]),
                    lin_vel_w_error = Box(low=-np.inf, high=np.inf, shape=[3]),
                    z_axis = Box(low=-np.inf, high=np.inf, shape=[3]),
                    last_action = Box(low=-np.inf, high=np.inf, shape=[3])
                ),
                chaser = Dict(
                    history_lidar = Box(low=-np.inf, high=np.inf, shape=[16, self.lidar_history_length]),
                    relative_pos = Box(low=-np.inf, high=np.inf, shape=[3]),
                    relative_vel = Box(low=-np.inf, high=np.inf, shape=[3]),
                    lin_vel_b = Box(low=-np.inf, high=np.inf, shape=[3]),
                    rotmat = Box(low=-np.inf, high=np.inf, shape=[9]),
                    ang_vel_b = Box(low=-np.inf, high=np.inf, shape=[3]),
                    last_action = Box(low=-np.inf, high=np.inf, shape=[4])
                )
            )