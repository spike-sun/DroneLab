import os
import torch
from torch import abs, max, atan2
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.envs.direct_rl_env import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster, ContactSensor, Camera
from isaaclab.utils import configclass
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.spaces.torch import flatten_tensorized_space
from _utils.controller import PositionController
from _utils.math import quat_rotate_inv, quat_to_rotation_matrix, quat_to_z_axis
from _utils.tensor_queue import TensorQueue
from models.evader import EvaderPolicy
from .scene import ForestEvaderSceneCfg, ForestChaserAssetCfg
ENABLE_CAMERAS = int(os.getenv('ENABLE_CAMERAS', default=0))
MAX_ANGLE = 0.463647609


@configclass
class ForestChaserSceneCfg(ForestEvaderSceneCfg, ForestChaserAssetCfg):
    pass

@configclass
class ForestChaserCfg(DirectRLEnvCfg):
    seed = 42
    history_length = 3
    observe_camera = False
    evader_policy = 'logs/policy/evader/PPO_2025-02-13_21-25-01/checkpoints/best_agent.pt'
    action_space = Box(low=-1.0, high=1.0, shape=[4])
    observation_space = Dict(
        evader = Dict(
            lidar = Box(low=-np.inf, high=np.inf, shape=[history_length, 16]),
            height_error = Box(low=-np.inf, high=np.inf, shape=[1]),
            lin_vel_w_error = Box(low=-np.inf, high=np.inf, shape=[3]),
            z_axis = Box(low=-np.inf, high=np.inf, shape=[3]),
            last_action = Box(low=-np.inf, high=np.inf, shape=[3])
        ),
        chaser = Dict(
            lidar = Box(low=-np.inf, high=np.inf, shape=[history_length, 16]),
            relative_pos = Box(low=-np.inf, high=np.inf, shape=[3]),
            relative_vel = Box(low=-np.inf, high=np.inf, shape=[3]),
            lin_vel_b = Box(low=-np.inf, high=np.inf, shape=[3]),
            rotmat = Box(low=-np.inf, high=np.inf, shape=[9]),
            ang_vel_b = Box(low=-np.inf, high=np.inf, shape=[3]),
            last_action = Box(low=-np.inf, high=np.inf, shape=[4])
        )
    )
    if ENABLE_CAMERAS and observe_camera:
        observation_space['chaser']['rgb'] = Box(low=-np.inf, high=np.inf, shape=[224, 224, 3])
        observation_space['chaser']['depth'] = Box(low=-np.inf, high=np.inf, shape=[224, 224])
    episode_length_s = 180
    decimation = 6
    sim = SimulationCfg(
        dt = 1.0 / 120.0,
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
    viewer = ViewerCfg(eye=(-45.0, 0.0, 7.5), lookat=(-45.0, 0.0, 0.0))
    scene = ForestChaserSceneCfg()

class ForestChaser(DirectRLEnv):

    def __init__(self, cfg: ForestChaserCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.history_length = self.cfg.history_length
        self.observe_camera = self.cfg.observe_camera

        # evader
        self.evader: Articulation = self.scene['evader']
        self.evader_body_id = self.evader.find_bodies('body')[0]
        self.lidar_e: RayCaster = self.scene['lidar_e']
        self.history_lidar_e = TensorQueue(shape=[16], batch_size=self.num_envs, maxlen=self.cfg.history_length, dtype=torch.float, device=self.device)
        self.contact_e: ContactSensor = self.scene['contact_e']
        self.collision_e = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.actions_e = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self.force_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)
        self.torque_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)

        # evader policy
        evader_action_space = Box(low=-1.0, high=1.0, shape=[3])
        evader_observation_space = Dict(evader=self.single_observation_space['policy']['evader'])
        self.evader_observation = torch.zeros((self.num_envs, gym.spaces.flatdim(evader_observation_space)), device=self.device)
        weights = torch.load(cfg.evader_policy, weights_only=True)
        self.evader_policy = EvaderPolicy(evader_observation_space, evader_action_space, device=self.device).to(self.device)
        self.evader_policy.load_state_dict(weights['policy'])
        self.evader_policy.eval()
        self.evader_state_preprocessor = RunningStandardScaler(size=evader_observation_space, device=self.device)
        self.evader_state_preprocessor.load_state_dict(weights['state_preprocessor'])
        self.evader_state_preprocessor.eval()
        
        # chaser
        self.chaser: Articulation = self.scene['chaser']
        self.chaser_body_id = self.chaser.find_bodies('body')[0]
        self.contact_c: ContactSensor = self.scene['contact_c']
        self.lidar_c: RayCaster = self.scene['lidar_c']
        self.history_lidar_c = TensorQueue(shape=[16], batch_size=self.num_envs, maxlen=self.cfg.history_length, dtype=torch.float, device=self.device)
        self.collision_c = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.actions_c = torch.zeros((self.num_envs, 4), device=self.device)
        self.force_c = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.torque_c = torch.zeros((self.num_envs, 1, 3), device=self.device)
        if ENABLE_CAMERAS:
            self.camera_c: Camera = self.scene['camera_c']
            self.camera_c.set_intrinsic_matrices(
                torch.tensor(
                    [[
                        [224, 0,   112],
                        [0,   224, 112],
                        [0,   0,   1  ]
                    ]], device=self.device
                ).repeat(self.num_envs, 1, 1)
            )

        # constants
        self.zeros = torch.zeros(self.num_envs, device=self.device)
        self.desired_velocity = torch.tensor([0.5, 0.0, 0.0], device=self.device)
        self.desired_height = 2.0
        self.robot_mass = self.chaser.root_physx_view.get_masses()[0].sum()
        self.gravity_vector = torch.tensor(self.sim.cfg.gravity, device=self.device)  # [0, 0, -9.81]
        self.robot_weight_vector = self.robot_mass * self.gravity_vector  # [0, 0, -mg]
        self.robot_weight = torch.norm(self.robot_weight_vector)  # mg
        self.controller = PositionController(num_envs=self.num_envs, device=self.device)

    def _setup_scene(self):        
        if ENABLE_CAMERAS:
            import omni.replicator.core as rep
            prim = rep.get.prim_at_path('/World/ground')
            with prim:
                rep.modify.semantics([('class', 'obstacles')])
    
    # 每step调用1次，policy应在外环
    def _pre_physics_step(self, actions):
        with torch.inference_mode():
            actions_e, _, _ = self.evader_policy.compute({'states': self.evader_state_preprocessor(self.evader_observation)})
            self.actions_e = actions_e.detach().clamp(-1.0, 1.0)
            self.actions_c = actions.detach().clamp(-1.0, 1.0)
    
    # 每step调用decimation次，控制器应在内环
    def _apply_action(self):
        
        # evader 控制世界坐标系下线加速度
        self.force_e[:, 0, 2], self.torque_e[:, 0, :] = self.controller.acc_yaw(
            self.evader.data.root_quat_w,
            self.evader.data.root_ang_vel_b,
            self.actions_e * 2.0,
            self.zeros
        )
        self.evader.set_external_force_and_torque(self.force_e, self.torque_e, body_ids=self.evader_body_id)

        # chaser 控制体坐标系下角速度和推力
        self.force_c[:, 0, 2] = self.robot_weight * (self.actions_c[:, 0] + 1.0)
        self.torque_c[:, 0, :] = self.controller.angvel(self.chaser.data.root_ang_vel_b, self.actions_c[:, 1:4] * 2.0)
        self.chaser.set_external_force_and_torque(self.force_c, self.torque_c, body_ids=self.chaser_body_id)
        
        # 统计碰撞
        self.collision_c |= torch.any(torch.abs(self.contact_c.data.net_forces_w) > 1e-8, dim=(1, 2))
        self.collision_e |= torch.any(torch.abs(self.contact_e.data.net_forces_w) > 1e-8, dim=(1, 2))
    
    # self.sim.step()
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_range_e = (self.evader.data.root_pos_w[:, 2] > 4.0) | (self.evader.data.root_pos_w[:, 2] < 0.1)     
        out_range_c = (self.chaser.data.root_pos_w[:, 2] > 4.0) | (self.chaser.data.root_pos_w[:, 2] < 0.1)
        dist = torch.norm(self.evader.data.root_pos_w - self.chaser.data.root_pos_w, dim=-1)
        terminated = out_range_e | out_range_c | self.collision_e | self.collision_c | (dist > 4.0)
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        
        # avoid collision
        dist = self._get_dist(self.lidar_e)
        minDist, _ = torch.min(dist, dim=1)
        prevDist = self.history_lidar_c.newest()
        prevMinDist, _ = torch.min(prevDist, dim=1)
        avoidance_reward = torch.clamp(minDist - prevMinDist, -1.0, 1.0)

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
        super()._reset_idx(env_ids)

        new_y = torch.rand(env_ids.shape) * 80.0 - 40.0  # (-40, 40)    

        # reset chaser
        root_state = self.chaser.data.default_root_state[env_ids]
        root_state[:, 0] = -52.0 + (0.4 * torch.rand(len(env_ids), device=self.device) - 0.2)
        root_state[:, 1] = new_y
        root_state[:, 7] = 0.5
        self.chaser.write_root_state_to_sim(root_state, env_ids)
        self.chaser.write_joint_state_to_sim(self.chaser.data.default_joint_pos[env_ids], self.chaser.data.default_joint_vel[env_ids], None, env_ids)
        
        # reset evader
        root_state = self.evader.data.default_root_state[env_ids]
        root_state[:, 0] = -51.0
        root_state[:, 1] = new_y
        root_state[:, 7] = 0.5
        self.evader.write_root_state_to_sim(root_state, env_ids)
        self.evader.write_joint_state_to_sim(self.evader.data.default_joint_pos[env_ids], self.evader.data.default_joint_vel[env_ids], None, env_ids)

        with torch.inference_mode():
            self.actions_e[env_ids] = 0.0
            self.actions_c[env_ids] = 0.0
        
        self.collision_e[env_ids] = False
        self.collision_c[env_ids] = False
        lidar_e = self._get_dist(self.lidar_e)
        lidar_c = self._get_dist(self.lidar_c)
        self.history_lidar_e.init(lidar_e, env_ids)
        self.history_lidar_c.init(lidar_c, env_ids)
    
    def _get_observations(self):
        
        lidar_e = self._get_dist(self.lidar_e)
        lidar_c = self._get_dist(self.lidar_c)
        self.history_lidar_e.append(lidar_e)
        self.history_lidar_c.append(lidar_c)

        observations = {
            'evader': {
                'lidar': self.history_lidar_e.buffer,
                'height_error': self.evader.data.root_pos_w[:, 2].unsqueeze(1) - self.desired_height,
                'lin_vel_w_error': self.evader.data.root_lin_vel_w - self.desired_velocity,
                'z_axis': quat_to_z_axis(self.evader.data.root_quat_w),
                'last_action': self.actions_e
            },
            'chaser': {
                'lidar': self.history_lidar_c.buffer,
                'relative_pos': quat_rotate_inv(self.chaser.data.root_quat_w, self.evader.data.root_pos_w - self.chaser.data.root_pos_w),
                'relative_vel': quat_rotate_inv(self.chaser.data.root_quat_w, self.evader.data.root_lin_vel_w - self.chaser.data.root_lin_vel_w),
                'lin_vel_b': self.chaser.data.root_lin_vel_b,
                'rotmat': quat_to_rotation_matrix(self.chaser.data.root_quat_w).view(-1,9),
                'ang_vel_b': self.chaser.data.root_ang_vel_b,
                'last_action': self.actions_c
            }
        }
        if ENABLE_CAMERAS:
            if self.observe_camera:
                observations['chaser']['rgb'] = self.camera_c.data['rgb']
                observations['chaser']['depth'] = self.camera_c.data['depth'][:,:,:,0]
            else:
                self.extras['rgb'] = self.camera_c.data.output['rgb']
                self.extras['depth'] = self.camera_c.data.output['depth'][:,:,:,0]
        
        self.evader_observation = flatten_tensorized_space({'evader': observations['evader']})
        return {'policy': observations}

    def _get_dist(self, lidar: RayCaster):
        return torch.norm(lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1), dim=-1).clamp_max(10.0) / 10.0