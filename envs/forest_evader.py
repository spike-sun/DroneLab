import numpy as np
import torch
from gymnasium.spaces import Dict, Box
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster, ContactSensor
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from _utils.controller import PositionController
from _utils.math import quat_rotate_inv, quat_to_rotation_matrix, quat_to_z_axis
from _utils.tensor_queue import TensorQueue
from .scene import ForestEvaderSceneCfg


@configclass
class ForestEvaderCfg(DirectRLEnvCfg):
    seed = 42
    history_length = 3
    action_space = Box(low=-1.0, high=1.0, shape=[3])
    observation_space = Dict(
        evader = Dict(
            lidar = Box(low=-np.inf, high=np.inf, shape=[history_length, 16]),
            height_error = Box(low=-np.inf, high=np.inf, shape=[1]),
            lin_vel_w_error = Box(low=-np.inf, high=np.inf, shape=[3]),
            z_axis = Box(low=-np.inf, high=np.inf, shape=[3]),
            last_action = Box(low=-np.inf, high=np.inf, shape=[3])
        )
    )
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
    scene = ForestEvaderSceneCfg()


class ForestEvader(DirectRLEnv):

    def __init__(self, cfg: ForestEvaderCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # assets
        self.evader: Articulation = self.scene['evader']
        self.lidar_e: RayCaster = self.scene['lidar_e']
        self.contact_e: ContactSensor = self.scene['contact_e']

        # buffers
        self.history_lidar_e = TensorQueue(shape=[16], batch_size=self.num_envs, maxlen=self.cfg.history_length, dtype=torch.float, device=self.device)
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
        self.controller = PositionController(num_envs=self.num_envs, device=self.device)
    
    # 每step调用1次，policy应在外环
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions_e = actions.detach().clamp(-1.0, 1.0)
    
    # 每step调用decimation次，控制器应在内环
    def _apply_action(self):

        # 控制世界坐标系下线加速度
        self.force_e[:, 0, 2], self.torque_e[:, 0, :] = self.controller.acc_yaw(
            self.evader.data.root_quat_w,
            self.evader.data.root_ang_vel_b,
            self.actions_e * 2.0,
            self.zeros
        )
        self.evader.set_external_force_and_torque(self.force_e, self.torque_e, body_ids=self.evader_body_id)

        self.collision_e |= torch.any(torch.abs(self.contact_e.data.net_forces_w) > 1e-8, dim=(1, 2))
    
    # self.sim.step()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_range_e = (self.evader.data.root_pos_w[:, 2] > 4.0) | (self.evader.data.root_pos_w[:, 2] < 0.1)
        terminated = out_range_e | self.collision_e
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        
        # collision
        dist = self._get_dist(self.lidar_e)
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
        lidar_e = self._get_dist(self.lidar_e)
        self.history_lidar_e.init(lidar_e, env_ids)
    
    def _get_observations(self):
        
        lidar_e = self._get_dist(self.lidar_e)
        self.history_lidar_e.append(lidar_e)

        observations = {
            'evader': {
                'lidar': self.history_lidar_e.buffer,
                'height_error': self.evader.data.root_pos_w[:, 2].unsqueeze(1) - self.desired_height,
                'lin_vel_w_error': self.evader.data.root_lin_vel_w - self.desired_velocity,
                'z_axis': quat_to_z_axis(self.evader.data.root_quat_w),
                'last_action': self.actions_e
            }
        }
        
        return {'policy': observations}
    
    def _get_dist(self, lidar: RayCaster):
        return torch.norm(lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1), dim=-1).clamp_max(10.0) / 10.0