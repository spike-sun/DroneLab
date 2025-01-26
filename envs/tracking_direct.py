import torch
from torch import abs, max, min, sin, cos, atan2
import gymnasium as gym
import numpy as np
import cv2
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.lab.envs import DirectRLEnvCfg, DirectRLEnv
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, CameraData
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from my_utils.controller import PositionController, quat_rotate_inv, quat_rotate
from my_utils.trajectory import EightTrajectory
from vae import VAE

SCALE = 1.0
MAX_ANGLE = 0.463647609

R = 0.5
M = 2  # 障碍物个数

@configclass
class MySceneCfg(InteractiveSceneCfg):

    terrain = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn = sim_utils.UsdFileCfg(
            usd_path = "/home/sbw/MyUAV/assets/ground/default_environment.usd",
            scale = (100.0, 100.0, 1.0),
            semantic_tags = [('class', 'ground')]
        )
    )
    
    light = AssetBaseCfg(
        prim_path = "/World/DomeLight",
        spawn = sim_utils.DomeLightCfg(intensity=2000.0)
    )
    
    evader = ArticulationCfg(
        prim_path = '{ENV_REGEX_NS}/evader',
        spawn = sim_utils.UsdFileCfg(
            usd_path = "/home/sbw/MyUAV/assets/cf2x/cf2x_blue.usd",
            scale = (SCALE, SCALE, SCALE),
            semantic_tags = [('class', 'cf2x')],
            activate_contact_sensors=True,
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
            pos = (0.0, 0.0, 2.0)
        ),
        actuators = {
            'dummy': ImplicitActuatorCfg(
                joint_names_expr = ['.*'],
                stiffness = 0.0,
                damping = 0.0,
            ),
        },
    )
    
    chaser = ArticulationCfg(
        prim_path = "{ENV_REGEX_NS}/chaser",
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"/home/sbw/MyUAV/assets/cf2x/cf2x_red.usd",
            semantic_tags = [("class", "cf2x")],
            scale = (SCALE, SCALE, SCALE),
            activate_contact_sensors=True,
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
            pos = (0.0, -0.5, 2.0),
            rot = (0.707, 0.0, 0.0, 0.707)  # yaw = pi/2
        ),
        actuators = {
            "dummy": ImplicitActuatorCfg(
                joint_names_expr = [".*"],
                stiffness = 0.0,
                damping = 0.0,
            ),
        }
    )
    '''
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/chaser/body/front_camera",
        height = 224,
        width = 224,
        data_types = ['depth'],
        spawn = sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e3)),
        offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention='world')
    )
    '''
    cylinder1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/cylinder1",
        spawn = sim_utils.UsdFileCfg(
            usd_path = "/home/sbw/MyUAV/assets/cylinder/cylinder.usd",
            scale = (R, R, 1.0),
            semantic_tags = [("class", "obstacles")]
        ),
        init_state = AssetBaseCfg.InitialStateCfg(
            pos = (1.2, 0.0, 0.0),
            rot = (1.0, 0.0, 0.0, 0.0)
        )
    )

    cylinder2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/cylinder2",
        spawn = sim_utils.UsdFileCfg(
            usd_path = "/home/sbw/MyUAV/assets/cylinder/cylinder.usd",
            scale = (R, R, 1.0),
            semantic_tags = [('class', 'obstacles')]
        ),
        init_state = AssetBaseCfg.InitialStateCfg(
            pos = (-1.2, 0.0, 0.0),
            rot = (1.0, 0.0, 0.0, 0.0)
        )
    )


@configclass
class TrackingEnvCfg(DirectRLEnvCfg):
    episode_length_s = 60
    decimation = 6
    num_actions = 4
    num_observations = 2 * M + 9 + num_actions + 12
    debug_vis = True
    sim = SimulationCfg(
        dt = 1.0 / 120.0,
        render_interval = decimation,
        disable_contact_processing = True,
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene = MySceneCfg(num_envs=16, env_spacing=8.0)


class TrackingEnv(DirectRLEnv):

    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._zeros = torch.zeros(self.num_envs, device=self.device)
        self._ones = torch.ones(self.num_envs, device=self.device)
        
        self._actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._force = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torque = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self._chaser: Articulation = self.scene['chaser']
        self._evader: Articulation = self.scene['evader']

        self._chaser_body_id = self._chaser.find_bodies("body")[0]
        self._evader_body_id = self._evader.find_bodies("body")[0]
        self._robot_mass = self._chaser.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._robot_weight_vector = torch.tensor([0.0, 0.0, self._robot_weight], device=self.device).repeat(self.num_envs, 1)
        self._unit_rotmat = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)

        self._traj_generator = EightTrajectory(self.num_envs, self.scene.env_origins, 4*R, self.device)
        self._controller = PositionController(num_envs=self.num_envs, device=self.device, scale=SCALE)

        self._pos_obs = torch.tensor([[
            [1.2, 0.0],
            [-1.2, 0.0]
        ]], device=self.device) + self.scene.env_origins[:, None, 0:2]

        self.set_debug_vis(self.cfg.debug_vis)
    
    # 每次step调用1次, policy应在外环
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
    
    # 每次step调用decimation次, 控制器应在内环
    def _apply_action(self):

        # chaser
        self._force[:, 0, 2] = self._robot_weight * (self._actions[:, 0] + 1.0)
        self._torque[:, 0, :] = self._controller.angvel_cmd(self._chaser.data.root_ang_vel_b, self._actions[:, 1:4] * 2.0)
        self._chaser.set_external_force_and_torque(self._force, self._torque, body_ids=self._chaser_body_id)
        
        # evader
        pos_des, vel_w_des, acc_w_des = self._traj_generator.step(self.cfg.sim.dt)
        force = torch.zeros((self.num_envs, 1, 3), device=self.device)
        force[:, 0, 2], torque = self._controller.pos_cmd(
            self._evader.data.root_pos_w,
            self._evader.data.root_quat_w,
            self._evader.data.root_lin_vel_w,
            self._evader.data.root_ang_vel_b,
            pos_des,
            vel_w_des,
            acc_w_des
        )
        self._evader.set_external_force_and_torque(force, torque.unsqueeze(1), body_ids=self._evader_body_id)
    
    # self.sim.step()
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.logical_or(self._chaser.data.root_pos_w[:, 2] > 4.0, self._chaser.data.root_pos_w[:, 2] < 0.1)
        terminated = torch.logical_or(terminated, torch.abs(self._chaser.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > 4.0)
        terminated = torch.logical_or(terminated, torch.abs(self._chaser.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > 4.0)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.collision | terminated, time_out
    
    def _get_rewards(self) -> torch.Tensor:
        
        # tracking
        dp = quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_pos_w - self._chaser.data.root_pos_w)
        dv = quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_lin_vel_w - self._chaser.data.root_lin_vel_w)
        rx = max(1.0 - abs(dp[:, 0] - 0.5), self._zeros)
        ry = max(1.0 - abs(atan2(dp[:, 1], dp[:, 0]) / MAX_ANGLE), self._zeros)
        rz = max(1.0 - abs(atan2(dp[:, 2], dp[:, 0]) / MAX_ANGLE), self._zeros)
        track_reward = torch.sqrt(rx * ry * rz)

        # action
        u = torch.norm(self._actions, p=2, dim=-1)
        action_penalty = u / (1.0 + u)
        
        reward = track_reward - 0.001 * action_penalty
        reward = torch.where(self.collision | self.occlusion, -1.0, reward)
        
        return reward

    def _reset_idx(self, env_ids):
        
        # reset scene(articulation(actuator, external force/torque), rigid_object, sensor, action/observation noise model), handle events
        super()._reset_idx(env_ids)

        # chaser
        joint_pos = self._chaser.data.default_joint_pos[env_ids]
        joint_vel = self._chaser.data.default_joint_vel[env_ids]
        default_root_state = self._chaser.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._chaser.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._chaser.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._chaser.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # evader
        joint_pos = self._evader.data.default_joint_pos[env_ids]
        joint_vel = self._evader.data.default_joint_vel[env_ids]
        default_root_state = self._evader.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._evader.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._evader.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._evader.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._actions[env_ids] = 0.0
        self._traj_generator.reset_idx(env_ids)
    
    def _get_observations(self):

        # collision
        p_obs = self._pos_obs - self._chaser.data.root_pos_w[:, None, 0:2]  # (N,M,2)
        p_obs = torch.concat([p_obs, torch.zeros(self.num_envs, M, 1, device=self.device)], dim=-1)  # (N,M,3)
        p_obs = quat_rotate_inv(self._chaser.data.root_quat_w.unsqueeze(1).repeat(1,2,1), p_obs)  # (N,M,3)
        p_obs = p_obs[:,:,0:2]  # (N,M,2)
        d_obs = torch.norm(p_obs, p=2, dim=-1)  # (N,M)
        d_target = torch.norm(self._evader.data.root_pos_w - self._chaser.data.root_pos_w, p=2, dim=-1)
        self.collision = torch.any(d_obs < R + 0.1, dim=1) | (d_target < 0.1)

        # occlusion
        _, occlusion = is_occluded(self._chaser.data.root_pos_w[:,0:2], self._evader.data.root_pos_w[:,0:2], self._pos_obs)  # (N,M,2), (N,M)
        self.occlusion = torch.any(occlusion, dim=-1)
        
        observations = {
            "policy": torch.concat(
                [
                    torch.flatten(p_obs, 1, 2),  # (N, 2*M)
                    self._chaser.data.root_lin_vel_b,
                    self._chaser.data.projected_gravity_b,
                    self._chaser.data.root_ang_vel_b,
                    self._actions,
                    quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_pos_w - self._chaser.data.root_pos_w),
                    quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_lin_vel_w - self._chaser.data.root_lin_vel_w),
                    self._evader.data.projected_gravity_b,
                    self._evader.data.root_ang_vel_b
                ], dim=-1
            )       
        }
        
        return observations

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self.scene.env_origins)
    
    # 修改DirectRLEnv类的动作空间, 让SAC能随机采样动作
    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations,))
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))  # 修改这一句, 原先范围是负无穷到正无穷

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


def save_tensor(image: torch.Tensor, path: str):
    image_np: np.array = image.cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    if image.dtype == torch.float32:
        image_np *= 255.0
        image_np = image_np.astype(np.uint8)
    elif image.dtype != torch.uint8:
        raise Exception(f"wtf {image.dtype}")
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_cv)


@torch.jit.script
def is_occluded(p1: torch.Tensor, p2: torch.Tensor, po: torch.Tensor, R=R):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, float) -> tuple[torch.Tensor, torch.Tensor]
    # 求障碍物在LOS坐标系（X轴指向目标，Y轴符合右手法则）下的坐标并判断遮挡
    # p1: (N,2), p2: (N,2), po: (N,M,2)
    # return: (N,M), (N)
    num_obs = po.size(1)
    Xl = p2 - p1  # (N,2)
    length = torch.norm(Xl, p=2, dim=-1)  # (N)
    Xl = Xl / length[:,None]  # (N,2)
    Rlw = torch.stack([Xl[:,0], Xl[:,1], -Xl[:,1], Xl[:,0]], dim=1).reshape(-1,2,2)  # (N,2,2)
    Rlw = Rlw.unsqueeze(1).repeat(1,num_obs,1,1)  # (N,M,2,2)
    uw = po - p1[:,None,:]  # (N,M,2)
    ul = Rlw @ uw.unsqueeze(-1)  # (N,M,2,1)
    ul = ul.squeeze(-1)  # (N,M,2)
    length = length.unsqueeze(-1).repeat(1,num_obs)  # (N,M)
    d = torch.abs(ul[:,:,1])  # (N,M)
    return ul, ((d < R) & (ul[:,:,0] > 0.0) & (ul[:,:,0] < length))