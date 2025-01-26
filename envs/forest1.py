RECORD = True
from collections import deque
import os
from datetime import datetime

import torch
from torch import abs, max, atan2
import gymnasium as gym
import numpy as np
import h5py
from time import time

if os.environ.get("ENABLE_CAMERAS", 0) == 1:
    import omni.replicator.core as rep
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationCfg, PinholeCameraCfg, PhysxCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.lab.envs import DirectRLEnvCfg, DirectRLEnv
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, ContactSensor, ContactSensorCfg, Camera, CameraCfg
from omni.isaac.lab.sensors.ray_caster.patterns import LidarPatternCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

from models.evader import EvaderPolicy, EvaderValue
from my_utils.controller import PositionController
from my_utils.math import quat_rotate_inv, quat_to_rotation_matrix, quat_to_z_axis


SCALE = 1.0  # 修改无人机的尺寸需要同时修改控制器参数
MAX_ANGLE = 0.463647609  # FOV角
LIDAR_RANGE = 5.0  # 激光雷达最大测距

@configclass
class MySceneCfg(InteractiveSceneCfg):

    num_envs = 2048

    env_spacing = 8.0

    terrain_cfg = TerrainImporterCfg(
        prim_path = "/World/ground",
        terrain_type = "generator",
        terrain_generator = TerrainGeneratorCfg(
            seed = 1,
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
            pos = (0.25, 0.0, 2.0),
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
    
    chaser = ArticulationCfg(
        prim_path = "{ENV_REGEX_NS}/chaser",
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"/home/sbw/MyUAV/assets/cf2x/cf2x_red.usd",
            scale = (SCALE, SCALE, SCALE),
            #semantic_tags = [("class", "cf2x")],
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
            "dummy": ImplicitActuatorCfg(
                joint_names_expr = [".*"],
                stiffness = 0.0,
                damping = 0.0,
            )
        }
    )
    
    lidar_e = RayCasterCfg(
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

    lidar_c = RayCasterCfg(
        prim_path = "{ENV_REGEX_NS}/chaser/body",
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

    contact_c = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/chaser/.*")

    if RECORD:
        camera_c = CameraCfg(
            prim_path = "{ENV_REGEX_NS}/chaser/body/front_camera",
            offset = CameraCfg.OffsetCfg(convention="world"),
            spawn = PinholeCameraCfg(),
            data_types = ["depth", "rgb"],
            width = 224,
            height = 224,
        )


@configclass
class ForestEnvCfg(DirectRLEnvCfg):
    episode_length_s = 180
    decimation = 6  # 控制周期0.05s
    num_actions = 4
    num_observations = 79
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
        
        if RECORD:
            assert cfg.scene.num_envs == 1, "num_envs should be 1 if recording"
        
        super().__init__(cfg, render_mode, **kwargs)

        # chaser
        self._chaser: Articulation = self.scene["chaser"]
        self._chaser_body_id = self._chaser.find_bodies("body")[0]
        self._contact_c: ContactSensor = self.scene["contact_c"]
        self._lidar_c: RayCaster = self.scene["lidar_c"]
        self._history_distance_c = deque(maxlen=3)
        self._collision_c = torch.tensor([False], device=self.device).repeat(self.num_envs)
        self._actions_c = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._force_c = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torque_c = torch.zeros((self.num_envs, 1, 3), device=self.device)
        
        # evader
        self._evader: Articulation = self.scene["evader"]
        self._evader_body_id = self._evader.find_bodies("body")[0]
        self._lidar_e: RayCaster = self.scene["lidar_e"]
        self._history_distance_e = deque(maxlen=3)
        self._actions_e = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self._force_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)
        self._torque_e = torch.zeros((self.num_envs, 1, 3), device=self.device, requires_grad=False)
        self._observation_e = torch.zeros((self.num_envs, 55), device=self.device, requires_grad=False)

        if RECORD:
            self._camera: Camera = self.scene['camera_c']
            intrinsic_matrix = torch.tensor(
                [
                    [224, 0,   112],
                    [0,   224, 112],
                    [0,   0,   1  ]
                ], device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1, 1)
            self._camera.set_intrinsic_matrices(intrinsic_matrix)

        # evader策略
        evader_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        evader_observation_space = gym.spaces.Dict({
            "history_distances": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,)),
            "evader_states": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
            "actions": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        })
        evader_observation_space_flatten = gym.spaces.flatten_space(evader_observation_space)
        agent_cfg = PPO_DEFAULT_CONFIG
        agent_cfg["state_preprocessor"] = RunningStandardScaler
        agent_cfg["state_preprocessor_kwargs"] = {"size": evader_observation_space_flatten, "device": self.device}
        agent_cfg["value_preprocessor"] = RunningStandardScaler
        agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}
        models = {
            "policy": EvaderPolicy(evader_observation_space, evader_action_space, device=self.device),
            "value": EvaderValue(evader_observation_space, 1, device=self.device)
        }
        agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
        agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        agent_cfg["random_timesteps"] = 0  # ignore random timesteps
        self._evader_agent = PPO(
            models=models,
            memory=None,
            cfg=agent_cfg,
            observation_space=evader_observation_space_flatten,
            action_space=evader_action_space,
            device=self.device,
        )
        self._evader_agent.init()
        self._evader_agent.load("logs/skrl/forest/PPO_2024-12-11_12-11-05/checkpoints/best_agent.pt")
        self._evader_agent.set_running_mode("eval")

        # 常量
        self._zeros = torch.zeros(self.num_envs, device=self.device)
        self._ones = torch.ones(self.num_envs, device=self.device)
        self._eye3 = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._desired_velocity = torch.tensor([0.5, 0.0, 0.0], device=self.device)
        self._desired_height = 2.0
        self._robot_mass = self._chaser.root_physx_view.get_masses()[0].sum()
        self._gravity_vector = torch.tensor(self.sim.cfg.gravity, device=self.device)  # [0, 0, -9.81]
        self._robot_weight_vector = self._robot_mass * self._gravity_vector  # [0, 0, -mg]
        self._robot_weight = torch.abs(self._robot_weight_vector[2])
        self._controller = PositionController(num_envs=self.num_envs, device=self.device, scale=SCALE)

        # 模仿学习数据集
        self._student_buf = {}
        self._student_dataset = []
        self.delta = 8
        self.data_path = f"data/{self.cfg.seed}"
        if os.path.exists(self.data_path):
            raise Exception(f"The directory '{self.data_path}' already exists.")
        else:
            os.makedirs(self.data_path)
            print(f"The directory '{self.data_path}' has been created.")

    def _setup_scene(self):
        if os.environ.get("ENABLE_CAMERAS", 0) == 1:
            prim = rep.get.prim_at_path("/World/ground")
            print(prim)
            with prim:
                rep.modify.semantics([('class', 'obstacles')])
    
    # 每step调用1次，policy应在外环
    def _pre_physics_step(self, actions: torch.Tensor):

        # 推理evader动作
        with torch.inference_mode():
            _, _, outputs = self._evader_agent.act(self._observation_e, timestep=0, timesteps=0)
            self._actions_e = outputs["mean_actions"].clone().clamp(-1.0, 1.0)

        # chaser动作
        self._actions_c = actions.clone().clamp(-1.0, 1.0)

        if RECORD:
            # 保存a_{t}
            # 此时字典里已经有last_action和其他观测了
            self._student_buf["action"] = self._actions_c[0].cpu()
            self._student_dataset.append(self._student_buf)
            self._student_buf = {}  # 不要用clear，用clear的话列表里存的都是引用
    
    # 每substep调用一次，每step调用decimation次，控制器应在内环
    def _apply_action(self):

        # evader 控制世界坐标系下线加速度
        self._force_e[:, 0, 2], self._torque_e[:, 0, :] = self._controller.acc_yaw_cmd(
            self._evader.data.root_quat_w,
            self._evader.data.root_ang_vel_b,
            self._actions_e * 2.0,
            self._zeros
        )
        self._evader.set_external_force_and_torque(self._force_e, self._torque_e, body_ids=self._evader_body_id)

        # chaser 控制体坐标系下角速度和推力
        self._force_c[:, 0, 2] = self._robot_weight * (self._actions_c[:, 0] + 1.0)
        self._torque_c[:, 0, :] = self._controller.angvel_cmd(self._chaser.data.root_ang_vel_b, self._actions_c[:, 1:4] * 2.0)
        self._chaser.set_external_force_and_torque(self._force_c, self._torque_c, body_ids=self._chaser_body_id)
        
        # 统计decimation个substep中是否出现碰撞
        self._collision_c = self._collision_c | torch.any(torch.abs(self._contact_c.data.net_forces_w) > 1e-8, dim=(1, 2))
    
    # self.sim.step()
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        out_range_e = torch.logical_or(self._evader.data.root_pos_w[:, 2] > 4.0, self._evader.data.root_pos_w[:, 2] < 0.1)
        if(torch.any(out_range_e)):
            print("WARNING: evader out of range")
        
        out_range_c = torch.logical_or(self._chaser.data.root_pos_w[:, 2] > 4.0, self._chaser.data.root_pos_w[:, 2] < 0.1)
        dist = torch.norm(self._evader.data.root_pos_w - self._chaser.data.root_pos_w, p=2, dim=-1)
        terminated = out_range_c | self._collision_c | (dist > 3.0)
        if(torch.any(terminated)):
            print("WARNING: terminated")

        timeout = self.episode_length_buf >= self.max_episode_length

        if RECORD:
            assert len(terminated) == 1 and len(timeout) == 1, "is there other envs?"
            # 只在跑完完整轨迹时保存
            # 保存 a_{t-1}, s_{t}, a_{t}
            if(timeout[0] | terminated[0]):
                if(timeout[0]):
                    stacked = stack_tensors(self._student_dataset)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    path = f"{self.data_path}/trajectory_{timestamp}.hdf5"
                    with h5py.File(path, "w") as f:
                        for k, v in stacked.items():
                            f.create_dataset(k, data=v)
                    print(f"Dataset saved at {path}.")
                self._student_dataset = []

        return terminated, timeout
    
    def _get_rewards(self) -> torch.Tensor:
        
        # avoid collision
        dist = torch.norm(self._lidar_c.data.ray_hits_w - self._lidar_c.data.pos_w.unsqueeze(1), p=2, dim=-1).clamp_max(LIDAR_RANGE)
        minDist, _ = torch.min(dist, dim=-1)
        prevDist = self._history_distance_c[-1]
        prevMinDist, _ = torch.min(prevDist, dim=-1)
        avoidance_reward = torch.tanh(minDist - prevMinDist)

        # tracking
        dp = quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_pos_w - self._chaser.data.root_pos_w)
        rx = max(1.0 - (abs(dp[:, 0] - 1.0)), self._zeros)
        ry = max(1.0 - abs(atan2(dp[:, 1], dp[:, 0]) / MAX_ANGLE), self._zeros)
        rz = max(1.0 - abs(atan2(dp[:, 2], dp[:, 0]) / MAX_ANGLE), self._zeros)
        track_reward = torch.sqrt(rx * ry * rz)

        # height
        dz = torch.abs(self._chaser.data.root_pos_w[:, 2] - self._desired_height)
        height_reward = torch.exp(-dz)
        
        # action
        action_weights = torch.tensor([1.0, 10.0, 10.0, 10.0], device=self.device)
        u = torch.sum(torch.square(self._actions_c), dim=-1)
        action_reward = -u

        # print(track_reward[2], height_reward[2], avoidance_reward[2], action_reward[2])
        
        reward = 0.7 * track_reward + 0.2 * height_reward + 0.1 * avoidance_reward + 0.01 * action_reward

        reward = torch.where(self._collision_c, -1.0, reward)

        return reward

    def _reset_idx(self, env_ids):

        # reset scene
        super()._reset_idx(env_ids)

        if RECORD:
            new_y = - 40.0 + self.delta * 10.0
            print(new_y)
            if self.delta > 8:
                exit()
            else:
                self.delta += 1
        else:
            new_y = torch.rand(env_ids.shape) * 80.0 - 40.0  # (-40, 40)    

        # reset chaser
        joint_pos = self._chaser.data.default_joint_pos[env_ids]
        joint_vel = self._chaser.data.default_joint_vel[env_ids]
        default_root_state = self._chaser.data.default_root_state[env_ids]
        default_root_state[:, 0] = -52.0
        default_root_state[:, 1] = new_y
        self._chaser.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._chaser.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._chaser.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # reset evader
        joint_pos = self._evader.data.default_joint_pos[env_ids]
        joint_vel = self._evader.data.default_joint_vel[env_ids]
        default_root_state = self._evader.data.default_root_state[env_ids]
        default_root_state[:, 0] = -51.0
        default_root_state[:, 1] = new_y
        self._evader.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._evader.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._evader.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        with torch.inference_mode():
            self._actions_e[env_ids] = 0.0
        self._history_distance_e.clear()
        
        self._actions_c[env_ids] = 0.0
        self._collision_c[env_ids] = False
        self._history_distance_c.clear()
    
    def _get_observations(self):

        # evader observations
        distance_e = torch.norm(self._lidar_e.data.ray_hits_w - self._lidar_e.data.pos_w.unsqueeze(1), p=2, dim=-1).clamp_max(LIDAR_RANGE)
        self._history_distance_e.append(distance_e)
        while(len(self._history_distance_e) < 3):
            self._history_distance_e.append(distance_e)
        history_distances_e = torch.stack(list(self._history_distance_e), dim=-1).flatten(1,2)
        self._observation_e = torch.concat([
            # 排列顺序手动和字典迭代顺序对齐
            self._actions_e,
            self._evader.data.root_pos_w[:, 2].unsqueeze(1) - self._desired_height,
            self._evader.data.root_lin_vel_w - self._desired_velocity,
            quat_to_z_axis(self._evader.data.root_quat_w),
            history_distances_e
        ], dim=-1)

        # chaser observations
        distance_c = torch.norm(self._lidar_c.data.ray_hits_w - self._lidar_c.data.pos_w.unsqueeze(1), p=2, dim=-1).clamp_max(LIDAR_RANGE)
        self._history_distance_c.append(distance_c)
        while(len(self._history_distance_c) < 3):
            self._history_distance_c.append(distance_c)
        history_distances_c = torch.stack(list(self._history_distance_c), dim=-1)
        observations = {
            "policy": {
                "history_distances": history_distances_c,
                "states": torch.concat([
                    quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_pos_w - self._chaser.data.root_pos_w),  # 3
                    quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_lin_vel_w - self._chaser.data.root_lin_vel_w),  # 3
                    quat_to_z_axis(self._evader.data.root_quat_w),  # 3
                    self._actions_e,  # 3
                    self._chaser.data.root_lin_vel_b,  # 3
                    quat_to_rotation_matrix(self._chaser.data.root_quat_w).view(-1,9),  # 9
                    self._chaser.data.root_ang_vel_b,  # 3
                    self._actions_c  # 4
                ], dim=-1)
            }
        }

        if RECORD:
            # 保存 a_{t-1}, s_{t}
            self._student_buf["evader_state"] = torch.concat([
                quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_pos_w - self._chaser.data.root_pos_w),  # 3
                quat_rotate_inv(self._chaser.data.root_quat_w, self._evader.data.root_lin_vel_w - self._chaser.data.root_lin_vel_w),  # 3
            ], dim=-1)[0].cpu()
            self._student_buf["chaser_state"] = torch.concat([
                self._chaser.data.root_lin_vel_b,  # 3
                quat_to_rotation_matrix(self._chaser.data.root_quat_w).view(-1,9),  # 9
                self._chaser.data.root_ang_vel_b,  # 3
            ], dim=-1)[0].cpu()
            self._student_buf["last_action"] = self._actions_c[0].cpu()
            self._student_buf["depth"] = self._camera.data.output["depth"][0,:,:,0].cpu()
            self._student_buf["rgb"] = self._camera.data.output["rgb"][0,:,:,0:3].cpu()

        return observations
    
    # override
    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""

        # action space
        self.num_actions = self.cfg.num_actions
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        
        # observation space
        self.single_observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Dict({
                "history_distances": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,16)),
                "states": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,)),
            })
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


def stack_tensors(dict_list):
    # 将列表中所有字典的值堆在一起
    result = {}

    # 遍历列表中的每个字典
    for d in dict_list:
        for key, tensor in d.items():
            if key not in result:
                result[key] = []
            # 将张量添加到相应的列表中
            result[key].append(tensor)
    
    # 遍历结果字典，将每个列表中的张量堆叠成一个张量
    for key in result:
        result[key] = np.stack(result[key], axis=0)

    return result