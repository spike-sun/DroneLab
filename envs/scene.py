import os
from isaaclab.sim import SimulationCfg, PinholeCameraCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sensors import RayCaster, RayCasterCfg, ContactSensor, ContactSensorCfg, Camera, CameraCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
ENABLE_CAMERAS = int(os.getenv('ENABLE_CAMERAS', default=0))

@configclass
class ForestEvaderSceneCfg(InteractiveSceneCfg):

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
            usd_path = '/home/sbw/DroneLab/assets/cf2x/cf2x_blue.usd',
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
class ForestChaserAssetCfg:
    chaser = ArticulationCfg(
        prim_path = '{ENV_REGEX_NS}/chaser',
        spawn = sim_utils.UsdFileCfg(
            usd_path = f'/home/sbw/DroneLab/assets/cf2x/cf2x_red.usd',
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
        init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
        actuators = {
            'dummy': ImplicitActuatorCfg(
                joint_names_expr = ['.*'],
                stiffness = 0.0,
                damping = 0.0,
            )
        }
    )
    lidar_c = RayCasterCfg(
        prim_path = '{ENV_REGEX_NS}/chaser/body',
        mesh_prim_paths = ['/World/ground/terrain'],
        attach_yaw_only = True,
        pattern_cfg = LidarPatternCfg(
            channels = 1,
            vertical_fov_range = (-0.0, 0.0),
            horizontal_fov_range = (-180.0, 180.0),
            horizontal_res = 22.5
        )
    )
    contact_c = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/chaser/.*')
    if ENABLE_CAMERAS:
        camera_c = CameraCfg(
            prim_path = '{ENV_REGEX_NS}/chaser/body/front_camera',
            offset = CameraCfg.OffsetCfg(convention='world'),
            spawn = PinholeCameraCfg(),
            data_types = ['depth', 'rgb'],
            width = 224,
            height = 224
        )