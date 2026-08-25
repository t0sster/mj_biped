from __future__ import annotations

import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp import dr, events as event_fns
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor import BuiltinSensorCfg, RayCastSensorCfg, GridPatternCfg, ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from mj_biped.tasks.tinker import mdp

_TINKER_XML = (
  Path(__file__).resolve().parents[2] / "assets" / "tinker" / "xml" / "world.xml"
)

_JOINT_NAMES = (
  "joint_l_yaw",
  "joint_l_roll",
  "joint_l_pitch",
  "joint_l_knee",
  "joint_l_ankle",
  "joint_r_yaw",
  "joint_r_roll",
  "joint_r_pitch",
  "joint_r_knee",
  "joint_r_ankle",
)

_ROBOT_CFG = SceneEntityCfg("tinker", joint_names=_JOINT_NAMES)
_ROBOT_ACTUATOR_CFG = SceneEntityCfg("tinker", actuator_names=_JOINT_NAMES)
_FOOT_SITE_CFG = SceneEntityCfg(
  "tinker",
  site_names=("left_foot", "right_foot"),
  preserve_order=True,
)
_IMU_SITE_CFG = SceneEntityCfg("tinker", site_names=("imu",))

_DECIMATION = 10
_PLAY_NUM_ENVS = 1
_FEET_CONTACT_SENSOR = "feet_ground_contact"
_ILLEGAL_CONTACT_SENSOR = "illegal_ground_contact"
_IMU_GYRO_SENSOR = "tinker/imu_gyro"
_IMU_ACCEL_SENSOR = "tinker/imu_accel"

_HEIGH_RAYCAST_SENSOR = "ray_cast_sensor"


_YAW_JOINT_NAMES_EXPR = (".*_yaw",)
_ROLL_JOINT_NAMES_EXPR = (".*_roll",)
_PITCH_JOINT_NAMES_EXPR = (".*_pitch",)
_KNEE_JOINT_NAMES_EXPR = (".*_knee",)
_ANKLE_JOINT_NAMES_EXPR = (".*_ankle",)

_TINKER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    # DM-6006
    BuiltinPositionActuatorCfg(
      target_names_expr=_YAW_JOINT_NAMES_EXPR,
      stiffness=15.0,
      damping=0.65,
      effort_limit=6.0,
      armature=0.024776,
      frictionloss=0.166980,
      viscous_damping=0.250651,
      delay_min_lag=0,
      delay_max_lag=30,
    ),
    # DM-8006
    BuiltinPositionActuatorCfg(
      target_names_expr=_ROLL_JOINT_NAMES_EXPR,
      stiffness=15.0,
      damping=0.65,
      effort_limit=10.0,
      armature=0.032850,
      frictionloss=0.259239,
      viscous_damping=0.001136,
      delay_min_lag=0,
      delay_max_lag=30,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=_PITCH_JOINT_NAMES_EXPR,
      stiffness=15.0,
      damping=0.65,
      effort_limit=10.0,
      armature=0.029625,
      frictionloss=0.048435,
      viscous_damping=0.267501,
      delay_min_lag=0,
      delay_max_lag=30,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=_KNEE_JOINT_NAMES_EXPR,
      stiffness=15.0,
      damping=0.65,
      effort_limit=10.0,
      armature=0.013130,
      frictionloss=0.0,
      viscous_damping=0.377046,
      delay_min_lag=0,
      delay_max_lag=30,
    ),
    # DM-6006
    BuiltinPositionActuatorCfg(
      target_names_expr=_ANKLE_JOINT_NAMES_EXPR,
      stiffness=15.0,
      damping=0.65,
      effort_limit=6.0,
      armature=0.013179,
      frictionloss=1.520994,
      viscous_damping=0.0,
      delay_min_lag=0,
      delay_max_lag=30,
    ),
  ),
  soft_joint_pos_limit_factor=0.95,
)


def _get_tinker_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_TINKER_XML))


_TINKER_INITIAL_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.325),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "joint_l_yaw": 0.0,
    "joint_l_roll": 0.0,
    "joint_l_pitch": 0.45,
    "joint_l_knee": 0.9,
    "joint_l_ankle": 0.45,
    "joint_r_yaw": 0.0,
    "joint_r_roll": 0.0,
    "joint_r_pitch": -0.45,
    "joint_r_knee": -0.9,
    "joint_r_ankle": -0.45,
  },
  joint_vel={".*": 0.0},
)
# _TINKER_INITIAL_STATE = EntityCfg.InitialStateCfg(
#   pos=(0.0, 0.0, 0.325),
#   rot=(1.0, 0.0, 0.0, 0.0),
#   joint_pos={
#     "joint_l_yaw": 0.0,
#     "joint_l_roll": -0.0,
#     "joint_l_pitch": 0.0,
#     "joint_l_knee": 0.0,
#     "joint_l_ankle": 0.0,
#     "joint_r_yaw": 0.0,
#     "joint_r_roll": 0.55,
#     "joint_r_pitch": -0.0,
#     "joint_r_knee": -0.23,
#     "joint_r_ankle": -0.0,
#   },
#   joint_vel={".*": 0.0},
# )

def _get_tinker_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_tinker_spec,
    articulation=_TINKER_ARTICULATION,
    init_state=_TINKER_INITIAL_STATE,
  )


def _make_env_cfg(num_envs: int) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "imu_rpy": ObservationTermCfg(
      func=mdp.imu_rpy,
      params={"asset_cfg": _IMU_SITE_CFG},
      noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
    ),
    "imu_gyro": ObservationTermCfg(
      func=mdp.builtin_sensor_data,
      params={"sensor_name": _IMU_GYRO_SENSOR},
      noise=UniformNoiseCfg(n_min=-0.15, n_max=0.15),
    ),
    "imu_accel": ObservationTermCfg(
      func=mdp.builtin_sensor_data,
      params={"sensor_name": _IMU_ACCEL_SENSOR},
      noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3),
    ),
    "velocity_command": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": "velocity"},
    ),
    "gait_phase": ObservationTermCfg(
      func=mdp.gait_phase_observation,
      params={"command_name": "gait"},
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_CFG},
      noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_CFG},
      noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
    ),
    "last_action": ObservationTermCfg(func=env_mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(
      {
        **actor_terms,
        "foot_air_time": ObservationTermCfg(
          func=velocity_mdp.foot_air_time,
          params={"sensor_name": _FEET_CONTACT_SENSOR},
        ),
        "foot_contact_forces": ObservationTermCfg(
          func=velocity_mdp.foot_contact_forces,
          params={"sensor_name": _FEET_CONTACT_SENSOR},
        ),
        "base_height": ObservationTermCfg(
          func=mdp.current_base_height,
          params={"asset_cfg": _ROBOT_CFG},
        ),
        # Privileged: real base velocity isn't directly measurable on hardware
        # (no direct sensor for it — actor gets imu_accel/imu_gyro instead).
        "base_lin_vel": ObservationTermCfg(
          func=env_mdp.base_lin_vel,
          params={"asset_cfg": _ROBOT_CFG},
        ),
      },
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="tinker",
      actuator_names=_JOINT_NAMES,
      scale={
        ".*_yaw": 0.25,
        ".*_roll": 0.15,
        ".*_pitch": 0.4,
        ".*_knee": 0.35,
        ".*_ankle": 0.25,
      },
      use_default_offset=True,
      preserve_order=True,
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "velocity": velocity_mdp.UniformVelocityCommandCfg(
      entity_name="tinker",
      resampling_time_range=(4.0, 8.0),
      rel_standing_envs=0.1,
      rel_forward_envs=0.3,
      heading_command=False,
      ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.30, 0.30),
        lin_vel_y=(-0.2, 0.2),
        ang_vel_z=(-0.3, 0.3),
      ),
    ),
    "gait": mdp.UniformGaitCommandCfg(
      resampling_time_range=(1.0e6, 1.0e6),
      ranges=mdp.UniformGaitCommandCfg.Ranges(
        frequencies=(0.2, 1.5),
        duty_cycle=(0.25, 1.2),
      ),
    ),
  }

  rewards = {
    "base_height_track": RewardTermCfg(
      func=mdp.base_height,
      weight=1.0,
      params={
        "asset_cfg": _ROBOT_CFG,
        "target_height": 0.21,
        "std": math.sqrt(0.001),
      }
    ),
    "step_width": RewardTermCfg(
        func=mdp.step_width,
        weight=0.5,
        params={
          "asset_cfg": _FOOT_SITE_CFG,
          "target_width": 0.19,
          "std": math.sqrt(0.001),
        },
    ),

    "track_linear_velocity": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=2.75,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std": math.sqrt(0.1),
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=1.0,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std": math.sqrt(0.2),
      },
    ),
    "heading_travel_alignment": RewardTermCfg(
      func=mdp.heading_travel_alignment,
      weight=1.5,
      params={
        "asset_cfg": _ROBOT_CFG,
        "std": math.sqrt(0.1),
        "min_speed": 0.1,
      },
    ),
    "posture": RewardTermCfg(
      func=velocity_mdp.variable_posture,
      weight=0.2,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std_standing": {".*": math.sqrt(0.08)},
        "std_walking": {
          ".*_yaw": math.sqrt(0.05),
          ".*_roll": math.sqrt(0.05),
          ".*_pitch": math.sqrt(0.05),
          ".*_knee": math.sqrt(0.4),
          ".*_ankle": math.sqrt(0.25),
        },
        "std_running": {
          ".*_yaw": math.sqrt(0.08),
          ".*_roll": math.sqrt(0.1),
          ".*_pitch": math.sqrt(0.05),
          ".*_knee": math.sqrt(0.5),
          ".*_ankle": math.sqrt(0.35),
        },
        "walking_threshold": 0.05,
        "running_threshold": 0.4,
      },
    ),
    "swing_foot_force": RewardTermCfg(
      func=mdp.swing_foot_force_l2,
      weight=-0.0,
      params={
        "command_name": "gait",
        "motion_command_name": "velocity",
        "sensor_name": _FEET_CONTACT_SENSOR,
        "force_scale": 50.0,
        "command_threshold": 0.05,
      },
    ),
    "stance_foot_velocity": RewardTermCfg(
      func=mdp.stance_foot_velocity_l2,
      weight=-0.0,
      params={
        "asset_cfg": _FOOT_SITE_CFG,
        "command_name": "gait",
        "motion_command_name": "velocity",
        "command_threshold": 0.05,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.biped_air_time,
      weight=1.75,
      params={
        "sensor_name": _FEET_CONTACT_SENSOR,
        "command_name": "velocity",
        "max_reward_time": 0.25,
        "command_threshold": 0.05,
      },
    ),
    "soft_landing": RewardTermCfg(
      func=velocity_mdp.soft_landing,
      weight=-5.0e-3,
      params={
        "sensor_name": _FEET_CONTACT_SENSOR,
        "command_name": "velocity",
        "command_threshold": 0.05,
      },
    ),
    "excessive_foot_force": RewardTermCfg(
      func=mdp.excessive_foot_force,
      weight=-2.0e-3,
      params={
        "sensor_name": _FEET_CONTACT_SENSOR,
        "max_force": 80.0,
      },
    ),
    "base_vertical_velocity": RewardTermCfg(
      func=mdp.base_vertical_velocity_l2,
      weight=-0.0,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_torques": RewardTermCfg(
      func=env_mdp.joint_torques_l2,
      weight=-1.0e-4,
      params={"asset_cfg": _ROBOT_ACTUATOR_CFG},
    ),
    "joint_velocity": RewardTermCfg(
      func=env_mdp.joint_vel_l2,
      weight=-1.0e-4,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "flat_orientation": RewardTermCfg(
      func=env_mdp.flat_orientation_l2,
      weight=-0.8,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_limits": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-0.05,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "action_rate": RewardTermCfg(func=env_mdp.action_rate_l2, weight=-5.0e-3),
    "action_acceleration": RewardTermCfg(
      func=env_mdp.action_acc_l2,
      weight=-1.0e-3,
    ),


  }

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "root_height": TerminationTermCfg(
      func=env_mdp.root_height_below_minimum,
      params={"minimum_height": 0.2, "asset_cfg": _ROBOT_CFG},
    ),
    "bad_orientation": TerminationTermCfg(
      func=env_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0), "asset_cfg": _ROBOT_CFG},
    ),
    "illegal_ground_contact": TerminationTermCfg(
      func=velocity_mdp.illegal_contact,
      params={
        "sensor_name": _ILLEGAL_CONTACT_SENSOR,
        "force_threshold": 5.0,
      },
    ),
  }

  events = {
      # Reset all entities to their default state each episode.
      "reset_scene": EventTermCfg(
          func=event_fns.reset_scene_to_default,
          mode="reset",
      ),
      # Randomize foot friction once at startup.
      "foot_friction": EventTermCfg(
          func=dr.geom_friction,
          mode="startup",
          params={
              "asset_cfg": SceneEntityCfg("tinker", geom_names=["left_foot_collision", "right_foot_collision"]),
              "ranges": (0.8, 1.0),
              "operation": "abs",
          },
      ),
      # Push the robot at random intervals during the episode.
      "push_robot": EventTermCfg(
          func=event_fns.push_by_setting_velocity,
          mode="interval",
          interval_range_s=(1.0, 5.0),
          params={
              "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
              "asset_cfg": _ROBOT_CFG,
          },
      ),
      # Transient random impulses with duration and cooldown.
      "impulse": EventTermCfg(
          func=event_fns.apply_body_impulse,
          mode="step",
          params={
              "force_range": (-15.0, 15.0),
              "torque_range": (0.0, 0.0),
              "duration_s": (0.05, 0.1),
              "cooldown_s": (1.0, 5.0),
              "asset_cfg": SceneEntityCfg("tinker", body_names=("base_link")),
          },
      ),
      "body_mass": EventTermCfg(
        func=dr.pseudo_inertia,
        mode="reset",
        params={
          # Excludes "base_link": it's the massless freejoint wrapper (all
          # real mass/inertia live on "torso", see tinker_range.xml) -- its
          # pseudo-inertia matrix is identically zero, and Cholesky-decomposing
          # a zero matrix produces NaN, which then corrupts the whole sim.
          "asset_cfg": SceneEntityCfg("tinker", body_names="torso|link_.*"),
          "alpha_range": (0.5 * math.log(0.9), 0.5 * math.log(1.1)),
        }
      ),
      "encoder_bias": EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={
          "asset_cfg": SceneEntityCfg("tinker"),
          "bias_range": (-0.015, 0.015),
        },
      ),
      "base_com": EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
          "asset_cfg": SceneEntityCfg("tinker", body_names=("base_link")),  # Set per-robot.
          "operation": "add",
          "ranges": {
            0: (-0.025, 0.025),
            1: (-0.025, 0.025),
            2: (-0.03, 0.03),
          },
        },
      ),
      "armature": EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
          "asset_cfg": SceneEntityCfg("tinker", joint_names=(".*")),
          "operation": "scale",
          "ranges": (0.95, 1.05),
        },
      ),
      "frictioloss": EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
          "asset_cfg": SceneEntityCfg("tinker", joint_names=(".*")),
          "operation": "scale",
          "ranges": (0.95, 1.05),
        },
      ),
      "damping": EventTermCfg(
        mode="startup",
        func=dr.joint_damping,
        params={
          "asset_cfg": SceneEntityCfg("tinker", joint_names=(".*")),
          "operation": "scale",
          "ranges": (0.95, 1.05),
        },
      ),
      "effort_limits": EventTermCfg(
        mode="startup",
        func=dr.effort_limits,
        params={
          "asset_cfg": _ROBOT_ACTUATOR_CFG,
          "operation": "scale",
          "effort_limit_range": (0.8, 1.2),
        },
      ),
  }

  raycast_cfg = RayCastSensorCfg(
      name=_HEIGH_RAYCAST_SENSOR,
      frame=ObjRef(type="body", name="base_link", entity="tinker"),
      pattern=GridPatternCfg(
          size=(0.2, 0.2),
          resolution=0.1,
          direction=(0.0, 0.0, -1.0),
      ),
      ray_alignment="yaw",
      max_distance=2.0,
  )

  imu_gyro_sensor = BuiltinSensorCfg(
    name="imu_gyro",
    sensor_type="gyro",
    obj=ObjRef(type="site", name="imu", entity="tinker"),
  )
  imu_accel_sensor = BuiltinSensorCfg(
    name="imu_accel",
    sensor_type="accelerometer",
    obj=ObjRef(type="site", name="imu", entity="tinker"),
  )

  feet_contact_sensor = ContactSensorCfg(
    name=_FEET_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern=("link_l_ankle", "link_r_ankle"),
      entity="tinker",
    ),
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=True,
    history_length=_DECIMATION,
  )
  illegal_contact_sensor = ContactSensorCfg(
    name=_ILLEGAL_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern=(
        "torso",
        "link_l_pitch",
        "link_l_knee",
        "link_r_pitch",
        "link_r_knee",
      ),
      entity="tinker",
    ),
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    history_length=_DECIMATION,
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"tinker": _get_tinker_cfg()},
      sensors=(
        feet_contact_sensor,
        illegal_contact_sensor,
        raycast_cfg,
        imu_gyro_sensor,
        imu_accel_sensor,
      ),
      num_envs=num_envs,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    metrics={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="tinker",
      body_name="base_link",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=48,
      njmax=128,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=_DECIMATION,
    episode_length_s=20.0,
  )


def tinker_env_cfg(
  play: bool = False,
  num_envs: int = 1024,
  play_num_envs: int = _PLAY_NUM_ENVS,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(play_num_envs if play else num_envs)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg
