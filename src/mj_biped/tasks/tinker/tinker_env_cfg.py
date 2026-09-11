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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
  Path(__file__).resolve().parents[2] / "assets" / "tinker" / "mjcf" / "world.xml"
)

# In the new model each hinge sits on its motor body (actuator_*), and the
# output link is rigidly attached to the motor -- there are no passive joints.
# These ten hinges are all of the robot's DOFs, in motor order (yaw, roll,
# pitch, knee, ankle per leg): the same names and order as the old model.
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


# Per-joint system-identification results (armature, viscous damping,
# frictionloss, and fixed encoder bias). Replaces the old per-type-regex
# grouping now that left/right differ per joint.
_JOINT_SYSID_PARAMS: dict[str, dict[str, float]] = {
  "joint_l_yaw":   dict(armature=0.00010, viscous_damping=0.00015, frictionloss=0.06376, bias=0.0034),
  "joint_l_roll":  dict(armature=0.02428, viscous_damping=0.00055, frictionloss=0.20956, bias=-0.0900),
  "joint_l_pitch": dict(armature=0.01508, viscous_damping=0.20577, frictionloss=0.20281, bias=0.0194),
  "joint_l_knee":  dict(armature=0.00716, viscous_damping=0.12399, frictionloss=0.10800, bias=-0.0050),
  "joint_l_ankle": dict(armature=0.00010, viscous_damping=0.02544, frictionloss=0.05319, bias=0.0089),
  # Right leg not yet independently measured -- mirrored from left (same
  # values, not negated).
  "joint_r_yaw":            dict(armature=0.00010, viscous_damping=0.00015, frictionloss=0.06376, bias=0.0034),
  "joint_r_roll":  dict(armature=0.02428, viscous_damping=0.00055, frictionloss=0.20956, bias=-0.0900),
  "joint_r_pitch":   dict(armature=0.01508, viscous_damping=0.20577, frictionloss=0.20281, bias=0.0194),
  "joint_r_knee":  dict(armature=0.00716, viscous_damping=0.12399, frictionloss=0.10800, bias=-0.0050),
  "joint_r_ankle": dict(armature=0.00010, viscous_damping=0.02544, frictionloss=0.05319, bias=0.0089),
}

# effort_limit is still per joint *type* (not individually measured above).
_JOINT_TYPE_EFFORT_LIMIT = {
  "yaw": 4.0,
  "roll": 8.0,
  "pitch": 8.0,
  "knee": 8.0,
  "ankle": 4.0,
}

# Measured command delay is shared across all joints, converted to whole
# physics steps (35.95 ms / 2 ms per step = 17.975 -> rounds to 18 steps).
_MEASURED_DELAY_MS = 35.95
_PHYSICS_TIMESTEP_MS = 2.0  # must match MujocoCfg(timestep=0.002) below
_DELAY_LAG_STEPS = round(_MEASURED_DELAY_MS / _PHYSICS_TIMESTEP_MS)

_TINKER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=10.0,
      damping=0.5,
      effort_limit=_JOINT_TYPE_EFFORT_LIMIT[joint_name.rsplit("_", 1)[-1]],
      armature=params["armature"],
      frictionloss=params["frictionloss"],
      viscous_damping=params["viscous_damping"],
      delay_min_lag=_DELAY_LAG_STEPS,
      delay_max_lag=_DELAY_LAG_STEPS,
    )
    for joint_name, params in _JOINT_SYSID_PARAMS.items()
  ),
  soft_joint_pos_limit_factor=0.99,
)


def _get_tinker_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_TINKER_XML))


_TINKER_INITIAL_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.325),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "joint_l_yaw": 0.0,
    "joint_l_roll": 0.0,
    "joint_l_pitch": 0.85,
    "joint_l_knee": 1.5,
    "joint_l_ankle": 0.85,
    "joint_r_yaw": 0.0,
    "joint_r_roll": 0.0,
    "joint_r_pitch": -0.85,
    "joint_r_knee": -1.5,
    "joint_r_ankle": -0.85,
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
      noise=UniformNoiseCfg(n_min=(-0.1, -0.1, -0.25), n_max=(0.1, 0.1, 0.25)),
    ),
    "imu_gyro": ObservationTermCfg(
      func=mdp.builtin_sensor_data,
      params={"sensor_name": _IMU_GYRO_SENSOR},
      noise=UniformNoiseCfg(n_min=-0.15, n_max=0.15),
    ),
    "imu_accel": ObservationTermCfg(
      func=mdp.builtin_sensor_data,
      params={"sensor_name": _IMU_ACCEL_SENSOR},
      noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
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
      params={"asset_cfg": _ROBOT_CFG, "biased": True},
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
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
      rel_forward_envs=0.2,
      heading_command=False,
      ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.15, 0.15),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(-0.2, 0.2),
        # heading=(-math.pi, math.pi),
      ),
    ),
    "gait": mdp.UniformGaitCommandCfg(
      resampling_time_range=(1.0e6, 1.0e6),
      ranges=mdp.UniformGaitCommandCfg.Ranges(
        frequencies=(1.0, 1.0),
        duty_cycle=(0.5, 0.5),
      ),
    ),
  }

  rewards = {
    "base_height_track": RewardTermCfg(
      func=mdp.base_height,
      weight=0.5,
      params={
        "asset_cfg": _ROBOT_CFG,
        "target_height": 0.22,
        "std": math.sqrt(0.05),
      }
    ),
    "step_width": RewardTermCfg(
        func=mdp.step_width,
        weight=0.5,
        params={
          "asset_cfg": _FOOT_SITE_CFG,
          "target_width": 0.2,
          "std": math.sqrt(0.2),
        },
    ),

    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity_world,
      weight=2.0,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std": math.sqrt(0.01),
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity_world,
      weight=2.5,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std": math.sqrt(0.01),
      },
    ),
    # "heading_travel_alignment": RewardTermCfg(
    #   func=mdp.heading_travel_alignment,
    #   weight=0.5,
    #   params={
    #     "asset_cfg": _ROBOT_CFG,
    #     "std": math.sqrt(0.1),
    #     "min_speed": 0.1,
    #   },
    # ),
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
    "gait_contact_match": RewardTermCfg(
      func=mdp.gait_contact_match,
      weight=1.0,
      params={
        "command_name": "gait",
        "motion_command_name": "velocity",
        "sensor_name": _FEET_CONTACT_SENSOR,
        "command_threshold": 0.025,
      },
    ),
    "swing_foot_force": RewardTermCfg(
      func=mdp.swing_foot_force_l2,
      weight=-0.75,
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
      weight=-0.75,
      params={
        "asset_cfg": _FOOT_SITE_CFG,
        "command_name": "gait",
        "motion_command_name": "velocity",
        "command_threshold": 0.05,
      },
    ),
    # "feet_slip": RewardTermCfg(
    #   func=velocity_mdp.feet_slip,
    #   weight=-0.0,
    #   params={
    #     "asset_cfg": _FOOT_SITE_CFG,
    #     "sensor_name": _FEET_CONTACT_SENSOR,
    #     "command_name": "velocity",
    #     "command_threshold": 0.05,
    #   },
    # ),
    "feet_air_time": RewardTermCfg(
      func=mdp.biped_air_time,
      weight=1.5,
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
    # "base_vertical_velocity": RewardTermCfg(
    #   func=mdp.base_vertical_velocity_l2,
    #   weight=-0.0,
    #   params={"asset_cfg": _ROBOT_CFG},
    # ),
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
      "reset_base_pose": EventTermCfg(
          func=event_fns.reset_root_state_uniform,
          mode="reset",
          params={
              "asset_cfg": SceneEntityCfg("tinker"),
              "pose_range": {
                  "roll": (-0.25, 0.25),
                  "pitch": (-0.25, 0.25),
                  "yaw": (-math.pi, math.pi),
              },
          },
      ),
      # Randomize initial joint pose around the default crouch, so every
      # episode doesn't start from the exact same pose.
      "reset_joints": EventTermCfg(
          func=event_fns.reset_joints_by_offset,
          mode="reset",
          params={
              "asset_cfg": _ROBOT_CFG,
              "position_range": (-0.1, 0.1),
              "velocity_range": (0.0, 0.0),
          },
      ),
      # Randomize foot friction every episode reset. mode="reset" (not
      # "startup") so the curriculum below can widen "ranges" over training
      # and have it actually take effect on later episodes.
      "foot_friction": EventTermCfg(
          func=dr.geom_friction,
          mode="reset",
          params={
              "asset_cfg": SceneEntityCfg("tinker", geom_names=["left_foot_collision", "right_foot_collision"]),
              "ranges": (0.92, 1.0),
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
              "force_range": (-10.0, 10.0),
              "torque_range": (0.0, 0.0),
              "duration_s": (0.05, 0.1),
              "cooldown_s": (1.0, 5.0),
              "asset_cfg": SceneEntityCfg("tinker", body_names=("base_link")),
          },
      ),
      # Mass/inertia/CoM randomization restricted to the base only -- legs
      # are no longer randomized now that per-joint sysid params are fixed.
      # t1/t2/t3_range (CoM offset) start narrow and widen via the
      # "com_offset_curriculum" curriculum term below; alpha_range
      # (mass scale) is left fixed at full difficulty from the start.
      "body_mass": EventTermCfg(
        func=dr.pseudo_inertia,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("tinker", body_names="torso"),
          "alpha_range": (0.5 * math.log(0.8), 0.5 * math.log(1.1)),
          "t1_range": (-0.01, 0.01),
          "t2_range": (-0.01, 0.01),
          "t3_range": (-0.01, 0.01),
        }
      ),
      # Fixed (not randomized) per-joint encoder bias from measured
      # calibration -- a degenerate (v, v) range makes dr.encoder_bias
      # sample the same constant every time.
      **{
        f"encoder_bias_{joint_name}": EventTermCfg(
          mode="startup",
          func=dr.encoder_bias,
          params={
            "asset_cfg": SceneEntityCfg("tinker", joint_names=(joint_name,)),
            "bias_range": (params["bias"], params["bias"]),
          },
        )
        for joint_name, params in _JOINT_SYSID_PARAMS.items()
      },
      "effort_limits": EventTermCfg(
        mode="startup",
        func=dr.effort_limits,
        params={
          "asset_cfg": _ROBOT_ACTUATOR_CFG,
          "operation": "scale",
          "effort_limit_range": (0.95, 1.05),
        },
      ),
  }

  # Widen two DR ranges over the course of training instead of randomizing
  # at full difficulty from step 0: base CoM offset (body_mass's t1/2/3_range)
  # and foot friction (foot_friction's ranges). Step thresholds are in
  # env.common_step_counter units (env.step() calls, i.e.
  # num_steps_per_env * PPO iteration -- see mdp/runner.py), spaced across
  # the ~120k steps of a max_iterations=5001, num_steps_per_env=24 run.
  curriculum = {
    "com_offset_curriculum": CurriculumTermCfg(
      func=mdp.event_curriculum,
      params={
        "event_name": "body_mass",
        "stages": [
          {"step": 0, "params": {
            "t1_range": (-0.04, 0.04),
            "t2_range": (-0.04, 0.04),
            "t3_range": (-0.04, 0.04),
          }},
          {"step": 24_000, "params": {
            "t1_range": (-0.08, 0.08),
            "t2_range": (-0.08, 0.08),
            "t3_range": (-0.08, 0.08),
          }},
          {"step": 48_000, "params": {
            "t1_range": (-0.12, 0.12),
            "t2_range": (-0.12, 0.12),
            "t3_range": (-0.12, 0.12),
          }},
        ],
      },
    ),
    "foot_friction_curriculum": CurriculumTermCfg(
      func=mdp.event_curriculum,
      params={
        "event_name": "foot_friction",
        "stages": [
          {"step": 0, "params": {"ranges": (0.8, 1.0)}},
          {"step": 24_000, "params": {"ranges": (0.6, 1.5)}},
          {"step": 48_000, "params": {"ranges": (0.3, 2.0)}},
        ],
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
        "actuator_roll",
        "actuator_hip",
        "actuator_knee",
        "actuator_ankle",
        "actuator_r_roll",
        "actuator_r_hip",
        "actuator_r_knee",
        "actuator_r_ankle",
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
    curriculum=curriculum,
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
      # New model has ~2x the collision geoms (8 actuator-housing cylinders
      # added) -- raised from 48/128 to give headroom past the ~52 contacts
      # observed at reset across the full random-pose population.
      nconmax=96,
      njmax=256,
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
