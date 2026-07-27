from __future__ import annotations

import math
from pathlib import Path

import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from optim.tasks.bd_optim import mdp

_BD_XML: Path = Path(__file__).resolve().parents[2] / "assets" / "bd" / "bd_prev_gear.xml"

_BD_JOINT_NAMES = (
  "JL0_hip_pitch",
  "JL1_hip_roll",
  "JL2_thigh_yaw",
  "JL3_knee_pitch",
  "JL4_ankle_pitch",
  "JR0_hip_pitch",
  "JR1_hip_roll",
  "JR2_thigh_yaw",
  "JR3_knee_pitch",
  "JR4_ankle_pitch",
)

_ACTUATED_JOINT_NAMES = (
  "JL0_hip_pitch",
  "JL1_hip_roll",
  "JL2_thigh_yaw",
  "JL3_knee_pitch",
  "JL4_ankle_pitch",
  "JR0_hip_pitch",
  "JR1_hip_roll",
  "JR2_thigh_yaw",
  "JR3_knee_pitch",
  "JR4_ankle_pitch",
)

_XML_ACTUATOR_NAMES = (
  "ML0_hip_pitch",
  "ML1_hip_roll",
  "ML2_thigh_yaw",
  "ML3_knee_pitch",
  "ML4_ankle_pitch",
  "MR0_hip_pitch",
  "MR1_hip_roll",
  "MR2_thigh_yaw",
  "MR3_knee_pitch",
  "MR4_ankle_pitch",
)

_ROBOT_CFG = SceneEntityCfg("bd", joint_names=_BD_JOINT_NAMES)
_BASE_CONTACT_CFG = SceneEntityCfg("bd", body_names=("pelvis", "head"))
_FEET_CONTACT_CFG = SceneEntityCfg(
  "bd",
  body_names=("left_foot", "right_foot"),
  preserve_order=True,
)
_TERRAIN_CFG = SceneEntityCfg("terrain", geom_names=("terrain",))

_ROBOT_ACTUATOR_CFG = SceneEntityCfg("bd", actuator_names=_XML_ACTUATOR_NAMES)

_PLAY_NUM_ENVS = 1
_DECIMATION = 4
_FEET_CONTACT_SENSOR = "feet_ground_contact"


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_BD_XML))


_BD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(target_names_expr=_ACTUATED_JOINT_NAMES),
  ),
  soft_joint_pos_limit_factor=0.9,
)

_BD_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.55),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "JL0_hip_pitch": 0.5,
    "JL1_hip_roll": 0.0,
    "JL2_thigh_yaw": 0.0,
    "JL3_knee_pitch": 1.1,
    "JL4_ankle_pitch": -0.6,

    "JR0_hip_pitch": -0.5,
    "JR1_hip_roll": 0.0,
    "JR2_thigh_yaw": 0.0,
    "JR3_knee_pitch": -1.1,
    "JR4_ankle_pitch": 0.6,
  },
  joint_vel={".*": 0.0},
)


def _get_bd_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_BD_ARTICULATION,
    init_state=_BD_INIT,
  )


def _make_env_cfg(num_envs: int = 1024) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=env_mdp.base_lin_vel,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "velocity_command": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": "velocity"},
    ),
    "gait": ObservationTermCfg(
      func=mdp.gait_sin_cos,
      params={"command_name": "gait"},
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_CFG},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({
      **actor_terms,
      "foot_air_time": ObservationTermCfg(
        func=velocity_mdp.foot_air_time,
        params={"sensor_name": _FEET_CONTACT_SENSOR},
      ),
      "foot_contact_forces": ObservationTermCfg(
        func=velocity_mdp.foot_contact_forces,
        params={"sensor_name": _FEET_CONTACT_SENSOR},
      ),
    }),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.JointPositionToMotorEffortActionCfg(
      entity_name="bd",
      actuator_names=_ACTUATED_JOINT_NAMES,
      scale={
        ".*hip_pitch": 0.5,
        ".*knee_pitch": 0.5,
        ".*ankle_pitch": 0.35,
        ".*hip_roll": 0.4,
        ".*thigh_yaw": 0.4,
      },
      use_default_offset=True,
      preserve_order=True,
      stiffness={
        "JL0_hip_pitch": 45.0,
        "JL1_hip_roll": 45.0,
        "JL2_thigh_yaw": 40.0,
        "JL3_knee_pitch": 45.0,
        "JL4_ankle_pitch": 40.0,

        "JR0_hip_pitch": 45.0,
        "JR1_hip_roll": 45.0,
        "JR2_thigh_yaw": 40.0,
        "JR3_knee_pitch": 45.0,
        "JR4_ankle_pitch": 40.0,
      },
      damping={
        "JL0_hip_pitch": 4.5,
        "JL1_hip_roll": 4.5,
        "JL2_thigh_yaw": 4.0,
        "JL3_knee_pitch": 4.5,
        "JL4_ankle_pitch": 4.0,

        "JR0_hip_pitch": 4.5,
        "JR1_hip_roll": 4.5,
        "JR2_thigh_yaw": 4.0,
        "JR3_knee_pitch": 4.5,
        "JR4_ankle_pitch": 4.0,
      },
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "velocity": velocity_mdp.UniformVelocityCommandCfg(
      entity_name="bd",
      resampling_time_range=(5.0, 10.0),
      rel_standing_envs=0.05,
      rel_forward_envs=0.5,
      heading_command=False,
      ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.2, 0.4),
        lin_vel_y=(-0.2, 0.2),
        ang_vel_z=(-0.5, 0.5),
      ),
    ),
    "gait": mdp.UniformGaitCommandCfg(
      resampling_time_range=(1.0e6, 1.0e6),
      ranges=mdp.UniformGaitCommandCfg.Ranges(
        frequencies=(0.75, 1.25),
        duty_cycle=(0.60, 0.60),
      ),
    ),
  }

  rewards = {
    "track_lin_vel_xy": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=1.5,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std": math.sqrt(0.15),
      },
    ),
    "track_ang_vel_z": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=1.0,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std": math.sqrt(0.20),
      },
    ),
    "posture": RewardTermCfg(
      func=velocity_mdp.variable_posture,
      weight=0.2,
      params={
        "asset_cfg": _ROBOT_CFG,
        "command_name": "velocity",
        "std_standing": {
          ".*": math.sqrt(0.15),
        },
        "std_walking": {
          ".*hip_pitch": math.sqrt(0.5),
          ".*knee_pitch": math.sqrt(0.5),
          ".*ankle_pitch": math.sqrt(0.35),
          ".*hip_roll": math.sqrt(0.10),
          ".*thigh_yaw": math.sqrt(0.10),
        },
        "std_running": {
          ".*hip_pitch": math.sqrt(0.5),
          ".*knee_pitch": math.sqrt(0.5),
          ".*ankle_pitch": math.sqrt(0.35),
          ".*hip_roll": math.sqrt(0.20),
          ".*thigh_yaw": math.sqrt(0.20),
        },
        "walking_threshold": 0.02,
        "running_threshold": 0.2,
      },
    ),
    "gait_swing_foot_force": RewardTermCfg(
      func=mdp.gait_swing_foot_force,
      weight=-0.5,
      params={
        "command_name": "gait",
        "sensor_name": _FEET_CONTACT_SENSOR,
        "force_scale": 50.0,
      },
    ),
    "gait_stance_foot_velocity": RewardTermCfg(
      func=mdp.gait_stance_foot_velocity,
      weight=-0.5,
      params={
        "asset_cfg": _FEET_CONTACT_CFG,
        "command_name": "gait",
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time_positive_biped,
      weight=0.5,
      params={
        "sensor_name": _FEET_CONTACT_SENSOR,
        "command_name": "velocity",
        "threshold": 0.25,
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
    "foot_contact_force": RewardTermCfg(
      func=mdp.feet_contact_forces,
      weight=-2.0e-3,
      params={
        "sensor_name": _FEET_CONTACT_SENSOR,
        "max_force": 80.0,
      },
    ),
    "base_lin_vel_z": RewardTermCfg(
      func=mdp.base_lin_vel_z_l2,
      weight=-0.05,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "torques": RewardTermCfg(
      func=mdp.torques,
      weight=-1.0e-4,
      params={"action_name": "joint_pos"},
    ),
    "torque_limits": RewardTermCfg(
      func=mdp.torque_limits,
      weight=-0.1,
      params={"action_name": "joint_pos", "soft_limit": 0.9},
    ),
    "joint_velocity": RewardTermCfg(
      func=env_mdp.joint_vel_l2,
      weight=-1.0e-4,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "flat_orientation": RewardTermCfg(
      func=env_mdp.flat_orientation_l2,
      weight=-0.1,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "ang_vel_xy": RewardTermCfg(
      func=mdp.ang_vel_xy,
      weight=-0.05,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_limits": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-0.05,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "action_rate": RewardTermCfg(func=env_mdp.action_rate_l2, weight=-5.0e-3),
    "action_acc": RewardTermCfg(func=env_mdp.action_acc_l2, weight=-1.0e-3),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "root_height": TerminationTermCfg(
      func=env_mdp.root_height_below_minimum,
      params={"minimum_height": 0.2, "asset_cfg": _ROBOT_CFG},
    ),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(110.0), "asset_cfg": _ROBOT_CFG},
    ),
    "base_contact_with_ground": TerminationTermCfg(
      func=mdp.base_contact_with_ground,
      params={"asset_cfg": _BASE_CONTACT_CFG, "terrain_cfg": _TERRAIN_CFG},
    ),
  }

  metrics = {
    "max_commanded_forward_speed": MetricsTermCfg(
      func=mdp.max_commanded_forward_speed,
      params={"asset_cfg": _ROBOT_CFG},
      reduce="last",
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"bd": _get_bd_cfg()},
      sensors=(
        ContactSensorCfg(
          name=_FEET_CONTACT_SENSOR,
          primary=ContactMatch(
            mode="body",
            pattern=("left_foot", "right_foot"),
            entity="bd",
          ),
          secondary=ContactMatch(mode="geom", pattern="terrain"),
          fields=("found", "force"),
          reduce="netforce",
          track_air_time=True,
          history_length=_DECIMATION,
        ),
      ),
      num_envs=num_envs,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="bd",
      body_name="pelvis",
      distance=5.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    decimation=_DECIMATION,
    episode_length_s=20.0,
  )


def bd_optim_env_cfg(
  play: bool = False,
  num_envs: int = 1024,
  play_num_envs: int = _PLAY_NUM_ENVS,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(play_num_envs if play else num_envs)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg
