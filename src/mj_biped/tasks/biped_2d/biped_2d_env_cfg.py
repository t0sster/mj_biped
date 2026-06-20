from __future__ import annotations

import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mj_biped.tasks.biped_2d import mdp

_BIPED_2D_XML: Path = (
  Path(__file__).resolve().parents[2] / "assets" / "biped_2d" / "biped_2d_prims.xml"
)

_BIPED_JOINT_NAMES = (
  "root_x",
  "root_z",
  "root_pitch",
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

_ROBOT_CFG = SceneEntityCfg("biped_2d", joint_names=_BIPED_JOINT_NAMES)
_BASE_CONTACT_CFG = SceneEntityCfg(
  "biped_2d",
  body_names=("pelvis", "head"),
)
_FEET_CONTACT_CFG = SceneEntityCfg(
  "biped_2d",
  body_names=("left_ankle_pitch_link", "right_ankle_pitch_link"),
  preserve_order=True,
)
_TERRAIN_CFG = SceneEntityCfg("terrain", geom_names=("terrain",))

_TORQUE_ACTUATOR_NAMES = tuple(f"{name}_pd_.*" for name in _ACTUATED_JOINT_NAMES)

_ROBOT_ACTUATOR_CFG = SceneEntityCfg(
  "biped_2d",
  actuator_names=_TORQUE_ACTUATOR_NAMES,
)

_PLAY_NUM_ENVS = 1


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_BIPED_2D_XML))


_BIPED_2D_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPdActuatorCfg(
      target_names_expr=(
        "JL0_hip_pitch", 
        "JR0_hip_pitch", 
        "JL1_hip_roll", 
        "JR1_hip_roll",
        "JL3_knee_pitch", 
        "JR3_knee_pitch"
      ),
      stiffness=60,
      damping=1.5,
      effort_limit=20.0,
    ),
    BuiltinPdActuatorCfg(
      target_names_expr=(
        "JL2_thigh_yaw", 
        "JR2_thigh_yaw", 
        "JL4_ankle_pitch", 
        "JR4_ankle_pitch"
      ),
      stiffness=50,
      damping=1.2,
      effort_limit=20.0,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

_BIPED_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.55),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "JL0_hip_pitch": 0.7,
    "JL1_hip_roll": -0.09,
    "JL2_thigh_yaw": -0.26,
    "JL3_knee_pitch": -1.3,
    "JL4_ankle_pitch": -0.7,

    "JR0_hip_pitch": -0.7,
    "JR1_hip_roll": 0.09,
    "JR2_thigh_yaw": 0.26,
    "JR3_knee_pitch": 1.3,
    "JR4_ankle_pitch": 0.7,
  },
  joint_vel={".*": 0.0},
)


def _get_biped_2d_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_BIPED_2D_ARTICULATION,
    init_state=_BIPED_INIT,
  )


def _make_env_cfg(num_envs: int = 1024) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel_2d,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel_2d,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "gait": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": "gait"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="biped_2d",
      actuator_names=_ACTUATED_JOINT_NAMES,
      scale=0.25,
      use_default_offset=True,
      preserve_order=True,
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "gait": mdp.UniformGaitCommandCfg(
      resampling_time_range=(1.0e6, 1.0e6),
      ranges=mdp.UniformGaitCommandCfg.Ranges(
        frequencies=(0.5, 1.5),
        duty_cycle=(0.5, 0.5),
      ),
    ),
  }

  rewards = {
    "forward_velocity": RewardTermCfg(
      func=mdp.forward_velocity,
      weight=1.0,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "joint_torque": RewardTermCfg(
      func=env_mdp.joint_torques_l2,
      weight=-1.0e-3,
      params={"asset_cfg": _ROBOT_ACTUATOR_CFG},
    ),
    "joint_velocity": RewardTermCfg(
      func=env_mdp.joint_vel_l2,
      weight=-1.0e-4,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "contact_schedule": RewardTermCfg(
      func=mdp.contact_schedule,
      weight=1.0,
      params={
        "asset_cfg": _FEET_CONTACT_CFG,
        "terrain_cfg": _TERRAIN_CFG,
        "command_name": "gait",
        "sigma": 0.15,
      },
    ),
    "action_acc": RewardTermCfg(func=env_mdp.action_acc_l2, weight=-1.0e-3),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_pitch,
      params={"limit_angle": math.radians(70.0), "asset_cfg": _ROBOT_CFG},
    ),
    "base_contact_with_ground": TerminationTermCfg(
      func=mdp.base_contact_with_ground,
      params={"asset_cfg": _BASE_CONTACT_CFG, "terrain_cfg": _TERRAIN_CFG},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"biped_2d": _get_biped_2d_cfg()},
      num_envs=num_envs,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="biped_2d",
      body_name="pelvis",
      distance=5.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005),
    ),
    decimation=4,
    episode_length_s=20.0,
  )


def biped_2d_env_cfg(
  play: bool = False,
  num_envs: int = 1024,
  play_num_envs: int = _PLAY_NUM_ENVS,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(play_num_envs if play else num_envs)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg
