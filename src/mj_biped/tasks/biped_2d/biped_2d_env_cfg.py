"""Biped 2D locomotion environment configuration."""

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
from mjlab.managers.event_manager import EventTermCfg
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
_ROBOT_ACTUATOR_CFG = SceneEntityCfg(
  "biped_2d",
  actuator_names=_ACTUATED_JOINT_NAMES,
)

_PLAY_NUM_ENVS = 1


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_BIPED_2D_XML))


_BIPED_2D_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPdActuatorCfg(
      target_names_expr=("JL0_hip_pitch", "JR0_hip_pitch"),
      stiffness=60.0,
      damping=1.5,
      effort_limit=20.0,
    ),
    BuiltinPdActuatorCfg(
      target_names_expr=("JL1_hip_roll", "JR1_hip_roll"),
      stiffness=60.0,
      damping=1.5,
      effort_limit=20.0,
    ),
    BuiltinPdActuatorCfg(
      target_names_expr=("JL2_thigh_yaw", "JR2_thigh_yaw"),
      stiffness=50.0,
      damping=1.2,
      effort_limit=20.0,
    ),
    BuiltinPdActuatorCfg(
      target_names_expr=("JL3_knee_pitch", "JR3_knee_pitch"),
      stiffness=60.0,
      damping=1.5,
      effort_limit=20.0,
    ),
    BuiltinPdActuatorCfg(
      target_names_expr=("JL4_ankle_pitch", "JR4_ankle_pitch"),
      stiffness=50.0,
      damping=1.2,
      effort_limit=20.0,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

_BIPED_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "root_x": 0.0,
    "root_z": 0.0,
    "root_pitch": 0.0,

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
    sort_actuators=True,
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

  events = {
    "reset_joints": EventTermCfg(
      func=env_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (-0.1, 0.1),
        "asset_cfg": _ROBOT_CFG,
      },
    ),
  }

  rewards = {
    "alive": RewardTermCfg(
      func=mdp.alive,
      weight=0.25,
    ),
    "forward_velocity": RewardTermCfg(
      func=mdp.forward_velocity,
      weight=1.5,
      params={"asset_cfg": _ROBOT_CFG},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={"asset_cfg": _ROBOT_CFG, "sigma": math.sqrt(0.35)},
    ),
    "lateral_centering": RewardTermCfg(
      func=mdp.lateral_centering,
      weight=1.0,
      params={"asset_cfg": _ROBOT_CFG, "sigma": math.sqrt(0.15)},
    ),
    "joint_torque": RewardTermCfg(
      func=mdp.joint_torque,
      weight=-1.0e-3,
      params={"actuator_cfg": _ROBOT_ACTUATOR_CFG, "sigma": math.sqrt(10.0)},
    ),
    "joint_velocity": RewardTermCfg(
      func=mdp.joint_velocity,
      weight=-1.0e-4,
      params={"asset_cfg": _ROBOT_CFG, "sigma": math.sqrt(10.0)},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=env_mdp.bad_orientation,
      params={"limit_angle": 1.25, "asset_cfg": _ROBOT_CFG},
    ),
    "root_height_below_minimum": TerminationTermCfg(
      func=env_mdp.root_height_below_minimum,
      params={"minimum_height": -0.2, "asset_cfg": _ROBOT_CFG},
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
    events=events,
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
