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

_ROBOT_2D_XML: Path = (
  Path(__file__).resolve().parents[2] / "assets" / "biped_2d" / "robot_2d.xml"
)

_ROBOT_JOINT_NAMES = (
  "root_x",
  "root_z",
  "root_pitch",
  "hip_right",
  "knee_right",
  "hip_left",
  "knee_left",
)

_ACTUATED_JOINT_NAMES = (
  "hip_right",
  "knee_right",
  "hip_left",
  "knee_left",
)

_ROBOT_CFG = SceneEntityCfg("robot_2d", joint_names=_ROBOT_JOINT_NAMES)
_BASE_CONTACT_CFG = SceneEntityCfg(
  "robot_2d",
  body_names=("base",),
)
_FEET_CONTACT_CFG = SceneEntityCfg(
  "robot_2d",
  body_names=("foot_left", "foot_right"),
  preserve_order=True,
)
_TERRAIN_CFG = SceneEntityCfg("terrain", geom_names=("terrain",))

_TORQUE_ACTUATOR_NAMES = tuple(f"{name}_pd_.*" for name in _ACTUATED_JOINT_NAMES)

_ROBOT_ACTUATOR_CFG = SceneEntityCfg(
  "robot_2d",
  actuator_names=_TORQUE_ACTUATOR_NAMES,
)

_PLAY_NUM_ENVS = 1


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_ROBOT_2D_XML))


_ROBOT_2D_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPdActuatorCfg(
      target_names_expr=(
        "hip_right",
        "hip_left",
      ),
      stiffness=60,
      damping=1.5,
      effort_limit=20.0,
    ),
    BuiltinPdActuatorCfg(
      target_names_expr=(
        "knee_right",
        "knee_left",
      ),
      stiffness=50,
      damping=1.2,
      effort_limit=20.0,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

_ROBOT_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "hip_left": 0.35,
    "knee_left": -0.5,
    "hip_right": 0.35,
    "knee_right": -0.5,
  },
  joint_vel={".*": 0.0},
)


def _get_robot_2d_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_ROBOT_2D_ARTICULATION,
    init_state=_ROBOT_INIT,
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
      entity_name="robot_2d",
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
        frequencies=(0.25, 1.0),
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
      entities={"robot_2d": _get_robot_2d_cfg()},
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
      entity_name="robot_2d",
      body_name="base",
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


def robot_2d_env_cfg(
  play: bool = False,
  num_envs: int = 1024,
  play_num_envs: int = _PLAY_NUM_ENVS,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(play_num_envs if play else num_envs)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg
