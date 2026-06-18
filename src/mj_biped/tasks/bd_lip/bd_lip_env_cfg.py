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
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mj_biped.tasks.bd_lip import mdp

BD_XML = Path(__file__).parents[2] / "assets" / "bd" / "bd.xml"

BD_JOINT_NAMES = (
  "J_L0",
  "J_R0",
  "J_L1",
  "J_R1",
  "J_L2",
  "J_R2",
  "J_L3",
  "J_R3",
  "J_L4_ankle",
  "J_R4_ankle",
)

BD_INIT_JOINT_POS = {
  "J_L0": 0.0,
  "J_L1": 0.0,
  "J_L2": 0.56,
  "J_L3": 1.12,
  "J_L4_ankle": 0.57,
  "J_R0": 0.0,
  "J_R1": 0.0,
  "J_R2": -0.56,
  "J_R3": -1.12,
  "J_R4_ankle": -0.57,
}

FEET_BODY_NAMES = ("R4_Link_ankle", "L4_Link_ankle")


def get_bd_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(BD_XML))
  root = spec.bodies[1]
  if not any(j.name == "floating_base_joint" for j in spec.joints):
    root.add_freejoint(name="floating_base_joint")
  return spec


BD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPdActuatorCfg(
      target_names_expr=("J_L0", "J_R0", "J_L4_ankle", "J_R4_ankle"),
      stiffness=50.0,
      damping=1.2,
      effort_limit=20.0,
    ),
    BuiltinPdActuatorCfg(
      target_names_expr=("J_L1", "J_R1", "J_L2", "J_R2", "J_L3", "J_R3"),
      stiffness=60.0,
      damping=1.5,
      effort_limit=20.0,
    ),
  ),
  soft_joint_pos_limit_factor=1.0,
)

BD_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.36),
  joint_pos=BD_INIT_JOINT_POS,
  joint_vel={".*": 0.0},
)


def get_bd_entity_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_bd_spec,
    articulation=BD_ARTICULATION,
    init_state=BD_INIT_STATE,
  )


def make_bd_lip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  right_foot_cfg = SceneEntityCfg("robot", body_names=("R4_Link_ankle",))
  left_foot_cfg = SceneEntityCfg("robot", body_names=("L4_Link_ankle",))
  feet_body_cfg = SceneEntityCfg(
    "robot",
    body_names=FEET_BODY_NAMES,
    preserve_order=True,
  )

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_heading": ObservationTermCfg(
          func=mdp.base_heading,
          noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "base_ang_vel": ObservationTermCfg(
          func=env_mdp.base_ang_vel,
          noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "projected_gravity": ObservationTermCfg(
          func=env_mdp.projected_gravity,
          noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "right_foot_state": ObservationTermCfg(
          func=mdp.foot_state_b,
          params={"asset_cfg": right_foot_cfg},
        ),
        "left_foot_state": ObservationTermCfg(
          func=mdp.foot_state_b,
          params={"asset_cfg": left_foot_cfg},
        ),
        "right_step_target": ObservationTermCfg(
          func=mdp.step_command_b,
          params={"index": 0},
        ),
        "left_step_target": ObservationTermCfg(
          func=mdp.step_command_b,
          params={"index": 1},
        ),
        "base_velocity_command": ObservationTermCfg(
          func=env_mdp.generated_commands,
          params={"command_name": "base_velocity"},
        ),
        "base_height_command": ObservationTermCfg(
          func=env_mdp.generated_commands,
          params={"command_name": "base_height_command"},
        ),
        "gait_phase": ObservationTermCfg(func=mdp.gait_phase),
        "joint_pos": ObservationTermCfg(
          func=env_mdp.joint_pos_rel,
          noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
          func=env_mdp.joint_vel_rel,
          noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
      },
      concatenate_terms=True,
      enable_corruption=not play,
    ),
  }
  observations["critic"] = ObservationGroupCfg(
    terms=dict(observations["actor"].terms),
    concatenate_terms=True,
    enable_corruption=False,
  )
  observations["critic"].terms.update(
    {
      "base_pos": ObservationTermCfg(func=mdp.robot_base_pose),
      "base_lin_vel": ObservationTermCfg(func=env_mdp.base_lin_vel),
      "joint_torque": ObservationTermCfg(func=mdp.robot_joint_torque),
      "joint_acc": ObservationTermCfg(func=mdp.robot_joint_acc),
      "robot_mass": ObservationTermCfg(func=mdp.robot_mass),
      "robot_inertia": ObservationTermCfg(func=mdp.robot_inertia),
      "robot_joint_stiffness": ObservationTermCfg(func=mdp.robot_joint_stiffness),
      "robot_joint_damping": ObservationTermCfg(func=mdp.robot_joint_damping),
      "feet_contact_force": ObservationTermCfg(
        func=mdp.robot_contact_force,
        params={"sensor_name": "feet_contact", "body_names": FEET_BODY_NAMES},
      ),
    }
  )

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=BD_JOINT_NAMES,
      scale=0.75,
      use_default_offset=True,
      preserve_order=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      heading_command=True,
      heading_control_stiffness=1.0,
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      rel_forward_envs=0.6,
      resampling_time_range=(5.0, 5.0),
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.2, 0.3),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(-0.15, 0.15),
        heading=(-math.pi / 4.0, math.pi / 4.0),
      ),
    ),
    "gait_command": mdp.UniformGaitCommandCfg(
      resampling_time_range=(5.0, 5.0),
      ranges=mdp.UniformGaitCommandCfg.Ranges(
        frequencies=(1.5, 2.5),
        offsets=(0.45, 0.55),
        durations=(0.45, 0.55),
      ),
    ),
    "lip_step_command": mdp.LipStepCommandCfg(
      entity_name="robot",
      foot_body_names=FEET_BODY_NAMES,
      nominal_step_length=None,
      nominal_step_width=0.20,
      step_period_s=None,
      use_cmd_heading=True,
      resampling_time_range=(1.0e6, 1.0e6),
      ranges=mdp.LipStepCommandCfg.Ranges(
        step_length=None,
        step_width=(0.18, 0.21),
        step_period_s=None,
      ),
    ),
    "base_height_command": mdp.BaseHeightCommandCfg(
      resampling_time_range=(1.0e6, 1.0e6),
      ranges=mdp.BaseHeightCommandCfg.Ranges(height=(0.26, 0.30)),
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=env_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
        "velocity_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (0.0, 0.0),
        }
      },
    ),
    "reset_joints": EventTermCfg(
      func=env_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=3.0,
      params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    ),
    "step_tracking": RewardTermCfg(
      func=mdp.step_command_tracking,
      weight=3.0,
      params={
        "asset_cfg": feet_body_cfg,
        "command_name": "lip_step_command",
        "gait_command_name": "gait_command",
        "velocity_command_name": "base_velocity",
        "position_sigma": 0.05,
        "yaw_sigma": 0.25,
        "command_threshold": 0.02,
      },
    ),
    "heading": RewardTermCfg(
      func=mdp.heading_tracking,
      weight=0.5,
      params={"command_name": "base_velocity", "heading_sigma": 0.15},
    ),
    "contact_schedule": RewardTermCfg(
      func=mdp.contact_schedule,
      weight=2.0,
      params={
        "sensor_name": "feet_contact",
        "command_name": "gait_command",
        "body_names": FEET_BODY_NAMES,
        "threshold": 1.0,
        "sigma": 0.25,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_contact",
        "command_name": "base_velocity",
        "gait_command_name": "gait_command",
        "body_names": FEET_BODY_NAMES,
        "threshold": 0.1,
        "swing_time_scale": 0.5,
        "min_threshold": 0.25,
        "dense": False,
        "contact_force_threshold": 1.0,
        "single_support_only": False,
      },
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_tracking_exp,
      weight=1.0,
      params={"command_name": "base_height_command", "height_sigma": 0.05},
    ),
    "joint_torques": RewardTermCfg(func=env_mdp.joint_torques_l2, weight=-1.0e-4),
    "joint_vel": RewardTermCfg(func=env_mdp.joint_vel_l2, weight=-1.0e-3),
    "joint_pos_limits": RewardTermCfg(func=env_mdp.joint_pos_limits, weight=-1.0),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-0.5,
      params={
        "command_name": "base_velocity",
        "lin_threshold": 0.02,
        "ang_threshold": 0.02,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.foot_slip_penalty,
      weight=-0.5,
      params={
        "sensor_name": "feet_contact",
        "asset_cfg": feet_body_cfg,
        "body_names": FEET_BODY_NAMES,
        "contact_threshold": 1.0,
      },
    ),
    "action_acc": RewardTermCfg(func=env_mdp.action_acc_l2, weight=-1.0e-3),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-1.0e-2),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0e-1),
    "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=env_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "base_contact": TerminationTermCfg(
      func=velocity_mdp.illegal_contact,
      params={"sensor_name": "base_contact", "force_threshold": 1.0},
    ),
    "max_velocity": TerminationTermCfg(
      func=mdp.exceeds_max_velocity,
      params={"max_velocity": 3.0},
    ),
  }

  num_envs = 4 if play else 2048
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_bd_entity_cfg()},
      sensors=(
        ContactSensorCfg(
          name="feet_contact",
          primary=ContactMatch(
            mode="body",
            pattern=FEET_BODY_NAMES,
            entity="robot",
          ),
          fields=("found", "force"),
          reduce="netforce",
          track_air_time=True,
          history_length=4,
        ),
        ContactSensorCfg(
          name="base_contact",
          primary=ContactMatch(
            mode="body",
            pattern="base_link",
            entity="robot",
          ),
          fields=("found", "force"),
          reduce="netforce",
          history_length=4,
        ),
      ),
      num_envs=num_envs,
      env_spacing=2.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=64,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.002,
        integrator="implicitfast",
        iterations=4,
        ls_iterations=4,
      ),
    ),
    decimation=10,
    episode_length_s=20.0,
    seed=42,
  )

  if play:
    cfg.commands["base_velocity"].ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.25, 0.25),
      lin_vel_y=(-0.1, 0.1),
      ang_vel_z=(-0.1, 0.1),
      heading=(-math.pi / 4.0, math.pi / 4.0),
    )
    cfg.commands["base_velocity"].rel_standing_envs = 0.0
    cfg.commands["lip_step_command"].nominal_step_width = 0.20
    cfg.commands["lip_step_command"].ranges = mdp.LipStepCommandCfg.Ranges(
      step_length=None,
      step_width=(0.20, 0.20),
      step_period_s=None,
    )

  return cfg


def bd_lip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_bd_lip_env_cfg(play=play)


def bd_lip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="bd_lip",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=5001,
    logger="tensorboard",
    upload_model=False,
  )
