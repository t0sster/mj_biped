from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

from mj_biped.tasks.bd_lip.mdp.lip import gait_phase_from_command


def base_heading(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.heading_w.unsqueeze(1)


def gait_phase(env, command_name: str = "gait_command") -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  phase = torch.remainder(env.episode_length_buf * env.step_dt * command[:, 0], 1.0)
  phase = phase.unsqueeze(1)
  return torch.cat(
    (torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)),
    dim=1,
  )


def gait_command(env, command_name: str = "gait_command") -> torch.Tensor:
  return env.command_manager.get_command(command_name)


def robot_joint_torque(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.qfrc_actuator[:, asset_cfg.joint_ids]


def robot_joint_acc(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_acc


def robot_mass(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  mass = torch.as_tensor(asset.data.model.body_mass, device=asset.data.device)
  return mass.reshape(mass.shape[0], -1)


def robot_inertia(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  inertia = torch.as_tensor(asset.data.model.body_inertia, device=asset.data.device)
  return inertia.reshape(inertia.shape[0], -1)


def robot_joint_stiffness(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  stiffness = torch.as_tensor(asset.data.model.jnt_stiffness, device=asset.data.device)
  return stiffness.reshape(stiffness.shape[0], -1)


def robot_joint_damping(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  damping = torch.as_tensor(asset.data.model.dof_damping, device=asset.data.device)
  return damping.reshape(damping.shape[0], -1)


def robot_base_pose(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w


def robot_contact_force(
  env,
  sensor_name: str,
  body_names: tuple[str, str],
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert hasattr(sensor, "data")
  order = torch.tensor(
    [sensor.primary_names.index(name) for name in body_names],
    device=env.device,
    dtype=torch.long,
  )
  force_history = sensor.data.force_history
  if force_history is not None:
    force_history = force_history.index_select(1, order)
    return force_history.reshape(force_history.shape[0], -1)
  assert sensor.data.force is not None
  force = sensor.data.force.index_select(1, order)
  return force.reshape(force.shape[0], -1)


def foot_state_b(
  env,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]

  root_pos_w = asset.data.root_link_pos_w
  root_quat_w = asset.data.root_link_quat_w
  root_heading_w = asset.data.heading_w.unsqueeze(1)

  rel_pos_b = math_utils.quat_apply_inverse(
    root_quat_w,
    foot_pos_w[:, 0, :] - root_pos_w,
  )
  forward_b = torch.tensor([1.0, 0.0, 0.0], device=asset.data.device).repeat(
    foot_quat_w.shape[0], 1
  )
  foot_forward_w = math_utils.quat_apply(foot_quat_w[:, 0, :], forward_b)
  foot_yaw_w = torch.atan2(foot_forward_w[:, 1], foot_forward_w[:, 0]).unsqueeze(1)
  rel_yaw = math_utils.wrap_to_pi(foot_yaw_w - root_heading_w)
  return torch.cat((rel_pos_b, rel_yaw), dim=1)


def step_command_b(
  env,
  index: int,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  command_name: str = "lip_step_command",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  target_w = command[:, 0:3] if index == 0 else command[:, 3:6]

  root_pos_w = asset.data.root_link_pos_w
  root_quat_w = asset.data.root_link_quat_w
  root_heading_w = asset.data.heading_w.unsqueeze(1)

  target_pos_w = torch.zeros(target_w.shape[0], 3, device=target_w.device)
  target_pos_w[:, :2] = target_w[:, :2]
  target_pos_w[:, 2] = root_pos_w[:, 2]
  rel_pos_b = math_utils.quat_apply_inverse(root_quat_w, target_pos_w - root_pos_w)
  rel_yaw = math_utils.wrap_to_pi(target_w[:, 2:3] - root_heading_w)
  return torch.cat((rel_pos_b, rel_yaw), dim=1)


def contact_schedule_phase(
  env,
  command_name: str = "gait_command",
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  right_phase, left_phase, duration = gait_phase_from_command(
    env.episode_length_buf, env.step_dt, command
  )
  return torch.stack((right_phase, left_phase, duration), dim=1)
