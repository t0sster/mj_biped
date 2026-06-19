from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

from mj_biped.tasks.bd_lip.mdp.footstep_planners import PFootstepPlanner
from mj_biped.tasks.bd_lip.mdp.lip import (
  gait_phase_from_command,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class UniformGaitCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    frequencies: tuple[float, float]
    offsets: tuple[float, float]
    durations: tuple[float, float]

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> GaitCommand:
    return GaitCommand(self, env)


class GaitCommand(CommandTerm):
  cfg: UniformGaitCommandCfg

  def __init__(self, cfg: UniformGaitCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.gait_command = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.gait_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
    self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.offsets)
    self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.durations)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class BaseHeightCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> BaseHeightCommand:
    return BaseHeightCommand(self, env)


class BaseHeightCommand(CommandTerm):
  cfg: BaseHeightCommandCfg

  def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class LipStepCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    step_length: tuple[float, float] | None = None
    step_width: tuple[float, float] | None = None
    step_period_s: tuple[float, float] | None = None

  entity_name: str = "robot"
  foot_body_names: tuple[str, str] = ("R4_Link_ankle", "L4_Link_ankle")
  nominal_step_length: float | None = None
  nominal_step_width: float = 0.20
  step_period_s: float | None = None
  use_cmd_heading: bool = True
  heading_speed_eps: float = 1.0e-3
  stride_compensation_gain: float = 0.0
  stride_compensation_max_ratio: float = 0.5
  turn_width_gain: float = 0.15
  turn_length_gain: float = 1.0
  ranges: Ranges | None = None

  def build(self, env: ManagerBasedRlEnv) -> LipStepCommand:
    return LipStepCommand(self, env)


class LipStepCommand(CommandTerm):
  """Generate right/left foot targets with the BD-Lip XCoM planner."""

  cfg: LipStepCommandCfg

  def __init__(self, cfg: LipStepCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.step_target_command = torch.zeros(self.num_envs, 6, device=self.device)

    foot_cfg = SceneEntityCfg(
      cfg.entity_name,
      body_names=cfg.foot_body_names,
      preserve_order=True,
    )
    foot_cfg.resolve(env.scene)
    self.foot_body_ids = foot_cfg.body_ids

    self.forward_b = torch.tensor([1.0, 0.0, 0.0], device=self.device)
    self.swing_state = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
    self.frozen_targets_w = torch.zeros(self.num_envs, 2, 3, device=self.device)
    self.footstep_planner = PFootstepPlanner(cfg, self.device)

    self.step_length = torch.full(
      (self.num_envs, 1),
      cfg.nominal_step_length if cfg.nominal_step_length is not None else 0.0,
      device=self.device,
    )
    self.step_width = torch.full(
      (self.num_envs, 1), cfg.nominal_step_width, device=self.device
    )
    self.step_period = torch.full(
      (self.num_envs, 1),
      cfg.step_period_s if cfg.step_period_s is not None else 0.0,
      device=self.device,
    )
    self._viz_radius = 0.5 * cfg.nominal_step_width if cfg.nominal_step_width > 0.0 else 0.03

  @property
  def command(self) -> torch.Tensor:
    return self.step_target_command

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    if isinstance(env_ids, torch.Tensor):
      self.swing_state[env_ids] = False
      self.frozen_targets_w[env_ids] = 0.0
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.ranges is None:
      return
    n = len(env_ids)
    if self.cfg.ranges.step_length is not None:
      self.step_length[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
        *self.cfg.ranges.step_length
      )
    if self.cfg.ranges.step_width is not None:
      self.step_width[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
        *self.cfg.ranges.step_width
      )
    if self.cfg.ranges.step_period_s is not None:
      self.step_period[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
        *self.cfg.ranges.step_period_s
      )

  def _update_command(self) -> None:
    root_pos_w = self.robot.data.root_link_pos_w
    root_vel_w = self.robot.data.root_link_lin_vel_w
    root_quat_w = self.robot.data.root_link_quat_w
    foot_pos_w = self.robot.data.body_link_pos_w[:, self.foot_body_ids, :]
    foot_quat_w = self.robot.data.body_link_quat_w[:, self.foot_body_ids, :]

    gait_command = self._env.command_manager.get_command("gait_command")
    right_phase, left_phase, duration = gait_phase_from_command(
      self._env.episode_length_buf,
      self._env.step_dt,
      gait_command,
    )

    right_contact = right_phase < duration
    left_contact = left_phase < duration
    swing_right = ~right_contact
    swing_left = ~left_contact

    tie_break = (right_contact & left_contact) | ((~right_contact) & (~left_contact))
    swing_right[tie_break] = right_phase[tie_break] > left_phase[tie_break]
    swing_left[tie_break] = ~swing_right[tie_break]

    cmd = self._env.command_manager.get_command("base_velocity")
    cmd_vel_b = cmd[:, :2]
    cmd_wz = cmd[:, 2:3]
    plan = self.footstep_planner.plan(
      root_pos_w=root_pos_w,
      root_vel_w=root_vel_w,
      root_quat_w=root_quat_w,
      foot_pos_w=foot_pos_w,
      foot_quat_w=foot_quat_w,
      cmd_vel_b=cmd_vel_b,
      cmd_wz=cmd_wz,
      gait_command=gait_command,
      step_length_prior=self.step_length,
      step_width_prior=self.step_width,
      step_period_prior=self.step_period,
      episode_length_buf=self._env.episode_length_buf,
      step_dt=self._env.step_dt,
      swing_right=swing_right,
      swing_left=swing_left,
    )

    foot_forward = self.forward_b.repeat(foot_quat_w.shape[0], 1)
    right_forward_w = math_utils.quat_apply(foot_quat_w[:, 0, :], foot_forward)
    left_forward_w = math_utils.quat_apply(foot_quat_w[:, 1, :], foot_forward)
    right_yaw_w = torch.atan2(right_forward_w[:, 1], right_forward_w[:, 0])
    left_yaw_w = torch.atan2(left_forward_w[:, 1], left_forward_w[:, 0])

    swing_now = torch.stack((swing_right, swing_left), dim=1)
    swing_start = swing_now & ~self.swing_state
    self.swing_state = swing_now

    if swing_start[:, 0].any():
      right_ids = swing_start[:, 0]
      self.frozen_targets_w[right_ids, 0, :2] = plan.target_xy_w[right_ids]
      self.frozen_targets_w[right_ids, 0, 2] = plan.target_heading_w[right_ids, 0]
    if swing_start[:, 1].any():
      left_ids = swing_start[:, 1]
      self.frozen_targets_w[left_ids, 1, :2] = plan.target_xy_w[left_ids]
      self.frozen_targets_w[left_ids, 1, 2] = plan.target_heading_w[left_ids, 0]

    right_target = torch.zeros(self.num_envs, 3, device=self.device)
    left_target = torch.zeros(self.num_envs, 3, device=self.device)
    right_target[:, :2] = torch.where(
      swing_right.unsqueeze(1),
      self.frozen_targets_w[:, 0, :2],
      foot_pos_w[:, 0, :2],
    )
    left_target[:, :2] = torch.where(
      swing_left.unsqueeze(1),
      self.frozen_targets_w[:, 1, :2],
      foot_pos_w[:, 1, :2],
    )
    right_target[:, 2] = torch.where(
      swing_right, self.frozen_targets_w[:, 0, 2], right_yaw_w
    )
    left_target[:, 2] = torch.where(
      swing_left, self.frozen_targets_w[:, 1, 2], left_yaw_w
    )

    self.step_target_command[:, 0:3] = right_target
    self.step_target_command[:, 3:6] = left_target

  def _debug_vis_impl(self, visualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    step_targets = self.step_target_command
    foot_pos_w = self.robot.data.body_link_pos_w[:, self.foot_body_ids, :]
    swing_now = self.swing_state
    target_z = 0.02

    for i in env_indices:
      right_target = step_targets[i, 0:3].cpu().numpy()
      left_target = step_targets[i, 3:6].cpu().numpy()
      right_actual = foot_pos_w[i, 0].cpu().numpy()
      left_actual = foot_pos_w[i, 1].cpu().numpy()

      right_target_vis = right_target.copy()
      left_target_vis = left_target.copy()
      right_target_vis[2] = target_z
      left_target_vis[2] = target_z

      # Target markers stay fixed in world coordinates during swing.
      visualizer.add_sphere(
        center=right_target_vis,
        radius=self._viz_radius,
        color=(1.0, 0.0, 0.0, 0.35),
        label=f"step_target_right_{i}",
      )
      visualizer.add_sphere(
        center=left_target_vis,
        radius=self._viz_radius,
        color=(0.0, 0.0, 1.0, 0.35),
        label=f"step_target_left_{i}",
      )

      # Actual foot centers.
      visualizer.add_sphere(
        center=right_actual,
        radius=0.35 * self._viz_radius,
        color=(1.0, 0.4, 0.4, 0.85),
        label=f"step_actual_right_{i}",
      )
      visualizer.add_sphere(
        center=left_actual,
        radius=0.35 * self._viz_radius,
        color=(0.4, 0.4, 1.0, 0.85),
        label=f"step_actual_left_{i}",
      )

      # Optional: highlight which foot is in swing.
      if swing_now[i, 0]:
        visualizer.add_sphere(
          center=right_actual,
          radius=0.6 * self._viz_radius,
          color=(1.0, 0.0, 0.0, 0.15),
          label=f"step_swing_right_{i}",
        )
      if swing_now[i, 1]:
        visualizer.add_sphere(
          center=left_actual,
          radius=0.6 * self._viz_radius,
          color=(0.0, 0.0, 1.0, 0.15),
          label=f"step_swing_left_{i}",
        )

  def _update_metrics(self) -> None:
    pass
