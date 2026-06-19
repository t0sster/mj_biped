from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from mjlab.utils.lab_api import math as math_utils

from mj_biped.tasks.bd_lip.mdp.lip import compute_xcom_step_targets_b


class LipStepPlannerCfg(Protocol):
  nominal_step_length: float | None
  nominal_step_width: float
  step_period_s: float | None
  use_cmd_heading: bool
  heading_speed_eps: float
  stride_compensation_gain: float
  stride_compensation_max_ratio: float
  turn_width_gain: float
  turn_length_gain: float


@dataclass(slots=True)
class FootstepPlan:
  step_time: torch.Tensor
  step_length: torch.Tensor
  step_width: torch.Tensor
  step_heading_b: torch.Tensor
  target_heading_w: torch.Tensor
  target_xy_w: torch.Tensor


class FootstepPlanner(Protocol):
  def plan(
    self,
    *,
    root_pos_w: torch.Tensor,
    root_vel_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    cmd_vel_b: torch.Tensor,
    cmd_wz: torch.Tensor,
    gait_command: torch.Tensor,
    step_length_prior: torch.Tensor | None,
    step_width_prior: torch.Tensor | None,
    step_period_prior: torch.Tensor | None,
    episode_length_buf: torch.Tensor,
    step_dt: float,
    swing_right: torch.Tensor,
    swing_left: torch.Tensor,
  ) -> FootstepPlan:
    raise NotImplementedError


class PFootstepPlanner:
  """Velocity-to-footstep planner with P feedback on velocity error.

  The planner keeps gait timing separate from velocity regulation so a future
  MPC implementation can replace only this class without touching the command
  term wiring.
  """

  def __init__(self, cfg: LipStepPlannerCfg, device: torch.device):
    self.cfg = cfg
    self.device = device

  def plan(
    self,
    *,
    root_pos_w: torch.Tensor,
    root_vel_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    cmd_vel_b: torch.Tensor,
    cmd_wz: torch.Tensor,
    gait_command: torch.Tensor,
    step_length_prior: torch.Tensor | None,
    step_width_prior: torch.Tensor | None,
    step_period_prior: torch.Tensor | None,
    episode_length_buf: torch.Tensor,
    step_dt: float,
    swing_right: torch.Tensor,
    swing_left: torch.Tensor,
  ) -> FootstepPlan:
    yaw_quat_w = math_utils.yaw_quat(root_quat_w)
    root_pos_b = torch.zeros_like(root_pos_w)
    root_pos_b[:, 2:3] = root_pos_w[:, 2:3]
    root_vel_b = math_utils.quat_apply_inverse(yaw_quat_w, root_vel_w)

    foot_pos_b = math_utils.quat_apply_inverse(
      yaw_quat_w.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1).reshape(-1, 4),
      (foot_pos_w - root_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).reshape(foot_pos_w.shape)

    if step_period_prior is not None and torch.any(step_period_prior.abs() > 1.0e-8):
      step_time = step_period_prior
    elif self.cfg.step_period_s is not None:
      step_time = torch.full((root_pos_w.shape[0], 1), self.cfg.step_period_s, device=self.device)
    else:
      freq = gait_command[:, 0].clamp(min=1.0e-3)
      step_time = (0.5 / freq).unsqueeze(1)

    cmd_forward = cmd_vel_b[:, 0:1]
    cmd_speed = torch.norm(cmd_vel_b, dim=1, keepdim=True)
    step_heading_b = torch.where(
      cmd_speed > self.cfg.heading_speed_eps,
      torch.atan2(cmd_vel_b[:, 1], cmd_vel_b[:, 0]).unsqueeze(1),
      torch.zeros_like(cmd_speed),
    )

    if step_length_prior is not None and torch.any(step_length_prior.abs() > 1.0e-8):
      nominal_step_length = step_length_prior
    elif self.cfg.nominal_step_length is not None:
      nominal_step_length = torch.full_like(step_time, self.cfg.nominal_step_length)
    else:
      nominal_step_length = torch.abs(cmd_forward) * step_time
    turn_length_boost = self.cfg.turn_length_gain * torch.abs(cmd_wz) * step_time
    step_length = torch.clamp(
      nominal_step_length + turn_length_boost,
      min=0.0,
      max=0.35,
    )

    if step_width_prior is not None and torch.any(step_width_prior.abs() > 1.0e-8):
      nominal_step_width = step_width_prior
    else:
      nominal_step_width = torch.full_like(step_time, self.cfg.nominal_step_width)
    step_width = (
      nominal_step_width
      + self.cfg.turn_width_gain * torch.abs(cmd_wz) * step_time
    )
    step_width = torch.clamp(step_width, min=0.05)
    step_width = torch.clamp(
      step_width,
      max=(1.0 + self.cfg.stride_compensation_max_ratio) * nominal_step_width,
    )

    target_heading_w = (
      math_utils.wrap_to_pi(self._base_heading(root_quat_w) + cmd_wz * step_time)
      if self.cfg.use_cmd_heading
      else self._base_heading(root_quat_w)
    )

    support_pos_b = torch.where(
      swing_right.unsqueeze(1),
      foot_pos_b[:, 1, :],
      foot_pos_b[:, 0, :],
    )
    target_b = compute_xcom_step_targets_b(
      root_pos_b,
      root_vel_b,
      support_pos_b,
      cmd_vel_b,
      step_heading_b,
      step_time,
      step_width,
      step_length,
      swing_left,
    )
    velocity_error_b = cmd_vel_b - root_vel_b[:, :2]
    velocity_correction_b = self.cfg.stride_compensation_gain * velocity_error_b * step_time
    forward_limit = self.cfg.stride_compensation_max_ratio * torch.clamp(step_length, min=0.05)
    lateral_limit = self.cfg.stride_compensation_max_ratio * torch.clamp(step_width, min=0.05)
    velocity_correction_b = torch.clamp(
      velocity_correction_b,
      min=torch.cat((-forward_limit, -lateral_limit), dim=1),
      max=torch.cat((forward_limit, lateral_limit), dim=1),
    )
    target_b[:, :2] = target_b[:, :2] + velocity_correction_b

    target_vec_b = torch.zeros_like(target_b)
    target_vec_b[:, :2] = target_b[:, :2]
    target_xy_w = root_pos_w[:, :2] + math_utils.quat_apply(yaw_quat_w, target_vec_b)[:, :2]

    return FootstepPlan(
      step_time=step_time,
      step_length=step_length,
      step_width=step_width,
      step_heading_b=step_heading_b,
      target_heading_w=target_heading_w,
      target_xy_w=target_xy_w,
    )

  def _base_heading(self, root_quat_w: torch.Tensor) -> torch.Tensor:
    forward_b = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(root_quat_w.shape[0], 1)
    forward_w = math_utils.quat_apply(math_utils.yaw_quat(root_quat_w), forward_b)
    return torch.atan2(forward_w[:, 1], forward_w[:, 0]).unsqueeze(1)
