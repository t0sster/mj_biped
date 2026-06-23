from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from mjlab.utils.lab_api import math as math_utils

from mj_biped.tasks.bd_lip.mdp.lip import compute_xcom_step_targets_w


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
    com_pos_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    cmd_vel_b: torch.Tensor,
    cmd_wz: torch.Tensor,
    target_height: torch.Tensor,
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
    com_pos_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_quat_w: torch.Tensor,
    cmd_vel_b: torch.Tensor,
    cmd_wz: torch.Tensor,
    target_height: torch.Tensor,
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
    root_vel_b = math_utils.quat_apply_inverse(yaw_quat_w, root_vel_w)
    cmd_vel_w_3d = torch.zeros(root_pos_w.shape[0], 3, device=self.device)
    cmd_vel_w_3d[:, :2] = cmd_vel_b
    cmd_vel_w = math_utils.quat_apply(yaw_quat_w, cmd_vel_w_3d)[:, :2]

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

    cmd_speed = torch.norm(cmd_vel_b, dim=1, keepdim=True)
    step_dir_b = torch.where(
      cmd_speed > self.cfg.heading_speed_eps,
      cmd_vel_b / torch.clamp(cmd_speed, min=1.0e-6),
      torch.tensor([1.0, 0.0], device=self.device).view(1, 2).expand(cmd_vel_b.shape[0], -1),
    )
    step_heading_b = torch.where(
      cmd_speed > self.cfg.heading_speed_eps,
      torch.atan2(cmd_vel_b[:, 1], cmd_vel_b[:, 0]).unsqueeze(1),
      torch.zeros_like(cmd_speed),
    )
    base_heading_w = self._base_heading(root_quat_w)
    step_heading_w = torch.where(
      cmd_speed > self.cfg.heading_speed_eps,
      torch.atan2(cmd_vel_w[:, 1], cmd_vel_w[:, 0]).unsqueeze(1),
      base_heading_w,
    )

    if step_length_prior is not None and torch.any(step_length_prior.abs() > 1.0e-8):
      nominal_step_length = step_length_prior
    elif self.cfg.nominal_step_length is not None:
      nominal_step_length = torch.full_like(step_time, self.cfg.nominal_step_length)
    else:
      nominal_step_length = cmd_speed * step_time
    step_length = torch.clamp(nominal_step_length, min=0.0, max=0.35)

    if step_width_prior is not None and torch.any(step_width_prior.abs() > 1.0e-8):
      nominal_step_width = step_width_prior
    else:
      nominal_step_width = torch.full_like(step_time, self.cfg.nominal_step_width)
    step_width = torch.clamp(nominal_step_width, min=0.05)

    target_heading_w = (
      math_utils.wrap_to_pi(base_heading_w + cmd_wz * step_time)
      if self.cfg.use_cmd_heading
      else base_heading_w
    )

    support_pos_w = torch.where(
      swing_right.unsqueeze(1),
      foot_pos_w[:, 1, :],
      foot_pos_w[:, 0, :],
    )
    heading_dir_b = torch.stack(
      (torch.cos(step_heading_b.squeeze(1)), torch.sin(step_heading_b.squeeze(1))),
      dim=1,
    )
    speed_scale = cmd_vel_b[:, 0:1].abs() / (
      cmd_vel_b[:, 0:1].abs() + cmd_vel_b[:, 1:2].abs() + 1.0e-6
    )
    delta_along = (
      (foot_pos_b[:, 0, :2] - foot_pos_b[:, 1, :2]).mul(heading_dir_b).sum(dim=1, keepdim=True)
    )
    comp = self.cfg.stride_compensation_gain * speed_scale * delta_along
    comp_limit = self.cfg.stride_compensation_max_ratio * step_length
    comp = torch.clamp(comp, min=-comp_limit, max=comp_limit)
    swing_sign = (swing_left.float() - swing_right.float()).unsqueeze(1)
    cmd_speed_along = cmd_speed
    act_speed_along = torch.sum(root_vel_b[:, :2] * step_dir_b, dim=1, keepdim=True)
    speed_error = cmd_speed_along - act_speed_along
    speed_feedback = self.cfg.stride_compensation_gain * speed_error * step_time
    speed_feedback = torch.clamp(speed_feedback, min=-comp_limit, max=comp_limit)
    step_length_eff = torch.clamp(step_length + speed_feedback + swing_sign * comp, min=0.0)
    target_w = compute_xcom_step_targets_w(
      com_pos_w,
      root_vel_w,
      support_pos_w,
      step_heading_w,
      step_time,
      step_width,
      step_length_eff,
      target_height,
      swing_left,
    )

    return FootstepPlan(
      step_time=step_time,
      step_length=step_length,
      step_width=step_width,
      step_heading_b=step_heading_b,
      target_heading_w=target_heading_w,
      target_xy_w=target_w[:, :2],
    )

  def _base_heading(self, root_quat_w: torch.Tensor) -> torch.Tensor:
    forward_b = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(root_quat_w.shape[0], 1)
    forward_w = math_utils.quat_apply(math_utils.yaw_quat(root_quat_w), forward_b)
    return torch.atan2(forward_w[:, 1], forward_w[:, 0]).unsqueeze(1)
