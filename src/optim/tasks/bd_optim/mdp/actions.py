from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg


@dataclass(kw_only=True)
class JointPositionToMotorEffortActionCfg(BaseActionCfg):
  """Position-target policy action applied through XML motor actuators."""

  stiffness: float | dict[str, float] = 30.0
  damping: float | dict[str, float] = 3.0
  use_default_offset: bool = True

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT

  def build(self, env) -> JointPositionToMotorEffortAction:
    return JointPositionToMotorEffortAction(self, env)


class JointPositionToMotorEffortAction(BaseAction):
  cfg: JointPositionToMotorEffortActionCfg

  def __init__(self, cfg: JointPositionToMotorEffortActionCfg, env) -> None:
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

    self._stiffness = self._resolve_gain(cfg.stiffness)
    self._damping = self._resolve_gain(cfg.damping)
    self._gear, self._ctrl_min, self._ctrl_max = self._xml_motor_params()

  def apply_actions(self) -> None:
    q_des = self._processed_actions - self._entity.data.encoder_bias[
      :,
      self._target_ids,
    ]
    q = self._entity.data.joint_pos[:, self._target_ids]
    qd = self._entity.data.joint_vel[:, self._target_ids]
    tau_joint = self._stiffness * (q_des - q) - self._damping * qd
    tau_motor = tau_joint / self._gear
    tau_motor = torch.clamp(tau_motor, min=self._ctrl_min, max=self._ctrl_max)
    self._entity.set_joint_effort_target(tau_motor, joint_ids=self._target_ids)

  def _resolve_gain(self, gain: float | dict[str, float]) -> torch.Tensor:
    if isinstance(gain, (float, int)):
      return torch.full(
        (self.num_envs, self.action_dim),
        float(gain),
        device=self.device,
      )

    values = torch.zeros(self.action_dim, device=self.device)
    for index, target_name in enumerate(self._target_names):
      if target_name not in gain:
        raise ValueError(f"No gain value provided for joint {target_name!r}.")
      values[index] = float(gain[target_name])
    return values.unsqueeze(0).repeat(self.num_envs, 1)

  def _xml_motor_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actuators = self._entity.data.indexing.actuators
    if actuators is None:
      raise ValueError("JointPositionToMotorEffortAction requires actuators.")

    actuator_by_target = {
      _strip_namespace(actuator.target): actuator for actuator in actuators
    }

    gear = torch.zeros(self.action_dim, device=self.device)
    ctrl_min = torch.zeros(self.action_dim, device=self.device)
    ctrl_max = torch.zeros(self.action_dim, device=self.device)
    for index, target_name in enumerate(self._target_names):
      actuator = actuator_by_target.get(target_name)
      if actuator is None:
        raise ValueError(f"No XML motor actuator found for joint {target_name!r}.")
      gear[index] = float(actuator.gear[0])
      ctrl_min[index] = float(actuator.ctrlrange[0])
      ctrl_max[index] = float(actuator.ctrlrange[1])

    return (
      gear.unsqueeze(0).repeat(self.num_envs, 1),
      ctrl_min.unsqueeze(0).repeat(self.num_envs, 1),
      ctrl_max.unsqueeze(0).repeat(self.num_envs, 1),
    )


def _strip_namespace(name: str) -> str:
  return name.split("/")[-1]
