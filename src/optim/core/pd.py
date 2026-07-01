from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import PDConfig
from .model import ModelContext, joint_positions, joint_velocities


@dataclass(frozen=True)
class ControlStep:
  q: np.ndarray
  qd: np.ndarray
  q_des: np.ndarray
  tau_joint_cmd: np.ndarray
  tau_motor_cmd: np.ndarray
  tau_motor: np.ndarray
  tau_joint: np.ndarray
  motor_speed: np.ndarray


class PDController:
  def __init__(self, context: ModelContext, cfg: PDConfig) -> None:
    self._context = context
    self._kp = cfg.kp
    self._kd = cfg.kd
    self._gear = np.array([item.gear for item in context.actuators])
    self._ctrl_min = np.array([item.ctrl_min for item in context.actuators])
    self._ctrl_max = np.array([item.ctrl_max for item in context.actuators])

  def apply(self, data: mujoco.MjData, q_des: np.ndarray) -> ControlStep:
    q = joint_positions(data, self._context)
    qd = joint_velocities(data, self._context)
    tau_joint_cmd = self._kp * (q_des - q) - self._kd * qd
    tau_motor_cmd = tau_joint_cmd / self._gear
    tau_motor = np.clip(tau_motor_cmd, self._ctrl_min, self._ctrl_max)
    tau_joint = tau_motor * self._gear

    for index, actuator in enumerate(self._context.actuators):
      data.ctrl[actuator.actuator_id] = tau_motor[index]

    return ControlStep(
      q=q,
      qd=qd,
      q_des=q_des.copy(),
      tau_joint_cmd=tau_joint_cmd,
      tau_motor_cmd=tau_motor_cmd,
      tau_motor=tau_motor,
      tau_joint=tau_joint,
      motor_speed=self._gear * qd,
    )

