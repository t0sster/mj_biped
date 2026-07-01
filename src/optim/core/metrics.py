from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import mujoco
import numpy as np

from .model import ModelContext
from .pd import ControlStep


@dataclass(frozen=True)
class MetricSummary:
  actuator: str
  joint: str
  max_abs_motor_torque: float
  rms_motor_torque: float
  max_abs_joint_torque: float
  rms_joint_torque: float
  max_abs_motor_speed: float
  max_abs_power: float
  saturation_fraction: float
  rms_tracking_error: float


class MotorMetricsLogger:
  def __init__(self, context: ModelContext) -> None:
    self._context = context
    n = len(context.actuators)
    self._count = 0
    self._max_abs_motor_torque = np.zeros(n)
    self._sum_sq_motor_torque = np.zeros(n)
    self._max_abs_joint_torque = np.zeros(n)
    self._sum_sq_joint_torque = np.zeros(n)
    self._max_abs_motor_speed = np.zeros(n)
    self._max_abs_power = np.zeros(n)
    self._saturation_count = np.zeros(n)
    self._sum_sq_tracking_error = np.zeros(n)
    self._samples: list[dict[str, float | str | bool]] = []

  def record(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control: ControlStep,
  ) -> None:
    del model
    power = control.tau_joint * control.qd
    ctrl_min = np.array([item.ctrl_min for item in self._context.actuators])
    ctrl_max = np.array([item.ctrl_max for item in self._context.actuators])
    limit = np.maximum(np.abs(ctrl_min), np.abs(ctrl_max))
    saturated = np.abs(control.tau_motor_cmd) >= (0.999 * limit)

    self._count += 1
    self._max_abs_motor_torque = np.maximum(
      self._max_abs_motor_torque,
      np.abs(control.tau_motor),
    )
    self._sum_sq_motor_torque += control.tau_motor**2
    self._max_abs_joint_torque = np.maximum(
      self._max_abs_joint_torque,
      np.abs(control.tau_joint),
    )
    self._sum_sq_joint_torque += control.tau_joint**2
    self._max_abs_motor_speed = np.maximum(
      self._max_abs_motor_speed,
      np.abs(control.motor_speed),
    )
    self._max_abs_power = np.maximum(self._max_abs_power, np.abs(power))
    self._saturation_count += saturated
    self._sum_sq_tracking_error += (control.q_des - control.q) ** 2
    self._record_samples(data, control, saturated)

    if not np.all(np.isfinite(data.qpos)):
      raise FloatingPointError("Simulation produced non-finite qpos values.")

  def summary(self) -> list[MetricSummary]:
    if self._count == 0:
      return []

    rms_motor_torque = np.sqrt(self._sum_sq_motor_torque / self._count)
    rms_joint_torque = np.sqrt(self._sum_sq_joint_torque / self._count)
    rms_tracking_error = np.sqrt(self._sum_sq_tracking_error / self._count)

    return [
      MetricSummary(
        actuator=item.name,
        joint=item.joint_name,
        max_abs_motor_torque=float(self._max_abs_motor_torque[index]),
        rms_motor_torque=float(rms_motor_torque[index]),
        max_abs_joint_torque=float(self._max_abs_joint_torque[index]),
        rms_joint_torque=float(rms_joint_torque[index]),
        max_abs_motor_speed=float(self._max_abs_motor_speed[index]),
        max_abs_power=float(self._max_abs_power[index]),
        saturation_fraction=float(self._saturation_count[index] / self._count),
        rms_tracking_error=float(rms_tracking_error[index]),
      )
      for index, item in enumerate(self._context.actuators)
    ]

  def write_csv(self, path: str | Path) -> None:
    rows = self.summary()
    with Path(path).open("w", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=MetricSummary.__dataclass_fields__)
      writer.writeheader()
      for row in rows:
        writer.writerow(row.__dict__)

  def write_timeseries_csv(self, path: str | Path) -> None:
    fieldnames = (
      "time",
      "actuator",
      "joint",
      "tau_motor_cmd",
      "tau_motor",
      "tau_joint",
      "motor_speed",
      "joint_velocity",
      "power",
      "q",
      "q_des",
      "q_error",
      "saturated",
    )
    with Path(path).open("w", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(self._samples)

  def _record_samples(
    self,
    data: mujoco.MjData,
    control: ControlStep,
    saturated: np.ndarray,
  ) -> None:
    power = control.tau_joint * control.qd
    for index, actuator in enumerate(self._context.actuators):
      self._samples.append({
        "time": float(data.time),
        "actuator": actuator.name,
        "joint": actuator.joint_name,
        "tau_motor_cmd": float(control.tau_motor_cmd[index]),
        "tau_motor": float(control.tau_motor[index]),
        "tau_joint": float(control.tau_joint[index]),
        "motor_speed": float(control.motor_speed[index]),
        "joint_velocity": float(control.qd[index]),
        "power": float(power[index]),
        "q": float(control.q[index]),
        "q_des": float(control.q_des[index]),
        "q_error": float(control.q_des[index] - control.q[index]),
        "saturated": bool(saturated[index]),
      })
