from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from optim.core.config import ACTUATED_JOINT_NAMES
from optim.core.model import ModelContext
from optim.tasks.scripted.stand import STAND_POSES, StandTask


SQUAT_STAND_POSES = {
  "stand": {
    "JL0_hip_pitch": 0.785,
    "JL1_hip_roll": 0.0,
    "JL2_thigh_yaw": 0.0,
    "JL3_knee_pitch": -1.57,
    "JL4_ankle_pitch": -0.785,

    "JR0_hip_pitch": -0.785,
    "JR1_hip_roll": 0.0,
    "JR2_thigh_yaw": 0.0,
    "JR3_knee_pitch": 1.57,
    "JR4_ankle_pitch": 0.785,
  },
  "crouch": dict(STAND_POSES["crouch"]),
}


@dataclass(frozen=True)
class SquatStandTiming:
  stand_hold: float = 0.5
  down: float = 1.0
  crouch_hold: float = 1.0
  up: float = 1.0
  final_hold: float = 0.5

  @property
  def total(self) -> float:
    return self.stand_hold + self.down + self.crouch_hold + self.up + self.final_hold


@dataclass(frozen=True)
class SquatStandTask:
  poses: dict[str, dict[str, float]] = field(default_factory=lambda: SQUAT_STAND_POSES)
  timing: SquatStandTiming = field(default_factory=SquatStandTiming)
  base_clearance: float = 5.0e-03

  @property
  def duration(self) -> float:
    return self.timing.total

  def initial_qpos(self, context: ModelContext) -> np.ndarray:
    return StandTask(
      joint_targets=self.poses["stand"],
      base_clearance=self.base_clearance,
    ).initial_qpos(context)

  def desired_joint_positions(self, time: float, context: ModelContext) -> np.ndarray:
    pose = self._target_pose(time)
    q_des = np.zeros(len(ACTUATED_JOINT_NAMES))
    for joint_name, value in pose.items():
      q_des[context.joint_name_to_index[joint_name]] = value
    return q_des

  def _target_pose(self, time: float) -> dict[str, float]:
    stand = self.poses["stand"]
    crouch = self.poses["crouch"]
    t0 = self.timing.stand_hold
    t1 = t0 + self.timing.down
    t2 = t1 + self.timing.crouch_hold
    t3 = t2 + self.timing.up

    if time < t0:
      return stand
    if time < t1:
      phase = (time - t0) / self.timing.down
      return _interpolate_pose(stand, crouch, _smooth_step(phase))
    if time < t2:
      return crouch
    if time < t3:
      phase = (time - t2) / self.timing.up
      return _interpolate_pose(crouch, stand, _smooth_step(phase))
    return stand


def _smooth_step(phase: float) -> float:
  phase = min(1.0, max(0.0, phase))
  return 0.5 - 0.5 * math.cos(math.pi * phase)


def _interpolate_pose(
  start: dict[str, float],
  end: dict[str, float],
  phase: float,
) -> dict[str, float]:
  joint_names = set(start) | set(end)
  return {
    joint_name: (1.0 - phase) * start.get(joint_name, 0.0)
    + phase * end.get(joint_name, 0.0)
    for joint_name in joint_names
  }
