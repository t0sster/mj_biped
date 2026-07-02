from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optim.core.config import ACTUATED_JOINT_NAMES
from optim.core.model import ModelContext
from optim.tasks.squat_stand import _interpolate_pose, _smooth_step
from optim.tasks.stand import STAND_POSES, StandTask


JUMP_VERTICAL_POSES = {
  "crouch": dict(STAND_POSES["crouch"]),
  "extend": {
    "JL0_hip_pitch": 0.9,
    "JL1_hip_roll": 0.0,
    "JL2_thigh_yaw": 0.0,
    "JL3_knee_pitch": -2.0,
    "JL4_ankle_pitch": -1.1,

    "JR0_hip_pitch": -0.9,
    "JR1_hip_roll": 0.0,
    "JR2_thigh_yaw": 0.0,
    "JR3_knee_pitch": 2.0,
    "JR4_ankle_pitch": 1.1,
  },
  "flight": {
    "JL0_hip_pitch": 0.2,
    "JL1_hip_roll": 0.0,
    "JL2_thigh_yaw": 0.0,
    "JL3_knee_pitch": -0.4,
    "JL4_ankle_pitch": -0.1,

    "JR0_hip_pitch": -0.2,
    "JR1_hip_roll": 0.0,
    "JR2_thigh_yaw": 0.0,
    "JR3_knee_pitch": 0.4,
    "JR4_ankle_pitch": 0.1,
  },
}


@dataclass(frozen=True)
class JumpVerticalTiming:
  crouch_hold: float = 0.1
  push: float = 0.1
  extend_hold: float = 0.2
  flight_pose_transition: float = 0.2
  final_hold: float = 0.5

  @property
  def total(self) -> float:
    return (
      self.crouch_hold
      + self.push
      + self.extend_hold
      + self.flight_pose_transition
      + self.final_hold
    )


@dataclass(frozen=True)
class JumpVerticalTask:
  poses: dict[str, dict[str, float]] = field(default_factory=lambda: JUMP_VERTICAL_POSES)
  timing: JumpVerticalTiming = field(default_factory=JumpVerticalTiming)
  base_clearance: float = 5.0e-03

  @property
  def duration(self) -> float:
    return self.timing.total

  def initial_qpos(self, context: ModelContext) -> np.ndarray:
    return StandTask(
      joint_targets=self.poses["crouch"],
      base_clearance=self.base_clearance,
    ).initial_qpos(context)

  def desired_joint_positions(self, time: float, context: ModelContext) -> np.ndarray:
    pose = self._target_pose(time)
    q_des = np.zeros(len(ACTUATED_JOINT_NAMES))
    for joint_name, value in pose.items():
      q_des[context.joint_name_to_index[joint_name]] = value
    return q_des

  def _target_pose(self, time: float) -> dict[str, float]:
    crouch = self.poses["crouch"]
    extend = self.poses["extend"]
    flight = self.poses["flight"]

    t0 = self.timing.crouch_hold
    t1 = t0 + self.timing.push
    t2 = t1 + self.timing.extend_hold
    t3 = t2 + self.timing.flight_pose_transition

    if time < t0:
      return crouch
    if time < t1:
      phase = (time - t0) / self.timing.push
      return _interpolate_pose(crouch, extend, _smooth_step(phase))
    if time < t2:
      return extend
    if time < t3:
      phase = (time - t2) / self.timing.flight_pose_transition
      return _interpolate_pose(extend, flight, _smooth_step(phase))
    return flight
