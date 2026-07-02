from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from optim.core.config import ACTUATED_JOINT_NAMES
from optim.core.model import ModelContext


STAND_POSES = {
  "neutral": {},
  "crouch": {
    "JL0_hip_pitch": 0.5,
    "JL1_hip_roll": 0.0,
    "JL2_thigh_yaw": 0.0,
    "JL3_knee_pitch": -1.1,
    "JL4_ankle_pitch": -0.6,

    "JR0_hip_pitch": -0.5,
    "JR1_hip_roll": 0.0,
    "JR2_thigh_yaw": 0.0,
    "JR3_knee_pitch": 1.1,
    "JR4_ankle_pitch": 0.6,
  },
}


@dataclass(frozen=True)
class StandTask:
  joint_targets: dict[str, float] = field(default_factory=dict)
  base_clearance: float = 5.0e-03

  def initial_qpos(self, context: ModelContext) -> np.ndarray:
    qpos = np.zeros(context.model.nq)
    # Freejoint qpos is [x, y, z, qw, qx, qy, qz]; qw=1 is identity rotation.
    qpos[3] = 1.0
    for actuator in context.actuators:
      qpos[actuator.qpos_addr] = self.joint_targets.get(actuator.joint_name, 0.0)
    qpos[2] += self._base_height_shift(context, qpos)
    qpos[2] += 0.02
    return qpos

  def desired_joint_positions(self, time: float, context: ModelContext) -> np.ndarray:
    del time
    q_des = np.zeros(len(ACTUATED_JOINT_NAMES))
    for joint_name, value in self.joint_targets.items():
      q_des[context.joint_name_to_index[joint_name]] = value
    return q_des

  def _base_height_shift(self, context: ModelContext, qpos: np.ndarray) -> float:
    model = context.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    floor_height = None
    robot_bottom = None
    for geom_id in range(model.ngeom):
      geom_type = model.geom_type[geom_id]
      if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        height = float(data.geom_xpos[geom_id, 2])
        floor_height = height if floor_height is None else max(floor_height, height)
        continue

      bottom = float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])
      robot_bottom = bottom if robot_bottom is None else min(robot_bottom, bottom)

    if floor_height is None or robot_bottom is None:
      return 0.0
    return floor_height + self.base_clearance - robot_bottom
