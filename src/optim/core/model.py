from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import ACTUATED_JOINT_NAMES, ACTUATOR_NAMES


@dataclass(frozen=True)
class ActuatorInfo:
  name: str
  joint_name: str
  actuator_id: int
  joint_id: int
  qpos_addr: int
  qvel_addr: int
  gear: float
  ctrl_min: float
  ctrl_max: float


@dataclass(frozen=True)
class ModelContext:
  model: mujoco.MjModel
  actuators: tuple[ActuatorInfo, ...]
  joint_name_to_index: dict[str, int]


def load_model(path: str | Path, timestep: float | None = None) -> ModelContext:
  model = mujoco.MjModel.from_xml_path(str(path))
  if timestep is not None:
    model.opt.timestep = timestep
  actuators = tuple(_actuator_info(model, actuator, joint) for actuator, joint in zip(
    ACTUATOR_NAMES,
    ACTUATED_JOINT_NAMES,
    strict=True,
  ))
  return ModelContext(
    model=model,
    actuators=actuators,
    joint_name_to_index={
      actuator.joint_name: index for index, actuator in enumerate(actuators)
    },
  )


def new_data(context: ModelContext) -> mujoco.MjData:
  data = mujoco.MjData(context.model)
  mujoco.mj_forward(context.model, data)
  return data


def joint_positions(data: mujoco.MjData, context: ModelContext) -> np.ndarray:
  return np.array([data.qpos[item.qpos_addr] for item in context.actuators])


def joint_velocities(data: mujoco.MjData, context: ModelContext) -> np.ndarray:
  return np.array([data.qvel[item.qvel_addr] for item in context.actuators])


def _actuator_info(
  model: mujoco.MjModel,
  actuator_name: str,
  joint_name: str,
) -> ActuatorInfo:
  actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    actuator_name,
  )
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if actuator_id < 0:
    raise ValueError(f"Unknown actuator: {actuator_name}")
  if joint_id < 0:
    raise ValueError(f"Unknown joint: {joint_name}")

  gear = float(model.actuator_gear[actuator_id, 0])
  ctrlrange = model.actuator_ctrlrange[actuator_id]
  return ActuatorInfo(
    name=actuator_name,
    joint_name=joint_name,
    actuator_id=actuator_id,
    joint_id=joint_id,
    qpos_addr=int(model.jnt_qposadr[joint_id]),
    qvel_addr=int(model.jnt_dofadr[joint_id]),
    gear=gear,
    ctrl_min=float(ctrlrange[0]),
    ctrl_max=float(ctrlrange[1]),
  )

