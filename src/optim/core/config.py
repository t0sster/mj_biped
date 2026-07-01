from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ACTUATED_JOINT_NAMES = (
  "JL0_hip_pitch",
  "JL1_hip_roll",
  "JL2_thigh_yaw",
  "JL3_knee_pitch",
  "JL4_ankle_pitch",
  "JR0_hip_pitch",
  "JR1_hip_roll",
  "JR2_thigh_yaw",
  "JR3_knee_pitch",
  "JR4_ankle_pitch",
)

ACTUATOR_NAMES = (
  "ML0_hip_pitch",
  "ML1_hip_roll",
  "ML2_thigh_yaw",
  "ML3_knee_pitch",
  "ML4_ankle_pitch",
  "MR0_hip_pitch",
  "MR1_hip_roll",
  "MR2_thigh_yaw",
  "MR3_knee_pitch",
  "MR4_ankle_pitch",
)

DEFAULT_MODEL_PATH = (
  Path(__file__).resolve().parents[1] / "assets" / "bd" / "bd_prev_gear.xml"
)


@dataclass(frozen=True)
class PDConfig:
  kp: float = 30.0
  kd: float = 3.0


@dataclass(frozen=True)
class SimulationConfig:
  model_path: Path = DEFAULT_MODEL_PATH
  duration: float = 2.0
  timestep: float | None = None
  pd: PDConfig = field(default_factory=PDConfig)

