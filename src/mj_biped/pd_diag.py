from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Literal

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import VerbosityLevel

from mj_biped.tasks.bd_lip.bd_lip_env_cfg import bd_lip_env_cfg

JOINT_NAMES = (
  "J_L0",
  "J_R0",
  "J_L1",
  "J_R1",
  "J_L2",
  "J_R2",
  "J_L3",
  "J_R3",
  "J_L4_ankle",
  "J_R4_ankle",
)

JOINT_SIGNS = {
  "J_L0": 1.0,
  "J_R0": 1.0,
  "J_L1": 1.0,
  "J_R1": 1.0,
  "J_L2": 1.0,
  "J_R2": 1.0,
  "J_L3": 1.0,
  "J_R3": 1.0,
  "J_L4_ankle": 1.0,
  "J_R4_ankle": -1.0,
}


@dataclass
class PDDiagConfig:
  viewer: Literal["auto", "native", "viser"] = "auto"
  device: str | None = None
  num_envs: int = 1
  profile: Literal["hold", "sine", "step"] = "step"
  frequency: float = 0.7
  amplitude: float = 0.6
  action_scale: float = 0.25
  raw_action_limit: float | None = None
  stiffness: float | None = None
  damping: float | None = None
  effort_limit: float | None = None
  no_terminations: bool = True
  log_every: int = 50
  verbosity: Literal["silent", "info", "debug"] = "info"


class OpenLoopJointPolicy:
  def __init__(self, cfg: PDDiagConfig, env: ManagerBasedRlEnv) -> None:
    self.cfg = cfg
    self.env = env
    self.step_idx = 0
    self._action_ids = {
      "J_L0": 0,
      "J_R0": 1,
      "J_L1": 2,
      "J_R1": 3,
      "J_L2": 4,
      "J_R2": 5,
      "J_L3": 6,
      "J_R3": 7,
      "J_L4_ankle": 8,
      "J_R4_ankle": 9,
    }
    self._joint_cfg = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)

  def reset(self) -> None:
    self.step_idx = 0

  def bind(self) -> None:
    self._joint_cfg.resolve(self.env.scene)

  def _make_actions(self) -> torch.Tensor:
    actions = torch.zeros(
      (self.env.num_envs, len(JOINT_NAMES)), device=self.env.device
    )

    if self.cfg.profile == "hold":
      return actions

    phase = self.step_idx * self.env.step_dt * self.cfg.frequency * 2.0 * math.pi
    s = math.sin(phase)
    left = max(0.0, s)
    right = max(0.0, -s)

    if self.cfg.profile == "sine":
      left = s
      right = -s

    for name in JOINT_NAMES:
      if name.startswith("J_L"):
        actions[:, self._action_ids[name]] = (
          JOINT_SIGNS[name] * self.cfg.amplitude * left
        )
      else:
        actions[:, self._action_ids[name]] = (
          JOINT_SIGNS[name] * self.cfg.amplitude * right
        )
    if self.cfg.raw_action_limit is not None:
      actions = torch.clamp(
        actions, -self.cfg.raw_action_limit, self.cfg.raw_action_limit
      )
    return actions

  def _log_state(self, actions: torch.Tensor) -> None:
    if self.cfg.log_every <= 0 or self.step_idx % self.cfg.log_every != 0:
      return

    robot = self.env.scene["robot"]
    joint_pos = robot.data.joint_pos[0, self._joint_cfg.joint_ids].detach().cpu()
    joint_vel = robot.data.joint_vel[0, self._joint_cfg.joint_ids].detach().cpu()
    joint_torque = robot.data.qfrc_actuator[0, self._joint_cfg.joint_ids].detach().cpu()
    root_pos = robot.data.root_link_pos_w[0].detach().cpu()
    root_lin_vel = robot.data.root_link_lin_vel_b[0].detach().cpu()
    root_ang_vel = robot.data.root_link_ang_vel_b[0].detach().cpu()

    names = ", ".join(
      f"{name}={joint_pos[self._action_ids[name]].item():+.3f}"
      for name in ("J_L2", "J_L3", "J_L4_ankle", "J_R2", "J_R3", "J_R4_ankle")
    )
    torques = ", ".join(
      f"{name}={joint_torque[self._action_ids[name]].item():+.2f}"
      for name in ("J_L2", "J_L3", "J_L4_ankle", "J_R2", "J_R3", "J_R4_ankle")
    )
    vels = ", ".join(
      f"{name}={joint_vel[self._action_ids[name]].item():+.3f}"
      for name in ("J_L2", "J_L3", "J_L4_ankle", "J_R2", "J_R3", "J_R4_ankle")
    )
    action_str = ", ".join(
      f"{name}={actions[0, self._action_ids[name]].item():+.3f}"
      for name in ("J_L2", "J_L3", "J_L4_ankle", "J_R2", "J_R3", "J_R4_ankle")
    )
    print(
      f"[PD-DIAG] step={self.step_idx:05d} "
      f"root_z={root_pos[2].item():+.3f} "
      f"lin_b=({root_lin_vel[0].item():+.3f}, {root_lin_vel[1].item():+.3f}, {root_lin_vel[2].item():+.3f}) "
      f"ang_b=({root_ang_vel[0].item():+.3f}, {root_ang_vel[1].item():+.3f}, {root_ang_vel[2].item():+.3f})"
    )
    print(f"[PD-DIAG]   actions: {action_str}")
    print(f"[PD-DIAG]   joints : {names}")
    print(f"[PD-DIAG]   vels   : {vels}")
    print(f"[PD-DIAG]   torques: {torques}")

  def __call__(self, obs) -> torch.Tensor:
    del obs
    actions = self._make_actions()
    self._log_state(actions)
    self.step_idx += 1
    return actions

'''uv run mj-biped-pd-diag --viewer viser'''
def _parse_args() -> PDDiagConfig:
  parser = argparse.ArgumentParser(
    prog="mj-biped pd-diag",
    description="Open-loop diagnostic viewer for BD-Lip PD tuning.",
  )
  parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  parser.add_argument("--device", default=None)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--profile", choices=("hold", "sine", "step"), default="sine")
  parser.add_argument("--frequency", type=float, default=0.7)
  parser.add_argument("--amplitude", type=float, default=0.6)
  parser.add_argument("--action-scale", type=float, default=0.25)
  parser.add_argument("--raw-action-limit", type=float, default=None)
  parser.add_argument("--stiffness", type=float, default=None)
  parser.add_argument("--damping", type=float, default=None)
  parser.add_argument("--effort-limit", type=float, default=20)
  parser.add_argument("--log-every", type=int, default=50)
  parser.add_argument("--allow-terminations", action="store_true")
  parser.add_argument("--verbosity", choices=("silent", "info", "debug"), default="info")
  ns = parser.parse_args()
  return PDDiagConfig(
    viewer=ns.viewer,
    device=ns.device,
    num_envs=ns.num_envs,
    profile=ns.profile,
    frequency=ns.frequency,
    amplitude=ns.amplitude,
    action_scale=ns.action_scale,
    raw_action_limit=ns.raw_action_limit,
    stiffness=ns.stiffness,
    damping=ns.damping,
    effort_limit=ns.effort_limit,
    no_terminations=not ns.allow_terminations,
    log_every=ns.log_every,
    verbosity=ns.verbosity,
  )


def _build_env(cfg: PDDiagConfig) -> ManagerBasedRlEnv:
  env_cfg = bd_lip_env_cfg(play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.actions["joint_pos"].scale = cfg.action_scale

  if cfg.stiffness is not None or cfg.damping is not None or cfg.effort_limit is not None:
    for actuator_cfg in env_cfg.scene.entities["robot"].articulation.actuators:
      if cfg.stiffness is not None:
        actuator_cfg.stiffness = cfg.stiffness
      if cfg.damping is not None:
        actuator_cfg.damping = cfg.damping
      if cfg.effort_limit is not None:
        actuator_cfg.effort_limit = cfg.effort_limit

  if cfg.no_terminations:
    env_cfg.terminations = {}

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def main() -> None:
  cfg = _parse_args()
  env = _build_env(cfg)
  policy = OpenLoopJointPolicy(cfg, env)
  policy.bind()

  try:
    verbosity = {
      "silent": VerbosityLevel.SILENT,
      "info": VerbosityLevel.INFO,
      "debug": VerbosityLevel.DEBUG,
    }[cfg.verbosity]

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = cfg.viewer
    if resolved_viewer == "auto":
      resolved_viewer = "native" if has_display else "viser"

    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy, verbosity=verbosity).run()
    else:
      ViserPlayViewer(env, policy, verbosity=verbosity).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
