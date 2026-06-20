"""Inspect the biped_2d action-to-actuator path for a zero policy command."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mj_biped.tasks.biped_2d.biped_2d_env_cfg import biped_2d_env_cfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402


def _row(name: str, value: float) -> str:
  return f"{name:>20s}: {value: .6f}"


def _tensor_1d(values: torch.Tensor) -> list[float]:
  return values.detach().cpu().flatten().tolist()


def _print_named_values(title: str, names: list[str] | tuple[str, ...], values: torch.Tensor) -> None:
  print(f"\n{title}")
  for name, value in zip(names, _tensor_1d(values), strict=False):
    print(_row(name, value))


def main() -> None:
  torch.set_printoptions(precision=5, sci_mode=False)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = biped_2d_env_cfg(play=True, play_num_envs=1)
  cfg.seed = 42

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()

  entity = env.scene["biped_2d"]
  term = env.action_manager.get_term("joint_pos")
  target_ids = term.target_ids
  target_names = tuple(term.target_names)

  print(f"device: {device}")
  print(f"action_dim: {env.action_manager.total_action_dim}")
  print(f"joint_names: {entity.joint_names}")
  print(f"actuator_names: {entity.actuator_names}")
  print(f"action target_names: {target_names}")
  print(f"action target_ids: {_tensor_1d(target_ids)}")

  _print_named_values(
    "default_joint_pos[target_ids]",
    target_names,
    entity.data.default_joint_pos[0, target_ids],
  )
  _print_named_values(
    "current joint_pos[target_ids] after reset",
    target_names,
    entity.data.joint_pos[0, target_ids],
  )

  offset = term.offset
  if isinstance(offset, torch.Tensor):
    _print_named_values("action offset", target_names, offset[0])
  else:
    print(f"\naction offset scalar: {offset}")

  zero_action = torch.zeros(
    (1, env.action_manager.total_action_dim),
    device=env.device,
    dtype=torch.float32,
  )
  env.action_manager.process_action(zero_action)
  term = env.action_manager.get_term("joint_pos")

  _print_named_values("processed zero action", target_names, term._processed_actions[0])

  env.action_manager.apply_action()
  _print_named_values(
    "entity joint_pos_target[target_ids] after apply_action",
    target_names,
    entity.data.joint_pos_target[0, target_ids],
  )

  env.scene.write_data_to_sim()
  ctrl_ids = entity.data.indexing.ctrl_ids
  ctrl_values = entity.data.data.ctrl[0, ctrl_ids]
  _print_named_values("mujoco ctrl by entity actuator", entity.actuator_names, ctrl_values)

  print("\nCheck:")
  print("  zero action should equal action offset")
  print("  action offset should equal default_joint_pos[target_ids]")
  print("  *_pd_pos ctrl should equal joint_pos_target; *_pd_vel ctrl should stay near 0")

  env.close()


if __name__ == "__main__":
  main()
