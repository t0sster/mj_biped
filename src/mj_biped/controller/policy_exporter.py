from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="policy_exporter",
    description="Export a trained mjlab policy checkpoint to ONNX.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--path",
    default=".",
    help=(
      "Checkpoint path under logs/rsl_rl, or a run directory. "
      "If a directory is given, the latest model_*.pt inside it is exported."
    ),
  )
  parser.add_argument(
    "--full-path",
    "--full_path",
    dest="full_path",
    action="store_true",
    help="Treat --path as an exact filesystem path instead of prefixing logs/rsl_rl.",
  )
  return parser


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
  stem = path.stem
  try:
    iteration = int(stem.split("_")[-1])
  except ValueError:
    iteration = -1
  return (iteration, path.stat().st_mtime, path.name)


def _resolve_checkpoint_path(raw_path: str, full_path: bool) -> Path:
  candidate = Path(raw_path).expanduser()
  if not full_path:
    candidate = Path("logs/rsl_rl") / candidate

  if candidate.is_file():
    return candidate.resolve()

  if candidate.is_dir():
    local_checkpoints = sorted(candidate.glob("model_*.pt"), key=_checkpoint_sort_key)
    if local_checkpoints:
      return local_checkpoints[-1].resolve()

    from mjlab.utils.os import get_checkpoint_path

    return get_checkpoint_path(candidate, run_dir=".*", checkpoint=r"model_.*\.pt")

  raise FileNotFoundError(
    f"Checkpoint path does not exist or is not a recognized directory: {candidate}"
  )


def _infer_task_id(checkpoint_path: Path) -> str:
  parts = {part.lower() for part in checkpoint_path.parts}
  if "robot_2d" in parts:
    return "Mjlab-Robot-2D"
  if "biped_2d" in parts:
    return "Mjlab-Biped-2D"
  raise ValueError(
    "Could not infer task from checkpoint path. "
    "Expected path segments containing 'robot_2d' or 'biped_2d'."
  )


def run_export(checkpoint_path: Path) -> Path:
  import mj_biped.tasks  # noqa: F401

  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  import torch

  configure_torch_backends()

  task_id = _infer_task_id(checkpoint_path)
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )

    export_dir = checkpoint_path.parent
    export_name = checkpoint_path.with_suffix(".onnx").name
    runner.export_policy_to_onnx(str(export_dir), export_name)
    return export_dir / export_name
  finally:
    env.close()


def main(argv: list[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)

  checkpoint_path = _resolve_checkpoint_path(args.path, args.full_path)
  exported_path = run_export(checkpoint_path)
  print(f"Exported ONNX policy: {exported_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
