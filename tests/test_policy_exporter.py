from __future__ import annotations

from pathlib import Path

import pytest

from mj_biped.controller.policy_exporter import (
  _infer_task_id,
  _resolve_checkpoint_path,
)


def test_infer_task_id_from_checkpoint_path() -> None:
  assert _infer_task_id(Path("logs/rsl_rl/robot_2d/run/model_350.pt")) == (
    "Mjlab-Robot-2D"
  )
  assert _infer_task_id(Path("logs/rsl_rl/biped_2d/run/model_100.pt")) == (
    "Mjlab-Biped-2D"
  )


def test_infer_task_id_rejects_unknown_path() -> None:
  with pytest.raises(ValueError, match="Could not infer task"):
    _infer_task_id(Path("logs/rsl_rl/unknown/run/model_1.pt"))


def test_resolve_checkpoint_path_accepts_exact_file(tmp_path: Path) -> None:
  checkpoint = tmp_path / "model_12.pt"
  checkpoint.touch()

  assert _resolve_checkpoint_path(str(checkpoint), full_path=True) == (
    checkpoint.resolve()
  )


def test_resolve_checkpoint_path_uses_highest_model_iteration(
  tmp_path: Path,
) -> None:
  run_dir = tmp_path / "robot_2d" / "run"
  run_dir.mkdir(parents=True)
  older_checkpoint = run_dir / "model_2.pt"
  latest_checkpoint = run_dir / "model_100.pt"
  older_checkpoint.touch()
  latest_checkpoint.touch()

  assert _resolve_checkpoint_path(str(run_dir), full_path=True) == (
    latest_checkpoint.resolve()
  )


def test_resolve_checkpoint_path_prefixes_logs_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  checkpoint = tmp_path / "logs" / "rsl_rl" / "robot_2d" / "run" / "model_1.pt"
  checkpoint.parent.mkdir(parents=True)
  checkpoint.touch()

  assert _resolve_checkpoint_path("robot_2d/run/model_1.pt", full_path=False) == (
    checkpoint.resolve()
  )
