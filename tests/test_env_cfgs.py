from __future__ import annotations

from mj_biped.tasks.biped_2d.biped_2d_env_cfg import biped_2d_env_cfg
from mj_biped.tasks.biped_2d.robot_2d_env_cfg import robot_2d_env_cfg


def test_robot_2d_train_cfg_uses_requested_env_count() -> None:
  cfg = robot_2d_env_cfg(num_envs=7)

  assert cfg.scene.num_envs == 7
  assert cfg.observations["actor"].enable_corruption is True
  assert set(cfg.actions) == {"joint_pos"}
  assert set(cfg.rewards) == {
    "forward_velocity",
    "joint_torque",
    "joint_velocity",
    "contact_schedule",
    "action_acc",
  }
  assert set(cfg.terminations) == {
    "time_out",
    "bad_orientation",
    "base_contact_with_ground",
  }


def test_robot_2d_play_cfg_disables_corruption_and_uses_play_env_count() -> None:
  cfg = robot_2d_env_cfg(play=True, num_envs=7, play_num_envs=2)

  assert cfg.scene.num_envs == 2
  assert cfg.episode_length_s == 1e10
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.viewer.entity_name == "robot_2d"
  assert cfg.viewer.body_name == "base"


def test_biped_2d_train_cfg_uses_requested_env_count() -> None:
  cfg = biped_2d_env_cfg(num_envs=5)

  assert cfg.scene.num_envs == 5
  assert cfg.observations["actor"].enable_corruption is True
  assert set(cfg.actions) == {"joint_pos"}
  assert set(cfg.rewards) == {
    "forward_velocity",
    "joint_torque",
    "joint_velocity",
    "contact_schedule",
    "action_acc",
  }
  assert set(cfg.terminations) == {
    "time_out",
    "bad_orientation",
    "base_contact_with_ground",
  }


def test_biped_2d_play_cfg_disables_corruption_and_uses_play_env_count() -> None:
  cfg = biped_2d_env_cfg(play=True, num_envs=5, play_num_envs=1)

  assert cfg.scene.num_envs == 1
  assert cfg.episode_length_s == 1e10
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.viewer.entity_name == "biped_2d"
  assert cfg.viewer.body_name == "pelvis"
