from __future__ import annotations

import mujoco

from mj_biped.tasks.biped_2d.biped_2d_env_cfg import biped_2d_env_cfg
from mj_biped.tasks.biped_2d.robot_2d_env_cfg import robot_2d_env_cfg
from mj_biped.tasks.tinker.tinker_env_cfg import tinker_env_cfg


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
  assert set(cfg.metrics) == {"max_forward_speed"}
  assert cfg.metrics["max_forward_speed"].reduce == "last"
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
  assert cfg.metrics == {}
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


def test_tinker_train_cfg_tracks_commands_without_speed_metric() -> None:
  cfg = tinker_env_cfg(num_envs=6)

  assert cfg.scene.num_envs == 6
  assert cfg.observations["actor"].enable_corruption is True
  assert set(cfg.actions) == {"joint_pos"}
  assert set(cfg.commands) == {"velocity", "gait"}
  assert set(cfg.rewards) == {
    "track_linear_velocity",
    "track_angular_velocity",
    "posture",
    "swing_foot_force",
    "stance_foot_velocity",
    "feet_air_time",
    "soft_landing",
    "excessive_foot_force",
    "base_vertical_velocity",
    "joint_torques",
    "joint_velocity",
    "flat_orientation",
    "joint_limits",
    "action_rate",
    "action_acceleration",
  }
  assert cfg.metrics == {}
  assert set(cfg.terminations) == {
    "time_out",
    "root_height",
    "bad_orientation",
    "illegal_ground_contact",
  }
  assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02


def test_tinker_play_cfg_disables_corruption_and_uses_play_env_count() -> None:
  cfg = tinker_env_cfg(play=True, num_envs=6, play_num_envs=2)

  assert cfg.scene.num_envs == 2
  assert cfg.episode_length_s == 1e10
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.viewer.entity_name == "tinker"
  assert cfg.viewer.body_name == "base_link"


def test_tinker_model_has_consistent_limited_position_actuators() -> None:
  entity = tinker_env_cfg(num_envs=1).scene.entities["tinker"].build()
  model = entity.compile()

  assert "floor" not in entity.geom_names
  assert entity.body_names[0] == "base_link"
  assert set(entity.site_names) >= {"left_foot", "right_foot"}
  assert {
    "base_collision",
    "left_foot_collision",
    "right_foot_collision",
  } <= set(entity.geom_names)

  for actuator_id in range(model.nu):
    joint_id = model.actuator_trnid[actuator_id, 0]
    assert model.actuator_forcelimited[actuator_id]
    assert model.actuator_forcerange[actuator_id].tolist() == [-12.0, 12.0]
    assert (
      model.actuator_ctrlrange[actuator_id] == model.jnt_range[joint_id]
    ).all()

  assert mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "joint_r_pitch",
  ) >= 0
