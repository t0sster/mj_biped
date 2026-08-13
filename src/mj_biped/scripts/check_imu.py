"""Live-print imu_rpy/imu_gyro/imu_accel while holding the default pose.

For calibrating the imu site's pos/quat in tinker_range.xml against the real
sensor: opens the native viewer (mouse-drag perturbation enabled) so you can
tilt/rotate the robot by hand while comparing the printed numbers against the
real IMU's telemetry in the same pose.
"""

from __future__ import annotations

import time

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer

from mj_biped.tasks.tinker import mdp
from mj_biped.tasks.tinker.tinker_env_cfg import tinker_env_cfg

_PRINT_INTERVAL_S = 0.5


class _HoldDefaultPosePolicy:
  """Zero-action policy (holds default joint pose); prints imu data as it goes."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    self._imu_cfg = SceneEntityCfg("tinker", site_names=("imu",))
    self._imu_cfg.resolve(env.scene)
    self._last_print = 0.0

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    now = time.monotonic()
    if now - self._last_print >= _PRINT_INTERVAL_S:
      self._last_print = now
      rpy = torch.rad2deg(mdp.imu_rpy(self._env, asset_cfg=self._imu_cfg)[0])
      gyro = torch.rad2deg(
        mdp.builtin_sensor_data(self._env, sensor_name="tinker/imu_gyro")[0]
      )
      accel = mdp.builtin_sensor_data(self._env, sensor_name="tinker/imu_accel")[0]
      print(
        f"rpy(deg)=[{rpy[0]:7.2f} {rpy[1]:7.2f} {rpy[2]:7.2f}]  "
        f"gyro(deg/s)=[{gyro[0]:7.2f} {gyro[1]:7.2f} {gyro[2]:7.2f}]  "
        f"accel(m/s2)=[{accel[0]:6.2f} {accel[1]:6.2f} {accel[2]:6.2f}]"
      )
    return torch.zeros(obs.shape[0], self._env.action_manager.total_action_dim)


def main() -> None:
  cfg = tinker_env_cfg(play=True, play_num_envs=1)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  policy = _HoldDefaultPosePolicy(env)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=None)
  print(
    "Robot held at default pose. Ctrl+right-drag in the viewer to tilt/rotate "
    f"it by hand. Printing imu_rpy/imu_gyro/imu_accel every {_PRINT_INTERVAL_S}s."
  )
  NativeMujocoViewer(wrapped_env, policy).run()


if __name__ == "__main__":
  main()
