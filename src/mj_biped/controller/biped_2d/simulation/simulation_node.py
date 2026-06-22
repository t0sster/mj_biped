import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

OBS_TOPIC = "/biped_2d/observation"
ACTION_TOPIC = "/biped_2d/action"
ACTION_DIM = 4

TIMESTEP = 0.005
DECIMATION = 4
CONTROL_DT = TIMESTEP * DECIMATION

JOINT_NAMES = (
  "root_x", "root_z", "root_pitch",
  "hip_right", "knee_right", "hip_left", "knee_left",
)
ACTUATED = ("hip_right", "knee_right", "hip_left", "knee_left")
DEFAULT_POS = np.array([0.0, 0.5, 0.0, 0.35, -0.5, 0.35, -0.5])  # root_z=0.5 -> база z=1.5
DEFAULT_OFFSET = DEFAULT_POS[3:7]
ACTION_SCALE = 0.25
KP = np.array([60.0, 50.0, 60.0, 50.0])
KD = np.array([1.5, 1.2, 1.5, 1.2])
GAIT_FREQUENCY = 1.0
GAIT_DUTY = 0.5

_SCENE = Path(__file__).resolve().parent / "scene_inference.xml"


class SimulationNode(Node):
  def __init__(self):
    super().__init__("biped_2d_simulation")
    self.model = mujoco.MjModel.from_xml_path(str(_SCENE))
    self.model.opt.timestep = TIMESTEP
    self.data = mujoco.MjData(self.model)

    jid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
    self._q = np.array([self.model.jnt_qposadr[jid(n)] for n in JOINT_NAMES])
    self._v = np.array([self.model.jnt_dofadr[jid(n)] for n in JOINT_NAMES])
    self._aq = np.array([self.model.jnt_qposadr[jid(n)] for n in ACTUATED])
    self._av = np.array([self.model.jnt_dofadr[jid(n)] for n in ACTUATED])

    self._phase = 0.0
    self._action = np.zeros(ACTION_DIM)
    self._got = False

    self._pub = self.create_publisher(Float64MultiArray, OBS_TOPIC, 10)
    self.create_subscription(Float64MultiArray, ACTION_TOPIC, self._on_action, 10)
    self._reset()

  def _on_action(self, msg):
    a = np.asarray(msg.data, dtype=np.float64)
    if a.shape[0] == ACTION_DIM:
      self._action, self._got = a, True

  def _reset(self):
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[self._q] = DEFAULT_POS
    self._phase = 0.0
    mujoco.mj_forward(self.model, self.data)

  def obs(self):
    q, qd = self.data.qpos[self._q], self.data.qvel[self._v]
    right = (self._phase + 0.5) % 1.0
    gait = [self._phase, right, float(self._phase < GAIT_DUTY), float(right < GAIT_DUTY), GAIT_FREQUENCY]
    return np.concatenate([
      [qd[0], 0.0, qd[1]],
      [0.0, qd[2], 0.0],
      q - DEFAULT_POS,
      qd,
      gait,
    ])

  def step(self, action):
    target = DEFAULT_OFFSET + ACTION_SCALE * action
    for _ in range(DECIMATION):
      # PD без клипа момента: в модели mjlab forcelimited=False.
      tau = KP * (target - self.data.qpos[self._aq]) - KD * self.data.qvel[self._av]
      self.data.qfrc_applied[self._av] = tau
      mujoco.mj_step(self.model, self.data)
    self._phase = (self._phase + CONTROL_DT * GAIT_FREQUENCY) % 1.0

  def wait_action(self, timeout=0.1):
    self._got = False
    deadline = time.perf_counter() + timeout
    while rclpy.ok() and not self._got and time.perf_counter() < deadline:
      rclpy.spin_once(self, timeout_sec=0.01)
    return self._action


def run(node, headless):
  viewer = None
  if not headless:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(node.model, node.data)
  try:
    while rclpy.ok() and (viewer is None or viewer.is_running()):
      t0 = time.perf_counter()
      node._pub.publish(Float64MultiArray(data=node.obs().tolist()))
      node.step(node.wait_action())
      if viewer is not None:
        viewer.sync()
      dt = CONTROL_DT - (time.perf_counter() - t0)
      if dt > 0:
        time.sleep(dt)
  finally:
    if viewer is not None:
      viewer.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--headless", action="store_true")
  args = parser.parse_args()

  rclpy.init()
  node = SimulationNode()
  try:
    run(node, args.headless)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()