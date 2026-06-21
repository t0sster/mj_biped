from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

OBS_TOPIC = "/biped_2d/observation"
ACTION_TOPIC = "/biped_2d/action"
ACTION_DIM = 4

_POLICY = Path(__file__).resolve().parents[1] / "policy" / "model_350.onnx"


class InferenceNode(Node):
  def __init__(self):
    super().__init__("biped_2d_inference")
    self._session = ort.InferenceSession(str(_POLICY), providers=["CPUExecutionProvider"])
    self._input = self._session.get_inputs()[0].name
    self._pub = self.create_publisher(Float64MultiArray, ACTION_TOPIC, 10)
    self.create_subscription(Float64MultiArray, OBS_TOPIC, self._on_obs, 10)

  def _on_obs(self, msg):
    obs = np.asarray(msg.data, dtype=np.float32)[None, :]
    action = self._session.run(None, {self._input: obs})[0].reshape(-1)[:ACTION_DIM]
    self._pub.publish(Float64MultiArray(data=action.astype(float).tolist()))


def main():
  rclpy.init()
  node = InferenceNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
