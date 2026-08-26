import time

import numpy as np
from scipy.interpolate import CubicHermiteSpline

import rclpy
from rclpy.node import Node

from tinker_msgs.msg import ControlCmd, LowCmd, LowState, MotorCmd

from biped_hardware.hardware_node import MOTOR_IDS

# Целевая поза, рад — индекс совпадает с MOTOR_IDS (сначала левая нога, потом правая).
INIT_POSE = [0.0, 0.0, 0.45, 0.9, 0.45, 0.0, 0.0, -0.45, -0.9, -0.45]

KP = 15.0
KD = 0.6

MOVE_DURATION_SEC = 3.0     # длительность плавного перехода
SETTLE_SEC = 0.5            # сколько ещё держим целевую позу после перехода
CONTROL_RATE_HZ = 50.0      # частота публикации LowCmd
FEEDBACK_WAIT_SEC = 2.0     # сколько ждём стартовый фидбек от hardware_node


class InitPoseNode(Node):

    def __init__(self):
        super().__init__("init_pose_node")
        self.control_pub = self.create_publisher(ControlCmd, "/control_command", 10)
        self.cmd_pub = self.create_publisher(LowCmd, "/low_level_cmd", 10)
        self.create_subscription(LowState, "/low_level_state_real", self._on_state, 10)
        self.last_state = None

    def _on_state(self, msg: LowState) -> None:
        self.last_state = msg

    def _tick(self, seconds: float) -> None:
        """Даёт ROS обработать входящие сообщения и ждёт заданное время."""
        rclpy.spin_once(self, timeout_sec=0.0)
        time.sleep(seconds)

    def _wait_for_feedback(self) -> bool:
        deadline = time.time() + FEEDBACK_WAIT_SEC
        while time.time() < deadline:
            self._tick(0.02)
            if self.last_state is not None:
                return True
        return False

    def _enable_motors(self) -> None:
        for motor_id in MOTOR_IDS:
            msg = ControlCmd()
            msg.motor_id = motor_id
            msg.cmd = ControlCmd.ENABLE
            self.control_pub.publish(msg)

    def _publish_cmd(self, positions, velocities) -> None:
        msg = LowCmd()
        for index in range(len(MOTOR_IDS)):
            motor_cmd = MotorCmd()
            motor_cmd.position = float(positions[index])
            motor_cmd.velocity = float(velocities[index])
            motor_cmd.kp = KP
            motor_cmd.kd = KD
            motor_cmd.torque = 0.0
            msg.motor_cmd[index] = motor_cmd
        self.cmd_pub.publish(msg)

    def run(self) -> None:
        self.get_logger().info("жду стартовый фидбек от hardware_node")
        if not self._wait_for_feedback():
            self.get_logger().error("нет фидбека от hardware_node, прерываю")
            return

        start = np.array(
            [self.last_state.motor_state[i].position for i in range(len(MOTOR_IDS))]
        )
        target = np.array(INIT_POSE)

        self.get_logger().info("включаю моторы")
        self._enable_motors()
        self._tick(0.3)

        # плавная траектория: кубический эрмитов сплайн с нулевой скоростью
        # на концах, одна траектория сразу на все 10 суставов
        x = [0.0, MOVE_DURATION_SEC]
        y = np.stack([start, target])
        dydx = np.zeros_like(y)
        pos_spline = CubicHermiteSpline(x, y, dydx, axis=0)
        vel_spline = pos_spline.derivative()

        self.get_logger().info(f"перевожу в начальную позу за {MOVE_DURATION_SEC:.1f} с")
        period = 1.0 / CONTROL_RATE_HZ
        t_start = time.time()
        while True:
            t = time.time() - t_start
            if t >= MOVE_DURATION_SEC:
                break
            self._publish_cmd(pos_spline(t), vel_spline(t))
            self._tick(period)

        # держим целевую позу ещё немного, чтобы она точно устоялась
        settle_deadline = time.time() + SETTLE_SEC
        zero_velocity = np.zeros_like(target)
        while time.time() < settle_deadline:
            self._publish_cmd(target, zero_velocity)
            self._tick(period)

        self.get_logger().info("начальная поза достигнута")


def main(args=None):
    rclpy.init(args=args)
    node = InitPoseNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
