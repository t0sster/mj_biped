#!/usr/bin/env python3
"""
ROS2 Node: zero_motors_node
============================
Разовая нода: включает все 10 моторов, обнуляет им позицию (SET_ZERO)
и по фидбеку от hardware_node проверяет, что позиция реально ушла в ноль.
После проверки выключает моторы и завершается.

Нужен уже запущенный hardware_node. Запуск:
    ros2 run biped_hardware zero_motors_node
"""

import time

import rclpy
from rclpy.node import Node

from tinker_msgs.msg import ControlCmd, LowState, MotorState

from biped_hardware.hardware_node import MOTOR_IDS

ZERO_TOLERANCE_RAD = 0.05   # насколько близко к 0 считаем "обнулилось"
ZERO_TIMEOUT_SEC = 3.0      # сколько ждём подтверждения по фидбеку
SETTLE_SEC = 0.3            # пауза после ENABLE и после SET_ZERO


def motor_is_zeroed(motor_state: MotorState) -> bool:
    if motor_state.error == MotorState.LOSS_CONNECTION:
        return False
    return abs(motor_state.position) < ZERO_TOLERANCE_RAD


def all_motors_zeroed(low_state: LowState) -> bool:
    return all(motor_is_zeroed(low_state.motor_state[i]) for i in range(len(MOTOR_IDS)))


class ZeroMotorsNode(Node):

    def __init__(self):
        super().__init__("zero_motors_node")
        self.control_pub = self.create_publisher(ControlCmd, "/control_command", 10)
        self.create_subscription(LowState, "/low_level_state_real", self._on_state, 10)
        self.last_state = None

    def _on_state(self, msg: LowState) -> None:
        self.last_state = msg

    def _send_control(self, motor_id: int, cmd: int) -> None:
        msg = ControlCmd()
        msg.motor_id = motor_id
        msg.cmd = cmd
        self.control_pub.publish(msg)

    def _spin_for(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self) -> bool:
        self.get_logger().info("включаю моторы")
        for motor_id in MOTOR_IDS:
            self._send_control(motor_id, ControlCmd.ENABLE)
        self._spin_for(SETTLE_SEC)

        self.get_logger().info("отправляю обнуление позиции")
        for motor_id in MOTOR_IDS:
            self._send_control(motor_id, ControlCmd.SET_ZERRO_POSITION)
        self._spin_for(SETTLE_SEC)

        ok = self._wait_until_zeroed()

        self.get_logger().info("выключаю моторы")
        for motor_id in MOTOR_IDS:
            self._send_control(motor_id, ControlCmd.DISABLE)
        self._spin_for(SETTLE_SEC)

        if ok:
            self.get_logger().info("все 10 моторов обнулены")
        else:
            self._log_problem_motors()
        return ok

    def _wait_until_zeroed(self) -> bool:
        deadline = time.time() + ZERO_TIMEOUT_SEC
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.last_state and all_motors_zeroed(self.last_state):
                return True
        return False

    def _log_problem_motors(self) -> None:
        if self.last_state is None:
            self.get_logger().error("нет фидбека от hardware_node")
            return
        for index, motor_id in enumerate(MOTOR_IDS):
            state = self.last_state.motor_state[index]
            if state.error == MotorState.LOSS_CONNECTION:
                self.get_logger().error(f"мотор {motor_id}: нет связи")
            elif abs(state.position) >= ZERO_TOLERANCE_RAD:
                self.get_logger().error(
                    f"мотор {motor_id}: позиция {state.position:+.3f} рад, не обнулился"
                )


def main(args=None):
    rclpy.init(args=args)
    node = ZeroMotorsNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
