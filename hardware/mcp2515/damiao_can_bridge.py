"""
ROS2 Node: damiao_can_bridge
============================
Мост ROS2 ↔ CAN для моторов Damiao.

Подписки:
  /low_cmd      (LowCmd)      — команды MIT mode для всех моторов
  /control_cmd  (ControlCmd)  — enable / disable / set_zero / clear_error

Публикации:
  /low_state    (LowState)    — состояние всех моторов

Замените 'your_msgs' на имя вашего ROS2 пакета с сообщениями.
"""

import time
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time

from your_msgs.msg import (   # ← замените your_msgs
    LowCmd,
    LowState,
    MotorState as RosMotorState,
    ControlCmd,
)

from damiao_can import DamiaoMotorBus, MotorState, MOTOR_IDS

MAX_MOTORS = 10
PUBLISH_RATE_HZ = 100.0   # частота публикации /low_state


class DamiaoCanBridge(Node):

    def __init__(self):
        super().__init__('damiao_can_bridge')

        self._bus = DamiaoMotorBus(rx_callback=self._on_motor_feedback)
        self._rx_tick: int = 0

        # ── Подписки ──────────────────────────────────────────────────
        self.create_subscription(LowCmd,     '/low_cmd',     self._cb_low_cmd,     10)
        self.create_subscription(ControlCmd, '/control_cmd', self._cb_control_cmd, 10)

        # ── Публикации ─────────────────────────────────────────────────
        self._pub_state = self.create_publisher(LowState, '/low_state', 10)

        # ── Таймер публикации ──────────────────────────────────────────
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_low_state)

        self.get_logger().info(
            f'damiao_can_bridge запущен, моторы: {MOTOR_IDS}')

    # ── Callbacks входящих топиков ─────────────────────────────────────

    def _cb_low_cmd(self, msg: LowCmd) -> None:
        """LowCmd → MIT frame для каждого мотора из MOTOR_IDS."""
        for idx, motor_id in enumerate(MOTOR_IDS):
            if idx >= MAX_MOTORS:
                break
            cmd = msg.motor_cmd[idx]
            self._bus.send_mit(
                motor_id=motor_id,
                pos=float(cmd.position),
                vel=float(cmd.velocity),
                kp=float(cmd.kp),
                kd=float(cmd.kd),
                torq=float(cmd.torque),
            )

    def _cb_control_cmd(self, msg: ControlCmd) -> None:
        """ControlCmd → enable / disable / set_zero / clear_error."""
        self._bus.handle_control_cmd(
            motor_id=int(msg.motor_id),
            cmd_byte=int(msg.cmd),
        )

    # ── Обратная связь от моторов ──────────────────────────────────────

    def _on_motor_feedback(self, state: MotorState) -> None:
        """Вызывается из потока CAN при каждом feedback frame."""
        self._rx_tick += 1

    # ── Публикация /low_state ──────────────────────────────────────────

    def _publish_low_state(self) -> None:
        now: Time = self.get_clock().now().to_msg()

        low_state = LowState()
        low_state.timestamp_state = now
        low_state.tick = self._rx_tick

        for idx, motor_id in enumerate(MOTOR_IDS):
            if idx >= MAX_MOTORS:
                break
            state = self._bus.get_state(motor_id)
            ms = RosMotorState()
            ms.timestamp_state = now
            if state is not None:
                ms.position           = state.position
                ms.velocity           = state.velocity
                ms.torque             = state.torque
                ms.temperature_mosfet = state.temperature_mosfet
                ms.temperature_rotor  = state.temperature_rotor
                ms.error              = state.error
            low_state.motor_state[idx] = ms

        self._pub_state.publish(low_state)

    # ── Завершение ────────────────────────────────────────────────────

    def destroy_node(self):
        self._bus.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DamiaoCanBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
