#!/usr/bin/env python3
"""
ROS2 Node: hardware_node
========================
Одна нода на Raspberry Pi: собирает данные со всех датчиков робота
(10 моторов Damiao по CAN + IMU HWT906 по UART) и отправляет их на ПК,
а команды с ПК раздаёт моторам.

Подписки (приходят с ПК, см. пример в laptop-node/test_talker):
  /low_level_cmd         (LowCmd)      — команды MIT mode на все 10 моторов
  /control_command       (ControlCmd)  — enable / disable / set_zero / clear_error

Публикации (уходят на ПК):
  /low_level_state_real  (LowState)    — состояние моторов + IMU

Запуск (can0 поднят, см. шапку damiao_can.py):
    ros2 run biped_hardware hardware_node
"""

import rclpy
from rclpy.node import Node

from tinker_msgs.msg import ControlCmd, IMUState, LowCmd, LowState, MotorState

from biped_hardware.damiao_can import DamiaoMotorBus
from biped_hardware.hwt906_imu import Hwt906Imu

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

# CAN ID моторов: 1..5 левая нога, 6..10 правая.
# Порядок задаёт индекс мотора в сообщениях LowCmd/LowState.
MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

PUBLISH_RATE_HZ = 100.0    # частота публикации /low_level_state_real
LOG_RATE_HZ = 1.0          # частота диагностических сообщений в консоль

USE_IMU = True             # False — работать только с моторами, без IMU

# ─────────────────────────────────────────────────────────────────────────────


class HardwareNode(Node):

    def __init__(self):
        super().__init__("hardware_node")

        self.motor_bus = DamiaoMotorBus(rx_callback=self._on_motor_feedback)
        self.imu = Hwt906Imu() if USE_IMU else None

        # счётчики принятых сообщений, растут всё время работы ноды
        self.motor_feedback_count = 0
        self.received_cmd_count = 0

        # значения счётчиков на момент прошлого лога — из них считаем частоту
        self.counts_at_last_log = (0, 0, 0)

        # ── Подписки ──────────────────────────────────────────────────────────
        self.create_subscription(
            LowCmd, "/low_level_cmd", self._on_low_cmd, 10)
        self.create_subscription(
            ControlCmd, "/control_command", self._on_control_cmd, 10)

        # ── Публикации ────────────────────────────────────────────────────────
        self.low_state_publisher = self.create_publisher(
            LowState, "/low_level_state_real", 10)

        # ── Таймеры ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_low_state)
        self.create_timer(1.0 / LOG_RATE_HZ, self._log_status)

        self.get_logger().info(
            f"hardware_node запущен: моторы {MOTOR_IDS}, "
            f"IMU {'вкл' if USE_IMU else 'выкл'}, "
            f"публикация {PUBLISH_RATE_HZ:.0f} Гц"
        )

    # ── Команды с ПК ──────────────────────────────────────────────────────────

    def _on_low_cmd(self, msg: LowCmd) -> None:
        """LowCmd → MIT-команда каждому мотору."""
        self.received_cmd_count += 1

        for index, motor_id in enumerate(MOTOR_IDS):
            cmd = msg.motor_cmd[index]
            self.motor_bus.send_mit(
                motor_id=motor_id,
                pos=float(cmd.position),
                vel=float(cmd.velocity),
                kp=float(cmd.kp),
                kd=float(cmd.kd),
                torq=float(cmd.torque),
            )

    def _on_control_cmd(self, msg: ControlCmd) -> None:
        """ControlCmd → enable / disable / set_zero / clear_error."""
        self.motor_bus.handle_control_cmd(
            motor_id=int(msg.motor_id),
            cmd_byte=int(msg.cmd),
        )

    # ── Данные с моторов ──────────────────────────────────────────────────────

    def _on_motor_feedback(self, state) -> None:
        """Вызывается из потока CAN на каждый пришедший фидбек."""
        self.motor_feedback_count += 1

    # ── Отправка состояния на ПК ──────────────────────────────────────────────

    def _publish_low_state(self) -> None:
        now = self.get_clock().now().to_msg()

        low_state = LowState()
        low_state.timestamp_state = now
        low_state.tick = self.motor_feedback_count % 2**32   # поле uint32
        low_state.imu_state = self._make_imu_state(now)

        for index, motor_id in enumerate(MOTOR_IDS):
            low_state.motor_state[index] = self._make_motor_state(motor_id, now)

        self.low_state_publisher.publish(low_state)

    def _make_motor_state(self, motor_id: int, now) -> MotorState:
        """Последний фидбек мотора → сообщение MotorState."""
        motor_state = MotorState()
        motor_state.timestamp_state = now

        state = self.motor_bus.get_state(motor_id)
        if state is None:
            # мотор ещё ни разу не ответил — отдаём нули и код потери связи
            motor_state.error = MotorState.LOSS_CONNECTION
            return motor_state

        motor_state.position = state.position
        motor_state.velocity = state.velocity
        motor_state.torque = state.torque
        motor_state.temperature_mosfet = state.temperature_mosfet
        motor_state.temperature_rotor = state.temperature_rotor
        motor_state.error = state.error
        return motor_state

    def _make_imu_state(self, now) -> IMUState:
        """Последние данные IMU → сообщение IMUState."""
        imu_state = IMUState()
        imu_state.timestamp_state = now

        if self.imu is None:
            return imu_state

        imu_state.quaternion = self.imu.quaternion
        imu_state.gyroscope = self.imu.gyroscope
        imu_state.accelerometer = self.imu.accelerometer
        imu_state.rpy = self.imu.rpy
        imu_state.temperature = int(self.imu.temperature)
        return imu_state

    # ── Диагностика ───────────────────────────────────────────────────────────

    def _log_status(self) -> None:
        """Раз в секунду печатает, сколько сообщений пришло за эту секунду."""
        counts_now = (
            self.received_cmd_count,
            self.motor_feedback_count,
            self.imu.packet_count if self.imu else 0,
        )
        cmd_rate, feedback_rate, imu_rate = [
            now - before for now, before in zip(counts_now, self.counts_at_last_log)
        ]
        self.counts_at_last_log = counts_now

        self.get_logger().info(
            f"команд принято: {cmd_rate}/с, "
            f"фидбек моторов: {feedback_rate}/с, "
            f"пакетов IMU: {imu_rate}/с"
        )

    # ── Завершение ────────────────────────────────────────────────────────────

    def destroy_node(self):
        # снимаем момент со всех моторов, чтобы робот не остался под управлением
        for motor_id in MOTOR_IDS:
            self.motor_bus.disable(motor_id)

        self.motor_bus.close()
        if self.imu:
            self.imu.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HardwareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Остановлено пользователем")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
