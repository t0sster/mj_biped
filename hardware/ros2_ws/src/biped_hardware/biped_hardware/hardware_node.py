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

import time

import rclpy
from rclpy.node import Node

from tinker_msgs.msg import ControlCmd, IMUState, LowCmd, LowState, MotorState

from biped_hardware.damiao_can import CAN_MASTER_ID, DamiaoMotorBus
from biped_hardware.hwt906_imu import Hwt906Imu

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

# CAN ID моторов: 1..5 левая нога, 6..10 правая.
# Порядок задаёт индекс мотора в сообщениях LowCmd/LowState.
MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Моторы отвечают только на пришедший к ним кадр, поэтому команды шлём
# непрерывно — иначе фидбека не будет. Чем выше частота, тем плотнее поток:
# CONTROL_RATE_HZ * len(MOTOR_IDS) кадров в секунду. Если в логах появится
# 'No buffer space available' — MCP2515 не успевает, частоту надо снизить.
CONTROL_RATE_HZ = 100.0

PUBLISH_RATE_HZ = 100.0    # частота публикации /low_level_state_real
LOG_RATE_HZ = 1.0          # частота диагностических сообщений в консоль

# Если команд с ПК нет дольше этого времени, считаем, что ПК молчит,
# и переходим на холостые кадры (моторы при этом не двигаются).
CMD_TIMEOUT_SEC = 0.5

# Если мотор не отвечает дольше этого времени, помечаем его как потерянного.
FEEDBACK_TIMEOUT_SEC = 0.2

USE_IMU = True             # False — работать только с моторами, без IMU

# ─────────────────────────────────────────────────────────────────────────────


class HardwareNode(Node):

    def __init__(self):
        super().__init__("hardware_node")

        self.motor_bus = DamiaoMotorBus(
            rx_callback=self._on_motor_feedback,
            error_callback=self._on_can_error,
        )
        self.imu = Hwt906Imu() if USE_IMU else None

        # последняя команда с ПК и время её получения: её и шлём в моторы,
        # пока не придёт следующая
        self.last_motor_cmd = None
        self.last_cmd_time = 0.0
        # что шлём прямо сейчас: команды с ПК или холостые кадры
        self.is_sending_pc_cmd = False

        # счётчики принятых сообщений, растут всё время работы ноды
        self.motor_feedback_count = 0
        self.received_cmd_count = 0

        # значения счётчиков на момент прошлого лога — из них считаем частоту
        self.counts_at_last_log = (0, 0, 0, 0, 0, 0)

        # ── Подписки ──────────────────────────────────────────────────────────
        self.create_subscription(
            LowCmd, "/low_level_cmd", self._on_low_cmd, 10)
        self.create_subscription(
            ControlCmd, "/control_command", self._on_control_cmd, 10)

        # ── Публикации ────────────────────────────────────────────────────────
        self.low_state_publisher = self.create_publisher(
            LowState, "/low_level_state_real", 10)

        # ── Таймеры ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / CONTROL_RATE_HZ, self._send_motor_commands)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_low_state)
        self.create_timer(1.0 / LOG_RATE_HZ, self._log_status)

        self.get_logger().info(
            f"hardware_node запущен: моторы {MOTOR_IDS}, "
            f"IMU {'вкл' if USE_IMU else 'выкл'}, "
            f"отправка {CONTROL_RATE_HZ:.0f} Гц, "
            f"публикация {PUBLISH_RATE_HZ:.0f} Гц"
        )

    # ── Команды с ПК ──────────────────────────────────────────────────────────

    def _on_low_cmd(self, msg: LowCmd) -> None:
        """Команду только запоминаем — в моторы её шлёт таймер."""
        self.received_cmd_count += 1
        self.last_motor_cmd = msg
        self.last_cmd_time = time.time()

    def _on_control_cmd(self, msg: ControlCmd) -> None:
        """ControlCmd → enable / disable / set_zero / clear_error."""
        self.motor_bus.handle_control_cmd(
            motor_id=int(msg.motor_id),
            cmd_byte=int(msg.cmd),
        )

    # ── Отправка команд в моторы ──────────────────────────────────────────────

    def _send_motor_commands(self) -> None:
        """
        Шлём кадры в моторы непрерывно: контроллер отвечает фидбеком только
        на пришедший к нему кадр, без потока команд состояние не обновляется.
        """
        has_fresh_cmd = (
            self.last_motor_cmd is not None
            and time.time() - self.last_cmd_time < CMD_TIMEOUT_SEC
        )

        # сообщаем в лог только о смене режима, а не каждый раз
        if has_fresh_cmd != self.is_sending_pc_cmd:
            self.is_sending_pc_cmd = has_fresh_cmd
            self.get_logger().info(
                "пошли команды с ПК" if has_fresh_cmd
                else "команд с ПК нет, шлю холостые кадры"
            )

        for index, motor_id in enumerate(MOTOR_IDS):
            if has_fresh_cmd:
                cmd = self.last_motor_cmd.motor_cmd[index]
                self.motor_bus.send_mit(
                    motor_id=motor_id,
                    pos=float(cmd.position),
                    vel=float(cmd.velocity),
                    kp=float(cmd.kp),
                    kd=float(cmd.kd),
                    torq=float(cmd.torque),
                )
            else:
                # холостой кадр: kp = kd = 0 и нулевой момент, поэтому мотор
                # ничего не делает, но кадр получает и отвечает фидбеком
                self.motor_bus.send_mit(
                    motor_id=motor_id, pos=0.0, vel=0.0, kp=0.0, kd=0.0, torq=0.0
                )

    # ── Данные с моторов ──────────────────────────────────────────────────────

    def _on_motor_feedback(self, state) -> None:
        """Вызывается из потока CAN на каждый пришедший фидбек."""
        self.motor_feedback_count += 1

    def _on_can_error(self, error_text: str) -> None:
        """Вызывается из драйвера при ошибке шины — не чаще раза в секунду в лог."""
        self.get_logger().error(f"ошибка CAN: {error_text}", throttle_duration_sec=1.0)

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

        # мотор ещё ни разу не ответил или замолчал — отдаём код потери связи
        if state is None or time.time() - state.timestamp > FEEDBACK_TIMEOUT_SEC:
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
            self.motor_bus.tx_frame_count,
            self.motor_bus.rx_frame_count,
            self.motor_bus.tx_error_count + self.motor_bus.error_frame_count,
        )
        cmd_rate, feedback_rate, imu_rate, tx_rate, rx_rate, error_rate = [
            now - before for now, before in zip(counts_now, self.counts_at_last_log)
        ]
        self.counts_at_last_log = counts_now

        self.get_logger().info(
            f"команд принято: {cmd_rate}/с, "
            f"фидбек моторов: {feedback_rate}/с, "
            f"пакетов IMU: {imu_rate}/с, "
            f"кадров в шину: {tx_rate}/с, "
            f"кадров с шины: {rx_rate}/с, "
            f"ошибок CAN: {error_rate}/с, "
            f"шина: {self.motor_bus.get_bus_state_text()}"
        )

        # молчит IMU — обычно порт занят другой программой
        if self.imu and imu_rate == 0:
            self.get_logger().warn(
                f"IMU молчит: {self.imu.last_error_text}" if self.imu.last_error_text
                else "IMU молчит: не занят ли порт другой программой (lsof /dev/ttyAMA0)"
            )

        # молчат все моторы — подсказываем, куда смотреть
        if feedback_rate == 0:
            other_ids = self.motor_bus.other_frame_ids
            if other_ids:
                # кадры с шины идут, но не с тем ID, который мы считаем своим
                ids_text = " ".join(f"0x{frame_id:03X}" for frame_id in sorted(other_ids))
                self.get_logger().warn(
                    f"фидбека нет, но на шине есть кадры с ID: {ids_text}. "
                    f"Ждём ID 0x{CAN_MASTER_ID:03X} — поправьте CAN_MASTER_ID "
                    f"в damiao_can.py или Master ID в моторах"
                )
            else:
                self.get_logger().warn(
                    "с шины не приходит ни одного кадра: проверьте питание "
                    "моторов, их CAN ID и что can0 поднят на 1 Мбит/с"
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
