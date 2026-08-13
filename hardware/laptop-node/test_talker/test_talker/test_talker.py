#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from tinker_msgs.msg import LowCmd, MotorCmd, LowState, ControlCmd


class SinusoidalTrajectoryTalker(Node):
    def __init__(self):
        super().__init__('sinusoidal_trajectory_talker')

        # ── Параметры синусоиды ───────────────────────────────────────
        self.amplitude     = 0.5
        self.frequency     = 0.35
        self.offset        = 0.0

        # ── Усиления ─────────────────────────────────────────────────
        self.kp_big        = 25.0
        self.kd_big        = 1.5
        self.torque_ff     = 0.0

        # ── Общие параметры ───────────────────────────────────────────
        self.rate_hz       = 100.0
        self.num_motors    = 10

        # ── Параметры огибающей ───────────────────────────────────────
        self.ramp_duration = 3.0   # секунды: за это время offset → 0, amplitude → max

        # ── Внутреннее состояние ──────────────────────────────────────
        self._ramp_start_t  = None          # время начала разгона
        self._start_pos     = None          # реальные позиции в момент старта [num_motors]
        self._ready         = False         # получили первый LowState

        self._pub_count = 0

        # ── ROS интерфейсы ────────────────────────────────────────────
        self.low_cmd_pub     = self.create_publisher(LowCmd,     '/low_level_cmd',   10)
        self.control_cmd_pub = self.create_publisher(ControlCmd, '/control_command', 10)
        self.lowstate_sub    = self.create_subscription(
            LowState, '/low_level_state_real', self.data_callback, 10)

        self.last_state = None

        self.create_timer(1.0 / self.rate_hz, self.publish_message)
        self.create_timer(1.0,                self._log_publish_rate)

        self.get_logger().info(
            f'Waiting for first /low_level_state_real ...\n'
            f'  ramp_duration={self.ramp_duration} s\n'
            f'  amplitude={self.amplitude}, frequency={self.frequency} Hz\n'
            f'  kp={self.kp_big}, kd={self.kd_big}, rate={self.rate_hz} Hz'
        )

    # ── Callback состояния ────────────────────────────────────────────

    def data_callback(self, msg: LowState):
        self.last_state = msg

        if not self._ready:
            t_now = self.get_clock().now().nanoseconds / 1e9

            self._start_pos    = [
                msg.motor_state[i].position for i in range(self.num_motors)
            ]
            self._ramp_start_t = t_now
            self._ready        = True

            self.get_logger().info(
                f'Got initial state, starting ramp.\n'
                f'  start_pos={[f"{p:.3f}" for p in self._start_pos]}'
            )

    # ── Основной цикл ─────────────────────────────────────────────────

    def publish_message(self):
        if not self._ready:
            return

        t_now   = self.get_clock().now().nanoseconds / 1e9
        elapsed = t_now - self._ramp_start_t
        alpha   = min(elapsed / self.ramp_duration, 1.0)  # [0.0 → 1.0]

        # Амплитуда нарастает, стартовый offset затухает
        effective_amplitude = self.amplitude * alpha
        sin_val = self.offset + effective_amplitude * math.sin(
            2.0 * math.pi * self.frequency * t_now
        )

        # Enable всех моторов
        for i in range(self.num_motors):
            ctrl_msg      = ControlCmd()
            ctrl_msg.motor_id = i
            ctrl_msg.cmd  = 252  # CMD_ENABLE
            self.control_cmd_pub.publish(ctrl_msg)

        # Формирование команд
        motor_cmds = []
        for i in range(self.num_motors):
            kp = 1.5    if i == 9 else self.kp_big
            kd = 1.0    if i == 9 else self.kd_big

            # Базовая синусоидальная позиция с учётом масштаба мотора
            sin_pos = sin_val / 8.0 if i in [0, 1, 5, 6] else sin_val

            # Стартовый offset затухает по мере роста alpha
            pos = self._start_pos[i] * (1.0 - alpha) + sin_pos

            m          = MotorCmd()
            m.position = float(pos)
            m.velocity = 0.0
            m.torque   = float(self.torque_ff)
            m.kp       = float(kp)
            m.kd       = float(kd)
            motor_cmds.append(m)

        low_cmd           = LowCmd()
        low_cmd.motor_cmd = motor_cmds
        self.low_cmd_pub.publish(low_cmd)
        self._pub_count  += 1

    # ── Лог частоты ───────────────────────────────────────────────────

    def _log_publish_rate(self):
        status = f'alpha={min((self.get_clock().now().nanoseconds/1e9 - self._ramp_start_t) / self.ramp_duration, 1.0):.2f}' \
            if self._ready else 'WAITING'
        self.get_logger().info(
            f'[{status}] publish rate: {self._pub_count} cmd/s'
        )
        self._pub_count = 0


def main():
    rclpy.init()
    node = SinusoidalTrajectoryTalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()