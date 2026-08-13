#!/usr/bin/env python3
"""
Нода init_pos: плавно интерполирует моторы из текущей позиции в целевую.
Использует обратную связь из /low_level_state_real.
Завершается автоматически когда все моторы достигли цели (по реальным позициям).
"""
import math
import rclpy
from rclpy.node import Node
from tinker_msgs.msg import LowCmd, MotorCmd, LowState, ControlCmd

# Целевые позиции для 10 моторов (рад)
TARGET_POS = [0.0, 0.08, 0.56, -1.12, -0.57,
              0.0, -0.08, -0.56, 1.12, 0.57]

KP_DEFAULT     = 13.0
KD_DEFAULT     = 0.65

RATE_HZ        = 100.0   # Гц
DURATION_S     = 5.0     # длительность интерполяции (секунды)
DONE_THRESHOLD = 0.03    # рад — все моторы должны быть ближе этого к цели


class InitPosNode(Node):
    def __init__(self):
        super().__init__('init_pos')

        self.num_motors = len(TARGET_POS)

        self._start_pos   = None   # позиции в момент первого state
        self._start_time  = None   # время начала интерполяции
        self._last_state  = None   # последний полученный state (для проверки позиций)
        self._ready       = False  # получили первый LowState

        self.low_cmd_pub     = self.create_publisher(LowCmd,     '/low_level_cmd',   10)
        self.control_cmd_pub = self.create_publisher(ControlCmd, '/control_command', 10)
        self.lowstate_sub    = self.create_subscription(
            LowState, '/low_level_state_real', self._cb_state, 10)

        self.create_timer(1.0 / RATE_HZ, self._tick)

        self.get_logger().info(
            f'init_pos: ожидаю /low_level_state_real...\n'
            f'  target={TARGET_POS}\n'
            f'  duration={DURATION_S} s, threshold={DONE_THRESHOLD} rad'
        )

    # ── Callback состояния ─────────────────────────────────────────────

    def _cb_state(self, msg: LowState):
        self._last_state = msg  # всегда обновляем для проверки текущих позиций

        if self._ready:
            return  # старт уже зафиксирован, дальше только _tick читает _last_state

        self._start_pos  = [msg.motor_state[i].position for i in range(self.num_motors)]
        self._start_time = self.get_clock().now().nanoseconds / 1e9
        self._ready      = True

        self.get_logger().info(
            f'Получили начальное состояние, начинаю интерполяцию.\n'
            f'  start_pos={[f"{p:.3f}" for p in self._start_pos]}'
        )

    # ── Основной цикл ──────────────────────────────────────────────────

    def _tick(self):
        if not self._ready:
            return

        t_now   = self.get_clock().now().nanoseconds / 1e9
        elapsed = t_now - self._start_time
        alpha   = min(elapsed / DURATION_S, 1.0)  # 0.0 → 1.0

        # Enable всех моторов каждый тик
        for i in range(self.num_motors):
            ctrl_msg          = ControlCmd()
            ctrl_msg.motor_id = i
            ctrl_msg.cmd      = 252  # CMD_ENABLE
            self.control_cmd_pub.publish(ctrl_msg)

        # Линейная интерполяция start → target
        motor_cmds = []
        for i in range(self.num_motors):
            pos = self._start_pos[i] + alpha * (TARGET_POS[i] - self._start_pos[i])

            m          = MotorCmd()
            m.position = float(pos)
            m.velocity = 0.0
            m.torque   = 0.0
            m.kp       = float(KP_DEFAULT)
            m.kd       = float(KD_DEFAULT)
            motor_cmds.append(m)

        low_cmd           = LowCmd()
        low_cmd.motor_cmd = motor_cmds
        self.low_cmd_pub.publish(low_cmd)

        # Проверка достижения цели по реальным позициям (из обратной связи)
        if alpha >= 1.0 and self._last_state is not None:
            errors = [
                abs(self._last_state.motor_state[i].position - TARGET_POS[i])
                for i in range(self.num_motors)
            ]
            max_err = max(errors)
            if max_err < DONE_THRESHOLD:
                self.get_logger().info(
                    f'Все моторы достигли цели (max_err={max_err:.4f} rad). Завершаю init_pos.'
                )
                raise SystemExit
            else:
                self.get_logger().info(
                    f'Интерполяция завершена, ожидаю сходимости (max_err={max_err:.4f} rad)...',
                    throttle_duration_sec=0.5,
                )


def main(args=None):
    rclpy.init(args=args)
    node = InitPosNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
