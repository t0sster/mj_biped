#!/usr/bin/env python3
"""
Нода motor_gui: GUI для ручного управления отдельным мотором.
Пока кнопка «SEND» зажата — команды идут в /low_level_cmd и /control_command.
Кнопка отпущена — публикация прекращается.
"""
import sys
import threading

import rclpy
from rclpy.node import Node

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel,
    QGroupBox, QHBoxLayout,
)
from PyQt5.QtCore import QTimer, Qt

from tinker_msgs.msg import LowCmd, MotorCmd, ControlCmd

NUM_MOTORS      = 10
PUBLISH_RATE_MS = 10   # 10 мс = 100 Гц


# ── ROS2 нода ─────────────────────────────────────────────────────────────────

class MotorGuiNode(Node):
    def __init__(self):
        super().__init__('motor_gui')
        self.low_cmd_pub     = self.create_publisher(LowCmd,     '/low_level_cmd',   10)
        self.control_cmd_pub = self.create_publisher(ControlCmd, '/control_command', 10)

    def send_cmd(self, motor_id: int, pos: float, vel: float,
                 torque: float, kp: float, kd: float) -> None:
        # Enable выбранного мотора
        ctrl          = ControlCmd()
        ctrl.motor_id = motor_id
        ctrl.cmd      = ControlCmd.ENABLE
        self.control_cmd_pub.publish(ctrl)

        # LowCmd: только выбранный мотор активен, остальные — нули (kp=kd=0)
        motor_cmds = []
        for i in range(NUM_MOTORS):
            m = MotorCmd()
            if i == motor_id:
                m.position = float(pos)
                m.velocity = float(vel)
                m.torque   = float(torque)
                m.kp       = float(kp)
                m.kd       = float(kd)
            motor_cmds.append(m)

        low_cmd           = LowCmd()
        low_cmd.motor_cmd = motor_cmds
        self.low_cmd_pub.publish(low_cmd)


# ── Qt GUI ────────────────────────────────────────────────────────────────────

class MotorGuiWindow(QWidget):
    def __init__(self, node: MotorGuiNode):
        super().__init__()
        self._node    = node
        self._sending = False

        self.setWindowTitle('Motor GUI — ROS2')
        self.setMinimumWidth(320)

        # ── Поля ввода ────────────────────────────────────────────────
        self._motor_id = QSpinBox()
        self._motor_id.setRange(0, NUM_MOTORS - 1)
        self._motor_id.setValue(0)

        self._pos = self._double_spin(-30.0, 30.0, 0.0,  decimals=4, step=0.01)
        self._vel = self._double_spin(-50.0, 50.0, 0.0,  decimals=3, step=0.1)
        self._trq = self._double_spin(-30.0, 30.0, 0.0,  decimals=3, step=0.1)
        self._kp  = self._double_spin(  0.0, 500.0, 13.0, decimals=2, step=0.5)
        self._kd  = self._double_spin(  0.0,  50.0,  0.65, decimals=3, step=0.05)

        form = QFormLayout()
        form.addRow('Motor ID (0–9):', self._motor_id)
        form.addRow('Position (rad):', self._pos)
        form.addRow('Velocity (rad/s):', self._vel)
        form.addRow('Torque (Nm):', self._trq)
        form.addRow('Kp:', self._kp)
        form.addRow('Kd:', self._kd)

        group = QGroupBox('Параметры команды')
        group.setLayout(form)

        # ── Кнопка и статус ───────────────────────────────────────────
        self._btn = QPushButton('SEND')
        self._btn.setCheckable(True)
        self._btn.setMinimumHeight(60)
        self._btn.setStyleSheet(
            'QPushButton { font-size: 16px; background: #444; color: white; border-radius: 6px; }'
            'QPushButton:checked { background: #1a7a1a; }'
        )
        self._btn.toggled.connect(self._on_toggle)

        self._status = QLabel('● Idle')
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet('color: gray; font-size: 13px;')

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self._btn)
        btn_layout.addWidget(self._status)

        # ── Главный layout ────────────────────────────────────────────
        main_layout = QVBoxLayout()
        main_layout.addWidget(group)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # ── Таймер публикации ─────────────────────────────────────────
        self._pub_timer = QTimer(self)
        self._pub_timer.setInterval(PUBLISH_RATE_MS)
        self._pub_timer.timeout.connect(self._publish)

        # ── Таймер spin rclpy ─────────────────────────────────────────
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(5)
        self._spin_timer.timeout.connect(lambda: rclpy.spin_once(self._node, timeout_sec=0))
        self._spin_timer.start()

    # ── Вспомогательные ───────────────────────────────────────────────

    @staticmethod
    def _double_spin(lo: float, hi: float, default: float,
                     decimals: int = 3, step: float = 0.1) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(default)
        sb.setDecimals(decimals)
        sb.setSingleStep(step)
        return sb

    # ── Обработчики кнопки ────────────────────────────────────────────

    def _on_toggle(self, checked: bool):
        self._sending = checked
        if checked:
            self._pub_timer.start()
            self._status.setText('● Sending...')
            self._status.setStyleSheet('color: #1aaa1a; font-weight: bold; font-size: 13px;')
        else:
            self._pub_timer.stop()
            self._status.setText('● Idle')
            self._status.setStyleSheet('color: gray; font-size: 13px;')

    # ── Публикация ────────────────────────────────────────────────────

    def _publish(self):
        if not self._sending:
            return
        self._node.send_cmd(
            motor_id=self._motor_id.value(),
            pos=self._pos.value(),
            vel=self._vel.value(),
            torque=self._trq.value(),
            kp=self._kp.value(),
            kd=self._kd.value(),
        )


# ── Точка входа ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MotorGuiNode()

    app = QApplication(sys.argv)
    window = MotorGuiWindow(node)
    window.show()

    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
