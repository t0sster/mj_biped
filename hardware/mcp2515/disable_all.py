"""
Аварийное отключение моторов Damiao по CAN — команда DISABLE (0xFD / 253).

Не ждёт подтверждения и ничего не удерживает — просто шлёт CMD_DISABLE
на каждый выбранный мотор и выходит. Полезно сразу после сбоя вроде
"CAN TX error: No buffer space available", чтобы снять момент с моторов,
не дожидаясь штатного завершения теста.

Запускать после source ros2_ws/install/setup.bash (нужен активный can0):
    python3 disable_all.py
"""

import time
from typing import List

from biped_hardware.damiao_can import DamiaoMotorBus

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

# 10 слотов — CAN ID моторов: 1..5 левая нога, 6..10 правая (как в hardware_node.MOTOR_IDS).
MOTOR_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Какие СЛОТЫ отключать. По умолчанию — все 10.
ACTIVE_SLOTS: List[int] = list(range(10))

# ─────────────────────────────────────────────────────────────────────────────


def run() -> None:
    active_ids = [MOTOR_IDS[slot] for slot in ACTIVE_SLOTS]

    with DamiaoMotorBus() as bus:
        for mid in active_ids:
            bus.disable(mid)
            time.sleep(0.005)

    print(f"DISABLE (253) отправлен на моторы: {active_ids}")


if __name__ == "__main__":
    run()
