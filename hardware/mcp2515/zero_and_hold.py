"""
Обнуление позиции моторов Damiao по CAN и удержание в позиции 0.

Логика та же, что перед чирпом в chirp_logger.py: SET_ZERO делает текущую
физическую позицию нулём (мотор не двигается), затем по фидбеку ждём
подтверждения, что позиция реально стала ~0, и только после этого начинаем
удерживать pos=0 — MIT-команду нужно слать непрерывно (watchdog драйвера
сбрасывает момент без потока команд), одной командой позицию не удержать.

Запускать после source ros2_ws/install/setup.bash (нужен активный can0):
    python3 zero_and_hold.py
Остановить — Ctrl+C (моторы будут выключены перед выходом).
"""

import time
from typing import List

from biped_hardware.damiao_can import DamiaoMotorBus

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

# 10 слотов — впишите сюда CAN ID своих моторов (как в hardware_node.MOTOR_IDS).
MOTOR_IDS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Какие СЛОТЫ (индексы 0..9 в MOTOR_IDS выше) обнулять и удерживать в этом запуске.
ACTIVE_SLOTS: List[int] = [4]

# MIT-параметры позиционного ПД при удержании pos=0.
KP = 15.0
KD = 0.65

CONTROL_RATE_HZ = 200.0   # частота отправки удерживающих MIT-команд

# Обнуление позиции: сколько ждать подтверждения по фидбеку и с какой точностью.
ZERO_TOLERANCE = 0.02   # рад
ZERO_TIMEOUT = 2.0      # с

# Сколько держать pos=0 после обнуления. None — держать, пока не нажмёте Ctrl+C.
HOLD_DURATION: float = None

# ─────────────────────────────────────────────────────────────────────────────


def zero_motors(
    bus: DamiaoMotorBus,
    motor_ids: List[int],
    tol: float = ZERO_TOLERANCE,
    timeout: float = ZERO_TIMEOUT,
) -> None:
    """Обнулить позицию каждого мотора (CMD_SET_ZERO сохраняет текущую
    физическую позицию как 0, сам мотор при этом не двигается) и дождаться
    по фидбеку, что позиция реально стала ~0."""
    for mid in motor_ids:
        bus.set_zero_position(mid)

    pending = set(motor_ids)
    deadline = time.perf_counter() + timeout
    while pending and time.perf_counter() < deadline:
        for mid in list(pending):
            state = bus.get_state(mid)
            if state is not None and abs(state.position) < tol:
                pending.discard(mid)
        if pending:
            time.sleep(0.01)

    if pending:
        print(f"ВНИМАНИЕ: моторы {sorted(pending)} не подтвердили обнуление "
              f"за {timeout}с (позиция ещё не ~0).")
    else:
        print(f"Обнуление подтверждено для моторов {motor_ids}.")


def run() -> None:
    active_ids = [MOTOR_IDS[slot] for slot in ACTIVE_SLOTS]

    with DamiaoMotorBus() as bus:
        for mid in active_ids:
            bus.enable(mid)
        time.sleep(0.2)

        zero_motors(bus, active_ids)

        print("Удержание pos=0" +
              (f" в течение {HOLD_DURATION}с..." if HOLD_DURATION else " (Ctrl+C для выхода)..."))

        period = 1.0 / CONTROL_RATE_HZ
        t_start = time.perf_counter()
        next_tick = t_start
        try:
            while HOLD_DURATION is None or time.perf_counter() - t_start < HOLD_DURATION:
                for mid in active_ids:
                    bus.send_mit(mid, pos=0.0, kp=KP, kd=KD)

                next_tick += period
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nОстановлено пользователем.")
        finally:
            for mid in active_ids:
                bus.disable(mid)
            print("Моторы выключены.")


if __name__ == "__main__":
    run()
