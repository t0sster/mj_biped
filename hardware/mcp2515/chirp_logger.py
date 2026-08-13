"""
Chirp-тест моторов Damiao по CAN — без ROS2.

Идея: та же логика, что в mujoco-примере set_chirp_target() (свип по частоте
f0 → f1, синусоидальная позиционная цель), но цель уходит по CAN через
send_mit(), а не в data.ctrl. Фидбек всех моторов копится в обычных
python-списках прямо в rx_callback (без записи на диск и без ROS2-топиков) —
это самый быстрый вариант логирования, диск трогаем один раз в конце теста.

После теста строится график: по одному subplot'у на каждый мотор из
ACTIVE_SLOTS, на нём — отправленная команда (target) поверх фактической
позиции мотора. Число графиков подстраивается под len(ACTIVE_SLOTS)
автоматически.

Запускать после source ros2_ws/install/setup.bash (нужен активный can0, см. шапку damiao_can.py):
    python3 chirp_logger.py
"""

import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

from biped_hardware.damiao_can import DamiaoMotorBus, MotorState

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

# 10 слотов — CAN ID моторов: 1..5 левая нога, 6..10 правая (как в hardware_node.MOTOR_IDS).
MOTOR_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Какие СЛОТЫ (индексы 0..9 в MOTOR_IDS выше) гонять чирпом в этом запуске.
# Например [0] — только мотор из слота 0, [0, 2] — сразу два мотора.
ACTIVE_SLOTS: List[int] = [0, 1, 2, 3, 4]

# MIT-параметры позиционного ПД на драйвере мотора во время теста.
KP = 15.0
KD = 0.65

# Параметры chirp-сигнала (как в mujoco-примере). Амплитуда — своя на каждый
# слот (индекс совпадает с MOTOR_IDS/JOINT_LIMITS: сначала левая нога, потом
# правая), частота и длительность свипа — общие для всех моторов теста.
# AMPLITUDES: List[float] = [0.25, 0.18, 1.5, 2.3, 0.65, 0.28, 0.18, 1.5, 2.3, 0.65]   # рад, AMPLITUDES[slot]
# AMPLITUDES: List[float] = [0.25, 0.18, 1.5, 2.3, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0]   # рад, AMPLITUDES[slot]
AMPLITUDES: List[float] = [0.25, 0.18, 1.5, 2.3, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0]   # рад, AMPLITUDES[slot]
F0 = 0.1            # Гц, начальная частота
F1 = 1.0            # Гц, конечная частота
DURATION = 20.0     # с

CONTROL_RATE_HZ = 200.0   # частота отправки MIT-команд

# Обнуление позиции перед чирпом (см. zero_motors()).
ZERO_TOLERANCE = 0.02   # рад — насколько близко фидбек должен подойти к 0
ZERO_TIMEOUT = 2.0      # с — сколько ждать подтверждения, прежде чем сдаться

LOG_DIR = Path("chirp_logs")

# Лимиты по СЛОТАМ (индекс совпадает с MOTOR_IDS/ACTIVE_SLOTS выше), рад.
# Взяты из <joint ... range="..."> в mujoco-описании tinymal (5 суставов на
# ногу: yaw, roll, pitch, knee, ankle). ПРОВЕРЬТЕ порядок под свою распиновку
# CAN ID — если слоты подключены не в этом порядке, переставьте строки.
JOINT_LIMITS: List[tuple] = [
    ("l_yaw",   -0.3,  0.3),   # слот 0
    ("l_roll",  -0.2,  0.2),   # слот 1
    ("l_pitch", -0.57, 1.57),  # слот 2
    ("l_knee",  -2.35, 0.05),  # слот 3
    ("l_ankle", -0.6,  0.7),   # слот 4
    ("r_yaw",   -0.3,  0.3),   # слот 5
    ("r_roll",  -0.2,  0.2),   # слот 6
    ("r_pitch", -0.57, 1.57),  # слот 7
    ("r_knee",  -0.05, 2.35),  # слот 8
    ("r_ankle", -0.7,  0.6),   # слот 9
]

# ─────────────────────────────────────────────────────────────────────────────


_clamp_warned: set = set()


def clamp_to_joint_limit(slot: int, pos: float) -> float:
    """Обрезать целевую позицию под лимит сустава в этом слоте (JOINT_LIMITS).
    При первом выходе за предел для слота печатает предупреждение один раз,
    дальше молча клэмпит — чтобы не заспамить консоль на 200 Гц."""
    name, lo, hi = JOINT_LIMITS[slot]
    clamped = max(lo, min(hi, pos))
    if clamped != pos and slot not in _clamp_warned:
        _clamp_warned.add(slot)
        print(f"ВНИМАНИЕ: слот {slot} ({name}): цель {pos:+.3f} рад вышла за "
              f"лимит [{lo}, {hi}] — команды будут обрезаться по границе.")
    return clamped


def chirp_target(
    t: float,
    amplitude: float,
    f0: float = F0,
    f1: float = F1,
    duration: float = DURATION,
) -> Optional[float]:
    """Аналог set_chirp_target() из mujoco-примера: возвращает целевую
    позицию (рад) в момент времени t, либо None, если чирп закончился."""
    if t > duration:
        return None
    k = (f1 - f0) / duration
    tt = min(t, duration)
    phase = 2.0 * math.pi * (f0 * tt + 0.5 * k * tt * tt)
    return amplitude * math.sin(phase)


class FastLogger:
    """Копит фидбек и отправленные команды в python-списках (append атомарен
    под GIL, поэтому безопасно вызывается из CAN rx-потока без явных локов).
    Никакого диска и ROS2-сериализации в процессе теста — только в save()."""

    def __init__(self) -> None:
        self._rows: Dict[int, list] = {}
        self._cmd_rows: Dict[int, list] = {}

    def on_state(self, state: MotorState) -> None:
        self._rows.setdefault(state.motor_id, []).append((
            state.timestamp,
            state.position,
            state.velocity,
            state.torque,
            state.temperature_mosfet,
            state.temperature_rotor,
            state.status_str,
        ))

    def on_command(self, motor_id: int, t: float, target_pos: float) -> None:
        """t — время с начала теста (perf_counter-дельта), см. run_chirp_test.
        wall_time (time.time()) пишем отдельно — в той же шкале, что и
        state.timestamp в on_state(), чтобы потом считать задержку напрямую
        вычитанием меток без пересчёта perf_counter <-> wall clock."""
        self._cmd_rows.setdefault(motor_id, []).append((time.time(), t, target_pos))

    def save(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        for motor_id, rows in self._rows.items():
            path = log_dir / f"motor_{motor_id}.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "position", "velocity", "torque",
                            "temp_mosfet", "temp_rotor", "status"])
                w.writerows(rows)
            print(f"[{motor_id}] {len(rows)} строк -> {path}")

        for motor_id, rows in self._cmd_rows.items():
            path = log_dir / f"motor_{motor_id}_cmd.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "t_rel", "target_pos"])
                w.writerows(rows)
            print(f"[{motor_id}] {len(rows)} команд -> {path}")

    def print_rates(self, active_ids: List[int]) -> None:
        """Фактическая частота логирования (фидбека) и отправки команд
        по каждому мотору — считается по интервалу между первой и
        последней записью, а не по номинальной CONTROL_RATE_HZ."""
        print("\nЧастота за время теста:")
        for motor_id in active_ids:
            state = self._rows.get(motor_id, [])
            if len(state) > 1:
                state_hz = (len(state) - 1) / (state[-1][0] - state[0][0])
                state_str = f"{state_hz:.1f} Гц ({len(state)} сэмплов)"
            else:
                state_str = "недостаточно данных"

            cmd = self._cmd_rows.get(motor_id, [])
            if len(cmd) > 1:
                cmd_hz = (len(cmd) - 1) / (cmd[-1][0] - cmd[0][0])
                cmd_str = f"{cmd_hz:.1f} Гц ({len(cmd)} команд)"
            else:
                cmd_str = "недостаточно данных"

            print(f"  [{motor_id}] логирование фидбека: {state_str}  |  "
                  f"отправка команд: {cmd_str}")

    def print_latency(self, active_ids: List[int]) -> None:
        """Задержка команда → фидбек: для каждой отправленной MIT-команды
        ищем ближайший следующий по времени фидбек-фрейм того же мотора
        (протокол request-response — на каждую команду мотор должен
        ответить одним фреймом) и считаем time(feedback) - time(command).
        Обе метки времени — time.time(), поэтому вычитаются напрямую."""
        print("\nЗадержка команда -> фидбек:")
        for motor_id in active_ids:
            cmd = self._cmd_rows.get(motor_id, [])
            state = self._rows.get(motor_id, [])
            if not cmd or not state:
                print(f"  [{motor_id}] недостаточно данных")
                continue

            state_times = [s[0] for s in state]
            delays = []
            j = 0
            for cmd_time, _t_rel, _target in cmd:
                while j < len(state_times) and state_times[j] < cmd_time:
                    j += 1
                if j >= len(state_times):
                    break
                delays.append(state_times[j] - cmd_time)

            if not delays:
                print(f"  [{motor_id}] не удалось сопоставить команды с фидбеком")
                continue

            delays.sort()
            n = len(delays)
            mean_ms = 1000 * sum(delays) / n
            median_ms = 1000 * delays[n // 2]
            max_ms = 1000 * delays[-1]
            print(f"  [{motor_id}] среднее={mean_ms:.2f} мс  "
                  f"медиана={median_ms:.2f} мс  макс={max_ms:.2f} мс  "
                  f"({n} пар команда/фидбек)")

    def plot(self, active_ids: List[int], wall_t0: float, out_path: Path) -> None:
        """Один subplot на мотор: отправленная команда (target) поверх
        фактической позиции. Backend Agg — работает без дисплея по SSH."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(active_ids)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True, squeeze=False)

        for row, motor_id in enumerate(active_ids):
            ax = axes[row][0]

            cmd = self._cmd_rows.get(motor_id, [])
            if cmd:
                _, cmd_t, cmd_pos = zip(*cmd)
                ax.plot(cmd_t, cmd_pos, label="команда (target)",
                        color="tab:orange", linewidth=1.2)

            state = self._rows.get(motor_id, [])
            if state:
                state_t = [s[0] - wall_t0 for s in state]
                state_pos = [s[1] for s in state]
                ax.plot(state_t, state_pos, label="факт (position)",
                        color="tab:blue", linewidth=1.0)

            ax.set_ylabel("рад")
            ax.set_title(f"Мотор {motor_id}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1][0].set_xlabel("время, с")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"график -> {out_path}")


def zero_motors(
    bus: DamiaoMotorBus,
    motor_ids: List[int],
    tol: float = ZERO_TOLERANCE,
    timeout: float = ZERO_TIMEOUT,
) -> None:
    """Обнулить позицию каждого мотора (CMD_SET_ZERO сохраняет текущую
    физическую позицию как 0, сам мотор при этом не двигается) и дождаться
    по фидбеку, что позиция реально стала ~0. Чирп начинается только после
    этого — его первая команда (t=0) сама по себе равна 0, так что если
    фактическая позиция к этому моменту тоже ~0, мотор не дёрнется."""
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
              f"за {timeout}с (позиция ещё не ~0) — чирп всё равно стартует.")
    else:
        print(f"Обнуление подтверждено для моторов {motor_ids}.")


def run_chirp_test() -> None:
    active_ids = [MOTOR_IDS[slot] for slot in ACTIVE_SLOTS]
    logger = FastLogger()

    with DamiaoMotorBus(rx_callback=logger.on_state) as bus:
        for mid in active_ids:
            bus.enable(mid)
        time.sleep(0.2)

        zero_motors(bus, active_ids)

        period = 1.0 / CONTROL_RATE_HZ
        t_start = time.perf_counter()
        wall_t0 = time.time()   # для сопоставления с state.timestamp на графике
        next_tick = t_start

        while True:
            t = time.perf_counter() - t_start
            if t > DURATION:
                break

            for slot in ACTIVE_SLOTS:
                mid = MOTOR_IDS[slot]
                target = chirp_target(t, amplitude=AMPLITUDES[slot])
                clamped = clamp_to_joint_limit(slot, target)
                bus.send_mit(mid, pos=clamped, kp=KP, kd=KD)
                logger.on_command(mid, t, clamped)

            next_tick += period
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        for mid in active_ids:
            bus.disable(mid)

    logger.save(LOG_DIR)
    logger.plot(active_ids, wall_t0, LOG_DIR / "chirp_plot.png")
    logger.print_rates(active_ids)
    logger.print_latency(active_ids)


if __name__ == "__main__":
    run_chirp_test()
