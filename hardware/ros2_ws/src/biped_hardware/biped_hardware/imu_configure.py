#!/usr/bin/env python3
"""
Настройка того, какие пакеты шлёт IMU HWT906.

Нужна, если датчик не отдаёт часть данных — например, приходят только пакеты
0x52 (гироскоп) и 0x53 (углы), а ускорения (0x51) нет, и accelerometer
в топике всегда нулевой.

Скрипт пишет настройку в датчик и сохраняет её в его памяти — то есть меняет
состояние самого устройства, а не только текущий сеанс. Запускать осознанно:

    ros2 run biped_hardware imu_configure

После записи проверить, что пакет 0x51 появился:

    python3 -u src/biped_hardware/biped_hardware/hwt906_imu.py
"""

import time

import serial

from biped_hardware.hwt906_imu import (
    IMU_BAUDRATE,
    IMU_PORT,
    PACKET_HEADER,
    PACKET_SIZE,
    parse_packet,
)

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

# Какие пакеты датчик должен слать. Чем больше типов, тем плотнее поток по UART.
# Ноде нужны accel + gyro + angle; quaternion можно не включать — тогда он
# считается из углов Эйлера (см. hwt906_imu.py).
OUTPUT_PACKETS = ["accel", "gyro", "angle"]

# ─────────────────────────────────────────────────────────────────────────────
# Константы протокола WitMotion
# ─────────────────────────────────────────────────────────────────────────────

# Каждому типу пакета соответствует свой бит в регистре RSW.
PACKET_BITS = {
    "time": 1 << 0,         # 0x50
    "accel": 1 << 1,        # 0x51
    "gyro": 1 << 2,         # 0x52
    "angle": 1 << 3,        # 0x53
    "magnetometer": 1 << 4,  # 0x54
    "quaternion": 1 << 9,   # 0x59
}

REGISTER_SAVE = 0x00   # сохранить настройки в памяти датчика
REGISTER_RSW = 0x02    # набор выводимых пакетов
REGISTER_READ = 0x27   # запрос значения регистра

PKG_REGISTER_REPLY = 0x5F   # ответ датчика на запрос значения регистра

UNLOCK_COMMAND = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])

READ_TIMEOUT = 1.0     # с, сколько ждём ответ датчика на запрос регистра


def write_register(port: serial.Serial, register: int, value: int) -> None:
    """Команда записи регистра: FF AA <регистр> <младший байт> <старший байт>."""
    # разблокировка действует недолго, поэтому шлём её перед каждой записью
    port.write(UNLOCK_COMMAND)
    port.flush()
    time.sleep(0.2)

    command = bytes([0xFF, 0xAA, register, value & 0xFF, (value >> 8) & 0xFF])
    port.write(command)
    port.flush()
    time.sleep(0.2)   # датчику нужно время на обработку команды


def read_register(port: serial.Serial, register: int):
    """
    Запрашивает значение регистра и ждёт ответ (пакет 0x5F).
    Возвращает число или None, если датчик не ответил.
    """
    port.reset_input_buffer()
    port.write(bytes([0xFF, 0xAA, REGISTER_READ, register, 0x00]))
    port.flush()

    buffer = bytearray()
    deadline = time.time() + READ_TIMEOUT

    while time.time() < deadline:
        byte = port.read(1)
        if not byte:
            continue

        if not buffer and byte[0] != PACKET_HEADER:
            continue

        buffer += byte
        if len(buffer) < PACKET_SIZE:
            continue

        result = parse_packet(bytes(buffer))
        buffer.clear()

        # в ответе лежат 4 регистра подряд, нам нужен первый — запрошенный
        if result is not None and result[0] == PKG_REGISTER_REPLY:
            return result[1][0] & 0xFFFF

    return None


def main() -> None:
    rsw_value = 0
    for packet_name in OUTPUT_PACKETS:
        rsw_value |= PACKET_BITS[packet_name]

    port = serial.Serial(IMU_PORT, IMU_BAUDRATE, timeout=1)
    print(f"Открыт {IMU_PORT} @ {IMU_BAUDRATE}")

    try:
        print(f"Включаю пакеты {OUTPUT_PACKETS}, RSW = 0x{rsw_value:04X}")
        write_register(port, REGISTER_RSW, rsw_value)
        write_register(port, REGISTER_SAVE, 0x0000)
        time.sleep(0.5)   # после сохранения датчик применяет настройки

        # читаем регистр обратно — только так видно, что датчик нас услышал
        actual_value = read_register(port, REGISTER_RSW)

        if actual_value is None:
            print(
                "Датчик не ответил на запрос регистра.\n"
                "Скорее всего команды до него не доходят: проверьте, что TX "
                "Raspberry Pi подключён к RX датчика (для приёма данных хватает "
                "одного провода RX, а для настройки нужен и второй)."
            )
        elif actual_value == rsw_value:
            print(f"Готово: в датчике RSW = 0x{actual_value:04X}")
        else:
            print(
                f"Настройка не применилась: в датчике RSW = 0x{actual_value:04X}, "
                f"ожидали 0x{rsw_value:04X}"
            )
    finally:
        port.close()


if __name__ == "__main__":
    main()
