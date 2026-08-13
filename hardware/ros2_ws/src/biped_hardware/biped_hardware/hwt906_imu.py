#!/usr/bin/env python3
"""
Парсер IMU WitMotion HWT906 (UART) + фоновое чтение порта.

Датчик сам, без запроса, шлёт пакеты по 11 байт:
    0x55, тип пакета, 8 байт данных, контрольная сумма

Класс Hwt906Imu читает порт в отдельном потоке и хранит последние значения,
чтобы ROS-нода могла забрать их в любой момент (см. hardware_node.py).

Зависимости:  pip install pyserial

Проверка порта (Raspberry Pi):
  sudo raspi-config  → отключить консоль на serial, включить сам serial port
  ls -l /dev/ttyAMA0
"""

import math
import struct
import threading

import serial

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

IMU_PORT = "/dev/ttyAMA0"
IMU_BAUDRATE = 921600

# ─────────────────────────────────────────────────────────────────────────────
# Константы протокола
# ─────────────────────────────────────────────────────────────────────────────

PACKET_HEADER = 0x55
PACKET_SIZE = 11

PKG_ACCEL = 0x51        # ускорение + температура
PKG_GYRO = 0x52         # угловая скорость
PKG_ANGLE = 0x53        # углы roll/pitch/yaw
PKG_MAG = 0x54          # магнитометр
PKG_QUATERNION = 0x59   # кватернион (если включён в настройках датчика)

GRAVITY = 9.80665       # м/с², для перевода из g


def parse_packet(packet: bytes):
    """
    Разбирает один пакет из 11 байт.
    Возвращает (тип пакета, список из 4 чисел) или None, если пакет битый.
    """
    if len(packet) != PACKET_SIZE or packet[0] != PACKET_HEADER:
        return None

    # контрольная сумма — сумма первых 10 байт по модулю 256
    checksum = sum(packet[0:10]) & 0xFF
    if checksum != packet[10]:
        return None

    packet_type = packet[1]
    raw_values = struct.unpack("<4h", packet[2:10])
    return packet_type, raw_values


def quaternion_from_rpy(roll: float, pitch: float, yaw: float):
    """Кватернион [w, x, y, z] из углов Эйлера в радианах."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


class Hwt906Imu:
    """
    Чтение HWT906 в фоновом потоке.

    Последние данные лежат в полях объекта — единицы измерения как в ROS:
        accelerometer  — м/с²
        gyroscope      — рад/с
        rpy            — рад
        quaternion     — [w, x, y, z]
        temperature    — °C

    Быстрый старт:
        imu = Hwt906Imu()
        print(imu.gyroscope)
        imu.close()
    """

    def __init__(self, port: str = IMU_PORT, baudrate: int = IMU_BAUDRATE):
        self.accelerometer = [0.0, 0.0, 0.0]
        self.gyroscope = [0.0, 0.0, 0.0]
        self.rpy = [0.0, 0.0, 0.0]
        self.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.temperature = 0.0
        self.packet_count = 0          # сколько корректных пакетов принято
        self.packet_counts = {}        # то же самое, но по типам пакетов: {тип: сколько}
        self.last_error_text = ''      # если чтение порта упало — текст ошибки

        # датчик может не присылать кватернион — тогда считаем его из углов
        self._quaternion_from_sensor = False

        self._serial = serial.Serial(port, baudrate, timeout=1)
        self._is_running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    # ── Фоновое чтение ────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Собирает байты в пакеты по 11 штук и обновляет поля объекта."""
        try:
            self._read_packets()
        except Exception as error:
            # без этого поток умирает молча и данные просто перестают идти
            self.last_error_text = str(error)
            print(f"IMU: чтение порта прервано — {error}")

    def _read_packets(self) -> None:
        buffer = bytearray()

        while self._is_running:
            byte = self._serial.read(1)
            if not byte:
                continue

            # начало пакета — всегда 0x55, всё до него пропускаем
            if not buffer and byte[0] != PACKET_HEADER:
                continue

            buffer += byte
            if len(buffer) < PACKET_SIZE:
                continue

            result = parse_packet(bytes(buffer))
            buffer.clear()
            if result is not None:
                packet_type, raw_values = result
                self._update_values(packet_type, raw_values)
                self.packet_count += 1
                self.packet_counts[packet_type] = \
                    self.packet_counts.get(packet_type, 0) + 1

    def _update_values(self, packet_type: int, raw_values) -> None:
        """Переводит сырые int16 в физические величины."""
        v0, v1, v2, v3 = raw_values

        if packet_type == PKG_ACCEL:
            # ±16 g на весь диапазон int16
            self.accelerometer = [
                v0 / 32768.0 * 16.0 * GRAVITY,
                v1 / 32768.0 * 16.0 * GRAVITY,
                v2 / 32768.0 * 16.0 * GRAVITY,
            ]
            self.temperature = v3 / 100.0

        elif packet_type == PKG_GYRO:
            # ±2000 °/с на весь диапазон int16
            self.gyroscope = [
                math.radians(v0 / 32768.0 * 2000.0),
                math.radians(v1 / 32768.0 * 2000.0),
                math.radians(v2 / 32768.0 * 2000.0),
            ]

        elif packet_type == PKG_ANGLE:
            # ±180° на весь диапазон int16
            self.rpy = [
                math.radians(v0 / 32768.0 * 180.0),
                math.radians(v1 / 32768.0 * 180.0),
                math.radians(v2 / 32768.0 * 180.0),
            ]
            if not self._quaternion_from_sensor:
                self.quaternion = quaternion_from_rpy(*self.rpy)

        elif packet_type == PKG_QUATERNION:
            self.quaternion = [
                v0 / 32768.0,
                v1 / 32768.0,
                v2 / 32768.0,
                v3 / 32768.0,
            ]
            self._quaternion_from_sensor = True

    # ── Завершение ────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._is_running = False
        self._thread.join(timeout=1.0)
        self._serial.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Ручное тестирование:  python3 hwt906_imu.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    with Hwt906Imu() as imu:
        print(f"Открыт {IMU_PORT} @ {IMU_BAUDRATE}")
        try:
            while True:
                time.sleep(0.5)
                roll, pitch, yaw = imu.rpy

                # какие типы пакетов реально шлёт датчик: нужны 0x51, 0x52, 0x53.
                # если какого-то типа нет — он отключён в настройках датчика
                types = " ".join(
                    f"0x{ptype:02X}:{count}"
                    for ptype, count in sorted(imu.packet_counts.items())
                )

                print(
                    f"rpy=({math.degrees(roll):+7.2f} "
                    f"{math.degrees(pitch):+7.2f} "
                    f"{math.degrees(yaw):+7.2f})°  "
                    f"gyro={[f'{g:+.3f}' for g in imu.gyroscope]}  "
                    f"accel={[f'{a:+.2f}' for a in imu.accelerometer]}  "
                    f"T={imu.temperature:.1f}°C  "
                    f"пакеты: {types}"
                )
        except KeyboardInterrupt:
            print("\nОстановлено пользователем")
