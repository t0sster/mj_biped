#!/usr/bin/env python3
"""
Минимальный парсер протокола WitMotion (HWT906) через UART.
Baud rate по умолчанию у HWT906: 921600.

pip install pyserial
"""

import serial
import struct

PORT = "/dev/ttyAMA0"
BAUD = 921600

# Коды типов пакетов WitMotion
PKG_ACCEL = 0x51   # ускорение
PKG_GYRO  = 0x52   # угловая скорость
PKG_ANGLE = 0x53   # углы (roll/pitch/yaw)
PKG_MAG   = 0x54   # магнитометр

def parse_packet(data: bytes):
    """data — 11 байт: 0x55, type, 8 байт полезной нагрузки, checksum"""
    if len(data) != 11 or data[0] != 0x55:
        return None

    # проверка контрольной суммы (сумма первых 10 байт по модулю 256)
    checksum = sum(data[0:10]) & 0xFF
    if checksum != data[10]:
        return None  # битый пакет, отбрасываем

    ptype = data[1]
    payload = data[2:10]

    # 4 int16 little-endian значения на пакет
    v0, v1, v2, v3 = struct.unpack("<4h", payload)

    if ptype == PKG_ACCEL:
        ax = v0 / 32768.0 * 16  # g
        ay = v1 / 32768.0 * 16
        az = v2 / 32768.0 * 16
        temp = v3 / 100.0
        return ("ACCEL", {"ax": ax, "ay": ay, "az": az, "temp": temp})

    elif ptype == PKG_GYRO:
        gx = v0 / 32768.0 * 2000  # deg/s
        gy = v1 / 32768.0 * 2000
        gz = v2 / 32768.0 * 2000
        return ("GYRO", {"gx": gx, "gy": gy, "gz": gz})

    elif ptype == PKG_ANGLE:
        roll  = v0 / 32768.0 * 180
        pitch = v1 / 32768.0 * 180
        yaw   = v2 / 32768.0 * 180
        return ("ANGLE", {"roll": roll, "pitch": pitch, "yaw": yaw})

    elif ptype == PKG_MAG:
        mx, my, mz = v0, v1, v2
        return ("MAG", {"mx": mx, "my": my, "mz": mz})

    return (f"TYPE_0x{ptype:02X}", {"raw": payload.hex()})


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Открыт {PORT} @ {BAUD}")

    buf = bytearray()

    try:
        while True:
            byte = ser.read(1)
            if not byte:
                continue

            # ищем начало пакета
            if not buf and byte[0] != 0x55:
                continue

            buf += byte

            if len(buf) == 11:
                result = parse_packet(bytes(buf))
                if result:
                    name, values = result
                    print(name, values)
                buf.clear()

    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
