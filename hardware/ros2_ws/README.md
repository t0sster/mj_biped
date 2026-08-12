# ros2_ws — воркспейс Raspberry Pi

Собирает данные со всего железа робота и отдаёт их на ПК по ROS2.

```
ros2_ws/src/
  tinker_msgs/      сообщения (общие с ПК: LowCmd, LowState, IMUState, ...)
  biped_hardware/   нода: 10 моторов Damiao по CAN + IMU HWT906 по UART
```

## Топики

| Топик | Тип | Направление |
|---|---|---|
| `/low_level_cmd` | `LowCmd` | ПК → Raspberry Pi, MIT-команды на 10 моторов |
| `/control_command` | `ControlCmd` | ПК → Raspberry Pi, enable / disable / set_zero / clear_error |
| `/low_level_state_real` | `LowState` | Raspberry Pi → ПК, состояние моторов + IMU, 100 Гц |

Имена топиков совпадают с примером `laptop-node/test_talker`, так что ПК-нода
работает с этой нодой без правок.

## Сборка

На Raspberry Pi ставим зависимости и собираем:

```bash
pip install python-can pyserial
cd hardware/ros2_ws
colcon build
source install/setup.bash
```

На ПК достаточно собрать только `tinker_msgs` — типы сообщений должны совпадать
с теми, что на роботе.

## Запуск

Перед запуском поднимаем CAN-интерфейс (MCP2515):

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0
```

Затем сама нода:

```bash
ros2 run biped_hardware hardware_node
# или
ros2 launch biped_hardware hardware.launch.py
```

Раз в секунду нода печатает, сколько команд, фидбеков моторов и пакетов IMU она
приняла за секунду — по этим числам сразу видно, что связь живая.

## Настройка

Все параметры — константами вверху файлов, ROS-параметры не используются:

- `biped_hardware/hardware_node.py` — `MOTOR_IDS` (CAN ID моторов и их порядок в
  сообщениях), `PUBLISH_RATE_HZ`, `USE_IMU`;
- `biped_hardware/damiao_can.py` — `CAN_CHANNEL`, `CAN_MASTER_ID`, диапазоны
  MIT-режима (должны совпадать с настройками в конфигураторе мотора);
- `biped_hardware/hwt906_imu.py` — `IMU_PORT`, `IMU_BAUDRATE`.

Единицы измерения IMU приведены к принятым в ROS: `accelerometer` — м/с²,
`gyroscope` — рад/с, `rpy` — рад, `quaternion` — `[w, x, y, z]`. Если датчик не
настроен на выдачу кватерниона (пакет `0x59`), он считается из углов Эйлера.

## Проверка без ROS

Модули запускаются и по отдельности — удобно, когда надо понять, отвечает ли
конкретная железка:

```bash
python3 src/biped_hardware/biped_hardware/hwt906_imu.py   # поток углов с IMU
python3 src/biped_hardware/biped_hardware/damiao_can.py   # enable/MIT/disable одного мотора
```

Стендовые скрипты для моторов (чирп, обнуление, аварийное выключение) лежат
в `hardware/mcp2515/` и используют драйвер из этого пакета, так что перед их
запуском нужно `source install/setup.bash`.
