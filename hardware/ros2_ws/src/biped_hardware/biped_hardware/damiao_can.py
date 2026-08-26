import math
import struct
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable

import can

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — меняйте здесь
# ─────────────────────────────────────────────────────────────────────────────

CAN_CHANNEL = 'can0'    # сетевой интерфейс SocketCAN
CAN_MASTER_ID = 0       # Master ID (Frame ID обратной связи, задаётся в 调试助手)

# CAN ID моторов задаются там, где драйвер используется:
# в hardware_node.MOTOR_IDS (ROS2) и в MOTOR_IDS стендовых скриптов mcp2515/.

# Диапазоны MIT mode (должны совпадать с настройками в 调试助手)
P_MIN, P_MAX   = -12.5, 12.5   # рад
V_MIN, V_MAX   = -45.0, 45.0   # рад/с
KP_MIN, KP_MAX =   0.0, 500.0
KD_MIN, KD_MAX =   0.0, 5.0
T_MIN, T_MAX   = -18.0, 18.0   # Нм

# ─────────────────────────────────────────────────────────────────────────────
# Константы протокола
# ─────────────────────────────────────────────────────────────────────────────

FRAME_MIT       = 0x000   # Frame ID = motor_id
FRAME_POS_SPEED = 0x100   # Frame ID = 0x100 + motor_id
FRAME_SPEED     = 0x200   # Frame ID = 0x200 + motor_id

CMD_ENABLE      = 0xFC    # включить мотор
CMD_DISABLE     = 0xFD    # выключить мотор
CMD_SET_ZERO    = 0xFE    # сохранить текущую позицию как ноль
CMD_CLEAR_ERROR = 0xFB    # сбросить ошибки

# Правдоподобные температуры мотора, °C. Значения вне диапазона означают,
# что кадр пришёл искажённым — такому кадру нельзя верить целиком.
TEMPERATURE_MIN = -40
TEMPERATURE_MAX = 150

STATUS_MAP = {
    0x0: 'DISABLING',
    0x1: 'ENABLE',
    0x8: 'OVERVOLTAGE',
    0x9: 'LOWVOLTAGE',
    0xA: 'OVERCURRENT',
    0xB: 'OVERTEMPERATURE_MOSFET',
    0xC: 'OVERTEMPERATURE_ROTOR',
    0xD: 'LOSS_CONNECTION',
    0xE: 'OVERLOAD',
}

# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotorState:
    motor_id:           int   = 0
    position:           float = 0.0   # рад
    velocity:           float = 0.0   # рад/с
    torque:             float = 0.0   # Нм
    temperature_mosfet: int   = 0     # °C
    temperature_rotor:  int   = 0     # °C
    error:              int   = 0     # raw ERR nibble
    status_str:         str   = 'UNKNOWN'
    timestamp:          float = field(default_factory=time.time)

    def is_enabled(self)  -> bool: return self.error == 0x1
    def has_error(self)   -> bool: return self.error not in (0x0, 0x1)


@dataclass
class MotorCmd:
    position: float = 0.0
    velocity: float = 0.0
    torque:   float = 0.0
    kp:       float = 0.0
    kd:       float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Кодирование / декодирование
# ─────────────────────────────────────────────────────────────────────────────

def _f2u(x, x_min, x_max, bits):
    # NaN/inf не отбрасываются сравнением — превращаем в безопасный 0
    # до клампа, иначе NaN тихо становится максимумом диапазона
    if not math.isfinite(x):
        x = 0.0
    x = max(x_min, min(x_max, x))
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))

def _u2f(x_int, x_min, x_max, bits):
    return float(x_int) * (x_max - x_min) / ((1 << bits) - 1) + x_min


def _pack_mit(motor_id, pos, vel, kp, kd, torq) -> can.Message:
    """MIT control frame (документ стр. 33–34). Frame ID = motor_id."""
    p  = _f2u(pos,  P_MIN,  P_MAX,  16)
    v  = _f2u(vel,  V_MIN,  V_MAX,  12)
    kp_ = _f2u(kp, KP_MIN, KP_MAX, 12)
    kd_ = _f2u(kd, KD_MIN, KD_MAX, 12)
    t  = _f2u(torq, T_MIN,  T_MAX,  12)
    data = [
        (p >> 8) & 0xFF,
        p & 0xFF,
        (v >> 4) & 0xFF,
        ((v & 0xF) << 4) | ((kp_ >> 8) & 0xF),
        kp_ & 0xFF,
        (kd_ >> 4) & 0xFF,
        ((kd_ & 0xF) << 4) | ((t >> 8) & 0xF),
        t & 0xFF,
    ]
    return can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)


def _pack_pos_speed(motor_id, pos, vel) -> can.Message:
    """Position/Speed frame (документ стр. 35). Frame ID = 0x100 + motor_id."""
    data = list(struct.pack('<ff', pos, vel))
    return can.Message(arbitration_id=FRAME_POS_SPEED + motor_id,
                       data=data, is_extended_id=False)


def _pack_speed(motor_id, vel) -> can.Message:
    """Speed frame (документ стр. 36). Frame ID = 0x200 + motor_id."""
    data = list(struct.pack('<f', vel))
    return can.Message(arbitration_id=FRAME_SPEED + motor_id,
                       data=data, is_extended_id=False)


def _pack_cmd(motor_id, cmd_byte) -> can.Message:
    """Служебные команды (документ стр. 37). Frame ID = motor_id."""
    return can.Message(
        arbitration_id=motor_id,
        data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, cmd_byte],
        is_extended_id=False,
    )


def decode_feedback(msg: can.Message) -> Optional[MotorState]:
    """
    Feedback frame (документ стр. 32).
    D[0]: ERR[7:4] | ID[3:0]
    D[1-2]: POS[15:0]
    D[3]: VEL[11:4]
    D[4]: VEL[3:0] | T[11:8]
    D[5]: T[7:0]
    D[6]: T_MOS
    D[7]: T_Rotor
    """
    if len(msg.data) < 8:
        return None
    d = msg.data

    motor_id = d[0] & 0xF
    err      = (d[0] >> 4) & 0xF

    p_int = (d[1] << 8) | d[2]
    v_int = (d[3] << 4) | ((d[4] >> 4) & 0xF)
    t_int = ((d[4] & 0xF) << 8) | d[5]

    return MotorState(
        motor_id=motor_id,
        position=_u2f(p_int, P_MIN, P_MAX, 16),
        velocity=_u2f(v_int, V_MIN, V_MAX, 12),
        torque=_u2f(t_int, T_MIN, T_MAX, 12),
        temperature_mosfet=d[6],
        temperature_rotor=d[7],
        error=err,
        status_str=STATUS_MAP.get(err, f'UNKNOWN(0x{err:X})'),
        timestamp=time.time(),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────────────────────────────────────

class DamiaoMotorBus:
    """
    Управление моторами Damiao через CAN (MCP2515 / SocketCAN).

    Быстрый старт:
        bus = DamiaoMotorBus()
        bus.enable(1)
        bus.send_mit(1, pos=0.5, kp=30.0, kd=1.0)
        bus.disable(1)
        bus.close()
    """

    # cmd_byte → имя метода, который его обрабатывает
    _CONTROL_HANDLERS = {
        CMD_ENABLE:      'enable',
        CMD_DISABLE:     'disable',
        CMD_SET_ZERO:    'set_zero_position',
        CMD_CLEAR_ERROR: 'clear_error',
    }

    def __init__(
        self,
        rx_callback: Optional[Callable[[MotorState], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ):
        self.rx_callback = rx_callback
        self.error_callback = error_callback
        self.states: Dict[int, MotorState] = {}

        # диагностика шины, растёт всё время работы
        self.tx_frame_count = 0      # отправлено кадров
        self.rx_frame_count = 0      # принято кадров, включая чужие
        self.tx_error_count = 0      # не удалось отправить кадр
        self.error_frame_count = 0   # шина прислала кадр ошибки
        self.bad_frame_count = 0     # фидбек с невозможными значениями
        self.last_error_text = ''
        self.other_frame_ids = set()  # ID кадров, которые мы отбросили как чужие

        self._bus = can.interface.Bus(
            channel=CAN_CHANNEL,
            interface='socketcan',
        )
        self._notifier = can.Notifier(self._bus, [self._on_message])
        logger.info(f'DamiaoMotorBus: {CAN_CHANNEL}, master_id={CAN_MASTER_ID}')

    # ── Служебные команды ─────────────────────────────────────────────────────

    def enable(self, motor_id: int) -> None:
        """Включить мотор. Обязательно перед отправкой команд движения."""
        self._send(_pack_cmd(motor_id, CMD_ENABLE))
        logger.info(f'[{motor_id}] ENABLE')

    def disable(self, motor_id: int) -> None:
        """Выключить мотор."""
        self._send(_pack_cmd(motor_id, CMD_DISABLE))
        logger.info(f'[{motor_id}] DISABLE')

    def set_zero_position(self, motor_id: int) -> None:
        """Обнулить позицию (сохранить текущую как ноль)."""
        self._send(_pack_cmd(motor_id, CMD_SET_ZERO))
        logger.info(f'[{motor_id}] SET_ZERO')

    def clear_error(self, motor_id: int) -> None:
        """Сбросить ошибки мотора."""
        self._send(_pack_cmd(motor_id, CMD_CLEAR_ERROR))
        logger.info(f'[{motor_id}] CLEAR_ERROR')

    # ── Команды движения ──────────────────────────────────────────────────────

    def send_mit(
        self,
        motor_id: int,
        pos: float,
        vel: float  = 0.0,
        kp:  float  = 30.0,
        kd:  float  = 1.0,
        torq: float = 0.0,
    ) -> None:
        """
        MIT mode: позиция + скорость feedforward + Kp/Kd + момент feedforward.
        pos  ∈ [±12.5 рад], vel ∈ [±45 рад/с], torq ∈ [±18 Нм]
        kp   ∈ [0..500],    kd  ∈ [0..5]
        """
        self._send(_pack_mit(motor_id, pos, vel, kp, kd, torq))

    def send_position(self, motor_id: int, pos: float, vel: float = 0.0) -> None:
        """
        Position/Speed mode: трапецеидальный профиль.
        pos — цель в радианах, vel — макс. скорость равномерного участка.
        """
        self._send(_pack_pos_speed(motor_id, pos, vel))

    def send_speed(self, motor_id: int, vel: float) -> None:
        """Speed mode: только заданная скорость (рад/с)."""
        self._send(_pack_speed(motor_id, vel))

    # ── ROS2-совместимые обёртки ──────────────────────────────────────────────

    def send_motor_cmd(self, motor_id: int, cmd: MotorCmd) -> None:
        """Отправить MotorCmd напрямую из ROS2 callback."""
        self.send_mit(motor_id, cmd.position, cmd.velocity,
                      cmd.kp, cmd.kd, cmd.torque)

    def handle_control_cmd(self, motor_id: int, cmd_byte: int) -> None:
        """
        Обработать ControlCmd.cmd:
          251 (0xFB) → clear_error
          252 (0xFC) → enable
          253 (0xFD) → disable
          254 (0xFE) → set_zero_position
        """
        handler_name = self._CONTROL_HANDLERS.get(cmd_byte)
        if handler_name is None:
            logger.warning(f'Неизвестная команда: 0x{cmd_byte:02X}')
            return
        getattr(self, handler_name)(motor_id)

    def get_state(self, motor_id: int) -> Optional[MotorState]:
        return self.states.get(motor_id)

    def get_bus_state_text(self) -> str:
        """
        Состояние контроллера CAN: ACTIVE — норма, PASSIVE/ERROR — много ошибок
        на шине (обрыв, нет терминаторов, никто не отвечает).
        """
        try:
            return str(self._bus.state).replace('BusState.', '')
        except NotImplementedError:
            return 'UNKNOWN'

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _send(self, msg: can.Message) -> None:
        try:
            self._bus.send(msg)
            self.tx_frame_count += 1
            logger.debug(f'TX  0x{msg.arbitration_id:03X}  '
                         f'{msg.data.hex(" ").upper()}')
        except can.CanError as e:
            # частый случай — 'No buffer space available': кадры уходят быстрее,
            # чем MCP2515 успевает их отдавать в шину
            self.tx_error_count += 1
            self._report_error(str(e))

    def _on_message(self, msg: can.Message) -> None:
        # кадр ошибки шины: обрыв, короткое замыкание, нет второго узла и т.п.
        if msg.is_error_frame:
            self.error_frame_count += 1
            self._report_error(f'error frame 0x{msg.arbitration_id:08X}')
            return

        self.rx_frame_count += 1

        # фидбек приходит с Frame ID = CAN_MASTER_ID, всё прочее не наше.
        # ID чужих кадров запоминаем: если моторы отвечают, а фидбека нет,
        # значит в моторах прошит другой Master ID и его видно в этом списке
        if msg.arbitration_id != CAN_MASTER_ID:
            self.other_frame_ids.add(msg.arbitration_id)
            return
        state = decode_feedback(msg)
        if state is None:
            return

        # искажённый кадр: температуры вне физически возможных значений.
        # Позициям из такого кадра тоже верить нельзя, поэтому пропускаем целиком
        if not (TEMPERATURE_MIN <= state.temperature_mosfet <= TEMPERATURE_MAX
                and TEMPERATURE_MIN <= state.temperature_rotor <= TEMPERATURE_MAX):
            self.bad_frame_count += 1
            self._report_error(
                f'мотор {state.motor_id}: странная температура '
                f'{state.temperature_mosfet}/{state.temperature_rotor} °C, кадр пропущен'
            )
            return

        self.states[state.motor_id] = state
        logger.debug(f'RX  motor={state.motor_id}  '
                     f'pos={state.position:+.3f}  '
                     f'vel={state.velocity:+.3f}  '
                     f'torq={state.torque:+.3f}  '
                     f'{state.status_str}')
        if self.rx_callback:
            self.rx_callback(state)

    def _report_error(self, error_text: str) -> None:
        # одна и та же ошибка сыплется сотнями в секунду — пишем только новую
        if error_text != self.last_error_text:
            logger.error(f'CAN: {error_text}')
        self.last_error_text = error_text
        if self.error_callback:
            self.error_callback(error_text)

    def close(self) -> None:
        self._notifier.stop()
        self._bus.shutdown()

    def __enter__(self):  return self
    def __exit__(self, *a): self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Ручное тестирование
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    def on_state(s: MotorState):
        print(f'  ← [{s.motor_id}] pos={s.position:+.3f} rad  '
              f'vel={s.velocity:+.3f} rad/s  torq={s.torque:+.3f} Nm  '
              f'T_mos={s.temperature_mosfet}°C  {s.status_str}')

    MOTOR = 1   # ← ID мотора для теста

    with DamiaoMotorBus(rx_callback=on_state) as bus:
        print('ENABLE')
        bus.enable(MOTOR)
        time.sleep(0.5)

        print('MIT → 0.5 рад')
        bus.send_mit(MOTOR, pos=0.5, kp=30.0, kd=1.0)
        time.sleep(2.0)

        print('SET_ZERO')
        bus.set_zero_position(MOTOR)
        time.sleep(0.5)

        print('DISABLE')
        bus.disable(MOTOR)
        time.sleep(0.3)
