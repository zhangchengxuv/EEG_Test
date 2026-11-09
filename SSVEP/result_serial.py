"""Independent result link: ASCII lines, 8N1, no flow control, no GUI dependencies."""
from dataclasses import dataclass
import math
import re
import threading
import time

import serial

TARGET_LABELS = {0: "无有效结果/停止", 1: "左肩", 2: "右肩", 3: "左肘", 4: "右肘"}
RESULT_MODES = ("EEG", "TEST", "SIM")
MAX_LINE_BYTES = 128


@dataclass(frozen=True)
class ResultMessage:
    sequence: int
    mode: str
    target_id: int
    frequency_hz: float


def _validate(mode, target_id, frequency_hz):
    if mode not in RESULT_MODES:
        raise ValueError("结果模式必须是 EEG、TEST 或 SIM")
    if type(target_id) is not int or target_id not in TARGET_LABELS:
        raise ValueError("目标编号必须是 0–4")
    if not math.isfinite(frequency_hz) or frequency_hz < 0:
        raise ValueError("频率必须是有限非负数")
    if (target_id == 0) != (frequency_hz == 0):
        raise ValueError("编号 0 的频率须为 0；编号 1–4 的频率须大于 0")
    if target_id and round(frequency_hz, 2) == 0:
        raise ValueError("目标频率不能在两位小数精度下变成零")


def encode_result(message):
    _validate(message.mode, message.target_id, message.frequency_hz)
    if type(message.sequence) is not int or message.sequence < 1:
        raise ValueError("消息序号必须为正整数")
    frequency = message.frequency_hz if message.target_id else 0.
    line = f"RESULT,{message.sequence},{message.mode},{message.target_id},{frequency:.2f}\r\n".encode("ascii")
    if len(line) > MAX_LINE_BYTES:
        raise ValueError("结果消息过长")
    return line


def parse_result(line):
    if len(line) > MAX_LINE_BYTES:
        raise ValueError("结果消息过长")
    try:
        fields = line.decode("ascii").rstrip("\r\n").split(",")
        if len(fields) != 5 or fields[0] != "RESULT":
            raise ValueError("结果帧格式错误")
        if not fields[1].isascii() or not fields[1].isdigit() or not re.fullmatch(r"[0-4]", fields[3]):
            raise ValueError("编号格式错误")
        if not re.fullmatch(r"[0-9]+\.[0-9]{2}", fields[4]):
            raise ValueError("频率格式错误")
        message = ResultMessage(int(fields[1]), fields[2], int(fields[3]), float(fields[4]))
        encode_result(message)  # Shared semantic validation.
        return message
    except (UnicodeError, OverflowError) as exc:
        raise ValueError("结果帧必须是有效 ASCII 文本") from exc


class ResultStreamParser:
    """Handle serial fragmentation, coalesced lines and oversized/corrupt input."""
    def __init__(self):
        self.pending = bytearray()
        self.discarding = False
        self.errors = 0

    def feed(self, chunk):
        messages = []
        for value in chunk:
            if self.discarding:
                if value == 10:
                    self.discarding = False
                continue
            self.pending.append(value)
            if len(self.pending) > MAX_LINE_BYTES:
                self.pending.clear()
                self.errors += 1
                self.discarding = value != 10
            elif value == 10:
                try:
                    messages.append(parse_result(bytes(self.pending)))
                except ValueError:
                    self.errors += 1
                self.pending.clear()
        return messages


def same_serial_port(first, second):
    def normalize(value):
        value = str(value).strip().upper()
        if value.startswith("\\\\.\\"):
            value = value[4:]
        match = re.fullmatch(r"COM0*([0-9]+)", value)
        return f"COM{int(match[1])}" if match else value
    return bool(str(first).strip() and str(second).strip() and normalize(first) == normalize(second))


def open_result_port(port, baudrate=115200, read_timeout=0):
    # do_not_open allows DTR/RTS to be set before opening; URLs support loopback QA.
    device = serial.serial_for_url(port, do_not_open=True)
    device.baudrate = baudrate
    device.bytesize, device.parity, device.stopbits = serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE
    device.timeout, device.write_timeout = read_timeout, .2
    device.xonxoff = device.rtscts = device.dsrdtr = False
    device.dtr = device.rts = False
    device.open()
    return device


class ResultSerialSender:
    """Single latest state, no backlog; serial I/O never runs in the presentation thread."""
    def __init__(self, port, baudrate=115200, mode="EEG", interval=.5, stale_after=1.5, port_factory=None):
        _validate(mode, 0, 0.)
        if not str(port).strip() or baudrate <= 0 or interval <= 0 or stale_after <= interval:
            raise ValueError("结果串口或发送间隔配置错误")
        self.port, self.baudrate = str(port).strip(), baudrate
        self.interval, self.stale_after = interval, stale_after
        self.port_factory = port_factory or open_result_port
        self.condition = threading.Condition()
        self.state = (mode, 0, 0.)
        self.updated_at = time.perf_counter()
        self.stop_requested = False
        self.thread = None
        self.stats = dict(port=self.port, baudrate=baudrate, connected=False, closing=False,
                          sent_count=0, last_line="", last_sent_monotonic_s=None, error="",
                          protocol="RESULT,sequence,mode,target_id,frequency_hz\\r\\n",
                          heartbeat_seconds=interval, stale_after_seconds=stale_after)

    def start(self):
        if self.thread is not None:
            raise RuntimeError("每次连接应创建新的结果发送器")
        self.thread = threading.Thread(target=self._run, name="SSVEP-result-serial", daemon=True)
        self.thread.start()

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def publish(self, mode, target_id=0, frequency_hz=0., evidence_time=None):
        _validate(mode, target_id, frequency_hz)
        stamp = time.perf_counter() if evidence_time is None else float(evidence_time)
        if not math.isfinite(stamp) or stamp > time.perf_counter()+.01:
            raise ValueError("结果时间戳无效")
        with self.condition:
            if self.stop_requested or stamp < self.updated_at:
                return False
            self.state = mode, target_id, float(frequency_hz)
            self.updated_at = stamp
            self.condition.notify_all()
        return True

    def snapshot(self):
        with self.condition:
            return dict(self.stats, running=self.is_running())

    def request_close(self):
        with self.condition:
            self.state = self.state[0], 0, 0.
            self.stop_requested = True
            self.stats["closing"] = True
            self.condition.notify_all()

    def close(self, timeout=.8):
        self.request_close()
        if self.thread is not None:
            self.thread.join(timeout)
        return not self.is_running()

    def _run(self):
        device = None
        try:
            device = self.port_factory(self.port, self.baudrate)
            with self.condition:
                self.stats["connected"] = True
            previous = None
            next_send = 0.
            while True:
                with self.condition:
                    now = time.perf_counter()
                    stopping = self.stop_requested
                    mode, target, frequency = self.state
                    expires = self.updated_at+self.stale_after
                    effective = (mode, 0, 0.) if stopping or now >= expires else (mode, target, frequency)
                    if not stopping and effective == previous and now < next_send:
                        wake_at = min(next_send, expires) if effective[1] else next_send
                        self.condition.wait(max(.001, wake_at-now))
                        continue
                    sequence = self.stats["sent_count"]+1
                line = encode_result(ResultMessage(sequence, *effective))
                if device.write(line) != len(line):
                    raise serial.SerialTimeoutException("结果消息未完整写入")
                with self.condition:
                    self.stats.update(sent_count=sequence, last_line=line.decode("ascii").strip(),
                                      last_sent_monotonic_s=time.perf_counter())
                previous = effective
                next_send = time.perf_counter()+self.interval
                if stopping:
                    device.flush()  # Background only; finish the final zero before closing.
                    break  # Final zero frame precedes closing, whenever the link still works.
        except (OSError, serial.SerialException, ValueError) as exc:
            with self.condition:
                self.stats["error"] = str(exc)
        finally:
            if device is not None:
                try:
                    device.close()
                except (OSError, serial.SerialException):
                    pass
            with self.condition:
                self.stats.update(connected=False, closing=False)
