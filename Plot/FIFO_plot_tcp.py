# -*- coding: utf-8 -*-
"""
与 Plot/FIFO_plot.py 功能一致：8 路波形、FFT、滤波、中值、UDP 打标、FIF 录制、UI 布局相同。
区别：数据来自 ESP32 无线固件 main_wifi 的 TCP 流（行格式仍为 channel:...，与串口版相同）。

运行（建议在 Plot 目录下）：  python FIFO_plot_tcp.py
固件：platformio 环境 seeed_xiao_esp32s3_wifi；电脑与板子同局域网。首次启动会提示输入板子 IP 与端口。
原 FIFO_plot.py 未修改，可继续使用 USB 串口。
"""
import sys
import os
import re
import time
import json
import socket
import queue
from collections import deque
from datetime import datetime, timezone

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from trigger_protocol import TriggerState, UDP_PORT, TRIG_VOLT_SCALE
import mne
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PyQt5.QtWidgets import QFrame

from qfluentwidgets import CheckBox, ComboBox, PushButton, LineEdit, SpinBox, InfoBar, InfoBarPosition
from qfluentwidgets import FluentIcon as FIF

from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, iirnotch, firwin, get_window
from scipy.signal import medfilt
# ================== 配置参数（可按需修改） ==================
SAMPLE_RATE = 500  # 采样率（与固件 CONFIG1 500 SPS 一致，见 main.cpp）
# CH1..CH8 → 10-20 电极名（与 ADS1299 硬件通道顺序一致）；CH9 为软件 UDP 打标 TRIG（非 µV）
EEG_CHANNEL_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
ALL_CHANNEL_NAMES = EEG_CHANNEL_NAMES + ["TRIG"]
N_SAVE_CHANNELS = 9
UPDATE_INTERVAL = 40  # UI 刷新间隔 ms（TCP 版略加快；仍远低于 500Hz 采样，避免多余负荷可调回 60）
# 状态栏更新上限：全速 channel 时若每行都 emit 会 500 次/秒 拖垮 Qt，与采集抢 CPU
TCP_STATUS_MAX_HZ = 12.0
DATA_LENGTH = 20000  # 环形缓存长度（每通道，增大以适应分时数据）
PLOT_LENGTH = 1000  # 时域显示长度
FFT_LENGTH = 500  # FFT长度（点数）

# 滤波参数
NOTCH_FREQ      = 50.0          # 保持50Hz陷波（工频干扰）
QUALITY_FACTOR  = 30            # 提高品质因数，更精准去除50Hz
LOW_CUTOFF      = 0.5           # 降低低频截止，保留更慢的变化
HIGH_CUTOFF     = 80.0          # 提高高频截止，保留方波边沿
FIR_ORDER       = 61            # 适当降低FIR阶数，减少延迟

# 中值滤波窗口（针对方波信号优化）
MEDIAN_WINDOW_SIZE = 5          # 5-7点（方波信号专用）

# 显示参数
DEFAULT_Y_LIM_UV = 100.0  # 固定Y轴半幅（±μV）

# 固件串口输出为电压 (V)，例如 10µV → 0.000010；本程序内部统一换算为 µV 再滤波/绘图，避免标成 µV 却用 V 的数值（小 10^6 倍）
# 若将来固件改为直接输出 µV，将此项改为 False，切勿与 V→µV 重复换算。
SERIAL_VALUES_IN_VOLTS = True
V_TO_UV = 1.0e6
UV_TO_V = 1.0 / V_TO_UV  # FIF/MNE 通道数据为伏特，须与 V_TO_UV 成互逆


def serial_amplitude_to_uv(values):
    """将解析到的幅度转为 µV（与 SERIAL_VALUES_IN_VOLTS 一致）。"""
    if values is None:
        return None
    if SERIAL_VALUES_IN_VOLTS:
        return [float(v) * V_TO_UV for v in values]
    return [float(v) for v in values]

# ================== 日志配置 ==================
# 与脚本同目录下的 logs（即 Plot/logs），不随当前工作目录变化
_PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_PLOT_DIR, "logs")  # 录制保存目录（.fif）
LOG_MAX_SIZE_MB = 100  # 单文件约当大小上限（按内存缓冲估算，过大则轮转）
LOG_MAX_AGE_HOURS = 24  # 单文件最长时长（小时，与大小二选一先达到者轮转）
LOG_QUEUE_MAX = 10000  # 日志队列最大长度（防内存溢出）
# 单文件最大样本数：受体积与时长双重限制（9 通道 float64）
_MAX_SAMPLES_BY_SIZE = max(1000, int(LOG_MAX_SIZE_MB * 1024 * 1024 / (N_SAVE_CHANNELS * 8)))
_MAX_SAMPLES_BY_TIME = max(1000, int(SAMPLE_RATE * 3600 * LOG_MAX_AGE_HOURS))
MAX_SAMPLES_PER_FILE = min(_MAX_SAMPLES_BY_SIZE, _MAX_SAMPLES_BY_TIME)

class ChannelBuffer:
    """
    修复满缓冲后波形消失问题
    核心思路：
    1. x_axis 从每个通道实际最晚时间戳减去 n，保证不会超出 buffer
    2. np.interp 时限制插值范围在已有数据区间
    """
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.data = [np.zeros(capacity, dtype=np.float32) for _ in range(8)]
        self.timestamps = [np.zeros(capacity, dtype=np.int64) for _ in range(8)]
        self.indices = [0]*8
        self.is_full = [False]*8

    def append(self, channel_id, values: list, start_index=None):
        if channel_id < 0 or channel_id >= 8 or len(values) == 0:
            return

        n = len(values)
        idx = self.indices[channel_id]
        ts = np.arange(start_index, start_index + n) if start_index is not None else np.arange(idx, idx + n)

        if n > self.capacity:
            values = values[-self.capacity:]
            ts = ts[-self.capacity:]
            n = self.capacity

        for i in range(n):
            self.data[channel_id][idx] = values[i]
            self.timestamps[channel_id][idx] = ts[i]
            idx = (idx + 1) % self.capacity
            if idx == 0:
                self.is_full[channel_id] = True
        self.indices[channel_id] = idx

    def get_recent(self, n=PLOT_LENGTH):
        """
        获取最近 n 个点的数据
        修复20000点满后波形消失问题：
        1. 不依赖全局索引计算x轴
        2. 始终返回连续数组
        """
        y_data = np.full((8, n), np.nan, dtype=np.float32)
        x_axis = np.arange(n)  # 绘图直接用 0~n-1

        for ch in range(8):
            idx = self.indices[ch]
            if self.is_full[ch]:
                # 已满，取环形缓冲最后n个点
                data = np.concatenate((self.data[ch][idx:], self.data[ch][:idx]))
            else:
                # 未满，直接取已有数据
                data = self.data[ch][:idx]
            # 截取最后n个点
            y_data[ch, -len(data):] = data[-n:]
        return x_axis, y_data


# ================== UDP 软件打标（与 UI_cue/cue_paradigm 配合） ==================
class TriggerUdpListener(QtCore.QThread):
    """监听 JSON：{\"code\": <int>}，更新 TriggerState；与串口采样合并为第 9 通道。"""

    bind_failed = QtCore.pyqtSignal(str)
    bind_ok = QtCore.pyqtSignal(int)

    def __init__(self, state: TriggerState, port: int, parent=None):
        super().__init__(parent)
        self._state = state
        self._port = port
        self._run_flag = True

    def stop(self):
        self._run_flag = False

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 绑定本机回环（与 cue 发往 127.0.0.1 一致；部分环境下 0.0.0.0 收不到环回包）
            s.bind(("127.0.0.1", self._port))
        except OSError as e:
            self.bind_failed.emit(
                f"端口 {self._port} 绑定失败: {e}（是否已开第二个 FIFO_plot 或其它程序占用？）"
            )
            return
        self.bind_ok.emit(self._port)
        s.settimeout(0.25)
        while self._run_flag:
            try:
                data, _ = s.recvfrom(4096)
                if not data:
                    continue
                msg = json.loads(data.decode("utf-8"))
                if isinstance(msg, dict) and "code" in msg:
                    self._state.set(int(msg["code"]))
            except socket.timeout:
                continue
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError):
                continue
        try:
            s.close()
        except Exception:
            pass


# ================== 异步 FIF 写入线程（MNE Raw） ==================
class LogWriterThread(QtCore.QThread):
    """
    缓冲样本后写入 MNE 可读的 FIF（-raw.fif）：前 8 通道 EEG 为 µV→V；
    第 9 通道 TRIG 为软件打标码×TRIG_VOLT_SCALE（伏特存盘）。
    轮转：样本数达到 MAX_SAMPLES_PER_FILE，或单段时长超过 LOG_MAX_AGE_HOURS。
    """
    log_rotated = QtCore.pyqtSignal(str)  # 发出当前/最新 FIF 路径

    def __init__(self, base_dir: str, log_queue: queue.Queue):
        super().__init__()
        self.base_dir = base_dir
        self.log_queue = log_queue
        self.running = False  # stop() 置 False，便于在 queue 空时退出
        self._current_path = None
        self._rows = []
        self._segment_t0 = None  # 本段第一个样本的 wall time（time.time()）
        self._segment_start_mono = None  # 本段起始 monotonic，用于时长判断

    def _save_fif(self):
        """将当前缓冲写入一个 FIF 文件。"""
        if not self._rows:
            return
        os.makedirs(self.base_dir, exist_ok=True)
        data_uv = np.asarray(self._rows, dtype=np.float64)  # (n_samples, 9)：前 8 为 µV，末列为打标码
        if data_uv.ndim != 2 or data_uv.shape[1] != N_SAVE_CHANNELS:
            self._rows.clear()
            self._segment_t0 = None
            self._segment_start_mono = None
            return
        eeg_v = (data_uv[:, :8].T * UV_TO_V).astype(np.float64)  # (8, n)
        codes = data_uv[:, 8].astype(np.float64)
        trig_v = (codes * TRIG_VOLT_SCALE).reshape(1, -1).astype(np.float64)  # (1, n)
        data_v = np.vstack([eeg_v, trig_v])
        ch_types = ["eeg"] * 8 + ["misc"]
        info = mne.create_info(
            ALL_CHANNEL_NAMES,
            SAMPLE_RATE,
            ch_types=ch_types,
            verbose=False,
        )
        raw = mne.io.RawArray(data_v, info, verbose=False)
        if self._segment_t0 is not None:
            raw.set_meas_date(datetime.fromtimestamp(self._segment_t0, tz=timezone.utc))
        try:
            raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
        except Exception:
            pass

        t0 = self._segment_t0 or time.time()
        fname = f"EEG_{datetime.fromtimestamp(t0, tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}-raw.fif"
        path = os.path.join(self.base_dir, fname)
        raw.save(path, overwrite=True, verbose=False)
        self._current_path = path
        self.log_rotated.emit(path)

        self._rows.clear()
        self._segment_t0 = None
        self._segment_start_mono = None

    def _should_rotate(self):
        if not self._rows:
            return False
        if len(self._rows) >= MAX_SAMPLES_PER_FILE:
            return True
        if self._segment_start_mono is not None:
            if time.monotonic() - self._segment_start_mono >= LOG_MAX_AGE_HOURS * 3600.0:
                return True
        return False

    def run(self):
        self.running = True
        while True:
            try:
                item = self.log_queue.get(timeout=0.2)
            except queue.Empty:
                if not self.running:
                    self._save_fif()
                    return
                continue

            if item is None:
                self._save_fif()
                return

            wall_t, row = item
            if self._segment_t0 is None:
                self._segment_t0 = wall_t
                self._segment_start_mono = time.monotonic()

            self._rows.append(row)

            if self._should_rotate():
                self._save_fif()

    def stop(self):
        self.running = False
        try:
            self.log_queue.put_nowait(None)
        except Exception:
            pass
        self.wait(2000)


# ================== TCP 线程（解析/滤波/缓冲与串口版一致，数据源为 main_wifi） ==================
class TcpThread(QtCore.QThread):
    data_received = QtCore.pyqtSignal(str)  # 原始数据行（用于状态栏）
    connection_status = QtCore.pyqtSignal(str)  # 已连接/失败原因，主线程可显示在状态栏

    def __init__(self, host: str, port: int, log_queue: queue.Queue, trigger_state: TriggerState = None):
        super().__init__()
        self.host = str(host).strip()
        self.port = int(port)
        self._tcp_sock = None
        self.running = False
        self.log_queue = log_queue  # 日志队列
        self._trigger_state = trigger_state

        # 使用通道专用缓冲区（修复核心问题）
        self.buffer = ChannelBuffer(capacity=DATA_LENGTH)
        self.global_index = 0  # 全局时间索引

        # 过滤开关
        self.filter_enabled = True
        self.median_enabled = False  # 中值滤波开关（UI 控制）

        # 滤波器初始化
        self._init_filters()
        self._last_status_emit_t = 0.0  # 限制 data_received 频率，见 TCP_STATUS_MAX_HZ

    # -------- 滤波器设计与状态 --------
    def _init_filters(self):
        nyquist = 0.5 * SAMPLE_RATE

        # 1) 50Hz 陷波（提高Q值）
        w0 = NOTCH_FREQ / nyquist
        self.b_notch, self.a_notch = iirnotch(w0, QUALITY_FACTOR)
        self.zi_notch = [lfilter_zi(self.b_notch, self.a_notch) for _ in range(8)]

        # 2) 方波优化带通（更宽通带）
        self.fir_coeff = firwin(FIR_ORDER, [LOW_CUTOFF / nyquist, HIGH_CUTOFF / nyquist], pass_zero=False)
        self.zi_fir = [np.zeros(FIR_ORDER - 1) for _ in range(8)]

        # 3) 中值滤波缓冲（针对方波优化）
        self.median_buffers = [deque(maxlen=MEDIAN_WINDOW_SIZE) for _ in range(8)]

    @staticmethod
    def _read_lines_from_socket(sock, buffer_b: bytearray):
        """
        从 socket 收字节并切出按 \\n 分隔的完整行（与 makefile.readline 等效但更稳，避免半包/编码问题）。
        返回 (lines_str_list, 更新后的 buffer_b)。
        """
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            return [], buffer_b
        except (OSError, ConnectionError, BrokenPipeError):
            return None, buffer_b
        if not chunk:
            return None, buffer_b
        buffer_b += chunk
        out = []
        while True:
            i = buffer_b.find(b"\n")
            if i < 0:
                break
            line_b = buffer_b[:i]
            del buffer_b[: i + 1]
            try:
                out.append(line_b.decode("utf-8", errors="replace").strip())
            except Exception:
                out.append("")
        return out, buffer_b

    # -------- 主循环：读 TCP 行，解析，入缓冲（行格式同串口 channel:...）--------
    def run(self):
        self.running = True
        sock = None
        line_buf = bytearray()
        try:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=30.0)
            except OSError as e:
                self.connection_status.emit(
                    f"TCP 连接失败 {self.host}:{self.port} — {e!s}（检查 IP/端口/固件与防火墙）"
                )
                return

            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
            except OSError:
                pass
            # 有数据立即返回，无数据则短超时，便于轮询 self.running
            sock.settimeout(0.2)
            self._tcp_sock = sock
            self.connection_status.emit(
                f"已连接 {self.host}:{self.port}，正在等待 channel: 数据行…"
            )
            try:
                sock.sendall(b"1")
            except Exception:
                pass

            while self.running:
                lines, line_buf = self._read_lines_from_socket(sock, line_buf)
                if lines is None:
                    self.connection_status.emit("TCP 连接已关闭")
                    break
                if not lines:
                    continue
                for raw in lines:
                    if not raw:
                        continue
                    # 1. 原始数据送入队列
                    try:
                        values = self._parse_line(raw)
                        if values and len(values) == 8:
                            values_uv = serial_amplitude_to_uv(values)
                            trig = (
                                self._trigger_state.get()
                                if self._trigger_state is not None
                                else 0
                            )
                            row = values_uv + [float(trig)]
                            self.log_queue.put_nowait((time.time(), row))
                    except queue.Full:
                        pass

                    # 2. 状态栏：限流，避免 ~500 次/秒 刷新拖垮主线程
                    st = time.time()
                    if st - self._last_status_emit_t >= 1.0 / TCP_STATUS_MAX_HZ:
                        self._last_status_emit_t = st
                        self.data_received.emit(raw)
                    # 3. 滤波与缓冲（全速率，保证曲线跟得上采样）
                    self._process_serial_data(raw)
        except Exception as e:
            # 任一线程内未预料错误，避免静默退出导致界面无数据
            self.connection_status.emit(f"采集线程异常: {e!s}")
        finally:
            self._tcp_sock = None
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    # -------- 处理串口数据（一次发8通道）--------
    def _process_serial_data(self, text: str):
        """处理串口数据：一次性8通道（针对方波信号优化）"""
        values = self._parse_line(text)
        if values is None or len(values) != 8:
            return
        values = serial_amplitude_to_uv(values)

        # 遍历8个通道，逐一滤波并写入buffer（单位为 µV）
        for ch in range(8):
            val = values[ch]

            if self.filter_enabled:
                # ===== 方波信号专用优化滤波 =====
                # 1) 中值滤波优先（去除脉冲噪声，保留方波边沿）
                if self.median_enabled:
                    # 添加当前值到中值缓冲区
                    self.median_buffers[ch].append(val)

                    # 当缓冲区满时计算中值（窗口大小建议5-7点）
                    if len(self.median_buffers[ch]) >= self.median_buffers[ch].maxlen:
                        # 使用中值滤波去除脉冲噪声
                        median_val = float(np.median(self.median_buffers[ch]))
                        val = median_val  # 直接使用中值（去除了脉冲）

                # 2) 带通滤波（针对方波优化）
                # 注意：这里保留IIR滤波用于实时处理，但调整了参数
                y1, self.zi_notch[ch] = lfilter(
                    self.b_notch, self.a_notch, [val], zi=self.zi_notch[ch]
                )

                # 方波信号带通优化：更宽的通带，保留更多谐波成分
                y2, self.zi_fir[ch] = lfilter(
                    self.fir_coeff, [1.0], y1, zi=self.zi_fir[ch]
                )
                val = float(y2[0])

            # 写入对应通道的缓冲区（一次1个点）
            self.buffer.append(ch, [val], self.global_index)

        # 每行数据只增加一次全局时间索引
        self.global_index += 1

    # -------- 行解析：从任意前缀中抽取浮点数 --------
    def _parse_line(self, text: str):
        """
        兼容形如 "channel: v1,...,v8" 等含 8 个浮点的行。
        固件当前输出为电压 (V)；是否在后续转为 µV 由 serial_amplitude_to_uv 处理。
        """
        if ":" in text or "：" in text:
            part = re.split(r'[:：]', text, maxsplit=1)[1]
        else:
            part = text
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', part)
        if len(nums) < 8:
            return None
        # 转换为浮点数
        try:
            return list(map(float, nums))
        except ValueError:
            return None

    # -------- 提供绘图数据 --------
    def get_plot_data(self, length=PLOT_LENGTH):
        """
        获取最近 length 个点的数据
        修复20000点满后波形消失问题：
        1. 使用ChannelBuffer安全x轴
        2. np.interp插值在已有数据范围内
        """
        return self.buffer.get_recent(length)

    def get_fft_data(self, length=FFT_LENGTH):
        """
        获取FFT数据，直接用安全的get_plot_data
        """
        return self.get_plot_data(length)

    # -------- 外部控制 --------
    def set_filter_enabled(self, on: bool):
        self.filter_enabled = on

    def set_median_enabled(self, on: bool):
        self.median_enabled = on

    def send_mode(self, key: str):
        """经 TCP 向固件发送模式：'1'连续、'2'阻抗、'3'自检（与串口发单字节相同）。"""
        try:
            s = self._tcp_sock
            if s is not None:
                s.sendall(key.encode("ascii"))
        except Exception:
            pass

    def stop(self):
        self.running = False
        s = self._tcp_sock
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        self.wait(2000)


# ================== 主窗口（与 FIFO_plot 相同 UI；数据来自 TcpThread） ==================
class TCPPlotter(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YuEEG Data Viewer (TCP / WiFi)")
        self.setWindowIcon(QtGui.QIcon('./school_logo.ico'))

        # 全局绘图配置（TCP 版关闭 OpenGL：部分显卡/驱动下 useOpenGL=True 会导致曲线空白）
        pg.setConfigOptions(antialias=True, useOpenGL=False)

        # ========= 1. 创建日志系统 =========
        self.log_queue = queue.Queue(maxsize=LOG_QUEUE_MAX)
        self.log_thread = LogWriterThread(LOG_DIR, self.log_queue)
        self.log_thread.log_rotated.connect(self._on_log_rotated)
        self.log_thread.start()

        # ========= 软件打标（UDP，与 UI_cue/cue_paradigm 同步） =========
        self.trigger_state = TriggerState()
        self.trigger_listener = TriggerUdpListener(self.trigger_state, UDP_PORT)

        self._setup_ui()

        self.trigger_listener.bind_failed.connect(
            lambda m: self.statusBar().showMessage(m, 12000)
        )
        self.trigger_listener.bind_ok.connect(
            lambda p: self.statusBar().showMessage(
                f"UDP 打标就绪 127.0.0.1:{p}（请运行 cue_paradigm 并开始范式）", 6000
            )
        )
        self.trigger_listener.start()

        host, port = self._get_tcp_target()
        if not host or port is None:
            self.log_thread.stop()
            sys.exit(1)

        self.serial_thread = TcpThread(host, port, self.log_queue, self.trigger_state)
        self.serial_thread.data_received.connect(self._on_serial_line)
        self.serial_thread.connection_status.connect(
            lambda m: self.statusBar().showMessage(m, 15000)
        )
        self.serial_thread.start()

        self._start_timers()

        if self.log_thread._current_path:
            self._on_log_rotated(self.log_thread._current_path)

        # 默认启用自动Y轴，更适合稀疏信号
        self.auto_y_box.setChecked(True)
        self._on_auto_y_changed()

    # -------- UI 构建 --------
    def _setup_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        main = QtWidgets.QVBoxLayout(cw)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        # 绘图区
        self.time_plot = pg.PlotWidget()
        self.fft_plot = pg.PlotWidget()
        for p in (self.time_plot, self.fft_plot):
            p.showGrid(x=True, y=True, alpha=0.3)

        self.time_plot.setLabel('bottom', 'Samples')
        self.time_plot.setLabel('left', 'Amplitude (μV)')
        self.fft_plot.setLabel('bottom', 'Frequency (Hz)')
        self.fft_plot.setLabel('left', 'Amplitude (a.u.)')

        main.addWidget(self.time_plot)
        main.addWidget(self.fft_plot)

        # 曲线（8条）
        palette = ['#FF3B30', '#34C759', '#007AFF', '#5AC8FA',
                   '#AF52DE', '#FFCC00', '#FF9500', '#8E8E93']
        self.time_curves = [self.time_plot.plot(pen=pg.mkPen(color=c, width=1.5)) for c in palette]
        self.fft_curves = [self.fft_plot.plot(pen=pg.mkPen(color=c, width=1.2)) for c in palette]

        # 控制栏
        ctrl = QtWidgets.QHBoxLayout()
        main.addLayout(ctrl)
        # 我想在这里加一条分割线，作为ctrl和串口数据的分割线，但写的好像不对
        # 创建一条水平分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)  # 水平线
        line.setFrameShadow(QFrame.Sunken)  # 阴影效果（看起来是“凹下去”的）
        line.setStyleSheet("color: #cccccc;")  # 设置颜色（可选）

        main.addWidget(line)

        # 通道勾选
        self.ch_boxes = [CheckBox(f"CH{i + 1}") for i in range(8)]
        for cb in self.ch_boxes:
            cb.setChecked(True)
            ctrl.addWidget(cb)

        # 显示模式
        self.mode_combo = ComboBox()
        self.mode_combo.addItems(["波形", "频谱", "并列"])
        self.mode_combo.currentIndexChanged.connect(self._update_display_mode)
        ctrl.addWidget(self.mode_combo)

        # 滤波开关
        self.filter_box = CheckBox("滤波开启")
        self.filter_box.setChecked(True)
        self.filter_box.stateChanged.connect(
            lambda s: self.serial_thread.set_filter_enabled(self.filter_box.isChecked()))
        ctrl.addWidget(self.filter_box)

        # 中值滤波开关
        self.median_box = CheckBox("中值滤波(去基线)")
        self.median_box.setChecked(False)
        self.median_box.stateChanged.connect(
            lambda s: self.serial_thread.set_median_enabled(self.median_box.isChecked()))
        ctrl.addWidget(self.median_box)

        # Y轴控制
        self.auto_y_box = CheckBox("自动Y轴")
        self.auto_y_box.setChecked(False)  # 注意：初始化后会在__init__中设为True
        self.auto_y_box.stateChanged.connect(self._on_auto_y_changed)
        ctrl.addWidget(self.auto_y_box)

        self.y_spin = SpinBox()
        self.y_spin.setRange(10, 1000)
        self.y_spin.setValue(int(DEFAULT_Y_LIM_UV))
        self.y_spin.valueChanged.connect(self._apply_y_range)
        ctrl.addWidget(self.y_spin)

        # 模式切换按钮（标准/测试）
        self.fun_btn = PushButton("标准")
        self.fun_btn.clicked.connect(self._toggle_fun_mode)
        ctrl.addWidget(self.fun_btn)

        # 暂停
        self.pause_btn = PushButton("暂停")
        self.pause_btn.clicked.connect(self._toggle_pause)
        ctrl.addWidget(self.pause_btn)

        self.lbl_trig = QtWidgets.QLabel("打标: 0")
        self.lbl_trig.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        self.lbl_trig.setToolTip(
            "第9通道 TRIG：0 无；1 左；2 右；10 休息；11 准备（见 trigger_protocol.py）"
        )
        ctrl.addWidget(self.lbl_trig)

        # 初始显示
        self._update_display_mode()
        self._apply_y_range()

    def _start_timers(self):
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._refresh_plots)
        self.plot_timer.start(UPDATE_INTERVAL)
        self.trig_timer = QtCore.QTimer(self)
        self.trig_timer.timeout.connect(self._refresh_trig_label)
        self.trig_timer.start(120)

    def _refresh_trig_label(self):
        c = self.trigger_state.get()
        self.lbl_trig.setText(f"打标: {c}")

    # -------- 事件处理 --------
    def _toggle_fun_mode(self):
        # 与固件保持一致：'1' 连续、'3' 自检
        if self.fun_btn.text() == "标准":
            self.serial_thread.send_mode('3')  # 切到自检
            self.serial_thread.set_filter_enabled(False)
            self.fun_btn.setText("测试")
            self._info("已切换到自检信号(测试)")
        else:
            self.serial_thread.send_mode('1')  # 回到连续采样
            self.serial_thread.set_filter_enabled(True)
            self.fun_btn.setText("标准")
            self._info("已切换到标准采样")

    def _toggle_pause(self):
        if self.plot_timer.isActive():
            self.plot_timer.stop()
            self.pause_btn.setText("继续")
        else:
            self.plot_timer.start()
            self.pause_btn.setText("暂停")

    def _update_display_mode(self):
        mode = self.mode_combo.currentText()
        if mode == "波形":
            self.time_plot.show()
            self.fft_plot.hide()
        elif mode == "频谱":
            self.time_plot.hide()
            self.fft_plot.show()
        else:
            self.time_plot.show()
            self.fft_plot.show()

    def _on_auto_y_changed(self):
        if self.auto_y_box.isChecked():
            self.time_plot.enableAutoRange(axis='y', enable=True)
        else:
            self.time_plot.enableAutoRange(axis='y', enable=False)
            self._apply_y_range()

    def _apply_y_range(self):
        if not self.auto_y_box.isChecked():
            lim = float(self.y_spin.value())
            self.time_plot.setYRange(-lim, +lim)

    def _refresh_plots(self):
        mode = self.mode_combo.currentText()
        if mode in ("波形", "并列"):
            self._update_time_plot()
        if mode in ("频谱", "并列"):
            self._update_fft_plot()

    def _update_time_plot(self):
        x, data = self.serial_thread.get_plot_data(PLOT_LENGTH)
        if data.size == 0:
            return

        active = [i for i, cb in enumerate(self.ch_boxes) if cb.isChecked()]
        if not active:
            return

        for i in active:
            self.time_curves[i].setData(x, data[i])

        # 自动滚动X轴
        if len(x) > 0:
            self.time_plot.setXRange(x[0], x[-1])

        # 自动Y轴范围（若开启）
        if self.auto_y_box.isChecked():
            # 只考虑有数据的区域
            mask = ~np.isnan(data)
            if np.any(mask):
                valid_data = data[mask]
                y_min, y_max = np.min(valid_data), np.max(valid_data)
                if y_min != y_max:
                    margin = (y_max - y_min) * 0.1 or 1.0
                    self.time_plot.setYRange(y_min - margin, y_max + margin)
                else:
                    # 单一值情况
                    self.time_plot.setYRange(y_min - 1, y_max + 1)

    def _update_fft_plot(self):
        x, data = self.serial_thread.get_fft_data(FFT_LENGTH)
        if data.size == 0:
            return

        fs = SAMPLE_RATE
        win = get_window('hann', data.shape[1])
        freq = np.fft.rfftfreq(len(win), 1.0 / fs)

        max_amp = 0.0
        for i in range(8):
            if not self.ch_boxes[i].isChecked():
                self.fft_curves[i].setData([], [])
                continue

            sig = data[i].astype(float)
            # 跳过全NaN的情况
            if np.all(np.isnan(sig)) or len(sig) < 8:
                self.fft_curves[i].setData([], [])
                continue

            # 处理NaN值
            sig = np.nan_to_num(sig)
            sig = sig - np.mean(sig)
            sig = sig * win

            amp = np.abs(np.fft.rfft(sig))
            # 仅展示 3~50 Hz
            mask = (freq >= 3.0) & (freq <= 50.0)
            self.fft_curves[i].setData(freq[mask], amp[mask])
            if amp[mask].size > 0:
                max_amp = max(max_amp, float(np.max(amp[mask])))

        if max_amp > 0:
            self.fft_plot.setYRange(0, max_amp * 1.1)

    def _on_serial_line(self, text: str):
        # 状态栏里看最后200字符
        try:
            self.statusBar().showMessage(text[-200:], 1500)
        except Exception:
            pass

    def _on_log_rotated(self, path: str):
        """录制文件切片通知，更新状态栏"""
        if not path:
            return
        filename = os.path.basename(path)
        self.statusBar().showMessage(f"录制: {filename}", 3000)

    def _get_tcp_target(self):
        """
        询问板子 IP 与 TCP 端口（与 main_wifi 串口监视器、TCP_SERVER_PORT 一致）。
        返回 (host_str, port_int)；取消则 (None, None)。
        """
        host, ok = QtWidgets.QInputDialog.getText(
            self,
            "TCP 连接",
            "ESP32 IP（与「已连接 WiFi」中一致）:",
            text="192.168.137.122",
        )
        if not ok or not str(host).strip():
            return None, None
        port, ok2 = QtWidgets.QInputDialog.getInt(
            self,
            "TCP 连接",
            "端口（与固件 main_wifi TCP_SERVER_PORT 一致，默认 3333）:",
            3333,
            1,
            65535,
            1,
        )
        if not ok2:
            return None, None
        return str(host).strip(), int(port)

    def _info(self, msg: str):
        try:
            InfoBar.success(
                title='提示',
                content=msg,
                orient=QtCore.Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=1500,
                parent=self
            )
        except Exception:
            # 若 qfluentwidgets 不可用则忽略
            pass

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        try:
            self.trigger_listener.stop()
            self.trigger_listener.wait(3000)
        except Exception:
            pass
        # 先停止采集线程（TCP）
        try:
            self.serial_thread.stop()
        except Exception:
            pass

        # 再停止日志线程（确保数据完整写入）
        try:
            self.log_thread.stop()
        except Exception:
            pass

        event.accept()


# ================== 程序入口 ==================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = TCPPlotter()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())