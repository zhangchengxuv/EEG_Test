# -*- coding: utf-8 -*-
"""
软件打标协议：cue_paradigm（UDP 发送）与 FIFO_plot（UDP 接收）共用。
请勿只改一侧的数字含义。

第 9 通道 TRIG 在 FIF 中存为「伏特」：value_V = code * TRIG_VOLT_SCALE。
读回：code = rint(data[8] / TRIG_VOLT_SCALE)。
"""
import threading

UDP_HOST = "127.0.0.1"
UDP_PORT = 45454


class TriggerState:
    """线程安全的当前打标码（与每个串口采样点合并）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._code = 0

    def set(self, c: int) -> None:
        with self._lock:
            self._code = int(c)

    def get(self) -> int:
        with self._lock:
            return self._code

# 离散事件码（与脑电 µV 量级区分，见 TRIG_VOLT_SCALE）
TRIG_OFF = 0
TRIG_MI_LEFT = 1
TRIG_MI_RIGHT = 2
TRIG_REST = 10
TRIG_PREP = 11

# FIF 第 9 通道：物理存储系数（V），解码时整除还原整数码
TRIG_VOLT_SCALE = 1e-9
