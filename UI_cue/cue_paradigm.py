# -*- coding: utf-8 -*-
"""
枕区左右运动想象（MI）Cue 范式：准备 → 想象 → 休息 循环。
配合 Plot/FIFO_plot：UDP 软件打标（第 9 通道 TRIG），码表见项目根 trigger_protocol.py。
"""
from __future__ import annotations

import csv
import json
import os
import random
import socket
import sys
import time
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from trigger_protocol import (
    UDP_HOST,
    UDP_PORT,
    TRIG_OFF,
    TRIG_MI_LEFT,
    TRIG_MI_RIGHT,
    TRIG_PREP,
    TRIG_REST,
)


class Phase(Enum):
    IDLE = auto()
    REST = auto()  # 试次间休息 / 首段基线
    PREP = auto()  # 准备（注视）
    MI = auto()  # 运动想象 + 方向箭头


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class ArrowWidget(QtWidgets.QWidget):
    """中央提示区：休息/准备文案，或左/右箭头。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "blank"  # blank | rest | prep | left | right
        self.setMinimumSize(760, 420)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_mode(self, mode: str):
        self._mode = mode
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor("#1a1a1e"))

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        sc = max(1.0, min(w, h) / 520.0)
        font_big = QtGui.QFont("Segoe UI", int(34 * sc))
        font_hint = QtGui.QFont("Segoe UI", int(20 * sc))

        if self._mode == "rest":
            p.setPen(QtGui.QColor("#a0a0a8"))
            p.setFont(font_big)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "休息")
            p.setFont(font_hint)
            p.drawText(
                QtCore.QRect(0, cy + 40, w, 60),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                "放松，勿刻意想象",
            )
        elif self._mode == "prep":
            p.setPen(QtGui.QColor("#e8e8f0"))
            p.setFont(font_big)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "+")
            p.setFont(font_hint)
            p.drawText(
                QtCore.QRect(0, cy + 50, w, 60),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                "准备",
            )
        elif self._mode == "left":
            self._draw_arrow(p, cx, cy, left=True)
        elif self._mode == "right":
            self._draw_arrow(p, cx, cy, left=False)
        else:
            p.setPen(QtGui.QColor("#606068"))
            p.setFont(font_hint)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "点击下方「开始范式」")

    def _draw_arrow(self, p: QtGui.QPainter, cx: int, cy: int, left: bool):
        p.setPen(QtGui.QColor("#4ec9b0"))
        p.setBrush(QtGui.QColor("#4ec9b0"))
        size = min(self.width(), self.height()) // 3
        body_w, body_h = size * 1.2, size * 0.35
        head = size * 0.55
        if left:
            tip_x, tip_y = cx - body_w * 0.5 - head, cy
            poly = QtGui.QPolygonF(
                [
                    QtCore.QPointF(tip_x, tip_y),
                    QtCore.QPointF(tip_x + head, tip_y - head * 0.55),
                    QtCore.QPointF(tip_x + head, tip_y - body_h * 0.5),
                    QtCore.QPointF(tip_x + head + body_w, tip_y - body_h * 0.5),
                    QtCore.QPointF(tip_x + head + body_w, tip_y + body_h * 0.5),
                    QtCore.QPointF(tip_x + head, tip_y + body_h * 0.5),
                    QtCore.QPointF(tip_x + head, tip_y + head * 0.55),
                ]
            )
        else:
            tip_x, tip_y = cx + body_w * 0.5 + head, cy
            poly = QtGui.QPolygonF(
                [
                    QtCore.QPointF(tip_x, tip_y),
                    QtCore.QPointF(tip_x - head, tip_y - head * 0.55),
                    QtCore.QPointF(tip_x - head, tip_y - body_h * 0.5),
                    QtCore.QPointF(tip_x - head - body_w, tip_y - body_h * 0.5),
                    QtCore.QPointF(tip_x - head - body_w, tip_y + body_h * 0.5),
                    QtCore.QPointF(tip_x - head, tip_y + body_h * 0.5),
                    QtCore.QPointF(tip_x - head, tip_y + head * 0.55),
                ]
            )
        p.drawPolygon(poly)

        p.setPen(QtGui.QColor("#c8c8d0"))
        sc = max(1.0, min(self.width(), self.height()) / 520.0)
        f = QtGui.QFont("Segoe UI", int(20 * sc))
        p.setFont(f)
        lab = "想象向左运动" if left else "想象向右运动"
        lh = int(52 * sc)
        p.drawText(
            QtCore.QRect(0, cy + int(size * 0.85), self.width(), lh),
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
            lab,
        )


class CueParadigmWindow(QtWidgets.QMainWindow):
    TICK_MS = 50

    def __init__(self):
        super().__init__()
        self.setWindowTitle("枕区 MI Cue 范式（左 / 右）")
        self.resize(1040, 780)
        self.setMinimumSize(880, 640)

        self._phase = Phase.IDLE
        self._trial_idx = 0
        self._total_trials = 20
        self._directions: List[str] = []  # 'left' | 'right'
        self._phase_deadline_m: float = 0.0  # monotonic end time
        self._session_start_m: float = 0.0
        self._marker_file: Optional[object] = None
        self._marker_writer: Optional[csv.writer] = None
        self._paused_remain: float = 0.0
        self._udp_sock: Optional[socket.socket] = None

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # --- 参数区：两栏并排，减少竖向占用，把空间留给中央范式 ---
        form = QtWidgets.QGroupBox("范式参数（典型：准备 1.5s，想象 4s，休息 3s）")
        gl = QtWidgets.QGridLayout(form)
        self.sp_trials = QtWidgets.QSpinBox()
        self.sp_trials.setRange(4, 200)
        self.sp_trials.setValue(20)
        self.sp_prep = QtWidgets.QDoubleSpinBox()
        self.sp_prep.setRange(0.5, 5.0)
        self.sp_prep.setSingleStep(0.1)
        self.sp_prep.setDecimals(1)
        self.sp_prep.setValue(1.5)
        self.sp_mi = QtWidgets.QDoubleSpinBox()
        self.sp_mi.setRange(2.0, 15.0)
        self.sp_mi.setSingleStep(0.5)
        self.sp_mi.setDecimals(1)
        self.sp_mi.setValue(4.0)
        self.sp_rest = QtWidgets.QDoubleSpinBox()
        self.sp_rest.setRange(1.0, 20.0)
        self.sp_rest.setSingleStep(0.5)
        self.sp_rest.setDecimals(1)
        self.sp_rest.setValue(3.0)
        self.cb_balance = QtWidgets.QCheckBox("左右试次数量相等（随机交错）")
        self.cb_balance.setChecked(False)

        r = 0
        gl.addWidget(QtWidgets.QLabel("总试次数"), r, 0)
        gl.addWidget(self.sp_trials, r, 1)
        r += 1
        gl.addWidget(QtWidgets.QLabel("准备期 (s)"), r, 0)
        gl.addWidget(self.sp_prep, r, 1)
        r += 1
        gl.addWidget(QtWidgets.QLabel("运动想象期 (s)"), r, 0)
        gl.addWidget(self.sp_mi, r, 1)
        r += 1
        gl.addWidget(QtWidgets.QLabel("试次间休息 (s)"), r, 0)
        gl.addWidget(self.sp_rest, r, 1)
        r += 1
        gl.addWidget(self.cb_balance, r, 0, 1, 2)
        form.setMaximumHeight(240)

        udp_form = QtWidgets.QGroupBox("FIFO_plot 软件打标（UDP，与第 9 通道 TRIG 同步）")
        ug = QtWidgets.QGridLayout(udp_form)
        self.cb_udp = QtWidgets.QCheckBox("向采集端发送打标码")
        self.cb_udp.setChecked(True)
        self.cb_udp.setToolTip("需先运行 FIFO_plot.py；默认 127.0.0.1:45454")
        self.ed_udp_host = QtWidgets.QLineEdit(UDP_HOST)
        self.sp_udp_port = QtWidgets.QSpinBox()
        self.sp_udp_port.setRange(1, 65535)
        self.sp_udp_port.setValue(UDP_PORT)
        ur = 0
        ug.addWidget(self.cb_udp, ur, 0, 1, 2)
        ur += 1
        ug.addWidget(QtWidgets.QLabel("主机"), ur, 0)
        ug.addWidget(self.ed_udp_host, ur, 1)
        ur += 1
        ug.addWidget(QtWidgets.QLabel("端口"), ur, 0)
        ug.addWidget(self.sp_udp_port, ur, 1)
        udp_form.setMaximumHeight(240)

        settings_row = QtWidgets.QHBoxLayout()
        settings_row.addWidget(form, stretch=1)
        settings_row.addWidget(udp_form, stretch=1)
        root.addLayout(settings_row, stretch=0)

        self.arrow = ArrowWidget()
        root.addWidget(self.arrow, stretch=10)

        # 本阶段进度（当前 Phase 内剩余时间可视化）
        self.lbl_phase = QtWidgets.QLabel("阶段：—")
        self.bar_phase = QtWidgets.QProgressBar()
        self.bar_phase.setRange(0, 1000)
        self.bar_phase.setValue(0)
        self.bar_phase.setFormat("%p%")
        self.bar_phase.setTextVisible(True)

        # 会话进度（已完成试次数 / 总试次数）
        self.lbl_session = QtWidgets.QLabel("会话：未开始")
        self.bar_session = QtWidgets.QProgressBar()
        self.bar_session.setRange(0, 1000)
        self.bar_session.setValue(0)
        self.bar_session.setFormat("%p%")

        root.addWidget(self.lbl_phase)
        root.addWidget(self.bar_phase)
        root.addWidget(self.lbl_session)
        root.addWidget(self.bar_session)

        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("开始范式")
        self.btn_pause = QtWidgets.QPushButton("暂停")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause)
        row.addWidget(self.btn_stop)
        root.addLayout(row)

        self.statusBar().showMessage(
            "就绪。CSV 与 FIF 均在 Plot/logs；打标码见 trigger_protocol.py"
        )

    def _send_trigger(self, code: int) -> None:
        if not self.cb_udp.isChecked():
            return
        try:
            payload = json.dumps({"code": int(code)}).encode("utf-8")
            host = (self.ed_udp_host.text() or "").strip() or UDP_HOST
            port = int(self.sp_udp_port.value())
            if self._udp_sock is None:
                self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.sendto(payload, (host, port))
        except Exception:
            pass

    def _set_udp_controls_enabled(self, on: bool):
        self.cb_udp.setEnabled(on)
        self.ed_udp_host.setEnabled(on)
        self.sp_udp_port.setEnabled(on)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #252526; color: #d4d4d4; }
            QGroupBox { font-weight: bold; border: 1px solid #3c3c3c; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QPushButton { background: #0e639c; color: white; border: none; padding: 8px 16px; border-radius: 4px; min-width: 88px; }
            QPushButton:hover { background: #1177bb; }
            QPushButton:disabled { background: #3c3c3c; color: #808080; }
            QProgressBar { border: 1px solid #3c3c3c; border-radius: 4px; text-align: center; height: 22px; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #264f78, stop:1 #4ec9b0); }
            QSpinBox, QDoubleSpinBox { padding: 4px; background: #3c3c3c; border: 1px solid #555; border-radius: 3px; }
            QLabel { color: #c8c8d0; }
            """
        )

    # --- 时序 ---
    def _prep_sec(self) -> float:
        return float(self.sp_prep.value())

    def _mi_sec(self) -> float:
        return float(self.sp_mi.value())

    def _rest_sec(self) -> float:
        return float(self.sp_rest.value())

    def _build_directions(self, n: int, balanced: bool) -> List[str]:
        if balanced:
            if n < 2:
                n = 2
                self.sp_trials.setValue(n)
            elif n % 2 != 0:
                n = n - 1
                self.sp_trials.setValue(n)
                self.statusBar().showMessage(
                    "左右平衡需偶数试次，已自动改为 {} 次。".format(n), 6000
                )
        if balanced:
            half = n // 2
            dirs = ["left"] * half + ["right"] * half
            random.shuffle(dirs)
        else:
            dirs = [random.choice(["left", "right"]) for _ in range(n)]
        return dirs

    def _open_markers(self):
        # 与 FIFO_plot 录制的 FIF 同目录：项目 Plot/logs
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _plot_logs = os.path.join(_root, "Plot", "logs")
        os.makedirs(_plot_logs, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_plot_logs, f"cue_markers_{ts}.csv")
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp_iso",
                "event",
                "trial_index",
                "direction",
                "phase_elapsed_s",
                "note",
            ]
        )
        self._marker_file = f
        self._marker_writer = w
        self.statusBar().showMessage(f"事件标记：{path}", 5000)

    def _write_marker(self, event: str, direction: str = "", note: str = ""):
        if self._marker_writer is None:
            return
        self._marker_writer.writerow(
            [_now_iso(), event, self._trial_idx, direction, "", note]
        )
        self._marker_file.flush()

    def _close_markers(self):
        if self._marker_file:
            try:
                self._marker_file.close()
            except Exception:
                pass
        self._marker_file = None
        self._marker_writer = None

    def _start(self):
        if self._timer.isActive():
            return
        self._directions = self._build_directions(
            int(self.sp_trials.value()), self.cb_balance.isChecked()
        )
        self._total_trials = len(self._directions)
        self._trial_idx = 0
        self._open_markers()
        self._write_marker("session_start", "", f"trials={self._total_trials}")
        self._session_start_m = time.monotonic()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.sp_trials.setEnabled(False)
        self.sp_prep.setEnabled(False)
        self.sp_mi.setEnabled(False)
        self.sp_rest.setEnabled(False)
        self.cb_balance.setEnabled(False)
        self._set_udp_controls_enabled(False)

        self._enter_phase(Phase.REST, first=True)
        self._timer.start(self.TICK_MS)

    def _enter_phase(self, phase: Phase, first: bool = False):
        self._phase = phase
        now = time.monotonic()

        if phase == Phase.REST:
            d = self._rest_sec()
            self._phase_deadline_m = now + d
            self.arrow.set_mode("rest")
            self.lbl_phase.setText(f"阶段：休息（试次间 {d:.1f}s）")
            ev = "rest_start" if not first else "baseline_rest_start"
            self._write_marker(ev, "", f"duration_s={d}")
            self._send_trigger(TRIG_REST)
        elif phase == Phase.PREP:
            d = self._prep_sec()
            self._phase_deadline_m = now + d
            self.arrow.set_mode("prep")
            self.lbl_phase.setText(f"阶段：准备（{d:.1f}s）")
            self._write_marker("prep_start", "", f"trial={self._trial_idx}")
            self._send_trigger(TRIG_PREP)
        elif phase == Phase.MI:
            d = self._mi_sec()
            self._phase_deadline_m = now + d
            direction = self._directions[self._trial_idx]
            self.arrow.set_mode("left" if direction == "left" else "right")
            self.lbl_phase.setText(
                f"阶段：运动想象（{d:.1f}s）— {'左' if direction == 'left' else '右'}"
            )
            self._write_marker(
                "mi_cue_on",
                direction,
                f"trial={self._trial_idx} duration_s={d}",
            )
            self._send_trigger(TRIG_MI_LEFT if direction == "left" else TRIG_MI_RIGHT)
        self._update_progress_bars()
        self._update_phase_label_remain()

    def _update_progress_bars(self):
        # 本会话：已完成 MI 试次数 / 总试次数
        done_mi = min(self._trial_idx, self._total_trials)
        self.bar_session.setMaximum(max(1, self._total_trials * 1000))
        self.bar_session.setValue(done_mi * 1000)
        self.lbl_session.setText(f"会话：已完成 {done_mi} / {self._total_trials} 个想象试次")

        # 本阶段：剩余时间 -> 进度条「已过去比例」
        if self._phase == Phase.IDLE:
            self.bar_phase.setValue(0)
            return
        total = {
            Phase.REST: self._rest_sec(),
            Phase.PREP: self._prep_sec(),
            Phase.MI: self._mi_sec(),
        }.get(self._phase, 1.0)
        remain = max(0.0, self._phase_deadline_m - time.monotonic())
        elapsed = max(0.0, total - remain)
        frac = min(1.0, elapsed / total) if total > 0 else 1.0
        self.bar_phase.setMaximum(1000)
        self.bar_phase.setValue(int(frac * 1000))

    def _on_tick(self):
        if self._phase == Phase.IDLE:
            return
        if time.monotonic() >= self._phase_deadline_m:
            self._advance_state_machine()
        self._update_phase_label_remain()
        self._update_progress_bars()

    def _update_phase_label_remain(self):
        if self._phase in (Phase.IDLE,):
            return
        remain = max(0.0, self._phase_deadline_m - time.monotonic())
        total = {
            Phase.REST: self._rest_sec(),
            Phase.PREP: self._prep_sec(),
            Phase.MI: self._mi_sec(),
        }[self._phase]
        pname = {Phase.REST: "休息", Phase.PREP: "准备", Phase.MI: "运动想象"}.get(
            self._phase, ""
        )
        if self._phase == Phase.MI and self._trial_idx < len(self._directions):
            d = self._directions[self._trial_idx]
            lr = "左" if d == "left" else "右"
            self.lbl_phase.setText(
                f"阶段：{pname}（{total:.1f}s）— {lr} — 剩余 {remain:.1f}s"
            )
        else:
            self.lbl_phase.setText(f"阶段：{pname} — 剩余 {remain:.1f}s")

    def _advance_state_machine(self):
        if self._phase == Phase.REST:
            self._write_marker("rest_end", "", f"trial={self._trial_idx}")
            self._enter_phase(Phase.PREP)
        elif self._phase == Phase.PREP:
            self._write_marker("prep_end", "", f"trial={self._trial_idx}")
            self._enter_phase(Phase.MI)
        elif self._phase == Phase.MI:
            d = self._directions[self._trial_idx]
            self._write_marker("mi_cue_off", d, f"trial={self._trial_idx}")
            self._trial_idx += 1
            if self._trial_idx >= self._total_trials:
                self._finish_session()
                return
            self._enter_phase(Phase.REST)

    def _finish_session(self):
        self._timer.stop()
        self._send_trigger(TRIG_OFF)
        self._write_marker("session_end", "", f"trials_completed={self._total_trials}")
        self._close_markers()
        self.arrow.set_mode("blank")
        self.lbl_phase.setText("阶段：结束")
        self.lbl_session.setText(
            f"会话：已完成 {self._total_trials} / {self._total_trials} 个想象试次"
        )
        self.bar_session.setValue(self.bar_session.maximum())
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.btn_stop.setEnabled(False)
        self.sp_trials.setEnabled(True)
        self.sp_prep.setEnabled(True)
        self.sp_mi.setEnabled(True)
        self.sp_rest.setEnabled(True)
        self.cb_balance.setEnabled(True)
        self._set_udp_controls_enabled(True)
        self._phase = Phase.IDLE
        self.statusBar().showMessage("范式结束。", 8000)

    def _toggle_pause(self):
        if self._phase == Phase.IDLE:
            return
        if self.btn_pause.text() == "暂停":
            self._paused_remain = max(0.0, self._phase_deadline_m - time.monotonic())
            self._timer.stop()
            self.btn_pause.setText("继续")
            self._write_marker("pause", "", f"remain_s={self._paused_remain:.3f}")
        else:
            self._phase_deadline_m = time.monotonic() + self._paused_remain
            self._timer.start(self.TICK_MS)
            self.btn_pause.setText("暂停")
            self._write_marker("resume", "", "")

    def _stop(self):
        self._timer.stop()
        self._send_trigger(TRIG_OFF)
        self._write_marker("session_abort", "", f"at_trial={self._trial_idx}")
        self._close_markers()
        self.arrow.set_mode("blank")
        self.lbl_phase.setText("阶段：已停止")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.btn_stop.setEnabled(False)
        self.sp_trials.setEnabled(True)
        self.sp_prep.setEnabled(True)
        self.sp_mi.setEnabled(True)
        self.sp_rest.setEnabled(True)
        self.cb_balance.setEnabled(True)
        self._set_udp_controls_enabled(True)
        self._phase = Phase.IDLE
        self.bar_phase.setValue(0)
        self.statusBar().showMessage("已停止。", 5000)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self._timer.isActive():
            self._timer.stop()
        self._close_markers()
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = CueParadigmWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
