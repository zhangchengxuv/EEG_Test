# -*- coding: utf-8 -*-
r"""Run from EEG_Test: .venv\Scripts\python.exe SSVEP\run_demo.py"""
import argparse
from collections import deque
from dataclasses import asdict
import json
import os
import queue
from pathlib import Path
import sys
import time

# Small CCA matrices do not benefit from oversubscribing CPU cores during presentation.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import serial.tools.list_ports

try:
    from .core import (CHANNEL_NAMES, SAMPLE_RATE, FrameTiming, SampleBuffer, assess_window,
                       make_targets, luminance, RELAXED_GATE, MouseTargetLatch)
    from .acquisition import AcquisitionThread, DecodingThread, SessionLog
    from .audience_bridge import AudienceBridge
    from .result_serial import ResultSerialSender, same_serial_port
except ImportError:
    from core import (CHANNEL_NAMES, SAMPLE_RATE, FrameTiming, SampleBuffer, assess_window,
                      make_targets, luminance, RELAXED_GATE, MouseTargetLatch)
    from acquisition import AcquisitionThread, DecodingThread, SessionLog
    from audience_bridge import AudienceBridge
    from result_serial import ResultSerialSender, same_serial_port


class StimulusWidget(QtWidgets.QWidget):
    first_frame = QtCore.pyqtSignal(float)
    stop_requested = QtCore.pyqtSignal()
    fullscreen_requested = QtCore.pyqtSignal()
    mouse_target_changed = QtCore.pyqtSignal(int)

    def __init__(self, targets, frame_timing, parent=None):
        super().__init__(parent)
        self.targets, self.timing = targets, frame_timing
        self.setMinimumSize(QtCore.QSize(600, 360))
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.running = False
        self.frame = 0
        self.display_frame = 0
        self.frame_due = False
        self.pending_frame = 0
        self.confirmed = None
        self.recorder = None
        self.neutral_render_test = False  # Used only by the no-flash GUI integration check.
        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.frame_timer.setSingleShot(True)
        self.frame_timer.timeout.connect(self._request_frame)
        self.frame_epoch = None
        self.mouse_test_enabled = False
        self.setMouseTracking(True)

    def begin(self, recorder):
        self.frame = 0
        self.display_frame = 0
        self.confirmed = None
        self.recorder = recorder
        self.timing.reset()
        self.frame_epoch = None
        self.running = True
        self._request_frame()

    def end(self):
        self.running = False
        self.frame_due = False
        self.frame_timer.stop()
        self.recorder = None
        self.confirmed = None
        self.update()

    def _request_frame(self):
        self.frame_due = True
        self.update()

    def _painted(self):
        if not self.running:
            return
        # This is CPU paint completion, not a swap or a measured display scan.
        stamp = time.perf_counter()
        self.timing.record(stamp)
        if self.frame_epoch is None:
            self.frame_epoch = stamp
            self.first_frame.emit(stamp)
        if self.recorder:
            self.recorder.submit("frame", (self.pending_frame, stamp))
        self.frame = self.pending_frame + 1
        # Absolute deadlines prevent cumulative timer drift; never burst to catch up.
        deadline = self.frame_epoch + self.frame / self.timing.refresh_hz
        delay_ms = max(0, round((deadline - time.perf_counter()) * 1000))
        self.frame_timer.start(delay_ms)

    def paintEvent(self, event):
        advance = self.running and self.frame_due
        if advance and self.frame_epoch is not None:
            # Skip missed phase steps after a stall instead of displaying a burst
            # of old frames. The timing gate still rejects windows with the stall.
            scheduled_frame = round((time.perf_counter() - self.frame_epoch) * self.timing.refresh_hz)
            self.frame = max(self.frame, scheduled_frame)
        if advance:
            self.display_frame = self.pending_frame = self.frame
            self.frame_due = False
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(QtCore.QRect(0, 0, self.width(), self.height()), QtGui.QColor("#101923"))
        # Four fixed quadrants; names and frequencies stay outside the flicker.
        painter.setPen(QtGui.QPen(QtGui.QColor("#4a627b"), 2))
        painter.drawLine(QtCore.QPointF(self.width()/2, 0), QtCore.QPointF(self.width()/2, self.height()))
        painter.drawLine(QtCore.QPointF(0, self.height()/2), QtCore.QPointF(self.width(), self.height()/2))
        for target, (rect, label, frequency_label) in zip(self.targets, self.target_layout()):
            value = luminance(self.display_frame, target.frames_per_cycle) if self.running and not self.neutral_render_test else .4
            brightness = int(255*value)
            painter.setPen(QtGui.QPen(QtGui.QColor("#506174"), 1))
            painter.setBrush(QtGui.QColor(brightness, brightness, brightness))
            painter.drawRoundedRect(rect, 12, 12)
            painter.setPen(QtGui.QColor("#e4edf5"))
            name_font = QtGui.QFont("Microsoft YaHei", weight=QtGui.QFont.Bold)
            name_font.setPixelSize(30)
            painter.setFont(name_font)
            painter.drawText(label, QtCore.Qt.AlignCenter, target.character)
            frequency_font = QtGui.QFont("Microsoft YaHei")
            frequency_font.setPixelSize(14)
            painter.setFont(frequency_font)
            painter.setPen(QtGui.QColor("#9fb4c8"))
            painter.drawText(frequency_label, QtCore.Qt.AlignCenter, f"{target.frequency:.3f} Hz")
        painter.end()
        if advance:
            self._painted()

    def target_layout(self):
        outer_x = max(18, min(48, self.width()*.02))
        outer_y = max(18, min(40, self.height()*.03))
        gap_x = max(96, min(240, self.width()*.10))
        gap_y = max(80, min(180, self.height()*.14))
        cw, ch = self.width()/2, self.height()/2
        result = []
        for i in range(4):
            column, row = i%2, i//2
            x = outer_x if column == 0 else cw+gap_x/2
            right = cw-gap_x/2 if column == 0 else self.width()-outer_x
            y = outer_y if row == 0 else ch+gap_y/2
            bottom = ch-gap_y/2 if row == 0 else self.height()-outer_y
            rect = QtCore.QRectF(x, y, right-x, max(30, bottom-y-68))
            name = QtCore.QRectF(x, rect.bottom()+7, rect.width(), 37)
            frequency = QtCore.QRectF(x, name.bottom()+2, rect.width(), 20)
            result.append((rect, name, frequency))
        return result

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.stop_requested.emit()
        elif event.key() == QtCore.Qt.Key_F11:
            self.fullscreen_requested.emit()
        else:
            super().keyPressEvent(event)

    def target_at(self, point):
        # Match the full four-way layout, including padding, labels and idle areas.
        point = QtCore.QPointF(point)
        if not (0 <= point.x() < self.width() and 0 <= point.y() < self.height()):
            return None
        return int(point.x() >= self.width()/2) + 2*int(point.y() >= self.height()/2)

    def mouseMoveEvent(self, event):
        if self.mouse_test_enabled:
            target = self.target_at(event.pos())
            if target is not None:
                self.mouse_target_changed.emit(target)
        super().mouseMoveEvent(event)

    def enterEvent(self, event):
        if self.mouse_test_enabled:
            target = self.target_at(self.mapFromGlobal(QtGui.QCursor.pos()))
            if target is not None:
                self.mouse_target_changed.emit(target)
        super().enterEvent(event)

    def check_mouse_position(self):
        point = QtGui.QCursor.pos()
        if self.mouse_test_enabled and QtWidgets.QApplication.widgetAt(point) is self:
            target = self.target_at(self.mapFromGlobal(point))
            if target is not None:
                self.mouse_target_changed.emit(target)


class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("EEG_Test · 八通道 SSVEP Demo")
        available = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1640, available.width()), min(1200, available.height()))
        self.display_refresh_hz = args.refresh_hz or QtWidgets.QApplication.primaryScreen().refreshRate()
        # Use an integer display-refresh divisor near 60 fps, rather than require Python
        # to repaint 240 times/sec. Every presentation frame spans whole display scans.
        self.scan_divisor = max(1, round(self.display_refresh_hz / 60))
        self.refresh_hz = self.display_refresh_hz / self.scan_divisor
        self.targets = make_targets(self.refresh_hz, args.frequency_profile)
        self.timing = FrameTiming(self.refresh_hz)
        self.buffer = SampleBuffer()
        self.acquisition = None
        self.decoder = None
        self.recorder = None
        self.stimulus_start = None
        self.presentation_requested_at = None
        self.result_sender = None
        self.mouse_latch = MouseTargetLatch()
        self.audience = AudienceBridge(self.targets, self.timing, self)
        self.audience.stop_requested.connect(self.stop_stimulus)
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(250)
        self.mouse_result_timer = QtCore.QTimer(self)
        self.mouse_result_timer.timeout.connect(self._publish_mouse_result)
        self.mouse_result_timer.start(500)
        QtWidgets.QApplication.instance().applicationStateChanged.connect(self._application_state)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.stop_stimulus)
        QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        if args.result_port:
            QtCore.QTimer.singleShot(0, self.toggle_result_output)

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#101923; color:#dce7f2; font-family:'Microsoft YaHei'; }
            QLabel#title { font-size:22px; font-weight:bold; }
            QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background:#223245; border:1px solid #456078; border-radius:5px; padding:6px; }
            QPushButton:disabled { color:#697e91; } QPushButton:hover { background:#304b60; }
            QProgressBar { border:0; background:#223245; border-radius:4px; text-align:center; }
            QProgressBar::chunk { background:#249d85; border-radius:4px; }
        """)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        title = QtWidgets.QLabel("SSVEP · 四区注视：左肩 / 右肩 / 左肘 / 右肘")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel("1–8：PO7 · PO3 · POz · PO4 · PO8 · O1 · Oz · O2　|　500 Hz　|　Esc 停止，F11 全屏"))
        self.settings = QtWidgets.QGroupBox("采集与解码设置")
        self.settings.setCheckable(True)
        self.settings.setChecked(True)
        settings_layout = QtWidgets.QVBoxLayout(self.settings)
        self.settings_body = QtWidgets.QWidget()
        settings_layout.addWidget(self.settings_body)
        settings_body_layout = QtWidgets.QVBoxLayout(self.settings_body)
        settings_body_layout.setContentsMargins(0, 0, 0, 0)
        self.settings.toggled.connect(self.settings_body.setVisible)
        layout.addWidget(self.settings)
        bar = QtWidgets.QHBoxLayout()
        settings_body_layout.addLayout(bar)
        self.source = QtWidgets.QComboBox()
        for label, value in (("USB 串口", "serial"), ("WiFi TCP", "tcp"), ("模拟数据（非脑电）", "simulation")):
            self.source.addItem(label, value)
        self.source.setCurrentIndex(2 if self.args.simulate else (1 if self.args.tcp else 0))
        self.endpoint = QtWidgets.QLineEdit(self.args.tcp or self.args.port)
        self.endpoint.setPlaceholderText("串口名 COM6 / 板子 IP")
        self.tcp_port = QtWidgets.QSpinBox()
        self.tcp_port.setRange(1, 65535)
        self.tcp_port.setValue(self.args.tcp_port)
        self.sim_target = QtWidgets.QComboBox()
        for target in self.targets:
            self.sim_target.addItem(f"模拟：{target.character} {target.frequency:.2f} Hz")
        self.sim_target.setCurrentIndex(2)
        self.connect_button = QtWidgets.QPushButton("连接采集")
        self.connect_button.clicked.connect(self.toggle_connection)
        for widget in (self.source, self.endpoint, self.tcp_port, self.sim_target, self.connect_button):
            bar.addWidget(widget)
        self.source.currentIndexChanged.connect(self._source_changed)
        self._source_changed()
        controls = QtWidgets.QHBoxLayout()
        settings_body_layout.addLayout(controls)
        controls.addWidget(QtWidgets.QLabel("解码窗"))
        self.window_seconds = QtWidgets.QDoubleSpinBox()
        self.window_seconds.setRange(2, 6)
        self.window_seconds.setValue(3)
        self.window_seconds.setSuffix(" 秒")
        controls.addWidget(self.window_seconds)
        controls.addWidget(QtWidgets.QLabel("测试时长"))
        self.duration = QtWidgets.QSpinBox()
        self.duration.setRange(10, 120)
        self.duration.setValue(20)
        self.duration.setSuffix(" 秒")
        controls.addWidget(self.duration)
        self.annotation = QtWidgets.QComboBox()
        self.annotation.addItem("自由注视（未标注）", 0)
        for i, target in enumerate(self.targets, 1):
            self.annotation.addItem(f"记录标签：注视「{target.character}」", i)
        controls.addWidget(self.annotation)
        self.frequency_profile = QtWidgets.QComboBox()
        self.frequency_profile.addItem("整帧频率（原方案）", "integer")
        self.frequency_profile.addItem("宽频 8 / 13 / 18 / 23 Hz", "wide")
        self.frequency_profile.setCurrentIndex(1 if self.args.frequency_profile == "wide" else 0)
        controls.addWidget(self.frequency_profile)
        self.decoder_method = QtWidgets.QComboBox()
        self.decoder_method.addItem("FBCCA · 滤波器组", "fbcca")
        self.decoder_method.addItem("CCA · 基线对照", "cca")
        self.decoder_method.setCurrentIndex(0 if self.args.decoder == "fbcca" else 1)
        controls.addWidget(self.decoder_method)
        mode_controls = QtWidgets.QHBoxLayout()
        settings_body_layout.addLayout(mode_controls)
        mode_controls.addWidget(QtWidgets.QLabel("运行模式"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("脑电模式", "eeg")
        self.mode.addItem("测试模式", "test")
        self.mode.setCurrentIndex(1 if self.args.mode == "test" else 0)
        mode_controls.addWidget(self.mode)
        mode_controls.addWidget(QtWidgets.QLabel("分数门限"))
        self.score_threshold = QtWidgets.QDoubleSpinBox()
        self.score_threshold.setDecimals(3)
        self.score_threshold.setRange(0, 1)
        self.score_threshold.setSingleStep(.01)
        self.score_threshold.setValue(.22)
        mode_controls.addWidget(self.score_threshold)
        mode_controls.addWidget(QtWidgets.QLabel("分差门限"))
        self.margin_threshold = QtWidgets.QDoubleSpinBox()
        self.margin_threshold.setDecimals(3)
        self.margin_threshold.setRange(0, 1)
        self.margin_threshold.setSingleStep(.005)
        self.margin_threshold.setValue(.015)
        mode_controls.addWidget(self.margin_threshold)
        mode_controls.addWidget(QtWidgets.QLabel("通道质量仅提示；SNR 不再硬性拦截"), 1)
        output_controls = QtWidgets.QHBoxLayout()
        settings_body_layout.addLayout(output_controls)
        output_controls.addWidget(QtWidgets.QLabel("结果输出串口"))
        self.result_port = QtWidgets.QComboBox()
        self.result_port.setEditable(True)
        self.result_port.setMinimumWidth(130)
        self.result_port.lineEdit().setPlaceholderText("例如 COM7，与采集口不同")
        output_controls.addWidget(self.result_port)
        self.result_port_refresh = QtWidgets.QPushButton("刷新串口")
        self.result_port_refresh.clicked.connect(self._refresh_result_ports)
        output_controls.addWidget(self.result_port_refresh)
        output_controls.addWidget(QtWidgets.QLabel("波特率"))
        self.result_baud = QtWidgets.QComboBox()
        self.result_baud.addItems([str(value) for value in (9600, 19200, 38400, 57600, 115200, 230400)])
        self.result_baud.setCurrentText(str(self.args.result_baud))
        output_controls.addWidget(self.result_baud)
        self.result_connect = QtWidgets.QPushButton("连接结果串口")
        self.result_connect.clicked.connect(self.toggle_result_output)
        output_controls.addWidget(self.result_connect)
        output_controls.addWidget(QtWidgets.QLabel("8N1 · 无流控 · 独立发送到另一台电脑"), 1)
        self._refresh_result_ports()
        self.result_port.setEditText(self.args.result_port or "")
        actions = QtWidgets.QHBoxLayout()
        layout.addLayout(actions)
        self.start_button = QtWidgets.QPushButton("开始闪烁")
        self.start_button.clicked.connect(self.start_stimulus)
        self.stop_button = QtWidgets.QPushButton("停止 / Esc")
        self.stop_button.clicked.connect(self.stop_stimulus)
        self.fullscreen_button = QtWidgets.QPushButton("全屏 / F11")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.audience_button = QtWidgets.QPushButton("展示大屏")
        self.audience_button.clicked.connect(lambda: self.audience.show_presentation())
        for widget in (self.start_button, self.stop_button, self.fullscreen_button, self.audience_button):
            actions.addWidget(widget)
        self.source_label = QtWidgets.QLabel("未连接。点击开始后才会闪烁；若不适，立即按 Esc。")
        layout.addWidget(self.source_label)
        self.result_status = QtWidgets.QLabel("结果输出：未连接（可选）")
        self.result_status.setStyleSheet("font-size:12px; color:#9fb4c8;")
        layout.addWidget(self.result_status)
        self.stimulus = StimulusWidget(self.targets, self.timing)
        self.stimulus.first_frame.connect(self._first_frame)
        self.stimulus.stop_requested.connect(self.stop_stimulus)
        self.stimulus.fullscreen_requested.connect(self.toggle_fullscreen)
        self.stimulus.mouse_target_changed.connect(self._mouse_target_changed)
        self.stimulus_container = self.stimulus
        self.stimulus_container.setMinimumSize(600, 360)
        layout.addWidget(self.stimulus_container, 1)
        self.notice_label = QtWidgets.QLabel("连接采集后开始；识别结果和所有图表仅在“展示大屏”显示。")
        self.notice_label.setStyleSheet("font-size:12px; color:#9fb4c8;")
        layout.addWidget(self.notice_label)
        self.health_label = QtWidgets.QLabel(f"显示器 {self.display_refresh_hz:.3f} Hz；计划呈现 {self.refresh_hz:.3f} 帧/秒（整帧分频）")
        layout.addWidget(self.health_label)
        self.frequency_profile.currentIndexChanged.connect(self._frequency_changed)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _mode_changed(self):
        self.mouse_latch.reset()
        test = self.mode.currentData() == "test"
        self.stimulus.mouse_test_enabled = test
        for widget in (self.decoder_method, self.score_threshold, self.margin_threshold):
            widget.setEnabled(not test)
        self.audience.set_mode(self.mode.currentData())
        self._clear_result_output()
        self.notice_label.setText(
            "测试模式：按最近进入的四分区输出，离开保持；脑电照常采集。" if test else
            "脑电模式：低门限确认，通道异常仅提示；结果来自脑电分类。")

    def _mouse_target_changed(self, target):
        if self.mode.currentData() == "test" and self.mouse_latch.observe(target):
            self._publish_mouse_result()

    def _publish_mouse_result(self):
        if self.recorder is None or self.mode.currentData() != "test":
            return
        result = self.mouse_latch.result(self.targets)
        self.recorder.submit("decision", result)
        self.audience.receive_result(result)
        self._forward_result(result)

    def _wire_mode(self):
        # The receiving host uses one tag; actual source/mode remain in local logs.
        return "EEG"

    def _refresh_result_ports(self):
        selected = self.result_port.currentText()
        self.result_port.clear()
        self.result_port.addItems([item.device for item in serial.tools.list_ports.comports()])
        self.result_port.setEditText(selected)

    def toggle_result_output(self):
        if self.result_sender is not None and self.result_sender.is_running():
            self.result_sender.request_close()
        else:
            port = self.result_port.currentText().strip()
            if not port:
                self.result_status.setText("结果输出：请先选择或填写输出串口，例如 COM7")
                return
            if self.source.currentData() == "serial" and same_serial_port(port, self.endpoint.text()):
                self.result_status.setText("结果输出：不能与脑电采集使用同一个串口，请选择另一个 USB 转串口")
                return
            self.result_sender = ResultSerialSender(port, int(self.result_baud.currentText()), mode=self._wire_mode())
            self.result_sender.start()
        self._refresh_result_status()

    def _refresh_result_status(self):
        if self.result_sender is None:
            return
        info = self.result_sender.snapshot()
        running = info["running"]
        for widget in (self.result_port, self.result_baud, self.result_port_refresh):
            widget.setEnabled(not running)
        self.result_connect.setEnabled(not info["closing"])
        self.result_connect.setText("断开结果串口" if running else "连接结果串口")
        stage = "正在断开" if info["closing"] else "已连接" if info["connected"] else "正在连接" if running else "已断开"
        message = f"结果输出 {info['port']} / {info['baudrate']}：{stage} · 已写入 {info['sent_count']} 条"
        if info["error"]:
            message += f" · {info['error']}"
        self.result_status.setText(message)
        self.result_status.setToolTip(info["last_line"] or info["error"])

    def _result_output_snapshot(self):
        return self.result_sender.snapshot() if self.result_sender is not None else {"connected": False, "sent_count": 0}

    def _clear_result_output(self):
        if self.result_sender is not None:
            self.result_sender.publish(self._wire_mode())

    def _forward_result(self, result):
        if self.result_sender is None:
            return
        target = result.get("confirmed")
        self.result_sender.publish(self._wire_mode(), 0 if target is None else target+1,
                                   0. if target is None else self.targets[target].frequency,
                                   evidence_time=result["monotonic_s"])

    def _frequency_changed(self):
        try:
            targets = make_targets(self.refresh_hz, self.frequency_profile.currentData())
        except ValueError as exc:
            self.notice_label.setText(str(exc))
            self.frequency_profile.blockSignals(True)
            self.frequency_profile.setCurrentIndex(1-self.frequency_profile.currentIndex())
            self.frequency_profile.blockSignals(False)
            return
        if self.acquisition is not None:
            self.toggle_connection()
        visible = self.audience.process is not None and self.audience.process.is_alive()
        self.audience.shutdown()
        self.audience.deleteLater()
        self.targets = targets
        self.stimulus.targets = targets
        self.stimulus.update()
        self.sim_target.clear()
        self.annotation.clear()
        self.annotation.addItem("自由注视（未标注）", 0)
        for i, target in enumerate(targets, 1):
            self.sim_target.addItem(f"模拟：{target.character} {target.frequency:.2f} Hz")
            self.annotation.addItem(f"记录标签：注视「{target.character}」", i)
        self.sim_target.setCurrentIndex(2)
        self.audience = AudienceBridge(targets, self.timing, self)
        self.audience.stop_requested.connect(self.stop_stimulus)
        self.mouse_latch.reset()
        self.audience.set_mode(self.mode.currentData())
        self._clear_result_output()
        if visible:
            self.audience.show_presentation()
        self.notice_label.setText("频率方案已切换，请重新连接采集；以每个区域下方标出的频率为准。")

    def _source_changed(self):
        source = self.source.currentData()
        self.tcp_port.setVisible(source == "tcp")
        self.sim_target.setVisible(source == "simulation")
        self.endpoint.setVisible(source != "simulation")
        if source == "tcp" and self.endpoint.text().upper().startswith("COM"):
            self.endpoint.setText(self.args.tcp or "192.168.137.122")
        elif source == "serial" and "." in self.endpoint.text():
            self.endpoint.setText(self.args.port)
        if self.result_sender is not None:
            self._clear_result_output()

    def toggle_connection(self):
        if self.acquisition is not None:
            self.stop_stimulus()
            self.acquisition.stop()
            self.acquisition = None
            self.audience.set_stream(None, self.source.currentData(), self.endpoint.text().strip())
            self.connect_button.setText("连接采集")
            for widget in (self.source, self.endpoint, self.tcp_port, self.sim_target):
                widget.setEnabled(True)
            self.source_label.setText("采集已断开")
            return
        if (self.source.currentData() == "serial" and self.result_sender is not None
                and self.result_sender.is_running() and same_serial_port(self.endpoint.text(), self.result_sender.port)):
            self.notice_label.setText("脑电采集串口与结果输出串口相同，请修改其中一个串口")
            return
        self.buffer = SampleBuffer()
        self.acquisition = AcquisitionThread(
            self.source.currentData(), self.buffer, self.endpoint.text().strip(), self.tcp_port.value(),
            self.targets[self.sim_target.currentIndex()].frequency, self)
        self.acquisition.status.connect(self.source_label.setText)
        self.acquisition.received_rate.connect(self._rate_updated)
        self.acquisition.start()
        self.audience.set_stream(self.buffer, self.source.currentData(), self.endpoint.text().strip())
        self.connect_button.setText("断开采集")
        for widget in (self.source, self.endpoint, self.tcp_port, self.sim_target):
            widget.setEnabled(False)

    def _rate_updated(self, rate):
        prefix = "模拟数据（不代表人体识别准确率）" if self.source.currentData() == "simulation" else "真实板卡数据"
        self.source_label.setText(f"{prefix}　|　接收 {rate:.0f} 行/秒　|　信号单位 µV")

    def start_stimulus(self):
        if self.recorder is not None:
            return
        current_screen = self.windowHandle().screen() if self.windowHandle() else QtWidgets.QApplication.primaryScreen()
        if not self.args.refresh_hz and abs(current_screen.refreshRate() - self.display_refresh_hz) > 0.2:
            self.notice_label.setText("窗口已移到不同刷新率的屏幕，请在该屏幕上重新启动并设置 --refresh-hz")
            return
        if self.acquisition is None or not self.acquisition.isRunning():
            self.notice_label.setText("请先连接采集")
            return
        data, stamps = self.buffer.snapshot(1.5)
        reason, _ = assess_window(data, stamps)
        self.active_gate = dict(RELAXED_GATE, minimum_score=self.score_threshold.value(),
                                minimum_margin=self.margin_threshold.value())
        self._clear_result_output()
        metadata = {
            "source": self.source.currentData(), "endpoint": self.endpoint.text().strip(),
            "mode": self.mode.currentData(), "mode_name": self.mode.currentText(),
            "result_source": "mouse" if self.mode.currentData() == "test" else "eeg",
            "result_serial_start": self._result_output_snapshot(),
            "result_serial_mode": self._wire_mode(),
            "channel_quality_blocks_decoding": False,
            "initial_quality_warning": reason,
            "neutral_render_test": self.stimulus.neutral_render_test,
            "sample_rate_hz": SAMPLE_RATE, "presentation_rate_hz": self.refresh_hz,
            "display_refresh_hz": self.display_refresh_hz,
            "planned_scans_per_presentation_frame": self.scan_divisor,
            "renderer": "Qt QWidget raster",
            "frame_timestamp_kind": "CPU paint completion; not display scan or photon time",
            "targets": [dict(target_id=i, **asdict(t)) for i, t in enumerate(self.targets, 1)],
            "annotation_target_id": self.annotation.currentData(),
            "annotation_note": "Manually declared ground truth; NEVER given to the decoder",
            "simulation_target_id": self.sim_target.currentIndex() + 1 if self.source.currentData() == "simulation" else None,
            "window_seconds": self.window_seconds.value(), "update_seconds": 0.5,
            "frequency_profile": self.frequency_profile.currentData(),
            "target_layout": ["top_left", "top_right", "bottom_left", "bottom_right"],
            "decoder": self.decoder_method.currentData() if self.mode.currentData() == "eeg" else "mouse_latch",
            "decoder_config": None if self.mode.currentData() == "test" else {"harmonics": 3, "filter_bands_hz": [[f, 90] for f in (6, 14, 22, 30, 38)],
                               "filter_order": 4, "weight_law": "n**-1.25 + 0.25"} if self.decoder_method.currentData() == "fbcca" else {"harmonics": 3},
            "decision_gate": self.active_gate if self.mode.currentData() == "eeg" else None,
            "preprocessing": "omit constant/mostly railed channels; median/MAD clipping at 6 scales; no minimum channel count",
            "mouse_result_note": "Last entered full stimulus quadrant (including labels and padding), latched outside; never an EEG classification or EEG ground-truth label",
        }
        try:
            self.recorder = SessionLog(self.args.output, metadata)
        except OSError as exc:
            self.notice_label.setText(f"无法创建记录目录: {exc}")
            return
        self.notice_label.setToolTip(f"记录目录：{self.recorder.path}")
        self.acquisition.set_recording(self.recorder, False, self.annotation.currentData())
        if self.mode.currentData() == "eeg":
            self.decoder = DecodingThread(self.buffer, self.targets, self.timing, self.recorder,
                                          self.window_seconds.value(), self, method=self.decoder_method.currentData(),
                                          gate_config=self.active_gate)
            self.decoder.result.connect(self._decoded)
            self.decoder.start()
        self.stimulus_start = None
        self.presentation_requested_at = time.perf_counter()
        self.audience.begin_trial()
        self.settings_was_expanded = self.settings.isChecked()
        self.settings.setChecked(False)
        self.settings.setEnabled(False)
        self.stimulus.begin(self.recorder)
        QtCore.QTimer.singleShot(0, self.stimulus.check_mouse_position)
        self.notice_label.setText("测试模式：进入对应四分区即可选择，离开保持；脑电照常采集。"
                                  if self.mode.currentData() == "test" else "积累刺激后的数据；通道异常仅提示…")
        self._publish_mouse_result()
        for widget in (self.window_seconds, self.duration, self.annotation, self.start_button, self.frequency_profile, self.decoder_method):
            widget.setEnabled(False)

    def _first_frame(self, stamp):
        self.stimulus_start = stamp
        if self.decoder:
            self.decoder.started_at = stamp
        if self.acquisition:
            self.acquisition.mark_stimulus_on()

    def _decoded(self, result):
        if (self.recorder is None or self.mode.currentData() != "eeg"
                or result.get("monotonic_s", 0) < (self.presentation_requested_at or float("inf"))):
            return
        self.audience.receive_result(result)
        self._forward_result(result)

    def _poll(self):
        self._refresh_result_status()
        if self.recorder is not None:
            if self.acquisition is None or not self.acquisition.isRunning():
                self.stop_stimulus()
                self.notice_label.setText("采集已停止，本轮已结束；结果串口输出编号 0")
                return
            # A separate audience process is part of this experiment. Its focus
            # heartbeat preserves the stop-on-switch behavior for other apps.
            if (QtWidgets.QApplication.applicationState() != QtCore.Qt.ApplicationActive
                    and not self.audience.is_active()):
                self.stop_stimulus()
                return
            if self.stimulus_start is None and time.perf_counter() - self.presentation_requested_at > 2:
                self.stop_stimulus()
                self.notice_label.setText("画面未能开始更新，请停止后重试")
                return
            if self.recorder.error:
                error = self.recorder.error
                self.stop_stimulus()
                self.notice_label.setText(error)
                return
            stats = self.timing.summary()
            remaining = self.duration.value() - (time.perf_counter() - self.stimulus_start) if self.stimulus_start else self.duration.value()
            self.health_label.setText(
                f"画面更新 {stats['fps']:.1f} 帧/秒　|　最长间隔 {stats['max_interval_ms']:.0f} ms　"
                f"|　长帧 {stats['late_fraction']:.1%}　|　剩余 {max(0,remaining):.0f} 秒")
            if remaining <= 0:
                self.stop_stimulus()
        elif self.acquisition is not None and self.acquisition.isRunning():
            data, stamps = self.buffer.snapshot(1.5)
            reason, good = assess_window(data, stamps)
            self.health_label.setText(("质量提示（不阻止开始）："+reason) if reason else
                                     f"数据已到达，质量通过 {np.count_nonzero(good)}/8；可开始闪烁")

    def stop_stimulus(self):
        self._clear_result_output()
        if self.recorder is None:
            self.stimulus.end()
            return
        stats = self.timing.summary()
        self.stimulus.end()
        self.audience.end_trial()
        if self.decoder is not None:
            self.decoder.stop()
            self.decoder = None
        if self.acquisition is not None:
            self.acquisition.set_recording(None)
        recorder, self.recorder = self.recorder, None
        try:
            recorder.close(stopped_local=time.strftime("%Y-%m-%d %H:%M:%S"),
                           final_frame_timing=stats, stimulus_first_paint_monotonic_s=self.stimulus_start,
                           result_serial_end=self._result_output_snapshot())
            message = f"已停止，记录保存到 {recorder.path}"
            if recorder.error:
                message += f"（{recorder.error}）"
            self.health_label.setText(message)
        except (OSError, queue.Full) as exc:
            self.health_label.setText(f"已停止，记录保存失败: {exc}")
        self.notice_label.setText("已停止闪烁")
        for widget in (self.window_seconds, self.duration, self.annotation, self.start_button, self.frequency_profile, self.decoder_method):
            widget.setEnabled(True)
        self.settings.setEnabled(True)
        self.settings.setChecked(self.settings_was_expanded)
        test = self.mode.currentData() == "test"
        for widget in (self.decoder_method, self.score_threshold, self.margin_threshold):
            widget.setEnabled(not test)
        self.stimulus_start = None
        self.presentation_requested_at = None

    def toggle_fullscreen(self):
        # Resizing during a run changes frame timing; restart explicitly after switching.
        self.stop_stimulus()
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _application_state(self, state):
        if state != QtCore.Qt.ApplicationActive:
            QtCore.QTimer.singleShot(300, self._check_application_focus)

    def _check_application_focus(self):
        if (QtWidgets.QApplication.applicationState() != QtCore.Qt.ApplicationActive
                and not self.audience.is_active()):
            self.stop_stimulus()

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.WindowStateChange and self.isMinimized():
            self.stop_stimulus()
        super().changeEvent(event)

    def closeEvent(self, event):
        self.stop_stimulus()
        if self.result_sender is not None:
            self.result_sender.close()
        if self.acquisition is not None:
            self.acquisition.stop()
        self.audience.shutdown()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EEG_Test 八通道 SSVEP：四区闪烁 + FBCCA + 串口/TCP/模拟")
    parser.add_argument("--port", default="COM6", help="USB 串口，默认 COM6")
    parser.add_argument("--result-port", help="独立结果输出串口，例如 COM7；设置后启动时连接")
    parser.add_argument("--result-baud", type=int, default=115200, choices=(9600, 19200, 38400, 57600, 115200, 230400))
    parser.add_argument("--tcp", metavar="IP", help="使用 WiFi TCP 数据源")
    parser.add_argument("--tcp-port", type=int, default=3333)
    parser.add_argument("--simulate", action="store_true", help="明确标识的合成信号，用于验证软件")
    parser.add_argument("--mode", choices=("eeg", "test"), default="eeg", help="脑电模式或测试模式（鼠标最近区域决定结果）")
    parser.add_argument("--frequency-profile", choices=("integer", "wide"), default="integer", help="原整帧方案或 8/13/18/23 Hz 宽频方案")
    parser.add_argument("--decoder", choices=("fbcca", "cca"), default="fbcca", help="默认 FBCCA；CCA 用于对照")
    parser.add_argument("--no-audience", action="store_true", help="启动时不自动打开展示大屏，仍可点击按钮打开")
    parser.add_argument("--refresh-hz", type=float, help="覆盖 Qt 报告的刷新率，须与显示器实际设置一致")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "sessions")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        window = DemoWindow(args)
    except ValueError as exc:
        QtWidgets.QMessageBox.critical(None, "显示器配置", str(exc))
        return 2
    window.showMaximized()
    if not args.no_audience:
        window.audience.show_presentation()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
