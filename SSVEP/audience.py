# -*- coding: utf-8 -*-
"""Audience display: shared acquisition, background analysis, raster-only plots."""
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from .core import CHANNEL_NAMES
    from .display_data import DisplayAnalysisThread, MAP_EXTENT, POSTERIOR_POSITIONS, posterior_field
except ImportError:
    from core import CHANNEL_NAMES
    from display_data import DisplayAnalysisThread, MAP_EXTENT, POSTERIOR_POSITIONS, posterior_field


BG = "#08121f"
CARD = "#101f30"
TEXT = "#e6f1ff"
MUTED = "#8fa9c2"
ACCENT = "#53e0be"
COLORS = ("#58c8ff", "#6be0c4", "#a6d977", "#eed47c", "#f3a27d", "#d19bff", "#78a1ff", "#f18cba")


def font(pixels, bold=False):
    f = QtGui.QFont("Microsoft YaHei")
    f.setPixelSize(pixels)
    f.setBold(bold)
    return f


def text(painter, rect, value, pixels=13, color=TEXT, align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, bold=False):
    painter.setFont(font(pixels, bold))
    painter.setPen(QtGui.QColor(color))
    rectangle = QtCore.QRectF(*rect) if isinstance(rect, tuple) else QtCore.QRectF(rect)
    painter.drawText(rectangle, align, str(value))


class PlotCard(QtWidgets.QWidget):
    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.title, self.subtitle = title, subtitle
        self.setMinimumSize(260, 160)

    def painter(self):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(QtGui.QColor("#253b51"), 1))
        p.setBrush(QtGui.QColor(CARD))
        p.drawRoundedRect(QtCore.QRectF(1, 1, self.width()-2, self.height()-2), 13, 13)
        text(p, (20, 10, self.width()-40, 29), self.title, 17, bold=True)
        text(p, (20, 40, self.width()-40, 23), self.subtitle, 12, MUTED)
        return p


class WaveformCard(PlotCard):
    def __init__(self):
        super().__init__("01  实时脑电波形", "八个后脑通道 · 显示滤波 1–45 Hz · 原始数据另行记录")
        self.payload = None
        self.scale = 25
        self.setMinimumHeight(280)

    def paintEvent(self, event):
        p = self.painter()
        area = QtCore.QRectF(83, 78, self.width()-110, self.height()-116)
        p.setPen(QtGui.QPen(QtGui.QColor("#26384b"), 1, QtCore.Qt.DotLine))
        for sec in range(5):
            x = area.left() + sec / 4 * area.width()
            p.drawLine(QtCore.QPointF(x, area.top()), QtCore.QPointF(x, area.bottom()))
            text(p, (x-25, area.bottom()+5, 50, 22), f"{sec-4} s", 11, MUTED, QtCore.Qt.AlignCenter)
        step = area.height()/8
        for i, name in enumerate(CHANNEL_NAMES):
            y = area.top() + (i+.5)*step
            text(p, (13, y-13, 62, 26), f"{i+1}  {name}", 12, COLORS[i])
            p.setPen(QtGui.QPen(QtGui.QColor("#223247"), 1))
            p.drawLine(QtCore.QPointF(area.left(), y), QtCore.QPointF(area.right(), y))
        data = self.payload
        if data and data.get("available") and not data.get("stale"):
            wave, times = data["waveform"], data["times"]
            # Limit geometry work to display pixels, preserving this 45 Hz display band.
            stride = max(1, len(times)//max(400, round(area.width())))
            xx = area.left() + (times[::stride]+4)/4*area.width()
            p.save()
            p.setClipRect(area)
            for i in range(8):
                center = area.top()+(i+.5)*step
                yy = center-np.clip(wave[::stride, i]/self.scale, -1, 1)*step*.40
                path = QtGui.QPainterPath()
                for j, (x, y) in enumerate(zip(xx, yy)):
                    path.moveTo(float(x), float(y)) if j == 0 else path.lineTo(float(x), float(y))
                p.setPen(QtGui.QPen(QtGui.QColor(COLORS[i] if data["good"][i] else "#657689"), 1.25))
                p.drawPath(path)
            p.restore()
        else:
            text(p, area, "等待采集数据" if not data else data.get("reason", "数据已中断"), 18, MUTED, QtCore.Qt.AlignCenter)
        text(p, (self.width()-260, 13, 235, 22), f"每通道 ±{self.scale} µV（超出截幅）", 11, MUTED, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        p.end()


class PowerCard(PlotCard):
    def __init__(self, targets):
        super().__init__("02  功率谱 · 寻找频率响应", "细线：有数据通道 · 白线：均值 · 虚线：刺激频率")
        self.targets = targets
        self.payload = None
        self.setMinimumHeight(220)

    def paintEvent(self, event):
        p = self.painter()
        area = QtCore.QRectF(65, 91, self.width()-90, self.height()-137)
        lo, hi = -25., 25.
        data = self.payload
        channels = data.get("plot_channels", data.get("good")) if data else None
        if data and data.get("available") and np.any(channels) and not data.get("stale"):
            db = 10*np.log10(np.maximum(data["psd"][:, channels], 1e-12))
            lo = min(-20., np.floor(np.percentile(db, 5)/10)*10)
            hi = max(20., np.ceil(np.max(db)/10)*10)
        for v in np.linspace(lo, hi, 5):
            y = area.bottom()-(v-lo)/(hi-lo)*area.height()
            p.setPen(QtGui.QPen(QtGui.QColor("#25374b"), 1))
            p.drawLine(QtCore.QPointF(area.left(), y), QtCore.QPointF(area.right(), y))
            text(p, (5, y-10, 51, 20), f"{v:.0f}", 10, MUTED, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        for f in (5, 10, 15, 20, 25, 30, 35, 40, 45):
            x = area.left()+(f-2)/43*area.width()
            text(p, (x-15, area.bottom()+5, 30, 20), f"{f}", 10, MUTED, QtCore.Qt.AlignCenter)
        for i, target in enumerate(self.targets):
            x = area.left()+(target.frequency-2)/43*area.width()
            p.setPen(QtGui.QPen(QtGui.QColor("#426976"), 1, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(x, area.top()), QtCore.QPointF(x, area.bottom()))
            text(p, (x-38, area.top()-24, 76, 20), f"{target.character} {target.frequency:.2f}", 10, ACCENT, QtCore.Qt.AlignCenter)
        if data and data.get("available") and np.any(channels) and not data.get("stale"):
            frequencies, psd = data["frequencies"], data["psd"]
            traces = [(psd[:, i], COLORS[i], .8) for i in range(8) if channels[i]]
            traces.append((psd[:, channels].mean(axis=1), "#e6f1ff", 2.2))
            p.save()
            p.setClipRect(area)
            for power, color, width in traces:
                xx = area.left()+(frequencies-2)/43*area.width()
                yy = area.bottom()-(10*np.log10(np.maximum(power, 1e-12))-lo)/(hi-lo)*area.height()
                path = QtGui.QPainterPath()
                for j, (x, y) in enumerate(zip(xx, yy)):
                    path.moveTo(float(x), float(y)) if j == 0 else path.lineTo(float(x), float(y))
                p.setPen(QtGui.QPen(QtGui.QColor(color), width))
                p.drawPath(path)
            p.restore()
        else:
            text(p, area, "等待有效脑电", 17, MUTED, QtCore.Qt.AlignCenter)
        text(p, (18, 66, 140, 19), "dB (µV²/Hz)", 10, MUTED)
        text(p, (self.width()-56, self.height()-29, 37, 20), "Hz", 11, MUTED)
        p.end()


class TopographyCard(PlotCard):
    def __init__(self):
        super().__init__("03  后脑局部功率分布", "后脑实测功率 · 异常通道黄色标记")
        self.image = None
        self.good = np.zeros(8, dtype=bool)
        self.quality_good = np.zeros(8, dtype=bool)
        self.value_range = (0., 1.)
        self.band_label = "6–30 Hz"
        self.key = None
        self.setMinimumHeight(280)

    def set_data(self, payload, band_index, label):
        if payload is None or not payload.get("available") or payload.get("stale"):
            self.image, self.key = None, None
            self.good = np.zeros(8, dtype=bool)
            self.quality_good = np.zeros(8, dtype=bool)
            self.update()
            return
        channels = payload.get("plot_channels", payload["good"])
        key = (payload["spectral_id"], band_index, tuple(channels), tuple(payload["good"]))
        if key == self.key:
            return
        self.key, self.band_label = key, label
        self.good = channels
        self.quality_good = payload["good"]
        values = payload["band_db"][band_index]
        field = posterior_field(values, self.good)
        inside = np.isfinite(field)
        if not np.any(inside):
            self.image = None
        else:
            low, high = float(np.min(values[self.good])), float(np.max(values[self.good]))
            if high-low < 6:
                center = (high+low)/2
                low, high = center-3, center+3
            self.value_range = (low, high)
            z = np.nan_to_num(np.clip((field-low)/(high-low), 0, 1), nan=0.)
            stops = np.array([[24, 61, 116], [31, 157, 179], [82, 224, 190], [238, 211, 107], [249, 125, 97]])
            rgba = np.empty((*field.shape, 4), dtype=np.uint8)
            for channel in range(3):
                rgba[..., channel] = np.interp(z, np.linspace(0, 1, len(stops)), stops[:, channel])
            rgba[..., 3] = inside.astype(np.uint8)*240
            self.image = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QtGui.QImage.Format_RGBA8888).copy()
        self.update()

    def paintEvent(self, event):
        p = self.painter()
        size = min(self.width()-70, (self.height()-145)*1.1)
        area = QtCore.QRectF((self.width()-size)/2, 75, size, size/1.1)
        p.setPen(QtGui.QPen(QtGui.QColor("#41617b"), 2))
        p.setBrush(QtGui.QColor("#15283b"))
        p.drawEllipse(area.adjusted(8, 0, -8, 0))
        if self.image is not None:
            p.drawImage(area, self.image)
        # Coordinates are schematic; a legend identifies this as a local rear view.
        for i, (x, y) in enumerate(POSTERIOR_POSITIONS):
            px = area.left()+(x-MAP_EXTENT[0])/(MAP_EXTENT[1]-MAP_EXTENT[0])*area.width()
            py = area.top()+(y-MAP_EXTENT[2])/(MAP_EXTENT[3]-MAP_EXTENT[2])*area.height()
            node_color = TEXT if self.quality_good[i] else "#e9b96e" if self.good[i] else MUTED
            p.setPen(QtGui.QPen(QtGui.QColor(node_color), 1.5))
            p.setBrush(QtGui.QColor("#0a1828"))
            p.drawEllipse(QtCore.QPointF(px, py), 5, 5)
            text(p, (px-27, py+8, 54, 20), CHANNEL_NAMES[i], 11, node_color, QtCore.Qt.AlignCenter)
        text(p, (area.left(), area.top()+6, area.width(), 25), "未测量区域", 11, MUTED, QtCore.Qt.AlignCenter)
        legend_y = area.bottom()+8
        if self.image is not None:
            gradient = QtGui.QLinearGradient(35, 0, self.width()-35, 0)
            for stop, color in zip(np.linspace(0, 1, 5), ("#183d74", "#1f9db3", "#52e0be", "#eed36b", "#f97d61")):
                gradient.setColorAt(float(stop), QtGui.QColor(color))
            p.fillRect(QtCore.QRectF(35, legend_y, self.width()-70, 7), gradient)
            low, high = self.value_range
            text(p, (35, legend_y+9, self.width()-70, 21), f"{low:.1f}                  {self.band_label} · dB µV²                  {high:.1f}", 10, MUTED, QtCore.Qt.AlignCenter)
        else:
            text(p, (25, legend_y, self.width()-50, 24), "等待足够的实测数据生成分布图", 12, MUTED, QtCore.Qt.AlignCenter)
        text(p, (15, self.height()-27, self.width()-30, 20), "示意位置 · 局部自动色阶 · 不代表全脑活动或源定位", 10, MUTED, QtCore.Qt.AlignCenter)
        p.end()


class ResultCard(PlotCard):
    def __init__(self, targets):
        super().__init__("识别结果", "候选比较 → 质量检查 → 连续确认")
        self.targets = targets
        self.result = None
        self.last_confirmed = None
        self.active = False
        self.simulated = False
        self.finished = False
        self.test_mode = False
        self.setMinimumHeight(310)

    def begin(self):
        self.result = self.last_confirmed = None
        self.active, self.finished = True, False
        self.update()

    def receive(self, result):
        self.result = result
        if result.get("confirmed") is not None:
            self.last_confirmed = result
        self.update()

    def paintEvent(self, event):
        result = self.last_confirmed if self.finished else self.result
        mouse_result = self.test_mode or bool(result and result.get("result_source") == "mouse")
        self.title = "" if mouse_result else "脑电识别结果"
        self.subtitle = "" if mouse_result else "低门限分类 → 分数平滑 → 连续确认"
        p = self.painter()
        if mouse_result:
            self._paint_mouse_result(p, result)
            p.end()
            return
        confirmed = result.get("confirmed") if result else None
        y = 70
        if confirmed is not None:
            target = self.targets[confirmed]
            text(p, (20, y, self.width()*.44, 86), target.character, 68, ACCENT, QtCore.Qt.AlignCenter, True)
            text(p, (self.width()*.52, y+7, self.width()*.46-20, 43), f"{target.frequency:.2f} Hz", 28, ACCENT, bold=True)
            note = "本轮最近确认" if self.finished else "已通过连续确认"
            if self.simulated:
                note = "模拟信号 · " + note
            text(p, (self.width()*.52, y+50, self.width()*.46-20, 30), note, 12, MUTED)
        else:
            text(p, (20, y+8, self.width()-40, 65), "识别中" if self.active else "等待实验", 37, TEXT, QtCore.Qt.AlignCenter, True)
        reason = result.get("reason", "") if result else "连接采集后，在范式窗口点击“开始闪烁”"
        if confirmed is not None:
            reason = f"{result.get('method', 'CCA')} 平滑分数 {result['score']:.3f} · 分差 {result['margin']:.3f}"
            reason += f"\n脑电主峰 {result['spectral_peak_hz']:.1f} Hz"
            if "snr_db" in result:
                reason += f" · 目标 SNR {result['snr_db'][confirmed]:.1f} dB"
        elif result and "best" in result:
            target = self.targets[result["best"]]
            reason = f"候选 {target.character} {target.frequency:.2f} Hz；尚未确认\n{reason}"
        text(p, (22, y+92, self.width()-44, 45), reason, 11, MUTED, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap)
        bars_y = y+145
        scores = result.get("scores", [0]*4) if result else [0]*4
        bar_height = max(14, min(19, (self.height()-bars_y-28)/4))
        for i, (target, score) in enumerate(zip(self.targets, scores)):
            yy = bars_y+i*(bar_height+5)
            text(p, (23, yy, 54, bar_height), target.character, 12, TEXT)
            width = self.width()-158
            p.fillRect(QtCore.QRectF(83, yy+3, width, bar_height-6), QtGui.QColor("#24394e"))
            p.fillRect(QtCore.QRectF(83, yy+3, width*np.clip(score, 0, 1), bar_height-6), QtGui.QColor(ACCENT if i == confirmed else "#4b89b3"))
            text(p, (self.width()-66, yy, 48, bar_height), f"{score:.3f}", 11, MUTED)
        p.end()

    def _paint_mouse_result(self, p, result):
        selected = result.get("confirmed") if result else None
        if selected is None:
            return
        target = self.targets[selected]
        name_size = int(np.clip(self.width()*.18, 48, 108))
        frequency_size = int(np.clip(self.width()*.08, 24, 42))
        name_height, frequency_height = name_size*1.4, frequency_size*1.5
        top = (self.height()-name_height-frequency_height-18)/2
        text(p, (20, top, self.width()-40, name_height), target.character,
             name_size, ACCENT, QtCore.Qt.AlignCenter, True)
        text(p, (20, top+name_height+18, self.width()-40, frequency_height),
             f"{target.frequency:.2f} Hz", frequency_size, ACCENT, QtCore.Qt.AlignCenter)


class AudienceWindow(QtWidgets.QMainWindow):
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, targets, timing, parent=None, analysis=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.targets, self.timing = targets, timing
        self.setWindowTitle("EEG_Test · 脑电实验展示大屏")
        self.resize(1600, 1050)
        self.setMinimumSize(1080, 780)
        self.connected = self.active = self.simulated = False
        self.source_name = "未连接"
        self.latest = None
        self.remote_state = None
        self.remote_generation = -1
        self.show_sequence = 0
        self.result_key = None
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TEXT}; font-family:'Microsoft YaHei'; }}
            QComboBox, QPushButton {{ background:#162a3e; color:{TEXT}; border:1px solid #36516b; border-radius:6px; padding:6px 10px; }}
            QPushButton:hover {{ background:#23425a; }}
            QLabel#metric {{ background:{CARD}; border:1px solid #253b51; border-radius:10px; padding:12px 16px; font-size:14px; }}
        """)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(22, 17, 22, 15)
        layout.setSpacing(12)
        header = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("脑电 · 看见注视的过程")
        title.setStyleSheet("font-size:27px; font-weight:bold;")
        heading.addWidget(title)
        heading.addWidget(QtWidgets.QLabel("EEG_Test   /   八通道 SSVEP 实时实验展示"))
        header.addLayout(heading, 1)
        self.mode_label = QtWidgets.QLabel("等待连接")
        self.mode_label.setStyleSheet(f"font-size:17px; color:{ACCENT}; padding:10px;")
        header.addWidget(self.mode_label)
        fullscreen = QtWidgets.QPushButton("全屏 / F11")
        fullscreen.clicked.connect(self.toggle_fullscreen)
        header.addWidget(fullscreen)
        layout.addLayout(header)
        metrics = QtWidgets.QHBoxLayout()
        self.metrics = []
        for value in ("数据源  ·  未连接", "有效通道  ·  — / 8", "采样接收  ·  — 行/秒", "实验阶段  ·  等待连接"):
            label = QtWidgets.QLabel(value)
            label.setObjectName("metric")
            metrics.addWidget(label, 1)
            self.metrics.append(label)
        layout.addLayout(metrics)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        left, right = QtWidgets.QVBoxLayout(), QtWidgets.QVBoxLayout()
        left.setSpacing(12)
        right.setSpacing(12)
        self.waveform = WaveformCard()
        self.power = PowerCard(targets)
        self.result = ResultCard(targets)
        self.topography = TopographyCard()
        left.addWidget(self.waveform, 3)
        left.addWidget(self.power, 2)
        right.addWidget(self.result, 1)
        right.addWidget(self.topography, 1)
        body.addLayout(left, 7)
        body.addLayout(right, 4)
        layout.addLayout(body, 1)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("波形量程"))
        self.scale = QtWidgets.QComboBox()
        for value in (25, 50, 100, 200):
            self.scale.addItem(f"±{value} µV", value)
        self.scale.currentIndexChanged.connect(self._scale_changed)
        controls.addWidget(self.scale)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("地形图频带"))
        self.band = QtWidgets.QComboBox()
        self.band.addItem("宽带 6–30 Hz", "6–30 Hz")
        for target in targets:
            self.band.addItem(f"{target.character} · {target.frequency:.2f} Hz ±1", f"{target.frequency:.2f}±1 Hz")
        self.band.currentIndexChanged.connect(self._band_changed)
        controls.addWidget(self.band)
        controls.addStretch()
        self.status = QtWidgets.QLabel("连接与闪烁由范式窗口控制")
        controls.addWidget(self.status)
        layout.addLayout(controls)
        self.footer = QtWidgets.QLabel("展示与范式共用采集数据 · 图形处理在后台运行 · CCA 分数不是概率")
        self.footer.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        layout.addWidget(self.footer)
        self.analysis = analysis or DisplayAnalysisThread(targets, self)
        self.analysis.start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)
        QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.stop_requested.emit)

    def set_stream(self, buffer, source, endpoint):
        self.set_connection(buffer is not None, source, endpoint)
        self.analysis.set_buffer(buffer)

    def set_connection(self, connected, source, endpoint):
        self.connected = connected
        self.simulated = source == "simulation"
        self.source_name = "模拟信号" if self.simulated else (f"USB · {endpoint}" if source == "serial" else f"WiFi · {endpoint}")
        self.mode_label.setText("模拟信号 / 非人体脑电" if self.simulated else "真实脑电数据源" if self.connected else "采集已断开")
        self.result.simulated = self.simulated
        self.latest = None
        self.waveform.payload = self.power.payload = None
        self.topography.set_data(None, 0, "")
        self.result.result = self.result.last_confirmed = None
        self.result.finished = self.result.active = False
        self.result.update()
        self.metrics[0].setText("数据源  ·  " + (self.source_name if self.connected else "未连接"))
        self.waveform.update()
        self.power.update()

    def begin_trial(self):
        self.active = True
        self.result.begin()

    def receive_result(self, result):
        self.result.receive(result)

    def end_trial(self):
        self.active = False
        self.result.active = False
        self.result.finished = True
        self.result.update()

    def _scale_changed(self):
        self.waveform.scale = self.scale.currentData()
        self.waveform.update()

    def _band_changed(self):
        self.topography.set_data(self.latest, self.band.currentIndex(), self.band.currentData())

    def _refresh(self):
        payload = self.analysis.take_latest()
        if payload is not None:
            state = payload.get("bridge_state")
            if state is not None:
                self._apply_remote_state(state)
            self.latest = payload
        data = self.latest
        valid = bool(data and data.get("available") and not data.get("stale"))
        good = int(np.count_nonzero(data["good"])) if valid else 0
        rate = data.get("rate", 0) if valid else 0
        self.metrics[1].setText(f"质量通过  ·  {good} / 8")
        self.metrics[2].setText(f"采样接收  ·  {rate:.0f} 行/秒" if valid else "采样接收  ·  等待数据")
        stage = "刺激与解码中" if self.active else "采集中 / 待刺激" if self.connected else "等待连接"
        if self.result.test_mode:
            stage = "测试模式 · 运行中" if self.active else "测试模式 · 本轮结束" if self.result.finished else "测试模式 · 待开始"
        if self.connected and data and data.get("reason"):
            stage += " · 质量提示"
        self.metrics[3].setText("实验阶段  ·  " + stage)
        if self.active:
            timing = self.remote_state["frame_timing"] if self.remote_state else self.timing.summary()
            self.status.setText(f"范式更新 {timing['fps']:.1f} 帧/秒 · 最长 {timing['max_interval_ms']:.0f} ms")
        else:
            self.status.setText(data.get("reason", "") if data and data.get("reason") else "连接与闪烁由范式窗口控制")
        if payload is not None and self.isVisible():
            self.waveform.payload = self.power.payload = payload
            self.waveform.update()
            self.power.update()
            self.topography.set_data(payload, self.band.currentIndex(), self.band.currentData())

    def _apply_remote_state(self, state):
        self.remote_state = state
        if self.remote_generation != state["generation"]:
            self.remote_generation = state["generation"]
            self.set_connection(state["connected"], state["source"], state["endpoint"])
        self.active = self.result.active = state["active"]
        self.result.finished = state["finished"]
        self.result.result = state["result"]
        self.result.last_confirmed = state["last_confirmed"]
        self.result.test_mode = state.get("mode", "eeg") == "test"
        if self.result.test_mode:
            self.mode_label.setText("测试模式 · " + ("模拟数据" if self.simulated else "脑电采集"))
            self.footer.setText("测试模式：按最近进入的区域输出；波形、频谱和功率图照常使用采集数据。")
        else:
            self.mode_label.setText("脑电模式 · " + ("模拟信号 / 非人体脑电" if self.simulated else "脑电采集"))
            self.footer.setText("脑电模式：低门限确认，质量仅提示；分类分数不是概率。")
        key = (state.get("mode", "eeg"), state["active"], state["finished"], state["result"].get("monotonic_s") if state["result"] else None)
        if key != self.result_key:
            self.result_key = key
            self.result.update()
        if state["show_sequence"] != self.show_sequence:
            self.show_sequence = state["show_sequence"]
            self.show_presentation()

    def show_presentation(self):
        screens = QtWidgets.QApplication.screens()
        if len(screens) > 1 and not self.isVisible():
            primary_name = self.remote_state.get("stimulus_screen") if self.remote_state else ""
            if not primary_name:
                primary_name = QtWidgets.QApplication.primaryScreen().name()
            screen = next((s for s in screens if s.name() != primary_name), screens[-1])
            self.move(screen.availableGeometry().topLeft())
            self.showMaximized()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def shutdown(self):
        self.timer.stop()
        self.analysis.stop()
        self.hide()

    def closeEvent(self, event):
        self.hide()  # Closing this view must not close the experiment or acquisition.
        event.ignore()
