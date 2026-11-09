"""Serial/TCP/simulated input and a separate buffered session writer."""
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import socket
import threading
import time

import numpy as np
from PyQt5 import QtCore
import serial

try:
    from .core import (CHANNEL_NAMES, SAMPLE_RATE, CCADecoder, FBCCADecoder, StableDecisionGate,
                       RELAXED_GATE, prepare_decode_window, assess_window, parse_line, simulated_samples)
except ImportError:
    from core import (CHANNEL_NAMES, SAMPLE_RATE, CCADecoder, FBCCADecoder, StableDecisionGate,
                      RELAXED_GATE, prepare_decode_window, assess_window, parse_line, simulated_samples)


class SessionLog:
    def __init__(self, base_dir, metadata):
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = Path(base_dir) / name
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, created_local=datetime.now().isoformat(),
                             channel_names=list(CHANNEL_NAMES), units="uV",
                             timestamp_note="PC receipt/CPU paint completion times; not hardware/photodiode timestamps")
        self.messages = queue.Queue(maxsize=4096)
        self.error = ""
        self.dropped_messages = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="SSVEP-session-writer", daemon=True)
        self.thread.start()

    def submit(self, kind, rows):
        if self.closed:
            return
        try:
            self.messages.put_nowait((kind, rows))
        except queue.Full:
            self.dropped_messages += 1
            self.error = "记录队列已满：本次记录不完整"

    def _run(self):
        try:
            with (self.path / "raw_eeg.csv").open("w", newline="", encoding="utf-8") as raw, \
                 (self.path / "frames.csv").open("w", newline="", encoding="utf-8") as frames, \
                 (self.path / "decisions.jsonl").open("w", encoding="utf-8") as decisions:
                raw_writer, frame_writer = csv.writer(raw), csv.writer(frames)
                raw_writer.writerow(["sample_index", "received_monotonic_s", "stimulus_on",
                                     "attended_target_id", *CHANNEL_NAMES])
                frame_writer.writerow(["frame_index", "paint_finished_monotonic_s"])
                while True:
                    message = self.messages.get()
                    if message is None:
                        break
                    kind, rows = message
                    if kind == "raw":
                        raw_writer.writerows(rows)
                    elif kind == "frame":
                        frame_writer.writerow(rows)
                    else:
                        decisions.write(json.dumps(rows, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception as exc:
            self.error = f"记录写入失败: {exc}"

    def close(self, **extra_metadata):
        # Called only after acquisition has detached this logger and the renderer has stopped.
        self.closed = True
        if self.thread.is_alive():
            self.messages.put(None, timeout=3)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error = self.error or "记录线程未能及时完成，记录可能不完整"
        self.metadata.update(extra_metadata, dropped_log_messages=self.dropped_messages,
                             recording_error=self.error,
                             writer_finished=not self.thread.is_alive())
        with (self.path / "session.json").open("w", encoding="utf-8") as fp:
            json.dump(self.metadata, fp, ensure_ascii=False, indent=2, allow_nan=False)


class AcquisitionThread(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    received_rate = QtCore.pyqtSignal(float)

    def __init__(self, source, sample_buffer, endpoint="COM6", port=3333,
                 simulation_frequency=12.0, parent=None):
        super().__init__(parent)
        self.source = source
        self.buffer = sample_buffer
        self.endpoint = endpoint
        self.port = port
        self.simulation_frequency = simulation_frequency
        self.stop_event = threading.Event()
        self.handle = None
        self.recording_lock = threading.Lock()
        self.recorder = None
        self.stimulus_on = False
        self.attended_target_id = 0
        self.sample_index = 0

    def set_recording(self, recorder, stimulus_on=False, attended_target_id=0):
        with self.recording_lock:
            self.recorder = recorder
            self.stimulus_on = stimulus_on
            self.attended_target_id = attended_target_id

    def mark_stimulus_on(self):
        with self.recording_lock:
            self.stimulus_on = True

    def run(self):
        try:
            if self.source == "serial":
                dev = serial.Serial()
                dev.port, dev.baudrate, dev.timeout = self.endpoint, 115200, 0.2
                dev.dtr, dev.rts = False, False
                dev.open()
                self.handle = dev
                dev.reset_input_buffer()
                dev.write(b"1")  # Board's continuous acquisition command, never self-test.
            elif self.source == "tcp":
                self.handle = socket.create_connection((self.endpoint, self.port), timeout=3)
                self.handle.settimeout(0.2)
                self.handle.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.handle.sendall(b"1")
            self.status.emit("模拟数据已连接（不是人体脑电）" if self.source == "simulation" else "采集已连接")
            pending = bytearray()
            last_report = time.perf_counter()
            report_count = 0
            simulation_clock = last_report
            while not self.stop_event.is_set():
                rows = []
                if self.source == "simulation":
                    now = time.perf_counter()
                    due = int((now - simulation_clock) * SAMPLE_RATE) - self.sample_index
                    if due <= 0:
                        self.stop_event.wait(0.005)
                        continue
                    indices = np.arange(self.sample_index, self.sample_index + min(due, 100))
                    rows = list(simulated_samples(indices, self.simulation_frequency))
                else:
                    try:
                        if self.source == "serial":
                            chunk = self.handle.read(min(max(self.handle.in_waiting, 1), 65536))
                        else:
                            chunk = self.handle.recv(65536)
                            if not chunk:
                                raise ConnectionError("TCP 服务已关闭连接")
                    except socket.timeout:
                        continue
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, remaining = pending.partition(b"\n")
                        pending = bytearray(remaining)
                        values = parse_line(line)
                        if values is not None:
                            rows.append(values)
                    if len(pending) > 65536:
                        raise ValueError("数据流长时间没有换行，请检查固件和连接端口")
                now = time.perf_counter()
                with self.recording_lock:
                    saved = []
                    for values in rows:
                        self.buffer.append(values, now)
                        if self.recorder:
                            saved.append([self.sample_index, now, int(self.stimulus_on),
                                          self.attended_target_id, *values.tolist()])
                        self.sample_index += 1
                    if self.recorder and saved:
                        self.recorder.submit("raw", saved)
                report_count += len(rows)
                if now - last_report >= 1:
                    self.received_rate.emit(report_count / (now - last_report))
                    report_count, last_report = 0, now
        except Exception as exc:
            self.status.emit(f"采集失败: {exc}")
        finally:
            if self.handle is not None:
                try:
                    self.handle.close()
                except (OSError, serial.SerialException):
                    pass
            self.handle = None

    def stop(self):
        self.stop_event.set()
        self.wait(4500)


class DecodingThread(QtCore.QThread):
    result = QtCore.pyqtSignal(dict)

    def __init__(self, sample_buffer, targets, frame_timing, recorder,
                 window_seconds=3.0, parent=None, method="fbcca", gate_config=None):
        super().__init__(parent)
        self.buffer, self.targets = sample_buffer, targets
        self.timing, self.recorder = frame_timing, recorder
        self.window_seconds = window_seconds
        self.method = method
        self.gate_config = dict(RELAXED_GATE if gate_config is None else gate_config)
        self.started_at = None
        self.stop_event = threading.Event()

    def run(self):
        decoder_class = FBCCADecoder if self.method == "fbcca" else CCADecoder
        decoder = decoder_class([t.frequency for t in self.targets])
        gate = StableDecisionGate(**self.gate_config)
        while not self.stop_event.wait(0.5):
            data, stamps = self.buffer.snapshot(self.window_seconds)
            warning, _ = assess_window(data, stamps)
            reason = ""
            used = np.zeros(8, dtype=bool)
            try:
                prepared, used = prepare_decode_window(data)
            except ValueError as exc:
                reason = str(exc)
            if not reason and time.perf_counter()-stamps[-1] > .6:
                reason = "采集已中断：没有新数据"
            timing = self.timing.summary(self.window_seconds)
            if not reason and (self.started_at is None or stamps[0] < self.started_at + 0.25):
                reason = f"积累刺激后的 {self.window_seconds:.1f} 秒数据"
            if not reason and not timing["ok"]:
                reason = timing["reason"]
            payload = {"monotonic_s": time.perf_counter(), "reason": reason, "timing": timing,
                       "confirmed": None, "method": "FBCCA" if self.method == "fbcca" else "CCA",
                       "streak": 0, "required_streak": self.gate_config["consecutive"],
                       "mode": "eeg", "result_source": "eeg", "quality_warning": warning}
            if reason:
                gate.reset()
            else:
                try:
                    payload.update(decoder.decode(prepared))
                    payload["confirmed"] = gate.update(payload)
                    payload["used_channels"] = [c for c, ok in zip(CHANNEL_NAMES, used) if ok]
                except (ValueError, np.linalg.LinAlgError) as exc:
                    gate.reset()
                    payload["reason"] = f"解码失败: {exc}"
            self.recorder.submit("decision", payload)
            self.result.emit(payload)

    def stop(self):
        self.stop_event.set()
        self.wait(5000)
