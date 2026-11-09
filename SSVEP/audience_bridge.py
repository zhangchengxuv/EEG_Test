"""A bounded, one-way EEG feed isolates the presentation GUI from audience plots."""
import multiprocessing as mp
import queue
import threading
import time

from PyQt5 import QtCore


class AudienceBridge(QtCore.QObject):
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, targets, timing, parent=None):
        super().__init__(parent)
        self.targets, self.timing = targets, timing
        self.lock = threading.Lock()
        self.buffer = None
        self.state = {"generation": 0, "connected": False, "source": "serial", "endpoint": "",
                      "active": False, "finished": False, "result": None, "last_confirmed": None,
                      "show_sequence": 0, "mode": "eeg"}
        self.process = self.publisher = None
        self.inbox = self.commands = self.shutdown_event = None
        self.publisher_stop = threading.Event()
        self.viewer_active = False
        self.last_heartbeat = 0.
        self.error = ""
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll_commands)
        self.timer.start(100)

    def set_stream(self, buffer, source, endpoint):
        with self.lock:
            self.buffer = buffer
            self.state.update(generation=self.state["generation"]+1, connected=buffer is not None,
                              source=source, endpoint=endpoint, active=False, finished=False,
                              result=None, last_confirmed=None)

    def begin_trial(self):
        with self.lock:
            self.state.update(active=True, finished=False, result=None, last_confirmed=None)

    def set_mode(self, mode):
        with self.lock:
            self.state.update(mode=mode, result=None, last_confirmed=None, finished=False)

    def receive_result(self, result):
        with self.lock:
            self.state["result"] = result
            if result.get("confirmed") is not None:
                self.state["last_confirmed"] = result

    def end_trial(self):
        with self.lock:
            self.state.update(active=False, finished=True)

    def show_presentation(self):
        with self.lock:
            self.state["show_sequence"] += 1
            handle = self.parent().windowHandle() if self.parent() else None
            self.state["stimulus_screen"] = handle.screen().name() if handle else ""
        if self.process is not None and self.process.is_alive():
            return
        self._stop_process()
        context = mp.get_context("spawn")
        self.inbox = context.Queue(maxsize=2)
        self.commands = context.Queue(maxsize=16)
        self.shutdown_event = context.Event()
        self.process = context.Process(target=run_audience_process,
                                       args=(self.targets, self.inbox, self.commands, self.shutdown_event),
                                       name="SSVEP-audience", daemon=True)
        self.error = ""
        self.process.start()
        self.publisher_stop.clear()
        self.publisher = threading.Thread(target=self._publish, name="SSVEP-audience-feed", daemon=True)
        self.publisher.start()

    def _publish(self):
        while not self.publisher_stop.wait(.1):
            with self.lock:
                buffer, state = self.buffer, dict(self.state)
            data, stamps = None, None
            if buffer is not None:
                for seconds in (4, 2, 1):
                    data, stamps = buffer.snapshot(seconds)
                    if data is not None:
                        break
            state["frame_timing"] = self.timing.summary()
            packet = {"state": state, "samples": data, "stamps": stamps}
            try:
                self.inbox.put_nowait(packet)
            except queue.Full:
                pass  # Never block the acquisition/presentation if the viewer is slow.
            except (OSError, ValueError):
                break

    def _poll_commands(self):
        if self.commands is None:
            return
        while True:
            try:
                command = self.commands.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break
            if command["kind"] == "stop":
                self.stop_requested.emit()
            elif command["kind"] == "heartbeat":
                self.viewer_active = command["active"]
                self.last_heartbeat = time.perf_counter()
            elif command["kind"] == "error":
                self.error = command["message"]
        if self.process is not None and not self.process.is_alive():
            self.viewer_active = False
            self.error = self.error or "展示进程已结束，点击“展示大屏”可重新打开"

    def is_active(self):
        return self.viewer_active and time.perf_counter()-self.last_heartbeat < .5

    def _stop_process(self):
        self.publisher_stop.set()
        if self.publisher is not None:
            self.publisher.join(timeout=2)
            self.publisher = None
        if self.shutdown_event is not None:
            self.shutdown_event.set()
        if self.process is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()  # Only the helper created by this bridge.
                self.process.join(timeout=1)
            self.process.close()
            self.process = None
        for channel in (self.inbox, self.commands):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self.inbox = self.commands = self.shutdown_event = None
        self.viewer_active = False

    def shutdown(self):
        self.timer.stop()
        self._stop_process()


def run_audience_process(targets, inbox, commands, shutdown_event):
    # Import GUI and NumPy analysis here: the participant process never paints charts.
    import sys
    from PyQt5 import QtWidgets
    try:
        from .audience import AudienceWindow
        from .display_data import RemoteDisplayAnalysisThread
    except ImportError:
        from audience import AudienceWindow
        from display_data import RemoteDisplayAnalysisThread

    def send(kind, **kwargs):
        try:
            commands.put_nowait(dict(kind=kind, **kwargs))
        except queue.Full:
            pass

    def exception_hook(exc_type, exception, tb):
        send("error", message=f"展示界面异常：{exception}")
        sys.__excepthook__(exc_type, exception, tb)

    sys.excepthook = exception_hook
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    analysis = RemoteDisplayAnalysisThread(targets, inbox)
    window = AudienceWindow(targets, None, analysis=analysis)
    window.stop_requested.connect(lambda: send("stop"))
    # The first packet chooses the audience monitor and restores the window.
    heartbeat = QtCore.QTimer()

    def poll():
        if shutdown_event.is_set():
            app.quit()
            return
        send("heartbeat", active=bool(window.isVisible() and app.applicationState() == QtCore.Qt.ApplicationActive))

    heartbeat.timeout.connect(poll)
    heartbeat.start(100)
    try:
        app.exec_()
    finally:
        window.shutdown()
        commands.cancel_join_thread()
