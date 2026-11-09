"""Display-only analysis. Never supplies samples or labels to the SSVEP decoder."""
import threading
import time
import queue

import numpy as np
from scipy import signal
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError
from PyQt5 import QtCore

try:
    from .core import SAMPLE_RATE, assess_window
except ImportError:
    from core import SAMPLE_RATE, assess_window


# Schematic rear view, matching board 1..8. These are not digitized scalp coordinates.
POSTERIOR_POSITIONS = np.array([
    [-.78, -.20], [-.40, -.31], [0., -.35], [.40, -.31], [.78, -.20],
    [-.39, .38], [0., .46], [.39, .38],
])
MAP_EXTENT = (-1.05, 1.05, -.95, .95)


def power_in_band(frequencies, psd, low, high):
    """Integrate a density (uV²/Hz) on FFT bins using bin-edge overlap."""
    df = float(frequencies[1] - frequencies[0])
    widths = np.maximum(0, np.minimum(frequencies + df / 2, high)
                        - np.maximum(frequencies - df / 2, low))
    return widths @ psd


def posterior_field(values, good, resolution=144):
    """Linear interpolation only inside the measured, valid electrodes' convex hull."""
    x = np.linspace(*MAP_EXTENT[:2], resolution)
    y = np.linspace(*MAP_EXTENT[2:], resolution)
    xx, yy = np.meshgrid(x, y)
    values, good = np.asarray(values), np.asarray(good, dtype=bool)
    if np.count_nonzero(good) < 3:
        return np.full(xx.shape, np.nan)
    try:
        return LinearNDInterpolator(POSTERIOR_POSITIONS[good], values[good])(xx, yy)
    except QhullError:
        return np.full(xx.shape, np.nan)


class DisplayProcessor:
    def __init__(self, targets, fs=SAMPLE_RATE):
        self.fs = fs
        self.targets = targets
        self.sos = signal.butter(3, (1, 45), btype="bandpass", fs=fs, output="sos")
        self.spectral = {}
        self.last_spectrum = -float("inf")

    def process(self, samples, stamps, now=None):
        now = time.perf_counter() if now is None else now
        reason, good = assess_window(samples, stamps, self.fs, now)
        if samples is None:
            return {"available": False, "reason": reason}
        if samples.ndim != 2 or samples.shape[1] != 8 or not np.all(np.isfinite(samples)):
            return {"available": False, "reason": reason}
        centered = signal.detrend(samples, axis=0)
        filtered = signal.sosfiltfilt(self.sos, centered, axis=0)
        # Trim only the display filter's edge transients, without changing decoder input.
        trim = round(.15 * self.fs)
        filtered = filtered[trim:-trim]
        elapsed = stamps[-1] - stamps[0]
        rate = (len(stamps) - 1) / elapsed if elapsed > 0 else 0.
        # Quality and display coverage are separate: do not hide every plot when
        # fewer than three sensors pass the old quality screen.
        deviations = np.std(centered, axis=0)
        valid = (deviations >= .25) & (np.ptp(centered, axis=0) <= 2000)
        valid &= np.mean(np.abs(samples) >= 187499, axis=0) < .5
        plot_channels = deviations > 1e-6
        if now-stamps[-1] > .6:
            valid[:] = False
        if now - self.last_spectrum >= .5 or not self.spectral:
            frequencies, psd = signal.welch(filtered, fs=self.fs,
                                            nperseg=min(len(filtered), 2 * self.fs), axis=0)
            # Include every channel in waveform, but do not color invalid sensors.
            powers = [power_in_band(frequencies, psd, 6, 30)]
            powers += [power_in_band(frequencies, psd, t.frequency - 1, t.frequency + 1)
                       for t in self.targets]
            mask = (frequencies >= 2) & (frequencies <= 45)
            self.spectral = {"frequencies": frequencies[mask], "psd": psd[mask],
                             "band_db": 10 * np.log10(np.maximum(powers, 1e-12)),
                             "spectral_id": float(stamps[-1])}
            self.last_spectrum = now
        # 250 Hz waveform is adequate for the displayed 1..45 Hz passband.
        stride = 2
        waveform = filtered[::stride].astype(np.float32)
        times = (np.arange(len(filtered))[::stride] + trim - len(samples) + 1) / self.fs
        return dict(self.spectral, available=True, reason=reason, good=valid, plot_channels=plot_channels,
                    waveform=waveform, times=times, rate=rate,
                    rms=np.std(centered, axis=0), stale=now - stamps[-1] > .6,
                    received_at=float(stamps[-1]))


class DisplayAnalysisThread(QtCore.QThread):
    """A latest-frame mailbox prevents queued plot updates from delaying presentation."""
    def __init__(self, targets, parent=None):
        super().__init__(parent)
        self.targets = targets
        self.lock = threading.Lock()
        self.buffer = None
        self.generation = 0
        self.latest = None
        self.stop_event = threading.Event()

    def set_buffer(self, buffer):
        with self.lock:
            self.buffer = buffer
            self.generation += 1
            self.latest = None

    def take_latest(self):
        with self.lock:
            latest, self.latest = self.latest, None
            return latest

    def run(self):
        previous_generation = -1
        processor = None
        while not self.stop_event.wait(.1):
            with self.lock:
                buffer, generation = self.buffer, self.generation
            if buffer is None:
                continue
            if generation != previous_generation:
                processor = DisplayProcessor(self.targets)
                previous_generation = generation
            data, stamps = None, None
            for seconds in (4, 2, 1):
                data, stamps = buffer.snapshot(seconds)
                if data is not None:
                    break
            try:
                payload = processor.process(data, stamps)
            except (ValueError, FloatingPointError) as exc:
                payload = {"available": False, "reason": f"展示分析异常：{exc}"}
            with self.lock:
                if generation == self.generation:
                    self.latest = payload

    def stop(self):
        self.stop_event.set()
        self.wait(3000)


class RemoteDisplayAnalysisThread(DisplayAnalysisThread):
    """Consume the bounded feed in the audience process, keeping only the newest packet."""
    def __init__(self, targets, inbox):
        super().__init__(targets)
        self.inbox = inbox

    def run(self):
        processor = DisplayProcessor(self.targets)
        generation = -1
        packet = None
        while not self.stop_event.is_set():
            try:
                packet = self.inbox.get(timeout=.15)
                while True:
                    try:
                        packet = self.inbox.get_nowait()
                    except queue.Empty:
                        break
            except queue.Empty:
                if packet is None:
                    continue
            except (OSError, EOFError, ValueError):
                break
            state = packet["state"]
            if generation != state["generation"]:
                processor = DisplayProcessor(self.targets)
                generation = state["generation"]
            try:
                payload = processor.process(packet["samples"], packet["stamps"])
            except (ValueError, FloatingPointError) as exc:
                payload = {"available": False, "reason": f"展示分析异常：{exc}"}
            payload["bridge_state"] = state
            with self.lock:
                self.latest = payload
