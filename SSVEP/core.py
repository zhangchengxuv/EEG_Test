"""Pure signal processing; amplitudes are microvolts, arrays are (samples, channels)."""
from collections import deque
from dataclasses import dataclass
from itertools import combinations
import threading
import time

import numpy as np
from scipy import signal

CHANNEL_NAMES = ("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")
CHARACTERS = ("左肩", "右肩", "左肘", "右肘")
NOMINAL_FREQUENCIES = (8.5, 12.0, 15.0, 20.0)
WIDE_FREQUENCIES = (8.0, 13.0, 18.0, 23.0)
SAMPLE_RATE = 500
RELAXED_GATE = {"minimum_score": .22, "minimum_margin": .015, "consecutive": 2,
                "minimum_snr_db": None, "smoothing": .6}


@dataclass(frozen=True)
class Target:
    character: str
    frequency: float
    frames_per_cycle: float


def make_targets(refresh_hz, profile="integer"):
    if not np.isfinite(refresh_hz) or not 40 <= refresh_hz <= 360:
        raise ValueError("显示器刷新率须在 40–360 Hz 之间")
    if profile == "wide":
        if max(WIDE_FREQUENCIES) >= refresh_hz / 2:
            raise ValueError("宽频方案需要软件更新率高于 46 帧/秒")
        return tuple(Target(c, f, refresh_hz / f) for c, f in zip(CHARACTERS, WIDE_FREQUENCIES))
    if profile != "integer":
        raise ValueError("未知频率方案")
    # Search whole-frame periods near the intended range. Rounding alone can turn
    # the low/high targets into an octave pair, e.g. 9 and 18 Hz at 72 fps.
    periods = range(max(3, int(np.ceil(refresh_hz / 22))), int(refresh_hz / 7.5) + 1)
    candidates = []
    for ascending in combinations(periods, len(CHARACTERS)):
        if any(high % low == 0 for low, high in combinations(ascending, 2)):
            continue  # No target fundamental is another target's integer harmonic.
        trial_cycles = tuple(reversed(ascending))
        frequencies = np.asarray([refresh_hz / n for n in trial_cycles])
        if np.min(np.diff(frequencies)) < 2.0:
            continue
        cost = float(np.sum((frequencies - NOMINAL_FREQUENCIES) ** 2))
        candidates.append((cost, trial_cycles))
    if not candidates:
        raise ValueError("当前呈现率无法生成四个间隔足够且无整数倍频的目标，请使用至少 60 Hz 的显示器")
    _, cycles = min(candidates)
    return tuple(Target(c, refresh_hz / n, n) for c, n in zip(CHARACTERS, cycles))


def luminance(frame, frames_per_cycle):
    return 0.5 + 0.45 * np.sin(2 * np.pi * frame / frames_per_cycle)


def parse_line(line):
    """Accept only the board's eight finite V values, converted exactly once to µV."""
    if isinstance(line, (bytes, bytearray)):
        line = line.decode("ascii", errors="replace")
    prefix, sep, body = line.strip().partition(":")
    if not sep or prefix.lower() != "channel":
        return None
    parts = body.split(",")
    if len(parts) != 8:
        return None
    try:
        values = np.asarray([float(x) for x in parts], dtype=np.float64)
    except ValueError:
        return None
    return values * 1e6 if np.all(np.isfinite(values)) else None


class SampleBuffer:
    def __init__(self, fs=SAMPLE_RATE, seconds=15):
        self.fs = fs
        self.samples = deque(maxlen=int(fs * seconds))
        self.times = deque(maxlen=int(fs * seconds))
        self.lock = threading.Lock()

    def append(self, values, received_time=None):
        stamp = time.perf_counter() if received_time is None else received_time
        with self.lock:
            self.samples.append(np.asarray(values, dtype=np.float64).copy())
            self.times.append(float(stamp))

    def snapshot(self, seconds):
        n = round(seconds * self.fs)
        with self.lock:
            if len(self.samples) < n:
                return None, None
            return np.asarray(list(self.samples)[-n:]), np.asarray(list(self.times)[-n:])


def prepare_decode_window(samples):
    """Keep usable variation without a minimum good-channel count; limit impulsive artifacts."""
    if samples is None or len(samples) < SAMPLE_RATE:
        raise ValueError("等待足够的脑电数据")
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != 8 or not np.all(np.isfinite(x)):
        raise ValueError("脑电数据格式或数值异常")
    varying = np.std(x, axis=0) > 1e-6
    non_railed = np.mean(np.abs(x) >= 187499, axis=0) < .5
    used = varying & non_railed
    if not np.any(used):
        # No fabricated EEG evidence: the decoder will return zero scores.
        return np.zeros((len(x), 1)), used
    x = x[:, used]
    center = np.median(x, axis=0)
    scale = 1.4826*np.median(np.abs(x-center), axis=0)
    # A single large transient must not inflate an otherwise meaningful MAD.
    scale = np.where(scale > 1e-6, scale, np.maximum(np.std(x, axis=0)*.05, 1e-6))
    x = np.clip(x-center, -6*scale, 6*scale)
    return x, used


class MouseTargetLatch:
    """A mouse choice is a separate result source, never evidence for EEG decoding."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.selected_at = None

    def observe(self, target, now=None):
        if target is None or target == self.target:
            return False
        if target not in range(4):
            raise ValueError("鼠标目标必须是 0–3")
        self.target = target
        self.selected_at = time.perf_counter() if now is None else now
        return True

    def result(self, targets, now=None):
        payload = {"monotonic_s": time.perf_counter() if now is None else now,
                   "mode": "test", "mode_name": "测试模式", "result_source": "mouse",
                   "method": "MOUSE", "confirmed": self.target, "best": self.target,
                   "mouse_selected_monotonic_s": self.selected_at,
                   "reason": "等待鼠标进入任一四分区" if self.target is None else "鼠标选择保持至进入其他四分区"}
        if self.target is not None:
            payload.update(target_name=targets[self.target].character,
                           target_frequency_hz=targets[self.target].frequency)
        return payload


def assess_window(samples, timestamps, fs=SAMPLE_RATE, now=None):
    """Reject bad transport/flat data before CCA; arrival rate is not hardware sample timing."""
    if samples is None or len(samples) < fs:
        return "等待足够的脑电数据", None
    if samples.ndim != 2 or samples.shape[1] != 8 or not np.all(np.isfinite(samples)):
        return "脑电数据格式或数值异常", None
    stamp = time.perf_counter() if now is None else now
    if stamp - timestamps[-1] > 0.6:
        return "采集已中断：没有新数据", None
    elapsed = timestamps[-1] - timestamps[0]
    rate = (len(samples) - 1) / elapsed if elapsed > 0 else 0
    if not 0.88 * fs <= rate <= 1.12 * fs:
        return f"接收速率异常 {rate:.0f}/s，期望 {fs}/s", None
    centered = signal.detrend(samples, axis=0)
    rms = np.std(centered, axis=0)
    peak_to_peak = np.ptp(centered, axis=0)
    good = (rms >= 0.25) & (peak_to_peak <= 2000)
    if np.max(np.abs(samples)) < 0.01:
        return "八通道全零：请先检查 ADS1299 通信与供电", None
    if np.count_nonzero(good) < 3:
        return "有效通道不足 3 路：信号过平或幅度过大", None
    return "", good


def _basis(matrix):
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or singular[0] < 1e-12:
        return u[:, :0]
    return u[:, singular > singular[0] * 1e-8]


class CCADecoder:
    def __init__(self, frequencies, fs=SAMPLE_RATE, harmonics=3):
        self.fs = fs
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.harmonics = harmonics
        self.sos = signal.butter(4, (6, 70), btype="bandpass", fs=fs, output="sos")
        self.notch = signal.iirnotch(50, 30, fs=fs)
        self.cached_references = {}

    def decode(self, samples, good_channels=None):
        # Preserve SRB1 hardware reference; no average reference or automatic Oz/EOG ICA.
        if good_channels is not None:
            samples = samples[:, good_channels]
        filtered = signal.sosfiltfilt(self.sos, signal.detrend(samples, axis=0), axis=0)
        filtered = signal.filtfilt(*self.notch, filtered, axis=0)
        # Remove filter transients equally from EEG and references.
        trim = round(0.15 * self.fs)
        filtered = filtered[trim:-trim]
        qx = _basis(filtered)
        n = len(filtered)
        if n not in self.cached_references:
            t = np.arange(n) / self.fs
            self.cached_references[n] = [
                _basis(np.column_stack([
                    wave(2 * np.pi * f * h * t)
                    for h in range(1, self.harmonics + 1)
                    for wave in (np.sin, np.cos)
                ])) for f in self.frequencies
            ]
        scores = np.zeros(len(self.frequencies))
        if qx.shape[1]:
            for i, qref in enumerate(self.cached_references[n]):
                scores[i] = np.linalg.svd(qx.T @ qref, compute_uv=False)[0]
        freq, psd = signal.welch(filtered, fs=self.fs, nperseg=min(n, 2 * self.fs), axis=0)
        mask = (freq >= 6) & (freq <= 30)
        spectrum = np.mean(psd, axis=1)
        peak = float(freq[mask][np.argmax(spectrum[mask])])
        order = np.argsort(scores)
        best = int(order[-1])
        return {
            "method": "CCA",
            "scores": scores.tolist(), "best": best, "score": float(scores[best]),
            "margin": float(scores[best] - scores[order[-2]]), "spectral_peak_hz": peak,
            "spectrum_hz": freq[mask].tolist(), "spectrum_power": spectrum[mask].tolist(),
        }


class FBCCADecoder(CCADecoder):
    """Training-free filter-bank CCA, with normalized weighted squared correlations.

    This demo uses Butterworth sub-bands and the n^-1.25 + .25 weighting law.
    It is an adaptation, not a reproduction of the Chen et al. (2015) experiment.
    """
    def __init__(self, frequencies, fs=SAMPLE_RATE, harmonics=3):
        super().__init__(frequencies, fs, harmonics)
        self.filter_bands = tuple((low, 90) for low in (6, 14, 22, 30, 38))
        self.bank = [signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
                     for band in self.filter_bands]
        self.weights = np.arange(1, len(self.bank)+1, dtype=float)**-1.25 + .25
        self.weights /= self.weights.sum()

    def decode(self, samples, good_channels=None):
        if good_channels is not None:
            samples = samples[:, good_channels]
        samples = signal.detrend(samples, axis=0)
        samples = signal.filtfilt(*self.notch, samples, axis=0)
        trim = round(.15*self.fs)
        n = len(samples)-2*trim
        if n not in self.cached_references:
            t = np.arange(n)/self.fs
            self.cached_references[n] = [
                _basis(np.column_stack([wave(2*np.pi*f*h*t)
                                       for h in range(1, self.harmonics+1)
                                       for wave in (np.sin, np.cos)]))
                for f in self.frequencies
            ]
        correlations = np.zeros((len(self.bank), len(self.frequencies)))
        first_band = None
        for b, sos in enumerate(self.bank):
            filtered = signal.sosfiltfilt(sos, samples, axis=0)[trim:-trim]
            if b == 0:
                first_band = filtered
            qx = _basis(filtered)
            if qx.shape[1]:
                for i, qref in enumerate(self.cached_references[n]):
                    correlations[b, i] = np.clip(np.linalg.svd(qx.T @ qref, compute_uv=False)[0], 0, 1)
        scores = np.sqrt(self.weights @ correlations**2)
        freq, psd = signal.welch(first_band, fs=self.fs, nperseg=min(n, 2*self.fs), axis=0)
        spectrum = np.mean(psd, axis=1)
        snrs = []
        df = freq[1]-freq[0]
        for f in self.frequencies:
            distance = np.abs(freq-f)
            central = distance <= max(.55, df)
            surrounding = (distance >= 1.5) & (distance <= 3.5)
            for other in self.frequencies:
                if other != f:
                    surrounding &= np.abs(freq-other) > .75
            floor = max(float(np.mean(spectrum[surrounding])), 1e-12)
            power = max(float(np.mean(spectrum[central])), 1e-12)
            snrs.append(float(10*np.log10(power/floor)))
        mask = (freq >= 6) & (freq <= 30)
        order = np.argsort(scores)
        best = int(order[-1])
        return {"method": "FBCCA", "scores": scores.tolist(), "best": best,
                "score": float(scores[best]), "margin": float(scores[best]-scores[order[-2]]),
                "subband_scores": correlations.tolist(), "snr_db": snrs,
                "spectral_peak_hz": float(freq[mask][np.argmax(spectrum[mask])]),
                "spectrum_hz": freq[mask].tolist(), "spectrum_power": spectrum[mask].tolist()}


class StableDecisionGate:
    """Smooth evidence, report why a window is rejected, and never hold a stale command."""
    def __init__(self, minimum_score=.35, minimum_margin=.06, consecutive=3,
                 minimum_snr_db=3., smoothing=.45):
        self.minimum_score, self.minimum_margin = minimum_score, minimum_margin
        self.consecutive, self.minimum_snr_db = consecutive, minimum_snr_db
        self.smoothing = smoothing
        self.reset()

    def reset(self):
        self.smoothed = None
        self.previous = None
        self.streak = 0

    def update(self, result):
        raw = np.asarray(result["scores"], dtype=float)
        if self.smoothed is None:
            self.smoothed = raw.copy()
        else:
            self.smoothed = self.smoothing*raw + (1-self.smoothing)*self.smoothed
        order = np.argsort(self.smoothed)
        best = int(order[-1])
        score = float(self.smoothed[best])
        margin = float(score-self.smoothed[order[-2]])
        raw_margin = float(raw[best]-np.max(np.delete(raw, best)))
        result.update(raw_scores=raw.tolist(), scores=self.smoothed.tolist(), best=best,
                      score=score, margin=margin, confirmed=None, required_streak=self.consecutive)
        # Check current evidence as well as smoothed evidence when attention changes.
        if int(np.argmax(raw)) != best:
            reason = "候选正在变化，等待新目标稳定"
        elif score < self.minimum_score or raw[best] < self.minimum_score:
            reason = f"分数不足 {score:.3f} / {self.minimum_score:.3f}"
        elif margin < self.minimum_margin or raw_margin < self.minimum_margin*.5:
            reason = f"候选分差不足 {margin:.3f} / {self.minimum_margin:.3f}"
        elif self.minimum_snr_db is not None and "snr_db" in result and result["snr_db"][best] < self.minimum_snr_db:
            reason = f"目标频率信噪比不足 {result['snr_db'][best]:.1f} dB / {self.minimum_snr_db:.0f} dB"
        else:
            self.streak = self.streak+1 if self.previous == best else 1
            self.previous = best
            result["streak"] = min(self.streak, self.consecutive)
            if self.streak >= self.consecutive:
                result.update(confirmed=best, reason="")
                return best
            result["reason"] = f"连续确认 {self.streak}/{self.consecutive}"
            return None
        self.previous, self.streak = None, 0
        result.update(streak=0, reason=reason)
        return None


class DecisionGate:
    """Heuristic rejection and repeatability gate, not a calibrated attention detector."""
    def __init__(self, minimum_score=0.35, minimum_margin=0.08, consecutive=3):
        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin
        self.consecutive = consecutive
        self.reset()

    def reset(self):
        self.previous = None
        self.streak = 0

    def update(self, result):
        if result["score"] < self.minimum_score or result["margin"] < self.minimum_margin:
            self.reset()
            return None
        best = result["best"]
        self.streak = self.streak + 1 if best == self.previous else 1
        self.previous = best
        return best if self.streak >= self.consecutive else None


class FrameTiming:
    def __init__(self, refresh_hz):
        self.refresh_hz = refresh_hz
        self.times = deque(maxlen=4000)
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.times.clear()

    def record(self, stamp):
        with self.lock:
            self.times.append(float(stamp))

    def summary(self, seconds=3, now=None):
        with self.lock:
            stamps = np.asarray(self.times)
        if len(stamps) < 30:
            return {"ok": False, "fps": 0.0, "late_fraction": 1.0,
                    "max_interval_ms": 0.0, "reason": "等待画面时序稳定"}
        stamps = stamps[stamps >= stamps[-1] - seconds]
        if len(stamps) < 2:
            return {"ok": False, "fps": 0.0, "late_fraction": 1.0,
                    "max_interval_ms": 0.0, "reason": "等待画面时序稳定"}
        intervals = np.diff(stamps)
        fps = float(1 / intervals.mean()) if len(intervals) and intervals.mean() > 0 else 0.0
        late = float(np.mean(intervals > 1.5 / self.refresh_hz))
        maximum_interval = float(np.max(intervals)) if len(intervals) else float("inf")
        current = time.perf_counter() if now is None else now
        maximum_ms = maximum_interval * 1000
        if current - stamps[-1] > 0.3:
            reason = "画面更新已中断"
        elif maximum_interval > 3 / self.refresh_hz:
            reason = f"画面出现 {maximum_ms:.0f} ms 停顿，等待稳定数据窗"
        elif abs(fps / self.refresh_hz - 1) > 0.07:
            reason = f"画面更新 {fps:.1f} 帧/秒，目标 {self.refresh_hz:.1f}"
        elif late > 0.05:
            reason = f"画面长帧比例 {late:.1%}，等待稳定数据窗"
        else:
            reason = ""
        return {"ok": not reason, "fps": fps, "late_fraction": late,
                "max_interval_ms": maximum_ms, "reason": reason}


def simulated_samples(indices, frequency, fs=SAMPLE_RATE, seed=20261015):
    """Independent eight-channel synthetic source, explicitly not a human recording."""
    t = np.asarray(indices) / fs
    rng = np.random.default_rng(seed + int(indices[0]))
    phases = np.arange(8) * 0.18
    fundamental = 10 * np.sin(2 * np.pi * frequency * t[:, None] + phases)
    harmonic = 4 * np.sin(4 * np.pi * frequency * t[:, None] + 0.5 * phases)
    return fundamental + harmonic + rng.normal(0, 6, (len(t), 8))
