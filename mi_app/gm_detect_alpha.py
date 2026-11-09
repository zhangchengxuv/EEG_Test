
import os
# 避免 Windows + MKL 某些情况下的内存问题
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
# import mne
from scipy import signal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# ===================== 一些通用函数 =====================

def bandpass_filter(data, fs, lowcut=1.0, highcut=40.0, order=4):
    """离线零相位带通滤波（用于标定阶段）"""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data)

def moving_average(x, k=5):
    """长度不变的滑动平均"""
    if k <= 1:
        return x.copy()
    assert k % 2 == 1, "平滑窗口 k 必须是奇数"
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode='same')

def compute_band_powers(eeg, fs,
                        win_length=2.0,
                        step=0.5,
                        alpha_band=(8.0, 12.0),
                        low_band=(1.0, 4.0),
                        total_band=(1.0, 40.0)):
    """离线：对一段 eeg 计算滑窗的 alpha/低频/总功率（相对值）"""
    win_samples = int(win_length * fs)
    step_samples = int(step * fs)

    n_samples = len(eeg)
    alpha_rel_list = []
    low_rel_list = []
    times = []

    for start in range(0, n_samples - win_samples, step_samples):
        end = start + win_samples
        segment = eeg[start:end]

        f, psd = signal.welch(
            segment,
            fs=fs,
            nperseg=win_samples // 2,
            noverlap=win_samples // 4
        )

        alpha_idx = (f >= alpha_band[0]) & (f <= alpha_band[1])
        low_idx   = (f >= low_band[0])   & (f <= low_band[1])
        total_idx = (f >= total_band[0]) & (f <= total_band[1])

        alpha_power = np.trapz(psd[alpha_idx], f[alpha_idx])
        low_power   = np.trapz(psd[low_idx],   f[low_idx])
        total_power = np.trapz(psd[total_idx], f[total_idx])

        if total_power <= 0:
            alpha_rel = 0.0
            low_rel   = 0.0
        else:
            alpha_rel = alpha_power / total_power
            low_rel   = low_power   / total_power

        alpha_rel_list.append(alpha_rel)
        low_rel_list.append(low_rel)

        center_time = (start + end) / 2 / fs
        times.append(center_time)

    return np.array(times), np.array(alpha_rel_list), np.array(low_rel_list)


# ===================== 实时检测类（用标定好的阈值） =====================

class RealTimeAlphaDetectorRT:
    def __init__(self, fs,
                 win_length=2.0,
                 step=0.5,
                 bandpass=(1.0, 40.0),
                 alpha_band=(8.0, 12.0),
                 low_band=(1.0, 4.0),
                 total_band=(1.0, 40.0),
                 low_thr=None,
                 alpha_thr=None):
        self.fs = fs
        self.win_length = win_length
        self.step = step
        self.win_samples = int(win_length * fs)
        self.step_samples = int(step * fs)

        self.bandpass = bandpass
        self.alpha_band = alpha_band
        self.low_band = low_band
        self.total_band = total_band

        # IIR 带通滤波器（实时）
        nyq = fs / 2.0
        low = bandpass[0] / nyq
        high = bandpass[1] / nyq
        self.sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
        self.zi = signal.sosfilt_zi(self.sos)

        self.buffer = np.zeros(0, dtype=float)
        self._samples_since_last_step = 0

        # 标定得到的阈值
        self.low_thr = low_thr
        self.alpha_thr = alpha_thr

    def _update_filter(self, x_block):
        y, self.zi = signal.sosfilt(self.sos, x_block, zi=self.zi)
        return y

    def _compute_features_from_window(self, segment):
        fs = self.fs
        win_samples = len(segment)

        f, psd = signal.welch(
            segment,
            fs=fs,
            nperseg=win_samples // 2,
            noverlap=win_samples // 4
        )

        alpha_idx = (f >= self.alpha_band[0]) & (f <= self.alpha_band[1])
        low_idx   = (f >= self.low_band[0])   & (f <= self.low_band[1])
        total_idx = (f >= self.total_band[0]) & (f <= self.total_band[1])

        alpha_power = np.trapz(psd[alpha_idx], f[alpha_idx])
        low_power   = np.trapz(psd[low_idx],   f[low_idx])
        total_power = np.trapz(psd[total_idx], f[total_idx])

        if total_power <= 0:
            return 0.0, 0.0

        alpha_rel = alpha_power / total_power
        low_rel   = low_power   / total_power
        return alpha_rel, low_rel

    def _classify(self, alpha_rel, low_rel):
        # 先看伪迹
        if low_rel > self.low_thr:
            return "artifact"
        # 再用 alpha 阈值判 open / closed
        if alpha_rel > self.alpha_thr:
            return "closed"
        else:
            return "open"

    def add_block(self, x_block):
        """实时接口"""
        y_block = self._update_filter(x_block)
        self.buffer = np.concatenate([self.buffer, y_block])
        self._samples_since_last_step += len(y_block)

        results = []

        while (self._samples_since_last_step >= self.step_samples and
               len(self.buffer) >= self.win_samples):
            segment = self.buffer[-self.win_samples:]
            alpha_rel, low_rel = self._compute_features_from_window(segment)
            current_time = (len(self.buffer) - self.win_samples / 2) / self.fs

            state = self._classify(alpha_rel, low_rel)

            results.append({
                "t": current_time,
                "alpha_rel": alpha_rel,
                "low_rel": low_rel,
                "state": state,
            })

            self._samples_since_last_step -= self.step_samples

        return results
        


# class EEGFileRead:
#     def __init__(self):
#         pass


#     def open_file(self,filePathName="D:\\learning_practice\\python_dsp\\detect_alpha\\12022333_.edf"):
#         raw = mne.io.read_raw_edf(filePathName, preload=True)
#         fs = raw.info["sfreq"]

        # raw.notch_filter(freqs=50)
        # raw.filter(l_freq=.3, h_freq=40,method='iir') # 5, 10, 20uv use this
        # raw.compute_psd().plot()
        # raw.plot(duration=5,show=True)
        # events_from_annot,event_dict = mne.events_from_annotations(raw)
        # fig = mne.viz.plot_events(events_from_annot,event_id=event_dict,sfreq=raw.info['sfreq'],first_samp=raw.first_samp)



