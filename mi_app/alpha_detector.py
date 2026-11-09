#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
alpha_detector.py

Real-time EEG alpha detector
- load calibration parameters (.npz)
- process EEG stream block by block
- output state: open / closed / artifact
"""

# ===================== 必须最先设置（防止 Windows + MKL 崩溃） =====================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ===================== 标准库 =====================
import argparse
import sys

# ===================== 第三方库 =====================
import numpy as np
# import mne
from scipy import signal


# ===================== 核心实时检测类 =====================

class AlphaRealtimeDetector:
    """
    Real-time alpha detector

    Usage:
        detector = AlphaRealtimeDetector.from_npz(params)
        results = detector.process_block(eeg_block)
    """

    def __init__(
        self,
        fs,
        bandpass,
        alpha_band,
        low_band,
        win_length,
        step,
        alpha_thr,
        low_thr,
    ):
        self.fs = float(fs)
        self.bandpass = tuple(bandpass)
        self.alpha_band = tuple(alpha_band)
        self.low_band = tuple(low_band)
        self.win_length = float(win_length)
        self.step = float(step)
        self.alpha_thr = float(alpha_thr)
        self.low_thr = float(low_thr)

        # window sizes
        self.win_samples = int(self.win_length * self.fs)
        self.step_samples = int(self.step * self.fs)

        # IIR bandpass filter (causal)
        nyq = self.fs / 2.0
        self.sos = signal.butter(
            4,
            [self.bandpass[0] / nyq, self.bandpass[1] / nyq],
            btype="bandpass",
            output="sos",
        )
        self.zi = signal.sosfilt_zi(self.sos)

        # buffers
        self.buffer = np.zeros(0, dtype=float)
        self._since_last = 0

    # ---------- factory ----------

    @classmethod
    def from_npz(cls, npz):
        """Create detector from calibration npz"""
        return cls(
            fs=npz["fs"],
            bandpass=npz["bandpass"],
            alpha_band=npz["alpha_band"],
            low_band=npz["low_band"],
            win_length=npz["win_length"],
            step=npz["step"],
            alpha_thr=npz["alpha_thr"],
            low_thr=npz["low_thr"],
        )

    # ---------- internal ----------

    @staticmethod
    def _band_power(f, psd, band):
        idx = (f >= band[0]) & (f <= band[1])
        return np.trapz(psd[idx], f[idx])

    def _extract_features(self, segment):
        f, psd = signal.welch(
            segment,
            fs=self.fs,
            nperseg=len(segment) // 2,
            noverlap=len(segment) // 4,
        )

        p_alpha = self._band_power(f, psd, self.alpha_band)
        p_low = self._band_power(f, psd, self.low_band)
        p_total = self._band_power(f, psd, self.bandpass)

        if p_total <= 0:
            return 0.0, 0.0

        return p_alpha / p_total, p_low / p_total

    def _classify(self, alpha_rel, low_rel):
        if low_rel > self.low_thr:
            return "artifact"
        elif alpha_rel > self.alpha_thr:
            return "closed"
        else:
            return "open"

    # ---------- public API ----------

    def process_block(self, x):
        """
        Process a block of EEG samples.

        Parameters
        ----------
        x : 1D numpy array

        Returns
        -------
        results : list of dict
            Each dict contains:
                {
                    "alpha_rel": float,
                    "low_rel": float,
                    "state": "open"/"closed"/"artifact"
                }
        """
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        self.buffer = np.concatenate([self.buffer, y])
        self._since_last += len(y)

        results = []

        while (
            self._since_last >= self.step_samples
            and len(self.buffer) >= self.win_samples
        ):
            seg = self.buffer[-self.win_samples:]
            alpha_rel, low_rel = self._extract_features(seg)
            state = self._classify(alpha_rel, low_rel)

            results.append({
                "alpha_rel": alpha_rel,
                "low_rel": low_rel,
                "state": state,
            })

            self._since_last -= self.step_samples

        return results


# ===================== CLI（用于 EDF 回放 / 调试） =====================

def run_cli(args):
    print("=== Alpha detector (CLI mode) ===")
    print("EDF:", args.edf)
    print("Params:", args.params)

    # params = np.load(args.params, allow_pickle=True)
    # detector = AlphaRealtimeDetector.from_npz(params)

    # raw = mne.io.read_raw_edf(args.edf, preload=True, verbose="ERROR")
    # fs = raw.info["sfreq"]

    # if abs(fs - detector.fs) > 1e-6:
    #     raise RuntimeError("Sampling rate mismatch between EDF and calibration params")

    # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    # if len(picks) == 0:
    #     raise RuntimeError("No EEG channel found in EDF")

    # eeg = raw.get_data(picks=[picks[0]])[0]

    # block_size = args.block_size
    # counts = {"open": 0, "closed": 0, "artifact": 0}

    # for i in range(0, len(eeg), block_size):
    #     block = eeg[i:i + block_size]
    #     if len(block) == 0:
    #         break

    #     results = detector.process_block(block)
    #     for r in results:
    #         counts[r["state"]] += 1
    #         print(f"{r['state']:>9} | alpha={r['alpha_rel']:.4f}")

    # print("\nSummary:")
    # for k, v in counts.items():
    #     print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time EEG alpha detector (CLI / module)"
    )
    parser.add_argument("--edf", help="EDF file for offline replay")
    parser.add_argument("--params", required=True,
                        help="Calibration parameter file (.npz)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Block size for simulated real-time processing")

    args = parser.parse_args()

    if args.edf is None:
        print("No --edf given, CLI mode skipped.")
        return

    try:
        run_cli(args)
    except Exception as e:
        print("Detector failed:", str(e), file=sys.stderr)
        sys.exit(1)

class AlphaStateMachine:
    """
    Debounce / hysteresis state machine for alpha detection.
    Input each step: (alpha_rel, low_rel, raw_state) -> stable_state
    """

    def __init__(
        self,
        step_sec: float,
        alpha_thr: float,
        low_thr: float,
        # hysteresis
        hysteresis_ratio: float = 0.05,   # 5%
        # debounce times
        close_min_sec: float = 1.5,
        open_min_sec: float = 1.0,
        # artifact handling
        artifact_grace_sec: float = 2.0,  # 连续伪迹超过这个时间才进入 ARTIFACT
        keep_last_on_artifact: bool = True,
    ):
        self.step_sec = float(step_sec)
        self.low_thr = float(low_thr)

        # hysteresis thresholds
        hr = float(hysteresis_ratio)
        self.alpha_thr_on = float(alpha_thr) * (1.0 + hr)
        self.alpha_thr_off = float(alpha_thr) * (1.0 - hr)

        # convert seconds to counts
        self.close_min_n = max(1, int(round(close_min_sec / self.step_sec)))
        self.open_min_n = max(1, int(round(open_min_sec / self.step_sec)))
        self.artifact_grace_n = max(1, int(round(artifact_grace_sec / self.step_sec)))

        self.keep_last_on_artifact = bool(keep_last_on_artifact)

        # internal state
        self.state = "open"          # stable output state: open/closed/artifact
        self._close_streak = 0
        self._open_streak = 0
        self._artifact_streak = 0

    def reset(self):
        self.state = "open"
        self._close_streak = 0
        self._open_streak = 0
        self._artifact_streak = 0

    def update(self, alpha_rel: float, low_rel: float):
        """
        Update with new features (per STEP).
        Returns stable state: "open" / "closed" / "artifact"
        """
        # 1) artifact detection (low frequency)
        is_artifact = (low_rel > self.low_thr)

        if is_artifact:
            self._artifact_streak += 1
            self._close_streak = 0
            self._open_streak = 0

            # 连续伪迹太久，才进入 artifact 状态（可选）
            if self._artifact_streak >= self.artifact_grace_n:
                self.state = "artifact"
                return self.state

            # grace 期间：不改变 state（更稳）
            return self.state if self.keep_last_on_artifact else "artifact"

        # 非伪迹：清空 artifact streak
        self._artifact_streak = 0
        if self.state == "artifact":
            # 从 artifact 恢复时，不立刻切换，让 streak 来决定
            self.state = "open"

        # 2) hysteresis logic: decide candidate open/closed based on current stable state
        if self.state == "open":
            # open -> closed 需要 alpha_rel 超过 alpha_thr_on 并持续
            if alpha_rel >= self.alpha_thr_on:
                self._close_streak += 1
                self._open_streak = 0
                if self._close_streak >= self.close_min_n:
                    self.state = "closed"
                    self._close_streak = 0
            else:
                self._close_streak = 0

        elif self.state == "closed":
            # closed -> open 需要 alpha_rel 低于 alpha_thr_off 并持续
            if alpha_rel <= self.alpha_thr_off:
                self._open_streak += 1
                self._close_streak = 0
                if self._open_streak >= self.open_min_n:
                    self.state = "open"
                    self._open_streak = 0
            else:
                self._open_streak = 0

        return self.state



if __name__ == "__main__":
    main()
