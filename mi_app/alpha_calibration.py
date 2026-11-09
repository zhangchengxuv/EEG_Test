#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration CLI tool for EEG alpha detection
"""

# ===================== 必须最先设置（防止 Windows + MKL 崩溃） =====================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")   # 禁止 GUI 后端，避免 Qt 冲突

# ===================== 标准库 =====================
import argparse
import sys

# ===================== 第三方库 =====================
import numpy as np
import mne
from scipy import signal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# ===================== 工具函数 =====================

def bandpass_filter(data, fs, band=(1.0, 40.0), order=4):
    nyq = fs / 2.0
    sos = signal.butter(
        order,
        [band[0] / nyq, band[1] / nyq],
        btype="bandpass",
        output="sos"
    )
    return signal.sosfiltfilt(sos, data)


def moving_average(x, k):
    if k <= 1:
        return x.copy()
    assert k % 2 == 1, "smooth_k must be odd"
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def compute_relative_powers(
    eeg,
    fs,
    win_length,
    step,
    alpha_band,
    low_band,
    total_band,
):
    win_samples = int(win_length * fs)
    step_samples = int(step * fs)

    alpha_rel = []
    low_rel = []
    times = []

    for start in range(0, len(eeg) - win_samples, step_samples):
        seg = eeg[start:start + win_samples]

        f, psd = signal.welch(
            seg,
            fs=fs,
            nperseg=win_samples // 2,
            noverlap=win_samples // 4
        )

        def band_power(band):
            idx = (f >= band[0]) & (f <= band[1])
            return np.trapz(psd[idx], f[idx])

        p_alpha = band_power(alpha_band)
        p_low = band_power(low_band)
        p_total = band_power(total_band)

        if p_total > 0:
            alpha_rel.append(p_alpha / p_total)
            low_rel.append(p_low / p_total)
        else:
            alpha_rel.append(0.0)
            low_rel.append(0.0)

        times.append((start + win_samples / 2) / fs)

    return np.array(times), np.array(alpha_rel), np.array(low_rel)


# ===================== 主逻辑 =====================

def run_calibration(args):
    print("=== Alpha calibration started ===")
    print("EDF file:", args.edf)

    raw = mne.io.read_raw_edf(args.edf, preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]

    # 选第一个 EEG 通道
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise RuntimeError("No EEG channel found in EDF.")
    ch_idx = picks[0]
    ch_name = raw.ch_names[ch_idx]

    eeg = raw.get_data(picks=[ch_idx])[0]
    duration = len(eeg) / fs

    print(f"Using channel: {ch_name}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Duration: {duration:.1f} s")

    # ---------- 带通滤波 ----------
    eeg_bp = bandpass_filter(eeg, fs, band=args.bandpass)

    # ---------- 滑窗功率 ----------
    times, alpha_rel, low_rel = compute_relative_powers(
        eeg_bp,
        fs,
        args.win_length,
        args.step,
        args.alpha_band,
        args.low_band,
        args.bandpass,
    )

    # ---------- 伪迹阈值 ----------
    low_thr = low_rel.mean() + args.low_k_std * low_rel.std()
    artifact_mask = low_rel > low_thr

    # ---------- alpha 平滑 ----------
    alpha_rel_smooth = moving_average(alpha_rel, args.smooth_k)

    valid_mask = ~artifact_mask
    alpha_valid = alpha_rel_smooth[valid_mask]

    if len(alpha_valid) < 10:
        raise RuntimeError("Not enough valid windows for GMM calibration.")

    # ---------- GMM ----------
    gmm = GaussianMixture(
        n_components=2,
        random_state=0,
        init_params="random",
        n_init=1,
    ).fit(alpha_valid.reshape(-1, 1))

    means = np.sort(gmm.means_.flatten())
    mu_open, mu_closed = means[0], means[1]

    alpha_thr = mu_open + args.alpha_ratio * (mu_closed - mu_open)

    print("mu_open   =", mu_open)
    print("mu_closed =", mu_closed)
    print("alpha_thr =", alpha_thr)
    print("low_thr   =", low_thr)

    # ---------- 保存参数 ----------
    np.savez(
        args.out,
        fs=fs,
        bandpass=args.bandpass,
        alpha_band=args.alpha_band,
        low_band=args.low_band,
        win_length=args.win_length,
        step=args.step,
        alpha_thr=alpha_thr,
        low_thr=low_thr,
    )
    print("Saved parameters to:", args.out)

    # ---------- 画图 ----------
    t_all = np.arange(len(eeg)) / fs

    plt.figure(figsize=(13, 8))

    # EEG
    plt.subplot(3, 1, 1)
    plt.plot(t_all, eeg_bp, linewidth=0.5)
    plt.title(f"EEG band-passed {args.bandpass[0]}–{args.bandpass[1]} Hz ({ch_name})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # alpha_rel
    plt.subplot(3, 1, 2)
    plt.plot(times, alpha_rel, label="alpha_rel")
    plt.scatter(times[artifact_mask],
                alpha_rel[artifact_mask],
                color="k", marker="x", s=25, label="artifact")
    plt.axhline(alpha_thr, color="r", linestyle="--", label="alpha_thr")
    plt.title("Calibration: alpha relative power")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative power")
    plt.legend()
    plt.grid(True)

    # classification
    state = np.full_like(alpha_rel, "open", dtype=object)
    state[alpha_rel_smooth > alpha_thr] = "closed"
    state[artifact_mask] = "artifact"

    plt.subplot(3, 1, 3)
    plt.plot(times, alpha_rel, label="alpha_rel")
    plt.scatter(times[state == "closed"],
                alpha_rel[state == "closed"],
                color="g", s=20, label="closed")
    plt.scatter(times[state == "artifact"],
                alpha_rel[state == "artifact"],
                color="k", marker="x", s=30, label="artifact")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative power")
    plt.title("Calibration classification result")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.fig, dpi=150)
    plt.close()

    print("Saved figure to:", args.fig)
    print("=== Calibration finished successfully ===")


# ===================== CLI 入口 =====================

from pathlib import Path
import argparse
import sys

def find_default_edf():
    edfs = list(Path(".").glob("*.edf"))
    if len(edfs) == 0:
        return None
    if len(edfs) > 1:
        print("Multiple .edf files found, please specify --edf explicitly:")
        for p in edfs:
            print("  ", p)
        sys.exit(1)
    return str(edfs[0])


def main():
    parser = argparse.ArgumentParser(
        description="EEG alpha calibration tool"
    )
    # parser.add_argument("--edf", required=True, help="Calibration EDF file")
    parser.add_argument(
        "--edf",
        # default=r"*.edf",
        help="Calibration EDF file"
    )

    # args = parser.parse_args()



    parser.add_argument("--out", default="alpha_calibration_params.npz",
                        help="Output calibration parameter file (.npz)")
    parser.add_argument("--fig", default="calibration_result.png",
                        help="Output figure file (.png)")

    parser.add_argument("--win-length", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--smooth-k", type=int, default=5)
    parser.add_argument("--low-k-std", type=float, default=2.0)
    parser.add_argument("--alpha-ratio", type=float, default=0.4)

    parser.add_argument("--bandpass", type=float, nargs=2, default=(1.0, 40.0))
    parser.add_argument("--alpha-band", type=float, nargs=2, default=(8.0, 12.0))
    parser.add_argument("--low-band", type=float, nargs=2, default=(1.0, 4.0))

    args = parser.parse_args()


    if args.edf is None:
        args.edf = find_default_edf()
        if args.edf is None:
            parser.error("No --edf given and no *.edf found in current directory")

    print("Using EDF:", args.edf)


    try:
        run_calibration(args)
    except Exception as e:
        print("Calibration failed:", str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
