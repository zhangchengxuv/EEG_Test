"""Run: python -B -m unittest discover -s SSVEP -p test_demo.py -v"""
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

import numpy as np

try:
    from .core import (CCADecoder, DecisionGate, FrameTiming, SampleBuffer, assess_window,
                       luminance, make_targets, parse_line, simulated_samples)
    from .acquisition import AcquisitionThread, SessionLog
except ImportError:
    from core import (CCADecoder, DecisionGate, FrameTiming, SampleBuffer, assess_window,
                      luminance, make_targets, parse_line, simulated_samples)
    from acquisition import AcquisitionThread, SessionLog


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.targets = make_targets(60)
        self.decoder = CCADecoder([t.frequency for t in self.targets])

    def test_refresh_rate_determines_real_frequency(self):
        self.assertEqual([t.frames_per_cycle for t in self.targets], [7, 5, 4, 3])
        self.assertAlmostEqual(self.targets[0].frequency, 60 / 7)
        self.assertGreaterEqual(min(np.diff([t.frequency for t in self.targets])), 3)
        for hz in (59.94, 75, 120, 144):
            for target in make_targets(hz):
                self.assertAlmostEqual(target.frequency * target.frames_per_cycle, hz)

    def test_quantized_targets_do_not_become_integer_harmonic_pairs(self):
        for hz in (59.94, 60, 72, 75, 120, 144):
            frequencies = [target.frequency for target in make_targets(hz)]
            self.assertGreaterEqual(min(np.diff(frequencies)), 2)
            for i, low in enumerate(frequencies):
                for high in frequencies[i + 1:]:
                    ratio = high / low
                    self.assertGreater(abs(ratio - round(ratio)), 0.001)

    def test_sampled_luminance_spectrum(self):
        frames = np.arange(420)
        for target in self.targets:
            x = luminance(frames, target.frames_per_cycle)
            spectrum = abs(np.fft.rfft(x - x.mean()))
            measured = np.fft.rfftfreq(len(x), 1 / 60)[np.argmax(spectrum)]
            self.assertAlmostEqual(measured, target.frequency)

    def test_parse_exactly_eight_volts_and_ignore_noise(self):
        np.testing.assert_allclose(parse_line("channel:1e-6,-2e-6,0,0,0,0,0,0"), [1, -2, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(parse_line(bytearray(b"channel:1e-6,-2e-6,0,0,0,0,0,0")), [1, -2, 0, 0, 0, 0, 0, 0])
        for text in ("Device ID: 0b111110", "channel:0,0,0", "channel:" + ",".join(["0"] * 9),
                     "channel:nan,0,0,0,0,0,0,0", "channel:inf,0,0,0,0,0,0,0"):
            self.assertIsNone(parse_line(text))

    def test_all_four_frequencies_with_noise_and_phases(self):
        for i, target in enumerate(self.targets):
            for seed in (12, 50, 100):
                x = simulated_samples(np.arange(1500), target.frequency, seed=seed)
                result = self.decoder.decode(x)
                self.assertEqual(result["best"], i)
                self.assertGreater(result["score"], 0.7)
                self.assertGreater(result["margin"], 0.08)
                self.assertLess(abs(result["spectral_peak_hz"] - target.frequency), 0.6)

    def test_channel_permutation_and_dc_offset_do_not_change_identification(self):
        x = simulated_samples(np.arange(1500), self.targets[2].frequency)
        x += np.arange(8) * 100
        original = self.decoder.decode(x)
        permuted = self.decoder.decode(x[:, ::-1])
        self.assertEqual(original["best"], 2)
        np.testing.assert_allclose(original["scores"], permuted["scores"], atol=1e-9)

    def test_independent_noise_is_not_confirmed(self):
        gate = DecisionGate()
        for seed in range(8):
            x = np.random.default_rng(seed).normal(0, 10, (1500, 8))
            self.assertIsNone(gate.update(self.decoder.decode(x)))

    def test_flat_rank_deficient_window_has_no_cca_evidence(self):
        result = self.decoder.decode(np.zeros((1500, 8)))
        self.assertEqual(result["score"], 0)
        self.assertIsNone(DecisionGate().update(result))
        tone = simulated_samples(np.arange(1500), self.targets[2].frequency)[:, :1]
        self.assertEqual(self.decoder.decode(np.repeat(tone, 8, axis=1))["best"], 2)

    def test_decision_requires_repeated_evidence_and_resets(self):
        gate = DecisionGate()
        strong = {"best": 2, "score": 0.8, "margin": 0.3}
        self.assertIsNone(gate.update(strong))
        self.assertIsNone(gate.update(strong))
        self.assertEqual(gate.update(strong), 2)
        self.assertIsNone(gate.update(dict(strong, margin=0.01)))
        self.assertIsNone(gate.update(strong))

    def test_signal_health_zero_slow_stale_and_partially_flat(self):
        stamps = np.arange(1500) / 500 + 10
        x = simulated_samples(np.arange(1500), 12)
        self.assertEqual(assess_window(x, stamps, now=13)[0], "")
        self.assertIn("全零", assess_window(np.zeros_like(x), stamps, now=13)[0])
        self.assertIn("速率", assess_window(x, np.arange(1500) / 250 + 7, now=13)[0])
        self.assertIn("中断", assess_window(x, stamps, now=20)[0])
        x[:, :5] = 0
        reason, good = assess_window(x, stamps, now=13)
        self.assertEqual(reason, "")
        self.assertEqual(np.count_nonzero(good), 3)
        x[:, 5] = 0
        self.assertIn("不足", assess_window(x, stamps, now=13)[0])

    def test_frame_timing_rejects_wrong_fps_stalls_and_dropped_frames(self):
        for fps, ok in ((60, True), (30, False), (600, False)):
            timing = FrameTiming(60)
            for stamp in 10 + np.arange(180) / fps:
                timing.record(stamp)
            self.assertEqual(timing.summary(now=stamp)["ok"], ok)
            json.dumps(timing.summary(now=stamp), allow_nan=False)
        self.assertFalse(timing.summary(now=stamp + 1)["ok"])
        timing = FrameTiming(60)
        for stamp in 10 + np.cumsum(np.tile([1 / 60] * 4 + [2 / 60], 40)):
            timing.record(stamp)
        self.assertFalse(timing.summary(now=stamp)["ok"])
        # A rare 400 ms stall followed by rapid catch-up can average 60 fps and
        # have less than 1% late frames; it must still fail presentation checks.
        intervals = np.full(180, 1 / 60)
        intervals[59] = 0.4
        intervals[60:84] = (25 / 60 - 0.4) / 24
        timing = FrameTiming(60)
        for stamp in 10 + np.cumsum(intervals):
            timing.record(stamp)
        report = timing.summary(now=stamp)
        self.assertFalse(report["ok"])
        self.assertAlmostEqual(report["max_interval_ms"], 400)
        self.assertIn("400 ms", report["reason"])
        # A new complete stable window must recover without restarting the app.
        for stamp in stamp + np.arange(1, 241) / 60:
            timing.record(stamp)
        self.assertTrue(timing.summary(now=stamp)["ok"])
        self.assertEqual(timing.summary(now=stamp)["reason"], "")

    def test_empty_and_sparse_frame_reports_are_finite_json(self):
        timing = FrameTiming(60)
        for count in (0, 1, 30):
            timing.reset()
            for stamp in np.arange(count):
                timing.record(stamp)
            report = timing.summary(seconds=0.1, now=31)
            self.assertFalse(report["ok"])
            json.dumps(report, allow_nan=False)


class IntegrationTests(unittest.TestCase):
    def test_buffer_returns_copy_and_latest_window(self):
        buffer = SampleBuffer(fs=2, seconds=2)
        for i in range(6):
            buffer.append(np.ones(8) * i, i)
        data, stamps = buffer.snapshot(1)
        np.testing.assert_equal(stamps, [4, 5])
        data[0, 0] = -100
        self.assertEqual(buffer.snapshot(1)[0][0, 0], 4)

    def test_session_preserves_raw_order_annotation_and_independent_decision(self):
        with tempfile.TemporaryDirectory() as root:
            log = SessionLog(root, {"annotation_target_id": 1, "source": "simulation"})
            log.submit("raw", [[5, 10, 1, 1, *range(8)]])
            log.submit("frame", [0, 10])
            log.submit("decision", {"best": 2, "confirmed": 2})
            log.close()
            self.assertFalse(log.error)
            raw = (log.path / "raw_eeg.csv").read_text().splitlines()
            self.assertIn("PO7,PO3,POz,PO4,PO8,O1,Oz,O2", raw[0])
            self.assertTrue(raw[1].endswith("0,1,2,3,4,5,6,7"))
            metadata = json.loads((log.path / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["annotation_target_id"], 1)
            decision = json.loads((log.path / "decisions.jsonl").read_text())
            self.assertEqual(decision["confirmed"], 2)

    def test_fragmented_tcp_stream_reads_eight_channels(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        def server():
            try:
                conn, _ = listener.accept()
                with conn:
                    conn.recv(1)
                    conn.sendall(b"startup information\ncha")
                    conn.sendall(b"nnel:0.000001,0.000002,0.000003,0.000004,0.000005,0.000006,0.000007,0.000008\n")
                    time.sleep(0.1)
            finally:
                listener.close()
        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        buffer = SampleBuffer(fs=1)
        receiver = AcquisitionThread("tcp", buffer, "127.0.0.1", port)
        receiver.start()
        deadline = time.perf_counter() + 2
        data = None
        while time.perf_counter() < deadline:
            data, _ = buffer.snapshot(1)
            if data is not None:
                break
            time.sleep(0.01)
        receiver.stop()
        thread.join(2)
        self.assertIsNotNone(data)
        np.testing.assert_allclose(data[0], np.arange(1, 9))


if __name__ == "__main__":
    unittest.main()
