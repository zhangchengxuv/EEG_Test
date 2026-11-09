"""Relaxed decoding and explicit, persistent mouse-driven test results."""
import copy
import json
import tempfile
import unittest

import numpy as np

try:
    from .core import (RELAXED_GATE, FBCCADecoder, MouseTargetLatch, StableDecisionGate,
                       assess_window, make_targets, prepare_decode_window, simulated_samples)
    from .acquisition import SessionLog
    from .display_data import DisplayProcessor
except ImportError:
    from core import (RELAXED_GATE, FBCCADecoder, MouseTargetLatch, StableDecisionGate,
                      assess_window, make_targets, prepare_decode_window, simulated_samples)
    from acquisition import SessionLog
    from display_data import DisplayProcessor


class ModeTests(unittest.TestCase):
    def test_mouse_latches_after_leave_and_only_changes_to_another_target(self):
        latch = MouseTargetLatch()
        targets = make_targets(60)
        self.assertIsNone(latch.result(targets)["confirmed"])
        self.assertTrue(latch.observe(2, now=10))
        self.assertFalse(latch.observe(None, now=11))
        self.assertFalse(latch.observe(2, now=12))
        held = latch.result(targets, now=30)
        self.assertEqual(held["confirmed"], 2)
        self.assertEqual(held["mouse_selected_monotonic_s"], 10)
        self.assertEqual(held["result_source"], "mouse")
        self.assertEqual(held["mode_name"], "测试模式")
        self.assertEqual(held["target_name"], "左肘")
        for field in ("scores", "score", "snr_db", "spectral_peak_hz"):
            self.assertNotIn(field, held)
        self.assertTrue(latch.observe(3, now=31))
        self.assertEqual(latch.result(targets)["target_name"], "右肘")
        latch.reset()
        self.assertIsNone(latch.result(targets)["confirmed"])

    def test_lower_gate_accepts_weak_stable_evidence_without_snr_rejection(self):
        result = {"scores": [.1, .12, .27, .24], "snr_db": [0, 0, -4, 0]}
        strict = StableDecisionGate()
        self.assertIsNone(strict.update(copy.deepcopy(result)))
        gate = StableDecisionGate(**RELAXED_GATE)
        self.assertIsNone(gate.update(copy.deepcopy(result)))
        self.assertEqual(gate.update(copy.deepcopy(result)), 2)
        flat = {"scores": [0, 0, 0, 0], "snr_db": [0]*4}
        self.assertIsNone(gate.update(flat))

    def test_one_varying_channel_can_decode_despite_seven_bad_channels(self):
        targets = make_targets(60)
        x = np.full((1500, 8), -187500.)
        x[:, 6] = simulated_samples(np.arange(1500), targets[2].frequency)[:, 6]
        stamps = 10+np.arange(1500)/500
        self.assertIn("有效通道不足", assess_window(x, stamps, now=13)[0])
        prepared, used = prepare_decode_window(x)
        self.assertEqual(np.count_nonzero(used), 1)
        result = FBCCADecoder([t.frequency for t in targets]).decode(prepared)
        self.assertEqual(result["best"], 2)
        gate = StableDecisionGate(**RELAXED_GATE)
        for _ in range(2):
            accepted = gate.update(copy.deepcopy(result))
        self.assertEqual(accepted, 2)

    def test_robust_clipping_limits_spikes_and_constant_data_stays_uninformative(self):
        x = simulated_samples(np.arange(1500), 15)
        x[700, :] += 1e6
        prepared, used = prepare_decode_window(x)
        self.assertTrue(np.all(used))
        self.assertLess(np.max(abs(prepared)), 200)
        constant, used = prepare_decode_window(np.full((1500, 8), 187500.))
        self.assertFalse(np.any(used))
        result = FBCCADecoder([t.frequency for t in make_targets(60)]).decode(constant)
        self.assertEqual(result["score"], 0)
        with self.assertRaises(ValueError):
            prepare_decode_window(np.full((1500, 8), np.nan))

    def test_display_keeps_partial_quality_count_and_real_spectrum(self):
        x = np.zeros((2000, 8))
        x[:, :2] = simulated_samples(np.arange(2000), 15)[:, :2]
        payload = DisplayProcessor(make_targets(60)).process(x, 10+np.arange(2000)/500, now=14)
        self.assertEqual(np.count_nonzero(payload["good"]), 2)
        self.assertEqual(np.count_nonzero(payload["plot_channels"]), 2)
        self.assertTrue(payload["available"])
        self.assertGreater(np.max(payload["psd"]), 0)

    def test_mouse_records_preserve_actual_raw_values_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            log = SessionLog(directory, {"mode": "test", "result_source": "mouse"})
            log.submit("raw", [[1, 10, 1, 0, *range(8)]])
            latch = MouseTargetLatch()
            latch.observe(3, now=10)
            log.submit("decision", latch.result(make_targets(60), now=10.1))
            log.close()
            meta = json.loads((log.path/"session.json").read_text(encoding="utf-8"))
            result = json.loads((log.path/"decisions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(meta["result_source"], "mouse")
            self.assertEqual(result["confirmed"], 3)
            self.assertNotIn("score", result)
            self.assertTrue((log.path/"raw_eeg.csv").read_text().splitlines()[1].endswith("0,1,2,3,4,5,6,7"))


if __name__ == "__main__":
    unittest.main()
