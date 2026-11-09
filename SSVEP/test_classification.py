"""Frequency synthesis, filter-bank decoding and freshness of command evidence."""
import copy
import json
import unittest

import numpy as np

try:
    from .core import (FBCCADecoder, StableDecisionGate, make_targets, luminance, simulated_samples)
except ImportError:
    from core import (FBCCADecoder, StableDecisionGate, make_targets, luminance, simulated_samples)


class ClassificationTests(unittest.TestCase):
    def test_fixed_limb_mapping_and_wide_profile_spectrum(self):
        for refresh in (59.94, 60, 72, 75):
            targets = make_targets(refresh, "wide")
            self.assertEqual([t.character for t in targets], ["左肩", "右肩", "左肘", "右肘"])
            self.assertEqual([t.frequency for t in targets], [8, 13, 18, 23])
            self.assertGreaterEqual(min(np.diff([t.frequency for t in targets])), 5)
            n = round(refresh*20)
            for target in targets:
                x = luminance(np.arange(n), target.frames_per_cycle)
                peak = np.fft.rfftfreq(n, 1/refresh)[np.argmax(abs(np.fft.rfft(x-x.mean()))) ]
                self.assertLess(abs(peak-target.frequency), .06)
        with self.assertRaises(ValueError):
            make_targets(45, "wide")

    def test_filter_bank_recovers_all_targets_with_noise_and_dc(self):
        for profile in ("integer", "wide"):
            targets = make_targets(60, profile)
            decoder = FBCCADecoder([t.frequency for t in targets])
            for i, target in enumerate(targets):
                for seed in (8, 29, 101):
                    x = simulated_samples(np.arange(1500), target.frequency, seed=seed)+np.arange(8)*500
                    result = decoder.decode(x)
                    self.assertEqual(result["best"], i)
                    self.assertGreater(result["snr_db"][i], 3)
                    self.assertTrue(np.all((np.array(result["scores"]) >= 0) & (np.array(result["scores"]) <= 1)))
                    gate = StableDecisionGate()
                    for _ in range(3):
                        accepted = gate.update(copy.deepcopy(result))
                    self.assertEqual(accepted, i)
                    json.dumps(result, allow_nan=False)

    def test_rank_deficiency_and_permutation_do_not_create_different_classes(self):
        targets = make_targets(60)
        decoder = FBCCADecoder([t.frequency for t in targets])
        x = simulated_samples(np.arange(1500), targets[2].frequency)
        result = decoder.decode(x)
        np.testing.assert_allclose(result["scores"], decoder.decode(x[:, ::-1])["scores"], atol=1e-9)
        self.assertEqual(decoder.decode(np.repeat(x[:, :1], 8, axis=1))["best"], 2)
        self.assertEqual(decoder.decode(np.zeros((1500, 8)))["score"], 0.)

    def test_noise_and_non_candidate_tone_are_not_confirmed(self):
        targets = make_targets(60, "wide")
        decoder = FBCCADecoder([t.frequency for t in targets])
        gate = StableDecisionGate()
        for seed in range(20):
            x = np.random.default_rng(seed).normal(0, 15, (1500, 8))
            self.assertIsNone(gate.update(decoder.decode(x)))
        t = np.arange(1500)[:, None]/500
        for seed in range(6):
            x = 20*np.sin(2*np.pi*10*t)+np.random.default_rng(seed).normal(0, 6, (1500, 8))
            self.assertIsNone(gate.update(decoder.decode(x)))

    def test_gate_rejects_current_switch_and_snr_without_holding_old_result(self):
        gate = StableDecisionGate()
        def result(scores, snr=12):
            return {"scores": scores, "snr_db": [snr]*4}
        for _ in range(3):
            accepted = gate.update(result([.85, .1, .1, .1]))
        self.assertEqual(accepted, 0)
        switched = result([.1, .85, .1, .1])
        self.assertIsNone(gate.update(switched))
        self.assertIsNone(switched["confirmed"])
        for _ in range(5):
            switched = result([.1, .85, .1, .1])
            accepted = gate.update(switched)
        self.assertEqual(accepted, 1)
        poor = result([.1, .85, .1, .1], snr=0)
        self.assertIsNone(gate.update(poor))
        self.assertIn("信噪比", poor["reason"])
        self.assertEqual(poor["streak"], 0)
        gate.reset()
        self.assertIsNone(gate.update(result([.1, .85, .1, .1])))


if __name__ == "__main__":
    unittest.main()
