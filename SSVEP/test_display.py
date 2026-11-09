"""Signal units, display quality gates, and measured-area interpolation."""
import unittest

import numpy as np

try:
    from .core import make_targets
    from .display_data import DisplayProcessor, power_in_band, posterior_field
except ImportError:
    from core import make_targets
    from display_data import DisplayProcessor, power_in_band, posterior_field


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.targets = make_targets(60)
        self.stamps = 10 + np.arange(2000)/500
        self.amplitude = np.arange(1, 9)*2.
        self.samples = np.sin(2*np.pi*15*np.arange(2000)[:, None]/500)*self.amplitude

    def test_waveform_and_spectrum_keep_physical_units(self):
        payload = DisplayProcessor(self.targets).process(self.samples, self.stamps, now=14)
        self.assertEqual(payload["reason"], "")
        self.assertEqual(np.count_nonzero(payload["good"]), 8)
        peak = payload["frequencies"][np.argmax(payload["psd"], axis=0)]
        np.testing.assert_allclose(peak, 15, atol=.3)
        power = 10**(payload["band_db"][3]/10)  # 左, 15 Hz +/- 1 Hz
        np.testing.assert_allclose(power, self.amplitude**2/2, rtol=.035)
        np.testing.assert_allclose(np.std(payload["waveform"], axis=0), self.amplitude/np.sqrt(2), rtol=.035)
        self.assertLessEqual(payload["times"][-1], -.15)

    def test_density_integration_uses_bin_width_and_band_edges(self):
        freq = np.arange(0, 51, .5)
        psd = np.full((len(freq), 8), 2.)
        np.testing.assert_allclose(power_in_band(freq, psd, 6, 30), 48.)
        np.testing.assert_allclose(power_in_band(freq, psd, 14, 16), 4.)

    def test_zero_stale_and_missing_input_never_become_valid_topography(self):
        processor = DisplayProcessor(self.targets)
        empty = processor.process(None, None, now=14)
        self.assertFalse(empty["available"])
        zero = processor.process(np.zeros_like(self.samples), self.stamps, now=14)
        self.assertIn("全零", zero["reason"])
        self.assertFalse(np.any(zero["good"]))
        stale = processor.process(self.samples, self.stamps, now=15)
        self.assertTrue(stale["stale"])
        self.assertFalse(np.any(stale["good"]))

    def test_display_spectrum_is_throttled_but_waveforms_refresh(self):
        processor = DisplayProcessor(self.targets)
        first = processor.process(self.samples, self.stamps, now=14)
        second = processor.process(self.samples*2, self.stamps+.1, now=14.1)
        self.assertEqual(first["spectral_id"], second["spectral_id"])
        np.testing.assert_allclose(second["waveform"], first["waveform"]*2)
        third = processor.process(self.samples*2, self.stamps+.6, now=14.6)
        self.assertNotEqual(first["spectral_id"], third["spectral_id"])

    def test_topography_masks_unmeasured_regions_and_insufficient_sensors(self):
        field = posterior_field(np.ones(8)*7, np.ones(8, dtype=bool))
        self.assertTrue(np.any(np.isfinite(field)))
        np.testing.assert_allclose(field[np.isfinite(field)], 7.)
        self.assertTrue(np.all(np.isnan(field[:20])))
        self.assertTrue(np.isnan(field[0, 0]))
        self.assertTrue(np.all(np.isnan(posterior_field(np.ones(8), [1, 1, 0, 0, 0, 0, 0, 0]))))
        partial = posterior_field([1000, 7, 7, 7, 1000, 1000, 1000, 1000], [0, 1, 1, 1, 0, 0, 0, 0])
        self.assertTrue(np.any(np.isfinite(partial)))
        np.testing.assert_allclose(partial[np.isfinite(partial)], 7.)
        self.assertTrue(np.all(np.isnan(partial[80:])))


if __name__ == "__main__":
    unittest.main()
