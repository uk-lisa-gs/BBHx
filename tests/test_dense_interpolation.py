"""The single-launch dense interpolation (fill=True, combine=False) against the per-binary path."""

import unittest

import numpy as np

from bbhx.utils.constants import PC_SI, YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD

DAY = 86400.0
DT = 10.0
T0 = 150 * DAY
N = round(30 * DAY / DT)
FREQS = np.fft.rfftfreq(N, DT)[1:]
MODES = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]


def batch():
    # three different binaries (different masses, merger times and sky positions),
    # so their frequency ranges in the data differ
    m1 = np.array([3.0e6, 1.2e6, 6.0e6])
    m2 = np.array([1.5e6, 1.0e6, 2.0e6])
    t_ref = T0 + np.array([28.0, 20.0, 29.5]) * DAY
    return [m1, m2, np.array([0.4, 0.0, -0.5]), np.array([-0.2, 0.3, 0.1]),
            np.full(3, 3.0e4 * 1e6 * PC_SI), np.array([0.7, 2.0, 4.0]), np.zeros(3),
            np.array([0.9, 2.0, 0.3]), np.array([2.1, 4.0, 0.5]), np.array([0.3, -0.8, 1.1]),
            np.array([0.5, 1.5, 2.5]), t_ref]


class DenseInterpolationTest(unittest.TestCase):
    def test_matches_per_binary_interpolation(self):
        wave = BBHWaveformFD(amp_phase_kwargs={"run_phenomd": False}, force_backend="cpu")
        kw = dict(freqs=FREQS, modes=MODES, direct=False, length=1024,
                  t_obs_start=(T0 - 2 * DAY) / YRSID_SI, t_obs_end=(T0 + 29 * DAY) / YRSID_SI)
        dense = wave(*batch(), fill=True, combine=False, **kw)

        # the per-binary path: TemplateInterpFD output, filled by hand
        channels, start_inds, lengths = wave(*batch(), fill=False, **kw)
        expected = np.zeros_like(dense)
        for i, (temp, start, length) in enumerate(zip(channels, start_inds, lengths)):
            expected[i, :, start : start + length] = temp

        self.assertEqual(dense.shape, (3, 3, FREQS.size))
        self.assertGreater(len(set(int(s) for s in start_inds)), 1)
        np.testing.assert_allclose(dense, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


if __name__ == "__main__":
    unittest.main()
