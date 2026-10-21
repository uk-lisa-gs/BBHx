"""Regression checks for PhenomHM at (near-)equal masses."""

import multiprocessing
import os
import unittest

import numpy as np

from bbhx.waveforms.phenomhm import PhenomHMAmpPhase


MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
T_REF = 1.86e7


def _masses(mchirp, q):
    m1 = mchirp * (1 + q) ** (1 / 5) / q ** (3 / 5)
    m2 = mchirp * q ** (2 / 5) * (1 + q) ** (1 / 5)
    return m1, m2


def _generate_and_exit(m1, m2, chi1, chi2):
    waveform = PhenomHMAmpPhase(force_backend="cpu")
    waveform.run_wave(m1, m2, chi1, chi2, 1e27, 0.0, 0.0, T_REF, 1024, modes=MODES)
    finite = all(
        np.all(np.isfinite(np.asarray(x)))
        for x in (waveform.amp, waveform.phase, waveform.tf)
    )
    os._exit(0 if finite else 2)


class EqualMassTest(unittest.TestCase):
    def test_rounding_past_equal_mass_gives_finite_waveforms(self):
        # These masses round to eta > 1/4; the unguarded sqrt(1 - 4 eta) gave a
        # NaN final spin and an out-of-bounds ringdown-table read. Run in a
        # child process so a crash fails the test rather than the runner.
        m1, m2 = _masses(2496791.0256973505, 0.9999999994963521)
        self.assertLess(1 - 4 * m1 * m2 / (m1 + m2) ** 2, 0.0)
        child = multiprocessing.get_context("fork").Process(
            target=_generate_and_exit, args=(m1, m2, -0.8227, -0.4804)
        )
        child.start()
        child.join()
        self.assertEqual(child.exitcode, 0)

    def test_21_mode_is_smooth_up_to_equal_mass(self):
        # London et al. (arXiv:1708.00404) Eq. (4) divides by PN_21(2f/m), which
        # passes through zero in the merger-ringdown as q -> 1 when chi1 != chi2,
        # turning the (2,1) mode into a narrow spectral line (x87 at ~5 mHz here,
        # at 3% of the mode's peak). Between q = 0.99 and q = 1 the mode should
        # change little wherever it carries signal; with the cap the largest
        # change is ~1.5x, at < 1e-3 of the peak. (Deep in the ringdown tail,
        # < 1e-6 of the peak, ratios of tiny numbers are not meaningful.)
        frequencies = np.geomspace(1e-5, 3e-2, 4000)
        amp21 = {}
        for q in (0.99, 0.999, 0.9999, 0.99999, 1.0):
            mtot = 5.77e6
            waveform = PhenomHMAmpPhase(force_backend="cpu")
            waveform.run_wave(
                mtot / (1 + q), mtot * q / (1 + q), -0.25, -0.97, 1e27, 0.0, 0.0,
                T_REF, frequencies.size, freqs=frequencies, modes=MODES,
            )
            amp21[q] = np.asarray(waveform.amp).reshape(len(MODES), -1)[1]
        reference = amp21[0.99]
        positive = reference > 1e-6 * reference.max()
        for q, amp in amp21.items():
            with self.subTest(q=q):
                self.assertTrue(np.all(np.isfinite(amp)))
                self.assertLess(np.max(amp[positive] / reference[positive]), 2.0)


if __name__ == "__main__":
    unittest.main()
