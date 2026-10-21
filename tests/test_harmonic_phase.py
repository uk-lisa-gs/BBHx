"""Regression checks for the intrinsic inspiral harmonic phase alignment."""

import unittest

import numpy as np

from bbhx.waveforms.phenomhm import PhenomHMAmpPhase


MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
M = np.array([m for _, m in MODES])
# Leading-order complex-amplitude shifts in the PhenomHM mode convention.
MODE_PHASE = np.array([0.0, np.pi / 2, -np.pi / 2, 0.0, np.pi, -np.pi / 2])


class HarmonicPhaseTest(unittest.TestCase):
    def test_inspiral_modes_share_the_same_orbital_phase(self):
        # Compare the modes at the same orbital frequency, well inside every
        # inspiral branch. Common time and orbital-phase shifts then cancel.
        f22 = np.geomspace(2e-5, 8e-5, 32)
        frequencies = (M[:, None] * f22 / 2)[None]
        configurations = [
            (1.5e6, 0.5e6, 0.0, 0.0),
            (1.5e6, 0.5e6, 0.066, -0.211),
            (1.5e6, 0.5e6, -0.068, 0.217),
            (3.0e6, 1.0e6, 0.6, -0.4),
        ]
        for m1, m2, chi1, chi2 in configurations:
            for t_ref in (0.0, 1e6):
                with self.subTest(
                    m1=m1, m2=m2, chi1=chi1, chi2=chi2, t_ref=t_ref
                ):
                    waveform = PhenomHMAmpPhase(force_backend="cpu")
                    waveform.run_wave(
                        m1, m2, chi1, chi2, 1e27, 0.0, 0.0, t_ref,
                        len(f22), freqs=frequencies, modes=MODES,
                    )
                    phase = waveform.phase[0]
                    relative = phase - M[:, None] * phase[0] / 2
                    expected = np.broadcast_to(MODE_PHASE[:, None], relative.shape)
                    np.testing.assert_allclose(
                        relative, expected, rtol=0.0, atol=2e-8
                    )
