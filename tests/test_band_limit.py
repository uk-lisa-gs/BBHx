"""Interpolated waveforms for observations starting at t > 0, and the end-of-observation taper."""

import unittest

import numpy as np

from bbhx.utils.constants import PC_SI, YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD

DAY = 86400.0
DT = 10.0
T0 = 150 * DAY  # observation starts mid-year
N = round(30 * DAY / DT)
FREQS = np.fft.rfftfreq(N, DT)[1:]
T_MERGE = T0 + 28 * DAY
MODES = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]
PARAMS = [3.0e6, 1.5e6, 0.4, -0.2, 3.0e4 * 1e6 * PC_SI, 0.7, 0.0, 0.9, 2.1, 0.3, 0.5, T_MERGE]
WINDOW = dict(t_obs_start=(T0 - 2 * DAY) / YRSID_SI, t_obs_end=(T0 + 30 * DAY) / YRSID_SI)


def generate(**kwargs):
    wave = BBHWaveformFD(amp_phase_kwargs={"run_phenomd": False}, force_backend="cpu")
    args = [np.atleast_1d(p) for p in PARAMS]
    kw = {**WINDOW, **kwargs}
    h = wave(*args, freqs=FREQS, modes=MODES, fill=True, combine=False, squeeze=False, **kw)
    return h[0, :2], wave


def mismatch(a, b):
    ab = np.real(np.vdot(a, b))
    return 1.0 - ab / np.sqrt(np.real(np.vdot(a, a)) * np.real(np.vdot(b, b)))


def frequency_at(wave, mode, t):
    """Frequency where the (direct) harmonic's t_f first reaches t."""
    tf = wave.out_buffer_final[2, 0, MODES.index(mode)]
    return FREQS[np.argmax(tf > t)]


class BandLimitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.direct, cls.direct_wave = generate(direct=True)

    def test_interpolated_matches_direct_for_late_observation_start(self):
        # The sparse grid used to be clipped to t_f in [0, t_obs_end - t_obs_start],
        # i.e. the wrong place for any observation not starting at t = 0.
        interp, _ = generate(direct=False, length=4096)
        self.assertLess(mismatch(self.direct, interp), 1e-6)
        ratio = np.linalg.norm(interp) / np.linalg.norm(self.direct)
        self.assertAlmostEqual(ratio, 1.0, places=4)

    def test_taper_leaves_early_frequencies_and_removes_late_ones(self):
        t_taper, t_end = T_MERGE - 2.0 * DAY, T_MERGE - 0.5 * DAY
        tapered, _ = generate(direct=True, t_obs_end_taper=t_taper / YRSID_SI, t_obs_end=t_end / YRSID_SI)
        # At a given frequency the (2,1) harmonic is the last to arrive and (4,4)
        # the first: below f_21(t_taper) nothing is tapered, above f_44(t_end)
        # everything is gone.
        f_untouched = frequency_at(self.direct_wave, (2, 1), t_taper) * 0.99  # L-frame vs SSB slack
        f_gone = frequency_at(self.direct_wave, (4, 4), t_end) * 1.01
        below, above = FREQS < f_untouched, FREQS > f_gone
        np.testing.assert_array_equal(tapered[:, below], self.direct[:, below])
        self.assertTrue(np.all(tapered[:, above] == 0.0))

    def test_interpolated_taper_matches_direct_and_stops_early(self):
        t_taper, t_end = T_MERGE - 2.0 * DAY, T_MERGE - 0.5 * DAY
        window = dict(t_obs_end_taper=t_taper / YRSID_SI, t_obs_end=t_end / YRSID_SI)
        direct, _ = generate(direct=True, **window)
        interp, _ = generate(direct=False, length=4096, **window)
        self.assertLess(mismatch(direct, interp), 1e-6)
        # the dense frequencies evaluated stop where the (4,4) taper ends
        f_gone = frequency_at(self.direct_wave, (4, 4), t_end) * 1.01
        self.assertLess(FREQS[np.flatnonzero(np.abs(interp[0]) > 0)[-1]], f_gone)

    def test_per_binary_taper_matches_separate_calls(self):
        # Two binaries merging a day apart, each tapered relative to its own merger.
        wave = BBHWaveformFD(amp_phase_kwargs={"run_phenomd": False}, force_backend="cpu")
        args = [np.array([p, p]) for p in PARAMS]
        args[-1] = np.array([T_MERGE, T_MERGE - DAY])
        t_taper = (args[-1] - 2.0 * DAY) / YRSID_SI
        t_end = (args[-1] - 0.5 * DAY) / YRSID_SI
        kw = dict(freqs=FREQS, modes=MODES, fill=True, combine=False, squeeze=False, direct=False, length=4096,
                  t_obs_start=WINDOW["t_obs_start"])
        both = wave(*args, t_obs_end_taper=t_taper, t_obs_end=t_end, **kw)
        for i in range(2):
            one = wave(*[a[i : i + 1] for a in args], t_obs_end_taper=t_taper[i], t_obs_end=t_end[i], **kw)
            np.testing.assert_allclose(both[i], one[0], rtol=0, atol=1e-12 * np.abs(one).max())

    def test_taper_must_precede_end(self):
        with self.assertRaises(ValueError):
            generate(direct=False, length=1024, t_obs_end_taper=WINDOW["t_obs_end"] * 1.01)


if __name__ == "__main__":
    unittest.main()
