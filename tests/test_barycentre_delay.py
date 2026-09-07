"""Regression checks for orbital-delay bookkeeping with absolute orbit positions."""

import numpy as np
import pytest
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.utils.constants import C_SI, YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD
from lisatools.detector import EqualArmlengthOrbits


class FrozenOrbits(EqualArmlengthOrbits):
    """Supply identical geometry at every time, including an absolute SSB centre."""

    translation = np.zeros(3)

    @property
    def x_base(self):
        positions = super().x_base
        return np.broadcast_to(positions[0] + self.translation, positions.shape).copy()

    @property
    def n_base(self):
        normals = super().n_base
        return np.broadcast_to(normals[0], normals.shape).copy()

    @property
    def ltt_base(self):
        times = super().ltt_base
        return np.broadcast_to(times[0], times.shape).copy()

    @property
    def v_base(self):
        return np.zeros_like(super().v_base)


class TranslatedFrozenOrbits(FrozenOrbits):
    translation = np.array([3.0e10, -2.0e10, 1.0e10])


class CenteredFrozenOrbits(FrozenOrbits):
    @property
    def x_base(self):
        positions = super().x_base
        return positions - positions.mean(axis=1, keepdims=True)


def frozen_response(orbits, epoch, longitude, latitude, tag):
    frequencies = np.linspace(0.002, 0.008, 513)
    response = LISATDIResponse(orbits=orbits, TDItag=tag, force_backend="cpu")
    shape = (1, 1, len(frequencies))
    response(
        frequencies,
        [0.9],
        [longitude],
        [latitude],
        [0.4],
        [0.7],
        len(frequencies),
        modes=[(2, 2)],
        phase=np.zeros(shape),
        tf=np.full(shape, epoch),
    )
    transfer = np.stack(
        [response.transferL1, response.transferL2, response.transferL3]
    )[:, 0, 0]
    return frequencies, np.conj(transfer * np.exp(1j * response.phase[0, 0]))


@pytest.mark.parametrize("tag", ["AET", "XYZ"])
@pytest.mark.parametrize("longitude,latitude", [(0.4, 0.2), (2.7, -0.6), (1.5, 0.4)])
def test_static_geometry_has_no_spurious_orbital_modulation(tag, longitude, latitude):
    # A hard-coded analytic orbit may split the phase internally, but cannot
    # change the physical response when the supplied detector geometry is fixed.
    _, first = frozen_response(FrozenOrbits(), 1e6, longitude, latitude, tag)
    _, later = frozen_response(FrozenOrbits(), 8e6, longitude, latitude, tag)
    np.testing.assert_allclose(later, first, rtol=2e-11, atol=1e-14)


@pytest.mark.parametrize("tag", ["AET", "XYZ"])
@pytest.mark.parametrize("longitude,latitude", [(0.4, 0.2), (2.7, -0.6), (1.5, 0.4)])
def test_translation_contributes_one_geometric_delay(tag, longitude, latitude):
    frequencies, original = frozen_response(
        FrozenOrbits(), 2e6, longitude, latitude, tag
    )
    _, translated = frozen_response(
        TranslatedFrozenOrbits(), 2e6, longitude, latitude, tag
    )
    direction = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    delay = -direction @ TranslatedFrozenOrbits.translation / C_SI
    expected = original * np.exp(-2j * np.pi * frequencies * delay)
    np.testing.assert_allclose(translated, expected, rtol=2e-11, atol=1e-14)


@pytest.mark.parametrize("tag", ["AET", "XYZ"])
@pytest.mark.parametrize("longitude,latitude", [(0.4, 0.2), (2.7, -0.6), (1.5, 0.4)])
def test_absolute_response_contains_only_one_barycentre_delay(tag, longitude, latitude):
    orbits = FrozenOrbits()
    frequencies, absolute = frozen_response(orbits, 1e6, longitude, latitude, tag)
    # Independent arm-only reference: centre the fixed geometry and evaluate
    # where the analytic p0 is perpendicular to the source, so its phase is zero.
    # This avoids relying on the implementation's split between transfer and
    # waveform-phase arrays, which changes when the double counting is fixed.
    null_epoch = ((longitude + np.pi / 2) % (2 * np.pi)) * YRSID_SI / (2 * np.pi)
    _, arm_only = frozen_response(
        CenteredFrozenOrbits(), null_epoch, longitude, latitude, tag
    )
    direction = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    centre = np.mean([orbits.get_pos(1e6, sc) for sc in [1, 2, 3]], axis=0)
    delay = -direction @ centre / C_SI
    expected = arm_only * np.exp(-2j * np.pi * frequencies * delay)
    np.testing.assert_allclose(absolute, expected, rtol=2e-11, atol=1e-14)


def test_full_waveform_time_shift_with_static_detector():
    generator = BBHWaveformFD(
        force_backend="cpu",
        amp_phase_kwargs={"run_phenomd": False},
        response_kwargs={"orbits": FrozenOrbits()},
    )
    frequencies = np.linspace(0.002, 0.008, 1025)
    # m1, m2, spin1, spin2, distance, phi_ref, f_ref, inc, lam, beta, psi, t_ref.
    parameters = [8e5, 3e5, 0.4, -0.2, 1e27, 0.7, 0, 0.9, 0.4, 0.2, 0.4, 1e6]
    kwargs = dict(
        freqs=frequencies,
        length=len(frequencies),
        modes=[(2, 2)],
        direct=True,
        fill=True,
        squeeze=False,
        t_obs_start=0,
        t_obs_end=1,
    )
    first = np.asarray(generator(*parameters, **kwargs)).copy()
    shift = 1000.0
    parameters[-1] += shift
    later = np.asarray(generator(*parameters, **kwargs)).copy()
    expected = first * np.exp(-2j * np.pi * frequencies * shift)
    assert np.linalg.norm(first) > 0
    relative_error = np.linalg.norm(later - expected) / np.linalg.norm(first)
    assert relative_error < 1e-9
