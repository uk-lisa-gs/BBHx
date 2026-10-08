"""Regression checks for the arm unit vectors used in the single-link response.

The reference below rebuilds first-generation TDI X, Y, Z from the spacecraft
positions alone (not the orbit file's link normals), following the lisabeta
convention used by ``d_EvaluateGslr``: arm l is opposite spacecraft l, with
n1 = (p3 - p2)/L, n2 = (p1 - p3)/L and n3 = (p2 - p1)/L. lisatools defines
n_ij = (p_i - p_j)/L, so these are links 32, 13 and 21; reading links 12, 23
and 31 instead pairs each link's projection with another arm's delay.
"""

import numpy as np
import pytest
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.utils.constants import C_SI
from lisatools.detector import EqualArmlengthOrbits


def spin_weighted_y2(m, inc, phi):
    sign = 1.0 if m == 2 else -1.0
    return np.sqrt(5.0 / (64.0 * np.pi)) * (1.0 + sign * np.cos(inc)) ** 2 * np.exp(1j * m * phi)


def polarization_matrix(inc, lam, beta, psi, phi):
    """Complex H for the (2, 2) mode, as built in ``responseCore``."""
    u = np.array([np.sin(lam), -np.cos(lam), 0.0])
    v = np.array([-np.sin(beta) * np.cos(lam), -np.sin(beta) * np.sin(lam), np.cos(beta)])
    p = np.cos(psi) * u + np.sin(psi) * v
    q = -np.sin(psi) * u + np.cos(psi) * v
    h_plus = np.outer(p, p) - np.outer(q, q)
    h_cross = np.outer(p, q) + np.outer(q, p)
    ylm = spin_weighted_y2(2, inc, phi)
    yl_m = np.conj(spin_weighted_y2(-2, inc, phi))
    return 0.5 * (ylm + yl_m) * h_plus + 0.5j * (ylm - yl_m) * h_cross


def reference_xyz(orbits, freqs, tf, inc, lam, beta, psi, phi):
    k = -np.array([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)])
    H = polarization_matrix(inc, lam, beta, psi, phi)
    L = orbits.armlength
    pos = {sc: np.asarray(orbits.get_pos(tf, sc)) for sc in (1, 2, 3)}

    def unit(a, b):
        d = pos[a] - pos[b]
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    arm = {1: unit(3, 2), 2: unit(1, 3), 3: unit(2, 1)}
    x = np.pi * freqs * L / C_SI

    def link(r, s, a, sign):
        n = arm[a]
        kn = n @ k
        nHn = np.einsum("fi,ij,fj->f", n, H, n)
        delay = np.exp(1j * x * (1.0 + (pos[r] + pos[s]) @ k / L))
        return 1j * x * np.sinc(x * (1.0 + sign * kn) / np.pi) * delay * nHn

    G = {
        (1, 2): link(1, 2, 3, -1), (2, 1): link(1, 2, 3, +1),
        (2, 3): link(2, 3, 1, -1), (3, 2): link(2, 3, 1, +1),
        (3, 1): link(3, 1, 2, -1), (1, 3): link(3, 1, 2, +1),
    }
    z = np.exp(2j * x)
    factor = 2j * np.sin(2 * x) * z
    X = factor * (G[2, 1] + z * G[1, 2] - G[3, 1] - z * G[1, 3])
    Y = factor * (G[3, 2] + z * G[2, 3] - G[1, 2] - z * G[2, 1])
    Z = factor * (G[1, 3] + z * G[3, 1] - G[2, 3] - z * G[3, 2])
    return np.stack([X, Y, Z])


def bbhx_xyz(orbits, freqs, tf, inc, lam, beta, psi, phi):
    response = LISATDIResponse(orbits=orbits, TDItag="XYZ", force_backend="cpu")
    shape = (1, 1, len(freqs))
    response(
        freqs.copy(), [inc], [lam], [beta], [psi], [phi], len(freqs),
        modes=[(2, 2)], phase=np.zeros(shape), tf=np.broadcast_to(tf, shape).copy(),
    )
    transfer = np.stack([response.transferL1, response.transferL2, response.transferL3])[:, 0, 0]
    return response, transfer * np.exp(1j * response.phase[0, 0])


@pytest.mark.parametrize(
    "inc,lam,beta,psi,phi",
    [(0.9, 0.4, 0.2, 0.4, 0.7), (2.1, 2.7, -0.6, 1.2, 4.0), (1.4, 5.1, 1.1, 2.9, 2.2)],
)
def test_link_response_matches_position_reference(inc, lam, beta, psi, phi):
    freqs = np.logspace(-4, np.log10(5e-2), 400)
    # sweep the orbit while sweeping frequency, as a chirp does
    tf = np.linspace(2e5, 3e7, len(freqs))
    response, bbhx = bbhx_xyz(EqualArmlengthOrbits(), freqs, tf, inc, lam, beta, psi, phi)
    reference = reference_xyz(response.orbits, freqs, tf, inc, lam, beta, psi, phi)
    for channel in range(3):
        error = np.linalg.norm(bbhx[channel] - reference[channel]) / np.linalg.norm(reference[channel])
        # the orbit file's normals include light-travel-time effects (~v/c) that
        # the position-difference reference omits; mislabelled arms give O(1)
        assert error < 2e-3
