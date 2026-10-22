"""
:code:`bbhx.utils.timefrequency`:

Helpers for the time-frequency correspondence :math:`t_f(f)` of each harmonic,
shared by the waveform generator and the band-limiting taper.

"""

import numpy as np


def first_crossing_frequency(freqs, tf, t_bound, xp=np):
    """Frequency at which each row's :math:`t_f(f)` first rises through ``t_bound``.

    PhenomHM's :math:`t_f(f)` rises monotonically through the inspiral and merger
    but turns back down above the ringdown, so only the first upward crossing is
    meaningful; it is found by linear interpolation between the bracketing points.

    Args:
        freqs (xp.ndarray): Frequencies, shape ``(..., length)``.
        tf (xp.ndarray): :math:`t_f` at ``freqs``, same shape.
        t_bound (xp.ndarray): Times to cross, shape ``(...)``.
        xp (module): ``numpy`` or ``cupy``.

    Returns:
        tuple: ``(f, found)``, both of shape ``(...)``. ``f`` is NaN where ``found``
        is ``False`` (no upward crossing).

    """
    t_bound = xp.asarray(t_bound)[..., None]
    up = (tf[..., :-1] < t_bound) & (tf[..., 1:] >= t_bound)
    found = up.any(axis=-1)
    i = xp.argmax(up, axis=-1)[..., None]
    f1 = xp.take_along_axis(freqs, i, axis=-1)
    f2 = xp.take_along_axis(freqs, i + 1, axis=-1)
    t1 = xp.take_along_axis(tf, i, axis=-1)
    t2 = xp.take_along_axis(tf, i + 1, axis=-1)
    f = (f1 + (f2 - f1) * (t_bound - t1) / (t2 - t1))[..., 0]
    return xp.where(found, f, xp.nan), found


def planck_taper_fall(x, xp=np):
    """Planck taper falling from 1 at ``x <= 0`` to 0 at ``x >= 1``.

    Infinitely differentiable, so a frequency-domain taper of this shape leaks
    far less into the time domain than a cosine one.
    """
    xs = xp.clip(x, 1e-12, 1.0 - 1e-12)
    with np.errstate(over="ignore"):
        w = 1.0 / (1.0 + xp.exp(1.0 / (1.0 - xs) - 1.0 / xs))
    return xp.where(x <= 0.0, 1.0, xp.where(x >= 1.0, 0.0, w))
