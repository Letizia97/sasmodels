"""
Conversion of scattering cross section from SANS (I(q), or rather, ds/dO) in absolute
units (cm-1)into SESANS correlation function G using a Hankel transformation, then converting
the SESANS correlation function into polarisation from the SESANS experiment

Everything is in units of metres except specified otherwise (NOT TRUE!!!)
Everything is in conventional units (nm for spin echo length)

Wim Bouwman (w.g.bouwman@tudelft.nl), June 2013

Three transforms are available. All compute

.. math::   G(\\xi) = \\frac{1}{2\\pi} \\int_0^\\infty q\\, J_0(q\\xi)\\, I(q)\\, dq

:class:`SesansTransform` uses a dense log-spaced q grid.
:class:`DHTSesansTransform` uses a digital filter (abscissae = table / xi).
:class:`LibhankelSesansTransform` uses the same dense grid but delegates
integration to ``libhankel`` (optional dependency).
All three expose *q*, *q_calc* and *apply*;
:data:`.direct_model.SESANS_TRANSFORM` selects between them.
"""

import numpy as np  # type: ignore
from numpy import pi  # type: ignore
from scipy.special import j0

from .sesans_filter import ABSCISSAE, WEIGHTS


class SesansTransform:
    """
    Spin-Echo SANS transform calculator.  Similar to a resolution function,
    the SesansTransform object takes I(q) for the set of *q_calc* values and
    produces a transformed dataset

    *SElength* (A) is the set of spin-echo lengths in the measured data.

    *zaccept* (1/A) is the maximum acceptance of scattering vector in the spin
    echo encoding dimension (for ToF: Q of min(R) and max(lam)).

    *Rmax* (A) is the maximum size sensitivity; larger radius requires more
    computation time.
    """
    #: SElength from the data in the original data units; not used by transform
    #: but the GUI uses it, so make sure that it is present.
    q = None  # type: np.ndarray

    #: q values to calculate when computing transform
    q_calc = None  # type: np.ndarray

    # transform arrays
    _H = None   # type: np.ndarray
    _H0 = None  # type: np.ndarray

    def __init__(self, z, SElength, lam, zaccept, Rmax, log_spacing=1.0003):
        # type: (np.ndarray, float, float, float, float, float) -> None
        self.q = z
        self.log_spacing = log_spacing
        self._set_hankel(SElength, lam, zaccept, Rmax)

    def apply(self, Iq):
        # type: (np.ndarray) -> np.ndarray
        """
        Apply the SESANS transform to the computed I(q).
        """
        G0 = np.dot(self._H0, Iq)
        G = np.dot(self._H.T, Iq)
        P = G - G0
        return P

    def _set_hankel(self, SElength, lam, zaccept, Rmax):
        # type: (np.ndarray, float, float, float) -> None
        SElength = np.asarray(SElength)
        if len(SElength) == 1:
            # TODO: Do we care that this fails for xi = 0?
            q_min, q_max = 0.01 * 2*pi/SElength[-1], 10*2*pi / SElength[0]
        else:
            # TODO: Why does q_min depend on the number of correlation lengths?
            # TODO: Why does q_max depend on the correlation step size?
            q_min = 0.1 * 2*pi / (np.size(SElength) * SElength[-1])
            q_max = 2*pi / (SElength[1] - SElength[0])
        #print("Hankel xi, Qmin, Qmax", SElength[0], q_min, q_max, len(SElength))
        q = np.exp(np.arange(np.log(q_min), np.log(q_max),
                             np.log(self.log_spacing)))
        #print(q)

        dq = np.diff(q)
        dq = np.insert(dq, 0, dq[0])

        H0 = dq/(2*pi) * q

        H = np.outer(q, SElength)
        j0(H, out=H)
        H *= (dq * q / (2*pi)).reshape((-1, 1))

        reptheta = np.outer(q, lam/(2*pi))
        # Note: Using inplace update with reptheta => arcsin(reptheta).
        # When q L / 2 pi > 1 that means wavelength is too large to
        # reach that q value at any angle. These should produce theta = NaN
        # without any warnings.
        with np.errstate(invalid='ignore'):
            np.arcsin(reptheta, out=reptheta)
        # Reverse the condition to protect against NaN. We can't use
        # theta > zaccept since all comparisons with NaN return False.
        mask = ~(reptheta <= zaccept)
        H[mask] = 0
        #print("number of masked points", np.sum(mask))

        self.q_calc = q
        self._H, self._H0 = H, H0


class DHTSesansTransform:
    """
    Spin-Echo SANS transform using a digital Hankel filter.

    Interchangeable with :class:`SesansTransform`: same *q*, *q_calc*, *apply*
    and constructor arguments.

    Evaluates G(xi) = sum(I(a[i]/xi) * (a[i]/xi) * w[i] / xi) / (2 pi) with
    *a*, *w* from :mod:`.sesans_filter`.  Because the abscissae are a/xi, all
    required q values are known before the model is evaluated.

    Work scales with n_xi (vs. fixed cost for the dense grid); the two cross
    over somewhere above 150 spin-echo lengths.  *Rmax* is ignored.
    """
    #: SElength in the original data units; not used by the transform, but the
    #: GUI reads it, so make sure that it is present.
    q = None  # type: np.ndarray

    #: q values to calculate when computing transform
    q_calc = None  # type: np.ndarray

    # transform arrays
    _H = None   # type: np.ndarray
    _H0 = None  # type: np.ndarray

    def __init__(self, z, SElength, lam, zaccept, Rmax):
        # type: (np.ndarray, np.ndarray, np.ndarray, float, float) -> None
        self.q = z
        self._set_filter(SElength, lam, zaccept)

    def apply(self, Iq):
        # type: (np.ndarray) -> np.ndarray
        # multiply-and-sum rather than np.dot: threaded BLAS overhead (~2.9 ms)
        # far exceeds the arithmetic cost (~0.015 ms) for a single-vector reduction.
        G0 = np.sum(self._H0 * Iq)
        G = np.sum(np.reshape(Iq, self._H.shape) * self._H, axis=0)
        P = G - G0
        return P

    def _set_filter(self, SElength, lam, zaccept):
        # type: (np.ndarray, np.ndarray, float) -> None
        SElength = np.asarray(SElength, dtype=float)
        lam = np.broadcast_to(np.asarray(lam, dtype=float), SElength.shape)

        # The filter needs I(q) at abscissa/xi, one column of q per spin-echo
        # length. Both the abscissae and the weights carry a 1/xi.
        q = ABSCISSAE[:, None] / SElength[None, :]
        H = q * WEIGHTS[:, None] / SElength[None, :] / (2*pi)

        reptheta = q * lam[None, :] / (2*pi)
        # Note: as in SesansTransform._set_hankel. When q L / 2 pi > 1 the
        # wavelength is too large to reach that q at any angle, which should
        # give theta = NaN without any warnings.
        with np.errstate(invalid='ignore'):
            np.arcsin(reptheta, out=reptheta)
        # Reverse the condition to protect against NaN. We can't use
        # theta > zaccept since all comparisons with NaN return False.
        mask = ~(reptheta <= zaccept)
        H[mask] = 0

        self.q_calc = q.ravel()
        self._H, self._H0 = H, self._g0_weights(self.q_calc)

    @staticmethod
    def _g0_weights(q_calc):
        # type: (np.ndarray) -> np.ndarray
        """
        Trapezoid weights for G(0) = (1/2pi) int q^2 I(q) d(ln q).

        The filter cannot evaluate G(0) directly (abscissae are a/xi → inf as
        xi → 0), but the union of abscissae over all xi already covers q
        densely at no extra model evaluations.  Not acceptance-masked.
        """
        order = np.argsort(q_calc)
        log_q = np.log(q_calc[order])

        # Trapezoid weights: each point takes half of the interval on either
        # side. Repeated abscissae give a zero-width interval and drop out.
        step = np.diff(log_q)
        trapezoid = np.zeros_like(log_q)
        trapezoid[:-1] += step/2
        trapezoid[1:] += step/2

        weights = np.empty_like(q_calc)
        weights[order] = trapezoid * q_calc[order]**2 / (2*pi)
        return weights


class LibhankelSesansTransform:
    """
    Spin-Echo SANS transform backed by ``libhankel`` (optional dependency).

    Interchangeable with :class:`SesansTransform` and :class:`DHTSesansTransform`.
    Builds a dense log-spaced q grid in ``__init__`` (same approach as
    :class:`SesansTransform`), then passes it as a tabulated form factor to
    ``libhankel.hankel_transform`` at each *apply* call.  *strategy_name* and
    *strategy_params* are forwarded to ``libhankel.hankel_transform`` unchanged.
    """

    q = None        # type: np.ndarray
    q_calc = None   # type: np.ndarray

    #: Override on the class before setting :data:`.direct_model.SESANS_TRANSFORM`.
    strategy_name = "Adaptive_DE_Ooura"              # type: str
    strategy_params = {"n_eval": 50, "eps_rel": 1e-6}  # type: dict
    log_spacing = 1.003                              # type: float
    tail = "power_law"                               # type: str
    #: Porod exponent; 0 lets libhankel fit it from the last few points.
    tail_exponent = 4.0                              # type: float

    def __init__(self, z, SElength, lam, zaccept, Rmax):
        # type: (np.ndarray, np.ndarray, np.ndarray, float, float) -> None
        self.q = z
        self._SElength = None   # type: np.ndarray
        self._set_grid(SElength, lam, zaccept, Rmax)

    def apply(self, Iq):
        # type: (np.ndarray) -> np.ndarray
        """Apply the SESANS transform to the computed I(q)."""
        # Deferred import: libhankel is optional; a top-level import would
        # prevent loading this module when libhankel is not installed.
        from libhankel import hankel_transform
        q_dict = {
            'q': self.q_calc,
            'f': Iq,
            'interp_type': 'linear',
            'tail': self.tail,
            'exponent': self.tail_exponent,
        }
        G = np.asarray(hankel_transform(
            0, q_dict, self._SElength, [], self.strategy_name, self.strategy_params,
        )) / (2 * pi)
        # G(0) = (1/2pi) int q I(q) dq: trapezoid in log q plus a rectangle
        # correction for [0, q_min] (linear extrapolation ≈ I(q_min)).
        q_min = self.q_calc[0]
        G0 = (np.trapezoid(self.q_calc**2 * Iq, np.log(self.q_calc))
              + q_min**2 * Iq[0] / 2) / (2 * pi)
        return G - G0

    def _set_grid(self, SElength, lam, zaccept, Rmax):
        # type: (np.ndarray, np.ndarray, float, float) -> None
        # lam, zaccept, Rmax ignored: the per-xi acceptance mask cannot be
        # expressed as a single tabulated I(q), and libhankel sets its own range.
        self._SElength = np.asarray(SElength, dtype=float)
        q_min = 0.1 / self._SElength[-1]
        q_max = 100 / self._SElength[0]
        self.q_calc = np.exp(np.arange(np.log(q_min), np.log(q_max),
                                       np.log(self.log_spacing)))

