"""
Conversion of scattering cross section from SANS (I(q), or rather, ds/dO) in absolute
units (cm-1)into SESANS correlation function G using a Hankel transformation, then converting
the SESANS correlation function into polarisation from the SESANS experiment

Everything is in units of metres except specified otherwise (NOT TRUE!!!)
Everything is in conventional units (nm for spin echo length)

Wim Bouwman (w.g.bouwman@tudelft.nl), June 2013

Three transforms are available. All compute

.. math::   G(\\xi) = \\frac{1}{2\\pi} \\int_0^\\infty q\\, J_0(q\\xi)\\, I(q)\\, dq

:class:`SesansTransform` evaluates it on a dense log-spaced grid of q whose
bounds and spacing are derived from the spin-echo lengths.
:class:`DHTSesansTransform` evaluates it with a digital filter, a quadrature
rule designed for Hankel transforms whose abscissae are a fixed table divided
by :math:`\\xi`. :class:`LibhankelSesansTransform` uses the same dense q grid
as :class:`SesansTransform` but delegates the integration to ``libhankel``
(optional dependency), supporting strategies including ``DHT_Key_201`` and
``Adaptive_DE_Ooura``. All three can act as a resolution object because all
know the q values they need before the model is evaluated;
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
    Spin-Echo SANS transform calculator, using a digital filter.

    Interchangeable with :class:`SesansTransform`: it exposes the same *q*,
    *q_calc* and *apply*, and takes the same arguments.

    A digital filter evaluates a Hankel transform as a weighted sum over
    abscissae that are a fixed table divided by the transform variable::

        G(xi) = sum(I(a[i]/xi) * (a[i]/xi) * w[i] / xi) / (2 pi)

    with *a* and *w* from :mod:`.sesans_filter`. Because the abscissae follow
    from the table and from xi alone, the q values at which the model is
    needed are known before the model is evaluated, which is what *q_calc*
    has to promise. Unlike :class:`SesansTransform` there is no grid to size,
    so there is no q_min, no q_max and no spacing to choose: the filter is
    valid over the whole range of q, and the number of model evaluations is
    the filter length times the number of spin-echo lengths rather than a
    single dense grid shared between them.

    That cost model is the trade. For the filter, work grows with the number
    of spin-echo lengths; for the dense grid it does not. On the SESANS
    datasets tested (40 to 84 spin-echo lengths) the filter is several times
    cheaper, but the two cross over somewhere above 150 spin-echo lengths.

    *z* is the spin-echo length in the original data units. The transform does
    not use it; the SasView GUI does, so it is stored unchanged.

    *SElength* (A) is the set of spin-echo lengths in the measured data.

    *lam* (A) is the wavelength, either a scalar or one value per spin-echo
    length (time-of-flight instruments vary it point by point).

    *zaccept* (1/A) is the maximum acceptance of the scattering vector in the
    spin-echo encoding direction.

    *Rmax* (A) is accepted for signature compatibility and ignored, as it is
    by :meth:`SesansTransform._set_hankel`. A filter has no grid to size.
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
        """
        Apply the SESANS transform to the computed I(q).

        Note that these are multiply-and-sum rather than *np.dot*. Numpy sends
        *np.dot* to BLAS, and for a reduction over a single vector threaded
        BLAS spends far longer starting and joining threads than it does on
        the arithmetic; measured here it costs about 2.9 ms against 0.015 ms
        for the multiply and sum, on 11457 points.
        """
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
        Quadrature weights for G(0) over the abscissae the filter already uses.

        *apply* returns G(xi) - G(0), and G(0) is the limit of the transform as
        xi goes to zero. A filter cannot be evaluated there, since its
        abscissae are a/xi, so G(0) needs its own quadrature.
        :class:`SesansTransform` gets it free from its dense grid
        (``H0 = dq/(2 pi) * q``, a rectangle rule); a filter has no such grid.

        But G(0) is a plain integral, not a Hankel transform:

        .. math::   G(0) = \\frac{1}{2\\pi} \\int_0^\\infty q\\, I(q)\\, dq
                         = \\frac{1}{2\\pi} \\int q^2 I(q)\\, d\\ln q

        and the union of the abscissae over all the spin-echo lengths already
        covers the q range densely, since each spin-echo length shifts the
        same table by a different 1/xi. So a trapezoid rule in log q over that
        union costs no extra model evaluations at all.

        The one case this does not cover well is a single spin-echo length,
        where the union is just the table itself, at about 19 points per
        decade.

        Like *H0* in :class:`SesansTransform`, these weights are not acceptance
        masked.
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
    Spin-Echo SANS transform calculator backed by ``libhankel``.

    Interchangeable with :class:`SesansTransform` and
    :class:`DHTSesansTransform`: exposes the same *q*, *q_calc* and *apply*
    and accepts the same constructor arguments.

    Like :class:`SesansTransform` it builds a dense log-spaced q grid in
    ``__init__`` so that *q_calc* is known before the model is evaluated.
    Unlike :class:`SesansTransform`, *apply* passes that grid as a tabulated
    form factor to ``libhankel.hankel_transform`` rather than doing a
    matrix multiply, which lets libhankel interpolate and integrate with its
    own quadrature.

    *strategy_name* and *strategy_params* are forwarded to
    ``libhankel.hankel_transform`` unchanged; see the libhankel documentation
    for the list of available strategies and their parameters.  The default
    ``DHT_Key_201`` works well for tabulated form factors; ``QWE_Key`` is
    an alternative that may be more accurate for smooth integrands but
    requires ``strategy_params = {"n_eval": ..., "eps_rel": ...}``.
    """

    q = None        # type: np.ndarray
    q_calc = None   # type: np.ndarray

    #: libhankel strategy name; override on the class before setting
    #: :data:`.direct_model.SESANS_TRANSFORM`.
    strategy_name = "Adaptive_DE_Ooura"   # type: str

    #: libhankel strategy parameters; override on the class before setting
    #: :data:`.direct_model.SESANS_TRANSFORM`.
    strategy_params = {"n_eval": 50, "eps_rel": 1e-6}  # type: dict

    #: log spacing for the dense q grid, same role as in SesansTransform.
    log_spacing = 1.0003   # type: float

    #: High-q tail type passed to ``libhankel.hankel_transform``: either
    #: ``"power_law"`` or ``"zero"``.  Override on the class before setting
    #: :data:`.direct_model.SESANS_TRANSFORM`.
    tail = "power_law"   # type: str

    #: Exponent for the power-law tail.  Porod regime (compact particles)
    #: gives exponent 4.  Set to 0 to let libhankel fit the exponent from
    #: the last few tabulated points (fails if those points are all zero).
    tail_exponent = 4.0   # type: float

    def __init__(self, z, SElength, lam, zaccept, Rmax):
        # type: (np.ndarray, np.ndarray, np.ndarray, float, float) -> None
        self.q = z
        self._SElength = None   # type: np.ndarray
        self._set_grid(SElength, lam, zaccept, Rmax)

    def apply(self, Iq):
        # type: (np.ndarray) -> np.ndarray
        """Apply the SESANS transform to the computed I(q)."""
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
        # G(0) = integral of q I(q) dq / (2 pi), no Bessel function needed.
        # Trapezoid rule in log q covers [q_min, q_max]. Add the missing
        # [0, q_min] piece: libhankel's QWE quadrature uses linear
        # extrapolation below q_min (so I(q) ≈ I(q_min)), giving a
        # rectangle contribution of q_min^2 * I(q_min) / 2.
        q_min = self.q_calc[0]
        G0 = (np.trapezoid(self.q_calc**2 * Iq, np.log(self.q_calc))
              + q_min**2 * Iq[0] / 2) / (2 * pi)
        return G - G0

    def _set_grid(self, SElength, lam, zaccept, Rmax):
        # type: (np.ndarray, np.ndarray, float, float) -> None
        """
        Build the dense log-spaced q grid and store the spin-echo lengths.

        *lam* and *zaccept* are accepted for signature compatibility but not
        used: the per-xi acceptance mask applied by :class:`SesansTransform`
        and :class:`DHTSesansTransform` cannot be expressed as a single
        tabulated I(q) passed to libhankel.  *Rmax* is likewise ignored
        because libhankel chooses its own integration range internally.
        """
        self._SElength = np.asarray(SElength, dtype=float)
        # q_min: a decade below 1/xi_max (the coarsest scale in the data).
        # q_max: two decades above 1/xi_min (the finest scale); libhankel
        # extrapolates beyond this with the power-law tail set in apply().
        q_min = 0.1 / self._SElength[-1]
        q_max = 100 / self._SElength[0]
        self.q_calc = np.exp(np.arange(np.log(q_min), np.log(q_max),
                                       np.log(self.log_spacing)))

