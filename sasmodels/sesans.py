"""
Conversion of scattering cross section from SANS (I(q), or rather, ds/dO) in absolute
units (cm-1)into SESANS correlation function G using a Hankel transformation, then converting
the SESANS correlation function into polarisation from the SESANS experiment

Everything is in units of metres except specified otherwise (NOT TRUE!!!)
Everything is in conventional units (nm for spin echo length)

Wim Bouwman (w.g.bouwman@tudelft.nl), June 2013

Two transforms are available. Both compute

.. math::   G(\\xi) = \\frac{1}{2\\pi} \\int_0^\\infty q\\, J_0(q\\xi)\\, I(q)\\, dq

:class:`SesansTransform` evaluates it on a dense log-spaced grid of q whose
bounds and spacing are derived from the spin-echo lengths.
:class:`DHTSesansTransform` evaluates it with a digital filter, a quadrature
rule designed for Hankel transforms whose abscissae are a fixed table divided
by :math:`\\xi`. Both can act as a resolution object because both know the q
values they need before the model is evaluated;
:data:`.direct_model.SESANS_TRANSFORM` selects between them.
"""


import unittest

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


# ==========================================================================


#: Spin-echo lengths (A) and wavelength (A) for the tests, roughly matching a
#: real measurement.
TEST_SELENGTH = np.logspace(2, np.log10(5000), 40)
TEST_LAMBDA = 6.0
#: Width (A) of the test scatterer. See :func:`gaussian_correlation`.
TEST_WIDTH = 1000.0
#: Large enough that no point is acceptance masked.
NO_MASKING = 10.0


def gaussian_correlation(q, width=TEST_WIDTH):
    # type: (np.ndarray, float) -> np.ndarray
    """
    A scattering curve whose SESANS transform is known in closed form.

    For :math:`I(q) = \\exp(-w^2 q^2/2)`,

    .. math::

        G(\\xi) = \\frac{1}{2\\pi} \\int_0^\\infty q J_0(q\\xi) I(q)\\,dq
                = \\frac{1}{2\\pi w^2} \\exp(-\\xi^2/2w^2)

    so both G(xi) and G(0) are available to compare a transform against.
    """
    return np.exp(-(width*q)**2/2)


def gaussian_polarisation(SElength, width=TEST_WIDTH):
    # type: (np.ndarray, float) -> np.ndarray
    """
    P(xi) = G(xi) - G(0) for :func:`gaussian_correlation`.
    """
    return (np.exp(-(SElength/width)**2/2) - 1)/(2*pi*width**2)


class DHTSesansTransformTest(unittest.TestCase):
    """
    Check the digital filter transform against a closed form, and against the
    dense grid transform it is an alternative to.
    """
    def setUp(self):
        self.transform = DHTSesansTransform(
            TEST_SELENGTH, TEST_SELENGTH, TEST_LAMBDA, NO_MASKING, Rmax=None)

    def test_filter_table(self):
        """Abscissae and weights are usable as a quadrature rule."""
        self.assertEqual(ABSCISSAE.shape, WEIGHTS.shape)
        self.assertTrue(np.all(np.isfinite(ABSCISSAE)))
        self.assertTrue(np.all(np.isfinite(WEIGHTS)))
        self.assertTrue(np.all(ABSCISSAE > 0))
        self.assertTrue(np.all(np.diff(ABSCISSAE) > 0))

    def test_q_calc(self):
        """
        The model is evaluated once per (abscissa, spin-echo length) pair, and
        apply returns one value per spin-echo length.
        """
        n_xi = len(TEST_SELENGTH)
        self.assertEqual(self.transform.q_calc.shape, (len(ABSCISSAE)*n_xi,))
        self.assertTrue(np.all(self.transform.q_calc > 0))
        Iq = gaussian_correlation(self.transform.q_calc)
        self.assertEqual(self.transform.apply(Iq).shape, (n_xi,))

    def test_analytic(self):
        """P(xi) matches the closed form for a Gaussian scatterer."""
        P = self.transform.apply(gaussian_correlation(self.transform.q_calc))
        target = gaussian_polarisation(TEST_SELENGTH)
        # Relative to P(inf) = -G(0), which is the scale P is measured on;
        # P itself passes through zero at xi = 0 so a pure relative
        # comparison is not meaningful near the short spin-echo lengths.
        scale = 1/(2*pi*TEST_WIDTH**2)
        self.assertLess(np.max(np.abs(P - target))/scale, 1e-9)

    def test_g0(self):
        """
        G(0) matches the closed form.

        This is the piece a filter cannot supply and _g0_weights has to make
        up, so check it on its own rather than only through P.
        """
        Iq = gaussian_correlation(self.transform.q_calc)
        G0 = np.sum(self.transform._H0*Iq)
        self.assertLess(abs(G0 - 1/(2*pi*TEST_WIDTH**2))*2*pi*TEST_WIDTH**2,
                        1e-10)

    def test_single_spin_echo_length(self):
        """
        One spin-echo length works too.

        This is the thin case for G(0): the union of abscissae is just the
        filter table, at about 19 points per decade, where 40 spin-echo
        lengths interleave to 8040 points. It is accurate here, but a smooth
        Gaussian is the easy case for a trapezoid rule and this test should
        not be read as saying the sparse case is always fine.
        """
        SElength = np.array([TEST_WIDTH])
        transform = DHTSesansTransform(
            SElength, SElength, TEST_LAMBDA, NO_MASKING, Rmax=None)
        P = transform.apply(gaussian_correlation(transform.q_calc))
        target = gaussian_polarisation(SElength)
        scale = 1/(2*pi*TEST_WIDTH**2)
        self.assertLess(np.max(np.abs(P - target))/scale, 1e-9)

    def test_acceptance_mask(self):
        """
        Neutrons scattered beyond the analyser acceptance contribute nothing.

        Masking is applied to the G(xi) weights only, matching
        SesansTransform, where H0 is not masked either.
        """
        masked = DHTSesansTransform(
            TEST_SELENGTH, TEST_SELENGTH, TEST_LAMBDA, 0.01, Rmax=None)
        self.assertTrue(np.any(masked._H == 0))
        self.assertTrue(np.all(masked._H0 == self.transform._H0))
        with np.errstate(invalid='ignore'):
            theta = np.arcsin(masked.q_calc*TEST_LAMBDA/(2*pi))
        expected = ~(theta <= 0.01)
        self.assertTrue(np.all(masked._H.ravel()[expected] == 0))

    def test_interchangeable_with_dense_grid(self):
        """
        The two transforms take the same arguments and agree on a smooth
        integrand, to the accuracy the dense grid can manage.

        On this integrand the whole of the difference between them is the
        dense grid's own error: it differs from the closed form by as much as
        it differs from the filter, on five times as many model evaluations.
        That is a property of a scatterer chosen to have an exact transform,
        not a general claim -- for models where q I(q) is not integrable, or
        where the dense grid's q_min and q_max happen to suit the sample,
        the ordering can go the other way.
        """
        # SesansTransform needs one wavelength per spin-echo length: its
        # np.outer(q, lam) has to come out the same shape as H. The filter
        # broadcasts a scalar, as tested above.
        lam = np.full_like(TEST_SELENGTH, TEST_LAMBDA)
        dense = SesansTransform(
            TEST_SELENGTH, TEST_SELENGTH, lam, NO_MASKING, Rmax=None)
        P_dense = dense.apply(gaussian_correlation(dense.q_calc))
        P_filter = self.transform.apply(
            gaussian_correlation(self.transform.q_calc))
        self.assertEqual(P_dense.shape, P_filter.shape)
        self.assertLess(len(self.transform.q_calc), len(dense.q_calc))
        scale = 1/(2*pi*TEST_WIDTH**2)
        target = gaussian_polarisation(TEST_SELENGTH)
        difference = np.max(np.abs(P_dense - P_filter))/scale
        dense_error = np.max(np.abs(P_dense - target))/scale
        self.assertLess(difference, 1e-3)
        self.assertAlmostEqual(difference, dense_error, places=6)
