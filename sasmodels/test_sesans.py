"""
Tests for the SESANS transforms in :mod:`.sesans`.

:class:`.sesans.SesansTransform` evaluates the SESANS Hankel transform on a
dense log-spaced grid of q; :class:`.sesans.DHTSesansTransform` evaluates it
with a digital filter; :class:`.sesans.LibhankelSesansTransform` delegates to
``libhankel`` (optional dependency). First closed-form tests against a Gaussian
scatterer, then all three run against each other on the ``.ses`` files in
``example/``, refereed by :func:`reference_polarisation`.

``README-sesans-transform.md`` has the measured numbers and the reasoning
behind them; this file only guards them.

The measured-data tests need *sasdata* and a compiler, and skip without
either. Their model parameters are the starting values from the fit scripts
in ``example/``, not fitted values: what is compared is two ways of
evaluating one integral for one I(q).
"""

import time
import unittest
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import numpy as np  # type: ignore
from numpy import pi  # type: ignore
from scipy.special import j0

from . import direct_model, sesans
from .core import load_model
from .direct_model import DirectModel, call_kernel
from .sesans import DHTSesansTransform


try:
    from libhankel import hankel_transform as _hankel_transform  # noqa: F401
    _LIBHANKEL_AVAILABLE = True
except ImportError:
    _LIBHANKEL_AVAILABLE = False

#: Spin-echo lengths (A), wavelength (A) and scatterer width (A) for the
#: closed-form tests, roughly matching a real measurement. The acceptance is
#: large enough that no point is masked.
TEST_SELENGTH = np.logspace(2, np.log10(5000), 40)
TEST_LAMBDA = 6.0
TEST_WIDTH = 1000.0
NO_MASKING = 10.0
#: P is measured on the scale of P(inf) = -G(0); it passes through zero at
#: xi = 0, so a pointwise relative error says nothing at short xi.
TEST_SCALE = 1/(2*pi*TEST_WIDTH**2)


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
    """P(xi) = G(xi) - G(0) for :func:`gaussian_correlation`."""
    return (np.exp(-(SElength/width)**2/2) - 1)/(2*pi*width**2)


class DigitalFiltersClosedFormTest(unittest.TestCase):
    """
    Check the digital filter against a case with an exact answer.

    A Gaussian is the smoothest integrand either rule will ever see, so this
    says the filter is wired up correctly and little else. The measured-data
    tests below are the ones to argue from.
    """
    def setUp(self):
        self.transform = DHTSesansTransform(
            TEST_SELENGTH, TEST_SELENGTH, TEST_LAMBDA, NO_MASKING, Rmax=None)

    def test_analytic(self):
        """P(xi) matches the closed form for a Gaussian scatterer."""
        P = self.transform.apply(gaussian_correlation(self.transform.q_calc))
        target = gaussian_polarisation(TEST_SELENGTH)
        self.assertEqual(P.shape, TEST_SELENGTH.shape)
        self.assertLess(np.max(np.abs(P - target))/TEST_SCALE, 1e-9)

    def test_g0(self):
        """
        G(0) matches the closed form.

        The piece a filter cannot supply -- its abscissae are a/xi, so there
        is nothing to evaluate at xi = 0 -- and the one part of
        DHTSesansTransform that is not a published rule, so check it on its
        own rather than only through P.
        """
        Iq = gaussian_correlation(self.transform.q_calc)
        G0 = np.sum(self.transform._H0*Iq)
        self.assertLess(abs(G0 - 1/(2*pi*TEST_WIDTH**2))/TEST_SCALE, 1e-10)


@unittest.skipUnless(_LIBHANKEL_AVAILABLE, "libhankel is not installed")
class LibhankelClosedFormTest(unittest.TestCase):
    """
    Check LibhankelSesansTransform against a case with an exact answer.

    Mirrors ClosedFormTest but for the libhankel-backed transform.  The
    default strategy (QWE_Key, eps_rel=1e-9) on a smooth Gaussian should
    match the analytic answer to a few parts in 1e-8.
    """
    def setUp(self):
        self.transform = sesans.LibhankelSesansTransform(
            TEST_SELENGTH, TEST_SELENGTH, TEST_LAMBDA, NO_MASKING, Rmax=None)

    def test_analytic(self):
        """P(xi) matches the closed form for a Gaussian scatterer."""
        P = self.transform.apply(gaussian_correlation(self.transform.q_calc))
        target = gaussian_polarisation(TEST_SELENGTH)
        self.assertEqual(P.shape, TEST_SELENGTH.shape)
        self.assertLess(np.max(np.abs(P - target))/TEST_SCALE, 1e-7)


# The measured data below. Everything above needs only numpy and scipy;
# everything below needs sasdata, a compiler, and example/.


#: The measured data lives beside the package rather than inside it, so it is
#: only there in a source checkout.
EXAMPLE_DIR = Path(__file__).parent.parent / "example"

#: A ``.ses`` file with a model to evaluate against it.
Measurement = namedtuple("Measurement", "filename model_name pars")

#: Every SESANS file in ``example/``. The spread matters more than the
#: individual entries: 25 to 120 spin-echo lengths, one instrument that fixes
#: the wavelength and one that scans it, and spin-echo lengths that are
#: linearly spaced or piecewise linear, never log-spaced.
MEASUREMENTS = [
    Measurement(
        "sphere.ses", "sphere",
        dict(scale=0.09, background=0.0, sld=7.0, radius=1000.,
             sld_solvent=1.0)),
    Measurement(
        "spheres2micron.ses", "sphere",
        dict(scale=0.0782, background=0.0, sld=1.41, radius=10000.,
             sld_solvent=2.70)),
    # The only one with a structure factor.
    Measurement(
        "core_shell.ses", "core_shell_sphere@hardsphere",
        dict(scale=0.2475, background=0.0, sld_core=1.05, sld_shell=2.733,
             sld_solvent=2.88, radius=730., thickness=20., volfraction=0.45)),
    Measurement(
        "se008724_01.ses", "sphere",
        dict(scale=0.09, background=0.0, sld=1.41, radius=5000.,
             sld_solvent=2.70)),
    Measurement(
        "se008731_01_40pcorr.ses", "sphere",
        dict(scale=0.09, background=0.0, sld=1.41, radius=10000.,
             sld_solvent=2.70)),
    # Time of flight: the wavelength scans 2.2 to 11.8 A across the 25 points.
    Measurement(
        "SiO2_100pc_H2O_0pc_D2O.ses", "sphere",
        dict(scale=0.09, background=0.0, sld=3.47, radius=500.,
             sld_solvent=6.36)),
]


def environment_problem():
    # type: () -> str
    """Why the measured-data tests cannot run here, or "" if they can."""
    if not EXAMPLE_DIR.is_dir():
        return "no example/ directory; needs a source checkout"
    try:
        import sasdata  # pylint: disable=unused-import
    except ImportError:
        return "sasdata is not installed"
    try:
        load_model("sphere")
    except Exception as exc:  # pylint: disable=broad-except
        return "cannot build a model kernel: %s" % exc
    return ""


SKIP_REASON = environment_problem()


@contextmanager
def using_transform(transform):
    """
    Run the body with *transform* as the SESANS transform.

    :func:`.direct_model._make_sesans_transform` reads the module-level
    :data:`.direct_model.SESANS_TRANSFORM` when the data is interpreted, so
    this has to be set before DirectModel is built, not before it is called.
    """
    saved = direct_model.SESANS_TRANSFORM
    direct_model.SESANS_TRANSFORM = transform
    try:
        yield
    finally:
        direct_model.SESANS_TRANSFORM = saved


def load(measurement):
    """Load the data, with the metadata the transform needs."""
    # Imported here rather than at the top: sasmodels.data imports cleanly
    # without sasdata but load_data raises, and these tests should skip
    # rather than error in that case.
    from .data import load_data
    return load_data(str(EXAMPLE_DIR / measurement.filename))


def make_calculator(measurement, transform):
    """
    A DirectModel for *measurement* that will use *transform*.

    The whole SasView path with only the transform swapped, so the two are
    compared as they are actually used rather than as bare quadrature rules.
    """
    data = load(measurement)
    with using_transform(transform):
        return DirectModel(data, load_model(measurement.model_name))


def reference_polarisation(measurement, n_linear=200001, n_log=200001):
    """
    P(xi) = G(xi) - G(0) by brute force, to judge both transforms against.

    Neither transform can referee the other and measured data has no closed
    form, so evaluate the integral directly: a trapezoid rule on a linear
    grid of q fine enough to resolve J0(q xi) at the longest spin-echo
    length, out to the acceptance edge where the integrand is cut off anyway.
    G(0) has no oscillation, so it gets a trapezoid rule in log q over a
    range wide enough to hold the whole of q I(q), unmasked, matching what
    both transforms do with it.

    Slow, useless for fitting, and shares no code with either transform,
    which is the point.
    """
    data = load(measurement)
    SElength = data.x
    wavelength = np.broadcast_to(
        np.asarray(data.source.wavelength, dtype=float), SElength.shape)
    model = load_model(measurement.model_name)

    # Beyond q = 2 pi sin(theta_max)/lam the analyser does not accept the
    # scattered neutron, so the grid need not go any further.
    q_edge = 2*pi*np.sin(data.sample.zacceptance[0])/wavelength
    q = np.linspace(0, np.max(q_edge), n_linear)[1:]  # J0 grid, skip q = 0
    Iq = call_kernel(model.make_kernel([q]), measurement.pars)
    G = np.empty_like(SElength)
    for i, (xi, edge) in enumerate(zip(SElength, q_edge)):
        accepted = q <= edge
        G[i] = np.trapezoid((q*j0(q*xi)*Iq)[accepted], q[accepted])/(2*pi)

    q_wide = np.logspace(-9, 2, n_log)
    Iq_wide = call_kernel(model.make_kernel([q_wide]), measurement.pars)
    # q I(q) dq = q^2 I(q) dlog(q)
    G0 = np.trapezoid(q_wide**2*Iq_wide, np.log(q_wide))/(2*pi)

    return G - G0


def relative_error(P, reference):
    """
    Largest difference, over max|reference|.

    P passes through zero at short xi, so a pointwise relative error says
    nothing there; max|P| is the scale the whole curve is read on.
    """
    return np.max(np.abs(P - reference))/np.max(np.abs(reference))


#: Largest error against :func:`reference_polarisation` expected of each
#: transform, as a fraction of max|P|. Upper bounds, roughly twice what is
#: measured today, so a transform that improves still passes. The dense
#: grid's numbers are the interesting ones: the files it does worst on are
#: the ones with the widest spread of spin-echo lengths. core_shell.ses is
#: the filter's worst file because it is the only one with a structure
#: factor, whose peak the fixed abscissae barely resolve.
EXPECTED_ERROR = {
    #                              dense    filter  libhankel
    "sphere.ses":                  (2e-2,   1e-3,   5e-4),
    "spheres2micron.ses":          (1e-3,   1e-3,   1e-3),
    "core_shell.ses":              (1e-1,   1e-2,   1e-2),
    "se008724_01.ses":             (1e-3,   1e-3,   1e-3),
    "se008731_01_40pcorr.ses":     (1e-3,   1e-3,   1e-3),
    # libhankel does not apply the acceptance mask, so the time-of-flight
    # file (varying wavelength, strong masking) is noticeably worse than
    # the filter but still well within 1e-2.
    "SiO2_100pc_H2O_0pc_D2O.ses":  (2e-1,   1e-3,   1e-2),
}


@unittest.skipIf(SKIP_REASON, SKIP_REASON)
class TransformComparisonTest(unittest.TestCase):
    """
    Run both transforms over every measured data set in ``example/``.

    Computed once for the class: building the six models and running the
    reference takes a few seconds.
    """
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for measurement in MEASUREMENTS:
            row = {'reference': reference_polarisation(measurement)}
            for strategy, transform in (
                ('dense',    sesans.SesansTransform),
                ('filter',   sesans.DHTSesansTransform),
                ('libhankel', sesans.LibhankelSesansTransform),
            ):
                if strategy == 'libhankel' and not _LIBHANKEL_AVAILABLE:
                    continue
                row[strategy] = make_calculator(
                    measurement, transform)(**measurement.pars)
            cls.results[measurement.filename] = row

    def test_reference_converged(self):
        """
        Refining the reference does not move it.

        If it did it would not be a reference, and nothing else here would
        mean anything.
        """
        for measurement in MEASUREMENTS:
            with self.subTest(measurement.filename):
                coarse = reference_polarisation(measurement, n_linear=100001)
                fine = reference_polarisation(measurement, n_linear=200001)
                self.assertLess(relative_error(coarse, fine), 1e-5)

    def test_error_against_reference(self):
        """
        Both stay within the error expected of them.

        The filter holds 1e-3 of the signal everywhere. The dense grid does
        too on three of the six files and is one to two orders of magnitude
        worse on the other three.
        """
        for filename, result in self.results.items():
            with self.subTest(filename):
                reference = result['reference']
                for strategy, limit in zip(('dense', 'filter', 'libhankel'),
                                           EXPECTED_ERROR[filename]):
                    if strategy not in result or limit is None:
                        continue
                    self.assertLess(
                        relative_error(result[strategy], reference), limit,
                        "%s transform on %s" % (strategy, filename))


def compare(repeats=3):
    # type: (int) -> None
    """
    Print the comparison table. Run with ``python -m sasmodels.test_sesans``.
    """
    if SKIP_REASON:
        print("cannot run:", SKIP_REASON)
        return
    strategies = [('dense', sesans.SesansTransform),
                  ('filter', sesans.DHTSesansTransform)]
    if _LIBHANKEL_AVAILABLE:
        strategies.append(('libhankel', sesans.LibhankelSesansTransform))
    names = [s for s, _ in strategies]
    header = ("%-28s %4s" % ("file", "n")
              + "".join(" %8s" % ("nq " + n) for n in names)
              + "".join(" %7s" % ("t " + n) for n in names)
              + "".join(" %10s" % ("err " + n) for n in names))
    print(header)
    print("-"*len(header))
    for measurement in MEASUREMENTS:
        reference = reference_polarisation(measurement)
        row = {}
        for strategy, transform in strategies:
            # Time the call, not the setup: a fitter builds the calculator
            # once and then calls it thousands of times.
            calculator = make_calculator(measurement, transform)
            P = calculator(**measurement.pars)
            start = time.perf_counter()
            for _ in range(repeats):
                calculator(**measurement.pars)
            row[strategy] = (len(calculator.resolution.q_calc),
                             (time.perf_counter() - start)/repeats,
                             relative_error(P, reference))
        print(("%-28s %4d" % (measurement.filename, len(reference)))
              + "".join(" %8d" % row[n][0] for n in names)
              + "".join(" %6.1fms" % (row[n][1]*1e3) for n in names)
              + "".join(" %10.2e" % row[n][2] for n in names))


if __name__ == "__main__":
    compare()
