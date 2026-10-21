"""
Tests for the SESANS transforms in :mod:`.sesans`.

Closed-form tests against a Gaussian scatterer, then all three transforms run
against each other on the ``.ses`` files in ``example/``, refereed by
:func:`reference_polarisation`. Measured-data tests need *sasdata* and a
compiler and skip without either.  ``README-sesans-transform.md`` has the
measured numbers and the reasoning.
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
    """I(q) = exp(-w^2 q^2/2), whose Hankel transform is known 
    in closed form."""
    return np.exp(-(width*q)**2/2)


def gaussian_polarisation(SElength, width=TEST_WIDTH):
    # type: (np.ndarray, float) -> np.ndarray
    """P(xi) = G(xi) - G(0) for :func:`gaussian_correlation`."""
    return (np.exp(-(SElength/width)**2/2) - 1)/(2*pi*width**2)


class DigitalFiltersClosedFormTest(unittest.TestCase):
    """Check DHTSesansTransform against a Gaussian scatterer 
    with an exact answer."""
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
        """G(0) matches the closed form (the one non-published part of the filter)."""
        Iq = gaussian_correlation(self.transform.q_calc)
        G0 = np.sum(self.transform._H0*Iq)
        self.assertLess(abs(G0 - 1/(2*pi*TEST_WIDTH**2))/TEST_SCALE, 1e-10)


# The measured data below. Everything above needs only numpy and scipy;
# everything below needs sasdata, a compiler, and example/.


#: The measured data lives beside the package rather than inside it, so it is
#: only there in a source checkout.
EXAMPLE_DIR = Path(__file__).parent.parent / "example"

#: A ``.ses`` file with a model to evaluate against it.
Measurement = namedtuple("Measurement", "filename model_name pars")

#: Every SESANS file in ``example/``: 25–120 spin-echo lengths, fixed and
#: scanning wavelength, linear and piecewise-linear xi spacing.
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
    """Set SESANS_TRANSFORM for the duration; 
    must be set before DirectModel is built."""
    saved = direct_model.SESANS_TRANSFORM
    direct_model.SESANS_TRANSFORM = transform
    try:
        yield
    finally:
        direct_model.SESANS_TRANSFORM = saved


def load(measurement):
    """Load the data file; import deferred so missing 
    sasdata causes a skip, not an error."""
    from .data import load_data
    return load_data(str(EXAMPLE_DIR / measurement.filename))


def make_calculator(measurement, transform):
    """DirectModel for *measurement* using *transform*, 
    exercising the full SasView path."""
    data = load(measurement)
    with using_transform(transform):
        return DirectModel(data, load_model(measurement.model_name))


def reference_polarisation(measurement, n_linear=200001, n_log=200001):
    """
    P(xi) by brute-force trapezoid integration, sharing no code 
    with either transform.

    G(xi): linear q grid, fine enough to resolve J0 at the longest xi, masked
    at the acceptance edge.  G(0): log q grid over the full q range.
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
    """max|P - reference| / max|reference|; max-normalised 
    because P passes through zero."""
    return np.max(np.abs(P - reference))/np.max(np.abs(reference))


#: Error tolerances (fraction of max|P|) for each transform on each file.
#: Roughly 2× the measured values so an improvement still passes.
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
    """All three transforms over every measured data set in ``example/``."""
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
        """Doubling the reference grid does not move it."""
        for measurement in MEASUREMENTS:
            with self.subTest(measurement.filename):
                coarse = reference_polarisation(measurement, n_linear=100001)
                fine = reference_polarisation(measurement, n_linear=200001)
                self.assertLess(relative_error(coarse, fine), 1e-5)

    def test_error_against_reference(self):
        """Each transform stays within its expected error bound on every file."""
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


if __name__ == "__main__":
    compare()
