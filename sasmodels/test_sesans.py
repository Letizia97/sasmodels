"""
Tests for the SESANS transforms in :mod:`.sesans`.

:class:`.sesans.SesansTransform` evaluates the SESANS Hankel transform on a
dense log-spaced grid of q; :class:`.sesans.DHTSesansTransform` evaluates it
with a digital filter. First a closed form on a Gaussian scatterer, which
says the filter is wired up correctly and little else; then the two run
against each other on the ``.ses`` files in ``example/``, refereed by
:func:`reference_polarisation`. ``README-sesans-transform.md`` has the
measured numbers and the reasoning behind them.

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
from .sesans import SesansTransform, DHTSesansTransform

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


class ClosedFormTest(unittest.TestCase):
    """
    Check the digital filter against a case with an exact answer.
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

    def test_indistinguishable_from_dense_grid(self):
        """
        On a smooth integrand the two agree, and agreement says nothing.

        The whole of the difference between them here is the dense grid's own
        error: it differs from the closed form by as much as it differs from
        the filter. That is why the measured-data tests below exist -- on a
        Gaussian, agreement and accuracy cannot be told apart.
        """
        # SesansTransform needs one wavelength per spin-echo length: its
        # np.outer(q, lam) has to come out the same shape as H. The filter
        # broadcasts a scalar.
        lam = np.full_like(TEST_SELENGTH, TEST_LAMBDA)
        dense = SesansTransform(
            TEST_SELENGTH, TEST_SELENGTH, lam, NO_MASKING, Rmax=None)
        P_dense = dense.apply(gaussian_correlation(dense.q_calc))
        P_filter = self.transform.apply(
            gaussian_correlation(self.transform.q_calc))
        self.assertLess(len(self.transform.q_calc), len(dense.q_calc))
        target = gaussian_polarisation(TEST_SELENGTH)
        difference = np.max(np.abs(P_dense - P_filter))/TEST_SCALE
        dense_error = np.max(np.abs(P_dense - target))/TEST_SCALE
        self.assertLess(difference, 1e-3)
        self.assertAlmostEqual(difference, dense_error, places=6)


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


def evaluate(measurement, transform):
    """P(xi) under *transform*, plus the transform itself."""
    calculator = make_calculator(measurement, transform)
    return calculator(**measurement.pars), calculator.resolution


def reference_transform(measurement, n_linear=200001, n_log=200001):
    """
    G(xi) and G(0) by brute force, to judge both transforms against.

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

    return G, G0


def reference_polarisation(measurement, n_linear=200001, n_log=200001):
    """G(xi) - G(0), the part directly comparable to what *apply* returns."""
    G, G0 = reference_transform(measurement, n_linear, n_log)
    return G - G0


def transform_parts(measurement, transform):
    """
    G(xi) and G(0) separately, to see which half carries the error.

    *apply* only returns the difference, but the halves come from different
    machinery. G(0) is recovered the way both classes compute it, by summing
    the unmasked H0 weights.
    """
    P, resolution = evaluate(measurement, transform)
    model = load_model(measurement.model_name)
    Iq = call_kernel(model.make_kernel([resolution.q_calc]), measurement.pars)
    G0 = np.sum(resolution._H0*Iq)
    return P + G0, G0


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
#: the ones with the widest spread of spin-echo lengths.
EXPECTED_ERROR = {
    #                              dense    filter
    "sphere.ses":                  (2e-2,   1e-3),
    "spheres2micron.ses":          (1e-3,   1e-3),
    # The filter's worst file, by a factor of sixteen; see
    # test_filter_barely_resolves_a_structure_factor.
    "core_shell.ses":              (1e-1,   1e-2),
    "se008724_01.ses":             (1e-3,   1e-3),
    "se008731_01_40pcorr.ses":     (1e-3,   1e-3),
    "SiO2_100pc_H2O_0pc_D2O.ses":  (2e-1,   1e-3),
}

#: A difference this big, as a fraction of max|P|, means the two are
#: answering the question differently rather than rounding differently.
DISAGREEMENT = 1e-3


@unittest.skipIf(SKIP_REASON, SKIP_REASON)
class TransformComparisonTest(unittest.TestCase):
    """
    Run both transforms over every measured data set in ``example/``.

    Computed once for the class: building the six models and running the
    reference takes a few seconds, and every test looks at the same numbers
    from a different angle.
    """
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for measurement in MEASUREMENTS:
            P_dense, dense = evaluate(measurement, sesans.SesansTransform)
            P_filter, filt = evaluate(measurement, sesans.DHTSesansTransform)
            cls.results[measurement.filename] = {
                'measurement': measurement,
                'reference': reference_polarisation(measurement),
                'dense': (P_dense, dense),
                'filter': (P_filter, filt),
            }

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
                for strategy, limit in zip(('dense', 'filter'),
                                           EXPECTED_ERROR[filename]):
                    self.assertLess(
                        relative_error(result[strategy][0], reference), limit,
                        "%s transform on %s" % (strategy, filename))

    def test_filter_is_closer_where_they_disagree(self):
        """
        Where the two differ by more than rounding, the filter is right.

        The question the closed-form test cannot answer. On measured data the
        two come apart, and on every file where they do it is the dense grid
        that has moved away from the reference. There has to be at least one
        such file or the test is vacuous.
        """
        disagreements = 0
        for filename, result in self.results.items():
            with self.subTest(filename):
                reference = result['reference']
                P_dense, P_filter = result['dense'][0], result['filter'][0]
                if relative_error(P_dense, P_filter) < DISAGREEMENT:
                    continue
                disagreements += 1
                self.assertLess(relative_error(P_filter, reference),
                                relative_error(P_dense, reference),
                                "on %s" % filename)
        self.assertGreater(disagreements, 0)

    def test_refining_the_dense_grid_does_not_help(self):
        """
        The dense grid's error is in its bounds, not its spacing.

        Rerun it with the spacing ten times finer, so ten times the model
        evaluations, leaving q_min and q_max alone; the error barely moves.
        Whatever it is missing is missing outside [q_min, q_max], where no
        refinement inside will find it -- and those two bounds are exactly
        what the filter does not have to choose.
        """
        for filename, result in self.results.items():
            with self.subTest(filename):
                measurement = result['measurement']
                data = load(measurement)
                wavelength = data.source.wavelength
                zaccept = (2*pi/np.max(wavelength)
                           * np.sin(data.sample.zacceptance[0]))
                refined = sesans.SesansTransform(
                    data.x, data.x, wavelength, zaccept, Rmax=None,
                    log_spacing=1.00003)
                model = load_model(measurement.model_name)
                Iq = call_kernel(
                    model.make_kernel([refined.q_calc]), measurement.pars)

                coarse_error = relative_error(result['dense'][0],
                                              result['reference'])
                refined_error = relative_error(refined.apply(Iq),
                                               result['reference'])
                self.assertGreater(len(refined.q_calc),
                                   9*len(result['dense'][1].q_calc))
                if coarse_error > DISAGREEMENT:
                    self.assertGreater(refined_error, coarse_error/2,
                                       "on %s" % filename)

    def test_g0_is_the_dense_grid_weak_point_not_the_filter(self):
        """
        G(0), not G(xi), is what the dense grid gets wrong.

        This reverses the expectation the filter was written under:
        _g0_weights was the obvious thing to be suspicious of, and on
        measured data it is the strongest part, two to four orders of
        magnitude below the filter's own G(xi) error. The dense grid is the
        other way round, because G(0) needs the whole q range and the grid
        stops at q_max = 2 pi / (xi[1] - xi[0]), set by the spacing of the
        data and knowing nothing about where I(q) ends.
        """
        for filename, result in self.results.items():
            with self.subTest(filename):
                G_ref, G0_ref = reference_transform(result['measurement'])
                scale = np.max(np.abs(G_ref - G0_ref))

                errors = {}
                for strategy, transform in (
                        ('dense', sesans.SesansTransform),
                        ('filter', sesans.DHTSesansTransform)):
                    G, G0 = transform_parts(result['measurement'], transform)
                    errors[strategy] = (np.max(np.abs(G - G_ref))/scale,
                                        abs(G0 - G0_ref)/scale)

                self.assertGreater(errors['dense'][1], errors['dense'][0],
                                   "dense grid on %s" % filename)
                self.assertLess(errors['filter'][1], errors['filter'][0],
                                "filter on %s" % filename)
                self.assertLess(errors['filter'][1], errors['dense'][1]/10,
                                "G(0) on %s" % filename)

    def test_filter_barely_resolves_a_structure_factor(self):
        """
        The one place the dense grid's density earns its keep.

        The filter's abscissae are a fixed table, about 19 per decade, the
        same everywhere: it cannot be refined for a sharp feature. The
        hardsphere peak in core_shell.ses is about a third of a decade wide,
        so the filter gets a handful of points across it where the dense grid
        gets thousands, and that is the filter's worst file.

        It does not change the ranking -- the dense grid is still an order of
        magnitude further out there, losing more to truncation than it gains
        from resolving the peak -- but a sharper feature would eventually
        cross over, and this is what would notice a filter table too short.
        """
        q = np.logspace(-4, -1, 4000)
        S = call_kernel(load_model('hardsphere').make_kernel([q]),
                        dict(radius_effective=730., volfraction=0.45))
        # Full width at half the peak height above the q -> 0 floor.
        above = q[S > (np.max(S) + S[0])/2]
        peak_decades = np.log10(np.max(above)/np.min(above))

        table = sesans.ABSCISSAE
        per_decade = len(table)/np.log10(table[-1]/table[0])
        self.assertLess(per_decade*peak_decades, 10)

        dense = self.results['core_shell.ses']['dense'][1].q_calc
        dense_per_decade = len(dense)/np.log10(dense[-1]/dense[0])
        self.assertGreater(dense_per_decade*peak_decades, 1000)


def compare(repeats=3):
    # type: (int) -> None
    """
    Print the comparison table. Run with ``python -m sasmodels.test_sesans``.
    """
    if SKIP_REASON:
        print("cannot run:", SKIP_REASON)
        return
    header = ("%-28s %4s %9s %8s %8s %7s %10s %9s"
              % ("file", "n", "nq dense", "nq filt", "t dense", "t filt",
                 "err dense", "err filt"))
    print(header)
    print("-"*len(header))
    for measurement in MEASUREMENTS:
        reference = reference_polarisation(measurement)
        row = {}
        for strategy, transform in (('dense', sesans.SesansTransform),
                                    ('filter', sesans.DHTSesansTransform)):
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
        print("%-28s %4d %9d %8d %7.1fms %6.1fms %10.2e %9.2e"
              % (measurement.filename, len(reference),
                 row['dense'][0], row['filter'][0],
                 row['dense'][1]*1e3, row['filter'][1]*1e3,
                 row['dense'][2], row['filter'][2]))


if __name__ == "__main__":
    compare()
