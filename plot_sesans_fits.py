"""
Plot P(xi) for each .ses example file, comparing all four strategies
against the brute-force reference.

Run from the repo root with the sasmodels venv:

    /tmp/sasenv/bin/python plot_sesans_fits.py

Produces sesans_fits.pdf (one page per file) in the current directory.
"""

import sys
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from sasmodels import sesans
from sasmodels.test_sesans import (
    MEASUREMENTS, SKIP_REASON,
    make_calculator, reference_polarisation,
)

try:
    import libhankel as _lh  # noqa: F401
    _LIBHANKEL_AVAILABLE = True
except ImportError:
    _LIBHANKEL_AVAILABLE = False


def make_libhankel_class(strategy_name, strategy_params=None):
    return types.new_class(
        "Lh_" + strategy_name,
        (sesans.LibhankelSesansTransform,),
        {},
        lambda ns: ns.update(
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
        ),
    )


STRATEGIES = [
    ("reference",      None,                          "k",       "-",  2.5),
    ("dense",          sesans.SesansTransform,         "#888888", "--", 1.5),
    ("filter",         sesans.DHTSesansTransform,      "#2196F3", "-",  1.5),
]

if _LIBHANKEL_AVAILABLE:
    STRATEGIES += [
        ("libhankel dht",    make_libhankel_class("DHT_Key_201"),                         "#FF9800", "-",  1.5),
        ("libhankel ooura",  make_libhankel_class("Adaptive_DE_Ooura",
                                                   {"n_eval": 50, "eps_rel": 1e-6}),      "#4CAF50", "-",  1.5),
        ("libhankel fftlog", make_libhankel_class("FFTLog"),                               "#E91E63", "--", 1.5),
    ]


def collect_curves():
    curves = {}
    for m in MEASUREMENTS:
        xi = None
        row = {}
        ref = reference_polarisation(m)

        # x axis: spin-echo lengths from the data file
        from sasmodels.test_sesans import load
        data = load(m)
        xi = data.x

        row["reference"] = (xi, ref)

        for label, transform, *_ in STRATEGIES:
            if label == "reference":
                continue
            try:
                calc = make_calculator(m, transform)
                P = calc(**m.pars)
                row[label] = (calc.resolution.q if hasattr(calc.resolution, 'q') else xi, P)
                # use xi from the data, not q
                row[label] = (xi, P)
            except RuntimeError:
                row[label] = None

        curves[m.filename] = row
    return curves


def plot_all(curves, path="sesans_fits.pdf"):
    with PdfPages(path) as pdf:
        for filename, row in curves.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            fig.suptitle(filename, fontsize=11)

            xi_ref, P_ref = row["reference"]

            for label, transform, color, ls, lw in STRATEGIES:
                if label not in row:
                    continue
                if label == "reference":
                    xi, P = xi_ref, P_ref
                elif row[label] is None:
                    continue
                else:
                    xi, P = row[label]

                for ax in axes:
                    ax.plot(xi, P, color=color, ls=ls, lw=lw, label=label)

            # Left: linear scale
            axes[0].set_xlabel("spin-echo length (Å)")
            axes[0].set_ylabel("P(ξ)")
            axes[0].set_title("linear scale")
            axes[0].legend(fontsize=8)

            # Right: log x scale, better to see short-xi behaviour
            axes[1].set_xscale("log")
            axes[1].set_xlabel("spin-echo length (Å)")
            axes[1].set_ylabel("P(ξ)")
            axes[1].set_title("log ξ scale")
            axes[1].legend(fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print("plotted:", filename)

    print("saved:", path)


if __name__ == "__main__":
    if SKIP_REASON:
        print("cannot run:", SKIP_REASON)
        sys.exit(1)
    curves = collect_curves()
    plot_all(curves)
