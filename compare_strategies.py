"""
Four-way SESANS timing and accuracy comparison.

Run from the repo root with the sasmodels venv:

    /tmp/sasenv/bin/python compare_strategies.py

Strategies compared:
    dense      -- SesansTransform (old, rectangle rule on dense log q grid)
    filter     -- DHTSesansTransform (digital Hankel filter, no extra dep)
    lh_dht     -- LibhankelSesansTransform with DHT_Key_201
    lh_ooura   -- LibhankelSesansTransform with Adaptive_DE_Ooura
"""

import sys
import time
import types

import numpy as np

# Run as a top-level script, not a package module, so add the repo to the path.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from sasmodels import sesans, direct_model
from sasmodels.test_sesans import (
    MEASUREMENTS, SKIP_REASON,
    make_calculator, reference_polarisation, relative_error,
)

try:
    import libhankel as _lh  # noqa: F401
    _LIBHANKEL_AVAILABLE = True
except ImportError:
    _LIBHANKEL_AVAILABLE = False


def make_libhankel_class(strategy_name, strategy_params=None):
    """Return a LibhankelSesansTransform subclass pinned to one strategy."""
    return types.new_class(
        "Lh_" + strategy_name,
        (sesans.LibhankelSesansTransform,),
        {},
        lambda ns: ns.update(
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
        ),
    )


def compare(repeats=3):
    if SKIP_REASON:
        print("cannot run:", SKIP_REASON)
        return

    strategies = [
        ("dense",    sesans.SesansTransform),
        ("filter",   sesans.DHTSesansTransform),
    ]
    if _LIBHANKEL_AVAILABLE:
        strategies += [
            ("libhankel_dht",   make_libhankel_class("DHT_Key_201")),
            ("libhankel_ooura", make_libhankel_class(
                "Adaptive_DE_Ooura", {"n_eval": 50, "eps_rel": 1e-6})),
            ("libhankel_qwe",   make_libhankel_class(
                "QWE_Chave", {"n_eval": 50, "eps_rel": 1e-6})),
        ]

    names = [s for s, _ in strategies]
    col_w = 10

    header = (
        "%-28s %4s" % ("file", "n")
        + "".join(" %*s" % (col_w, "t " + n) for n in names)
        + "".join(" %*s" % (col_w, "err " + n) for n in names)
    )
    print(header)
    print("-" * len(header))

    for measurement in MEASUREMENTS:
        reference = reference_polarisation(measurement)
        row = {}
        for strategy, transform in strategies:
            calc = make_calculator(measurement, transform)
            P = calc(**measurement.pars)
            start = time.perf_counter()
            for _ in range(repeats):
                calc(**measurement.pars)
            elapsed = (time.perf_counter() - start) / repeats
            row[strategy] = (elapsed, relative_error(P, reference))

        print(
            "%-28s %4d" % (measurement.filename, len(reference))
            + "".join(" %*.1fms" % (col_w - 2, row[n][0] * 1e3) for n in names)
            + "".join(" %*.2e" % (col_w, row[n][1]) for n in names)
        )


if __name__ == "__main__":
    compare()
