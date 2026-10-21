"""
Four-way SESANS comparison exported to Excel with per-row colour coding.

Run from the repo root with the sasmodels venv:

    /tmp/sasenv/bin/python make_comparison_excel.py

Produces sesans_comparison.xlsx in the current directory.
Colour scale applied per row: green = best (lowest), red = worst (highest).
Both timing and error columns are "lower is better".
"""

import sys
import time
import types

import numpy as np
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

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
    return types.new_class(
        "Lh_" + strategy_name,
        (sesans.LibhankelSesansTransform,),
        {},
        lambda ns: ns.update(
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
        ),
    )


def lerp_colour(lo_rgb, hi_rgb, t):
    """Interpolate between two RGB tuples; t=0 → lo, t=1 → hi."""
    r = int(lo_rgb[0] + t * (hi_rgb[0] - lo_rgb[0]))
    g = int(lo_rgb[1] + t * (hi_rgb[1] - lo_rgb[1]))
    b = int(lo_rgb[2] + t * (hi_rgb[2] - lo_rgb[2]))
    return "%02X%02X%02X" % (r, g, b)


GREEN = (0x63, 0xBE, 0x7B)   # Excel green
YELLOW = (0xFF, 0xEB, 0x84)  # Excel yellow (mid)
RED = (0xF8, 0x69, 0x6B)     # Excel red


def value_to_colour(value, min_val, max_val):
    """Map a value in [min_val, max_val] to a green→yellow→red hex string."""
    if max_val == min_val:
        return lerp_colour(GREEN, RED, 0.5)
    t = (value - min_val) / (max_val - min_val)  # 0=best(green), 1=worst(red)
    if t <= 0.5:
        return lerp_colour(GREEN, YELLOW, t * 2)
    else:
        return lerp_colour(YELLOW, RED, (t - 0.5) * 2)


def collect(repeats=3):
    strategies = [
        ("dense",    sesans.SesansTransform),
        ("filter",   sesans.DHTSesansTransform),
    ]
    if _LIBHANKEL_AVAILABLE:
        strategies += [
            ("libhankel_dht_key_201",         make_libhankel_class("DHT_Key_201")),
            ("libhankel_dht_anderson_801",    make_libhankel_class("DHT_Anderson_801")),
            ("libhankel_ooura",               make_libhankel_class(
                "Adaptive_DE_Ooura", {"n_eval": 50, "eps_rel": 1e-6})),
            ("libhankel_qwe",                 make_libhankel_class(
                "QWE_Key", {"n_eval": 2000, "eps_rel": 1e-6})),
            ("libhankel_dht_key_101",         make_libhankel_class("DHT_Key_101")),
            ("libhankel_dht_key_51",          make_libhankel_class("DHT_Key_51")),
            ("libhankel_dht_guptasarma",      make_libhankel_class("DHT_Guptasarma")),
            ("libhankel_dht_guptasarma_fast", make_libhankel_class("DHT_Guptasarma_Fast")),
        ]

    rows = []
    for measurement in MEASUREMENTS:
        reference = reference_polarisation(measurement)
        row = {"file": measurement.filename, "n": len(reference)}
        for strategy, transform in strategies:
            try:
                calc = make_calculator(measurement, transform)
                P = calc(**measurement.pars)
                start = time.perf_counter()
                for _ in range(repeats):
                    calc(**measurement.pars)
                elapsed = (time.perf_counter() - start) / repeats
                row["t_" + strategy] = elapsed * 1e3
                row["err_" + strategy] = relative_error(P, reference)
            except RuntimeError:
                row["t_" + strategy] = None
                row["err_" + strategy] = None
        rows.append(row)
    return rows, [s for s, _ in strategies]


def write_table(ws, rows, title, data_keys, num_format, start_row, strategy_names):
    """Write one table (timing or error) starting at *start_row*. Returns next free row."""
    header_font = Font(color="FFFFFF", bold=True)
    group_fill  = PatternFill("solid", fgColor="2F5496")
    header_fill = PatternFill("solid", fgColor="4472C4")
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    n_cols = 1 + len(data_keys)  # file + one per strategy

    # Title row
    ws.cell(start_row, 1, title)
    ws.merge_cells(start_row=start_row, start_column=1,
                   end_row=start_row, end_column=n_cols)
    title_cell = ws.cell(start_row, 1)
    title_cell.font = Font(color="FFFFFF", bold=True, size=12)
    title_cell.fill = group_fill
    title_cell.alignment = Alignment(horizontal="center")

    # Column header row
    hr = start_row + 1
    ws.cell(hr, 1, "file")
    for ci, s in enumerate(strategy_names, start=2):
        cell = ws.cell(hr, ci, s.replace("_", " "))
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    ws.cell(hr, 1).font = header_font
    ws.cell(hr, 1).fill = header_fill
    ws.cell(hr, 1).alignment = Alignment(horizontal="center")

    fail_fill = PatternFill("solid", fgColor="D9D9D9")  # light grey for failed
    fail_font = Font(color="FF0000", italic=True)

    # Data rows
    for ri, row in enumerate(rows, start=hr + 1):
        ws.cell(ri, 1, row["file"]).alignment = Alignment(horizontal="left")
        ws.cell(ri, 1).border = border

        vals = [row[k] for k in data_keys]
        # Exclude QWE from the colour scale so it doesn't compress the others.
        scale_vals = [v for k, v in zip(data_keys, vals)
                      if v is not None and "qwe" not in k]
        mn = min(scale_vals) if scale_vals else 0
        mx = max(scale_vals) if scale_vals else 1
        qwe_fill = PatternFill("solid", fgColor="E2EFDA")  # pale green-grey, neutral
        for ci, (k, v) in enumerate(zip(data_keys, vals), start=2):
            cell = ws.cell(ri, ci)
            cell.alignment = Alignment(horizontal="right")
            cell.border = border
            if v is None:
                cell.value = "failed"
                cell.fill = fail_fill
                cell.font = fail_font
            elif "qwe" in k:
                cell.value = v
                cell.number_format = num_format
                cell.fill = qwe_fill
            else:
                cell.value = v
                cell.number_format = num_format
                cell.fill = PatternFill("solid", fgColor=value_to_colour(v, mn, mx))

    return hr + len(rows) + 1  # +1 blank gap


def write_excel(rows, strategy_names, path="sesans_comparison.xlsx"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SESANS comparison"

    t_keys   = ["t_"   + s for s in strategy_names]
    err_keys = ["err_" + s for s in strategy_names]

    next_row = write_table(ws, rows, "Time per call (ms)",
                           t_keys, "0.0", start_row=1, strategy_names=strategy_names)
    write_table(ws, rows, "Error vs brute-force reference",
                err_keys, "0.00E+00", start_row=next_row + 1, strategy_names=strategy_names)

    # Column widths
    ws.column_dimensions["A"].width = 32
    for ci in range(2, len(strategy_names) + 2):
        ws.column_dimensions[get_column_letter(ci)].width = 16

    wb.save(path)
    print("saved:", path)


if __name__ == "__main__":
    if SKIP_REASON:
        print("cannot run:", SKIP_REASON)
        sys.exit(1)
    rows, strategy_names = collect(repeats=5)
    write_excel(rows, strategy_names)
