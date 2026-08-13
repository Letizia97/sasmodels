# Two ways to do the SESANS transform

This branch adds `sesans.DHTSesansTransform` alongside the existing
`sesans.SesansTransform`. Both turn a computed scattering curve `I(q)` into the
SESANS polarisation `P(xi)`; they differ only in how the integral is
evaluated. This note explains that difference, what it buys, and what it
costs.

Nothing changes by default. `direct_model.SESANS_TRANSFORM` selects the class
and still points at `SesansTransform`.

## What both of them compute

Both evaluate the same J0 Hankel transform,

```
G(xi) = 1/(2 pi) * integral_0^inf  q J0(q xi) I(q) dq
P(xi) = G(xi) - G(0)
```

and both do it under the same contract, the one sasmodels uses for resolution
objects:

1. pick the q values you need, up front, before the model is evaluated, and
   expose them as `q_calc`;
2. receive `I(q_calc)` and reduce it to one value per data point in `apply`.

That contract is the reason the two are interchangeable. It is also the reason
neither can be adaptive: you commit to the abscissae before you have seen a
single value of the integrand.

## The existing strategy: a dense log-spaced grid

`SesansTransform` builds one grid of q, shared by every spin-echo length, and
integrates with a rectangle rule in q:

```python
q = exp(arange(log(q_min), log(q_max), log(1.0003)))
H = J0(outer(q, SElength)) * (dq * q / (2 pi))
```

The oscillation of `J0(q xi)` is handled by resolving it: the grid is fine
enough (`log_spacing = 1.0003`, about 7700 points per decade) that the rectangle
rule sees a smooth function. The grid bounds come from heuristics that the
source itself flags as unexplained:

```python
# TODO: Why does q_min depend on the number of correlation lengths?
# TODO: Why does q_max depend on the correlation step size?
q_min = 0.1 * 2*pi / (size(SElength) * SElength[-1])
q_max = 2*pi / (SElength[1] - SElength[0])
```

Two consequences follow from this shape:

- **No convergence criterion.** There is no cheap second estimate to compare
  against, so there is no way to tell from the output whether a given answer is
  converged. Accuracy is inherited from whatever the heuristics produced for
  this particular set of spin-echo lengths.
- **Cost is independent of the number of data points.** One grid serves all of
  them. Adding spin-echo lengths costs nothing in model evaluations (it does
  cost memory and matrix work, see below).

## The new strategy: a digital filter

`DHTSesansTransform` evaluates the same integral as a digital Hankel
transform. A digital filter is a quadrature rule built specifically for Hankel
transforms: a fixed table of abscissae `a[i]` and weights `w[i]` such that

```
G(xi) = 1/(2 pi) * sum_i  I(a[i]/xi) * (a[i]/xi) * w[i] / xi
```

The abscissae are the same table for every problem, shifted by `1/xi`. Because
they follow from the table and from `xi` alone, they are known before the model
is evaluated, which is exactly what `q_calc` has to promise — so the filter
drops into the existing contract without changes anywhere else.

The oscillation of `J0` is not resolved here, it is absorbed into the weights,
which alternate in sign. That is why 201 points spanning eleven decades
(about 19 per decade) can do what the dense grid needs tens of thousands of
points for.

There is no `q_min`, no `q_max` and no spacing to choose. The trade is that the
abscissae depend on `xi`, so each spin-echo length needs its own column of q,
and the model evaluation count grows with the number of data points.

## Side by side

|                          | `SesansTransform`                          | `DHTSesansTransform`                    |
| ------------------------ | ------------------------------------------ | --------------------------------------- |
| quadrature               | rectangle rule in log q, `J0` resolved      | digital filter, `J0` in the weights     |
| abscissae                | one grid shared by all `xi`                 | fixed table / `xi`, one column per `xi` |
| tuning parameters        | `q_min`, `q_max`, `log_spacing`             | none                                    |
| where they come from     | undocumented heuristics on `SElength`       | published filter coefficients           |
| model evaluations        | `log(q_max/q_min)/log(1.0003)`, independent of the number of `xi` | `201 * n_xi` |
| transform matrix         | `(n_q, n_xi)` — grows as the product        | `(201, n_xi)`                           |
| `apply`                  | two `np.dot` against that matrix            | elementwise multiply and sum            |
| `G(0)`                   | free, from the same grid                    | needs its own rule (see below)          |
| `Rmax`                   | accepted and ignored                        | accepted and ignored                    |
| acceptance masking       | applied to `G(xi)` weights only             | same                                    |

The cost rows are the crossover. The dense grid comes out at 32000–44000 q
values across the measured files in `example/`, almost independently of how
many spin-echo lengths they have, against `201 * n_xi` for the filter. Add
spin-echo lengths and the filter catches up; the two meet at around 200
points, after which the dense grid is the cheaper one. The measured files run
from 25 to 120 spin-echo lengths, all on the filter's side of that line.

The matrix row matters for memory: the dense grid's `H` is `n_q * n_xi`, so
3.3M entries on `sphere.ses` against 16080 for the filter, and `apply` does a
`np.dot` over all of it. That, not the model evaluations, is why the wall-clock
gap (6–29×) is wider than the evaluation-count gap (1.8–6.5×).

## G(0), which turned out to be the other way round

`apply` returns `G(xi) - G(0)`, and `G(0)` is the limit as `xi -> 0`. A filter
cannot be evaluated there — its abscissae are `a[i]/xi`. `SesansTransform` gets
`G(0)` for free from its grid (`H0 = dq/(2 pi) * q`, the same rectangle rule).
The filter has no grid, so `_g0_weights` supplies one.

It rests on the observation that `G(0)` is a plain integral, not a Hankel
transform:

```
G(0) = 1/(2 pi) integral q I(q) dq = 1/(2 pi) integral q^2 I(q) d(ln q)
```

and that the union of the abscissae over all spin-echo lengths already covers q
densely, since each `xi` shifts the same table by a different `1/xi`. So a
trapezoid rule in log q over that union costs no extra model evaluations at
all.

This was written up as the filter's weak point — it is the one piece that is
not a published rule, and the obvious thing to be suspicious of. **On measured
data it is the strongest part, and `G(0)` is instead where the dense grid
loses.** Splitting each transform's error into its two halves, relative to
`max|P|`:

| file | dense G(ξ) | dense **G(0)** | filter G(ξ) | filter G(0) |
| --- | --- | --- | --- | --- |
| `sphere.ses` | 1.4e-3 | **8.4e-3** | 2.0e-4 | 6.5e-7 |
| `spheres2micron.ses` | 1.3e-4 | **5.0e-4** | 3.5e-4 | 7.4e-7 |
| `core_shell.ses` | 8.5e-3 | **4.4e-2** | 4.2e-3 | 2.0e-6 |
| `se008724_01.ses` | 1.5e-4 | **1.7e-4** | 3.0e-4 | 4.6e-7 |
| `se008731_01_40pcorr.ses` | 1.3e-4 | **5.0e-4** | 3.5e-4 | 7.4e-7 |
| `SiO2_100pc_H2O_0pc_D2O.ses` | 3.4e-3 | **8.6e-2** | 4.1e-5 | 1.3e-5 |

The dense grid's `G(0)` error exceeds its `G(ξ)` error on every file, and
dominates its total. The filter's `G(0)` error is two to four orders of
magnitude *below* its `G(ξ)` error, and 40× to 6700× below the dense grid's.

The reason is structural. `G(0)` is an integral over the whole of q, and the
dense grid stops at `q_max = 2π/(ξ[1] - ξ[0])` — a bound set by the spacing of
the data, which knows nothing about where `I(q)` actually ends. `H0` is that
same truncated grid, so the missing tail goes straight into `G(0)`. The
filter's union of abscissae spans eleven decades and runs off both ends of any
model, so its trapezoid rule has the whole integrand to work with.

The thin case for `_g0_weights` remains a single spin-echo length, where the
union is just the table at ~19 points per decade. There is a test for it on a
Gaussian, which is the easy case for a trapezoid rule; none of the measured
files exercise it.

## Accuracy on measured data

`sasmodels/test_sesans.py` runs both transforms over every `.ses`
file in `example/`, through `DirectModel` with real models, exactly the way
SasView runs them. Because measured data has no closed form, it compares both
against a third evaluation of the same integral — `reference_polarisation`,
a trapezoid rule on a q grid fine enough to resolve `J0(q xi)` at the longest
spin-echo length, sharing no code with either transform. Doubling its
resolution moves it by less than 1e-5, so it can referee.

Errors as a fraction of `max|P|`; times are per call, the thing a fitter pays
thousands of times.

| file | n | nq dense | nq filter | t dense | t filter | err dense | err filter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sphere.ses` | 80 | 41552 | 16080 | 10.7 ms | 1.0 ms | 9.8e-3 | **2.0e-4** |
| `spheres2micron.ses` | 40 | 32216 | 8040 | 9.6 ms | 0.7 ms | 5.0e-4 | 3.5e-4 |
| `core_shell.ses` | 80 | 41552 | 16080 | 17.4 ms | 2.7 ms | 5.3e-2 | **4.2e-3** |
| `se008724_01.ses` | 120 | 44005 | 24120 | 11.2 ms | 1.3 ms | **1.7e-4** | 3.0e-4 |
| `se008731_01_40pcorr.ses` | 40 | 32216 | 8040 | 11.0 ms | 0.6 ms | 5.0e-4 | 3.5e-4 |
| `SiO2_100pc_H2O_0pc_D2O.ses` | 25 | 32691 | 5025 | 8.8 ms | 0.3 ms | 9.0e-2 | **5.3e-5** |

Three things come out of this.

**Where the two disagree, the filter is the one that is right.** On three of
the six files they agree to better than 1e-3 of the signal and the argument is
academic. On the other three the dense grid is off by 1% to 9% of the signal
and the filter is 12× to 1700× closer to the reference. There is no file where
they disagree materially and the dense grid wins. `sphere.ses` at 9.8e-3 is
about 20σ on that file's error bars, so this is not below the noise.

**The dense grid's error is in its bounds, not its spacing.** Rerunning it with
`log_spacing = 1.00003` — ten times finer, ten times the model evaluations —
leaves the error unchanged to two figures (9.8e-3 → 9.8e-3, 5.3e-2 → 5.3e-2,
9.0e-2 → 9.0e-2). Whatever it is missing lies outside `[q_min, q_max]`, and no
refinement inside will find it. Those two bounds are precisely what the filter
does not have to choose, which is the whole argument for it.

**The filter is 6× to 29× faster per call** on these files, on 1.8× to 6.5×
fewer model evaluations. The extra factor is the dense grid's `np.dot` against
an `n_q × n_xi` matrix.

The one case that goes the other way is worth naming: `core_shell.ses` is the
filter's worst file by a factor of sixteen, and it is the only one with a
structure factor. The hardsphere peak is about a third of a decade wide, and
the filter's table is a fixed ~19 points per decade, so it gets about six
points across the peak where the dense grid gets thousands. The filter still
wins on that file overall, because it loses less to the peak than the dense
grid loses to truncation — but a sharper feature would eventually cross over,
and the filter has no refinement knob to answer with.

## Accuracy on a closed form

As a sanity check rather than a verdict, on a Gaussian scatterer
`I(q) = exp(-w^2 q^2 / 2)` with 40 log-spaced spin-echo lengths:

| | `P(xi)` error | `G(0)` error | model evaluations |
| --- | --- | --- | --- |
| dense grid | 1.5e-4 | 1.6e-4 | 40515 |
| filter     | 5e-11  | 3e-13  | 8040  |

Do not read too much into the size of that gap. A Gaussian is the smoothest
integrand either rule will ever see, and log-spaced spin-echo lengths are not
something an instrument produces — every real file above is linear or
piecewise linear in ξ. Both of those flatter the filter. The measured-data
table is the one to argue from.

## Switching between them

```python
from sasmodels import direct_model, sesans
direct_model.SESANS_TRANSFORM = sesans.DHTSesansTransform
```

Anything with the same constructor signature and the same `q`, `q_calc` and
`apply` will do. `_make_sesans_transform` reads the module-level name instead of
referring to `SesansTransform` directly.

## Provenance of the coefficients

`sesans_filter.py` vendors the `DHT_Key_201` table — 201 abscissae and weights
— so the branch adds no new dependency. It is generated by
`examples/sesans/export_filter_table.py` in LibHankel and should not be edited
by hand. The coefficients themselves are Kerry Key's published Hankel filters,
original to neither project.

## Tests

All in `sasmodels/test_sesans.py`, in two parts:

- the filter against a closed form on a Gaussian scatterer. Fast, no
  dependencies beyond numpy and scipy.
- the measured-data comparison above: both transforms over every `.ses` file in
  `example/`, refereed by an independent brute-force evaluation. Needs *sasdata*
  to read the files and a compiler to build the models, and skips if either is
  missing.

Run `python -m sasmodels.test_sesans` to print the comparison table.

## Status

The filter is implemented, tested against a closed form and against measured
data, and wired in behind a switch that is off.

Two of the three concerns this branch was opened with have been answered, and
in the filter's favour:

- ~~`_g0_weights` is home-made and is the accuracy floor.~~ It is the most
  accurate part of the filter on every measured file. `G(0)` is where the
  *dense grid* loses.
- ~~The disagreement with the dense grid is unexplained, and resolving it needs
  a reference neither transform provides.~~ There is now such a reference. The
  disagreement is the dense grid truncating at `q_max`, it does not go away
  under refinement, and where the two disagree the filter is closer.

What is still open:

- **The structure factor case.** `core_shell.ses` costs the filter a factor of
  sixteen, and the filter has no way to refine. One file is not enough to know
  how sharp a feature has to be before it loses outright.
- **Cost at many spin-echo lengths.** The filter is 201 points per spin-echo
  length; above roughly 200 points the dense grid is cheaper. The largest file
  here is 120. This is a real crossover, just not one these datasets reach.
- **The single-spin-echo-length case** for `_g0_weights`, which nothing
  measured exercises.
- **Whether any of this changes a fit.** Everything above is quadrature error
  against a reference. Whether a 1–9% error on `P(ξ)` moves fitted parameters,
  and by how much against their uncertainties, has not been tested.
