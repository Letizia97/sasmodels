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

The cost rows are the crossover. On the 40-point test set the dense grid works
out at 40515 q values regardless of how many of those 40 points you have,
against `201 * 40 = 8040` for the filter. Add spin-echo lengths and the filter
catches up; the two meet somewhere above roughly 150 points, after which the
dense grid is the cheaper one. Real SESANS datasets tested here have 40 to 84
spin-echo lengths, comfortably on the filter's side of that line.

The matrix row matters for memory: the dense grid's `H` is `n_q * n_xi`, so
1.6M entries for the test set against 8040 for the filter, and `apply` does a
`np.dot` over all of it.

## Where the filter is weak: G(0)

This is the part that is not a published rule, and it deserves the most
scrutiny in review.

`apply` returns `G(xi) - G(0)`, and `G(0)` is the limit as `xi -> 0`. A filter
cannot be evaluated there — its abscissae are `a[i]/xi`. `SesansTransform` gets
`G(0)` for free from its grid (`H0 = dq/(2 pi) * q`, the same rectangle rule).
The filter has no grid, so `_g0_weights` supplies one.

The observation it rests on is that `G(0)` is a plain integral, not a Hankel
transform:

```
G(0) = 1/(2 pi) integral q I(q) dq = 1/(2 pi) integral q^2 I(q) d(ln q)
```

and the union of the abscissae over all spin-echo lengths already covers q
densely, since each `xi` shifts the same table by a different `1/xi`. So a
trapezoid rule in log q over that union costs no extra model evaluations at
all.

That union is dense with many spin-echo lengths (8040 interleaved points for
the test set) and thin with one (just the table, ~19 points per decade). The
single-spin-echo-length case is covered by a test and is accurate there, but a
Gaussian is the easy case for a trapezoid rule and that test should not be read
as a guarantee. `G(0)` is also where the two transforms disagree most on real
models.

## Accuracy

Measured on a Gaussian scatterer, `I(q) = exp(-w^2 q^2 / 2)`, whose transform is
known in closed form (`sesans.gaussian_correlation` /
`gaussian_polarisation`), with 40 spin-echo lengths:

| | `P(xi)` error | `G(0)` error | model evaluations |
| --- | --- | --- | --- |
| dense grid | 1.5e-4 | — | 40515 |
| filter     | 5e-11  | 3e-13 | 8040 |

Errors are relative to `1/(2 pi w^2)`, the scale `P` is measured on. On a
sphere over 40 spin-echo lengths the filter is 36x cheaper per evaluation.

Two caveats on reading that table:

- A Gaussian is a smooth, rapidly decaying integrand — the case a filter is
  built for. For models where `q I(q)` is not integrable, or where the dense
  grid's `q_min`/`q_max` heuristics happen to suit the sample, the ordering can
  go the other way.
- Across the model library the two transforms **do not** agree well. That
  measures agreement, not accuracy, and for real models neither route is
  established as the better one on `P(xi)`. This is the reason the default is
  unchanged.

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

## Status

Exploratory. The filter is implemented, unit tested against a closed form and
against `SesansTransform`, and wired in behind a switch that is off. Open
before it could be a default:

- `_g0_weights` is home-made and is the accuracy floor of the whole thing.
- The disagreement with the dense grid across the model library is unexplained,
  and resolving it needs a reference neither transform provides.
- The crossover in cost against the number of spin-echo lengths means neither
  is the right default for every dataset.
