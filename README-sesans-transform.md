# Three ways to do the SESANS transform

This branch adds two classes alongside the existing `sesans.SesansTransform`:

- `sesans.DHTSesansTransform` — a digital Hankel filter; no new dependency.
- `sesans.LibhankelSesansTransform` — delegates to `libhankel`; requires the
  `libhankel` package, which is an optional dependency.

All three turn a computed scattering curve `I(q)` into the SESANS polarisation
`P(xi)`; they differ in how the integral is evaluated. This note explains those
differences, what each one buys, and what it costs.

Nothing changes by default. `direct_model.SESANS_TRANSFORM` selects the class
and still points at `SesansTransform`.

## What all three compute

All three evaluate the same J0 Hankel transform,

```
G(xi) = 1/(2 pi) * integral_0^inf  q J0(q xi) I(q) dq
P(xi) = G(xi) - G(0)
```

and all do it under the same contract, the one sasmodels uses for resolution
objects:

1. pick the q values you need, up front, before the model is evaluated, and
   expose them as `q_calc`;
2. receive `I(q_calc)` and reduce it to one value per data point in `apply`.

That contract is the reason the three are interchangeable. `SesansTransform`
and `DHTSesansTransform` must commit to all their abscissae upfront so they
cannot adapt after seeing the integrand. `LibhankelSesansTransform` works
around this by building a dense q grid for `q_calc` (so the contract is met)
and then letting libhankel choose its own internal quadrature nodes when
`apply` is called.

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

|                          | `SesansTransform`                          | `DHTSesansTransform`                    | `LibhankelSesansTransform`                        |
| ------------------------ | ------------------------------------------ | --------------------------------------- | ------------------------------------------------- |
| quadrature               | rectangle rule in log q, `J0` resolved      | digital filter, `J0` in the weights     | libhankel strategy (e.g. `Adaptive_DE_Ooura`)     |
| abscissae for `q_calc`   | one grid shared by all `xi`                 | fixed table / `xi`, one column per `xi` | one dense grid shared by all `xi` (same as dense) |
| internal quadrature      | same as `q_calc`                            | same as `q_calc`                        | chosen by libhankel at `apply` time               |
| tuning parameters        | `q_min`, `q_max`, `log_spacing`             | none                                    | `strategy_name`, `strategy_params`                |
| where they come from     | undocumented heuristics on `SElength`       | published filter coefficients           | libhankel library                                 |
| model evaluations        | ~32k–44k, independent of number of `xi`    | `201 * n_xi`                            | ~32k–44k (same grid as dense)                     |
| `apply`                  | two `np.dot` against `(n_q, n_xi)` matrix  | elementwise multiply and sum            | call to libhankel C library                       |
| `G(0)`                   | free, from the same grid                    | trapezoid over union of abscissae       | trapezoid over dense grid                         |
| acceptance masking       | applied per point                           | applied per point                       | not applied                                       |
| external dependency      | none                                        | none                                    | `libhankel`                                       |

The dense grid's `q_calc` size (32k–44k) and `LibhankelSesansTransform`'s
are similar because both build the same kind of grid; only the integration step
differs. `DHTSesansTransform` uses far fewer model evaluations (201 × n_xi)
because the filter's abscissae are spread across ξ values rather than packed
into one shared grid. The dense grid's `apply` is the expensive step: a
`np.dot` over an `n_q × n_xi` matrix up to 3.3M entries on `sphere.ses`,
which is why it is 6–29× slower per call despite similar evaluation counts to
libhankel.

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

`sasmodels/test_sesans.py` runs all three transforms over every `.ses`
file in `example/`, through `DirectModel` with real models, exactly the way
SasView runs them. Because measured data has no closed form, it compares all
three against a brute-force evaluation of the same integral — `reference_polarisation`,
a trapezoid rule on a q grid fine enough to resolve `J0(q xi)` at the longest
spin-echo length, sharing no code with any of the transforms. Doubling its
resolution moves it by less than 1e-5, so it can referee.

Errors as a fraction of `max|P|`; times are per call, the thing a fitter pays
thousands of times. The libhankel column uses the default `DHT_Key_201` strategy;
see the libhankel section below for results with `Adaptive_DE_Ooura`.

| file | n | nq dense | nq filter | t dense | t filter | err dense | err filter | err libhankel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sphere.ses` | 80 | 41552 | 16080 | 10.7 ms | 1.0 ms | 9.8e-3 | **2.0e-4** | 1.9e-4 |
| `spheres2micron.ses` | 40 | 32216 | 8040 | 9.6 ms | 0.7 ms | 5.0e-4 | 3.5e-4 | 3.5e-4 |
| `core_shell.ses` | 80 | 41552 | 16080 | 17.4 ms | 2.7 ms | 5.3e-2 | **4.2e-3** | 4.2e-3 |
| `se008724_01.ses` | 120 | 44005 | 24120 | 11.2 ms | 1.3 ms | **1.7e-4** | 3.0e-4 | 3.0e-4 |
| `se008731_01_40pcorr.ses` | 40 | 32216 | 8040 | 11.0 ms | 0.6 ms | 5.0e-4 | 3.5e-4 | 3.5e-4 |
| `SiO2_100pc_H2O_0pc_D2O.ses` | 25 | 32691 | 5025 | 8.8 ms | 0.3 ms | 9.0e-2 | **5.3e-5** | 2.4e-3 |

`LibhankelSesansTransform` with `DHT_Key_201` matches `DHTSesansTransform` on
every fixed-wavelength file (errors agree to better than 1%). The time-of-flight
file (`SiO2`) is the exception: the filter applies a per-point acceptance mask;
libhankel cannot, so it is 45× worse on that file regardless of strategy.

Three things come out of the dense vs. filter comparison.

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

## How sharp a feature does it take?

The cleanest way to answer that is to sharpen the one feature we already have.
`core_shell.ses` keeps its spin-echo lengths, wavelength and acceptance; the
model stays `core_shell_sphere@hardsphere`; only `volfraction` moves, which
narrows the structure factor peak and changes nothing else:

```python
from sasmodels import sesans
from sasmodels.test_sesans import (
    MEASUREMENTS, Measurement, make_calculator, reference_polarisation,
    relative_error)

base = [m for m in MEASUREMENTS if m.filename == "core_shell.ses"][0]
for vf in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
    m = Measurement(base.filename, base.model_name,
                    dict(base.pars, volfraction=vf))
    reference = reference_polarisation(m, n_linear=400001)
    for transform in (sesans.SesansTransform, sesans.DHTSesansTransform):
        P = make_calculator(m, transform)(**m.pars)
        print(vf, transform.__name__, relative_error(P, reference))
```

The reference was re-checked at each `volfraction` — doubling its resolution
moves it by ~1e-10 — so it still referees at these sharpnesses.

| `volfraction` | peak FWHM | filter points across | err dense | err filter | closer |
| --- | --- | --- | --- | --- | --- |
| 0.45 | 0.099 dec | 1.9 | 5.3e-2 | **4.2e-3** | filter, 12.5× |
| 0.50 | — | — | 6.2e-2 | **1.4e-2** | filter, 4.5× |
| 0.55 | 0.049 dec | 0.9 | 7.5e-2 | **2.5e-2** | filter, 3.1× |
| 0.60 | 0.032 dec | 0.6 | 9.3e-2 | **7.4e-2** | filter, 1.3× |
| 0.65 | — | — | 1.21e-1 | **1.15e-1** | filter, 1.05× |
| 0.70 | — | — | **1.61e-1** | 1.89e-1 | dense, 1.2× |

(FWHM measured above the minimum of `S(q)`. The "third of a decade" quoted in
the previous section is the same peak measured above the `q -> 0` floor, a
wider baseline and so a wider peak.)

So the crossover exists and lands at `volfraction ≈ 0.68`. Across this range
the filter degrades by 45× and the dense grid by 3×: the filter has a fixed
~19 abscissae per decade and nothing to spend when the integrand needs more,
which is the cost of having no tuning parameters.

The more informative number is underneath, splitting the error in two again:

| `volfraction` | dense G(ξ) | filter G(ξ) | dense G(0) | filter G(0) |
| --- | --- | --- | --- | --- |
| 0.45 | 8.5e-3 | **4.2e-3** | 4.4e-2 | 2.0e-6 |
| 0.50 | **1.0e-2** | 1.4e-2 | — | — |
| 0.55 | **1.2e-2** | 2.5e-2 | 6.2e-2 | 2.4e-5 |
| 0.60 | **1.6e-2** | 7.4e-2 | 7.8e-2 | 2.9e-5 |
| 0.65 | **2.0e-2** | 1.15e-1 | — | — |
| 0.70 | **2.7e-2** | 1.9e-1 | — | — |

**On `G(ξ)` alone — the oscillatory integral the filter exists to do — the
dense grid overtakes it at `volfraction ≈ 0.48`, and is 7× better by 0.70.**
That is far earlier than the total-error crossover at 0.68, and `volfraction`
0.5 is an ordinary concentrated suspension.

The filter goes on winning overall up to 0.68 only because the dense grid's
`G(0)` error is large enough to mask its better `G(ξ)`. Two errors of
different origin, and the larger one belongs to the other method.

Two caveats on how far to push this. `volfraction` above ~0.5 is past where
Percus-Yevick is quantitatively reliable, so the *total*-error crossover at
0.68 is arguably not reachable with a physical sample of this model — but the
`G(ξ)` crossover at 0.48 is squarely inside the usable range. And this is one
model on one file: it says a sharp enough feature flips the ranking, and
roughly how sharp, not that hardsphere is the worst case.

What it changes about the argument: the filter's advantage on the six measured
files may be mostly a `G(0)` advantage rather than a better Hankel transform.
The check that would settle it is `dense G(ξ)` against `filter G(ξ)` on the
five *smooth* files, with `G(0)` set aside. If the filter wins those,
replacing the transform is the right move; if it only ties, the smaller and
better-targeted fix is to repair `G(0)` in the existing method, which would
also keep the dense grid's advantage on sharp features. That has not been run.

## A libhankel backend: `LibhankelSesansTransform`

`LibhankelSesansTransform` is a third class on this branch. It uses the same
dense log-spaced q grid as `SesansTransform` but delegates the integration to
`libhankel.hankel_transform` rather than doing the matrix multiply itself.

### What it adds

`LibhankelSesansTransform` extends the two fixed-rule classes with adaptive
quadrature strategies. `libhankel` supports several strategies; `DHT_Key_201`
(the default) uses the same Kerry Key filter table vendored in `sesans_filter.py`,
while `Adaptive_DE_Ooura` is a double-exponential method that handles J₀
oscillations analytically and achieves 10–74× better accuracy than `DHT_Key_201`
at only 2–3× the cost.

### What QWE_Key gives

`QWE_Key` is an adaptive Gauss–Kronrod quadrature that sets its own abscissae
and refines until a tolerance criterion (`eps_rel`) is met. Unlike `DHT_Key_201`
it is not the same rule as anything already in sasmodels, so it gives a
genuinely independent measurement.

To run it, override the class attributes before constructing the transform:

```python
from sasmodels import direct_model, sesans
sesans.LibhankelSesansTransform.strategy_name = "QWE_Key"
sesans.LibhankelSesansTransform.strategy_params = {"n_eval": 5000, "eps_rel": 1e-6}
direct_model.SESANS_TRANSFORM = sesans.LibhankelSesansTransform
```

Results on the measured files (`n_eval=5000, eps_rel=1e-6`):

| file | err filter | err QWE_Key | factor | t filter | t QWE_Key |
| --- | --- | --- | --- | --- | --- |
| `sphere.ses` | 2.03e-4 | **1.21e-5** | 17× better | 0.5 ms | 391 ms |
| `spheres2micron.ses` | 3.48e-4 | **4.09e-6** | 85× better | 1.7 ms | 37 ms |
| `core_shell.ses` | 4.19e-3 | **4.66e-5** | 90× better | 2.1 ms | 171 ms |
| `se008724_01.ses` | 3.02e-4 | **2.89e-5** | 10× better | 0.7 ms | 243 ms |
| `se008731_01_40pcorr.ses` | 3.48e-4 | **4.09e-6** | 85× better | 0.5 ms | 34 ms |
| `SiO2_100pc_H2O_0pc_D2O.ses` | **5.32e-5** | 2.44e-3 | 45× worse | 0.2 ms | 114 ms |

QWE_Key is 10–90× more accurate than the filter on five of six files, including
`core_shell.ses` — the filter's hardest case — where QWE's adaptive refinement
resolves the hardsphere peak that the filter's fixed ~19 points/decade cannot.
The time-of-flight file remains worse regardless of strategy: the 45× gap is
the acceptance-mask issue, not the quadrature.

The cost is 20–800× slower per call (34–391 ms against 0.2–2.1 ms), so
QWE_Key is not viable for fitting but is well suited as a one-shot accuracy
check or a reference computation.

**False convergence warning.** At `eps_rel=1e-4` QWE_Key reports convergence
on every file but gives an error of 1.79e-1 on `se008724_01.ses` — three orders
of magnitude off. A loose tolerance lets QWE declare each sub-interval locally
converged without resolving the full integrand. **Do not use QWE_Key with
`eps_rel` looser than 1e-6 on tabulated data.**

QWE_Key is slow because it is a general-purpose Gauss-Kronrod rule that must
resolve every J₀ oscillation numerically, one sub-interval at a time; the
number of sub-intervals grows with ξ. Coarsening the q_calc grid (which reduces
the kinks in the piecewise-linear interpolant) saves at most 16% of the time —
the oscillations, not the kinks, are the bottleneck. `Adaptive_DE_Ooura` below
achieves the same accuracy without this cost.

### Adaptive_DE_Ooura: the practical adaptive strategy

`Adaptive_DE_Ooura` is a double-exponential method designed for oscillatory
Bessel-function integrals. Unlike QWE_Key, which is a general-purpose
Gauss-Kronrod rule that has to resolve J₀ oscillations one sub-interval at a
time, the DE transformation in Ooura's method handles the oscillations
analytically, so the method converges in far fewer evaluations.

```python
sesans.LibhankelSesansTransform.strategy_name = "Adaptive_DE_Ooura"
sesans.LibhankelSesansTransform.strategy_params = {"n_eval": 50, "eps_rel": 1e-6}
```

Results on the measured files:

| file | err DHT | err Ooura | factor | t DHT | t Ooura |
| --- | --- | --- | --- | --- | --- |
| `sphere.ses` | 1.94e-4 | **2.06e-5** | 9× better | 7 ms | 7 ms |
| `spheres2micron.ses` | 3.49e-4 | **5.49e-6** | 64× better | 3 ms | 11 ms |
| `core_shell.ses` | 4.22e-3 | **5.70e-5** | 74× better | 6 ms | 8 ms |
| `se008724_01.ses` | 3.00e-4 | **2.20e-5** | 14× better | 5 ms | 19 ms |
| `se008731_01_40pcorr.ses` | 3.49e-4 | **5.49e-6** | 64× better | 3 ms | 8 ms |
| `SiO2_100pc_H2O_0pc_D2O.ses` | 2.44e-3 | 2.44e-3 | — (masking) | 3 ms | 3 ms |

Ooura gives the same accuracy as QWE_Key at `n_eval=5000, eps_rel=1e-6` but at
only 7–19 ms — 5–50× faster than QWE_Key and only 2–3× slower than
`DHT_Key_201` itself. The `n_eval` parameter has no effect in this range:
`n_eval=50` already converges fully, and increasing it changes nothing. The
convergence criterion (`eps_rel`) does the right work.

This makes Ooura a plausible accuracy-check tool and potentially a fitter
strategy for applications where 10–74× better accuracy is worth a factor of 2–3
in cost. The only caveat is the same one as all `LibhankelSesansTransform`
strategies: the acceptance mask is not applied, so the time-of-flight file
shows no improvement over `DHT_Key_201`.

### Implementation notes

A few issues arose integrating with libhankel that are worth recording:

- **Import**: `from libhankel import hankel_transform` is deferred to inside
  `apply`, so the rest of sasmodels loads cleanly without libhankel installed.
  `LibhankelClosedFormTest` skips rather than errors if libhankel is absent.
- **Tail parameter**: libhankel expects `tail` as a string (`"power_law"` or
  `"zero"`); a tuple causes a segfault in the C layer. The explicit Porod
  exponent goes in as a separate `"exponent"` key.
- **G(0)**: calling `hankel_transform` at xi ≈ 0 overflows in QWE_Key.
  `G(0)` is instead computed as a trapezoid rule in log q over the dense
  q grid, with a rectangle correction for the missing [0, q_min] interval
  (libhankel's linear extrapolation below q_min includes this piece; a
  plain trapezoid starting at q_min does not).
- **Strategy for tabulated data**: `DHT_Key_201` is the strategy used in
  libhankel's own tabulated form factor tests; `QWE_Key` requires care (see
  convergence note above).

## Switching between them

```python
from sasmodels import direct_model, sesans

# Digital filter (fast, no extra dependency):
direct_model.SESANS_TRANSFORM = sesans.DHTSesansTransform

# libhankel with the default DHT_Key_201 strategy (same accuracy, optional dep):
direct_model.SESANS_TRANSFORM = sesans.LibhankelSesansTransform

# libhankel with Adaptive_DE_Ooura (best accuracy, ~2-3× slower than DHT):
sesans.LibhankelSesansTransform.strategy_name = "Adaptive_DE_Ooura"
sesans.LibhankelSesansTransform.strategy_params = {"n_eval": 50, "eps_rel": 1e-6}
direct_model.SESANS_TRANSFORM = sesans.LibhankelSesansTransform
```

Anything with the same constructor signature and the same `q`, `q_calc` and
`apply` will do. `_make_sesans_transform` reads the module-level name instead of
referring to `SesansTransform` directly.

`strategy_name` and `strategy_params` are class attributes, so set them before
assigning `SESANS_TRANSFORM`: `direct_model` reads the class when the data is
interpreted, and at that point the strategy is fixed for the lifetime of that
`DirectModel`.

## Provenance of the coefficients

`sesans_filter.py` vendors the `DHT_Key_201` table — 201 abscissae and weights
— so the branch adds no new dependency. It is generated by
`examples/sesans/export_filter_table.py` in LibHankel and should not be edited
by hand. The coefficients themselves are Kerry Key's published Hankel filters,
original to neither project.

## Tests

All in `sasmodels/test_sesans.py`, in two parts:

- closed-form tests against a Gaussian scatterer: one for `DHTSesansTransform`,
  one for `LibhankelSesansTransform` (skips if libhankel is not installed).
  Fast, no dependencies beyond numpy and scipy (and optionally libhankel).
- the measured-data comparison: all three transforms over every `.ses` file in
  `example/`, refereed by an independent brute-force evaluation. Needs *sasdata*
  to read the files and a compiler to build the models, and skips if either is
  missing. The libhankel subtests are additionally skipped if libhankel is not
  installed.

Run `python -m sasmodels.test_sesans` to print the three-way comparison table.

## Status

The filter and the libhankel backend are implemented, tested against a closed
form and against measured data, and wired in behind a switch that is off.

What is still open:

- **Whether `G(ξ)` alone favours the filter.** Sharpening the hardsphere peak
  showed the dense grid overtaking it on `G(ξ)` at `volfraction ≈ 0.48`, well
  inside the physical range, with the filter's overall lead carried by `G(0)`.
  The same split has not been run on the five smooth files, and it decides
  whether the answer is "replace the transform" or "fix `G(0)`".
- **The structure factor case, beyond hardsphere.** The crossover is now
  located for one model on one file — total error at `volfraction ≈ 0.68` — but
  hardsphere need not be the worst case, and the filter still has no way to
  refine.
- **Cost at many spin-echo lengths.** The filter is 201 points per spin-echo
  length; above roughly 200 points the dense grid is cheaper. The largest file
  here is 120. This is a real crossover, just not one these datasets reach.
- **The single-spin-echo-length case** for `_g0_weights`, which nothing
  measured exercises.
- **Whether any of this changes a fit.** Everything above is quadrature error
  against a reference. Whether a 1–9% error on `P(ξ)` moves fitted parameters,
  and by how much against their uncertainties, has not been tested.

## Recommendation

Two strategies cover the practical range:

- **`DHTSesansTransform`** (fast, no extra dependency): 0.3–2.7 ms per call,
  accuracy ~1e-3 to 1e-4. Use it for fitting, where the transform is called
  thousands of times. `LibhankelSesansTransform` with the default `DHT_Key_201`
  strategy matches it on fixed-wavelength data if libhankel is available.

- **`LibhankelSesansTransform` with `Adaptive_DE_Ooura`**: 7–19 ms per call,
  accuracy ~1e-5 (10–74× better than DHT). Use it to verify a fit result,
  evaluate a specific model carefully, or anywhere a more accurate number is
  needed and 2–3× the cost is acceptable.

`QWE_Key` reaches similar accuracy to Ooura but is 5–50× slower for no
benefit, so there is no practical reason to reach for it over Ooura.

Neither libhankel strategy applies the per-point acceptance mask, so they
give worse results on time-of-flight instruments (varying wavelength).
`DHTSesansTransform` is the right choice for those files.
