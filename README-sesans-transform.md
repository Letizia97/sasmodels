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

## `SesansTransform`: dense log-spaced grid

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

## `DHTSesansTransform`: digital Hankel filter

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

## How G(0) is computed

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
thousands of times. The libhankel column uses `Adaptive_DE_Ooura` with
`n_eval=50, eps_rel=1e-6` — the recommended strategy (see below). Using
`DHT_Key_201` instead gives errors within 1% of the filter column on every
fixed-wavelength file, since it is the same filter table.

| file | n | nq dense | nq filter | t dense | t filter | t Ooura | err dense | err filter | err Ooura |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sphere.ses` | 80 | 41552 | 16080 | 10.7 ms | 1.0 ms | 7 ms | 9.8e-3 | 2.0e-4 | **2.1e-5** |
| `spheres2micron.ses` | 40 | 32216 | 8040 | 9.6 ms | 0.7 ms | 11 ms | 5.0e-4 | 3.5e-4 | **5.5e-6** |
| `core_shell.ses` | 80 | 41552 | 16080 | 17.4 ms | 2.7 ms | 8 ms | 5.3e-2 | 4.2e-3 | **5.7e-5** |
| `se008724_01.ses` | 120 | 44005 | 24120 | 11.2 ms | 1.3 ms | 19 ms | 1.7e-4 | 3.0e-4 | **2.2e-5** |
| `se008731_01_40pcorr.ses` | 40 | 32216 | 8040 | 11.0 ms | 0.6 ms | 8 ms | 5.0e-4 | 3.5e-4 | **5.5e-6** |
| `SiO2_100pc_H2O_0pc_D2O.ses` | 25 | 32691 | 5025 | 8.8 ms | 0.3 ms | 3 ms | 9.0e-2 | **5.3e-5** | 2.4e-3 |

The SiO2 file (time-of-flight, varying wavelength) is the exception: the filter
applies a per-point acceptance mask that libhankel cannot replicate, so all
libhankel strategies are 45× worse on that file regardless of which quadrature
is used.

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

## `LibhankelSesansTransform`: adaptive quadrature via libhankel

`LibhankelSesansTransform` uses the same dense log-spaced q grid as
`SesansTransform` but delegates the integration to `libhankel.hankel_transform`
rather than doing the matrix multiply itself. The strategy is selectable at
runtime; the recommended one is `Adaptive_DE_Ooura`.

### `Adaptive_DE_Ooura`: the recommended strategy

`Adaptive_DE_Ooura` is a double-exponential method designed for oscillatory
Bessel-function integrals. It handles J₀ oscillations analytically rather than
resolving them numerically, so it converges in far fewer evaluations than a
general-purpose adaptive rule.

```python
sesans.LibhankelSesansTransform.strategy_name = "Adaptive_DE_Ooura"
sesans.LibhankelSesansTransform.strategy_params = {"n_eval": 50, "eps_rel": 1e-6}
```

Results on the measured files (same reference as the accuracy table above):

| file | err filter | err Ooura | factor | t filter | t Ooura |
| --- | --- | --- | --- | --- | --- |
| `sphere.ses` | 2.0e-4 | **2.1e-5** | 9× better | 1.0 ms | 7 ms |
| `spheres2micron.ses` | 3.5e-4 | **5.5e-6** | 64× better | 0.7 ms | 11 ms |
| `core_shell.ses` | 4.2e-3 | **5.7e-5** | 74× better | 2.7 ms | 8 ms |
| `se008724_01.ses` | 3.0e-4 | **2.2e-5** | 14× better | 1.3 ms | 19 ms |
| `se008731_01_40pcorr.ses` | 3.5e-4 | **5.5e-6** | 64× better | 0.6 ms | 8 ms |
| `SiO2_100pc_H2O_0pc_D2O.ses` | **5.3e-5** | 2.4e-3 | 45× worse (masking) | 0.3 ms | 3 ms |

Ooura is 9–74× more accurate than the filter at only 2–3× the cost on
fixed-wavelength files. The `n_eval` parameter has no effect in this range;
`n_eval=50` already converges fully. The convergence criterion (`eps_rel`)
does the work.

The SiO2 file is worse at every libhankel strategy — see the acceptance mask
note above.

### Other strategies

**`DHT_Key_201`** (the default) uses the same Kerry Key filter table as
`DHTSesansTransform`. It gives identical accuracy on fixed-wavelength files
and is the right choice if libhankel is available but accuracy is not the
priority.

**`QWE_Key`** is an adaptive Gauss–Kronrod rule that reaches the same accuracy
as Ooura but is 5–50× slower (34–391 ms per call), because it resolves J₀
oscillations numerically one sub-interval at a time rather than analytically.
There is no practical reason to use it over Ooura.

**False convergence warning.** At `eps_rel=1e-4` QWE_Key reports convergence
but gives errors up to 1.79e-1 — three orders of magnitude off.
**Do not use QWE_Key with `eps_rel` looser than 1e-6 on tabulated data.**

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
