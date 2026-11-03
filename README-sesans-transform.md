# SESANS transform selection

Three transforms are available. Set `data.sesans_transform` before building a
`DirectModel` to select one:

| key | class | dependency | when to use |
|---|---|---|---|
| `"dense"` | `SesansTransform` | none | legacy default; avoid for new work |
| `"fixed_grid"` | `DHTSesansTransform` | none | fitting — fast, no extra dependency |
| `"libhankel_fftlog"` | `LibhankelSesansTransform` | `libhankel` | accuracy matters — best on most files |

## Quick start

```python
from sasmodels.direct_model import DirectModel

data = load_data("my_file.ses")
data.sesans_transform = "fixed_grid"   # or "libhankel_fftlog"

calc = DirectModel(data, model)
P = calc(**pars)
```

## Which to choose

**`"fixed_grid"`** — use this when you don't want the extra LibHankel dependency, runtime is more important than accuracy. Faster than `"dense"` and more accurate on most files. 

**`"libhankel_fftlog"`** — use this when accuracy matters: verifying a fit
result, careful model evaluation, publication-quality numbers. More accurate
than `"fixed_grid"` on most files and with similar runtimes, but requires the `libhankel`
package. Does not apply the per-point acceptance mask, so gives worse results
on time-of-flight data (varying wavelength); use `"fixed_grid"` there instead.

**`"dense"`** — the original implementation. Slower and less accurate than
`"fixed_grid"` on most files due to truncation of the q integration range.
Kept for backwards compatibility.

## How the transforms differ

All three evaluate the same integral:

```
G(xi) = 1/(2 pi) * integral_0^inf  q J0(q xi) I(q) dq
P(xi) = G(xi) - G(0)
```

**`"dense"`** resolves J0 oscillations by using a very fine q grid (~32–44k
points). The bounds of that grid are heuristic and sometimes truncate the
integrand — that is the source of its errors, and refining the spacing does not
help.

**`"fixed_grid"`** uses a published digital Hankel filter: fixed abscissae
scaled by 1/xi, with J0 absorbed into signed weights. 201 points per spin-echo
length. No bounds to choose, so no truncation error.

**`"libhankel_fftlog"`** uses the same dense q grid as `"dense"` for model
evaluation, then hands the tabulated I(q) to `libhankel`'s FFTLog strategy,
which handles J0 analytically.

## Tests

All in `sasmodels/test_sesans.py`:

- Closed-form test of `DHTSesansTransform` against a Gaussian scatterer (numpy/scipy only).
- Measured-data comparison of all three transforms against a brute-force reference,
  over every `.ses` file in `example/`. Needs *sasdata* and a compiler; skips otherwise.
