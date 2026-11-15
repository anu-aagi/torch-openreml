# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.post`
**Purpose of Module:**
Computes post-hoc quantities from a fitted linear mixed model: fixed-effect estimates (BLUEs), random-effect estimates (BLUPs), conditional and marginal predictions, residuals, their covariance matrices, and the REML log-likelihood. Every function is a pure tensor computation over a response vector `y`, design matrices `X` and `Z`, and covariance matrices `G` (random effects) and `V` (marginal response covariance, `V = Z G Z' + R`). The module is the read-only reporting layer used after `MarginalREML`/`MixedModelREML` has produced a `V`.

**Functions Covered:**

- `blue(y, x, v)` — fixed-effect estimate `(X'V⁻¹X)⁻¹ X'V⁻¹y`
- `cov_blue(x, v)` — covariance of the BLUE, `(X'V⁻¹X)⁻¹`
- `var_blue(x, v)` — diagonal of `cov_blue`
- `marginal_predict(y, x, v)` — fixed-effect prediction `Xβ̂`
- `marginal_residual(y, x, v)` — marginal residual `y − Xβ̂`
- `blup(y, x, z, g, v)` — random-effect estimate `G Z' V⁻¹ e`
- `cov_blup(x, z, g, v)` — covariance of the BLUP, `G Z' P Z G`
- `var_blup(x, z, g, v)` — diagonal of `cov_blup`
- `predict(y, x, z, g, v)` — conditional prediction `Xβ̂ + Zb̂`
- `residual(y, x, z, g, v)` — conditional residual `y − predict`
- `loglik(y, x, v)` — REML log-likelihood `−½(log|V| + log|X'V⁻¹X| + e'V⁻¹e)`

**Testing Goal:**
Pin the documented formulas on small hand-computable fixtures, the algebraic identities between the functions (`predict = marginal_predict + Z·blup`, `residual = y − predict`, marginal variants of the conditional ones), shape and dtype/device behaviour, the input-validation paths inherited from `torch_openreml.utils` and `torch.linalg`, and the zero-fixed-effects / zero-random-effects / singular-`G` edge cases. The module had no tests before this file.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Public Surface and Signatures

- `torch_openreml` re-exports the seven functions that the package documents (`blue`, `blup`, `loglik`, `marginal_predict`, `marginal_residual`, `predict`, `residual`)
- `torch_openreml.post` exposes all eleven public callables, including the four covariance helpers not re-exported by `__init__` (`cov_blue`, `cov_blup`, `var_blue`, `var_blup`)
- Exact positional parameter lists and no default arguments for all eleven functions

#### B. `blue`

- Exact value on an orthogonal, hand-computable fixture (`β = [2, 0.5]`)
- Agreement with the textbook GLS formula `(X'V⁻¹X)⁻¹X'V⁻¹y`
- Agreement with explicit weighted least squares when `V` is diagonal
- Scale invariance in `V` (`blue(y, x, cV) == blue(y, x, V)` for `c > 0`)
- Constant response recovers the intercept with a zero slope
- Single fixed effect (`p = 1`): the value is the sample mean
- Single fixed effect: shape is `(p,)`, i.e. `(1,)` (regression-tested after the `.squeeze(-1)` fix)
- No fixed effects (`p = 0`) returns an empty `(0,)` tensor
- Row permutation invariance of `(y, X, V)`

#### C. `cov_blue` / `var_blue`

- Agreement with `(X'V⁻¹X)⁻¹`
- Symmetry of the covariance matrix
- `var_blue` equals the diagonal of `cov_blue` and the explicit diagonal formula
- Positive diagonal for positive-definite `V`
- `(1, 1)` / `(1,)` shapes and exact value `0.25` for an intercept-only design with `V = I` and `n = 4`
- `(0, 0)` / `(0,)` shapes with no fixed effects
- Linear scaling with `V` (`cov_blue(x, cV) == c·cov_blue(x, V)`)

#### D. `loglik`

- Exact value on the orthogonal fixture: `−½(log 20 + 1)`
- Agreement with the explicit REML expression in `V`
- Agreement with the quadratic form `−½(log|V| + log|X'V⁻¹X| + y'Py)`
- Single fixed effect hand value: `−½(log 4 + 5)`
- No fixed effects reduces to the marginal likelihood `−½(log|V| + y'V⁻¹y)`
- Quadratic form equals `e'V⁻¹e` with `e = marginal_residual(y, x, v)` (both sides use the same ridge)
- Scalar (`0-d`), `float64` result

#### E. `marginal_predict` / `marginal_residual`

- `marginal_predict == X @ blue`
- Exact orthogonal-fixture values (`Xβ` and `e`)
- `marginal_residual == y − marginal_predict`
- Residual orthogonality to the design in the `V` metric (`X'V⁻¹e = 0`)
- No fixed effects: prediction is zero and residual is `y`
- `p = 1` prediction and residual: shapes `(n,)` and values agree with `X @ blue` and `y - marginal_predict`

#### F. `blup`

- Agreement with `G Z' V⁻¹ (y − Xβ̂)`
- Agreement with `G Z' V⁻¹ (y − marginal_predict)` (identical ridge on both sides)
- Zero `G` returns zeros
- Zero random effects (`q = 0`) returns an empty `(0,)` tensor
- No fixed effects: `G Z' V⁻¹ y`
- Singular non-zero `G` (rank-one `ones(2, 2)`) is accepted and matches the formula — no `G` inversion is performed
- Single random effect: `(1,)` shape and formula value
- `p = 1`: works through `marginal_residual` and returns the formula value

#### G. `cov_blup` / `var_blup`

- Agreement with `G Z' P Z G`, where `P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹`
- Symmetry and positive semidefiniteness
- `var_blup` equals the diagonal of `cov_blup`
- Block identity `G − C22`, where `C` is the inverse of the mixed-model coefficient matrix built with **`R⁻¹`** (`R = V − ZGZ'`), not `V⁻¹`
- Zero `G` returns a zero matrix and zero variances
- Singular non-zero `G` is accepted and matches the formula (the docstring says only `V` must be invertible)
- Zero random effects returns `(0, 0)` / `(0,)`
- No fixed effects: `G Z' V⁻¹ Z G`
- Single random effect: `(1, 1)` shape and formula value

#### H. `predict` / `residual`

- `predict == marginal_predict + Z @ blup`
- `predict == X @ blue + Z @ blup`
- `residual == y − predict`
- Zero random effects makes both equal their marginal counterparts
- Zero `G` makes `predict` equal `marginal_predict`
- No fixed effects makes `predict` equal `Z @ blup`
- `p = 1` prediction works (it does not go through `blue`), with the correct value
- `n = 1` residual restores the `(1,)` shape by broadcasting
- `n = 1` prediction returns the documented `(1,)` (regression-tested after the `.squeeze(-1)` fix)

#### I. Input Validation and Error Paths

- Mixed dtypes across arguments → `ValueError` matching `"Dtype mismatch"` for all eleven functions
- Non-positive-definite `V` → `torch.linalg.LinAlgError` for all eleven functions
- Rank-deficient `X` (zero column, so `X'V⁻¹X = diag(a, 0)`) → `torch.linalg.LinAlgError` for all eleven
- Wrong `y` length → `RuntimeError` for the seven functions taking `y`
- Wrong `Z` row count or wrong `G` shape → `RuntimeError` for the five `Z`/`G` functions
- Wrong `V` shape (e.g. `5 × 5` for `n = 6`) → `RuntimeError` for all eleven

#### J. Dtype and Device Contract

- Every function preserves `float32` input and returns finite values
- Every function runs on the first available accelerator (`cuda` → `mps` → skip) and agrees with the CPU `float32` result
- Mixing an accelerator tensor with CPU tensors → `ValueError` matching `"Device mismatch"` for all eleven
- No function mutates any input tensor

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Build the fixtures directly with `torch`, so every expected value is either hand-computable or an explicit textbook formula:
  - `_lmm_fixture(dtype)` — `n = 6`, `p = 2`, `q = 2`, `G = [[0.6, 0.2], [0.2, 0.4]]`, diagonal `R`, and `V = ZGZ' + R`.
  - `_exact_fixture()` — `n = 4`, `V = I`, `X` columns orthogonal (intercept and `[-1.5, -0.5, 0.5, 1.5]`), `β = [2, 0.5]`, `e = [0.5, -0.5, -0.5, 0.5]` orthogonal to `X`; hence `blue = β`, `e'e = 1`, `X'X = diag(4, 5)`, `loglik = −½(log 20 + 1)`.
- Use `torch.allclose` with explicit tolerances:
  - `FORMULA = {rtol: 1e-5, atol: 1e-5}` for comparisons against formulas in `V`. The implementation silently factorises `V + 1e-6 I`, so the textbook expression differs at the `1e-6` level.
  - `EXACT = {rtol: 1e-10, atol: 1e-10}` for identities where both sides go through the same ridge (`predict` vs `marginal_predict + Z·blup`, `residual` vs `y − predict`, `var_*` vs the diagonal of `cov_*`).
  - `ACCEL = {rtol: 1e-3, atol: 1e-3}` for accelerator `float32` comparisons against CPU `float32`.
- Parameterise dtype/device and error-path tests over the eleven function names through a `_call(name, ...)` dispatcher, so each function is exercised without duplicating the case logic.
- Reference formulas use `torch.linalg.inv`/`solve` on the **unjittered** `V`; identity tests that need an exact match reconstruct the jittered matrix with `_ridge(v)`.
- Device tests follow the suite convention: pick `cuda` → `mps` → `pytest.skip("no accelerator available")` inline. MPS does not support `float64`, so the device-mismatch and accelerator tests build `float32` fixtures.
- Do not modify the library and do not set the global default dtype inside tests: every fixture specifies `dtype` explicitly.

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**, grouped by function:

```python
class TestBlueValues:
    def test_matches_explicit_gls_formula(self):
        # Arrange
        y, x, _, _, _, v = _lmm_fixture()
        expected = _gls_beta(y, x, v)

        # Act
        beta = blue(y, x, v)

        # Assert
        assert torch.allclose(beta, expected, **FORMULA)
```

Expected values are never copied from the implementation; they come from `_gls_beta`, `_p_matrix`, `_explicit_loglik`, or direct hand calculation.

### 3.3 Test Execution

```bash
conda run -n torch-openreml python -m pytest tests/test_post.py -q
```

Result on 2026-11-15 (torch 2.14.0, Python 3.12.12, CPU + MPS):

```text
175 tests collected
175 passed in 1.70s
```

All 175 tests pass; the former squeeze failures are now fixed in `post.py` and covered as regressions.

---

## 4. Key Risks / Notes

- **`p == 1` / `n == 1` shapes (fixed).** `blue` previously ended with `.squeeze()` (`post.py:87`), collapsing a single-column estimate to 0-d and breaking `marginal_predict`, `marginal_residual` and `blup`; `predict` had the same problem at `post.py:387`. Both now use `.squeeze(-1)`, so the documented `(p,)` / `(n,)` shape is preserved, and the five former xfails run as ordinary regression tests.
- **Ridge is documented (was a gap).** Every function factorises `V + 1e-6·I`, not `V`; the module docstring and the API page now state the fixed ridge. Formula-level tests still use `1e-5` tolerances, and tests that compare two library functions use `1e-10` because both sides see the same ridge.
- **No validation of `V`'s shape/rank.** The docstring says `v` is `(n, n)`, but a 1-D tensor of length `n` silently broadcasts against `1e-6·eye(n)` and produces an `(n, n)` matrix. No test asserts a specific error for this because the contract only specifies the shape; noted as a robustness observation.
- **Singular designs are not reliably detected.** An exactly collinear design such as `torch.ones(n, 2)` can still factorise because round-off leaves a tiny positive pivot; `blue` then returns a finite but meaningless estimate instead of raising. The error-path test uses a zero column (`X'V⁻¹X = diag(a, 0)`), which deterministically raises `LinAlgError`. This is a library robustness limitation, not a documented contract.
- **`cov_blup` block identity uses `R⁻¹`, not `V⁻¹`.** The docstring's `G − C22` form refers to the mixed-model coefficient matrix built from the residual covariance `R = V − ZGZ'`. Building it with `V⁻¹` produces a different matrix; the test uses `R⁻¹` and matches `cov_blup` to `~1e-7` (ridge-level).
- **No `G` inversion.** `blup` and `cov_blup` are well-defined for singular `G`, including `G = 0`. This is tested explicitly; the zero-`G` and rank-one-`G` cases return zeros / formula values rather than raising.
- **MPS dtype limitation.** MPS cannot hold `float64`. Accelerator and device-mismatch tests use `float32`; the `float64` formula tests run on CPU. If no accelerator is available, the four device tests skip instead of failing.
- **No mutations.** None of the functions writes to `y`, `x`, `z`, `g`, or `v`; the parametrised immutability test guards against future regressions.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Public surface and signatures | 13 |
| `blue` | 9 |
| `cov_blue` / `var_blue` | 8 |
| `loglik` | 6 |
| `marginal_predict` / `marginal_residual` | 7 |
| `blup` | 8 |
| `cov_blup` / `var_blup` | 10 |
| `predict` / `residual` | 9 |
| Input validation and error paths | 61 |
| Dtype and device | 44 |
| **Total** | **175** |

All **175 collected tests pass**; there are no xfails.

Breakdown of the parametrised categories:

- Input validation: 11 dtype mismatches, 11 non-PD `V`, 11 rank-deficient `X`, 7 wrong `y` lengths, 5 wrong `Z` rows, 5 wrong `G` shapes, 11 wrong `V` shapes = 61.
- Dtype/device: 11 `float32` preservation, 11 accelerator runs, 11 device mismatches, 11 input-immutability checks = 44.
