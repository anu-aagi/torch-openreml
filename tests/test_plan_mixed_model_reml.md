# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.mixed_model_reml`
**Purpose of Module:**
Fits linear mixed-effects models by AI-REML. The class owns the random-effect terms (design matrix `Z` and covariance `G` per term), the residual covariance `R`, and assembles the marginal response covariance `V = Z G Z' + R`. It selects between two evaluation backends — the C-form (`CForm`, which factorises the mixed-model coefficient matrix) and the marginal form (`Marginal`, which factorises `V`) — and exposes the REML log-likelihood, score vector, average-information matrix, an `optimize` loop, and the usual post-fit quantities (BLUE, BLUP, predictions, residuals, covariances).

**Classes Covered:**

- `MixedModelREML` — the estimator under test
- `CForm` and `Marginal` — exercised through `method=`, including instance reuse and fixed-only rejection
- `MarginalCovariance`, `Augment`, `BlockDiagonal` — the covariance assembly contracts the estimator relies on
- `DummyMatrix`, `ScalarMatrix`, `KnownMatrix` — the design/covariance fixtures used throughout
- `post.py` helpers — exercised indirectly through `blue`, `blup`, `predict`, `residual`, `loglik`

**Testing Goal:**
Pin the public surface and constructor validation, the design/covariance assembly, backend selection, the shape and symmetry of the score/AI matrices, optimizer convergence and history on a small synthetic data set and on the `john_alpha` example data, the post-fit accessors and their algebraic identities, fixed-effect and random-effect recovery on a known model, dtype/device propagation, and the fixed-only (no random effects) configuration. The module had no direct tests before this file. Exact values are avoided where the optimizer is involved; the assertions are convergence, finiteness, algebraic identities and physically plausible variance components.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Public Surface and Construction

- `MixedModelREML` is importable from `torch_openreml`
- A single positional term is stored as `random_term_0`; further positional terms are numbered in order
- Keyword terms keep their names (`rep`, `block`, …)
- A single term keeps the supplied objects by identity (`model.Z is z`, `model.G is g`)
- The default residual is `ScalarMatrix(n)`, with `n` inferred from the assembled `Z`
- An explicit `R=ScalarMatrix(n)` is kept; a fixed 2-D tensor `R` is wrapped in `KnownMatrix`
- `V` is a `MarginalCovariance` with operands `Z`, `G`, `R`; `V.free_param_names` and `num_free_params` reflect the parameterised terms
- Accessors `V`, `R`, `G`, `Z`, `G_terms`, `Z_terms`; `repr(model) == repr(model.V)`
- `method="auto"` builds both a `CForm` and a `Marginal` backend

#### B. Constructor Validation

- A positional term that is not a pair → `TypeError` naming the index
- A trailing positional term after pairs → `TypeError` naming its index
- A keyword term that is not a pair → `ValueError` naming the keyword
- A term of the wrong length → `ValueError` naming the term
- A design that is not a tensor/`Matrix` → `TypeError`
- A 1-D covariance tensor → `ValueError`; another covariance type → `TypeError`
- A duplicate term name → `ValueError`
- A non-`Matrix`/non-2-D-tensor `R` → `TypeError`; a 1-D `R` → `ValueError`
- No random terms and no `R` → `ValueError`
- A `V` with no free parameters (e.g. fixed-only with a fixed tensor `R`) → `ValueError`

#### C. Design/Covariance Assembly

- Single term: `V = MarginalCovariance(Z=z, G=g, R=r)` with the expected shape
- Multiple terms: `Z` is an `Augment` when any design is a `Matrix`, otherwise `torch.cat`; `G` is a `BlockDiagonal`
- Parameter namespaces follow `G/<term>/sigma^2`, `R/sigma^2`; the free-parameter order is the term order followed by `R`
- `tree()` exposes `/`, `Z`, `G`, `R`, and per-term nodes `Z/<name>`, `G/<name>` for an `Augment`/`BlockDiagonal` assembly; `/` evaluates to `V`, `Z/<name>` to the term design, `G/<name>` to the term covariance
- A fixed-only model has `tree() == {"/": V}`

#### D. Backend Selection (`method=`)

- `"auto"` (the default) uses `CForm` for an eligible model and reports `"cform"`
- `"marginal"` reports `"marginal"` from `loglik` and `ai_step`
- `"cform"` reports `"cform"`
- An unknown string behaves like `"auto"` and builds both backends
- A `Marginal`/`CForm` instance is stored as `model.method` and reused as `model._marginal`/`model._cform`
- C-form and marginal log-likelihoods agree at the same parameter vector (`atol=1e-4`)
- A fixed-only model with `method="cform"` raises `RuntimeError` from `loglik` and `ai_step`

#### E. `ai_step` — Score and Average Information

- Returns `(score (k,), ai (k, k), loglik scalar, method)`
- The AI matrix is symmetric and positive semidefinite
- The score matches a central finite difference of `loglik` in `float64`
- `require_loglik=False` keeps the score/AI but returns `nan` for the log-likelihood
- A wrong-length parameter vector raises `ValueError`

#### F. `optimize`

- Returns `(names, free_params)` matching `V.free_param_names`, with finite values
- History keys: `free_params`, `loglik`, `score`, `ai`, `delta`, `update`, `eta`, `accepted`, `method`; per-iteration entries are consistent in length
- The fit converges, the final log-likelihood is finite and no lower than the first
- Fitted variance components (after the `exp(2θ)` transform) are positive
- `get_free_params("last")` is the final iterate; `get_free_params("best")` is the argmax of the history log-likelihood
- `max_iter=1` stops after one iteration and is not converged
- The training `y`/`x` are stored so the post-fit accessors work without arguments
- Wrong-length `init_values` raises `ValueError`

#### G. Post-Fit Accessors

- Shapes: `blue (p,)`, `blup (q,)`, `predict (n,)`, `residual (n,)`, scalar `loglik`
- Covariance shapes: `cov_blue (p, p)`, `var_blue (p,)`, `cov_blup (q, q)`, `var_blup (q,)`; covariances symmetric
- `c_inverse_blocks` returns `(c11 (p, p), c12 (p, q), c21 (q, p), c22 (q, q))` with `c12 == c21.T` and `c11 == cov_blue`
- `predict == X @ blue + Z @ blup`; `residual == y - predict`
- `marginal_predict == X @ blue`; `marginal_residual == y - marginal_predict`
- Per-term accessors are dicts keyed by term name; the single-term values equal the joint quantities
- `tree()` keys and values as in §C; `build_params(trans=True)` equals `exp(2 * build_params(trans=False))`; dict output keys match the free-parameter names
- `loglik()` matches the last history entry and reports the backend

#### H. Multi-Term Models

- Two named terms produce `Augment`/`BlockDiagonal` assembly and three free parameters
- Per-term BLUPs concatenate to the joint BLUP
- `cov_blup_per_term` blocks equal the corresponding diagonal blocks of the joint covariance
- `tree()` contains `Z/rep`, `Z/block`, `G/rep`, `G/block`

#### I. Fixed-Only Models (No Random Effects)

- `Z`/`G` are `None`; `V` is the parameterised `R`; one free parameter
- `loglik` uses the marginal backend
- REML estimates the residual variance; with `V = σ²I` it equals the OLS residual variance `e'e / (n - p)`
- `blue` equals the OLS solution
- `predict`/`residual` reduce to the marginal quantities
- BLUP accessors raise `ValueError` ("no random-effect terms")
- `c_inverse_blocks` raises `RuntimeError` (C-form needs random effects)
- `predict_new` rejects random blocks with `ValueError`

#### J. Closed-Form Agreement

- `blue` matches the GLS solution `(X'V⁻¹X)⁻¹X'V⁻¹y`
- `blup` matches `G Z' V⁻¹ (y - Xβ̂)`
- Both use `V(fitted_params) + 1e-6·I`, matching the ridge that `post.py` adds

#### K. Fixed/Random-Effect Recovery

- On `y = Xβ + Zu + e` (`n = 50`, 5 groups, seed 0), the fixed effects are recovered within `atol=0.3`
- The BLUPs correlate with the simulated random effects (`r > 0.9`)
- The group and residual variance components are positive and physically plausible
- The conditional predictions correlate with the response (`r > 0.9`)

#### L. dtype and Device Propagation

- `V()` uses `float32`; `V(float64 params)` follows the parameter dtype
- A full fit initialised with `float64` yields `float64` for BLUE, BLUP, predictions, covariances, log-likelihood and history
- `optimize(init_values=None)` keeps the covariance default (`float32`) even when `y`/`x` are `float64`
- A fit runs on the first available accelerator (`cuda` → `mps` → skip) and the results stay on that device

#### M. `john_alpha` Example Data

- Fixture: 72 rows, `y = yield`, `x = [1, row]`, terms `rep` (3), `block` (6), `gen` (24), `R = ScalarMatrix(72)`
- The fit converges, reports `"cform"`, and the log-likelihood is finite
- Variance components are finite and plausible: `rep > 0.01`, `gen > 0.01`, `0 <= block < 1`, `0.02 < residual < 0.5`
- Post-fit shapes: `blue (2,)`, `blup (33,)`, `predict/residual (72,)`, `cov_blue (2, 2)`, `cov_blup (33, 33)`
- Per-term BLUPs have shapes `(3,)`, `(6,)`, `(24,)` and concatenate to the joint BLUP
- `predict == X @ blue + Z @ blup`
- `predict_new` accepts a dict with a subset of the random-effect terms
- The canonical intercept-only fit (the documented example configuration) also converges with finite log-likelihood and positive variance components; its post-fit BLUE/BLUP are not asserted here; the one-column shape regressions are covered in `TestEdgeContractRegressions`

#### N. Former API Gaps (fixed)

- `blue` with a one-column `X` returns `(1,)`
- `blup` with a one-column `X` returns `(q,)`
- `predict_new` with a one-column `X` returns `(n_new,)`
- `is_converged()` before a fit raises a `ValueError` asking the caller to run `optimize()` first

`ai_step(..., require_loglik=False)` returns the documented `torch.nan`
sentinel. Since `torch.nan` is a Python `float`, the skipped path differs in
type from the computed scalar; that behaviour is asserted by a passing test.

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- A synthetic fixture (`_toy_data`) uses `n = 40`, 5 groups, `DummyMatrix` + `ScalarMatrix(5)` and `R = ScalarMatrix(40)`, with `torch.manual_seed(0)`; `_fit_toy` fits it at `max_iter=100`. The parameter names are `["G/sigma^2", "R/sigma^2"]`.
- A two-term fixture (`_multi_term_model`) uses `rep` (3 levels) and `block` (6 levels) on `n = 60`, giving three free parameters.
- The recovery fixture (`_recovery_data`) uses `n = 50`, 5 groups, `β = [2, -1]`, `u ~ N(0, 1)`, `e ~ 0.25 N(0, 1)`, and a hand-built dummy `Z`. Seed 0 is used because the intercept is noisier under other seeds; the tolerance is deliberately loose (`atol=0.3`, correlation `> 0.9`).
- `john_alpha` uses `x = [1, row]` so that `p > 1`; the intercept-only configuration is fitted in its own test and only convergence/log-likelihood/variance components are asserted. The one-column `p == 1` shapes are covered separately in `TestEdgeContractRegressions`.
- Expected values are algebraic identities or textbook formulas, not numbers copied from the implementation:
  - `predict = Xβ + Zb`, `residual = y - predict`, `marginal_predict = Xβ`.
  - `blue` vs GLS and `blup` vs `GZ'V⁻¹(y - Xβ)`, using `V(fitted_params) + 1e-6·I` to match the ridge inside `post.py`.
  - Fixed-only: REML variance vs `e'e/(n-p)`, BLUE vs OLS.
- Tolerances: `atol=1e-5` for the algebraic identities and closed forms; `atol=1e-4` for the `john_alpha` decomposition (72 observations, float32); loose bounds for recovery.
- Optimizer tests assert convergence, finiteness, monotone improvement and plausible variance components, never exact parameter values.
- Device tests follow the suite convention: pick `cuda` → `mps` → `pytest.skip("no accelerator available")` inline, and build the matrices inside `with torch.device(device):`.
- The former contract failures are ordinary passing tests now that the library fixes landed; no xfails remain in this module.

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**, grouped by contract area:

```python
class TestPostFitShapes:
    def test_predict_and_residual_decompose(self):
        # Arrange
        model, y, x = _fit_toy()
        beta = model.blue()
        blup = model.blup()
        z = model.Z()

        # Act
        prediction = model.predict()
        residual = model.residual()

        # Assert
        assert torch.allclose(prediction, x @ beta + z @ blup, atol=1e-5)
        assert torch.allclose(residual, y - prediction, atol=1e-5)
```

The shared helpers (`_toy_data`, `_toy_model`, `_fit_toy`, `_multi_term_model`, `_recovery_data`) are module-level so each class reads as the contract it pins.

### 3.3 Test Execution

```bash
conda run -n torch-openreml python -m pytest tests/test_mixed_model_reml.py -q
```

Result on 2026-11-15 (torch 2.14.0, Python 3.12.12, CPU + MPS):

```text
108 tests collected
108 passed in 3.32s
```

All 108 tests pass; the former squeeze and pre-fit failures are covered as regressions.

---

## 4. Key Risks / Notes

- **`p == 1` squeeze (fixed).** `post.blue` previously used `.squeeze()` (`torch_openreml/post.py:87`), collapsing a one-column BLUE to 0-d and breaking `blup`, `marginal_predict` and `predict_new`. It now uses `.squeeze(-1)`, so the `(p,)` contract holds; the former xfails are regression tests.
- **`ai_step(require_loglik=False)` returns the `torch.nan` sentinel.** `torch.nan` is a Python `float`, so the skipped path intentionally differs in type from the computed scalar; `test_skipped_loglik_is_torch_nan` asserts `isinstance(loglik, float)` and `math.isnan(loglik)`.
- **`is_converged()` before a fit raises `ValueError`.** The error message asks the caller to run `optimize()` first; this is documented in the method and pinned by `test_is_converged_before_a_fit_raises`.
- **`get_free_params()` before a fit also raises `TypeError`** for the same reason. This is an API gap rather than a documented contract; it is recorded here but not pinned with a test.
- **`ai_step` validates parameter length per term, not globally.** A length-3 vector for a two-parameter model raises `"Parameters must have length 1, got 2!"` (from the `G` build) rather than a model-level message. The test therefore matches the stable prefix `"Parameters must have length"`. `optimize` does raise the global `"Parameters must have length 2, got 3!"`, which is pinned.
- **C-form fallback during optimisation.** `CForm` can fail its Cholesky factorisation at an intermediate iterate and fall back to `Marginal`, emitting `RuntimeWarning: CForm failed (...); falling back to Marginal.` This is expected behaviour (the estimator is designed to fall back), but it makes the reported backend per-iteration rather than constant. The tests assert `"cform"` only on converged fits where the fallback did not occur; a future test of an ill-conditioned fit should not assume a single backend.
- **`john_alpha` block variance collapses to ~7e-13.** The block effect is effectively zero for this data, so the test asserts `0 <= block < 1` rather than a positive lower bound. The `rep` and `gen` components are stable and positive.
- **`john_alpha` intercept-only fit works** and is the canonical example configuration; it converges and reports `"cform"`, and its BLUE/BLUP now have the documented shapes.
- **`V(int64 params)` raises `RuntimeError: expected m1 and m2 to have the same dtype...`** rather than promoting. The dtype tests cover `float32`/`float64` only; the integer case is an undocumented gap.
- **Library fixes are covered as regressions.** The tests document the current implementation and the repaired shape/error contracts; there are no xfails left in this module.
- **Runtime.** The file runs in ~3.3 s, well under the ~60 s budget; the `john_alpha` fixture is module-scoped and fitted once.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction and assembly | 8 |
| Constructor validation | 12 |
| Backend selection (`method=`) | 8 |
| `ai_step` | 6 |
| `optimize` | 8 |
| Post-fit shapes and accessors | 10 |
| Multi-term models | 4 |
| Fixed-only models | 14 |
| Accessor validation | 16 |
| Closed-form agreement | 2 |
| Fixed/random-effect recovery | 4 |
| dtype and device | 4 |
| `john_alpha` example data | 7 |
| **Total** | **108** |

All **108 collected tests pass**; there are no xfails.

Breakdown of the parametrised categories:

- Constructor validation: 12 invalid-construction cases = 12.
- Fixed-only: 8 ordinary tests plus 6 parametrised BLUP accessors = 14.
- Accessor validation: 2 response/design-pairing cases, 7 pre-fit accessor cases, and 7 `predict_new`/`update` tests = 16.
