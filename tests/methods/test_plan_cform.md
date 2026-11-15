# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.methods.cform`
**Purpose of Module:**
Implements the C-form (coefficient-form) evaluation of the marginal REML
score, average-information matrix and restricted log-likelihood. Instead of
forming `V` and `P = V^{-1} - V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}`, it factorises
the `(p + q) x (p + q)` coefficient matrix
`C = W'R^{-1}W + blockdiag(0, G^{-1})` with `W = [X, Z]`, which is cheaper
when the random-effect dimension `q` is small.

**Classes Covered:**

- `CForm` — stateless evaluator with model-aware and tensor-level entry points

**Testing Goal:**
Pin the full public contract of `CForm`: eligibility, the restricted
log-likelihood, the score vector and AI matrix on models small enough to check
against explicit matrix formulas and `torch.autograd`, the blocks of `C^{-1}`,
the tensor-extraction contract of `preprocess_input`, dtype/device
propagation, and agreement with the marginal method on a realistic
`john_alpha` model. The implementation is treated as the specification only
where it is self-consistent; where the source and docstring disagree the test
encodes the documented contract; the former fixed-tensor dtype gap is now
covered as a regression and there are no xfails.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Construction, Statelessness and Eligibility

- `CForm()` takes no arguments and stores no instance state (`vars(cform) == {}`)
- `eligible(model)` is `True` for a model with a random-effect term and at
  least one free marginal-covariance parameter
- `eligible(None)` is `False`
- `eligible` is `False` for a model without random effects (`Z`/`G` are `None`)
- `eligible` is `False` when the marginal covariance reports
  `num_free_params == 0`
- `eligible` needs only one free parameter, regardless of where it lives
- `loglik`, `ai_step` and `c_inverse_blocks` all raise
  `RuntimeError("CForm requires at least one random-effect term with free parameters.")`
  for an ineligible model, and for `model=None`
- The check is repeated in every entry point, so no entry point silently
  accepts an ineligible model

#### B. Restricted Log-Likelihood

- `loglik` matches the explicit C-form expression
  `-0.5 (logdet C + logdet R + logdet G + quad)` with
  `quad = y'R^{-1}y - (W'R^{-1}y)' C^{-1} (W'R^{-1}y)`
- `loglik` matches the explicit marginal expression
  `-0.5 (logdet V + logdet(X'V^{-1}X) + y'Py)` on the same model, confirming
  the two algebraic forms agree
- `loglik` agrees with `Marginal.loglik` within the residual jitter tolerance
  (`atol=1e-5`), since the marginal path jitters a composite `V` by `1e-6 I`
- `loglik_from_tensors` accepts a supplied `r_inv`, a supplied `g_inv`, and
  both, and each variant reproduces the computed-inverse value to roundoff
- Mismatched row counts between `x`, `y`, `z` and `r` raise `RuntimeError`
  from the linear algebra
- A non-positive-definite `r` raises `RuntimeError` from the Cholesky
  factorisation even when `r_inv` is supplied (the log-determinant path
  always factorises `R`)

#### C. Score Vector and AI Matrix

- The score has shape `(k,)` with `k = kz + kg + kr`
- Each score entry matches the explicit P-form reference
  `s_k = 0.5 (y'P dV_k P y - tr(P dV_k))`, evaluated with `V = ZGZ' + R` and
  `dV_k` from `model.V.grad`
- The score equals `torch.autograd.grad` of `CForm.loglik` with respect to
  `free_params` (CForm does not touch `model.V`, so the marginal covariance's
  intermediate cache cannot detach this graph)
- The AI matrix matches the explicit reference
  `AI_{kl} = 0.5 q_k' P q_l` with `q_k = dV_k P y`
- The AI matrix is symmetric to roundoff and positive semidefinite
  (`eigvalsh >= -1e-12`)
- `require_loglik=False` returns `torch.nan` for the log-likelihood and
  bitwise-identical score and AI values
- Parameter order is `[Z, G, R]`, verified with a `SimpleMatrix` design that
  contributes two `Z` parameters ahead of the `G` and `R` variances
- `ai_step_from_tensors` rejects the case where all of `dz`, `dg` and `dr`
  are empty with
  `ValueError("at least one of dz, dg, or dr must contain a parameter")`
- A derivative block whose trailing dimensions disagree with `z`, `g` or `r`
  raises `RuntimeError`
- A `KnownMatrix` random effect contributes no `G` score entry (`dg` is empty)
  and the remaining entries still match the reference
- A `KnownMatrix` residual contributes no `R` score entry (`dr` is empty)
- A model with parameters only in the design (`Z` free, `G` and `R` known)
  produces a full-length score and AI matrix that match the reference

#### D. Inverse Coefficient Blocks

- `c_inverse_blocks` returns four blocks with shapes
  `(p, p)`, `(p, q)`, `(q, p)`, `(q, q)`
- The four blocks reassemble into `torch.linalg.inv(C)` for the same `C`
- `c11` equals `(X'V^{-1}X)^{-1}` to roundoff
- `c12 == c21.T` exactly
- Supplying `g_inv`/`r_inv` reproduces the computed-inverse blocks
- When `R` has no closed-form inverse (`SimpleMatrix` residual), `R` is
  jittered by `1e-6 I` and the blocks match an explicit inverse built from the
  jittered `R`; `G` follows the Cholesky path without jitter
- The block dtype follows `free_params` (float32 parameters give float32
  blocks through a `DummyMatrix` design)

#### E. `preprocess_input` Contract

- With `grad=True` the returned dict has exactly
  `x, y, z, g, r, g_inv, r_inv, dz, dg, dr`
- With `grad=False` it has exactly `x, y, z, g, r, g_inv, r_inv`
- For the fixed-design tiny model the derivative shapes are
  `dz=(0, n, q)`, `dg=(kg, q, q)`, `dr=(kr, n, n)`
- For a parameterised design the `dz` block is `(2, n, q)`
- `x` and `y` are cast to `free_params`' dtype (and device)
- `r_inv` is `None` and `r` is jittered exactly when `R.manual_inverse` raises
  `NotImplementedError`; when a closed-form inverse exists, `r` is untouched
  and `r_inv` is populated
- A `G` without a closed-form inverse yields `g_inv=None` while `g` is the
  untransformed covariance
- `z`, `g` and `r` are sliced from `free_params` in `[Z, G, R]` order

#### F. dtype and Device Propagation

- Model-aware estimates follow the free-parameter dtype for float32 and
  float64
- The tensor-level kernel follows its input dtype for float32 and float64
- The model-aware entry point follows the free-parameter device; the test
  picks `cuda` → `mps` → `pytest.skip("no accelerator available")` inline and
  compares `device.type` because MPS reports `mps:0`
- A fixed-tensor `Z` is cast to the free-parameter dtype and device before
  the first product, so float32 parameters with a float64 design work and
  return float32 estimates

#### G. Realistic John's Alpha Case

- The single-genotype `john_alpha` model (`n = 72`) reproduces the explicit
  C-form log-likelihood
- Its score and AI matrix match the explicit P-form reference
- Its estimates agree with `Marginal.ai_step` within the jitter tolerance
- The AI matrix is symmetric and positive semidefinite
- The log-likelihood is finite

---

## 3. How to Test

### 3.1 Unit Testing Strategy

Two fixed models carry most of the coverage:

- A 4-observation model with `Z = [[1,0],[1,0],[0,1],[0,1]]`,
  `X = [[1,0],[1,1],[1,2],[1,3]]`, `G = ScalarMatrix(2)`,
  `R = ScalarMatrix(4)` and `free = [0.1, -0.2]`. Every entry of the score
  and AI matrix can be checked against an explicit `P` matrix built with
  `torch.linalg.inv`, and against `torch.autograd`.
- A parameterised-design variant using `SimpleMatrix(2, callable)` whose
  callable returns a 4x2 design with the last row set from the free
  parameters, exercising the non-empty `dz` path and `[Z, G, R]` ordering.

`KnownMatrix` substitutes for `G` or `R` to exercise the empty derivative
blocks. `john_alpha` is the realistic end-to-end case. All reference
calculations are explicit tensor formulas in the test module, not calls back
into the library, and float64 is used for value comparisons so that `atol`
can be set at `1e-12` rather than a loose default.

### 3.2 Test Structure

```python
def test_score_matches_autograd_of_loglik(self):
    # Arrange
    model, y, x, _ = _tiny_model()
    free = torch.tensor(TINY_FREE, dtype=DTYPE, requires_grad=True)

    # Act
    loglik = CForm().loglik(model, y, x, free)
    (autograd_score,) = torch.autograd.grad(loglik, free)

    # Assert
    score, _, _ = CForm().ai_step(model, y, x, free.detach())
    torch.testing.assert_close(score, autograd_score, rtol=0.0, atol=1e-12)
```

Tests are grouped into classes by contract area:
`TestCFormConstructionAndEligibility`, `TestCFormLoglik`,
`TestCFormScoreAndInformation`, `TestCFormInverseBlocks`,
`TestCFormPreprocessInput`, `TestCFormDtypeAndDevice`, `TestCFormJohnAlpha`.

### 3.3 Test Execution

```bash
conda run -n torch-openreml python -m pytest tests/methods/test_cform.py
```

---

## 4. Key Risks / Notes

- **The residual jitter is not uniform across derivative blocks.** When `R`
  has no closed-form inverse, `preprocess_input` adds `1e-6 I` to `R` but
  leaves `G` to the Cholesky path. The jitter test therefore builds its
  expected `C` from exactly `r + 1e-6 I` and asserts that `g` is unchanged;
  a future change to jitter both, or neither, surfaces here.
- **A supplied `r_inv` does not skip the `R` factorisation.** `loglik` and the
  AI path still call `torch.linalg.cholesky(r)` to obtain `logdet R`, so a
  non-positive-definite `R` fails even when `r_inv` is passed. This is pinned
  by `test_rejects_non_positive_definite_residual`.
- **Fixed-tensor designs follow the free parameters.** `x`, `y` and a plain
  tensor `Z` are all cast to `free_params`' dtype and device in
  `preprocess_input`; a `Matrix` design already followed through its
  parameter tensor. The test covers the mixed-dtype fixed-`Z` case as a
  regression.
- **Eligibility checks only `V.num_free_params`.** A model whose only free
  parameter lives in a fixed-effect design still has `V.num_free_params > 0`
  and is eligible; a model with a known `G` and known `R` but a parameterised
  `Z` is covered explicitly.
- **`Augment` is a `Matrix`, so multi-term designs keep `kz = 0`.** In the
  two-term `john_alpha` model the design is an `Augment` operator whose own
  `num_free_params` is zero, so `dz` is empty and only the `G`/`R` blocks
  contribute. CForm accepts it because `Operator` derives from `Matrix`;
  this plan only relies on that behaviour through the realistic case.
- **The AI matrix is not explicitly symmetrised.** It is computed as
  `0.5 Q P Q'`, mathematically symmetric but only up to roundoff. The tests
  assert symmetry at `atol=1e-14` and PSD at `-1e-12`, which is what the
  implementation can actually deliver.
- **`require_loglik=False` returns a NaN scalar, not `None`.** The score and
  AI matrix are asserted bitwise equal to the full call so that a future
  optimisation cannot silently perturb them.
- **Device tests must compare `device.type`.** On MPS the tensors carry
  `mps:0` while `torch.device("mps")` compares unequal; the tests compare
  `.device.type` to avoid a platform-specific failure.
- **The marginal comparison is tolerance-based by construction.** CForm and
  Marginal agree exactly in exact arithmetic but differ by the `1e-6`
  jitter on a composite `V`; the cross-method tests use `atol=1e-5` rather
  than pretending the two paths are bitwise identical.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction, statelessness and eligibility | 12 |
| Restricted log-likelihood | 8 |
| Score vector and AI matrix | 14 |
| Inverse coefficient blocks | 7 |
| `preprocess_input` | 9 |
| dtype and device | 6 |
| John's alpha realistic case | 5 |
| **Total** | **61** |

All **61 collected tests pass**; there are no xfails.
