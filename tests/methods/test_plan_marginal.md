# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.methods.marginal`
**Purpose of Module:**
Implements the marginal evaluation of the REML score, average-information
matrix and restricted log-likelihood directly from the marginal covariance
`V`. It offers an exact trace mode, a Hutch++ stochastic trace estimate, and
an `"auto"` mode that starts with the estimate for large problems and
switches to exact traces once the previous score norm is small. The AI matrix
and log-likelihood stay exact in every mode. Two tensor kernels are used: one
that forms `V` through Cholesky solves and never materialises `V^{-1}`, and
one that reuses a supplied `V^{-1}` and forms `P` explicitly.

**Classes Covered:**

- `Marginal` — configurable evaluator with model-aware and tensor-level entry
  points

**Testing Goal:**
Pin the constructor options and their validation, the `"auto"` trace
resolution and its score-norm history, the exact score/AI/log-likelihood on
models small enough to check against explicit matrix formulas and
`torch.autograd`, the stochastic trace behaviour under fixed seeds, the
supplied-inverse branch, the `preprocess_input` jitter contract, dtype/device
propagation, and agreement with the C-form method on a realistic `john_alpha`
model. The implementation and its docstrings are now aligned; the tests
encode the documented contract and there are no xfails.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Construction, Defaults and Validation

- Defaults: `trace_approx is False`, `subspace_fraction == 0.01`,
  `exact_score_threshold == 10.0`, `_last_score_norm is None`
- `MIN_N_FOR_TRACE == 100`
- Custom options are stored unchanged and `_last_score_norm` starts as `None`
- `trace_approx` rejects `None`, `"exact"`, `"approx"`, `2` and `-1` with
  `ValueError("`trace_approx` must be False, True or 'auto'.")`
- `subspace_fraction` outside `[0, 1]` (`-0.1`, `1.1`, `2.0`) raises
  `ValueError("`subspace_fraction` must lie in [0, 1].")`
- The boundary values `0.0` and `1.0` are accepted
- A negative `exact_score_threshold` (`-1.0`, `-1e-9`) raises
  `ValueError("`exact_score_threshold` must be non-negative.")`
- `reset()` clears `_last_score_norm`
- `trace_approx=0` and `trace_approx=1` are rejected at construction with
  `ValueError("`trace_approx` must be False, True or 'auto'.")`; the check is
  type-based, not equality-based, so numeric flags never reach `ai_step`
- A rejected numeric flag is never stored on the instance

#### B. Restricted Log-Likelihood

- `loglik(None, ...)` raises `ValueError("`model` must not be None.")`
- `loglik` matches the explicit reference
  `-0.5 (logdet V + logdet(X'V^{-1}X) + y'Py)` on the covariance the marginal
  path actually uses (a composite `V` plus the `1e-6 I` jitter)
- A model whose `V` is a plain `ScalarMatrix` (no random effects) uses the
  exact inverse with no jitter and matches the reference
- The tensor kernel matches the reference with and without a supplied `v_inv`
- `loglik` agrees with `CForm.loglik` within `atol=1e-5`

#### C. Score, AI Matrix and Trace Modes

- The score has shape `(k,)`
- The score and AI matrix match the explicit reference on the jittered
  covariance, with `dv = model.V.grad(free_params)[0]`
- The score equals `torch.autograd.grad` of `Marginal.loglik` after
  `model.V.reset_intermediates()` (without the reset the cached `V` is the
  detached tensor built by a previous `ai_step`)
- A no-random-effect model takes the with-inverse branch and matches the
  reference on the exact `V`
- The AI matrix is symmetric and positive semidefinite
- `require_loglik=False` returns `torch.nan` and bitwise-identical score and
  AI values
- The no-inverse and with-inverse exact kernels agree to roundoff on the same
  `V`
- `subspace_fraction=1.0` with `trace_approx=True` reproduces the exact score
  and gives bitwise-identical AI and log-likelihood
- `subspace_fraction=0.5` changes only the score; AI and log-likelihood stay
  bitwise equal to the exact values
- The stochastic estimate is reproducible for a fixed seed and differs across
  seeds
- The approximate path also runs with a supplied `v_inv`
- An empty `dv` raises
  `ValueError("dv must have at least one parameter; got dv.shape[0] == 0.")`
  through both kernels
- `subspace_fraction` outside `[0, 1]` raises
  `ValueError("subspace_fraction must lie in [0, 1].")` through both kernels
- A derivative block with the wrong trailing shape raises `RuntimeError`
- The model-aware `ai_step` is bitwise identical to the tensor kernel on
  `V + 1e-6 I` and `model.V.grad`

#### D. `"auto"` Trace Resolution

- The first call on a large problem uses the approximation (the score differs
  from the exact kernel)
- The call records `_last_score_norm == torch.norm(score)`
- A second call switches to exact once the recorded norm is below
  `exact_score_threshold`
- With `exact_score_threshold=0.0` the second call keeps approximating
- Below `MIN_N_FOR_TRACE` the first call is already exact
- `reset()` restarts the history so the next call approximates again
- The tests override `MIN_N_FOR_TRACE` on the instance (to `4`) so the
  `"auto"` logic is exercised without a 100-observation model

#### E. `preprocess_input` Contract

- With `grad=True` the dict has exactly `x, y, v, v_inv, dv`
- With `grad=False` it has exactly `x, y, v, v_inv`
- A composite `V` yields `v_inv is None` and `v == model.V(free) + 1e-6 I`
- A `ScalarMatrix` `V` yields a populated `v_inv` and `v == model.V(free)`
  with no jitter
- `dv` has shape `(k, n, n)` — `(2, 4, 4)` for the two-parameter tiny model
- `x` and `y` are cast to `free_params`' dtype
- `ai_step(None, ...)` raises the documented `AttributeError` from
  `preprocess_input` (it dereferences `model` rather than converting the
  error); `loglik(None, ...)` still raises its explicit `ValueError`

#### F. dtype and Device Propagation

- Model-aware estimates follow the free-parameter dtype for float32 and
  float64
- The tensor kernel follows its input dtype for float32 and float64
- The model-aware entry point follows the free-parameter device; the test
  picks `cuda` → `mps` → `pytest.skip("no accelerator available")` inline and
  compares `device.type` because MPS reports `mps:0`

#### G. Realistic John's Alpha Case

- The single-genotype `john_alpha` model (`n = 72`) reproduces the explicit
  reference log-likelihood
- Its score and AI matrix match the explicit reference
- Its estimates agree with `CForm.ai_step` within the jitter tolerance
- The AI matrix is symmetric and positive semidefinite
- The log-likelihood is finite

---

## 3. How to Test

### 3.1 Unit Testing Strategy

The same two tiny models as the C-form plan drive the value tests:

- A 4-observation model with `Z = [[1,0],[1,0],[0,1],[0,1]]`,
  `X = [[1,0],[1,1],[1,2],[1,3]]`, `G = ScalarMatrix(2)`,
  `R = ScalarMatrix(4)` and `free = [0.1, -0.2]`. Its `V` is a
  `MarginalCovariance`, which has no closed-form inverse, so the marginal
  path exercises the jitter plus the no-inverse kernel. Expected values are
  computed explicitly from `V + 1e-6 I`.
- A no-random-effect model (`V = ScalarMatrix(4)`) exercises the
  with-inverse kernel and the no-jitter path.

`DummyMatrix` designs supply the dtype-following model. `john_alpha` is the
realistic end-to-end case. The stochastic trace tests fix `torch.manual_seed`
and compare scores only, since AI and log-likelihood are exact in every mode.
All reference calculations are explicit tensor formulas in the test module,
not calls back into the library, and float64 is used for value comparisons.

### 3.2 Test Structure

```python
def test_approximate_trace_changes_score_but_not_ai_or_loglik(self):
    # Arrange
    model, y, x, free = _tiny_model()
    v = model.V(free)
    dv = model.V.grad(free)[0]
    expected = Marginal().ai_step_from_tensors(x, y, v, dv)

    # Act
    torch.manual_seed(0)
    actual = Marginal().ai_step_from_tensors(
        x, y, v, dv, trace_approx=True, subspace_fraction=0.5
    )

    # Assert
    assert not torch.allclose(actual[0], expected[0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
```

Tests are grouped into classes by contract area:
`TestMarginalConstruction`, `TestMarginalLoglik`,
`TestMarginalScoreAndInformation`, `TestMarginalAutoTrace`,
`TestMarginalPreprocessInput`, `TestMarginalDtypeAndDevice`,
`TestMarginalJohnAlpha`.

### 3.3 Test Execution

```bash
conda run -n torch-openreml python -m pytest tests/methods/test_marginal.py
```

---

## 4. Key Risks / Notes

- **`trace_approx` is type-checked.** The constructor accepts only `bool`
  values and the exact string `"auto"`; `0` and `1` are rejected even though
  they compare equal to `False`/`True`. This keeps the identity dispatch in
  `ai_step` total and prevents the former `UnboundLocalError`.
- **`ai_step(None, ...)` raises `AttributeError` by design.** Only `loglik`
  guards against `model=None` and raises `ValueError`; `ai_step` calls
  `preprocess_input` directly, which fails with `AttributeError` when it
  dereferences `model`. The docstring documents that behaviour.
- **The composite covariance always takes the jitter path.** `MarginalCovariance`
  does not override `manual_inverse`, so every model with a random-effect term
  gets `v + 1e-6 I` and the no-inverse kernel. The tests compute their
  reference on the jittered matrix; the cross-method comparison with CForm is
  therefore tolerance-based (`atol=1e-5`), not exact.
- **The covariance cache can silently detach autograd.** `Marginal.ai_step`
  is decorated `@torch.no_grad()` and populates `MarginalCovariance`'s
  intermediate cache with a detached `V`. A subsequent `loglik` with
  `requires_grad=True` at the same parameter values reuses that tensor, so
  `torch.autograd.grad` raises `RuntimeError: element 0 of tensors does not
  require grad`. The autograd test calls `model.V.reset_intermediates()`
  first and uses the same leaf tensor for both calls.
- **Only the trace is approximated.** The AI matrix and log-likelihood are
  exact in every mode, which is why the approximation tests assert bitwise
  equality for them. With `subspace_fraction=1.0` the Hutch++ subspace spans
  the whole space and the score matches the exact trace to roundoff.
- **`subspace_fraction` controls probe count through `randint`, so seeds are
  mandatory.** The reproducibility and seed-sensitivity tests fix
  `torch.manual_seed`; without it the score would be flaky.
- **Seeds are compared on this specific tiny model.** The seed-sensitivity
  test asserts that seeds `0` and `1` give different scores at
  `subspace_fraction=0.5`; this is stable for the fixed 4-observation model
  used here, not a claim about Hutch++ in general.
- **`"auto"` is exercised by overriding `MIN_N_FOR_TRACE` on the instance.**
  Building a 100-observation model for every dispatch test would be wasteful;
  the instance attribute is looked up on `self`, so the override is
  legitimate and keeps the fixtures small and deterministic.
- **The with-inverse branch needs a supplied `v_inv`.** Model-aware calls only
  use it when `V.manual_inverse` succeeds, which in this suite means the
  no-random-effect `ScalarMatrix` model; the tensor-kernel tests supply
  `torch.linalg.inv(v)` directly to cover the branch on any `V`.
- **Device tests must compare `device.type`.** MPS tensors carry `mps:0`,
  which compares unequal to `torch.device("mps")`.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction, defaults, validation and reset | 18 |
| Restricted log-likelihood | 6 |
| Score, AI matrix and trace modes | 22 |
| `"auto"` trace resolution | 6 |
| `preprocess_input` | 7 |
| dtype and device | 5 |
| John's alpha realistic case | 5 |
| **Total** | **69** |

All **69 collected tests pass**; there are no xfails.
