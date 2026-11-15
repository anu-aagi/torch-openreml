# Test Plan: `torch_openreml.covariance.operator_marginal_covariance`

## 1. Overview

`torch_openreml/covariance/operator_marginal_covariance.py` defines
`MarginalCovariance`, an `Operator` with exactly three operands `Z` (an
`n × q` incidence matrix), `G` (a `q × q` random-effect covariance), and
`R` (an `n × n` residual covariance). It evaluates `V = R + Z G Zᵀ`, caches
the operands and the result keyed by parameter values, and propagates the
per-operand Jacobians analytically. It is the covariance used by
`MixedModelREML`.

Covered by `tests/covariance/test_operator_marginal_covariance.py`.

## 2. Behaviour inventory

### Construction

- Three positional operands (`Z, G, R`), three keyword operands, or one dict
  with exactly those keys.
- Positional and keyword operands cannot be mixed (`ValueError`).
- Zero, one, two, or four positional operands raise `ValueError`; missing or
  extra named operands raise `ValueError` naming `Z, G and R`.
- Operands must be `Matrix` or tensor; at least one must be a `Matrix`
  (`TypeError` otherwise), including the all-fixed-tensor case.
- Parameter names are namespaced `Z/...`, `G/...`, `R/...`, in that order,
  with `Z` first, then `G`, then `R`. `free_param_names` matches
  `param_names`; `build_params` has one entry per free parameter.
- `repr` identifies the class and its operands.

### `__call__(free_params=None)`

- Matches `R + Z G Zᵀ` against explicit products for a fixed tensor `Z`, a
  one-hot `DummyMatrix` `Z`, a `Z = I` case, and a rank-one `Z`.
- Uses the operand values rebuilt from the free-parameter tensor, including
  dict input; default parameters equal explicit zeros.
- Result is symmetric and positive definite for a valid model; shape
  `(n, n)` is recorded on `shape` after the call.
- Rejects non-2-D `Z`, `G`, or `R`, non-square `R`, non-square `G`,
  `Z.shape[0] != R.shape[0]`, and `Z.shape[1] != G` dimensions, with the
  documented messages. Construction does not evaluate the operands, so the
  error surfaces on `__call__`.

### Caching

- Identical parameter values return the cached result object (`is`), and
  `manual_grad` reuses that cached value.
- New values or a dtype change invalidate the cache; `disable_cache` forces
  recomputation; `enable_cache` restores reuse.

### Gradients

- `manual_grad` matches `auto_grad` for the three-operand model and for a
  parameterised `Z` (`LowerTriangularMatrix`), with matching names.
- Works when only `G`, only `R`, or only `Z` is free; shapes are
  `(num_free_params, n, n)`.
- All-fixed operands return `(None, [])` from `manual_grad` and `grad`.
- `grad()` defaults to manual and honours `grad_mode`.
- Wrong-length tensors, unknown dict keys, and non-tensor arguments raise.

### dtype and device

- The result follows the input parameter tensor's dtype; without input it
  uses the free-parameter defaults, falling back to Torch's default dtype.
- Fixed tensor operands are cast to the resolved dtype.
- Result and gradient follow the input device (`cuda` → `mps` → skip).

## 3. How to test

- Arrange → Act → Assert, grouped by concern: construction, values,
  validation, cache, gradients, dtype/device.
- Compare against explicit `r + z @ g @ z.T` built from each operand's own
  `__call__`, so the test does not duplicate the implementation's parameter
  transforms.
- Assert cache reuse with `is` on successive identical calls, and
  invalidation with `is not` plus a value change.
- Cross-check every analytic Jacobian against `auto_grad`; for the
  parameterised-`Z` case allow a looser `atol` because the Jacobian composes
  three operand Jacobians.
- Device tests use the standard `cuda` → `mps` → `pytest.skip("no accelerator available")`
  ladder.

## 4. Key Risks

- **Shape validation**: the three shapes are interdependent (`Z` rows with
  `R`; `Z` columns with both `G` dimensions). Each mismatch must raise before
  the matrix product, or PyTorch would raise a less useful error.
- **Operator order**: names and Jacobians must follow `Z, G, R`. Swapping
  `G` and `R` would still produce a plausible matrix for some designs.
- **Cache invalidation**: the cache is a single entry keyed by values, dtype,
  and device. A stale hit after a parameter or dtype change would silently
  return the wrong covariance.
- **Z gradient**: `Z` contributes through both `Ż G Zᵀ` and `Z G Żᵀ`; a
  missing transpose or a dropped term is invisible in the value check, so it
  is checked against `auto_grad`.
- **Fixed operands**: `KnownMatrix` has no parameters, so the operator must
  return `(None, [])` rather than an empty tensor, and must still cast fixed
  tensors to the resolved dtype.

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction and naming | 16 |
| `__call__` values | 11 |
| Shape validation | 8 |
| Caching | 6 |
| Gradients | 15 |
| dtype and device | 7 |
| **Total** | **63** |
