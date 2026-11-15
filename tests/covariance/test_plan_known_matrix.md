# Test Plan: `torch_openreml.covariance.known_matrix`

## 1. Overview

`torch_openreml/covariance/known_matrix.py` defines `KnownMatrix`, a
`Matrix` subclass wrapping a user-supplied tensor as a fixed covariance
matrix. It has no parameters, returns a cast copy of the tensor from
`__call__`, and caches `torch.linalg.inv` at construction for square,
non-singular inputs so that `manual_inverse` is cheap.

It is the fixed-covariance escape hatch used by `MixedModelREML` (via
`KnownMatrix` inside the marginal covariance) and by examples that hold the
residual or random-effect covariance fixed.

Covered by `tests/covariance/test_known_matrix.py`.

## 2. Behaviour inventory

### Construction

- Accepts any 2-D `torch.Tensor`; keeps a reference (`matrix` property).
- `shape` is the tensor's shape as a tuple.
- Empty parameter surface: `param_specs == {}`, all name/index lists empty,
  `num_params == 0`.
- Rejects non-tensors with `TypeError`, non-2-D tensors with `ValueError`,
  and a missing argument with `TypeError`.

### `__call__(free_params=None)`

- Returns `self._matrix.to(device, dtype).clone()`: equal values, a distinct
  tensor, rectangular shapes supported, source untouched by mutation.
- Accepts `None`, an empty 1-D tensor, `{}`, and the `free_params` keyword.
- Rejects non-empty tensors (`ValueError`, "length 0"), non-empty dicts
  (`ValueError`, "Unexpected free parameters"), non-tensors (`TypeError`),
  non-1-D tensors (`ValueError`), and unknown keywords (`TypeError`).
- dtype follows the input tensor (including integer dtypes); device follows
  the input when present, otherwise the Torch default. Defaults are resolved
  at call time, not construction time.

### `build_params`

- Returns an empty tensor / empty dict; `include_fixed` and `trans` do not
  change that; `out_format` is validated.

### Gradients

- `grad`/`auto_grad` return `(None, [])`; `manual_grad` raises
  `NotImplementedError`; `grad_mode="manual"` raises; an unknown mode raises
  `RuntimeError`; `trans_grad` is empty.

### Intermediates

- `get_intermediates` is always `None`; `set_intermediates` is a no-op;
  both validate their parameter argument.

### `manual_inverse(free_params=None)`

- Matches `torch.linalg.inv` for square non-singular matrices (including
  1×1 and the identity), satisfies `V @ V⁻¹ = I`, and keeps symmetry.
- Raises `NotImplementedError` for rectangular, singular and all-zero
  matrices.
- Casts to the input dtype/device and returns a fresh copy per call.

## 3. How to test

- Arrange → Act → Assert, grouped by concern: construction, call values,
  dtype/device, validation, `build_params`, gradients, cache, inverse, repr.
- Compare values with `torch.equal` for exact casts and `torch.allclose` for
  inverses; assert copy semantics with `is not` plus mutation of the result.
- Device tests use the standard `cuda` → `mps` → `pytest.skip("no accelerator available")`
  ladder.
- Dtype tests pass an empty tensor of the target dtype; default-dtype tests
  temporarily change `torch.set_default_dtype` and restore it in `finally`.

## 4. Key risks

- **Aliasing**: `__call__` must clone, or a caller mutating the returned
  covariance would corrupt the known matrix (and the cached inverse).
- **Stale cache**: the inverse is computed once at construction. This is
  intentional, but it means the tensor must be treated as immutable; the tests
  pin the precomputation and the supported shapes.
- **Inverse availability**: rectangular and singular inputs must raise the
  documented `NotImplementedError` rather than a raw `RuntimeError` from
  `torch.linalg.inv`.
- **dtype/device resolution**: a known matrix has no parameters, so the empty
  parameter tensor is the only dtype/device channel; it must be followed, not
  ignored, and defaults must be resolved lazily.
- **Validation messages**: `build_params` is shared with every `Matrix`, so
  the empty-surface error paths must stay consistent with `IdentityMatrix`.

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction and parameter surface | 16 |
| `__call__` values and copy semantics | 11 |
| dtype and device | 13 |
| Free-parameter validation | 14 |
| `build_params` | 9 |
| Gradients | 14 |
| Intermediates | 7 |
| `manual_inverse` | 14 |
| Repr | 2 |
| **Total** | **100** |
