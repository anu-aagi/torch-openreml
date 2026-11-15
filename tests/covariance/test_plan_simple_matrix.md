# Test Plan: `torch_openreml.covariance.simple_matrix`

## 1. Overview

`torch_openreml/covariance/simple_matrix.py` defines `SimpleMatrix`, a
`Matrix` whose `__call__` is a user-supplied callable. It exposes exactly `n`
free parameters (`theta_0 … theta_{n-1}`) with identity transforms, an
optional `default` tensor of length `n`, and an optional `manual_grad`
callable. It is the escape hatch for covariance structures that are easier to
write as a function than as a dedicated class; `param.simple_param_specs` is
used by its constructor.

Covered by `tests/covariance/test_simple_matrix.py`.

## 2. Behaviour inventory

### Construction

- `SimpleMatrix(n, call=...)` is a `Matrix`; `call` is required and a missing
  or non-callable value raises `ValueError` ("'call' must be provided").
- `default` must be a 1-D `torch.Tensor`; a float raises `TypeError` ("1D
  torch tensor") and a wrong-length tensor raises `ValueError` naming the
  expected shape.
- With no `default`, all parameters default to zero; `n=0` is allowed and
  produces an empty parameter surface.

### `__call__`

- Calls the supplied function with the resolved parameter tensor: tensor
  input, dict input keyed by `theta_i`, and defaults all work, and the result
  is exactly the function's output.
- Rejects a wrong-length tensor (`ValueError`, "length n") and a dict with a
  missing key (`ValueError`, "Missing free parameters").
- dtype follows the input tensor, then the defaults, then Torch's default;
  input always wins over a differently typed default.

### Gradients

- With no `manual_grad`, `auto_grad` differentiates the callable; the
  Jacobian and names `theta_0 … theta_{n-1}` are correct for tensor, default,
  and dict inputs, and follow the input dtype.
- A supplied `manual_grad` is used by `grad()` and receives the resolved
  parameter tensor.
- `grad_mode="manual"` without a `manual_grad` raises `NotImplementedError`;
  the default mode falls back to automatic differentiation.
- With `n=0`, `grad()` returns `(None, [])`.

## 3. How to test

- Arrange → Act → Assert, grouped by concern: construction/validation,
  `__call__`, dtype, gradients.
- Use `diagonal_call(free_params) -> torch.diag(free_params)` as the simplest
  concrete callable, so expected values and Jacobians are obvious
  (`diag(e_i)` per parameter).
- Pass a dict with all `theta_i` keys to exercise the dict path; omit a key to
  exercise the error.
- Compare dtypes explicitly rather than relying on `allclose`.
- For the manual path, assert that the callable receives a tensor (including
  when defaults are used) and that its Jacobian is returned unchanged.

## 4. Key Risks

- **Callable resolution**: the function must receive the fully resolved
  parameter tensor, not the raw input; defaults and dict inputs must be
  normalised first.
- **dtype/device precedence**: input over defaults over Torch defaults. A
  default-typed tensor silently overriding an explicit float64 input would
  produce a wrong-precision result without an error.
- **Empty parameter surface**: `n=0` must still build a matrix and return
  `(None, [])` from `grad`, matching the other fixed matrices.
- **Manual gradient fallback**: a missing `manual_grad` must raise only in
  explicit manual mode, and the default mode must fall back to autodiff.
- **Validation messages**: the `default` tensor is validated at construction,
  while parameter lengths are validated per call, so both paths need pinning.

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction and default validation | 4 |
| `__call__` values and argument validation | 5 |
| dtype precedence | 3 |
| Gradients (automatic, manual, empty) | 8 |
| Zero-parameter defaults | 1 |
| **Total** | **21** |
