# Test Plan: `torch_openreml.covariance.param`

## 1. Overview

`torch_openreml/covariance/param.py` provides one helper, `simple_param_specs`.
It builds the parameter-specification dictionary consumed by
`Matrix.__init__` / `Operator` operands, giving `n` free parameters named
`theta_0 … theta_{n-1}` with a shared default value and a shared transform
(identity by default).

The helper is small but it is the default parameter surface for
`SimpleMatrix` and for any user constructing a `Matrix` subclass by hand.
Its contract is therefore mostly about the shape, aliasing and validation of
the produced specs rather than about numerical output.

Covered by `tests/covariance/test_param.py`.

## 2. Behaviour inventory

### `simple_param_specs(n, default=None, trans=None)`

- Returns a dict with exactly `n` entries, keys `theta_0 … theta_{n-1}` in
  index order.
- Every entry is `{"fixed": False, "default": <1-D tensor of shape (1,)>, "trans": <Transform>}`.
- `default=None` means `torch.tensor([0.0])`; `trans=None` means
  `TransformIdentity()`.
- The same transform object is shared by every parameter.
- Each `default` is a detached clone: specs do not alias the caller's tensor,
  and the entries do not alias each other.
- `n=0` and negative `n` return `{}`.

### Validation

- Non-tensor `default` raises `TypeError`.
- A `default` that is not exactly shape `(1,)` raises `ValueError`.
- A non-integer `n` raises `TypeError` from `range`.

## 3. How to test

- Arrange → Act → Assert, grouped by concern: names/order, spec contents,
  validation, integration.
- Compare tensors with `torch.equal` where exact and `torch.allclose` where the
  value passes through a transform.
- Assert aliasing with `is` / `is not`, then mutate a source tensor or a
  sibling default to prove independence.
- Integration cases instantiate `SimpleMatrix` (the production consumer of
  this helper) and check names, defaults, tensor/dict input and gradient names.

## 4. Key risks

- **Shared mutable defaults**: returning the caller's tensor in every entry
  would let one parameter's default change another's. The clone-per-entry
  behaviour is load-bearing.
- **Transform identity**: `trans=None` must mean identity, not "no transform"
  (which would break `Matrix.trans_grad`).
- **Shape validation**: a scalar default or a `(1, 1)` default silently
  broadcasts in some code paths; the helper must reject both early.
- **Ordering**: `Matrix` aligns names to free-parameter positions by dict
  order, so the insertion order must be `theta_0, theta_1, …`.
- **Zero parameters**: `SimpleMatrix(0, …)` is legal; the helper must return an
  empty dict rather than raise.

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Names and ordering | 5 |
| Specification contents and aliasing | 13 |
| Validation | 9 |
| Integration with `SimpleMatrix` | 5 |
| **Total** | **40** |
