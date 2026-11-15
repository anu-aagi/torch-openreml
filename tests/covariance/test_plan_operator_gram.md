# Test Plan: `torch_openreml.covariance.operator_gram`

## 1. Overview

`torch_openreml/covariance/operator_gram.py` defines `Gram`, an `Operator`
with exactly one operand `X` that evaluates either `XᵀX` (`gram_type="xtx"`,
the default) or `XXᵀ` (`gram_type="xxt"`). The operand may be any `Matrix`;
its free parameters are exposed namespaced under the operand name and its
Jacobian is propagated through the product rule.

Covered by `tests/covariance/test_operator_gram.py`.

## 2. Behaviour inventory

### Construction

- `gram_type` is positional/keyword, defaults to `"xtx"`, and accepts `"xxt"`.
- Any other value (`"XTX"`, `"xTx"`, whitespace, empty, `None`, non-string)
  raises `ValueError` mentioning `gram_type`.
- Exactly one operand: a positional operand is named `op_0`; a keyword or
  dict operand keeps its name. Two operands raise `ValueError("One operand...")`.
- Zero operands raise `TypeError` ("at least one Matrix") from the `Operator`
  base; a bare fixed tensor does too, because the base contract requires one
  `Matrix` operand.
- A fixed tensor can be wrapped in `KnownMatrix` and then works.
- Parameter names are the operand's names prefixed with `op_0/` (or the chosen
  name); `repr` includes the class, `gram_type`, and the operand.

### `__call__(free_params=None)`

- `"xtx"` equals `X.T @ X`; `"xxt"` equals `X @ X.T`.
- `X` of shape `(n, m)` gives `(m, m)` for `"xtx"` and `(n, n)` for `"xxt"`;
  the same shapes are recorded on `shape` after the call (before the first
  call `shape == ()`).
- Works for square, rectangular, single-column, and single-row operands;
  square inputs show that the two orientations generally differ.
- Result is symmetric and positive semidefinite; a zero operand gives a zero
  Gram; repeated calls agree.
- Values are rebuilt from the free-parameter tensor, including dict input.

### Gradients

- `manual_grad` matches `auto_grad` for both `gram_type`s and for a partially
  fixed parameter set (fixed names are dropped).
- Jacobian shape is `(num_free_params, m, m)` for `"xtx"` and
  `(num_free_params, n, n)` for `"xxt"`; names match `param_names`.
- `grad()` defaults to manual and honours `grad_mode`.
- A fixed operand (`KnownMatrix`) returns `(None, [])` from both `manual_grad`
  and `grad`.
- Wrong-length tensors, unknown dict keys, and non-tensor arguments raise.

### dtype and device

- The result follows the input parameter tensor's dtype; with no input it uses
  the free-parameter defaults, falling back to Torch's default dtype
  (`KnownMatrix` alone therefore yields `float32`).
- The operand and result follow the input device (`cuda` → `mps` → skip).

## 3. How to test

- Arrange → Act → Assert, grouped by concern: construction, values, values
  from parameters, gradients, dtype/device.
- Use `KnownMatrix(_x())` (float64 input) for exact product comparisons; when
  calling it, pass an empty `float64` tensor so the resolved dtype is
  `float64` rather than Torch's default.
- Cross-check every analytic Jacobian against `auto_grad`; assert shapes
  against the operand dimensions.
- Device tests use the standard `cuda` → `mps` → `pytest.skip("no accelerator available")`
  ladder.

## 4. Key risks

- **Contract inconsistency**: the `Gram` docstring says `X` "may be a fixed
  `torch.Tensor`", but the `Operator` base requires at least one `Matrix`
  operand, so a bare tensor is rejected. The tests pin the current
  `TypeError`; wrapping in `KnownMatrix` is the supported route.
- **Orientation**: `"xtx"` and `"xxt"` produce different shapes for
  rectangular `X`; mixing them silently would give a valid but wrong
  covariance, so both are checked explicitly.
- **dtype/device resolution**: with no free parameters, a fixed operand does
  not carry dtype/device, and the operator falls back to Torch defaults unless
  the caller supplies a parameter tensor.
- **Manual gradient**: the product rule must handle both orientation branches
  and partially fixed operands; a sign/transpose error would still look
  plausible, so it is checked against `auto_grad`.
- **Caching**: `Gram` does not use intermediates, so repeated evaluation must
  be value-correct rather than object-identical.

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction and operand naming | 19 |
| `__call__` values | 13 |
| Values from free parameters | 4 |
| Gradients | 13 |
| dtype and device | 7 |
| **Total** | **56** |
