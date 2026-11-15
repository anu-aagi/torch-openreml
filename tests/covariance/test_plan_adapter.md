# Test Plan: `torch_openreml.covariance.adapter`

## 1. Overview

`torch_openreml/covariance/adapter.py` defines `Adapter`, a `Matrix` that
reparameterises an existing `Matrix` (the *adaptee*). A user-supplied
`param_map` maps the adapter's own untransformed free parameters to the
adaptee's parameter vector, so a different parameterisation (for example two
variances summing to one) can be fitted without changing the adaptee. The
adapter's own transforms must be `TransformIdentity`; `manual_inverse` is
forwarded to the adaptee, and cache toggles propagate down.

Covered by `tests/covariance/test_adapter.py`.

## 2. Behaviour inventory

### Construction and parameter surface

- `Adapter(adaptee, param_specs, param_map)` is a `Matrix`; `shape` is the
  adaptee's shape; `num_free_params` counts only free adapter parameters.
- `param_specs` returns the stored dict (mutations take effect): flipping a
  spec to `fixed=True` drops it from `free_param_names` and `param_map`
  receives only free parameters.
- Non-identity transforms on the adapter raise `ValueError` mentioning
  `TransformIdentity`.
- `repr` contains `Adapter`.

### `__call__`

- Applies `param_map` to the raw (untransformed) free parameters and passes
  the resulting vector to the adaptee: the value equals building the adaptee
  directly from the mapped vector.
- One adapter parameter can drive several adaptee parameters, and changing it
  changes the matrix.
- Accepts defaults (`None`) and a dict keyed by adapter parameter names, both
  matching the equivalent tensor.

### `manual_inverse`

- Matches `torch.linalg.inv(V)` for the built matrix.
- Forwards the *mapped* parameters, not the raw ones, and does not call
  `torch.linalg.inv` when the adaptee has a closed form (asserted by
  monkeypatching `torch.linalg.inv` to fail).
- Accepts defaults and dict input.
- Propagates `NotImplementedError` when the adaptee has no closed-form
  inverse.

### Gradients

- `manual_grad` shape is `(num_free_params, *shape)` and its names match
  `free_param_names`; fixed adapter parameters are excluded.
- Manual and automatic Jacobians agree, with and without a fixed parameter.
- `grad()` uses the manual path.
- An all-fixed adapter returns `(None, [])`.

### Cache control

- `disable_cache` and `enable_cache` propagate to the adaptee, including a
  nested adapter; disabling clears the adaptee's intermediates and enabling
  restores caching.

## 3. How to test

- Arrange → Act → Assert, grouped by concern: construction, call, inverse,
  parameter mapping, gradients, cache, repr.
- Build a concrete adapter over `DiagonalMatrix(2)` whose `param_map` is
  `sigmoid(θ), 1 - sigmoid(θ)`; compare against `DiagonalMatrix(2)` called
  with the mapped vector.
- Use `AR1Matrix(3)` as a caching adaptee when asserting cache propagation.
- Compare manual and automatic Jacobians with `torch.allclose`.
- Monkeypatch `torch.linalg.inv` to prove the forward path uses the adaptee's
  closed form.

## 4. Key Risks

- **Transform bypass**: `param_map` must see raw free parameters. If the
  adapter applied its own transforms first, the mapping would be silently
  wrong; the tests record the arguments the map receives.
- **Free-only mapping**: fixed parameters must not be passed to `param_map`,
  or the map's expected length would change when a parameter is fixed.
- **Inverse forwarding**: the closed-form inverse must be the adaptee's,
  evaluated at the mapped parameters. Falling back to a dense inverse would
  be correct numerically but loses the closed form, so it is asserted via a
  monkeypatched `torch.linalg.inv`.
- **Cache propagation**: a stale adaptee cache after `disable_cache` would
  serve old values; both propagation and clearing are checked, including one
  level of nesting.
- **Identity transform requirement**: allowing a transform on the adapter
  would double-apply a reparameterisation; construction rejects it.

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Construction and parameter surface | 4 |
| `__call__` and parameter mapping | 6 |
| `manual_inverse` | 5 |
| Parameter-map validation | 3 |
| Gradients | 6 |
| Cache control | 5 |
| Repr | 1 |
| **Total** | **30** |
