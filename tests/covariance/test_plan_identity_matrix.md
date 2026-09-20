# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.identity_matrix`
**Purpose of Module:**
Provides a fixed `n × n` identity covariance matrix with no trainable parameters. Used to represent independent, homoscedastic residuals in linear mixed-effects models (i.e., `V = I_n`).

**Classes Covered:**

- `IdentityMatrix` — fixed identity matrix `V = I_n`

**Testing Goal:**
Ensure the matrix is correctly constructed as the identity, exposes a completely empty parameter surface, follows the dtype and device of its input, rejects any non-empty parameter input, and returns `(None, [])` from every gradient path. This is the simplest covariance matrix, so it also serves as the concrete vehicle for the cross-cutting dtype/device resolution rules and for the zero-parameter edge cases of the `Matrix` ABC.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor takes `n` and nothing else: `dtype` and `device` are not constructor arguments
- Instance is a subclass of `Matrix`
- Shape is `(n, n)`, stored as a tuple
- `n` is validated: `0` raises `ValueError` from the shape check, a negative `n` raises `RuntimeError` from `torch.eye`, a non-int `n` (float, str, bool, `None`) raises `TypeError`
- `n` may be passed as a keyword; a 0-dimensional tensor `n` is accepted by `torch.eye` and then rejected by the shape check with `ValueError`, not `TypeError`
- The whole parameter surface is empty: `param_specs`, `param_defaults`, `param_trans` and their free/fixed counterparts are `{}`; the name and index properties are `[]`; `num_params == num_free_params == num_fixed_params == 0`
- `grad_mode` defaults to `"default"`
- `repr_dict` is `{"shape": (n, n), "param_specs": {}}`

#### B. `__call__(free_params=None)` Behavior

- Returns `torch.eye(n)` for every accepted form of the argument: omitted, `None`, an empty tensor, an empty dict, or the `free_params` keyword
- Diagonal entries are 1, off-diagonal are 0, and the matrix is symmetric
- Correct size for a range of `n`, including the `n = 1` boundary
- Returns a freshly allocated tensor: mutating the result, or holding two results from consecutive calls, must not affect the matrix or later calls
- Rejects any non-empty free-parameter input: a non-empty tensor (`ValueError`, length 0) or a dict with any key (`ValueError`, unexpected free parameters)
- Rejects a non-tensor (`TypeError`) and a non-1D or 0-dimensional tensor (`ValueError`)
- Rejects an unexpected keyword argument (`TypeError`)

#### C. dtype and device resolution (cross-cutting contract)

- Rule 1: the dtype and the device of the input free-parameter tensor are followed, including non-float dtypes (e.g. `int64`), which are followed verbatim
- Rule 2: with no input, the Torch default dtype and device are used, resolved at call time rather than at construction time, for the device as well as the dtype
- An empty dict input also follows the Torch defaults, since it carries no tensor to read from
- A call that casts the matrix does not change the dtype of later default calls
- A dtype and a device that both differ from the defaults are followed together; the dtype used is one the accelerator supports (`float16`), since MPS rejects `float64` outright
- `build_params` and `trans_grad` follow the input dtype as well
- Device tests pick `cuda` → `mps` → `pytest.skip("no accelerator available")` inline

#### D. `build_params` Behavior

- Returns an empty 1D tensor by default, and `{}` in `"dict"` format
- `include_fixed` and `trans` do not change the empty result, in either output format
- Rejects an unknown `out_format` in both the empty-output branch and the parameterized branch
- Rejects non-empty tensor and dict input

#### E. Gradient Behavior

- `grad`, `auto_grad` return `(None, [])` for omitted, `None`, empty-tensor and empty-dict input
- `auto_grad` returns `(None, [])` even when `jacobian_method` is invalid, proving the zero-length early return precedes the Jacobian dispatch so no differentiation runs
- `manual_grad` raises `NotImplementedError`, with or without arguments
- `grad_mode = "manual"` surfaces that `NotImplementedError`; `grad_mode = "auto"` returns `(None, [])`; an unknown mode raises `RuntimeError`
- Gradient entry points still validate their input on the zero-parameter path: a non-empty tensor raises `ValueError`, a non-tensor raises `TypeError`
- `trans_grad` returns an empty tensor (also when called with no arguments) and validates its input
- `get_default_dtype_device` falls back to the Torch defaults and tracks changes to the default dtype

#### F. Intermediate Caching

- `get_intermediates` returns `None` before any `set_intermediates` call
- `set_intermediates` with empty parameters is a no-op returning `None`, so the cache can never be populated and a later `get_intermediates` is still `None`
- `reset_intermediates` leaves the cache empty
- Both methods validate `params`: a non-tensor raises `TypeError`, a non-1D tensor raises `ValueError`

#### G. `repr`

- Contains the class name and the shape, is a single line, and omits the empty `param_specs`

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Test with small `n` values (1, 2, 4, 5) for easy manual verification, plus the `n = 1` boundary
- Cover the full empty-parameter surface, using `pytest.mark.parametrize` over the property names rather than one test per property
- Test dtype and device preservation, including resolution at call time and the "cast does not stick" rule
- Use `torch.set_default_dtype` inside a `try`/`finally` for the default-resolution tests, so the global default is always restored
- Verify the error contract (type and message) at every entry point that accepts `free_params`: `__call__`, `build_params`, `grad`, `auto_grad`, `trans_grad`, `set_intermediates`, `get_intermediates`

### 3.2 Test Structure

Tests are grouped into classes by contract area, following **Arrange → Act → Assert**:

```python
class TestIdentityMatrixCallValues:
    """The matrix returned by ``__call__``."""

    def test_result_can_be_mutated_without_affecting_later_calls(self):
        mat = IdentityMatrix(3)
        result = mat()
        result[0, 0] = 99.0
        result[0, 1] = 99.0
        assert torch.equal(mat(), torch.eye(3))
```

### 3.3 Test Execution

`pytest tests/covariance/test_identity_matrix.py`

---

## 4. Key Risks / Notes

- **Aliasing of the returned tensor**: `__call__` resolves the dtype and device with `Tensor.to`, which returns the receiver when nothing needs converting. Returning that directly would hand the caller the internal `_matrix`, so an in-place edit such as `mat()[0, 0] = 99.0` would silently change every later call. `__call__` therefore clones on the no-conversion path, and `test_result_can_be_mutated_without_affecting_later_calls` and `test_consecutive_calls_are_distinct_tensors` pin that down. The same pattern is still present in `dummy_matrix.py`.
- **Resolution happens at call time, not construction time**: with no input, the dtype and device come from `Matrix.get_default_dtype_device`, which returns the Torch defaults for a matrix with no parameters. Changing the default dtype after construction therefore changes the result of `mat()`.
- **Empty dict input is not a dtype source**: `{}` carries no tensor, so it falls back to the Torch defaults rather than following any input dtype. `test_empty_dict_follows_torch_default_dtype` covers this.
- **Non-float dtypes are followed verbatim**: an empty `int64` parameter tensor yields an integer identity rather than being promoted or rejected, which is the general "follow the input" rule applied without a floating-point restriction.
- **`manual_grad` does not validate its input**: it raises `NotImplementedError` unconditionally, so it cannot be used to probe the parameter contract. Validation on the gradient path happens in `auto_grad` instead, which is why `grad(torch.ones(1))` raises `ValueError`.
- **The intermediate cache is dead weight here**: `set_intermediates` returns early when `params` has length 0, and an empty tensor is the only valid input, so the cache can never hold a value. The tests assert the no-op rather than an unreachable populated state.
- **Boundary of `n` validation**: `n = 0` is caught by `Matrix._check_shape` (`ValueError`), while a negative `n` is rejected earlier by `torch.eye` itself (`RuntimeError`). Both are pinned, so any future move to validate `n` inside the constructor will surface as a test change.
- **Removed REML interface**: `map_theta_to_v` and `map_theta_to_dv` no longer exist on `Matrix`, so this module no longer tests a REML-specific mapping; `__call__` and `grad` are the interface.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor, shape, empty parameter surface, `repr_dict` | 29 |
| `__call__` values, argument forms, freshness | 15 |
| dtype & device resolution | 14 |
| Free-parameter validation | 13 |
| `build_params` | 12 |
| `grad` / `auto_grad` / `manual_grad` / `trans_grad` | 21 |
| Intermediate caching | 8 |
| `repr` | 4 |
| **Total** | **116** |
