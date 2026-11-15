# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.equal_entry_matrix`
**Purpose of Module:**
Provides a dense covariance matrix `V = σ² · J_{n×m}` in which every entry equals the same variance `σ²`. It is the dense, single-variance counterpart of `ScalarMatrix` and is the natural vehicle for repeated-measure structures whose observations share a single variance component.

**Classes Covered:**

- `EqualEntryMatrix` — all-entries-equal covariance `V = σ² · J`

**Testing Goal:**
Pin the single-parameter contract (one free `"sigma^2"` parameter with an `TransformExpPow2()` transform by default), verify that `__call__` fills every entry of the `n × m` matrix with the transformed variance, that the manual gradient is `dσ²/dθ · J` and agrees with `auto_grad`, that fixed, custom and mixed configurations behave, and that the base-class dtype/device, cache, repr and `manual_inverse` contracts hold. The class now enforces the single-variance contract at construction; the spec-count tests pin that rejection path.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires `n`, accepts optional `m` (defaults to `n`) and optional `param_specs`
- Default param_specs: one free parameter `"sigma^2"` with `TransformExpPow2()`, default `[0.0]`
- Instance is a subclass of `Matrix`
- Shape is `(n, m)`, including rectangular `m != n`
- `num_params == num_free_params == 1`, `num_fixed_params == 0`, `free_param_index == [0]`
- `n` and `m` accept `int`, `list`, `tuple` and `pandas.Series` of levels: the dimension is the number of distinct elements, so `EqualEntryMatrix(["a", "b", "a"])` is `EqualEntryMatrix(2)`. An empty sequence resolves to `0` and is rejected by the shape check
- Level forms match the integer form in shape, parameter names and `__call__` values
- Non-positive and non-integer dimensions are rejected; a `bool` is an `int` in Python and is accepted as `1` at construction but fails on `__call__`
- Custom param_specs accepted; an empty or other falsy non-dict falls back to the defaults without mutating the class-level default
- Instances do not share the default spec dicts or transform objects

#### B. `__call__(free_params)` Behavior

- Returns an `n × m` matrix whose every entry is the transformed `σ²`
- Default transform `TransformExpPow2` makes every entry `exp(2θ) > 0`; defaults give an all-ones matrix
- The structure is exactly dense: no zeros, no triangular structure, rank one
- Works with `n = m = 1`, square `n = m` and rectangular pairs `(3, 2)`, `(2, 3)`, `(4, 1)`
- Accepts a flat tensor and a parameter dict, and no argument (defaults)
- Fixed-only and mixed configurations use fixed defaults for the fixed entries
- Cast calls do not change later default calls

#### C. `manual_grad(free_params)` Behavior

- Returns `(grad, names)` with `grad` of shape `(num_free_params, n, m)` and `names == free_param_names`
- Every entry of the single slice is the transform derivative; for the default spec `grad[0] = 2·exp(2θ)·J`
- All-fixed configuration returns `(None, [])`
- Mixed fixed/free configuration returns slices for free parameters only, but the same dense shape
- Agrees numerically with `auto_grad` for square, rectangular, float64 and mixed configurations
- `trans_grad` matches the manual gradient values

#### D. `auto_grad` and `grad` Dispatch

- `auto_grad` supports `jacfwd` (default), `jacrev` and `jacobian`, with and without `jacobian_chunk_size`
- An unknown Jacobian method raises `ValueError`
- `grad_mode` is `"default"` on construction; `"manual"` and `"auto"` select the corresponding path, an unknown mode raises `RuntimeError`
- `grad`, `manual_grad` and `auto_grad` accept a tensor, a dict, and the implicit defaults

#### E. Fixed and Mixed Parameters

- `num_free_params == 0`, `num_fixed_params == 1`, empty `free_param_names` and the full `fixed_param_index`
- `build_params(include_fixed=False)` is an empty tensor and an empty dict; `trans_grad` is empty
- Mixed spec: `("sigma^2"` free plus a second fixed parameter`)` is the structural stress case; the fixed value is used for the fixed slot and only free slices are returned
- A fixed-only matrix rejects any non-empty free parameter vector with the length error

#### F. Custom `param_specs`

- A custom single-parameter name (e.g. `"v"`) is honoured by `__call__`, `build_params`, `grad` and the dict format
- A custom transform (`TransformExp`) changes both the value and the derivative
- Custom names with `"/"` are rejected by the base class

#### G. `build_params` Integration

- Tensor output length 1, transformed by default and raw with `trans=False`
- Dict output maps `"sigma^2"` to a one-element tensor; all-fixed and `include_fixed=False` produce `{}` and the tensor path produces an empty tensor
- Mixed spec: full tensor length 2, `include_fixed=False` length 1, both with and without transforms
- Dict input is converted to the same tensor as the flat input; a dict of free-parameter defaults is accepted
- Unknown `out_format` raises `ValueError`, including when the output would be empty
- Missing and unexpected dict keys raise `ValueError`; mixed dtype/device dict values raise `ValueError` before concatenation

#### H. Free-Parameter Validation

- Every public entry point (`__call__`, `grad`, `manual_grad`, `auto_grad`, `trans_grad`, `build_params`) rejects a wrong length with the documented `"length 2"` message and a non-1D tensor with `"1D tensor"`
- A zero-dimensional tensor is rejected by `__call__`
- Non-tensor values (list, float, string) raise `TypeError`
- Dict input routes through the same checks: an empty dict reports missing parameters, an extra key reports unexpected parameters (the message contains a set repr, so exactly one unexpected key is used in the match)

#### I. dtype and device

- `__call__`, `build_params`, `manual_grad`, `auto_grad` and `trans_grad` follow the dtype of the supplied parameters for `float32`/`float64`
- The default dtype follows `torch.get_default_dtype()` at construction time and a cast call does not change it
- An integer parameter tensor is promoted to the default float dtype by `TransformExpPow2`; the result is still a valid all-equal matrix
- On an accelerator present (`cuda` then `mps`, otherwise `pytest.skip("no accelerator available")`), every method returns on the input device; a dtype/device pair (`float16` on the accelerator) is preserved together, and an accelerator call does not move later default calls

#### J. Intermediate Cache

- `get_intermediates` before a set returns `None`; set-then-get returns the value
- Keys are compared by value, dtype and device; the stored key is a copy so mutating the caller's tensor does not invalidate the entry
- `reset_intermediates` clears the cache; an empty parameter tensor is a no-op for both get and set
- Writing the same key twice replaces the value
- `auto_grad` clears the cache first; `manual_grad` and `__call__` leave it intact
- Both get and set validate a non-tensor (`TypeError`) and a non-1D tensor (`ValueError`)

#### K. `manual_inverse`

`EqualEntryMatrix` does **not** override the base implementation, so the documented contract is the base class's `NotImplementedError` naming the class and pointing at `torch.linalg.inv`. The test asserts the class dictionary has no `"manual_inverse"` entry and the raise happens with defaults, explicit free parameters, a dict and an all-fixed configuration, with both message fragments matched.

#### L. `repr`

- The single-line repr starts with `EqualEntryMatrix(`, contains the shape and the parameter name
- With one parameter the full spec is shown and no `...` abbreviation appears
- `repr_dict` exposes the shape tuple and the same `param_specs` object

#### M. Single-Spec Enforcement

The class is a one-variance model, and the constructor now rejects any `param_specs` dict that does not contain exactly one parameter:

- Two free parameters raise `ValueError` naming the required count and reporting the received count ("got 2")
- One free plus one fixed parameter raises the same error
- Three parameters raise the same error
- Empty specs still fall back to the default single `"sigma^2"` spec (the constructor's `or` default)
- One custom spec is accepted and drives `__call__`, `num_free_params` and the parameter names

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Start from the default all-free single-parameter configuration, then layer fixed, mixed and custom specs on top
- Verify exact values and dense structure, not just shapes: every entry of `__call__` must equal the same transformed scalar, and the single gradient slice must be uniformly `2·exp(2θ)`
- Use `torch.linalg`-free closed forms for the expected values so the tests do not mirror the implementation
- Keep the spec-count rejection in one deliberately named class so the single-variance contract is enforced rather than accidentally relaxed
- Repeat the dtype/device matrix on every public method; use the inline `cuda` → `mps` → `skip` pattern for device tests

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**. Grouped classes per contract, with module-level spec helpers. Representative structure:

```python
import pandas as pd
import pytest
import torch

from torch_openreml.covariance import EqualEntryMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExp,
    TransformExpPow2,
    TransformIdentity,
)


def single_spec(name="sigma^2", *, fixed=False, default=0.0, trans=None):
    return {
        "fixed": fixed,
        "default": torch.tensor([float(default)]),
        "trans": trans if trans is not None else TransformExpPow2(),
    }


class TestEqualEntryMatrixCallValues:
    """``__call__`` fills every entry with the same variance."""

    def test_call_fills_every_entry(self):
        mat = EqualEntryMatrix(3, 2)
        result = mat(torch.tensor([0.5]))
        assert torch.allclose(result, torch.full((3, 2), torch.exp(torch.tensor(1.0))))

    def test_call_agrees_with_manual_grad(self):
        mat = EqualEntryMatrix(3, 2)
        free_params = torch.tensor([0.3])
        manual, _ = mat.manual_grad(free_params)
        auto, _ = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
```

Other classes are `TestEqualEntryMatrixConstructor`, `...FixedParams`, `...BuildParams`, `...FreeParamsValidation`, `...Transforms`, `...Grad`, `...DtypeAndDevice`, `...Intermediates`, `...Inverse`, `...Repr` and `...MultiParamSpecLimitation`.

### 3.3 Test Execution

```bash
conda run -n torch-openreml python -m pytest tests/covariance/test_equal_entry_matrix.py -q
```

Expected: `237 passed`.

---

## 4. Key Risks / Notes

- **Single-variance model is enforced.** `EqualEntryMatrix.__init__` raises `ValueError` when `param_specs` contains anything other than one entry, so the former silent broadcast (`2 × 2`) and `RuntimeError` (`3 × 3`) paths are unreachable. `TestEqualEntryMatrixSpecCount` covers the rejection and the one-custom-spec acceptance path.
- **The default fallback is still `or`-based.** Empty `{}`/`[]` specs fall back to the single default `"sigma^2"` spec; only non-empty multi-parameter specs raise. The fallback tests document this boundary.
- **dtype follows the input, but the default transform promotes integers.** `TransformExpPow2` is an exponential, so an `int64` parameter vector yields a float matrix; tests assert the promotion explicitly rather than assuming the input dtype is preserved.
- **No closed-form inverse.** The absence of a `manual_inverse` override is itself part of the contract here: `assert "manual_inverse" not in EqualEntryMatrix.__dict__` guards against a future override silently changing error behaviour.
- **`repr` abbreviation threshold.** The base class abbreviates at three or more parameters; with the single default parameter the full spec must be printed, so the tests check both the presence of the name and the absence of `...`.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 51 |
| `__call__` values & dense structure | 20 |
| Fixed & mixed free/fixed configs | 12 |
| `build_params` | 21 |
| Free-parameter validation | 37 |
| Custom transforms | 8 |
| `manual_grad`, `auto_grad` & `grad` | 31 |
| dtype & device | 23 |
| intermediates cache | 16 |
| `manual_inverse` (base) | 6 |
| repr | 7 |
| Single-spec enforcement | 5 |
| **Total** | **237** |
