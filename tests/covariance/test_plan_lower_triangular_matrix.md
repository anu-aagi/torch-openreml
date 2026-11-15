# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.lower_triangular_matrix`
**Purpose of Module:**
Provides a covariance matrix whose free parameters are its lower-triangular entries and whose strictly upper-triangular entries are zero. It is the unstructured Cholesky-style building block used by composite operators and is the natural vehicle for a covariance that is not assumed symmetric.

**Classes Covered:**

- `LowerTriangularMatrix` — free lower-triangular entries, zeros above the diagonal

**Testing Goal:**
Pin the parameter naming and ordering (`L_{i}_{j}`, row-major over the lower triangle), the rectangular `n × m` placement including zero-filling above the diagonal, the default `TransformIdentity()` transform and `1.0` default, the closed-form `manual_grad` (one unit entry per lower-triangular slot, filtered to free parameters) and its agreement with `auto_grad`, and the base-class dtype/device, cache, repr and `manual_inverse` contracts. The class previously had only incidental coverage in `tests/covariance/test_matrix.py`; the custom-spec parameter-count contract is enforced and tested here.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires both `n` and `m` (there is no default for `m`) and accepts optional `param_specs`
- Shape is `(n, m)`; both dimensions accept `int`, `list`, `tuple` and `pandas.Series` of levels, resolved to the number of distinct elements
- Default parameter set is `L_{i}_{j}` for `i in range(n)` and `j in range(min(i + 1, m))`, i.e. `sum_i min(i + 1, m)` parameters
- Parameter order is row-major (all `j` for `i = 0`, then `i = 1`, ...), and `free_param_index == list(range(num_params))`
- Default spec per parameter: free, default `torch.tensor([1.0])`, `TransformIdentity()`
- Rectangular shapes: `m > n` caps each row at `m` slots but there are only `i + 1` lower entries, so `(1, 4)` has one parameter and `(2, 4)` has three; `m < n` produces a genuinely truncated lower triangle (e.g. `(4, 2)` has seven parameters)
- The matrix is square-agnostic: no symmetry is imposed or assumed
- Custom param_specs accepted; instances do not mutate or share the class-level default dict
- Non-positive and non-integer dimensions are rejected. Note that a non-integer `n` fails earlier, inside the default-spec comprehension (`range(n)` raises `TypeError: 'float' object cannot be interpreted as an integer`), than in the base shape check (`ValueError("All elements of 'shape' must be positive int!")`); the tests accept either exception class for the non-integer cases and pin the `ValueError` only for non-positive integer dimensions

#### B. `__call__(free_params)` Behavior

- Every lower-triangular slot `(i, j)` with `j <= i` and `j < m` receives the corresponding parameter value; entries with `j > i` are exactly zero
- Default call is the lower-triangular all-ones matrix
- Rectangular cases: `(1, 1)`, `(1, 4)`, `(3, 1)`, `(2, 4)` produce the documented zero-padded layouts
- Values are placed in the row-major order of `free_param_names`, not column-major (a permutation test with a strict monotone vector catches the ordering)
- Accepts a flat tensor, a parameter dict and no argument (defaults)
- Fixed and mixed configurations use fixed defaults for the fixed slots
- A cast call does not change later default calls

#### C. `manual_grad(free_params)` Behavior

- Returns `(grad, names)` with `grad` of shape `(num_free_params, n, m)` and `names == free_param_names`
- Only lower-triangular positions of each slice are populated; positions above the diagonal are zero in every slice
- With the default `TransformIdentity` the populated value is `1.0`; with `TransformExpPow2` on `L_0_0` it is `2e` at `(0, 0)`
- Mixed fixed/free configuration: fixed slot removed from the slices, remaining slices keep their original matrix positions and the name list shrinks accordingly
- All-fixed configuration returns `(None, [])`
- Agrees numerically with `auto_grad` for square, rectangular, float64 and mixed configurations
- `trans_grad` matches the populated manual-gradient values in row-major order

#### D. `auto_grad` and `grad` Dispatch

- `auto_grad` supports `jacfwd` (default), `jacrev` and `jacobian`, with and without `jacobian_chunk_size`
- An unknown Jacobian method raises `ValueError`
- `grad_mode` is `"default"` on construction; `"manual"` and `"auto"` select the corresponding path; an unknown mode raises `RuntimeError`
- `grad`, `manual_grad` and `auto_grad` accept a tensor, a dict and the implicit defaults

#### E. Fixed and Mixed Parameters

- All-fixed spec: `num_free_params == 0`, `num_fixed_params == num_params`, empty `free_param_names`, full `fixed_param_index`
- `build_params(include_fixed=False)` is empty in both tensor and dict form; `trans_grad` is empty
- Mixed 3×2 spec with `L_1_0` fixed at `9.0`: `__call__` places the fixed value, `manual_grad` returns four free slices and equal auto-grad
- A fixed-only matrix plus a non-empty free vector raises the base length error

#### F. Custom `param_specs`

- A custom set of names (e.g. `["a", "b", "c"]` for a 2×2) is honoured by `__call__`, `build_params`, `manual_grad` and dict input
- Per-parameter transforms are honoured independently: mixing `TransformExpPow2` and `TransformIdentity` changes individual entries and their derivatives
- A custom transform on a fixed parameter still affects its value in `__call__`/`build_params`
- Names containing `"/"` are rejected by the base class

#### G. `build_params` Integration

- Tensor output has length `num_params` and follows the default `TransformIdentity` (raw values)
- `trans=True` on a custom `TransformExpPow2` spec returns the transformed values; `trans=False` returns the raw ones
- Dict output maps each name to a one-element tensor, including fixed parameters
- `include_fixed=False` returns only free parameters in both formats; all-fixed produces an empty tensor and `{}`
- Dict input is converted to the same tensor as the flat input
- An unknown `out_format` raises `ValueError`, including on the empty-output path
- Missing and unexpected dict keys raise `ValueError`; mixed dtype/device dict values raise `ValueError` before concatenation

#### H. Free-Parameter Validation

- Every public entry point (`__call__`, `grad`, `manual_grad`, `auto_grad`, `trans_grad`, `build_params`) rejects a wrong length with the documented length message
- A non-1D tensor raises `ValueError("Parameters must be a 1D tensor!")`; a non-tensor raises `TypeError`
- Dict input routes through the same checks (missing/extra keys, values sharing dtype and device)

#### I. dtype and device

- `__call__`, `build_params`, `manual_grad`, `auto_grad` and `trans_grad` follow the dtype of the supplied parameters
- Because the default transform is `TransformIdentity`, an integer parameter tensor is preserved as an integer matrix; a custom exponential transform promotes it to float
- The default dtype follows `torch.get_default_dtype()` at construction and a cast call does not change it
- Accelerator tests (`cuda` → `mps` → `pytest.skip("no accelerator available")`) cover every public method, dtype preservation (`float16`), and non-mutation of later default calls

#### J. Intermediate Cache

- `get_intermediates` before a set returns `None`; set-then-get returns the value
- Keys are compared by value, dtype and device; the stored key is a copy
- `reset_intermediates` clears the cache; an empty parameter tensor is a no-op for get and set
- Writing the same key twice replaces the value
- `auto_grad` clears the cache first; `manual_grad` and `__call__` leave it intact
- Get and set validate a non-tensor (`TypeError`) and a non-1D tensor (`ValueError`)

#### K. `manual_inverse`

`LowerTriangularMatrix` does **not** override the base implementation. The documented contract is the base `NotImplementedError` naming the class and pointing at `torch.linalg.inv(self(free_params))`. The tests assert no `"manual_inverse"` key in the class dictionary and the raise for defaults, an explicit tensor, a dict and an all-fixed configuration.

#### L. `repr`

- The single-line repr starts with `LowerTriangularMatrix(`, contains the shape and the first parameter name
- With three or more parameters the base class abbreviates the spec dict with `...`; the first and last names are present and the middle ones are absent
- A one-parameter `(1, 1)` matrix prints the full spec without `...`
- `repr_dict` exposes the shape tuple and the same `param_specs` object

#### M. Spec-Count Enforcement

`LowerTriangularMatrix` has one parameter per lower-triangular slot, and the constructor rejects any other count:

- `(2, 2)` has three slots: one spec raises `ValueError: ... expected 3 parameter specs, got 1`; four specs raise `ValueError: ... expected 3 parameter specs, got 4`
- `(4, 2)` has seven slots (rows of 1, 2, 2, 2): six specs are rejected and seven are accepted
- A correctly sized custom surface keeps its names and order; fixed entries count toward the total, so three specs with one fixed still satisfy the check (three parameters, two free)
- Mutating `param_specs` after construction is unsupported and bypasses the check

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use the default `3 × 2` and `4 × 2` matrices for the ordering and rectangular-zero assertions, and `2 × 2` for the custom-spec and mixed cases
- Place parameter values with distinct, non-symmetric magnitudes so a row/column transposition or a diagonal offset is visible; assert the exact zero pattern separately from the values
- Cross-check `manual_grad` against `auto_grad` with `torch.allclose` on both square and rectangular shapes, and check the names list independently
- Keep spec-count enforcement in one class (`...SpecCount`); the constructor rejects too few and too many entries, and a rectangular shape pins the slot formula
- Repeat the dtype/device matrix on every public method using the inline `cuda` → `mps` → `skip` pattern

### 3.2 Test Structure

Every test follows **Arrange → Act → Assert**, with grouped classes per contract and module-level spec helpers:

```python
import pandas as pd
import pytest
import torch

from torch_openreml.covariance import LowerTriangularMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExpPow2,
    TransformIdentity,
)


def spec(*, fixed=False, default=1.0, trans=None):
    return {
        "fixed": fixed,
        "default": torch.tensor([float(default)]),
        "trans": trans if trans is not None else TransformIdentity(),
    }


def mixed_3x2_specs(fixed_name="L_1_0", fixed_value=9.0):
    specs = {}
    for i in range(3):
        for j in range(min(i + 1, 2)):
            name = f"L_{i}_{j}"
            specs[name] = spec(fixed=name == fixed_name, default=fixed_value if name == fixed_name else 1.0)
    return specs


class TestLowerTriangularMatrixCallValues:
    """``__call__`` places parameters in the lower triangle and zero-fills the rest."""

    def test_upper_triangle_is_zero(self):
        mat = LowerTriangularMatrix(3, 2)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mat(values)
        assert torch.all(result[0, 1] == 0)
        assert torch.all(result[1, 1:] == 0)
```

The remaining classes are `TestLowerTriangularMatrixConstructor`, `...FixedParams`, `...CustomTransforms`, `...Grad`, `...BuildParams`, `...FreeParamsValidation`, `...DtypeAndDevice`, `...Intermediates`, `...Inverse`, `...Repr` and `...SpecCount`.

### 3.3 Test Execution

```bash
conda run -n torch-openreml python -m pytest tests/covariance/test_lower_triangular_matrix.py -q
```

Expected: `233 passed`.

---

## 4. Key Risks / Notes

- **Spec count is validated only at construction.** The constructor rejects a `param_specs` dict whose length differs from the number of lower-triangular slots, which removes the old silent broadcast and masked `IndexError`. Directly mutating `param_specs` afterwards still bypasses the check, so the invariant is documented as construction-time only.
- **Ordering is row-major, not column-major.** `torch.tril_indices` returns row-major indices, and both `__call__` and `manual_grad` rely on that order matching `free_param_names`. A vector with distinct values catches a transposition; the strictly-increasing placement test is the load-bearing one.
- **Rectangular shapes truncate the lower triangle.** For `m > n` the number of parameters saturates at the triangle size (not `n · m`), so a test that assumes one parameter per column is wrong; `(1, 4)` has one parameter and four columns of output.
- **The default `TransformIdentity` preserves integer dtypes.** Unlike the exponential-transform classes, an `int64` parameter vector yields an `int64` matrix; the dtype tests assert that explicitly, and a custom exponential transform is used to cover the promotion path.
- **Fixed slots are removed by a boolean mask, not by re-indexing the tensor.** The mixed test uses a fixed interior slot (`L_1_0`), which is the case where an off-by-one in the mask shows up; fixing the first or last slot would pass even with a shifted mask.
- **No closed-form inverse.** The absence of a `manual_inverse` override is part of the contract: `assert "manual_inverse" not in LowerTriangularMatrix.__dict__` guards future overrides.
- **`torch.jit.script` deprecation warning.** `auto_grad` with the default `jacfwd` emits a `FutureWarning` on torch 2.14; it is harmless and does not fail the suite.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 51 |
| `__call__` values & triangular structure | 19 |
| Fixed & mixed free/fixed configs | 16 |
| Custom transforms | 4 |
| `manual_grad`, `auto_grad` & `grad` | 25 |
| `build_params` | 19 |
| Free-parameter validation | 42 |
| dtype & device | 23 |
| intermediates cache | 16 |
| `manual_inverse` (base) | 6 |
| repr | 7 |
| Spec-count enforcement | 5 |
| **Total** | **233** |
