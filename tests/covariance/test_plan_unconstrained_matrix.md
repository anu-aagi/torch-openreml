# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.unconstrained_matrix`
**Purpose of Module:**
Provides a symmetric covariance matrix parameterised by its lower-triangular entries (diagonal variances on the exponential scale, off-diagonal entries unconstrained), with the upper triangle mirrored from the lower one. It is the "unstructured but symmetric" covariance building block and, unlike `LowerTriangularMatrix`, imposes exact symmetry and positivity on the diagonal without guaranteeing positive definiteness.

**Classes Covered:**

- `UnconstrainedMatrix` — symmetric matrix from unconstrained lower-triangular parameters

**Testing Goal:**
Pin the parameter naming and ordering (`sigma^2_{i}_{j}` over the lower triangle, column-major), the different default transforms on the diagonal (`TransformExpPow2`) versus the off-diagonal (`TransformIdentity`), exact symmetry and mirrored entries in `__call__`, the closed-form `manual_grad` (diagonal `2·exp(2θ)`, off-diagonal unit entries at both `(i, j)` and `(j, i)`) and its agreement with `auto_grad`, plus the base-class dtype/device, cache, repr and `manual_inverse` contracts. The class was previously covered only incidentally through `tests/covariance/test_matrix.py`.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires `n` and accepts optional `param_specs`; the shape is always square `(n, n)`
- `n` accepts `int`, `list`, `tuple` and `pandas.Series` of levels, resolved to the number of distinct elements
- Default parameter set is `sigma^2_{i}_{j}` for `i in range(n)` and `j in range(i + 1)`, i.e. `n(n + 1) / 2` parameters
- Parameter order is column-major within the lower triangle: `(0,0)`, `(1,0)`, `(1,1)`, `(2,0)`, `(2,1)`, `(2,2)`, ...
- Default specs: diagonal parameters are free with default `[0.0]` and `TransformExpPow2()`; off-diagonal parameters are free with default `[0.0]` and `TransformIdentity()`
- `num_params == num_free_params == n(n + 1)/2`, `free_param_index == list(range(num_params))`, `num_fixed_params == 0`
- Custom param_specs accepted; instances do not mutate or share the class-level default dict
- Non-positive and non-integer dimensions are rejected: non-integer `n` fails inside the default-spec comprehension (`range(n)` raises `TypeError`) before the base shape check, so non-integer tests accept `(TypeError, ValueError)`; non-positive integer `n` raises `ValueError("All elements of 'shape' must be positive int!")`

#### B. `__call__(free_params)` Behavior

- Returns an `n × n` matrix with `M[i, j] == M[j, i]` for every pair (exact symmetry, including off-diagonal entries)
- Every lower-triangular parameter appears at `(i, j)` and mirrored at `(j, i)`; the matrix is fully determined by the lower triangle
- The default call is the identity matrix (diagonal `exp(0) = 1`, off-diagonal `0`)
- Values follow the per-parameter transform: a free diagonal default `0.0` becomes `1.0`, a diagonal default `0.5` becomes `exp(1.0)`, an off-diagonal default `0.5` stays `0.5`
- Works for `n = 1` (single entry), `n = 2`, `n = 3` and larger
- Accepts a flat tensor, a parameter dict and no argument (defaults)
- Fixed and mixed configurations use the fixed defaults, including transformed fixed defaults
- A cast call does not change later default calls

#### C. `manual_grad(free_params)` Behavior

- Returns `(grad, names)` with `grad` of shape `(num_free_params, n, n)` and `names == free_param_names`
- Diagonal slices are populated only at `(i, i)` with `2·exp(2θ)`; off-diagonal slices are populated at both `(i, j)` and `(j, i)` with the transform derivative (unit for `TransformIdentity`)
- All other positions in each slice are zero
- Mixed fixed/free configuration: slices only for free parameters, with the fixed parameter's diagonal/off-diagonal contributions absent, and the name list shrinks accordingly
- All-fixed configuration returns `(None, [])`
- Agrees numerically with `auto_grad` for `n = 2`, `n = 3`, float64 and mixed configurations
- `trans_grad` matches the slice values: `2·exp(2θ)` for a diagonal parameter and `1` for an off-diagonal one

#### D. `auto_grad` and `grad` Dispatch

- `auto_grad` supports `jacfwd` (default), `jacrev` and `jacobian`, with and without `jacobian_chunk_size`
- An unknown Jacobian method raises `ValueError`
- `grad_mode` is `"default"` on construction; `"manual"` and `"auto"` select the corresponding path; an unknown mode raises `RuntimeError`
- `grad`, `manual_grad` and `auto_grad` accept a tensor, a dict and the implicit defaults

#### E. Fixed and Mixed Parameters

- Mixed 3×3 spec with the `sigma^2_0_0` diagonal fixed at `9.0`: `__call__` has `exp(18)` at `(0, 0)`, the free diagonal values are transformed and the off-diagonal values are mirrored
- `manual_grad` returns five free slices; the fixed diagonal slice is absent and the free-diagonal slices are `2·exp(2θ)`
- `build_params(include_fixed=True)` returns all six entries in column-major order; `include_fixed=False` returns five
- All-fixed spec: `num_free_params == 0`, empty `free_param_names`, full `fixed_param_index`, empty `build_params`/`trans_grad`
- A fixed-only matrix plus a non-empty free vector raises the base length error

#### F. Custom `param_specs`

- A custom naming scheme (e.g. `["a", "b", "c"]` for `n = 2`) is honoured by `__call__`, `build_params`, `manual_grad` and dict input
- Per-parameter transforms are honoured independently: a custom `TransformIdentity` on a diagonal entry keeps the raw value and derivative `1`, while a custom `TransformExpPow2` on an off-diagonal entry scales both `(i, j)` and `(j, i)`
- The order of the custom dict determines the parameter order used by `build_params` and the gradient slices
- Names containing `"/"` are rejected by the base class

#### G. `build_params` Integration

- Tensor output has length `num_params` in column-major order; with the default transforms `[1, 0, 1, 0, 0, 1]` for `n = 3` because diagonal `exp(0) = 1`
- `trans=False` gives the raw defaults (all zeros) and raw custom values
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

- `__call__`, `build_params`, `manual_grad`, `auto_grad` and `trans_grad` follow the dtype of the supplied parameters (`float32`/`float64`)
- The default dtype follows `torch.get_default_dtype()` at construction and a cast call does not change it
- An integer parameter tensor is promoted to a float dtype because the diagonal transform is exponential; the result is still exactly symmetric
- Accelerator tests (`cuda` → `mps` → `pytest.skip("no accelerator available")`) cover every public method, dtype preservation (`float16`), and non-mutation of later default calls

#### J. Intermediate Cache

- `get_intermediates` before a set returns `None`; set-then-get returns the value
- Keys are compared by value, dtype and device; the stored key is a copy
- `reset_intermediates` clears the cache; an empty parameter tensor is a no-op for get and set
- Writing the same key twice replaces the value
- `auto_grad` clears the cache first; `manual_grad` and `__call__` leave it intact
- Get and set validate a non-tensor (`TypeError`) and a non-1D tensor (`ValueError`)

#### K. `manual_inverse`

`UnconstrainedMatrix` does **not** override the base implementation. The documented contract is the base `NotImplementedError` naming the class and pointing at `torch.linalg.inv(self(free_params))`. The tests assert no `"manual_inverse"` key in the class dictionary and the raise for defaults, an explicit tensor, a dict and an all-fixed configuration.

#### L. `repr`

- The single-line repr starts with `UnconstrainedMatrix(`, contains the shape and the first parameter name
- With `n = 2` there are three parameters, so the base class abbreviates the spec dict with `...`: the first (`sigma^2_0_0`) and last (`sigma^2_1_1`) names are present and `sigma^2_1_0` is absent
- `n = 1` has a single parameter and prints the full spec without `...`
- `repr_dict` exposes the shape tuple and the same `param_specs` object

#### M. Spec-Count Enforcement

`UnconstrainedMatrix` has `n(n+1)/2` lower-triangular slots, and the constructor rejects any other count:

- `n = 2` has three slots: one spec raises `ValueError: ... expected 3 parameter specs, got 1`; four specs raise `ValueError: ... expected 3 parameter specs, got 4`
- `n = 3` has six slots: five specs are rejected and six are accepted
- A correctly sized custom surface keeps its names and order; fixed entries count toward the total (three specs with one fixed → three parameters, two free)
- Mutating `param_specs` after construction is unsupported and bypasses the check

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use `n = 2` and `n = 3` for value and gradient tests; `n = 2` is the smallest size that exercises both a diagonal and an off-diagonal parameter
- Assert symmetry and mirroring independently: check `result == result.T` and, for each lower-triangular parameter, check both `(i, j)` and `(j, i)` against the expected transformed value
- Verify the two transform scales separately: diagonal `exp(2θ)`, off-diagonal raw `θ`; a single all-Identity custom spec is used when the intent is pure placement
- Cross-check `manual_grad` against `auto_grad` with `torch.allclose` on all-free, mixed and custom-transform configurations, and check names independently
- Keep spec-count enforcement in one class (`...SpecCount`); the constructor rejects too few and too many entries, and `n = 3` pins the `n(n+1)/2` formula
- Repeat the dtype/device matrix on every public method using the inline `cuda` → `mps` → `skip` pattern

### 3.2 Test Structure

Every test follows **Arrange → Act → Assert**, with grouped classes per contract and module-level spec helpers:

```python
import pandas as pd
import pytest
import torch

from torch_openreml.covariance import UnconstrainedMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExpPow2,
    TransformIdentity,
)


def lower_triangle_names(n):
    return [f"sigma^2_{i}_{j}" for i in range(n) for j in range(i + 1)]


def spec(*, fixed=False, default=0.0, trans=None):
    return {
        "fixed": fixed,
        "default": torch.tensor([float(default)]),
        "trans": trans if trans is not None else TransformExpPow2(),
    }


class TestUnconstrainedMatrixCallValues:
    """``__call__`` is symmetric and mirrors every lower-triangular entry."""

    def test_call_is_symmetric(self):
        mat = UnconstrainedMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 0.0, 1.0, 1.5, 0.0])
        result = mat(free_params)
        assert torch.allclose(result, result.T)

    def test_call_mirrors_off_diagonal(self):
        mat = UnconstrainedMatrix(2)
        result = mat(torch.tensor([0.0, 2.0, 0.0]))
        assert torch.allclose(result, torch.tensor([[1.0, 2.0], [2.0, 1.0]]))
```

The remaining classes are `TestUnconstrainedMatrixConstructor`, `...FixedParams`, `...CustomTransforms`, `...Grad`, `...BuildParams`, `...FreeParamsValidation`, `...DtypeAndDevice`, `...Intermediates`, `...Inverse`, `...Repr` and `...SpecCount`.

### 3.3 Test Execution

```bash
conda run -n torch-openreml python -m pytest tests/covariance/test_unconstrained_matrix.py -q
```

Expected: `226 passed`.

---

## 4. Key Risks / Notes

- **Different transforms on the same matrix.** The diagonal uses `TransformExpPow2` and the off-diagonal uses `TransformIdentity`. Tests that forget this produce plausible-looking but wrong expectations (e.g. expecting a raw diagonal default `0.0` instead of the identity value `1.0`). The `build_params` default vector `[1, 0, 1, 0, 0, 1]` and the `2·exp(2θ)` diagonal gradient are the load-bearing assertions here.
- **Symmetry can mask a missing mirror.** Checking only `result == result.T` passes for an all-zero matrix, and checking only `(i, j)` passes even if `(j, i)` is never written. Every value test therefore asserts both sides against the expected transformed value, with distinct magnitudes per parameter.
- **Diagonal entries are written twice by the index pair.** `torch.tril_indices` yields `(i, i)` on the diagonal for both `i` and `j`, so the manual-gradient loop assigns the diagonal slice twice; the result is correct but the duplicate write means an off-by-one in the free mask can behave differently on diagonal versus off-diagonal parameters, which the mixed test (fixed diagonal) exercises.
- **Spec count is validated only at construction.** The constructor rejects a `param_specs` dict whose length differs from `n(n+1)/2`, which removes the old silent broadcast and masked `IndexError`. Directly mutating `param_specs` afterwards still bypasses the check, so the invariant is documented as construction-time only.
- **Integer inputs are promoted by the exponential diagonal.** Unlike a pure-identity-transform matrix, `int64` parameters produce a float matrix; the dtype tests assert the promotion and that symmetry survives it.
- **No closed-form inverse.** The absence of a `manual_inverse` override is part of the contract: `assert "manual_inverse" not in UnconstrainedMatrix.__dict__` guards future overrides.
- **`torch.jit.script` deprecation warning.** `auto_grad` with the default `jacfwd` emits a `FutureWarning` on torch 2.14; it is harmless and does not fail the suite.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 45 |
| `__call__` values & symmetry | 16 |
| Fixed & mixed free/fixed configs | 17 |
| Custom transforms | 4 |
| `manual_grad`, `auto_grad` & `grad` | 25 |
| `build_params` | 20 |
| Free-parameter validation | 42 |
| dtype & device | 23 |
| intermediates cache | 16 |
| `manual_inverse` (base) | 6 |
| repr | 7 |
| Spec-count enforcement | 5 |
| **Total** | **226** |
