# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.equicorrelation_matrix`
**Purpose of Module:**
Provides an equicorrelation matrix `V = (1-ρ)I_n + ρ J_n` with a single correlation parameter. The diagonal entries are fixed at 1, and all off-diagonal entries equal ρ. This is the correlation-matrix counterpart of `CompoundSymmetricMatrix` (which additionally has a variance parameter σ²).

**Classes Covered:**

- `EquicorrelationMatrix` — equicorrelation matrix `V = (1-ρ)I_n + ρ J_n`

**Testing Goal:**
Ensure correct construction (unit diagonal, uniform off-diagonal), correlation range enforcement, and single-parameter gradient correctness. This is the simplest Matrix with a single constrained parameter and explicit positive-definiteness bounds.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires `n`, optional `param_specs`
- `rho_min = -1/(n-1)` for different n
- Default param_specs: one free param `rho` with transform Chain(Sigmoid, ScaleShift(1-rho_min, rho_min))
- Instance is a subclass of `Matrix`
- Shape is `(n, n)`
- `num_params` = 1, `num_free_params` = 1 (default)

#### B. `__call__(free_params)` Behavior

- Returns (1-ρ)I + ρ J
- All diagonal entries equal 1
- All off-diagonal entries equal ρ
- Matrix is symmetric
- For ρ = 0: returns I_n
- For ρ close to 1: all entries ≈ 1
- Default transform enforces ρ ∈ (rho_min, 1)
- Works for different n values (2, 3, 4)
- n=1: trivially [1]

#### C. Correlation Bounds

- ρ ∈ (-1/(n-1), 1) enforced by transform chain
- At free_param → -inf: ρ → rho_min
- At free_param → +inf: ρ → 1 (from below)

#### D. `manual_grad(free_params)` Behavior

- Single gradient slice: `dV/d(rho_free) = (J - I) * trans_grad_rho`
- Shape: `(1, n, n)`
- Diagonal of grad = 0
- Off-diagonal entries all equal (= trans_grad_rho)
- Agrees with `auto_grad`
- Returns `(None, [])` when rho is fixed

#### E. Fixed Parameter

- `rho` fixed: `num_free_params` = 0
- `grad` returns `(None, [])`
- `__call__` uses fixed default value (transformed)

#### F. Intermediate Caching

- Caches I_n, J_n, and the full matrix V
- Cache hit second time for same params
- Cache invalidated by `reset_intermediates`

#### G. Difference from CompoundSymmetricMatrix

- No sigma^2 parameter — diagonal is always 1
- V is cached directly (not assembled from sigma^2 * rho_mat)

#### H. REML Interface

- `map_theta_to_v` and `map_theta_to_dv` work correctly

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Test with small n (2, 3) for easy manual verification
- Use known free_param values for deterministic ρ values
- Compare `manual_grad` vs `auto_grad`
- Verify structure: unit diagonal, uniform off-diagonal

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
from math import log
from torch_openreml.covariance import EquicorrelationMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformChain,
    TransformScaleShift,
    TransformSigmoid,
)


class TestEquicorrelationMatrix:
    """Tests for the equicorrelation matrix."""

    def test_constructor(self):
        mat = EquicorrelationMatrix(3)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = EquicorrelationMatrix(3)
        assert mat.shape == (3, 3)

    def test_rho_min(self):
        mat = EquicorrelationMatrix(3)
        assert mat.rho_min == -0.5

    def test_rho_min_n2(self):
        mat = EquicorrelationMatrix(2)
        assert mat.rho_min == -1.0

    def test_default_params(self):
        mat = EquicorrelationMatrix(3)
        assert mat.num_params == 1
        assert mat.num_free_params == 1
        assert mat.num_fixed_params == 0

    def test_param_names(self):
        mat = EquicorrelationMatrix(3)
        assert mat.param_names == ["rho"]
        assert mat.free_param_names == ["rho"]

    def test_call_diagonal_is_one(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0))

    def test_call_off_diagonal_uniform(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        result = mat(free_params)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == off_diag[0]).all()

    def test_call_rho_zero_is_identity(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.0])
        result = mat(free_params)
        # sigmoid(0) = 0.5, scale: 0.5*(1-rho_min)+rho_min = 0.5*1.5-0.5 = 0.25
        # Actually rho = 0.25 at free_param=0, not 0.
        # To get rho=0: sigmoid(x) * (1-rho_min) + rho_min = 0 → sigmoid(x) = -rho_min/(1-rho_min)
        # For n=3: sigmoid(x) = 0.5/1.5 = 1/3 → x = logit(1/3) = log(0.5) ≈ -0.693
        x = log(1/3 / (2/3))
        free_params_zero = torch.tensor([x])
        result = mat(free_params_zero)
        assert torch.allclose(result, torch.eye(3))

    def test_call_rho_in_bounds_negative(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([-10.0])
        result = mat(free_params)
        n = 3
        off_diag = result[0, 1]
        assert off_diag > mat.rho_min

    def test_call_rho_in_bounds_positive(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([10.0])
        result = mat(free_params)
        off_diag = result[0, 1]
        assert off_diag < 1.0

    def test_call_symmetric(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_n1(self):
        mat = EquicorrelationMatrix(1)
        free_params = torch.tensor([0.0])
        result = mat(free_params)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0

    def test_manual_grad_shape(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["rho"]

    def test_manual_grad_diagonal_zero(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        grad, _ = mat.manual_grad(free_params)
        assert (grad[0].diag() == 0.0).all()

    def test_manual_grad_off_diagonal_uniform(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        grad, _ = mat.manual_grad(free_params)
        n = 3
        off_diag = grad[0][~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == off_diag[0]).all()

    def test_manual_grad_vs_auto_grad(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.2])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_grad_dispatches_to_manual(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.2])
        grad_default, _ = mat.grad(free_params)
        grad_manual, _ = mat.manual_grad(free_params)
        assert torch.allclose(grad_default, grad_manual)

    def test_fixed_rho(self):
        mat = EquicorrelationMatrix(3, param_specs={
            "rho": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(1.5, -0.5)]),
            }
        })
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 1
        assert mat.free_param_names == []

    def test_fixed_rho_grad(self):
        mat = EquicorrelationMatrix(3, param_specs={
            "rho": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(1.5, -0.5)]),
            }
        })
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_fixed_rho_call(self):
        mat = EquicorrelationMatrix(3, param_specs={
            "rho": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(1.5, -0.5)]),
            }
        })
        result = mat(torch.tensor([]))
        assert torch.allclose(result.diag(), torch.tensor(1.0))
        # rho = 0.25 (sigmoid(0)*1.5 - 0.5 = 0.75 - 0.5)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        expected_off = 0.25
        assert torch.allclose(off_diag, torch.tensor(expected_off))

    def test_intermediate_cache_hit(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        built = mat.build_params(free_params)
        assert mat.get_intermediates(built) is None
        mat(free_params)
        cache = mat.get_intermediates(built)
        assert cache is not None
        assert "v" in cache
        assert "i_n" in cache

    def test_intermediate_cache_reset(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        built = mat.build_params(free_params)
        mat(free_params)
        assert mat.get_intermediates(built) is not None
        mat.reset_intermediates()
        assert mat.get_intermediates(built) is None

    def test_dtype_float64(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_map_theta_to_v(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        mat = EquicorrelationMatrix(3)
        r = repr(mat)
        assert "EquicorrelationMatrix" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **No variance parameter**: Unlike `CompoundSymmetricMatrix`, this matrix has unit diagonal. It represents a pure correlation structure. The only parameter is ρ.
- **Rho bounds**: Same transform chain as `CompoundSymmetricMatrix.rho` — Sigmoid → ScaleShift(1-rho_min, rho_min). The ρ=0 (identity) test requires computing the correct `logit` value.
- **Gradient simplicity**: `dV/dρ = (J - I) * trans_grad`, which is constant-1 on off-diagonal and 0 on diagonal (before trans_grad scaling). This is the simplest possible gradient structure for a correlation parameter.
- **V in cache**: Unlike `CompoundSymmetricMatrix` which caches intermediate building blocks and assembles V in `__call__`, `EquicorrelationMatrix` caches the full V directly. This means `__call__` is just a cache lookup if the params haven't changed.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 5 |
| Param names | 1 |
| `__call__` correctness & structure | 7 |
| `manual_grad` & `grad` | 5 |
| Fixed rho config | 3 |
| Intermediate caching | 2 |
| dtype | 1 |
| REML interface | 2 |
| repr | 1 |
| **Total** | **27** |
