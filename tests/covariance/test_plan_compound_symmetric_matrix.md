# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.compound_symmetric_matrix`
**Purpose of Module:**
Provides a compound symmetric covariance matrix `V = σ²[(1-ρ)I_n + ρ J_n]` with shared variance `σ²` and correlation `ρ`. All diagonal entries equal `σ²`, all off-diagonal entries equal `σ²ρ`. Used for exchangeable correlation structures (e.g., repeated measures with constant within-group correlation).

**Classes Covered:**

- `CompoundSymmetricMatrix` — compound symmetric `V = σ²[(1-ρ)I_n + ρ J_n]`

**Testing Goal:**
Ensure correct matrix construction, correlation range enforcement via transform chain, correct manual gradient structure, and correct intermediate caching. This is the first Matrix with (a) a chained transform and (b) explicit parameter validation via `rho_min`.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires `n`, optional `param_specs`
- `rho_min = -1/(n-1)` computed correctly for different n
- Default param_specs: two free params — `sigma^2` (TransformExpPow2) and `rho` (Sigmoid → ScaleShift)
- Instance is a subclass of `Matrix`
- Shape is `(n, n)`
- `num_params` = 2, `num_free_params` = 2 (default)
- `n` also accepts a list, a tuple or a `pandas.Series` of levels: the dimension is the number of distinct elements (`n_distinct`), so `CompoundSymmetricMatrix(["a", "b", "a"])` is `CompoundSymmetricMatrix(2)`. `rho_min` follows the resolved dimension, and the result matches the integer form in shape, parameter names and `__call__` values. An empty sequence resolves to `0` and is rejected by the `n <= 1` check
- Custom param_specs accepted

#### B. `__call__(free_params)` Behavior

- Returns σ²[(1-ρ)I + ρJ]
- All diagonal entries equal σ²
- All off-diagonal entries equal σ²ρ
- Matrix is symmetric
- With default transforms: σ² > 0, ρ ∈ (rho_min, 1)
- Works for different n values (2, 3, 5)
- For ρ ≈ 0: nearly diagonal
- For ρ = 0 (free_param_rho → -inf): off-diagonals ≈ 0
- Intermediate caching correctly reuses I_n, J_n, rho_mat

#### C. Transform Chain for ρ

- Chain: `Sigmoid → ScaleShift(a=1-rho_min, b=rho_min)`
- Maps unconstrained ℝ → (rho_min, 1)
- When sigmoid output = 0: ρ = rho_min
- When sigmoid output = 0.5: ρ = (1 - rho_min)/2 + rho_min
- When sigmoid output = 1: ρ → 1 (from below)
- `trans_grad` correctly chains through both transforms

#### D. `manual_grad(free_params)` Behavior

- Two gradient slices (if both params free):
  - dV/d(sigma^2_free) = trans_grad_sigma² * rho_mat
  - dV/d(rho_free) = σ² * (J - I) * trans_grad_rho
- Shape: `(num_free_params, n, n)`
- dV/d(σ²) has equal diagonal and off-diagonal entries
- dV/d(ρ) has zeros on diagonal, σ² * trans_grad_rho on off-diagonal
- Agrees with `auto_grad`
- Returns `(None, [])` when both params fixed

#### E. Single Free Parameter Configs

- Only `sigma^2` free (rho fixed): grad has 1 slice = trans_grad_sigma² * rho_mat
- Only `rho` free (sigma^2 fixed): grad has 1 slice = σ² * (J - I) * trans_grad_rho
- Param names and indices correct in each case

#### F. Intermediate Caching

- `_get_or_build_intermediates` caches and retrieves correctly
- Cache returns None after `reset_intermediates`
- Cache valid only for same params, dtype, device

#### G. `build_params` Integration

- Merges free + fixed correctly
- Transforms applied correctly
- Correct param ordering (sigma^2 first, rho second)

#### H. REML Interface

- `map_theta_to_v` and `map_theta_to_dv` work correctly

#### I. `inverse(free_params)`

`CompoundSymmetricMatrix` overrides the base default with the Sherman-Morrison closed form `(1 / σ²) [I / (1 - ρ) - ρ J / ((1 - ρ)(1 + (n - 1)ρ))]`. The matrix is a scaled rank-one update of a scaled identity, so the inverse is one too, and inverting the built `V` numerically is pure overhead. The override must be indistinguishable from the default it replaces, and it takes the cache path.

- Agrees with `torch.linalg.inv(mat(free_params))`, and its diagonal and off-diagonal entries each match the closed form (uniform within each block, as in `V`)
- `mat(free_params) @ mat.inverse(free_params)` is the identity
- The `1 / σ²` factor is separable: inverse at a non-zero `sigma^2` equals the inverse at `sigma^2 = 1` divided by `sigma^2`, isolating the variance scaling from the correlation structure
- ρ = 0 gives `I / σ²`, matching the `__call__` test that uses the same `logit` value
- `n = 2`, where `1 + (n - 1)ρ` reduces to `1 + ρ` and the diagonal collapses to `1 / (σ²(1 - ρ²))`: both entries match that limit rather than the general formula
- Near `rho_min` (ρ → -1/(n-1), where `V` is singular and the denominator tends to 0), the closed form still matches `torch.linalg.inv` — the cancellation this risks is what the boundary test is for
- The argument contract is the one `__call__` already promises, and it is not bypassed by the override: `None` means the defaults, a dict of free parameters means the same as the flat tensor, and a non-tensor, a wrong-length vector and a bad dict each raise from `build_params`
- Both single-free-parameter configurations (only `sigma^2` free, only `rho` free) invert correctly, since the closed form reads both parameters regardless of which one is trainable
- dtype follows the input, and interleaving dtypes across calls does not let a stale cache entry leak through — the cache is keyed on dtype, so a cached float64 `sigma2`/`i_n`/`j_n` must not reach a float32 inverse
- The cache path is exercised in both directions: `inverse` after `__call__` reuses the entry without replacing it, and `inverse` alone populates it
- The output stays attached to `free_params`, and its gradient equals the gradient of `torch.linalg.inv(mat(free_params))` — the closed form must be the derivative of the same function, not merely equal to it in value

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Test with small n (2, 3) for easy manual verification
- Use free_params that map to known σ² and ρ values for verification
- Compare `manual_grad` vs `auto_grad`
- Test all fixed/free combinations (both free, only sigma free, only rho free, both fixed)
- Verify cache behavior

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance import CompoundSymmetricMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExpPow2,
    TransformChain,
    TransformScaleShift,
    TransformSigmoid,
)


class TestCompoundSymmetricMatrix:
    """Tests for the compound symmetric covariance matrix."""

    def test_constructor(self):
        mat = CompoundSymmetricMatrix(3)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = CompoundSymmetricMatrix(3)
        assert mat.shape == (3, 3)

    def test_rho_min(self):
        mat = CompoundSymmetricMatrix(3)
        assert mat.rho_min == -0.5  # -1/(3-1)

    def test_rho_min_n2(self):
        mat = CompoundSymmetricMatrix(2)
        assert mat.rho_min == -1.0

    def test_default_params(self):
        mat = CompoundSymmetricMatrix(3)
        assert mat.num_params == 2
        assert mat.num_free_params == 2
        assert mat.num_fixed_params == 0
        assert mat.param_names == ["sigma^2", "rho"]
        assert mat.free_param_names == ["sigma^2", "rho"]

    def test_call_diagonal(self):
        mat = CompoundSymmetricMatrix(3)
        # free_params = [0.0, -10.0] → sigma^2 = e^0 = 1, rho ≈ 0 (sigmoid(-10) ≈ 0, then scale-shift)
        free_params = torch.tensor([0.0, -10.0])
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0), atol=1e-4)

    def test_call_off_diagonal(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        # sigma^2 = e^0 = 1, rho = sigmoid(0) = 0.5, scaled: 0.5*(1-rho_min)+rho_min
        # For n=3, rho_min=-0.5: rho = 0.5*1.5 + (-0.5) = 0.75 - 0.5 = 0.25
        result = mat(free_params)
        sigma2 = torch.exp(torch.tensor(0.0))  # TransformExpPow2(0) = 1
        rho_min = -0.5
        rho = torch.sigmoid(torch.tensor(0.0)) * (1 - rho_min) + rho_min
        expected_off = sigma2 * rho
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert torch.allclose(off_diag, expected_off.expand_as(off_diag))

    def test_call_symmetric(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.5, 0.0])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 1.0])
        result = mat(free_params)
        # All diagonal entries equal
        assert result[0, 0] == result[1, 1] == result[2, 2]
        # All off-diagonal entries equal
        assert result[0, 1] == result[0, 2] == result[1, 2]

    def test_rho_in_bounds(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 100.0])
        result = mat(free_params)
        # rho should be within (-0.5, 1)
        sigma2 = result[0, 0]
        off_diag = result[0, 1]
        rho = off_diag / sigma2
        assert rho > mat.rho_min
        assert rho < 1.0

    def test_manual_grad_shape(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (2, 3, 3)
        assert grad_names == ["sigma^2", "rho"]

    def test_manual_grad_sigma_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        grad, _ = mat.manual_grad(free_params)
        # dV/d(sigma^2_free) = trans_grad_sigma * rho_mat
        # all entries should be equal (compound symmetric)
        grad_sigma = grad[0]
        assert torch.allclose(grad_sigma, grad_sigma[0, 0].expand_as(grad_sigma))

    def test_manual_grad_rho_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        grad, _ = mat.manual_grad(free_params)
        # dV/d(rho_free) = sigma^2 * (J - I) * trans_grad_rho
        # diagonal should be zero, off-diagonal equal
        grad_rho = grad[1]
        assert (grad_rho.diag() == 0.0).all()
        n = 3
        off_diag = grad_rho[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == off_diag[0]).all()

    def test_manual_grad_vs_auto_grad(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_only_sigma_free(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["sigma^2"]

    def test_only_sigma_free_grad(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        free_params = torch.tensor([0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["sigma^2"]

    def test_only_rho_free(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["rho"]

    def test_only_rho_free_grad(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        free_params = torch.tensor([0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["rho"]

    def test_both_fixed(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        built = mat.build_params(free_params)
        assert mat.get_intermediates(built) is None  # nothing cached yet
        mat(free_params)  # triggers cache write
        cache = mat.get_intermediates(built)
        assert cache is not None
        assert "sigma2" in cache
        assert "rho_mat" in cache

    def test_intermediate_cache_reset(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        built = mat.build_params(free_params)
        mat(free_params)
        assert mat.get_intermediates(built) is not None
        mat.reset_intermediates()
        assert mat.get_intermediates(built) is None

    def test_dtype_float64(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_inverse_closed_form_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7])
        sigma2, rho = mat.build_params(free_params)
        expected_diag = (1 / sigma2) * (1 / (1 - rho) - rho / ((1 - rho) * (1 + 2 * rho)))
        expected_off = -(1 / sigma2) * rho / ((1 - rho) * (1 + 2 * rho))
        result = mat.inverse(free_params)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert torch.allclose(result.diag(), expected_diag.expand(n))
        assert torch.allclose(off_diag, expected_off.expand_as(off_diag))

    def test_inverse_scales_by_reciprocal_sigma_squared(self):
        mat = CompoundSymmetricMatrix(3)
        base = mat.inverse(torch.tensor([0.0, 0.7]))
        sigma2 = mat.build_params(torch.tensor([1.0, 0.7]))[0]
        assert torch.allclose(mat.inverse(torch.tensor([1.0, 0.7])), base / sigma2)

    def test_inverse_near_rho_min(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, -6.0])
        rho = mat.build_params(free_params)[1]
        assert rho > mat.rho_min
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)), atol=1e-3)

    def test_inverse_follows_dtype_across_interleaved_calls(self):
        mat = CompoundSymmetricMatrix(3)
        mat(torch.tensor([0.4, 0.7], dtype=torch.float64))
        assert mat.inverse(torch.tensor([0.4, 0.7], dtype=torch.float32)).dtype == torch.float32

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        mat.reset_intermediates()
        torch.linalg.inv(mat(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_map_theta_to_v(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        mat = CompoundSymmetricMatrix(3)
        r = repr(mat)
        assert "CompoundSymmetricMatrix" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Correlation range**: The `rho_min = -1/(n-1)` bound ensures positive definiteness. For n=2, rho_min = -1; for n=3, rho_min = -0.5; for large n, rho_min → 0 from below. Tests verify this with different n.
- **Transform chain for ρ**: The default transform is `Sigmoid → ScaleShift(a=1-rho_min, b=rho_min)`. The `trans_grad` must correctly chain through both. The manual grad formula `sigma^2 * (J - I) * trans_grad_rho` depends on correct chain-rule evaluation.
- **Intermediate caching**: `_get_or_build_intermediates` caches I_n, J_n, and rho_mat. The cache key includes param hash, dtype, and device. Tests verify cache hit/miss behavior.
- **Single free parameter**: The `manual_grad` method handles partial freedom by checking `0 in free_param_index` and `1 in free_param_index` individually. The grad stack order corresponds to `self.free_param_names`.
- **Compound symmetry structure**: All diagonal entries equal, all off-diagonal entries equal — this is a stronger condition than just symmetry and should be verified.
- **`inverse` reads the cache**: the closed form is assembled from the cached `sigma2`, `i_n` and `j_n`, so it inherits the cache's dtype and device guards — a cache keyed on values alone would return a float64 `i_n` inside a float32 inverse, which `test_inverse_follows_dtype_across_interleaved_calls` pins. It reads `rho_mat`'s inputs rather than `rho_mat` itself, so the two forms of the same matrix stay derived from one cached set of blocks.
- **The closed form loses precision as `ρ → rho_min`**: `1 + (n - 1)ρ` is a difference of nearly equal terms there, and `V` becomes singular exactly at `rho_min`. The transform maps an unconstrained θ into the open interval, so `rho_min` is approached but never reached; the boundary test pins a tolerance (`atol=1e-3` at θ = -6) rather than exactness, since the float32 error there is genuinely larger than elsewhere.
- **The gradient test resets the cache between the two backward passes**: `Matrix`'s intermediate cache hands back the tensor objects from the last call with matching values, so a second autograd graph built on the same values reuses tensors whose graph has already been freed. The test calls `reset_intermediates` before differentiating `torch.linalg.inv(mat(free_params))`, which sidesteps the reuse rather than papering over it; the aliasing is a cache property, not an `inverse` property.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 8 |
| `__call__` correctness & structure | 4 |
| Rho bounds & transforms | 1 |
| `manual_grad` structure | 3 |
| `manual_grad` vs `auto_grad` | 1 |
| Single free param configs | 4 |
| Both fixed | 1 |
| Intermediate caching | 2 |
| dtype | 1 |
| REML interface | 2 |
| repr | 1 |
| `inverse` | 18 |
| **Total** | **46** |
