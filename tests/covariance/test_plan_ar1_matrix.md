# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.ar1_matrix`
**Purpose of Module:**
Provides a first-order autoregressive covariance matrix `V_{ij} = σ² ρ^{|i-j|}`. Covariance decays geometrically with the lag between observation indices, making it suitable for time-series or spatial data with decaying correlation over distance.

**Classes Covered:**

- `AR1Matrix` — AR(1) covariance `V_{ij} = σ² ρ^{|i-j|}`

**Testing Goal:**
Ensure correct construction of the geometric decay structure, correct manual gradient (including the rho-clamping safety measure), and correct intermediate caching. This matrix shares the 2-param pattern with `CompoundSymmetricMatrix` but has a distance-dependent off-diagonal structure.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires `n`, optional `param_specs`
- Default param_specs: two free params — `sigma^2` (TransformExpPow2) and `rho` (Sigmoid → ScaleShift(2, -1))
- Instance is a subclass of `Matrix`
- Shape is `(n, n)`
- `num_params` = 2, `num_free_params` = 2 (default)
- `n` also accepts a list, a tuple or a `pandas.Series` of levels: the dimension is the number of distinct elements (`n_distinct`), so `AR1Matrix(["a", "b", "a"])` is `AR1Matrix(2)`. An empty sequence resolves to `0` and is rejected by the shape check. The resolved matrix matches the integer form in shape, parameter names and `__call__` values

#### B. `__call__(free_params)` Behavior

- Returns σ² ρ^{|i-j|}
- Diagonal entries = σ² (always)
- Off-diagonal entries decay with distance: σ²ρ¹, σ²ρ², σ²ρ³, ...
- ρ^{|i-j|} = ρ^{|j-i|} — symmetric
- ρ = 0: all off-diagonals = 0 (diagonal only)
- ρ → 1: all entries approach σ²
- ρ → -1: signs alternate with distance
- Default transforms enforce σ² > 0, ρ ∈ (-1, 1)
- Works for different `n` (2, 4, 5)

#### C. Distance Matrix

- `diff = |i - j|` correctly computed
- diff diagonal = 0, first off-diagonal = 1, second = 2, etc.
- diff is cached in intermediates

#### D. `manual_grad(free_params)` Behavior

- Two gradient slices (if both params free):
  - dV/d(σ²_free) = trans_grad_σ² * ρ^{|i-j|}
  - dV/d(ρ_free) = trans_grad_ρ * σ² * |i-j| * ρ^{|i-j|-1}, with the lag-0 entry zero
- The rho slice is division-free: the exponent is `|i-j| - 1` clamped at 0, so it stays finite at ρ = 0
- At ρ = 0 the rho slice is σ² times the lag-1 adjacency matrix times `trans_grad_ρ`: lag 0 and lag ≥ 2 are zero, lag 1 is `trans_grad_ρ * σ²`
- Agrees with `auto_grad`, including at ρ = 0
- Returns `(None, [])` when both params fixed

#### E. Single Free Parameter Configurations

- Only sigma^2 free (rho fixed): grad has 1 slice = trans_grad_σ² * ρ^{|i-j|}
- Only rho free (sigma^2 fixed): grad has 1 slice = dV/dρ
- Correct param names

#### F. Intermediate Caching

- Caches sigma2, rho, diff, rho_power
- Cache hit after first `__call__`
- Cache invalidated by `reset_intermediates`

#### G. Edge Cases

- rho ≈ 0 (rho_free → -inf): check diagonal only, off-diagonals ≈ 0
- rho = 0: `__call__` is diagonal, and the rho gradient is finite with the lag-1 adjacency pattern
- rho ≈ 1 (rho_free → +inf): check all entries ≈ σ²
- n = 1 (trivial 1×1 case)

#### H. REML Interface

- `map_theta_to_v` and `map_theta_to_dv` work correctly

#### I. `inverse(free_params)`

`AR1Matrix` overrides the base default with the tridiagonal closed form: diagonal entries `1 + ρ²` (the two corner ones staying `1`), off-diagonal entries `-ρ`, all scaled by `1 / (σ²(1 - ρ²))`. The inverse of an AR(1) covariance is sparse, so forming `V` and inverting it numerically is pure overhead. The override must be indistinguishable from the default it replaces, and it takes the cache path plus its own `n = 1` branch.

- Agrees with `torch.linalg.inv(mat(free_params))`, and its entries match the closed form: corner diagonal entries are `1`, interior ones are `1 + ρ²`, the two nearest off-diagonals are `-ρ`, and everything further out is exactly zero (the tridiagonal structure is the point of the override)
- The result is symmetric, and `mat(free_params) @ mat.inverse(free_params)` is the identity
- Negative ρ, where `-ρ` flips sign: the off-diagonals become positive and the result still matches `torch.linalg.inv`
- `n = 1`, which the override special-cases before the tridiagonal construction: the result is `I / σ²`
- `n = 2`, where no entry satisfies "interior" and both diagonal entries are corner entries: the result is the `1 / (σ²(1 - ρ²))` limit, not the general formula
- The argument contract is the one `__call__` already promises, and it is not bypassed by the override: `None` means the defaults, a dict of free parameters means the same as the flat tensor, and a non-tensor, a wrong-length vector and a bad dict each raise from `build_params`
- Both single-free-parameter configurations (only `sigma^2` free, only `rho` free) invert correctly, since the closed form reads both parameters regardless of which one is trainable
- dtype follows the input, and interleaving dtypes across calls does not let a stale cache entry leak through — the cache is keyed on dtype, so a cached float64 `sigma2`/`rho` must not reach a float32 inverse
- The cache path is exercised in both directions: `inverse` after `__call__` reuses the entry without replacing it, and `inverse` alone populates it
- The output stays attached to `free_params`, and its gradient equals the gradient of `torch.linalg.inv(mat(free_params))` — the closed form must be the derivative of the same function, not merely equal to it in value

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Test with small n (3, 4) for easy manual verification
- Compare `manual_grad` vs `auto_grad`
- Test specific ρ values (0, 0.5, -0.5) for structure verification
- Test the geometric decay property
- Test single-param and both-fixed configs

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
from torch_openreml.covariance import AR1Matrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExpPow2,
    TransformChain,
    TransformScaleShift,
    TransformSigmoid,
)


class TestAR1Matrix:
    """Tests for the AR(1) covariance matrix."""

    def test_constructor(self):
        mat = AR1Matrix(4)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = AR1Matrix(4)
        assert mat.shape == (4, 4)

    def test_default_params(self):
        mat = AR1Matrix(4)
        assert mat.num_params == 2
        assert mat.num_free_params == 2
        assert mat.num_fixed_params == 0
        assert mat.param_names == ["sigma^2", "rho"]
        assert mat.free_param_names == ["sigma^2", "rho"]

    def test_call_diagonal(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 0.0])
        # sigma^2 = e^0 = 1, rho = sigmoid(0)*2 - 1 = 0.5*2 - 1 = 0
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0))

    def test_call_rho_zero_is_diagonal(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 0.0])
        result = mat(free_params)
        n = 4
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == 0.0).all()

    def test_call_geometric_decay(self):
        mat = AR1Matrix(4)
        # free_params chosen so rho = 0.5 (use known transform)
        # sigmoid(x) = 0.75 → x = logit(0.75) ≈ 1.099
        # rho = 0.75 * 2 - 1 = 0.5
        from math import log
        x = log(0.75 / 0.25)  # logit(0.75)
        free_params = torch.tensor([0.0, x])
        result = mat(free_params)
        sigma2 = torch.exp(torch.tensor(0.0))  # = 1
        # V[0,1] = rho^1, V[0,2] = rho^2, V[0,3] = rho^3
        rho = 0.5
        for i in range(4):
            for j in range(4):
                expected = sigma2 * (rho ** abs(i - j))
                assert torch.allclose(result[i, j], expected)

    def test_call_symmetric(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.5, 1.0])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_rho_positive_decay(self):
        mat = AR1Matrix(4)
        from math import log
        x = log(0.75 / 0.25)
        free_params = torch.tensor([0.0, x])
        result = mat(free_params)
        # Off-diagonal entries should decrease with distance
        assert result[0, 1] > result[0, 2]
        assert result[0, 2] > result[0, 3]

    def test_call_n1(self):
        mat = AR1Matrix(1)
        free_params = torch.tensor([0.0, 0.0])
        result = mat(free_params)
        assert result.shape == (1, 1)
        assert result[0, 0] > 0  # sigma^2 > 0

    def test_diff_matrix(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 0.0])
        mat(free_params)
        built = mat.build_params(free_params)
        diff = mat.get_intermediates(built)["diff"]
        for i in range(4):
            for j in range(4):
                assert diff[i, j] == abs(i - j)

    def test_manual_grad_shape(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (2, 4, 4)
        assert grad_names == ["sigma^2", "rho"]

    def test_manual_grad_sigma_structure(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        grad, _ = mat.manual_grad(free_params)
        grad_sigma = grad[0]
        # dV/d(sigma^2) should have same geometric decay pattern
        result = mat(free_params)
        assert torch.allclose(grad_sigma, result * 2.0)  # trans_grad = 2 at x=0

    def test_manual_grad_rho_diagonal_zero(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        grad, _ = mat.manual_grad(free_params)
        grad_rho = grad[1]
        assert (grad_rho.diag() == 0.0).all()

    def test_manual_grad_vs_auto_grad(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_only_sigma_free(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["sigma^2"]

    def test_only_sigma_free_grad(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        free_params = torch.tensor([0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 4, 4)
        assert grad_names == ["sigma^2"]

    def test_only_rho_free(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["rho"]

    def test_only_rho_free_grad(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        free_params = torch.tensor([1.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 4, 4)
        assert grad_names == ["rho"]

    def test_both_fixed(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        built = mat.build_params(free_params)
        assert mat.get_intermediates(built) is None
        mat(free_params)
        cache = mat.get_intermediates(built)
        assert cache is not None
        assert "rho_power" in cache
        assert "diff" in cache

    def test_intermediate_cache_reset(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        built = mat.build_params(free_params)
        mat(free_params)
        assert mat.get_intermediates(built) is not None
        mat.reset_intermediates()
        assert mat.get_intermediates(built) is None

    def test_dtype_float64(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_inverse_closed_form_structure(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        sigma2, rho = mat.build_params(free_params)
        d = 1 / (sigma2 * (1 - rho ** 2))
        expected_diag = torch.stack([d, d * (1 + rho ** 2), d * (1 + rho ** 2), d]).squeeze(-1)
        result = mat.inverse(free_params)
        assert torch.allclose(result.diag(), expected_diag)
        assert torch.allclose(result[0, 1], -rho * d)
        assert torch.allclose(result[3, 2], -rho * d)
        assert result[0, 2] == 0.0
        assert result[0, 3] == 0.0

    def test_inverse_n1(self):
        mat = AR1Matrix(1)
        free_params = torch.tensor([0.4, 1.0])
        sigma2 = mat.build_params(free_params)[0]
        assert torch.allclose(mat.inverse(free_params), torch.eye(1) / sigma2)
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)))

    def test_inverse_n2_both_diagonal_entries_are_corners(self):
        mat = AR1Matrix(2)
        free_params = torch.tensor([0.4, 1.0])
        sigma2, rho = mat.build_params(free_params)
        result = mat.inverse(free_params)
        assert torch.allclose(result.diag(), (1 / (sigma2 * (1 - rho ** 2))).expand(2))
        assert torch.allclose(result[0, 1], -rho / (sigma2 * (1 - rho ** 2)))

    def test_inverse_follows_dtype_across_interleaved_calls(self):
        mat = AR1Matrix(4)
        mat(torch.tensor([0.4, 1.0], dtype=torch.float64))
        assert mat.inverse(torch.tensor([0.4, 1.0], dtype=torch.float32)).dtype == torch.float32

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        mat.reset_intermediates()
        torch.linalg.inv(mat(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-5)

    def test_map_theta_to_v(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        mat = AR1Matrix(4)
        r = repr(mat)
        assert "AR1Matrix" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Rho gradient at ρ = 0**: the derivative is `|i-j| * ρ^(|i-j|-1)`, computed without dividing by ρ, with the exponent clamped at 0. At ρ = 0 the diagonal and lags ≥ 2 are zero and the first off-diagonals equal `trans_grad_ρ * σ²`; the gradient must contain no NaN or inf.
- **Geometric decay**: Unlike `CompoundSymmetricMatrix` where off-diagonals are all equal, AR(1) has distance-dependent off-diagonals. The gradient for rho preserves this structure, scaled by |i-j|.
- **No diagonal patch**: `fill_diagonal_` is unnecessary because the lag-0 term already carries the factor `|i-j| = 0`; the gradient stays finite at ρ = 0 without a post-hoc overwrite.
- **`diff` matrix**: The distance matrix `|i - j|` should be Toeplitz with constant diagonals. Verified in the test.
- **`inverse` reads the cache and branches on `n`**: the closed form is assembled from the cached `sigma2` and `rho`, so it inherits the cache's dtype and device guards — a cache keyed on values alone would return a float64 `sigma2` inside a float32 inverse, which `test_inverse_follows_dtype_across_interleaved_calls` pins. The `n == 1` branch exists because the tridiagonal construction would otherwise leave a lone entry classified as a corner, which is right by accident there; the `n = 1` and `n = 2` tests keep that accident honest for the sizes above it too.
- **The gradient tests reset the cache between the two backward passes**: `Matrix`'s intermediate cache hands back the tensor objects from the last call with matching values, so a second autograd graph built on the same values reuses tensors whose graph has already been freed. The test calls `reset_intermediates` before differentiating `torch.linalg.inv(mat(free_params))`, which sidesteps the reuse rather than papering over it; the aliasing is a cache property, not an `inverse` property.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 6 |
| `__call__` correctness & properties | 6 |
| Diff matrix | 1 |
| `manual_grad` structure & correctness | 3 |
| `manual_grad` vs `auto_grad` | 2 |
| Single free param configs | 4 |
| Both fixed | 1 |
| Intermediate caching | 2 |
| dtype | 1 |
| REML interface | 2 |
| repr | 1 |
| `inverse` | 18 |
| **Total** | **47** |
