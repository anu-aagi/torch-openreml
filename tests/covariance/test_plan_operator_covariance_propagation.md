# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_covariance_propagation`
**Purpose of Module:**
Provides a `CovariancePropagation` operator implementing `V = Z G Z^T`, the random-effects contribution to the marginal covariance in LMMs. Z is the random-effect design matrix and G is the random-effect covariance matrix. This operator involves a non-trivial product-rule gradient for Z parameters.

**Classes Covered:**

- `CovariancePropagation` (extends `Operator`) — `V = Z G Z^T`

**Testing Goal:**
Ensure correct propagation of Z through G to produce V, correct gradient computation for both Z and G parameters (including the product rule for Z), and proper handling of fixed vs trainable operands.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires exactly 2 operands → `ValueError` if not
- First operand is Z, second is G
- Supports positional, keyword, and dict specification
- Instance is a subclass of `Matrix` (via `Operator`)

#### B. `__call__(free_params)` Behavior

- Returns `Z @ G @ Z^T`
- Z can be rectangular (design matrix), G must be square
- Result shape is `(n_rows_Z, n_rows_Z)` — always square
- Fixed Tensor Z passed through as-is
- Intermediate caching works

#### C. Manual Gradient — G Parameters

- `dV/dθ_G = Z @ dG/dθ @ Z^T`
- This is linear in G: no product rule needed
- Shape `(num_free_params_G, n, n)` where n = Z rows

#### D. Manual Gradient — Z Parameters

- `dV/dθ_Z = dZ/dθ @ G @ Z^T + Z @ G @ dZ^T/dθ`
- Product rule: Z appears on both sides of G
- Two terms must be computed and summed
- `dZ^T/dθ` = `dZ/dθ.mT` (transpose of last two dims)

#### E. Common Configurations

- Z = fixed DummyMatrix (0 free params), G = DiagonalMatrix (n free params)
- Z = fixed Tensor, G = ScalarMatrix (1 free param)
- Both Z and G trainable
- Both fixed

#### F. Parameter Namespacing

- `"z/sigma^2_0"`, `"g/sigma^2"` etc. depending on operand names
- All Z params listed first, then G params (operand order)

#### G. REML Interface

- `map_theta_to_v` returns Z @ G @ Z^T
- `map_theta_to_dv` returns concatenated gradient

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Most common real-world case: Z = fixed DummyMatrix, G = trainable DiagonalMatrix
- Test simple: Z = fixed I, G = ScalarMatrix → V = σ²I
- Test both trainable: simple Z and G
- Compare `manual_grad` vs `auto_grad` as the primary validation
- Verify product rule correctness for Z gradient

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance import (
    CovariancePropagation, DummyMatrix,
    ScalarMatrix, DiagonalMatrix, IdentityMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestCovariancePropagation:
    """Tests for the CovariancePropagation operator."""

    def test_constructor(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            CovariancePropagation(ScalarMatrix(3))

    def test_param_namespacing(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        assert op.num_free_params == 1  # only g
        assert op.free_param_names == ["g/sigma^2"]

    def test_call_fixed_z_trainable_g(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        result = op(free_params)
        # Z=I, G=σ²I → V = I*(σ²I)*I = σ²I
        expected = torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_dummy_z_trainable_g(self):
        z = DummyMatrix(["a", "b", "a"])
        op = CovariancePropagation(z=z, g=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        result = op(free_params)
        # Manual: Z @ diag(σ²) @ Z^T
        z_mat = z()
        g_mat = torch.diag(torch.exp(2.0 * free_params))
        expected = z_mat @ g_mat @ z_mat.T
        assert torch.allclose(result, expected)

    def test_shape(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = CovariancePropagation(z=z, g=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        result = op(free_params)
        assert result.shape == (4, 4)

    def test_manual_grad_g_only(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["g/sigma^2"]

    def test_manual_grad_g_equals_propagated(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        # dV/dθ = Z @ dG/dθ @ Z^T = I @ dG/dθ @ I = dG/dθ
        g = ScalarMatrix(3)
        grad_g, _ = g.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, grad_g)

    def test_manual_grad_z_and_g_both_trainable(self):
        # Use trainable Z (ScalarMatrix as Z would be unusual but tests the product rule)
        op = CovariancePropagation(z=ScalarMatrix(2), g=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 2, 2)
        assert grad_names == ["z/sigma^2", "g/sigma^2"]

    def test_manual_grad_vs_auto_grad_fixed_z(self):
        z = DummyMatrix(["a", "b", "a"])
        op = CovariancePropagation(z=z, g=DiagonalMatrix(2))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_vs_auto_grad_both_trainable(self):
        op = CovariancePropagation(z=ScalarMatrix(2), g=ScalarMatrix(2))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "z" in cache
        assert "g" in cache

    def test_map_theta_to_v(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        r = repr(op)
        assert "CovariancePropagation" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Product rule for Z**: `V = Z G Z^T` means Z appears twice. The gradient w.r.t. Z has two terms: `dZ @ G @ Z^T + Z @ G @ dZ^T`. The code uses `dz.mT` for the transpose (matrix transpose of last two dims of a 3D gradient). This is the most mathematically complex gradient in the library.
- **G gradient is simpler**: `dV/dθ_G = Z @ dG/dθ @ Z^T` — just sandwich the G gradient between Z and Z^T. No product rule.
- **Real-world usage**: Typically Z is a fixed `DummyMatrix` (0 params) and G is a trainable `DiagonalMatrix` or `ScalarMatrix`. The Z-gradient code path exists for cases where Z itself is parameterized.
- **Shape propagation**: `Z` is `n×p`, `G` is `p×p` (square), `V` is `n×n` (square). The output shape should always be square.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor | 2 |
| Parameter namespacing | 1 |
| `__call__` (fixed Z, Dummy Z) | 3 |
| `manual_grad` G only | 2 |
| `manual_grad` both trainable | 1 |
| `manual_grad` vs `auto_grad` | 2 |
| All-fixed | 1 |
| Intermediate caching | 1 |
| REML interface | 2 |
| repr | 1 |
| **Total** | **16** |
