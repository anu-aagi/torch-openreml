# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_hadamard_product`
**Purpose of Module:**
Provides a `HadamardProduct` operator: `V = A ⊙ B` (element-wise product). Both operands must have the same shape. Commonly used to apply a scalar variance factor to a correlation matrix (e.g., `σ² * R` using a fixed scalar tensor and an `EquicorrelationMatrix`).

**Classes Covered:**

- `HadamardProduct` (extends `Operator`) — element-wise product `V = A ⊙ B`

**Testing Goal:**
Ensure correct element-wise product, correct gradient via `dA ⊙ B` and `A ⊙ dB`, and correct broadcasting of 3D gradient tensors with 2D operand matrices.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires exactly 2 operands → `ValueError`
- Both operands must have the same shape (enforced by runtime, not init)
- Instance is a subclass of `Matrix` (via `Operator`)

#### B. `__call__(free_params)` Behavior

- Returns `a * b` (element-wise)
- Same shape as operands
- Fixed Tensor operands included as-is
- Example: EquicorrelationMatrix * `torch.tensor([5.0])` scales the correlation matrix by 5
- Intermediate caching works

#### C. Manual Gradient — A Parameters

- `dV/dθ_A = dA/dθ ⊙ B`
- `da * b` where da is 3D `(k, n, n)` and b is 2D `(n, n)`
- Broadcasting over batch dimension

#### D. Manual Gradient — B Parameters

- `dV/dθ_B = A ⊙ dB/dθ`
- `a * db` where a is 2D, db is 3D

#### E. Common Configurations

- A = EquicorrelationMatrix (trainable), B = fixed scalar tensor (common LMM usage)
- A = ScalarMatrix, B = ScalarMatrix (both trainable)
- Only A trainable, only B trainable
- Both fixed

#### F. REML Interface

- `map_theta_to_v` returns element-wise product
- `map_theta_to_dv` returns concatenated gradient

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Test with simple matrices for easy manual verification
- Compare `manual_grad` vs `auto_grad`
- Use fixed scalar tensor as second operand (real-world pattern)
- Verify element-wise nature

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance import HadamardProduct, ScalarMatrix, IdentityMatrix, EquicorrelationMatrix
from torch_openreml.covariance.matrix import Matrix


class TestHadamardProduct:
    """Tests for the HadamardProduct operator."""

    def test_constructor(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            HadamardProduct(ScalarMatrix(3))

    def test_call_element_wise(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        result = op(free_params)
        # A = e^0 * I = I, B = e^1 * I = e*I
        # A ⊙ B = e * I (element-wise)
        import math
        expected = math.exp(1) * torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_with_scalar_tensor(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=torch.tensor([5.0]))
        free_params = torch.tensor([0.0])
        result = op(free_params)
        expected = 5.0 * torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_shape(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        assert op(free_params).shape == (3, 3)

    def test_param_namespacing(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert op.num_free_params == 2
        assert sorted(op.free_param_names) == ["a/sigma^2", "b/sigma^2"]

    def test_param_namespacing_one_trainable(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op.num_free_params == 1
        assert op.free_param_names == ["a/sigma^2"]

    def test_manual_grad_shape(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 3, 3)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_a_equals_da_times_b(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        b = torch.eye(3)
        expected = grad_a * b  # element-wise
        assert torch.allclose(grad_op, expected)

    def test_manual_grad_with_scalar_tensor(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=torch.tensor([5.0]))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        assert grad_op.shape == (1, 3, 3)
        # dV/dθ = dA/dθ * 5.0
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, grad_a * 5.0)

    def test_manual_grad_vs_auto_grad(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = HadamardProduct(a=IdentityMatrix(3), b=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "b" in cache

    def test_map_theta_to_v(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        r = repr(op)
        assert "HadamardProduct" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Element-wise nature**: Unlike all other operators, this one uses `*` (element-wise), not `@` (matrix multiply) or `torch.kron`. The gradient `dA ⊙ B` uses broadcasting: `da` is `(k, n, n)`, `b` is `(n, n)`, result is `(k, n, n)` via standard PyTorch broadcasting.
- **Common usage**: The example in the docstring shows `EquicorrelationMatrix(4)` * `torch.tensor([5.0])` — this is `5 * R`, turning a correlation matrix into a covariance matrix. This is a key pattern for the Hadamard product.
- **Shape constraint**: Both operands must have the same shape. Unlike `BlockDiagonal` which handles different sizes, or `Sum` which requires same shape, Hadamard product is inherently element-wise.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor | 2 |
| `__call__` | 3 |
| Parameter namespacing | 2 |
| `manual_grad` shape & structure | 3 |
| `manual_grad` vs `auto_grad` | 1 |
| All-fixed | 1 |
| Intermediate caching | 1 |
| REML interface | 2 |
| repr | 1 |
| **Total** | **16** |
