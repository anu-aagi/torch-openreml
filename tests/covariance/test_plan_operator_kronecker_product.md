# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_kronecker_product`
**Purpose of Module:**
Provides a `KroneckerProduct` operator: `V = A ⊗ B`. Used for separable covariance structures (e.g., space × time) where the total covariance factors as the Kronecker product of two component matrices.

**Classes Covered:**

- `KroneckerProduct` (extends `Operator`) — Kronecker product `V = A ⊗ B`

**Testing Goal:**
Ensure correct Kronecker product construction, correct gradient computation via `dA⊗B` and `A⊗dB`, and correct handling of the batched Kronecker product with 3D gradient tensors.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires exactly 2 operands → `ValueError`
- First operand = A, second = B
- Instance is a subclass of `Matrix` (via `Operator`)

#### B. `__call__(free_params)` Behavior

- Returns `torch.kron(A, B)`
- Shape: A(m×m) ⊗ B(n×n) → (mn, mn)
- Matches known Kronecker product values for small matrices
- Fixed Tensor operands included as-is
- Intermediate caching works

#### C. Manual Gradient — A Parameters

- `dV/dθ_A = dA/dθ ⊗ B`
- Batched Kronecker: `torch.kron(da_3d, b_2d)` broadcasts over first dim
- Each slice `da[i]` is kron'd with `b`

#### D. Manual Gradient — B Parameters

- `dV/dθ_B = A ⊗ dB/dθ`
- Batched Kronecker: `torch.kron(a_2d, db_3d)` broadcasts over first dim
- Each slice `db[i]` is kron'd with `a`

#### E. Parameter Namespacing

- Standard `"operand_name/param_name"` format
- A params first, then B params (operand order)

#### F. Common Configurations

- Both A and B trainable
- A trainable, B fixed (Tensor)
- A fixed, B trainable
- Both fixed

#### G. REML Interface

- `map_theta_to_v` returns Kronecker product
- `map_theta_to_dv` returns concatenated gradient

#### H. `inverse(free_params)`

`KroneckerProduct` overrides the base default with the closed form `V⁻¹ = A⁻¹ ⊗ B⁻¹`: each operand is
inverted through its own path and the two results are combined with `torch.kron`, instead of inverting the
`(mp) × (mp)` composite densely.

- Agrees with `torch.linalg.inv(op(free_params))`
- Is exactly `torch.kron(A.inverse(...), B.inverse(...))`, in operand order
- `op.inverse(free_params) @ op(free_params)` is the identity
- The composite is never inverted densely: `torch.linalg.inv` is not called
- A rectangular operand raises (the composite is singular)
- The argument contract is not bypassed: `None` means the defaults, a dict means the same as the flat
  tensor, and a non-tensor, a wrong-length vector and a bad dict each raise
- dtype follows the input, the result stays attached to `free_params`, and its gradient equals the gradient
  of `torch.linalg.inv(op(free_params))`

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use small matrices (2×2, 3×3) for easy manual verification
- Compare `manual_grad` vs `auto_grad`
- Verify Kronecker structure block-by-block for known inputs
- Test batched kron behavior

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance import KroneckerProduct, ScalarMatrix, IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


class TestKroneckerProduct:
    """Tests for the KroneckerProduct operator."""

    def test_constructor(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            KroneckerProduct(ScalarMatrix(3))

    def test_shape(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        result = op(free_params)
        assert result.shape == (6, 6)

    def test_call_identity(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=IdentityMatrix(3))
        free_params = torch.tensor([])
        result = op(free_params)
        assert torch.equal(result, torch.eye(6))

    def test_call_values(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        # A = e^0 * I2 = I2, B = e^1 * I3 = e*I3
        # A⊗B = [1*B, 0*B; 0*B, 1*B] = block diag of B
        import math
        result = op(free_params)
        expected_a = torch.eye(2)
        expected_b = math.exp(1) * torch.eye(3)  # e^(2*0.5) = e^1
        expected = torch.kron(expected_a, expected_b)
        assert torch.allclose(result, expected)

    def test_param_namespacing(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        assert op.num_free_params == 2
        assert sorted(op.free_param_names) == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_shape(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 6, 6)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_a_only(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 6, 6)
        assert grad_names == ["a/sigma^2"]

    def test_manual_grad_b_only(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 6, 6)
        assert grad_names == ["b/sigma^2"]

    def test_manual_grad_a_equals_kron(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        a = ScalarMatrix(2)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        b = torch.eye(3)
        expected = torch.kron(grad_a, b)
        assert torch.allclose(grad_op, expected)

    def test_manual_grad_b_equals_kron(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        a = torch.eye(2)
        b = ScalarMatrix(3)
        grad_b, _ = b.manual_grad(torch.tensor([0.5]))
        expected = torch.kron(a, grad_b)
        assert torch.allclose(grad_op, expected)

    def test_manual_grad_vs_auto_grad(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "b" in cache

    def test_map_theta_to_v(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_inverse_matches_inverse_of_call(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_is_kron_of_operand_inverses(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        expected = torch.kron(ScalarMatrix(2).inverse(free_params[:1]), DiagonalMatrix(2).inverse(free_params[1:]))
        assert torch.allclose(op.inverse(free_params), expected)

    def test_inverse_product_with_call_is_identity(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert torch.allclose(op.inverse(free_params) @ op(free_params), torch.eye(4), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        op.inverse(torch.tensor([0.5, 0.0, 1.0]))
        assert calls == []

    def test_inverse_rectangular_operand_raises(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=torch.ones(2, 3))
        with pytest.raises(RuntimeError):
            op.inverse(torch.tensor([0.5]))

    def test_inverse_uses_default_free_params(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_validation_is_not_bypassed(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.5, 0.0, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        r = repr(op)
        assert "KroneckerProduct" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Batched Kronecker product**: `torch.kron(da, b)` where `da` is 3D `(k, m, m)` and `b` is 2D `(n, n)`. PyTorch's `kron` handles this by treating the first dimension as batch, producing `(k, mn, mn)`. This is the key mechanism for the manual gradient.
- **Gradient simplicity**: Unlike `CovariancePropagation` (where Z appears twice requiring product rule), the Kronecker product gradient is a simple product rule: `d(A⊗B) = dA⊗B + A⊗dB`. Each term involves only one gradient.
- **Kronecker order**: `A ⊗ B` means each element of A is multiplied by the full B matrix. The order matters — `A ⊗ B ≠ B ⊗ A` in general. The first operand is the "outer" matrix.
- **`inverse` keeps the same operand order as the forward pass**: `(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹` does *not* reverse the operands, unlike `Product`, whose inverse does. A copy-pasted reversed implementation would still produce a square matrix of the right shape, so the operand-inverse test compares against `torch.kron(A.inverse(...), B.inverse(...))` built explicitly from the operands rather than from the implementation.
- **The saving is structural, not incidental**: the `(mp) × (mp)` inverse is never formed. The monkeypatched-`torch.linalg.inv` test asserts the dense inverse is not called at all, which a fallback through `__call__` would fail.
- **A rectangular operand makes the composite singular, not just malformed**: `rank(A ⊗ B) = rank(A)·rank(B) ≤ m·q < m·p` when B is `(q, p)` with `q < p`, so there is no square-composite corner case here (unlike `Product`). The override does no shape check of its own, so the failure comes from the operand's own inverse as `RuntimeError: linalg.inv: A must be batches of square matrices...`; the test matches `RuntimeError` only, not the wording.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor | 2 |
| Shape & `__call__` | 3 |
| Parameter namespacing | 1 |
| `manual_grad` shape & structure | 5 |
| `manual_grad` vs `auto_grad` | 1 |
| All-fixed | 1 |
| Intermediate caching | 1 |
| REML interface | 2 |
| repr | 1 |
| `inverse` override | 11 |
| **Total** | **28** |
