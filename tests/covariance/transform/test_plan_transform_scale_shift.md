# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.transform.transform_scale_shift`
**Purpose of Module:**
Provides a differentiable bijective affine transform `f(x) = ax + b` mapping ℝ → ℝ, with configurable scale `a` and shift `b`. Used for scaling and centering unconstrained optimization parameters.

**Classes Covered:**

- `TransformScaleShift` — affine transform `f(x) = ax + b`

**Testing Goal:**
Ensure correctness of forward, inverse, and gradient operations across varied scale and shift values. The class is stateful (stores `a`, `b`) and works for any `a ≠ 0` (bijection requires nonzero scale).

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires `a` (positional or kwarg), `b` defaults to `0.0`
- Constructor stores `a` and `b` as attributes
- Domain is ℝ, codomain is ℝ (class attributes)
- Instance is a subclass of `Transform` (`isinstance` check)
- `__repr__` returns `TransformScaleShift(a={a}, b={b})`

#### B. Method-Level Behavior

##### `__call__(x)`

- Correct forward output: `a*x + b`
- For `b=0`: pure scaling
- For `a=1`: pure shift
- For `a=-1, b=0`: sign flip
- Preserves shape of input tensor
- Works across float32 and float64 dtypes
- Preserves `requires_grad` through the forward pass
- Handles negative `a` correctly

##### `inverse(x)`

- Correctly inverts: `t.inverse(t(x)) ≈ x` for any `a ≠ 0`
- Formula: `(x - b) / a`
- Preserves shape of input tensor
- Works across float32 and float64 dtypes
- `a=0` should raise `ZeroDivisionError` or produce `inf`/`nan`

##### `grad(x)`

- Returns constant `a` regardless of input value
- Output shape should be broadcastable or scalar
- Agrees with `torch.autograd` numerical gradient of `__call__`
- Correct for scalar and multi-element tensors
- Output dtype and device match input

#### C. Interaction with TransformChain

- Composable with other transforms (e.g., `TransformChain([TransformScaleShift(a=2, b=1), TransformExp()])`)
- Forward/inverse/grad of chain containing `TransformScaleShift` is correct

#### D. Error Handling

- `a=0`: division by zero in `inverse` — verify behavior (should raise or produce inf)
- Non-tensor input to `__call__` — PyTorch `TypeError`

#### E. Performance (if relevant)

- All operations are O(n), constant-time for `grad`
- Suitable for use inside optimization loops

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Parametrize over `(a, b)` pairs covering: positive a, negative a, zero b, nonzero b, a=1, fractional a
- Seed inputs with `torch.abs()` where needed to avoid domain issues (but affine has no domain restrictions)
- Compare grad against autograd for validation

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformScaleShift,
    TransformExp,
    TransformChain,
)


class TestTransformScaleShift:
    """Tests for the affine transform."""

    def test_constructor(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        assert isinstance(t, Transform)
        assert t.a == 2.0
        assert t.b == 1.0

    def test_constructor_b_defaults_to_zero(self):
        t = TransformScaleShift(a=3.0)
        assert t.b == 0.0

    def test_repr(self):
        t = TransformScaleShift(a=2.0, b=-1.5)
        assert repr(t) == "TransformScaleShift(a=2.0, b=-1.5)"

    def test_domain_codomain(self):
        t = TransformScaleShift(a=2.0)
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, 3.0),
        (-3.0, -2.0),
    ])
    def test_forward_matches_formula(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), a * x + b)

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, 3.0),
        (-3.0, -2.0),
    ])
    def test_forward_shape_preserved(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 5.0),
        (-1.0, 3.0),
        (0.5, -2.0),
        (-3.0, 0.0),
    ])
    def test_inverse_roundtrip(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_known_values(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        # t(0) = 1, so inverse(1) = 0
        assert torch.allclose(t.inverse(torch.tensor(1.0)), torch.tensor(0.0))
        # t(1) = 3, so inverse(3) = 1
        assert torch.allclose(t.inverse(torch.tensor(3.0)), torch.tensor(1.0))

    def test_grad_constant(self):
        t = TransformScaleShift(a=2.5, b=10.0)
        x = torch.randn(5, dtype=torch.float64)
        expected = torch.full_like(x, 2.5)
        assert torch.allclose(t.grad(x).expand_as(x), expected)

    def test_grad_negative_a(self):
        t = TransformScaleShift(a=-3.0, b=1.0)
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), torch.tensor([-3.0], dtype=x.dtype))

    @pytest.mark.parametrize("a, b", [
        (2.0, 1.0),
        (1.0, 0.0),
        (-1.0, 5.0),
        (0.5, -2.0),
    ])
    def test_grad_matches_autograd(self, a, b):
        t = TransformScaleShift(a=a, b=b)
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical.expand_as(x.grad), x.grad)

    def test_dtype_float32(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformScaleShift(a=2.0, b=1.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformScaleShift(a=2.0)
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformScaleShift(a=2.0)
        x = torch.randn(5, requires_grad=True)
        assert t.inverse(x).requires_grad

    def test_inverse_zero_a(self):
        t = TransformScaleShift(a=0.0, b=1.0)
        x = torch.tensor([1.0, 2.0])
        result = t.inverse(x)
        assert torch.isinf(result).any() or torch.isnan(result).any()


class TestTransformChainWithScaleShift:
    """Integration tests with TransformChain."""

    def test_chain_scale_shift_then_exp_forward(self):
        t = TransformChain([TransformScaleShift(a=2.0, b=1.0), TransformExp()])
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        expected = torch.exp(2.0 * x + 1.0)
        assert torch.allclose(t(x), expected)

    def test_chain_scale_shift_then_exp_inverse(self):
        t = TransformChain([TransformScaleShift(a=2.0, b=1.0), TransformExp()])
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_chain_scale_shift_then_exp_grad(self):
        t = TransformChain([TransformScaleShift(a=2.0, b=1.0), TransformExp()])
        x = torch.randn(3, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **`a=0`**: Division by zero in `inverse`. The test verifies that this produces `inf`/`nan` rather than crashing, but `a=0` is not a valid bijection anyway.
- **`grad` shape**: The current implementation returns `torch.tensor([self.a])` (a 1-element 1D tensor). When used in a chain, this scalar-constant broadcast must compose correctly with element-wise grads from other transforms.
- **`grad` for negative `a`**: Should return the negative constant correctly.

---

## 5. Expected Test Count Summary

| Category | Tests | Parametrized |
|---|---|---|
| Constructor & repr | 3 | — |
| Domain/codomain | 1 | — |
| Forward correctness | 1 | ×5 pairs |
| Forward shape | 1 | ×5 pairs |
| Inverse roundtrip | 1 | ×5 pairs |
| Inverse known values | 1 | — |
| Gradient constant | 2 | — |
| Gradient vs autograd | 1 | ×4 pairs |
| Dtype preservation | 2 | — |
| requires_grad preservation | 2 | — |
| Edge cases (a=0) | 1 | — |
| Integration (chain) | 3 | — |
| **Estimated total** | | **~35** |
