# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.transform.transform_exppow2`
**Purpose of Module:**
Provides a differentiable bijective transform `f(x) = e^{2x}` mapping ℝ → ℝ₀⁺. This is a steeper variant of `TransformExp` (`e^x`), producing larger output magnitudes for positive inputs. Useful when stronger positive-domain regularization is desired.

**Classes Covered:**

- `TransformExpPow2` — scaled exponential transform `f(x) = e^{2x}`

**Testing Goal:**
Ensure correctness of forward, inverse, and gradient operations. Verify the factor-2 scaling is correctly applied in all three methods. The class is stateless.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor initializes without errors (no arguments required)
- Domain is ℝ, codomain is ℝ₀⁺ (class attributes)
- Instance is a subclass of `Transform` (`isinstance` check)

#### B. Method-Level Behavior

##### `__call__(x)`

- Correct forward output: `e^(2x)`
- At x=0: output is 1
- Equivalent to `TransformExp(2x)` — i.e., `torch.exp(2.0 * x)`
- Preserves shape of input tensor
- Works across different dtypes: `float32`, `float64`
- Preserves `requires_grad` through the forward pass
- Output is always strictly positive (> 0)
- Large positive x → large output
- Large negative x → output approaches 0

##### `inverse(x)`

- Correctly inverts: `t.inverse(t(x)) ≈ x`
- Formula: `log(x) / 2`
- For x=1: inverse = 0
- Non-positive inputs produce `-inf`/`nan`
- Preserves shape, dtype
- Preserves `requires_grad`

##### `grad(x)`

- Matches the analytic derivative: `2 * e^(2x)` = `2 * forward(x)`
- Agrees with `torch.autograd` numerical gradient of `__call__`
- grad(0) = 2
- Always positive
- Correct for scalar and multi-element tensors

#### C. Relationship to TransformExp

- `TransformExpPow2(x) = TransformExp(2*x)`
- This equivalence can be validated as an identity test

#### D. Interaction with TransformChain

- Composable with other transforms
- Forward/inverse/grad of a chain containing `TransformExpPow2` is correct

#### E. Error Handling

- `inverse` with zero or negative input: verify `-inf`/`nan` behavior
- Non-tensor input to `__call__` — PyTorch `TypeError`

#### F. Performance (if relevant)

- All operations are O(n)
- Suitable for use inside optimization loops

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Compare forward against `torch.exp(2*x)` as ground truth
- Compare grad against `2 * torch.exp(2*x)` analytically
- vs-autograd tests for numerical gradient agreement
- Equivalence test: `TransformExpPow2(x)` vs `TransformExp()(2*x)`

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformExpPow2,
    TransformExp,
    TransformChain,
)


class TestTransformExpPow2:
    """Tests for the scaled exponential transform."""

    def test_constructor(self):
        t = TransformExpPow2()
        assert isinstance(t, Transform)

    def test_domain_codomain(self):
        t = TransformExpPow2()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ₀⁺"

    def test_forward_matches_formula(self):
        t = TransformExpPow2()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.exp(2.0 * x))

    def test_forward_zero_is_one(self):
        t = TransformExpPow2()
        assert torch.allclose(t(torch.tensor(0.0)), torch.tensor(1.0))

    def test_forward_positive_is_larger_than_exp(self):
        t = TransformExpPow2()
        x = torch.tensor([1.0, 2.0])
        # e^(2x) > e^x for x > 0
        assert (t(x) > torch.exp(x)).all()

    def test_forward_always_positive(self):
        t = TransformExpPow2()
        x = torch.randn(100, dtype=torch.float64) * 10
        assert (t(x) > 0).all()

    def test_forward_shape_preserved(self):
        t = TransformExpPow2()
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    def test_forward_equivalent_to_exp_of_2x(self):
        t = TransformExpPow2()
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), exp(2.0 * x))

    def test_inverse_roundtrip(self):
        t = TransformExpPow2()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_matches_formula(self):
        t = TransformExpPow2()
        x = torch.tensor([1.0, 4.0, 10.0], dtype=torch.float64)
        assert torch.allclose(t.inverse(x), torch.log(x) / 2.0)

    def test_inverse_of_one_is_zero(self):
        t = TransformExpPow2()
        assert torch.allclose(t.inverse(torch.tensor(1.0)), torch.tensor(0.0))

    def test_inverse_non_positive(self):
        t = TransformExpPow2()
        assert torch.isneginf(t.inverse(torch.tensor(0.0)))
        assert torch.isnan(t.inverse(torch.tensor(-1.0)))

    def test_grad_matches_formula(self):
        t = TransformExpPow2()
        x = torch.randn(10, dtype=torch.float64)
        expected = 2.0 * torch.exp(2.0 * x)
        assert torch.allclose(t.grad(x), expected)

    def test_grad_is_twice_forward(self):
        t = TransformExpPow2()
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), 2.0 * t(x))

    def test_grad_at_zero(self):
        t = TransformExpPow2()
        assert torch.allclose(t.grad(torch.tensor(0.0)), torch.tensor(2.0))

    def test_grad_positive(self):
        t = TransformExpPow2()
        x = torch.randn(5, dtype=torch.float64)
        assert (t.grad(x) > 0).all()

    def test_grad_matches_autograd(self):
        t = TransformExpPow2()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformExpPow2()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformExpPow2()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformExpPow2()
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformExpPow2()
        x = torch.randn(5, requires_grad=True).exp()  # positive values
        assert t.inverse(x).requires_grad


class TestTransformChainWithExpPow2:
    """Integration tests with TransformChain."""

    def test_chain_exppow2_then_identity(self):
        t = TransformChain([TransformExpPow2(), TransformExp()])
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        expected = torch.exp(torch.exp(2.0 * x))
        assert torch.allclose(t(x), expected)

    def test_chain_exppow2_inverse_roundtrip(self):
        t = TransformChain([TransformExpPow2(), TransformExp()])
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_chain_exppow2_grad(self):
        t = TransformChain([TransformExpPow2(), TransformExp()])
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

- **Codomain naming**: The docstring says ℝ₀⁺ (non-negative), but `e^(2x) > 0` for all real x, so the actual codomain is strictly positive ℝ⁺. Tests verify output is always `> 0`.
- **Relationship to TransformExp**: `TransformExpPow2(x) ≡ TransformExp(2*x)`. This equivalence makes it technically redundant (you could chain `TransformScaleShift(a=2)` with `TransformExp`), but it exists as a convenience class. The equivalence test verifies this relationship.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 2 |
| Forward correctness & properties | 6 |
| Forward shape | 1 |
| Inverse roundtrip & correctness | 3 |
| Inverse edge cases | 1 |
| Gradient analytic & properties | 4 |
| Gradient vs autograd | 1 |
| Dtype preservation | 2 |
| requires_grad preservation | 2 |
| Integration (chain) | 3 |
| **Total** | **25** |
