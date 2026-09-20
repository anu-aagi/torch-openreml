# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.transform.transform_sigmoid`
**Purpose of Module:**
Provides a differentiable bijective sigmoid transform `f(x) = 1 / (1 + e^{-x})` mapping ℝ → (0, 1). Used for constraining parameters to the unit interval (e.g., correlation coefficients).

**Classes Covered:**

- `TransformSigmoid` — logistic sigmoid transform

**Testing Goal:**
Ensure correctness of forward (sigmoid), inverse (logit), and gradient operations. The class is stateless. Note the docstring states domain ℝ₀⁺ but the sigmoid is mathematically defined and useful on all of ℝ.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor initializes without errors (no arguments required)
- Domain is ℝ₀⁺, codomain is (0, 1) (class attributes)
- Instance is a subclass of `Transform` (`isinstance` check)

#### B. Method-Level Behavior

##### `__call__(x)`

- Correct forward output matching `torch.sigmoid`
- Maps 0 → 0.5
- Maps large positive values → ≈1
- Maps large negative values → ≈0
- Output is always strictly between 0 and 1
- Preserves shape of input tensor
- Works across different dtypes: `float32`, `float64`
- Preserves `requires_grad` through the forward pass

##### `inverse(x)`

- Correctly inverts: `t.inverse(t(x)) ≈ x`
- Matches `torch.logit` (logit function)
- For x=0.5, inverse = 0
- Inputs at 0 or 1 produce `-inf`/`+inf` (clamped by `torch.logit`)
- Inputs outside [0, 1] produce `nan`
- Preserves shape of input tensor
- Works across float32 and float64 dtypes

##### `grad(x)`

- Matches the analytic derivative: `sigmoid(x) * (1 - sigmoid(x))`
- grad(0) = 0.25 (maximum of the derivative)
- Agrees with `torch.autograd` numerical gradient of `__call__`
- Symmetric: `grad(x) == grad(-x)`
- Approaches 0 as `|x|` becomes large
- Always positive

#### C. Interaction with TransformChain

- Composable with other transforms (e.g., `TransformChain([TransformSigmoid(), TransformScaleShift(a=2, b=-1)])` to map to (-1, 1))
- Forward/inverse/grad of chain containing `TransformSigmoid` is correct

#### D. Error Handling

- `inverse` with values at domain boundaries (0, 1): verify `±inf` behavior
- `inverse` with values outside (0, 1): verify `nan` behavior
- Non-tensor input to `__call__` — PyTorch `TypeError`

#### E. Performance (if relevant)

- All operations are O(n)
- Suitable for use inside optimization loops

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use `torch.sigmoid` and `torch.logit` as reference implementations
- Roundtrip tests are the primary correctness check
- Compare grad against autograd for validation
- Test extreme inputs to verify bounded output and numerical stability

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformSigmoid,
    TransformScaleShift,
    TransformChain,
)


class TestTransformSigmoid:
    """Tests for the sigmoid transform."""

    def test_constructor(self):
        t = TransformSigmoid()
        assert isinstance(t, Transform)

    def test_domain_codomain(self):
        t = TransformSigmoid()
        assert t.domain == "ℝ₀⁺"
        assert t.codomain == "(0, 1)"

    def test_forward_matches_torch_sigmoid(self):
        t = TransformSigmoid()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.sigmoid(x))

    def test_forward_zero_is_half(self):
        t = TransformSigmoid()
        assert torch.allclose(t(torch.tensor(0.0)), torch.tensor(0.5))

    def test_forward_large_positive_approaches_one(self):
        t = TransformSigmoid()
        result = t(torch.tensor(100.0))
        assert torch.allclose(result, torch.tensor(1.0))

    def test_forward_large_negative_approaches_zero(self):
        t = TransformSigmoid()
        result = t(torch.tensor(-100.0))
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_output_bounded(self):
        t = TransformSigmoid()
        x = torch.randn(100, dtype=torch.float64) * 10
        result = t(x)
        assert (result > 0).all() and (result < 1).all()

    def test_forward_shape_preserved(self):
        t = TransformSigmoid()
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    def test_inverse_roundtrip(self):
        t = TransformSigmoid()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_matches_logit(self):
        t = TransformSigmoid()
        x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        assert torch.allclose(t.inverse(x), torch.logit(x))

    def test_inverse_of_half_is_zero(self):
        t = TransformSigmoid()
        assert torch.allclose(t.inverse(torch.tensor(0.5)), torch.tensor(0.0))

    def test_inverse_at_zero(self):
        t = TransformSigmoid()
        assert torch.isneginf(t.inverse(torch.tensor(0.0)))

    def test_inverse_at_one(self):
        t = TransformSigmoid()
        assert torch.isposinf(t.inverse(torch.tensor(1.0)))

    def test_inverse_outside_domain(self):
        t = TransformSigmoid()
        result = t.inverse(torch.tensor(-0.5))
        assert torch.isnan(result)

    def test_grad_matches_analytic(self):
        t = TransformSigmoid()
        x = torch.randn(10, dtype=torch.float64)
        s = torch.sigmoid(x)
        expected = s * (1 - s)
        assert torch.allclose(t.grad(x), expected)

    def test_grad_at_zero(self):
        t = TransformSigmoid()
        assert torch.allclose(t.grad(torch.tensor(0.0)), torch.tensor(0.25))

    def test_grad_symmetric(self):
        t = TransformSigmoid()
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.grad(x), t.grad(-x))

    def test_grad_positive(self):
        t = TransformSigmoid()
        x = torch.randn(5, dtype=torch.float64)
        assert (t.grad(x) > 0).all()

    def test_grad_approaches_zero_for_large_inputs(self):
        t = TransformSigmoid()
        g = t.grad(torch.tensor(50.0))
        assert torch.allclose(g, torch.tensor(0.0), atol=1e-10)

    def test_grad_matches_autograd(self):
        t = TransformSigmoid()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformSigmoid()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformSigmoid()
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformSigmoid()
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformSigmoid()
        x = torch.randn(5, requires_grad=True).sigmoid()
        assert t.inverse(x).requires_grad


class TestTransformChainWithSigmoid:
    """Integration tests with TransformChain."""

    def test_chain_sigmoid_then_scale_shift_forward(self):
        t = TransformChain([TransformSigmoid(), TransformScaleShift(a=2.0, b=-1.0)])
        x = torch.tensor([0.0], dtype=torch.float64)
        expected = 2.0 * torch.sigmoid(x) - 1.0  # maps to (-1, 1)
        assert torch.allclose(t(x), expected)

    def test_chain_sigmoid_then_scale_shift_inverse(self):
        t = TransformChain([TransformSigmoid(), TransformScaleShift(a=2.0, b=-1.0)])
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_chain_sigmoid_then_scale_shift_grad(self):
        t = TransformChain([TransformSigmoid(), TransformScaleShift(a=2.0, b=-1.0)])
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

- **Domain vs actual behavior**: The docstring states domain = ℝ₀⁺, but sigmoid works correctly on all of ℝ. Tests use negative inputs to verify actual behavior — if the docstring is intentionally restricting domain, this may need adjustment.
- **`torch.logit` clipping**: PyTorch's `logit` clips inputs to [0, 1] internally. Tests at exactly 0 and 1 verify the `±inf` behavior this produces.
- **grad saturation**: For |x| > ~40, sigmoid(x) ≈ 0 or 1, and grad ≈ 0. This is expected but worth being aware of for optimization use.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 2 |
| Forward correctness | 6 |
| Forward shape | 1 |
| Inverse roundtrip & correctness | 3 |
| Inverse edge cases (0, 1, outside) | 3 |
| Gradient analytic & properties | 5 |
| Gradient vs autograd | 1 |
| Dtype preservation | 2 |
| requires_grad preservation | 2 |
| Integration (chain) | 3 |
| **Total** | **28** |
