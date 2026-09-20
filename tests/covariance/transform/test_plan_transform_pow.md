# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.transform.transform_pow`
**Purpose of Module:**
Provides a parameterized power transform `f(x) = x^p` mapping ℝ → ℝ, with a configurable exponent `p`. Used for squaring, square-root, or other power-based parameter remappings.

**Classes Covered:**

- `TransformPow` — power transform `f(x) = x^p`

**Testing Goal:**
Ensure correctness of forward, inverse, and gradient operations for default and non-default exponent values. Unlike the Exp family, this class is stateful (stores `factor`), and its inverse/grad behavior depends on the exponent value.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor accepts `factor` kwarg, defaults to `2.0`
- Constructor stores `factor` as an attribute
- Constructor accepts integer and float exponents
- Domain is ℝ, codomain is ℝ (class attributes)
- Instance is a subclass of `Transform` (`isinstance` check)
- `__repr__` returns `TransformPow(factor={factor})` (overrides base class)

#### B. Method-Level Behavior

##### `__call__(x)`

- Correct forward output for scalar (0-d) tensors
- Correct forward output for 1-d tensors with multiple elements
- Correct forward output for 2-d tensors (preserves shape)
- For `factor=2`: matches `x ** 2`
- For `factor=3`: matches `x ** 3`
- For `factor=0.5`: matches `sqrt(x)`
- For `factor=1`: identity mapping
- Handles negative inputs for non-integer exponents (produces `nan` where applicable)
- Works across different dtypes: `float32`, `float64`
- Preserves `requires_grad` through the forward pass

##### `inverse(x)`

- Correctly inverts the forward transform: `t.inverse(t(x)) ≈ x` for `factor=2` (default)
- For arbitrary `factor`, the inverse should satisfy `t.inverse(t(x)) ≈ x` where defined
- The theoretical inverse is `x^(1/p)`; verify this holds for various exponents
- Handles zero input correctly
- Preserves shape of input tensor
- Works across float32 and float64 dtypes

##### `grad(x)`

- Matches the analytic derivative formula: `p * x^(p-1)`
- Agrees with `torch.autograd` numerical gradient of `__call__`
- For `factor=2`: `grad(x) = 2*x`
- For `factor=3`: `grad(x) = 3*x^2`
- For `factor=1`: `grad(x) = 1` (constant derivative)
- Correct for scalar (0-d) and multi-element tensors
- Correct for different dtypes

#### C. Interaction with TransformChain

- Composable with other transforms (e.g., `TransformChain([TransformPow(factor=2), TransformExp()])`)
- Forward/inverse/grad of a chain containing `TransformPow` is correct

#### D. Error Handling

- `__call__` with non-tensor input should raise `TypeError` (PyTorch handles this)
- Negative inputs with fractional exponents: verify `nan` behavior (expected from PyTorch)

#### E. Performance (if relevant)

- All operations are O(n) in the number of elements
- Suitable for use inside optimization loops

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use `pytest` with parametrize for different `factor` values
- Test default behavior (`factor=2`) plus non-default exponents (1, 3, 0.5)
- Inverse roundtrip is the key correctness test — if `t.inverse` doesn't match `x^(1/p)`, it's a bug
- Compare grad against autograd for validation

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance.transform import Transform, TransformPow


class TestTransformPow:
    """Tests for the power transform."""

    def test_constructor_default(self):
        t = TransformPow()
        assert isinstance(t, Transform)
        assert t.factor == 2.0

    def test_constructor_custom_factor(self):
        t = TransformPow(factor=3.0)
        assert t.factor == 3.0

    def test_repr(self):
        t = TransformPow(factor=3.0)
        assert repr(t) == "TransformPow(factor=3.0)"

    def test_domain_codomain(self):
        t = TransformPow()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    def test_forward_default_factor(self):
        t = TransformPow()
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(t(x), x ** 2)

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0, 0.5])
    def test_forward_matches_power(self, factor):
        t = TransformPow(factor=factor)
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.pow(x, factor))

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0, 0.5])
    def test_forward_shape_preserved(self, factor):
        t = TransformPow(factor=factor)
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
    def test_inverse_roundtrip(self, factor):
        t = TransformPow(factor=factor)
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_known_values(self):
        t = TransformPow(factor=2.0)
        assert torch.allclose(t.inverse(torch.tensor(4.0)), torch.tensor(2.0))
        assert torch.allclose(t.inverse(torch.tensor(9.0)), torch.tensor(3.0))

    def test_grad_default_factor(self):
        t = TransformPow()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        expected = 2.0 * x  # d/dx x^2 = 2x
        assert torch.allclose(t.grad(x), expected)

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0, 0.5])
    def test_grad_matches_analytic(self, factor):
        t = TransformPow(factor=factor)
        x = torch.randn(5, dtype=torch.float64) + 0.5  # avoid 0 for fractional exponents
        expected = factor * torch.pow(x, factor - 1.0)
        assert torch.allclose(t.grad(x), expected)

    @pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
    def test_grad_matches_autograd(self, factor):
        t = TransformPow(factor=factor)
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformPow(factor=2.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformPow(factor=2.0)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformPow(factor=2.0)
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Inverse correctness across all factors**: The `inverse` method is currently hardcoded as `torch.sqrt(x)`, which is only correct for `factor=2`. For any other exponent, `inverse(x)` should be `x^(1/p)`. This is a likely bug — the inverse roundtrip test with `factor != 2` will catch it.
- **Domain mismatch for odd exponents**: For `factor=2`, `inverse` expects non-negative inputs (sqrt domain), but `codomain` is ℝ. For `factor=3`, the inverse `cbrt` handles negative inputs fine.
- **Zero input with fractional exponents**: `0^(p-1)` for `0 < p < 1` produces `inf` in the grad — test that this doesn't cause downstream issues.

---

## 5. Expected Test Count Summary

| Category | Tests | Parametrized |
|---|---|---|
| Constructor & repr | 3 | — |
| Forward correctness | 3 | ×4 factors |
| Forward shape | 1 | ×4 factors |
| Inverse roundtrip | 1 | ×3 factors |
| Inverse known values | 1 | — |
| Gradient analytic | 2 | ×4 factors |
| Gradient autograd | 1 | ×3 factors |
| Dtype preservation | 2 | — |
| Integration (chain) | 2 | — |
| **Estimated total** | | **~40** |
