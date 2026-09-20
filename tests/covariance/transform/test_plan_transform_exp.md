# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.transform.transform_exp`
**Purpose of Module:**  
Provides three differentiable, bijective exponential transforms from ℝ → ℝ⁺ using different bases (e, 2, 10). Used to map unconstrained optimization parameters to positive-constrained domains (e.g., variance components).

**Classes Covered:**

- `TransformExp` — natural exponential `f(x) = e^x`
- `TransformExp2` — base-2 exponential `f(x) = 2^x`
- `TransformExp10` — base-10 exponential `f(x) = 10^x`

**Testing Goal:**  
Ensure correctness, numerical consistency, and robustness of forward, inverse, and gradient operations for all three transform classes. Each class is stateless, so testing focuses purely on functional behavior and mathematical correctness.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

For each class (`TransformExp`, `TransformExp2`, `TransformExp10`):

- Constructor initializes without errors (no arguments required)
- Domain is ℝ, codomain is ℝ⁺ (class attributes `domain` and `codomain`)
- `__repr__` returns the class name (inherited from `Transform`)
- `__str__` returns human-readable description (inherited from `Transform`)
- Instance is a subclass of `Transform` (`isinstance` check)

#### B. Method-Level Behavior

##### `__call__(x)`

- Correct forward output for scalar (0-d) tensors
- Correct forward output for 1-d tensors with multiple elements
- Correct forward output for 2-d tensors (preserves shape)
- Works across different dtypes: `float32`, `float64`
- Works across different devices: CPU, CUDA (if available)
- Handles large positive inputs (no overflow panic, just `inf` at extremes)
- Handles large negative inputs (approaches 0, no underflow panic)
- Preserves `requires_grad` through the forward pass
- Gradients flow correctly through `__call__` (composable with autograd)

##### `inverse(x)`

- Correctly inverts the forward transform: `t.inverse(t(x)) ≈ x`
- Correctly inverts for known reference values (e.g., `t.inverse(t(torch.zeros(1))) ≈ 0.0`)
- Correct output for 1-d and 2-d tensors (preserves shape)
- Raises appropriate error or produces `-inf`/`nan` for non-positive inputs (domain constraint)
- Works across float32 and float64 dtypes
- Preserves `requires_grad` through the inverse pass
- Gradients flow correctly through `inverse`

##### `grad(x)`

- Matches the analytic derivative formula for each base:
  - `TransformExp.grad(x) = e^x`
  - `TransformExp2.grad(x) = 2^x * ln(2)`
  - `TransformExp10.grad(x) = 10^x * ln(10)`
- Agrees with `torch.autograd` numerical gradient of `__call__`
- Correct for scalar (0-d) and multi-element tensors
- Correct for different dtypes and devices
- Returns positive values for all real inputs (derivative of monotonically increasing function)

#### C. Interaction Between Classes

- All three classes share the same `Transform` ABC and are interchangeable as drop-in components
- TransformChain can compose any of them with other transforms (e.g., `TransformChain([TransformExp(), TransformPow(2.0)])`)
- Forward/inverse/grad of a chain containing exponential transforms is correct

#### D. Error Handling

- `__call__` with non-tensor input should raise `TypeError` (PyTorch handles this natively)
- `inverse` with zero or negative input: verify whether `-inf`/`nan` is the expected behavior (log domain boundary) — document the behavior
- `grad` handles extreme values without `nan` in the intermediate range

#### E. Performance (if relevant)

- All operations are O(n) in the number of elements
- No unnecessary memory allocations (operation applies element-wise in-place semantics via PyTorch)
- Should be fast enough for use inside optimization loops (microseconds for typical tensor sizes)

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use `pytest` with parametrize to avoid duplicating identical tests across the three classes
- Test each method independently
- No external dependencies to mock — all transforms are pure functions of their input
- Use `torch.autograd.gradcheck` or manual `torch.autograd.grad` comparisons for derivative verification
- Where possible, compare against known mathematical identities rather than reimplementing the same formula

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance.transform import TransformExp, TransformExp2, TransformExp10


@pytest.mark.parametrize("transform_cls, base", [
    (TransformExp, None),       # e^x
    (TransformExp2, 2.0),       # 2^x
    (TransformExp10, 10.0),     # 10^x
])
class TestExponentialTransforms:
    """Shared test suite for all three exponential transform classes."""

    def test_constructor(self, transform_cls, base):
        t = transform_cls()
        assert isinstance(t, Transform)

    def test_forward_zero(self, transform_cls, base):
        t = transform_cls()
        result = t(torch.tensor(0.0))
        assert torch.allclose(result, torch.tensor(1.0))

    def test_forward_positive(self, transform_cls, base):
        t = transform_cls()
        x = torch.tensor([1.0, 2.0])
        expected = base ** x if base is not None else torch.exp(x)
        assert torch.allclose(t(x), expected)

    def test_forward_shape_preserved(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(3, 4, dtype=torch.float64)
        assert t(x).shape == x.shape

    def test_inverse_roundtrip(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t.inverse(t(x)), x)

    def test_inverse_known_values(self, transform_cls, base):
        t = transform_cls()
        # t(0) = 1, so inverse(1) = 0
        assert torch.allclose(t.inverse(torch.tensor(1.0)), torch.tensor(0.0))

    def test_inverse_non_positive(self, transform_cls, base):
        t = transform_cls()
        result = t.inverse(torch.tensor(0.0))
        assert torch.isneginf(result) or torch.isnan(result)

    def test_grad_matches_autograd(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_grad_positive(self, transform_cls, base):
        t = transform_cls()
        x = torch.randn(5, dtype=torch.float64)
        assert (t.grad(x) > 0).all()

    def test_dtype_float32(self, transform_cls, base):
        t = transform_cls()
        x = torch.tensor([1.0, -2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(t(x)).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self, transform_cls, base):
        t = transform_cls()
        x = torch.tensor([1.0, -2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(t(x)).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_domain_codomain_attributes(self, transform_cls, base):
        t = transform_cls()
        assert t.domain == "ℝ"    # ℝ
        assert t.codomain == "ℝ⁺"  # ℝ⁺


class TestTransformExpSpecific:
    """Tests specific to the natural exponential transform."""

    def test_forward_match_torch_exp(self):
        t = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.exp(x))

    def test_grad_is_self(self):
        t = TransformExp()
        x = torch.randn(5, dtype=torch.float64)
        # For e^x, grad equals forward
        assert torch.allclose(t.grad(x), t(x))


class TestTransformExp2Specific:
    """Tests specific to the base-2 exponential transform."""

    def test_forward_match_torch_exp2(self):
        t = TransformExp2()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(t(x), torch.exp2(x))

    def test_grad_includes_ln2(self):
        t = TransformExp2()
        x = torch.randn(5, dtype=torch.float64)
        ln2 = torch.log(torch.tensor(2.0, dtype=x.dtype))
        expected = torch.exp2(x) * ln2
        assert torch.allclose(t.grad(x), expected)


class TestTransformExp10Specific:
    """Tests specific to the base-10 exponential transform."""

    def test_inverse_known_values(self):
        t = TransformExp10()
        assert torch.allclose(t(torch.tensor(1.0)), torch.tensor(10.0))
        assert torch.allclose(t.inverse(torch.tensor(100.0)), torch.tensor(2.0))

    def test_grad_includes_ln10(self):
        t = TransformExp10()
        x = torch.randn(5, dtype=torch.float64)
        ln10 = torch.log(torch.tensor(10.0, dtype=x.dtype))
        expected = torch.pow(10.0, x) * ln10
        assert torch.allclose(t.grad(x), expected)
```

### 3.3 Test Execution

Don't need to execute any test.