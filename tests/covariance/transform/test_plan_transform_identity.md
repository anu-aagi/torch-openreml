# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.transform.transform_identity`
**Purpose of Module:**
Provides a trivial identity transform `f(x) = x` mapping ℝ → ℝ. Used as a no-op placeholder or neutral element in transform chains where no parameter transformation is needed.

**Classes Covered:**

- `TransformIdentity` — identity transform `f(x) = x`

**Testing Goal:**
Ensure the identity transform is a true no-op: forward and inverse both return the input unchanged, grad returns all ones, and it behaves as a neutral element in transform chains.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor initializes without errors (no arguments required)
- Domain is ℝ, codomain is ℝ (class attributes)
- Instance is a subclass of `Transform` (`isinstance` check)

#### B. Method-Level Behavior

##### `__call__(x)`

- Returns input unchanged (identity)
- Preserves values exactly (no floating-point drift)
- Preserves shape of input tensor
- Works for scalar (0-d) and multi-dimensional tensors
- Works across different dtypes: `float32`, `float64`
- Preserves `requires_grad`

##### `inverse(x)`

- Returns input unchanged (self-inverse)
- `t.inverse(t(x))` is `x` by identity
- Preserves values, shape, dtype
- Preserves `requires_grad`

##### `grad(x)`

- Returns ones with same shape and dtype as input
- Agrees with `torch.autograd` numerical gradient of `__call__`
- Correct for scalar and multi-element tensors

#### C. Interaction with TransformChain

- Acts as neutral element: `TransformChain([TransformIdentity(), T])` ≡ `T`
- Acts as neutral element: `TransformChain([T, TransformIdentity()])` ≡ `T`
- Composing with itself: `TransformChain([TransformIdentity(), TransformIdentity()])` is still identity
- Forward/inverse/grad of a chain with Identity is correct

#### D. Error Handling

- Non-tensor input should raise `TypeError` (PyTorch handles this)

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Straightforward value-equality checks — no numerical approximations needed
- Use `torch.equal` or `allclose` for exact match verification
- Chain tests verify neutral-element and idempotent composition

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformIdentity,
    TransformExp,
    TransformChain,
)


class TestTransformIdentity:
    """Tests for the identity transform."""

    def test_constructor(self):
        t = TransformIdentity()
        assert isinstance(t, Transform)

    def test_domain_codomain(self):
        t = TransformIdentity()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    def test_forward_returns_input(self):
        t = TransformIdentity()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(t(x), x)

    def test_forward_scalar(self):
        t = TransformIdentity()
        x = torch.tensor(3.5)
        assert torch.equal(t(x), x)

    def test_forward_2d_preserves_shape(self):
        t = TransformIdentity()
        x = torch.randn(3, 4, dtype=torch.float64)
        result = t(x)
        assert result.shape == x.shape
        assert torch.equal(result, x)

    def test_inverse_returns_input(self):
        t = TransformIdentity()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(t.inverse(x), x)

    def test_inverse_roundtrip(self):
        t = TransformIdentity()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(t.inverse(t(x)), x)

    def test_grad_returns_ones(self):
        t = TransformIdentity()
        x = torch.randn(5, dtype=torch.float64)
        expected = torch.ones_like(x)
        assert torch.equal(t.grad(x), expected)

    def test_grad_scalar(self):
        t = TransformIdentity()
        x = torch.tensor(5.0)
        assert torch.equal(t.grad(x), torch.tensor(1.0))

    def test_grad_matches_autograd(self):
        t = TransformIdentity()
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = t.grad(x.detach())
        y = t(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_dtype_float32(self):
        t = TransformIdentity()
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert t(x).dtype == torch.float32
        assert t.inverse(x).dtype == torch.float32
        assert t.grad(x).dtype == torch.float32

    def test_dtype_float64(self):
        t = TransformIdentity()
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert t(x).dtype == torch.float64
        assert t.inverse(x).dtype == torch.float64
        assert t.grad(x).dtype == torch.float64

    def test_forward_preserves_requires_grad(self):
        t = TransformIdentity()
        x = torch.randn(5, requires_grad=True)
        assert t(x).requires_grad

    def test_inverse_preserves_requires_grad(self):
        t = TransformIdentity()
        x = torch.randn(5, requires_grad=True)
        assert t.inverse(x).requires_grad


class TestTransformChainWithIdentity:
    """Integration tests with TransformChain — identity as neutral element."""

    def test_identity_then_exp_equals_exp(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain(x), exp(x))

    def test_exp_then_identity_equals_exp(self):
        chain = TransformChain([TransformExp(), TransformIdentity()])
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain(x), exp(x))

    def test_identity_then_identity_is_identity(self):
        chain = TransformChain([TransformIdentity(), TransformIdentity()])
        x = torch.randn(10, dtype=torch.float64)
        assert torch.equal(chain(x), x)

    def test_chain_with_identity_inverse_roundtrip(self):
        chain = TransformChain([TransformIdentity(), TransformExp(), TransformIdentity()])
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain.inverse(chain(x)), x)
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Triviality is the risk**: The identity transform is so simple that the main risk is accidental mutation of the input tensor (`__call__` should not clone or detach). Tests use `torch.equal` to verify exact value and object preservation where appropriate.
- **Neutral element**: The chain tests verify that `TransformIdentity` composes correctly and does not introduce floating-point drift when placed before/after other transforms.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor & class attributes | 2 |
| Forward correctness | 3 |
| Inverse correctness | 2 |
| Gradient correctness | 3 |
| Dtype preservation | 2 |
| requires_grad preservation | 2 |
| Chain (neutral element) | 4 |
| **Total** | **18** |
