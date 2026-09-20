# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.transform.transform`
**Purpose of Module:**
Defines the abstract base class `Transform` and the composite `TransformChain` class. All concrete transforms inherit from `Transform`. `TransformChain` allows composing multiple transforms sequentially.

**Classes Covered:**

- `Transform` — abstract base class defining the transform interface
- `TransformChain` — sequential composition of multiple transforms

**Testing Goal:**
Ensure the ABC enforces the correct interface contract, and that `TransformChain` correctly composes forward, inverse, and gradient operations in both order and reverse order.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Transform (ABC)

- Cannot be instantiated directly (`TypeError` due to abstract methods)
- Subclasses must implement `__call__`, `inverse`, `grad`
- Default `domain` and `codomain` are ℝ
- `__repr__` returns `ClassName()`
- `__str__` returns `ClassName: domain ↦ codomain`

#### B. TransformChain — Constructor

- Accepts a list of Transform objects
- Accepts a single Transform (auto-wraps to list)
- Raises `TypeError` if any element is not a Transform instance
- Stores chain as a list

#### C. TransformChain — `__call__(x)`

- Applies transforms in forward order: `t_n(...(t_1(x)))`
- Empty chain: returns input unchanged (identity behavior, if allowed)
- Single-element chain: equivalent to calling that transform directly
- Multi-element chain: composite output is correct
- Preserves shape of input
- Preserves `requires_grad`

#### D. TransformChain — `inverse(x)`

- Applies inverse transforms in reverse order: `t_1^{-1}(...(t_n^{-1}(x)))`
- Roundtrip: `chain.inverse(chain(x)) ≈ x`
- Single-element chain: equivalent to calling that transform's inverse directly

#### E. TransformChain — `grad(x)`

- Computes chain rule: multiplies `grad` of each transform evaluated at the intermediate forward values
- Matches `torch.autograd` numerical gradient of the chain's `__call__`
- Empty chain: grad returns 1 (or identity-like behavior)

#### F. TransformChain — repr/str

- `__repr__` returns `TransformChain([...])` with repr of inner transforms
- `__str__` returns `TransformChain([...])` with repr of inner transforms

#### G. Interaction with concrete transforms

- Works correctly with any combination of Exp, Pow, ScaleShift, Sigmoid, Identity, ExpPow2

#### H. Error Handling

- Constructor with non-Transform elements raises `TypeError`
- Constructor with non-list/tuple raises `TypeError` (attribute error on iteration)

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Test ABC contract with a minimal concrete subclass
- Test TransformChain with known simple transforms (Identity, Exp) to verify ordering
- Chain rule grad correctness is the most important test — compare against autograd
- Test edge cases: empty chain, single-element chain, three-element chain
- Verify inverse-reverse-order with asymmetric transforms (non-commutative)

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance.transform import (
    Transform,
    TransformChain,
    TransformIdentity,
    TransformExp,
    TransformPow,
    TransformScaleShift,
)


class TestTransformABC:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Transform()

    def test_concrete_subclass_instantiates(self):
        t = TransformIdentity()
        assert isinstance(t, Transform)

    def test_default_domain_codomain(self):
        class MinimalTransform(Transform):
            def __call__(self, x):
                return x
            def inverse(self, x):
                return x
            def grad(self, x):
                return torch.ones_like(x)

        t = MinimalTransform()
        assert t.domain == "ℝ"
        assert t.codomain == "ℝ"

    def test_repr(self):
        assert repr(TransformIdentity()) == "TransformIdentity()"

    def test_str(self):
        s = str(TransformIdentity())
        assert "TransformIdentity" in s
        assert "ℝ" in s


class TestTransformChain:
    """Tests for the TransformChain composite."""

    def test_constructor_with_list(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        assert len(chain.chain) == 2

    def test_constructor_with_single_transform(self):
        chain = TransformChain(TransformIdentity())
        assert len(chain.chain) == 1
        assert isinstance(chain.chain, list)

    def test_constructor_raises_on_non_transform(self):
        with pytest.raises(TypeError):
            TransformChain([TransformIdentity(), "not_a_transform"])

    def test_call_single_is_equivalent(self):
        chain = TransformChain(TransformExp())
        exp = TransformExp()
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain(x), exp(x))

    def test_call_applies_in_order(self):
        # Pow then ScaleShift: f(x) = 2*(x^2) + 1
        chain = TransformChain([TransformPow(factor=2.0), TransformScaleShift(a=2.0, b=1.0)])
        x = torch.randn(10, dtype=torch.float64)
        expected = 2.0 * (x**2) + 1.0
        assert torch.allclose(chain(x), expected)

    def test_call_shape_preserved(self):
        chain = TransformChain([TransformExp(), TransformPow(factor=3.0)])
        x = torch.randn(3, 4, dtype=torch.float64)
        assert chain(x).shape == x.shape

    def test_call_preserves_requires_grad(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        x = torch.randn(5, requires_grad=True)
        assert chain(x).requires_grad

    def test_inverse_applies_in_reverse_order(self):
        # forward: x → x^2 → exp(x^2)
        # inverse: y → log(y) → sqrt(log(y))
        chain = TransformChain([TransformPow(factor=2.0), TransformExp()])
        x = torch.randn(10, dtype=torch.float64)
        assert torch.allclose(chain.inverse(chain(x)), x)

    def test_inverse_single_is_equivalent(self):
        chain = TransformChain(TransformExp())
        exp = TransformExp()
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(chain.inverse(x), exp.inverse(x))

    def test_grad_matches_autograd_two_transform(self):
        chain = TransformChain([TransformExp(), TransformPow(factor=2.0)])
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = chain.grad(x.detach())
        y = chain(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_grad_matches_autograd_three_transform(self):
        chain = TransformChain([
            TransformScaleShift(a=2.0, b=1.0),
            TransformExp(),
            TransformPow(factor=3.0),
        ])
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        analytical = chain.grad(x.detach())
        y = chain(x)
        y.sum().backward()
        assert torch.allclose(analytical, x.grad)

    def test_grad_single_is_equivalent(self):
        chain = TransformChain(TransformExp())
        exp = TransformExp()
        x = torch.randn(5, dtype=torch.float64)
        assert torch.allclose(chain.grad(x), exp.grad(x))

    def test_repr(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        r = repr(chain)
        assert "TransformChain" in r
        assert "TransformIdentity" in r
        assert "TransformExp" in r

    def test_dtype_preserved(self):
        chain = TransformChain([TransformIdentity(), TransformExp()])
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert chain(x).dtype == torch.float64
        assert chain.inverse(chain(x)).dtype == torch.float64
        assert chain.grad(x).dtype == torch.float64
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Empty chain**: `TransformChain([])` — the code doesn't explicitly guard against this. An empty chain would cause identity-like behavior in `__call__` (no transforms applied). Verify this is handled or explicitly forbidden.
- **Chain rule correctness**: The `grad` method evaluates each transform's `grad` at the intermediate forward value. This is a subtle point — the derivative of `t2(t1(x))` is `t2'(t1(x)) * t1'(x)`. The code must evaluate `t1.grad(x)` first, then `t2.grad(t1(x))`, multiplying as it goes. Tests compare against autograd to catch ordering bugs.
- **Non-commutative transforms**: The inverse must reverse the order. `Chain([A, B]).inverse` should apply `A.inverse(B.inverse(y))`, not `B.inverse(A.inverse(y))`. The roundtrip test catches this: `chain.inverse(chain(x))` only equals x if inverse order is correct.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Transform ABC (instantiation, contract) | 5 |
| TransformChain constructor | 3 |
| TransformChain `__call__` | 4 |
| TransformChain `inverse` | 2 |
| TransformChain `grad` | 3 |
| TransformChain repr/str | 1 |
| Dtype preservation | 1 |
| **Total** | **19** |
