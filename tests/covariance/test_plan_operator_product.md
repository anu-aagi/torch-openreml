# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_product`
**Purpose of Module:**
Provides a `Product` operator: `V = A @ B` (matrix product). It is the plain matrix-multiply member of the product family, complementing the element-wise `HadamardProduct`, the external `KroneckerProduct`, and the fixed-structure `CovariancePropagation` (`Z G Z^T`) and `Gram` (`X^T X`). Operands may be rectangular: if `A` is `m × n` and `B` is `n × p`, the result is `m × p`, so the output need be neither square nor symmetric.

**Classes Covered:**

- `Product` (extends `Operator`) — matrix product `V = A @ B`

**Testing Goal:**
Ensure correct matrix-product evaluation, correct product-rule gradients for both operands (including rectangular operands and the 3D-Jacobian/2D-operand broadcast), correct parameter namespacing and ordering, intermediate caching, and dtype/device resolution across the two operands.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires exactly 2 operands → `ValueError` for 1 operand and for 3 operands
- Positional operands are named `op_0` / `op_1`
- Instance is a subclass of `Matrix` (via `Operator`)
- `repr` contains the class name

#### B. `__call__(free_params)` Behavior

- Returns `A @ B`, compared against evaluating each operand separately at its own parameter slice
- Transform anchor: `ScalarMatrix(2) @ IdentityMatrix(2)` at `θ = 0.5` gives `e · I` (`ExpPow2`)
- `IdentityMatrix @ IdentityMatrix` returns the identity exactly
- Fixed `torch.Tensor` operand in first position and in second position
- Dict parameter input gives the same result as the flat tensor input

#### C. Shape Behavior

- Square result, and `shape` attribute updated after the call
- Rectangular result: fixed design `DummyMatrix` (4 × 3) @ `DiagonalMatrix(3)` → 4 × 3
- Rectangular with fixed tensor first: (2 × 4) @ (4 × 3) → 2 × 3
- Incompatible adjacent dimensions raise torch's `RuntimeError` at call time (no hand-written check)

#### D. Parameter Namespacing

- `a/sigma^2`, `b/sigma^2_0`, `b/sigma^2_1`, ... in operand order
- Only the trainable operand's names appear when the other is a `Matrix` with no free parameters
- Second operand trainable only
- Positional operands still use `op_0` / `op_1` prefixes

#### E. Manual Gradient — A Parameters

- `dV/dθ_A = dA @ B`, checked against `A.manual_grad(...) @ B`
- Rectangular A: `ScalarMatrix(4)` @ fixed `DummyMatrix` (4 × 3), gradient shape `(1, 4, 3)`

#### F. Manual Gradient — B Parameters

- `dV/dθ_B = A @ dB`, checked against `A @ B.manual_grad(...)`
- Fixed tensor as A: gradient is `x @ dB`
- Rectangular B: fixed `DummyMatrix` (4 × 3) @ `DiagonalMatrix(3)`, gradient shape `(3, 4, 3)`

#### G. Gradient Structure and Consistency

- `manual_grad` shape `(num_free_params, *shape)` and concatenation order A-then-B
- `manual_grad` vs `auto_grad` for square, for rectangular operands (`LowerTriangularMatrix(3, 2)` @ `DiagonalMatrix(2)`, gradient `(7, 3, 2)`), and for the same `Matrix` instance passed as both operands
- Same instance twice still yields two namespaced gradient blocks and two free parameters
- All-fixed operands → `(None, [])`
- `grad()` agrees with `manual_grad` and names match `free_param_names`

#### H. Intermediate Caching

- Cache is populated under `"a"`, `"b"`, `"v"` after a call, keyed by built parameters
- `reset_intermediates` and `disable_cache` clear it
- A new parameter tensor recomputes (no stale reuse)

#### I. Dtype and Device Contract

- Input parameter dtype wins (flat tensor and dict input)
- Input device wins (`cuda` → `mps` → skip)
- Fixed tensor operand follows the input dtype, casting both down and up
- With no parameters at all, Torch defaults are used

#### J. Tree Integration

- `tree` paths `"/"`, `"a"`, `"b"`, with the root equal to `__call__`
- `free_param_tree` paths `"/"`, `"a"`, `"b"`, root the input tensor, each operand its own slice
- `grad_tree` root matches `manual_grad`, and nested under `Sum` the namespacing becomes `scaled/a/sigma^2`

#### K. REML Interface

- The old `map_theta_to_v` / `map_theta_to_dv` methods were removed from the library; the estimator uses `__call__` and `grad()`. Coverage is through the `__call__` value tests and the `grad()`-vs-`manual_grad` test; no separate REML-interface tests exist.

#### L. `inverse(free_params)`

`Product` overrides the base default with the closed form `V⁻¹ = B⁻¹A⁻¹`: each operand is inverted through its
own path and the two results are multiplied in **reverse** order, instead of inverting the composite densely.

- Agrees with `torch.linalg.inv(op(free_params))`
- Is exactly `B.inverse(...) @ A.inverse(...)`, and is *not* the forward-ordered product
  `A.inverse(...) @ B.inverse(...)` — the operands must be non-commuting for that test to mean anything
- `op.inverse(free_params) @ op(free_params)` is the identity
- The composite is never inverted densely: `torch.linalg.inv` is not called
- A fixed tensor operand is inverted directly, and a nested operand is handled the same way, by value
- A rectangular operand raises (the composite has no inverse)
- The argument contract is not bypassed: `None` means the defaults, a dict means the same as the flat
  tensor, and a non-tensor, a wrong-length vector and a bad dict each raise
- dtype follows the input, the result stays attached to `free_params`, and its gradient equals the gradient
  of `torch.linalg.inv(op(free_params))`

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use small matrices with values verifiable by hand, and compare `manual_grad` against `auto_grad`
- Compare against operand-by-operand evaluation (`a(params_a) @ b(params_b)`) rather than re-deriving transform arithmetic
- Cover both trainable positions, fixed tensors in both positions, and one genuinely rectangular trainable pair
- Follow the suite's dtype/device conventions: input dtype/device resolution, fixed-operand casting, inline `cuda` → `mps` → `pytest.skip`

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**.

```python
import torch
import math
import pytest
from torch_openreml.covariance import (
    Product,
    Sum,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    DummyMatrix,
    LowerTriangularMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestProduct:
    """Tests for the Product operator."""

    def test_constructor(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            Product(ScalarMatrix(3))

    def test_constructor_rejects_three(self):
        with pytest.raises(ValueError, match="Two operands"):
            Product(ScalarMatrix(3), ScalarMatrix(3), ScalarMatrix(3))

    def test_positional_operand_names(self):
        op = Product(ScalarMatrix(3), ScalarMatrix(3))
        assert op.free_param_names == ["op_0/sigma^2", "op_1/sigma^2"]

    def test_call_scales_by_operand_values(self):
        op = Product(a=ScalarMatrix(2), b=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        result = op(free_params)
        assert torch.allclose(result, math.exp(1) * torch.eye(2))

    def test_call_matches_matmul(self):
        op = Product(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 0.5])
        a = ScalarMatrix(2)
        b = DiagonalMatrix(2)
        assert torch.allclose(op(free_params), a(free_params[:1]) @ b(free_params[1:]))

    def test_call_identity_operands(self):
        op = Product(a=IdentityMatrix(3), b=IdentityMatrix(3))
        assert torch.equal(op(torch.tensor([])), torch.eye(3))

    def test_call_with_fixed_tensor_first(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = Product(a=x, b=IdentityMatrix(2))
        assert torch.allclose(op(torch.tensor([])), x)

    def test_call_with_fixed_tensor_second(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = Product(a=IdentityMatrix(2), b=x)
        assert torch.allclose(op(torch.tensor([])), x)

    def test_call_with_dict_params(self):
        op = Product(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 0.5])
        expected = op(free_params)
        result = op(
            {
                "a/sigma^2": free_params[:1],
                "b/sigma^2_0": free_params[1:2],
                "b/sigma^2_1": free_params[2:],
            }
        )
        assert torch.allclose(result, expected)

    def test_shape_square(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        assert op(free_params).shape == (3, 3)
        assert op.shape == (3, 3)

    def test_shape_rectangular(self):
        op = Product(a=DummyMatrix(["a", "b", "c", "a"]), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op(free_params).shape == (4, 3)
        assert op.shape == (4, 3)

    def test_shape_rectangular_fixed_first(self):
        x = torch.ones(2, 4)
        op = Product(a=x, b=DummyMatrix(["a", "b", "c", "a"]))
        result = op(torch.tensor([]))
        assert result.shape == (2, 3)
        assert torch.allclose(result, x @ DummyMatrix(["a", "b", "c", "a"])(torch.tensor([])))

    def test_mismatched_shapes_raise_at_runtime(self):
        op = Product(a=IdentityMatrix(2), b=IdentityMatrix(3))
        with pytest.raises(RuntimeError):
            op(torch.tensor([]))

    def test_param_namespacing(self):
        op = Product(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        assert op.num_free_params == 3
        assert op.free_param_names == ["a/sigma^2", "b/sigma^2_0", "b/sigma^2_1"]

    def test_param_namespacing_one_trainable(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op.num_free_params == 1
        assert op.free_param_names == ["a/sigma^2"]

    def test_param_namespacing_second_trainable(self):
        op = Product(a=IdentityMatrix(3), b=ScalarMatrix(3))
        assert op.num_free_params == 1
        assert op.free_param_names == ["b/sigma^2"]

    def test_inverse_matches_inverse_of_call(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_is_reversed_product_of_operand_inverses(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        expected = LowerTriangularMatrix(2, 2).inverse(free_params[2:]) @ DiagonalMatrix(2).inverse(free_params[:2])
        assert torch.allclose(op.inverse(free_params), expected)

    def test_inverse_is_not_the_forward_ordered_product(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        forward = DiagonalMatrix(2).inverse(free_params[:2]) @ LowerTriangularMatrix(2, 2).inverse(free_params[2:])
        assert not torch.allclose(op.inverse(free_params), forward)

    def test_inverse_product_with_call_is_identity(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        assert torch.allclose(op.inverse(free_params) @ op(free_params), torch.eye(2), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        op.inverse(torch.tensor([0.0, 0.5, 1.0]))
        assert calls == []

    def test_inverse_with_fixed_tensor_operand(self):
        op = Product(a=DiagonalMatrix(2), b=torch.eye(2) * 2.0)
        free_params = torch.tensor([0.0, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_rectangular_operand_raises(self):
        op = Product(a=LowerTriangularMatrix(3, 2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3, 1.0])
        with pytest.raises(RuntimeError):
            op.inverse(free_params)

    def test_inverse_uses_default_free_params(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_validation_is_not_bypassed(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.0, 0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert "Product" in repr(op)


class TestProductGrad:
    """Tests for Product gradient computation."""

    def test_manual_grad_shape(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 3, 3)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_a_equals_da_matmul_b(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, grad_names = op.manual_grad(free_params)
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, grad_a @ torch.eye(3))
        assert grad_names == ["a/sigma^2"]

    def test_manual_grad_b_equals_a_matmul_db(self):
        op = Product(a=IdentityMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, grad_names = op.manual_grad(free_params)
        b = ScalarMatrix(3)
        grad_b, _ = b.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, torch.eye(3) @ grad_b)
        assert grad_names == ["b/sigma^2"]

    def test_manual_grad_with_fixed_tensor_first(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = Product(a=x, b=ScalarMatrix(2))
        free_params = torch.tensor([0.5])
        grad_op, grad_names = op.manual_grad(free_params)
        b = ScalarMatrix(2)
        grad_b, _ = b.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, x @ grad_b)
        assert grad_names == ["b/sigma^2"]

    def test_manual_grad_rectangular_a(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = Product(a=ScalarMatrix(4), b=z)
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        a = ScalarMatrix(4)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        assert grad.shape == (1, 4, 3)
        assert torch.allclose(grad, grad_a @ z(torch.tensor([])))
        assert grad_names == ["a/sigma^2"]

    def test_manual_grad_rectangular_b(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = Product(a=z, b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.manual_grad(free_params)
        b = DiagonalMatrix(3)
        grad_b, _ = b.manual_grad(free_params)
        assert grad.shape == (3, 4, 3)
        assert torch.allclose(grad, z(torch.tensor([])) @ grad_b)
        assert grad_names == ["b/sigma^2_0", "b/sigma^2_1", "b/sigma^2_2"]

    def test_manual_grad_vs_auto_grad(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_vs_auto_grad_rectangular(self):
        op = Product(a=LowerTriangularMatrix(3, 2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5])
        manual, _ = op.manual_grad(free_params)
        auto, _ = op.auto_grad(free_params)
        assert manual.shape == (7, 3, 2)
        assert torch.allclose(manual, auto)

    def test_manual_grad_vs_auto_grad_same_instance(self):
        m = ScalarMatrix(3)
        op = Product(a=m, b=m)
        free_params = torch.tensor([0.5, 0.5])
        assert op.num_free_params == 2
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert manual.shape == (2, 3, 3)
        assert torch.allclose(manual, auto)
        assert names_m == names_a == ["a/sigma^2", "b/sigma^2"]

    def test_all_fixed(self):
        op = Product(a=IdentityMatrix(3), b=torch.eye(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_matches_manual_grad(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 1.5])
        grad, grad_names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert torch.allclose(grad, manual)
        assert grad_names == manual_names == op.free_param_names


class TestProductCache:
    """Tests for Product intermediate caching."""

    def test_intermediate_cache_hit(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "b" in cache
        assert "v" in cache

    def test_reset_intermediates_clears_cache(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.reset_intermediates()
        assert op.get_intermediates(built) is None

    def test_disable_cache_clears_cache(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.disable_cache()
        assert op.get_intermediates(built) is None

    def test_cache_recomputes_on_new_params(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        first = torch.tensor([0.0, 0.0])
        second = torch.tensor([0.5, 0.5])
        v_first = op(first)
        v_second = op(second)
        assert not torch.allclose(v_first, v_second)
        assert torch.allclose(v_second, op(second))


class TestProductDtypeDevice:
    """Tests for dtype and device resolution across the two operands."""

    def test_input_tensor_dtype_is_used(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_input_dict_dtype_is_used(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        v = op(
            {
                "a/sigma^2": torch.tensor([0.5], dtype=torch.float64),
                "b/sigma^2": torch.tensor([0.5], dtype=torch.float64),
            }
        )
        assert v.dtype == torch.float64

    def test_input_device_is_used(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.tensor([0.5], device=device)).device.type == device

    def test_tensor_operand_follows_input_dtype(self):
        op = Product(a=ScalarMatrix(3), b=torch.eye(3, dtype=torch.float64))
        assert op(torch.tensor([0.5], dtype=torch.float32)).dtype == torch.float32

    def test_tensor_operand_casts_up_to_input_dtype(self):
        op = Product(a=torch.eye(3), b=ScalarMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_no_params_uses_torch_defaults(self):
        op = Product(a=IdentityMatrix(3), b=IdentityMatrix(3))
        assert op.num_params == 0
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()


class TestProductTree:
    """Tests for Product integration with tree, free_param_tree and grad_tree."""

    def test_tree(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0, 1.5])
        results = op.tree(free_params)
        assert sorted(results) == ["/", "a", "b"]
        assert torch.allclose(results["/"], op(free_params))

    def test_free_param_tree(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0, 1.5])
        free_params_by_path = op.free_param_tree(free_params)
        assert sorted(free_params_by_path) == ["/", "a", "b"]
        assert free_params_by_path["/"] is free_params
        assert free_params_by_path["a"].shape == (1,)
        assert free_params_by_path["b"].shape == (3,)
        assert torch.equal(free_params_by_path["b"], free_params[1:])

    def test_grad_tree(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0, 1.5])
        grads = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a", "b"]
        manual, names = op.manual_grad(free_params)
        assert torch.allclose(grads["/"][0], manual)
        assert grads["/"][1] == names == op.free_param_names

    def test_nested_in_sum(self):
        op = Sum(scaled=Product(a=ScalarMatrix(2), b=IdentityMatrix(2)), extra=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["scaled/a/sigma^2"]
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Order is semantic**: matrix multiplication is not commutative, so `A @ B` and `B @ A` generally differ. The operator documents the first operand as `A` and the second as `B`, and both the constructor examples and tests use named operands (`a=`, `b=`) to keep the direction explicit. Positional operands give no such signal — they are only `op_0` / `op_1`.
- **3D Jacobian against 2D operand**: `da` has shape `(k, m, n)` and `b` is `(n, p)`, so `da @ b` broadcasts to `(k, m, p)`; symmetrically `a @ db` broadcasts a 2D left operand against `db`'s batch dimension. The right-multiplicand case (`a @ db`) is the one worth watching, because putting `db` on the left would silently compute the wrong (or an invalid) product. The rectangular tests pin both directions.
- **No shape validation, by design**: like `HadamardProduct` and `KroneckerProduct`, the constructor cannot know operand shapes (`Operator` passes `None` to `Matrix.__init__`, and shapes are only known after `__call__`), so compatibility is enforced by torch's matmul at call time. The test asserts a `RuntimeError` rather than a custom message, matching the other product operators.
- **Result need not be symmetric or positive definite**: `Product` is a structural building block like `Gram` / `Augment`; downstream code decides whether the product is a valid covariance.
- **Same instance as both operands is legal**: `Product(a=m, b=m)` exposes two namespaced parameter blocks (`a/sigma^2`, `b/sigma^2`) and the gradient sums the two roles' contributions implicitly through concatenation. This mirrors the operator framework's per-operand namespacing rather than per-instance aliasing.
- **Gradient concatenation order**: A's block comes first, then B's, matching `free_param_names`. The names from `operands_grad` already carry the operand prefix, so no reordering is needed.
- **Caching stores both operand values**, not just `v`, so `manual_grad` after a call needs no re-evaluation of the operands — the source of the `a` / `b` entries asserted in the cache tests.
- **`inverse` reverses the operand order**, the mirror image of the "order is semantic" risk above: `V⁻¹ = B⁻¹A⁻¹`, so the second operand's inverse comes *first*. The operands in the reverse-order test are deliberately non-commuting (`DiagonalMatrix` and `LowerTriangularMatrix(2, 2)`), because with commuting operands the reversed and forward products coincide and the test would pass either way; `test_inverse_is_not_the_forward_ordered_product` states the same fact from the other side.
- **The `inverse` path must not go through `__call__`**: a fallback that inverted `op(free_params)` would pass every value test, so the monkeypatched-`torch.linalg.inv` test is what pins the closed form; it asserts the dense inverse is not called at all. A fixed tensor operand is inverted directly by `operands_inverse`, which is why there is a dedicated fixed-operand inverse test rather than coverage through the trainable path.
- **A square composite does not imply invertible operands**: two rectangular operands can multiply to a square `(m, m)` matrix, but `Product` gives each operand's *own* inverse, which exists only for square operands. The override therefore covers square operands only; a rectangular pair raises from the operand's inverse instead of falling back. The rectangular test uses `LowerTriangularMatrix(3, 2) @ ScalarMatrix(2)`, whose composite is `(3, 2)` and has no inverse at all.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor, positional names, `repr` | 5 |
| `__call__` values | 6 |
| Shape behavior | 4 |
| Parameter namespacing | 3 |
| `manual_grad` structure & per-operand values | 7 |
| `manual_grad` vs `auto_grad` | 3 |
| All-fixed | 1 |
| Intermediate caching | 4 |
| Dtype / device | 6 |
| Tree integration | 4 |
| `inverse` override | 13 |
| **Total** | **56** |
