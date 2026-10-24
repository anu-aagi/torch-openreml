# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_transpose`
**Purpose of Module:**
Provides a `Transpose` operator: `V = Aᵀ`. If `A` is `m × n`, the result is `n × m`. It is the structural piece that makes transposes of trainable operands expressible for the first time — before it, only `Gram`'s fixed `XᵀX`/`XXᵀ` and the transpose hard-coded inside `CovariancePropagation`'s gradient used a transpose at all. With it, `Aᵀ B`, `A Bᵀ`, and general sandwiches `A B Aᵀ` compose out of `Product`, and `CovariancePropagation` becomes a special case of that sandwich.

**Classes Covered:**

- `Transpose` (extends `Operator`) — transpose `V = Aᵀ`

**Testing Goal:**
Ensure correct transpose evaluation for square and rectangular operands, correct Jacobian (`dAᵀ` via a last-two-dims transpose of the operand's 3D Jacobian), namespacing pass-through, intermediate caching, dtype/device resolution, tree integration, and equivalence with `CovariancePropagation` when composed into a sandwich.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor accepts exactly 1 operand → `ValueError` (`"One operand is required"`) for 2 operands
- No operand at all raises the base-class `TypeError` (no `Matrix` operand present)
- Positional operand is named `op_0`
- Instance is a subclass of `Matrix` (via `Operator`)
- `repr` contains the class name

#### B. `__call__(free_params)` Behavior

- Transform anchor: `ScalarMatrix(2)` at `θ = 0.5` transposes to `e · I` (symmetric, unchanged)
- Square operand (`DiagonalMatrix`) matches the operand's own evaluation
- Rectangular fixed operand: `DummyMatrix` (4 × 3) → 3 × 4, exactly the operand's `.T`
- Rectangular trainable operand: `LowerTriangularMatrix(3, 2)` → 2 × 3, matches the operand's `.T`
- Dict parameter input gives the same result as the flat tensor input

#### C. Shape Behavior

- `op.shape` is updated after the call and equals the transposed operand shape

#### D. Parameter Namespacing

- `a/L_0_0`, `a/L_1_0`, ... in operand order for a trainable rectangular operand
- A fixed operand contributes no free parameters (`num_free_params == 0`)
- Positional operand prefix `op_0/...`

#### E. Manual Gradient — Structure and Values

- `manual_grad` shape `(num_free_params, n, m)` — the transposed shape
- `dV/dθ = (dA/dθ)ᵀ`, checked against the operand's own `manual_grad(...).mT`
- Names pass through unchanged from `operands_grad`

#### F. Gradient Consistency

- `manual_grad` vs `auto_grad` for a square operand and for a rectangular operand
- All-fixed operand → `(None, [])`
- `grad()` agrees with `manual_grad`, names equal `free_param_names`

#### G. Intermediate Caching

- Cache is populated under `"a"` and `"v"` after a call, keyed by built parameters
- `reset_intermediates` and `disable_cache` clear it
- A new parameter tensor recomputes (no stale reuse)

#### H. Dtype and Device Contract

- Input parameter dtype wins (flat tensor and dict input)
- Input device wins (`cuda` → `mps` → skip)
- With no parameters at all, Torch defaults are used

#### I. Tree Integration

- `tree` paths `"/"` and `"a"`, root equal to `__call__`
- `free_param_tree` paths `"/"` and `"a"`, root the input tensor, `"a"` its own slice
- `grad_tree` root matches `manual_grad`
- Nested under `Sum`, namespacing becomes `transposed/a/sigma^2`

#### J. Composition

- `Transpose(Transpose(A))` returns `A` (involution)
- `Product(Z, Product(G, Transpose(Z)))` equals `CovariancePropagation(z=Z, g=G)` in both values and gradients — the sandwich equivalence that motivates the operator

#### K. REML Interface

- The old `map_theta_to_v` / `map_theta_to_dv` methods were removed from the library; the estimator uses `__call__` and `grad()`. Coverage is through the `__call__` value tests, the `grad()`-vs-`manual_grad` test, and the composition tests; no separate REML-interface tests exist.

#### L. `inverse(free_params)`

`Transpose` overrides the base default with the closed form `V⁻¹ = (A⁻¹)ᵀ`: the operand is inverted through its
own path (so a structured operand keeps its structured inverse) and the result is transposed, instead of
inverting the composite densely.

- Agrees with `torch.linalg.inv(op(free_params))`
- Is exactly the operand's own inverse transposed, `LowerTriangularMatrix(2, 2).inverse(free_params).T`
- `op(free_params) @ op.inverse(free_params)` is the identity
- The composite is never inverted densely: `torch.linalg.inv` is not called
- A nested operand (`Product`) is handled the same way, by value
- A non-square operand raises (`RuntimeError` from the operand's own inverse), since `op` has no inverse then
- The argument contract is not bypassed: `None` means the defaults, a dict means the same as the flat
  tensor, and a non-tensor, a wrong-length vector and a bad dict each raise
- dtype follows the input, the result stays attached to `free_params`, and its gradient equals the gradient
  of `torch.linalg.inv(op(free_params))`

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use small operands with values verifiable by hand, and compare `manual_grad` against `auto_grad`
- Compare against the operand's own evaluation (`.T` for values, `.mT` for the 3D Jacobian) rather than re-deriving transform arithmetic
- Use a rectangular trainable operand (`LowerTriangularMatrix(3, 2)`) so transposition is observably not a no-op in both value and gradient shape
- Use a fixed design matrix (`DummyMatrix`) for the rectangular fixed case
- Follow the suite's dtype/device conventions: input dtype/device resolution, inline `cuda` → `mps` → `pytest.skip`

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**.

```python
import torch
import math
import pytest
from torch_openreml.covariance import (
    Transpose,
    Sum,
    Product,
    CovariancePropagation,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    DummyMatrix,
    LowerTriangularMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestTranspose:
    """Tests for the Transpose operator."""

    def test_constructor(self):
        op = Transpose(a=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_one(self):
        with pytest.raises(ValueError, match="One operand"):
            Transpose(ScalarMatrix(3), ScalarMatrix(3))

    def test_constructor_rejects_no_operand(self):
        with pytest.raises(TypeError):
            Transpose()

    def test_positional_operand_names(self):
        op = Transpose(DiagonalMatrix(2))
        assert op.free_param_names == ["op_0/sigma^2_0", "op_0/sigma^2_1"]

    def test_call_scales_by_operand_values(self):
        op = Transpose(a=ScalarMatrix(2))
        free_params = torch.tensor([0.5])
        assert torch.allclose(op(free_params), math.exp(1) * torch.eye(2))

    def test_call_square_matches_operand(self):
        op = Transpose(a=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        assert torch.allclose(op(free_params), DiagonalMatrix(2)(free_params))

    def test_call_rectangular_fixed(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = Transpose(a=z)
        result = op(torch.tensor([]))
        assert result.shape == (3, 4)
        assert torch.equal(result, z(torch.tensor([])).T)

    def test_call_rectangular_trainable(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        operand = LowerTriangularMatrix(3, 2)
        assert torch.allclose(op(free_params), operand(free_params).T)

    def test_call_with_dict_params(self):
        op = Transpose(a=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        expected = op(free_params)
        result = op({"a/sigma^2_0": free_params[:1], "a/sigma^2_1": free_params[1:]})
        assert torch.allclose(result, expected)

    def test_shape_rectangular(self):
        op = Transpose(a=DummyMatrix(["a", "b", "c", "a"]))
        assert op(torch.tensor([])).shape == (3, 4)
        assert op.shape == (3, 4)

    def test_param_namespacing(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        assert op.num_free_params == 5
        assert op.free_param_names == ["a/L_0_0", "a/L_1_0", "a/L_1_1", "a/L_2_0", "a/L_2_1"]

    def test_param_namespacing_fixed_operand(self):
        op = Transpose(a=IdentityMatrix(3))
        assert op.num_free_params == 0
        assert op.free_param_names == []

    def test_inverse_matches_inverse_of_call(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_is_operand_inverse_transposed(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5])
        expected = LowerTriangularMatrix(2, 2).inverse(free_params).T
        assert torch.allclose(op.inverse(free_params), expected)

    def test_inverse_product_with_call_is_identity(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5])
        assert torch.allclose(op(free_params) @ op.inverse(free_params), torch.eye(2), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = Transpose(a=DiagonalMatrix(3))
        op.inverse(torch.tensor([0.0, 0.5, 1.0]))
        assert calls == []

    def test_inverse_nested_operand(self):
        op = Transpose(a=Product(a=DiagonalMatrix(2), b=ScalarMatrix(2)))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_rectangular_operand_raises(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        with pytest.raises(RuntimeError):
            op.inverse(free_params)

    def test_inverse_uses_default_free_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_validation_is_not_bypassed(self):
        op = Transpose(a=DiagonalMatrix(3))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.0, 0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = Transpose(a=ScalarMatrix(3))
        assert "Transpose" in repr(op)


class TestTransposeGrad:
    """Tests for Transpose gradient computation."""

    def test_manual_grad_shape(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (3, 3, 3)
        assert grad_names == ["a/sigma^2_0", "a/sigma^2_1", "a/sigma^2_2"]

    def test_manual_grad_equals_da_transpose(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        grad, grad_names = op.manual_grad(free_params)
        operand = LowerTriangularMatrix(3, 2)
        grad_operand, _ = operand.manual_grad(free_params)
        assert grad.shape == (5, 2, 3)
        assert torch.allclose(grad, grad_operand.mT)
        assert grad_names == ["a/L_0_0", "a/L_1_0", "a/L_1_1", "a/L_2_0", "a/L_2_1"]

    def test_manual_grad_vs_auto_grad(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.1])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_vs_auto_grad_rectangular(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        manual, _ = op.manual_grad(free_params)
        auto, _ = op.auto_grad(free_params)
        assert manual.shape == (5, 2, 3)
        assert torch.allclose(manual, auto)

    def test_all_fixed(self):
        op = Transpose(a=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_matches_manual_grad(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert torch.allclose(grad, manual)
        assert grad_names == manual_names == op.free_param_names


class TestTransposeCache:
    """Tests for Transpose intermediate caching."""

    def test_intermediate_cache_hit(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "v" in cache

    def test_reset_intermediates_clears_cache(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.reset_intermediates()
        assert op.get_intermediates(built) is None

    def test_disable_cache_clears_cache(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.disable_cache()
        assert op.get_intermediates(built) is None

    def test_cache_recomputes_on_new_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        first = torch.tensor([0.0, 0.5, 1.0])
        second = torch.tensor([1.0, 1.5, 2.0])
        v_first = op(first)
        v_second = op(second)
        assert not torch.allclose(v_first, v_second)
        assert torch.allclose(v_second, op(second))


class TestTransposeDtypeDevice:
    """Tests for dtype and device resolution for the operand."""

    def test_input_tensor_dtype_is_used(self):
        op = Transpose(a=ScalarMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_input_dict_dtype_is_used(self):
        op = Transpose(a=DiagonalMatrix(2))
        v = op(
            {
                "a/sigma^2_0": torch.tensor([0.5], dtype=torch.float64),
                "a/sigma^2_1": torch.tensor([0.5], dtype=torch.float64),
            }
        )
        assert v.dtype == torch.float64

    def test_input_device_is_used(self):
        op = Transpose(a=ScalarMatrix(3))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.tensor([0.5], device=device)).device.type == device

    def test_no_params_uses_torch_defaults(self):
        op = Transpose(a=IdentityMatrix(3))
        assert op.num_params == 0
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()


class TestTransposeTree:
    """Tests for Transpose integration with tree, free_param_tree and grad_tree."""

    def test_tree(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        results = op.tree(free_params)
        assert sorted(results) == ["/", "a"]
        assert torch.allclose(results["/"], op(free_params))

    def test_free_param_tree(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        free_params_by_path = op.free_param_tree(free_params)
        assert sorted(free_params_by_path) == ["/", "a"]
        assert free_params_by_path["/"] is free_params
        assert free_params_by_path["a"].shape == (3,)
        assert torch.equal(free_params_by_path["a"], free_params)

    def test_grad_tree(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        grads = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a"]
        manual, names = op.manual_grad(free_params)
        assert torch.allclose(grads["/"][0], manual)
        assert grads["/"][1] == names == op.free_param_names

    def test_nested_in_sum(self):
        op = Sum(transposed=Transpose(a=ScalarMatrix(2)), extra=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["transposed/a/sigma^2"]


class TestTransposeComposition:
    """Tests for Transpose in compositions with other operators."""

    def test_transpose_of_transpose_returns_operand(self):
        operand = LowerTriangularMatrix(3, 2)
        op = Transpose(a=Transpose(a=operand))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        assert torch.allclose(op(free_params), operand(free_params))

    def test_sandwich_equals_covariance_propagation(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        g = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])

        sandwich = Product(a=z, b=Product(a=g, b=Transpose(a=z)))
        propagation = CovariancePropagation(z=z, g=g)

        assert torch.allclose(sandwich(free_params), propagation(free_params))

        grad_sandwich, _ = sandwich.manual_grad(free_params)
        grad_propagation, _ = propagation.manual_grad(free_params)
        assert torch.allclose(grad_sandwich, grad_propagation)
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **`.T` forward, `.mT` in the gradient — deliberately different**: the forward pass transposes the operand with `.T` because a `Matrix` evaluation is 2D by contract; a 3D operand would be *fully* reversed rather than batched-transposed, which is what keeps this operator 2D-only. The gradient cannot use `.T`: `operands_grad` returns a 3D `(k, m, n)` Jacobian, and `.T` would reverse the parameter axis as well. `manual_grad` therefore uses `.mT` (last two dimensions), producing `(k, n, m)`. Getting this backwards yields a shape-valid but wrong gradient, so both directions are pinned by tests.
- **The single operand must be a `Matrix`**: `_check_operands` requires at least one `Matrix` operand, and with exactly one operand that means it is always a `Matrix`. Unlike `Product`, there is no fixed-tensor operand case — `Transpose(a=torch.eye(3))` raises `TypeError` from the base class. This is why the test file has no "fixed tensor operand" tests.
- **No shape compatibility rule to enforce**: a transpose has no pairing constraint, so unlike the product operators there is no runtime shape error to test. The only shape assertion is the output shape (and that `op.shape` updates).
- **Involution and nesting**: `Transpose(Transpose(A))` equals `A`, verified by value; the nested operand namespacing is `a/a/...`, which the test does not assert (it checks values only), so a future change to nesting names would not fail this test.
- **The sandwich equivalence is the operator's reason to exist**: `Product(Z, Product(G, Transpose(Z)))` reproduces `CovariancePropagation(z=Z, g=G)` in values and gradients. Note the parameter *names* differ between the two forms (`b/a/sigma^2_0` for the sandwich vs `g/sigma^2_0` for the propagation operator), so only values and gradient tensors are compared — a user switching forms gets equivalent numerics under different names.
- **Caching stores `a` and `v` but `manual_grad` does not use the cache**: there are no cross terms, so the gradient is just the operand's Jacobian transposed. The `"a"` entry keeps the cache shape consistent with the sibling operators and serves the forward/auto path.
- **The `inverse` override transposes with `.T`, not `.mT`**: the operand's `inverse` returns a 2D matrix, so a plain `.T` is correct; `.mT` (the batched last-two-axes transpose that `manual_grad` needs for its 3D Jacobian) would only be equivalent here by accident of dimensionality and would signal support for a batched operand that this operator does not have. The test pins `.T` by comparing against the operand's own inverse transposed with `.T`.
- **`inverse` inherits the operand's squareness requirement**: the override does no shape check of its own, so a rectangular operand (`LowerTriangularMatrix(3, 2)`) fails inside the operand's `inverse` with `RuntimeError: linalg.inv: A must be batches of square matrices...`. The test matches `RuntimeError` only, not the message wording.
- **The `inverse` path must not go through `__call__`**: a fallback that inverted `op(free_params)` would pass every value test, so the monkeypatched-`torch.linalg.inv` test is what pins the closed form; it asserts the dense inverse is not called at all.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor, positional names, `repr` | 5 |
| `__call__` values | 5 |
| Shape behavior | 1 |
| Parameter namespacing | 2 |
| `manual_grad` structure & values | 2 |
| `manual_grad` vs `auto_grad` | 2 |
| All-fixed | 1 |
| `grad()` consistency | 1 |
| Intermediate caching | 4 |
| Dtype / device | 4 |
| Tree integration | 4 |
| Composition | 2 |
| `inverse` override | 12 |
| **Total** | **45** |
