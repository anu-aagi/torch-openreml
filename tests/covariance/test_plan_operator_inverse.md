# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_inverse`
**Purpose of Module:**
Provides an `Inverse` operator: `V = A⁻¹` for a square operand. It is the structural counterpart of the estimator's precision-based work — where a model is parameterised through a precision (or a whitening) matrix rather than a covariance, the marginal covariance is the inverse of a trainable operand. Before this operator, the only inverse in the library was implicit inside `CovariancePropagation`'s gradient, and there was no composable way to express `A⁻¹` as a node.
**Classes Covered:**

- `Inverse` (extends `Operator`) — matrix inverse `V = A^{-1}`

**Testing Goal:**
Ensure correct inverse evaluation for square operands of several matrix types, correct Jacobian (`−A⁻¹ dA A⁻¹`, reused from the forward pass), namespacing pass-through, intermediate caching, dtype/device resolution, tree integration, and the algebraic identities (`A⁻¹⁻¹ = A`, `(Aᵀ)⁻¹ = (A⁻¹)ᵀ`, `(AB)⁻¹ = B⁻¹A⁻¹`, `A⁻¹A = I`) in both values and gradients.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor accepts exactly 1 operand → `ValueError` (`"One operand is required"`) for 2 operands
- No operand at all raises the base-class `TypeError` (no `Matrix` operand present)
- A fixed tensor operand (`torch.eye(3)`) raises `TypeError` — with one operand, the base-class rule forces it to be a `Matrix`
- Positional operand is named `op_0`
- Instance is a subclass of `Matrix` (via `Operator`)
- `repr` contains the class name

#### B. `__call__(free_params)` Behavior

- Transform anchor: `ScalarMatrix(2)` at `θ = 0.5` inverts to `e⁻¹ · I`
- Square operand (`DiagonalMatrix`) matches `torch.linalg.inv` of the operand's own evaluation
- Non-diagonal trainable operand (`UnconstrainedMatrix`) matches `torch.linalg.inv`
- Dict parameter input gives the same result as the flat tensor input

#### C. Shape Behavior

- `op.shape` is updated after the call and equals the operand's shape
- A **non-square** operand (fixed `DummyMatrix`, 4 × 3) raises `RuntimeError` matching `"square"` at call time — there is no construction-time check

#### D. Parameter Namespacing

- `a/sigma^2_0`, `a/sigma^2_1`, ... in operand order
- A fixed operand (`IdentityMatrix`) contributes no free parameters (`num_free_params == 0`)
- Positional operand prefix `op_0/...`

#### E. Manual Gradient — Structure and Values

- `manual_grad` shape `(num_free_params, n, n)`
- `dV/dθ = −A⁻¹ (dA/dθ) A⁻¹`, checked against the operand's own Jacobian and its own inverse
- Names pass through unchanged from `operands_grad`

#### F. Gradient Consistency

- `manual_grad` vs `auto_grad` for a non-symmetric trainable operand
- All-fixed operand → `(None, [])`
- `grad()` agrees with `manual_grad`, names equal `free_param_names`

#### G. Intermediate Caching

- Cache is populated under `"a"` and `"v"` after a call, keyed by built parameters
- `reset_intermediates` and `disable_cache` clear it
- A new parameter tensor recomputes (no stale reuse), and the new value equals `torch.linalg.inv` at the new parameters
- The gradient reads `"v"` from this cache — the forward pass's inverse, not a second inversion

#### H. Dtype and Device Contract

- Input parameter dtype wins (flat tensor and dict input)
- Input device wins (`cuda` → `mps` → skip); `torch.linalg.inv` is verified to run on MPS for `float32`
- With no parameters at all, Torch defaults are used

#### I. Tree Integration

- `tree` paths `"/"` and `"a"`, root equal to `__call__`
- `free_param_tree` paths `"/"` and `"a"`, root the input tensor, `"a"` its own slice
- `grad_tree` root matches `manual_grad`
- Nested under `Sum`, namespacing becomes `inverted/a/sigma^2`

#### J. Composition

- `Inverse(Inverse(A))` returns `A` (involution)
- `Inverse(Transpose(A))` equals `Transpose(Inverse(A))` in values **and** gradients
- `Inverse(A @ B)` equals `Inverse(B) @ Inverse(A)`, with each side fed its own parameter ordering
- `Inverse(A) @ A` is the identity

#### K. REML Interface

- The old `map_theta_to_v` / `map_theta_to_dv` methods were removed from the library; the estimator uses `__call__` and `grad()`. Coverage is through the `__call__` value tests, the `grad()`-vs-`manual_grad` test, and the composition tests; no separate REML-interface tests exist.

#### L. `inverse(free_params)`

`Inverse` overrides the base default with the closed form `V⁻¹ = A`: the operand is built once and returned
as it is, so the dense inversion the base default performs is skipped entirely. The override must be
indistinguishable from the default it replaces, in both value and argument handling.

- Agrees with `torch.linalg.inv(op(free_params))`
- Returns the operand matrix itself, equal to `DiagonalMatrix(3)(free_params)` rather than a re-inverted copy
- `op(free_params) @ op.inverse(free_params)` is the identity
- The composite is never inverted densely: `torch.linalg.inv` is not called
- A nested operand (`Product`) is handled the same way, by value
- The argument contract is not bypassed: `None` means the defaults, a dict of free parameters means the
  same as the flat tensor, and a non-tensor, a wrong-length vector and a bad dict each raise
- dtype follows the input, the result stays attached to `free_params`, and its gradient equals the gradient
  of `torch.linalg.inv(op(free_params))`
- A non-square operand is not checked — the operand is returned as the inverse, which the docstring warns
  about, so the shape of the returned tensor is pinned rather than an exception

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Compare values against `torch.linalg.inv` of the operand's own evaluation rather than re-deriving transform arithmetic
- Use **non-symmetric** trainable operands (`LowerTriangularMatrix(2, 2)`) wherever gradient *ordering* matters: with a symmetric operand and symmetric Jacobian, `−V dA V` is symmetric under any reordering, so a transposed or swapped product would pass unnoticed
- Use a rectangular fixed operand (`DummyMatrix`) for the non-square error, since a square operand can never exercise it
- Keep anchor values hand-checkable: `ScalarMatrix(2)` at `θ = 0.5` gives `e⁻¹ ≈ 0.368`
- Follow the suite's dtype/device conventions: input dtype/device resolution, inline `cuda` → `mps` → `pytest.skip`; MPS runs `linalg.inv` for `float32` (no float64 on MPS)

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**.

```python
import torch
import math
import pytest
from torch_openreml.covariance import (
    Inverse,
    Sum,
    Product,
    Transpose,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    DummyMatrix,
    UnconstrainedMatrix,
    LowerTriangularMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestInverse:
    """Tests for the Inverse operator."""

    def test_constructor(self):
        op = Inverse(a=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_one(self):
        with pytest.raises(ValueError, match="One operand"):
            Inverse(ScalarMatrix(3), ScalarMatrix(3))

    def test_constructor_rejects_no_operand(self):
        with pytest.raises(TypeError):
            Inverse()

    def test_constructor_rejects_tensor_operand(self):
        with pytest.raises(TypeError):
            Inverse(a=torch.eye(3))

    def test_positional_operand_names(self):
        op = Inverse(DiagonalMatrix(2))
        assert op.free_param_names == ["op_0/sigma^2_0", "op_0/sigma^2_1"]

    def test_call_scalar_inverts_operand(self):
        op = Inverse(a=ScalarMatrix(2))
        free_params = torch.tensor([0.5])
        assert torch.allclose(op(free_params), math.exp(-1) * torch.eye(2))

    def test_call_diagonal_matches_linalg_inv(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op(free_params), torch.linalg.inv(DiagonalMatrix(3)(free_params)))

    def test_call_unconstrained_matches_linalg_inv(self):
        operand = UnconstrainedMatrix(2)
        op = Inverse(a=UnconstrainedMatrix(2))
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(op(free_params), torch.linalg.inv(operand(free_params)))

    def test_call_with_dict_params(self):
        op = Inverse(a=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        expected = op(free_params)
        result = op({"a/sigma^2_0": free_params[:1], "a/sigma^2_1": free_params[1:]})
        assert torch.allclose(result, expected)

    def test_call_non_square_raises(self):
        op = Inverse(a=DummyMatrix(["a", "b", "c", "a"]))
        with pytest.raises(RuntimeError, match="square"):
            op(torch.tensor([]))

    def test_shape(self):
        op = Inverse(a=DiagonalMatrix(3))
        assert op(torch.tensor([0.0, 0.5, 1.0])).shape == (3, 3)
        assert op.shape == (3, 3)

    def test_param_namespacing(self):
        op = Inverse(a=DiagonalMatrix(2))
        assert op.num_free_params == 2
        assert op.free_param_names == ["a/sigma^2_0", "a/sigma^2_1"]

    def test_param_namespacing_fixed_operand(self):
        op = Inverse(a=IdentityMatrix(3))
        assert op.num_free_params == 0
        assert op.free_param_names == []

    def test_inverse_matches_inverse_of_call(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_returns_operand(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), DiagonalMatrix(3)(free_params))

    def test_inverse_product_with_call_is_identity(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op(free_params) @ op.inverse(free_params), torch.eye(3), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = Inverse(a=DiagonalMatrix(3))
        op.inverse(torch.tensor([0.0, 0.5, 1.0]))
        assert calls == []

    def test_inverse_nested_operand(self):
        op = Inverse(a=Product(a=DiagonalMatrix(2), b=ScalarMatrix(2)))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_uses_default_free_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_rectangular_operand_is_not_checked(self):
        op = Inverse(a=DummyMatrix(["a", "b", "c", "a"]))
        assert op.inverse(torch.tensor([])).shape == (4, 3)

    def test_inverse_validation_is_not_bypassed(self):
        op = Inverse(a=DiagonalMatrix(3))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.0, 0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = Inverse(a=ScalarMatrix(3))
        assert "Inverse" in repr(op)


class TestInverseGrad:
    """Tests for Inverse gradient computation."""

    def test_manual_grad_shape(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (3, 3, 3)
        assert grad_names == ["a/sigma^2_0", "a/sigma^2_1", "a/sigma^2_2"]

    def test_manual_grad_equals_neg_v_da_v(self):
        operand = LowerTriangularMatrix(2, 2)
        op = Inverse(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.2, -0.3, 0.5])
        grad, _ = op.manual_grad(free_params)
        da, _ = operand.manual_grad(free_params)
        v = torch.linalg.inv(operand(free_params))
        assert torch.allclose(grad, -v @ da @ v)

    def test_manual_grad_vs_auto_grad(self):
        op = Inverse(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.2, -0.3, 0.5])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert manual.shape == (3, 2, 2)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = Inverse(a=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_matches_manual_grad(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert torch.allclose(grad, manual)
        assert grad_names == manual_names == op.free_param_names


class TestInverseCache:
    """Tests for Inverse intermediate caching."""

    def test_intermediate_cache_hit(self):
        op = Inverse(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "v" in cache

    def test_reset_intermediates_clears_cache(self):
        op = Inverse(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.reset_intermediates()
        assert op.get_intermediates(built) is None

    def test_disable_cache_clears_cache(self):
        op = Inverse(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.disable_cache()
        assert op.get_intermediates(built) is None

    def test_cache_recomputes_on_new_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        first = torch.tensor([0.0, 0.5, 1.0])
        second = torch.tensor([1.0, 1.5, 2.0])
        v_first = op(first)
        v_second = op(second)
        assert not torch.allclose(v_first, v_second)
        assert torch.allclose(v_second, torch.linalg.inv(DiagonalMatrix(3)(second)))


class TestInverseDtypeDevice:
    """Tests for dtype and device resolution for the operand."""

    def test_input_tensor_dtype_is_used(self):
        op = Inverse(a=ScalarMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_input_dict_dtype_is_used(self):
        op = Inverse(a=DiagonalMatrix(2))
        v = op(
            {
                "a/sigma^2_0": torch.tensor([0.5], dtype=torch.float64),
                "a/sigma^2_1": torch.tensor([0.5], dtype=torch.float64),
            }
        )
        assert v.dtype == torch.float64

    def test_input_device_is_used(self):
        op = Inverse(a=ScalarMatrix(3))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.tensor([0.5], device=device)).device.type == device

    def test_no_params_uses_torch_defaults(self):
        op = Inverse(a=IdentityMatrix(3))
        assert op.num_params == 0
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()


class TestInverseTree:
    """Tests for Inverse integration with tree, free_param_tree and grad_tree."""

    def test_tree(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        results = op.tree(free_params)
        assert sorted(results) == ["/", "a"]
        assert torch.allclose(results["/"], op(free_params))

    def test_free_param_tree(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        free_params_by_path = op.free_param_tree(free_params)
        assert sorted(free_params_by_path) == ["/", "a"]
        assert free_params_by_path["/"] is free_params
        assert free_params_by_path["a"].shape == (3,)
        assert torch.equal(free_params_by_path["a"], free_params)

    def test_grad_tree(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        grads = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a"]
        manual, names = op.manual_grad(free_params)
        assert torch.allclose(grads["/"][0], manual)
        assert grads["/"][1] == names == op.free_param_names

    def test_nested_in_sum(self):
        op = Sum(inverted=Inverse(a=ScalarMatrix(2)), extra=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["inverted/a/sigma^2"]


class TestInverseComposition:
    """Tests for Inverse in compositions with other operators."""

    def test_inverse_of_inverse_returns_operand(self):
        operand = UnconstrainedMatrix(2)
        op = Inverse(a=Inverse(a=UnconstrainedMatrix(2)))
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(op(free_params), operand(free_params))

    def test_inverse_commutes_with_transpose(self):
        nested = Inverse(a=Transpose(a=LowerTriangularMatrix(2, 2)))
        commuted = Transpose(a=Inverse(a=LowerTriangularMatrix(2, 2)))
        free_params = torch.tensor([0.2, -0.3, 0.5])
        assert torch.allclose(nested(free_params), commuted(free_params))
        grad_nested, names_nested = nested.manual_grad(free_params)
        grad_commuted, names_commuted = commuted.manual_grad(free_params)
        assert names_nested == names_commuted
        assert torch.allclose(grad_nested, grad_commuted)

    def test_inverse_of_product_is_reversed_product(self):
        lhs = Inverse(a=Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2)))
        rhs = Product(
            a=Inverse(a=LowerTriangularMatrix(2, 2)), b=Inverse(a=DiagonalMatrix(2))
        )
        diag_params = torch.tensor([0.0, 0.5])
        lower_params = torch.tensor([0.2, -0.3, 0.4])
        assert torch.allclose(
            lhs(torch.cat([diag_params, lower_params])),
            rhs(torch.cat([lower_params, diag_params])),
        )

    def test_product_with_inverse_is_identity(self):
        op = Product(a=Inverse(a=DiagonalMatrix(3)), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        assert torch.allclose(op(free_params), torch.eye(3), atol=1e-6)
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Sign and ordering of the gradient**: `dV/dθ = −A⁻¹ dA A⁻¹`, not `+`, and the two factors must not be swapped. The explicit-formula test and the manual-vs-auto test both use a **non-symmetric** operand (`LowerTriangularMatrix(2, 2)`) precisely because with a symmetric `A` and symmetric `dA`, `V dA V` is symmetric under reordering (`(V dA V)ᵀ = V dA V` when `V` and `dA` are symmetric), so a transposed or swapped product would pass a symmetric-operand test. `test_inverse_commutes_with_transpose` is the complementary check: `(Aᵀ)⁻¹ = (A⁻¹)ᵀ` holds in values *and* gradients only if both sides are built from the same inverse.
- **Singular and ill-conditioned operands have no safety net**: `torch.linalg.inv` raises for exactly singular input, but silently returns garbage for merely ill-conditioned input, and this operator is designed to sit inside a REML optimizer that will steer parameters wherever the likelihood points. `UnconstrainedMatrix` is already documented as symmetric-but-not-guaranteed-PD, so `Inverse` follows the same convention: it does not check definiteness. NaNs produced here propagate into `manual_grad` rather than raising.
- **Non-square operand raises at call time, not construction**: there is no shape check (`Inverse(a=DummyMatrix(...))` constructs fine), and the failure is `RuntimeError: linalg.inv: A must be batches of square matrices...`. The test matches only `"square"` so it does not pin torch's exact wording.
- **The gradient reads the forward pass's cached inverse**: `manual_grad` uses `cache["v"]`, so `manual_grad` after `__call__` performs no second inversion. This is also a hazard: `__call__` returns the cached tensor itself, so a caller that mutates the returned matrix in place corrupts the value subsequent gradients are computed against. The same is true of every operator in the family; no test pins the aliasing, and none should be written to rely on a copy.
- **Composition tests must feed each side its own parameter ordering**: in `Inverse(A @ B)` versus `Inverse(B) @ Inverse(A)`, the `diag` and `lower` parameters are namespaced in opposite orders, so a single shared parameter tensor would silently evaluate the wrong point. The test constructs `torch.cat([diag, lower])` and `torch.cat([lower, diag])` explicitly for this reason; the same trap applies to any future composition test that compares two differently-ordered parameterisations.
- **MPS**: `torch.linalg.inv` runs on MPS for `float32` (verified on this machine, and exercised by `test_input_device_is_used` when MPS is the available accelerator). MPS has no float64, so the float64 dtype tests and the device test are necessarily separate.
- **The `inverse` override returns the operand unchecked**: `V⁻¹ = A` needs no inversion, so the override never calls `torch.linalg.inv` — the monkeypatched-`torch.linalg.inv` test pins that, since a fallback that inverted `__call__` would agree in value and pass every value test. The same shortcut means a non-square operand is returned as the inverse of a composite that has none: unlike `__call__`, `inverse` raises nothing, which the docstring's warning states and `test_inverse_rectangular_operand_is_not_checked` pins deliberately, so the behavior is not mistaken for a bug later.
- **No dedicated check that the inverse is computed once**: the "reuse the forward's inverse" property is an implementation detail, asserted only indirectly through the cache tests (`"v"` present after a call).

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor (arity, no operand, tensor operand), positional names, `repr` | 6 |
| `__call__` values (scalar, diagonal, unconstrained, dict) | 4 |
| Non-square raises | 1 |
| Shape behavior | 1 |
| Parameter namespacing (trainable, fixed) | 2 |
| `manual_grad` shape & names | 1 |
| `manual_grad` equals `−V dA V` | 1 |
| `manual_grad` vs `auto_grad` | 1 |
| All-fixed | 1 |
| `grad()` consistency | 1 |
| Intermediate caching | 4 |
| Dtype / device | 4 |
| Tree integration | 4 |
| Composition | 4 |
| `inverse` override | 12 |
| **Total** | **47** |
