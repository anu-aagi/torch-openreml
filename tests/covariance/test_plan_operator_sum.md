# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_sum`
**Purpose of Module:**
Provides a `Sum` operator that additively combines multiple covariance matrices: `V = Σ A_i`. Each operand can be a parameterized `Matrix` or a fixed `torch.Tensor`. The operator namespaces parameters as `"operand_name/param_name"` and delegates gradient computation to each operand.

**Classes Covered:**

- `Sum` (extends `Operator`) — additive composite of covariance matrices

**Testing Goal:**
Ensure correct additive composition, correct parameter namespace management, correct per-operand gradient concatenation, and proper handling of mixed operand types (Matrix + fixed Tensor, free + fixed params).

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires ≥ 2 operands → `ValueError` otherwise
- Supports positional args (auto-named `op_0`, `op_1`)
- Supports keyword args (user-named)
- Supports single dict arg
- Rejects mixed positional + keyword
- Instance is a subclass of `Matrix` (via `Operator`)
- `_repr_single_line` is `False`

#### B. Parameter Namespacing

- Parameters named `"operand_name/param_name"`
- `param_specs` aggregates from all operands with namespace prefix
- `param_names` reflects namespaced names
- `free_param_names` / `fixed_param_names` correctly namespaced
- `num_params` / `num_free_params` sum correctly across operands

#### C. `__call__(free_params)` Behavior

- Returns sum of all operand matrices
- Correctly splits free_params across operands
- Handles fixed Tensor operands (included as-is)
- Result shape matches operand shapes (all must be identical)

#### D. `build_operands(free_params)`

- Returns list of operand matrices in operand order
- Fixed Tensors included as-is
- Each Matrix operand evaluated with its slice of free_params

#### E. `build_params(free_params)`

- Concatenates built params from all Matrix operands
- `out_format="dict"` returns namespaced dict
- `trans=False` suppresses transforms

#### F. `manual_grad(free_params)` Behavior

- Concatenates per-operand gradients (no cross-terms in sum)
- Fixed Tensor operands contribute nothing (filtered out)
- All-fixed → returns `(None, [])`
- Agrees with `auto_grad`

#### G. Mixed Operand Types

- Matrix + Matrix: both contribute params and grads
- Matrix + Tensor: Tensor treated as fixed
- Multiple Tensors + one Matrix: only Matrix contributes params
- All Tensors: at least one Matrix required (enforced by Operator)

#### H. Mixed Free/Fixed Within Operands

- Some operands have fixed params, some free
- Free param ordering follows operand order
- `free_param_index` correctly computed across operands

#### I. REML Interface

- `map_theta_to_v` returns sum
- `map_theta_to_dv` returns concatenated gradient

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use known simple operands (`ScalarMatrix`, `IdentityMatrix`) for easy verification
- Test with 2, 3, and more operands
- Compare `manual_grad` vs `auto_grad`
- Verify namespace prefixes in param_specs
- Test mixed Tensor + Matrix operands
- Verify single-param edge case (at least 2 operands required)

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import pytest
from torch_openreml.covariance import Sum, ScalarMatrix, IdentityMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2


class TestSum:
    """Tests for the Sum operator."""

    def test_constructor_positional(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        assert isinstance(op, Matrix)
        assert list(op.operands.keys()) == ["op_0", "op_1"]

    def test_constructor_keyword(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert list(op.operands.keys()) == ["a", "b"]

    def test_constructor_dict(self):
        op = Sum({"a": ScalarMatrix(3), "b": ScalarMatrix(3)})
        assert list(op.operands.keys()) == ["a", "b"]

    def test_constructor_requires_two(self):
        with pytest.raises(ValueError, match="At least two operands"):
            Sum(ScalarMatrix(3))

    def test_constructor_rejects_mixed_args_kwargs(self):
        with pytest.raises(ValueError):
            Sum(ScalarMatrix(3), b=ScalarMatrix(3))

    def test_param_namespacing(self):
        op = Sum(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert "a/sigma^2" in op.param_names
        assert op.num_params == 1  # only ScalarMatrix has params
        assert op.num_free_params == 1

    def test_two_scalar_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert op.num_params == 2
        assert op.num_free_params == 2
        assert sorted(op.param_names) == ["a/sigma^2", "b/sigma^2"]
        assert sorted(op.free_param_names) == ["a/sigma^2", "b/sigma^2"]

    def test_call_sum(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        result = op(free_params)
        # Each ScalarMatrix(0) → e^0 = 1 → I, so sum = 2*I
        expected = 2.0 * torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_unequal_params(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 1.0])
        result = op(free_params)
        import math
        # op_0: sigma^2 = e^0 = 1 → I
        # op_1: sigma^2 = e^2 → e^2 * I
        expected = torch.eye(3) + math.exp(2) * torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_with_tensor_operand(self):
        fixed = torch.ones(3, 3)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        free_params = torch.tensor([0.0])
        result = op(free_params)
        expected = torch.eye(3) + torch.ones(3, 3)
        assert torch.allclose(result, expected)

    def test_build_operands(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        v_groups = op.build_operands(free_params)
        assert len(v_groups) == 2
        assert v_groups[0].shape == (3, 3)
        assert v_groups[1].shape == (3, 3)

    def test_build_params(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        built = op.build_params(free_params)
        assert built.numel() == 2

    def test_build_params_dict_format(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        result = op.build_params(free_params, out_format="dict")
        assert "a/sigma^2" in result
        assert "b/sigma^2" in result

    def test_manual_grad_shape(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 3, 3)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_sum_of_grads(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, _ = op.manual_grad(free_params)
        # d(sum)/d(a_sigma^2) = d(A)/d(a_sigma^2) (same as scalar grad alone)
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.0]))
        assert torch.allclose(grad[0], grad_a[0])
        # d(sum)/d(b_sigma^2) = d(B)/d(b_sigma^2)
        b = ScalarMatrix(3)
        grad_b, _ = b.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad[1], grad_b[0])

    def test_manual_grad_with_tensor(self):
        fixed = torch.ones(3, 3)
        op = Sum(a=ScalarMatrix(3), fixed=fixed)
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["a/sigma^2"]

    def test_manual_grad_vs_auto_grad(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed_operands(self):
        op = Sum(IdentityMatrix(3), IdentityMatrix(3))
        assert op.num_free_params == 0
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_map_theta_to_v(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        op = Sum(a=ScalarMatrix(3), b=ScalarMatrix(3))
        r = repr(op)
        assert "Sum" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Parameter ordering**: `free_params` is concatenated in operand iteration order (dict insertion order, Python 3.7+). The order of operands in `build_operands`, `build_params`, and `operands_grad` must be consistent.
- **No cross-terms**: Since `V = Σ A_i`, the derivative `dV/dθ_j = dA_i/dθ_j` where θ_j belongs to operand i. There are no cross-terms. The `manual_grad` simply concatenates per-operand grads.
- **At least 2 operands**: `Sum` adds this constraint on top of `Operator`'s "at least one Matrix" rule. A sum of one matrix is semantically just that matrix.
- **Fixed Tensor operands**: Pure tensors are passed through in `build_operands` and contribute `None` in `operands_grad`. They are effectively fixed additive components.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor (positional/keyword/dict/errors) | 5 |
| Parameter namespacing | 2 |
| `__call__` | 3 |
| `build_operands` | 1 |
| `build_params` | 2 |
| `manual_grad` & `grad` | 5 |
| REML interface | 2 |
| repr | 1 |
| **Total** | **21** |
