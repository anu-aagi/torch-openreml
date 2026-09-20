# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_augment`
**Purpose of Module:**
Defines `Augment`, an `Operator` that places its operands side by side column-wise, :math:`V = [A_0 \mid A_1 \mid \ldots]`. It owns the column layout of the composite, caches it as `col_offsets`, and implements `manual_grad` by placing each operand's block in its own columns.

**Classes Covered:**

- `Augment` (extends `Operator`) — column-wise augmentation of two or more operands

**Testing Goal:**
Ensure the constructor enforces the two-operand minimum, that the column layout, shape, and caching are correct, that `manual_grad` agrees with `auto_grad` and places each operand's block in the right columns, and that the operator follows the dtype and device of its input parameters.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Constructor

- Positional operands → auto-named `op_0`, `op_1`, ...
- Keyword operands → user-defined names
- Fewer than two operands → `ValueError`
- Inherits operand validation from `Operator` (non-dict, non-string keys, `"/"` in keys, non-Matrix/non-Tensor values, at least one Matrix)

#### B. Parameter Namespacing & Layout

- `free_param_names` namespaced `"operand/param"`, in operand order
- Operands may have differing numbers of columns and differing parameter counts
- Fixed `torch.Tensor` and fixed-parameter `Matrix` operands contribute no parameters

#### C. `__call__`

- Shape is `(rows, sum of operand columns)`
- Column layout matches operand order
- Accepts dict input equivalent to tensor input
- Repeated calls reuse the cached intermediates
- `_shape` is set from the result

#### D. Dtype & Device

- Input tensor dtype and device are used for the whole composite
- Dict input dtype and device are used
- Without input, resolved from the operands' free-param defaults
- Tensor operand is cast to the input dtype and device

#### E. `manual_grad`

- Shape is `(num_free_params, rows, total_cols)`
- `grad_names` matches `free_param_names`
- Each operand's block lands in its own columns; other columns are zero
- Agrees with `auto_grad`
- Follows the input dtype
- Accepts dict input, agreeing with tensor input
- Tensor and fixed-parameter `Matrix` operands contribute no rows
- Returns `(None, [])` when no operand has a free parameter

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use `Augment(a=ScalarMatrix(2), b=DiagonalMatrix(2))` as the standard vehicle: two operands, three free parameters, shapes `(2, 2)` and `(2, 4)`.
- Build dict input from `free_param_names` with a helper.
- Verify block placement by counting non-zeros outside an operand's column range.
- Compare `manual_grad` against `auto_grad` in `float64`.

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

```python
import torch
import pytest
from torch_openreml.covariance import Augment, ScalarMatrix, DiagonalMatrix, IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


def augmented():
    return Augment(a=ScalarMatrix(2), b=DiagonalMatrix(2))


def as_dict(op, free_params):
    return {name: free_params[i:i + 1] for i, name in enumerate(op.free_param_names)}


class TestAugment:
    """Tests for the Augment operator."""

    def test_constructor_requires_two_operands(self):
        with pytest.raises(ValueError, match="At least two operands"):
            Augment(a=ScalarMatrix(2))

    def test_call_shape(self):
        assert augmented()().shape == (2, 4)

    def test_call_with_dict_input(self):
        op = augmented()
        assert op(as_dict(op, torch.tensor([0.5, 0.5, 0.5]))).shape == (2, 4)

    def test_call_follows_input_dtype(self):
        op = augmented()
        assert op(torch.full((3,), 0.5, dtype=torch.float64)).dtype == torch.float64


class TestAugmentGrad:
    """Tests for Augment gradients."""

    def test_grad_matches_autograd(self):
        op = augmented()
        free_params = torch.tensor([0.7, 0.3, 1.4], dtype=torch.float64)
        manual, _ = op.grad(free_params)
        op.grad_mode = "auto"
        auto, _ = op.grad(free_params)
        assert torch.allclose(manual, auto)

    def test_grad_places_operand_blocks_in_columns(self):
        op = augmented()
        grad, _ = op.grad(torch.tensor([0.5, 0.5, 0.5]))
        assert torch.count_nonzero(grad[0, :, 2:]) == 0
        assert torch.count_nonzero(grad[1, :, :2]) == 0
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Column layout is the contract**: `manual_grad` and `__call__` both depend on `col_offsets` from `_get_or_build_intermediates`, so the two stay consistent by construction. A test that only checks the gradient's shape would not catch a wrong column assignment — check block placement.
- **Equal row counts**: `torch.cat(..., dim=1)` requires all operands to share their row count. A mismatch surfaces as a Torch error, not a validated `ValueError`.
- **Zero blocks are dropped, not zero-filled**: operands with no free parameters (`IdentityMatrix`, or a fixed tensor) contribute no rows to the gradient. With no trainable operand at all, `manual_grad` returns `(None, [])`.
- **Caching and accelerator devices**: intermediates are keyed on a hash of the built parameters, which is unavailable on MPS, so device tests on that backend may `xfail`.
- **Dtype and device resolution** is inherited from `Operator.build_operands`; see `test_plan_operator.md`.
