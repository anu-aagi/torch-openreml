# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_stack`
**Purpose of Module:**
Provides a `Stack` operator: `V = [A_0 ; A_1 ; ...]`, the row-wise (vertical) concatenation of two or more operands. It is the exact dual of `Augment`, which concatenates side by side. Operands may have different numbers of rows — that is the point of the operator — but must agree in columns, since that is the dimension being concatenated.
**Classes Covered:**

- `Stack` (extends `Operator`) — row-wise stacking `V = [A_0 ; A_1 ; \ldots]`

**Testing Goal:**
Ensure correct row-wise stacking for square and rectangular operands, correct row-block gradient placement with zero fill outside each operand's rows, namespacing, dict/tensor input parity, dtype/device resolution, and equivalence with the `Transpose(Augment(Transpose, Transpose))` composition it replaces.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor accepts two or more operands positionally (`op_0`, `op_1`) or by keyword (`a`, `b`)
- `ValueError` (`"At least two operands are required"`) for a single operand
- Instance is a subclass of `Matrix` (via `Operator`)

#### B. `__call__(free_params)` Behavior

- Two `2 × 2` operands stack to `4 × 2`
- Values are exactly `torch.cat([...], dim=0)` of the operand evaluations — each operand's block lands in its own rows
- Rectangular operands with different row counts (`LowerTriangularMatrix(3, 2)` over `DiagonalMatrix(2)`) stack to `5 × 2`
- Dict parameter input gives the same result as the flat tensor input

#### C. Shape Behavior

- `op.shape` is updated after the call and equals `(sum of operand rows, common columns)`

#### D. Parameter Namespacing

- `a/sigma^2`, `b/sigma^2_0`, `b/sigma^2_1` in operand order
- Positional operands are prefixed `op_0/...`
- A fixed operand (tensor or `IdentityMatrix`) contributes no free parameters

#### E. Manual Gradient — Structure and Values

- `manual_grad` shape `(num_free_params, total_rows, cols)`
- Each parameter's block is placed in its operand's rows and zero elsewhere — asserted both positively (the block is nonzero) and negatively (rows outside the block are zero)
- Rectangular case: shape `(7, 5, 2)` for `LowerTriangularMatrix(3, 2)` + `DiagonalMatrix(2)`
- Names pass through unchanged from `operands_grad`

#### F. Gradient Consistency

- `grad()` agrees with `auto_grad` (`grad_mode = "auto"`) for a square-operand mix
- `manual_grad` agrees with `auto_grad` for the rectangular case
- `manual_grad` with no input uses defaults
- All-fixed operands → `(None, [])`

#### G. Intermediate Caching

- No dedicated cache tests. The protocol is identical to `Augment`'s by construction (same `build_params` → `get_intermediates` → `build_operands` → `set_intermediates` flow, keys `v_groups` / `v` / `row_offsets`), and dict-vs-tensor parity is exercised through `test_grad_dict_matches_tensor`, which reuses the same cache entry across the two input forms. See §4 for the residual risk.

#### H. Dtype and Device Contract

- Input parameter dtype wins (`float64` input → `float64` output and gradient)
- Input device wins (`cuda` → `mps` → skip)
- A fixed tensor operand is cast to follow the input device (never a resolution source)

#### I. Tree Integration

- No dedicated `tree` / `grad_tree` tests. Nested namespacing is covered one level deep by `test_nested_in_sum` (paths become `s/a/sigma^2`, …). See §4.

#### J. Composition

- Three operands stack to `6 × 2` with three independent parameter sets
- `Stack(A, B)` equals `Transpose(Augment(Transpose(A), Transpose(B)))` in values — the composition that previously had to be written by hand
- Nested under `Sum`, with the forward value asserted so the enclosing operator's shape contract is actually exercised

#### K. REML Interface

- The old `map_theta_to_v` / `map_theta_to_dv` methods were removed from the library; the estimator uses `__call__` and `grad()`. Coverage is through the `__call__` value tests, the `grad()`-vs-`auto_grad` tests, and the composition tests; no separate REML-interface tests exist.

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use small operands with values verifiable by hand, and compare gradients across the manual, auto, and `grad_mode` paths
- Compare against `torch.cat` of the operand evaluations rather than re-deriving transform arithmetic
- Use operands with *different row counts* (`LowerTriangularMatrix(3, 2)` over `DiagonalMatrix(2)`) so the row/column axis distinction is observable — square operands of equal size would hide a transposed block placement
- Include a fixed tensor operand (`torch.eye(2)`) and a fixed matrix operand (`IdentityMatrix(2)`) so both fixed paths are covered
- Follow the suite's dtype/device conventions: input dtype/device resolution, inline `cuda` → `mps` → `pytest.skip`

### 3.2 Test Structure

Each test follows **Arrange → Act → Assert**.

```python
import torch
import pytest
from torch_openreml.covariance import (
    Stack,
    Sum,
    Augment,
    Transpose,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    DummyMatrix,
    LowerTriangularMatrix,
)
from torch_openreml.covariance.matrix import Matrix


def stacked():
    return Stack(a=ScalarMatrix(2), b=DiagonalMatrix(2))


def as_dict(op, free_params):
    return {name: free_params[i:i + 1] for i, name in enumerate(op.free_param_names)}


class TestStack:
    """Tests for the Stack operator."""

    def test_constructor_positional(self):
        op = Stack(ScalarMatrix(2), DiagonalMatrix(2))
        assert isinstance(op, Matrix)
        assert list(op.operands.keys()) == ["op_0", "op_1"]

    def test_constructor_keyword(self):
        op = stacked()
        assert list(op.operands.keys()) == ["a", "b"]

    def test_constructor_requires_two_operands(self):
        with pytest.raises(ValueError, match="At least two operands"):
            Stack(a=ScalarMatrix(2))

    def test_param_names_namespaced(self):
        op = stacked()
        assert op.free_param_names == ["a/sigma^2", "b/sigma^2_0", "b/sigma^2_1"]

    def test_call_shape(self):
        op = stacked()
        assert op().shape == (4, 2)

    def test_call_stacks_operands_rowwise(self):
        op = stacked()
        free_params = torch.tensor([0.5, 0.0, 0.5])
        result = op(free_params)
        expected = torch.cat(
            [ScalarMatrix(2)(free_params[:1]), DiagonalMatrix(2)(free_params[1:])],
            dim=0,
        )
        assert result.shape == (4, 2)
        assert torch.allclose(result, expected)

    def test_call_with_dict_input(self):
        op = stacked()
        v = op(as_dict(op, torch.tensor([0.5, 0.5, 0.5])))
        assert v.shape == (4, 2)

    def test_call_follows_input_dtype(self):
        op = stacked()
        assert op(torch.full((3,), 0.5, dtype=torch.float64)).dtype == torch.float64

    def test_call_follows_input_device(self):
        op = stacked()
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.full((3,), 0.5, device=device)).device.type == device

    def test_tensor_operand_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        op = Stack(a=ScalarMatrix(2), fixed=torch.eye(2, device=device))
        assert op(torch.tensor([0.5])).device == torch.device("cpu")


class TestStackGrad:
    """Tests for Stack gradients, including dict and default input."""

    def test_grad_without_input(self):
        op = stacked()
        grad, names = op.grad()
        assert grad.shape == (op.num_free_params, 4, 2)
        assert names == op.free_param_names

    def test_grad_with_dict_input(self):
        op = stacked()
        free_params = torch.tensor([0.5, 0.5, 0.5])
        grad, names = op.grad(as_dict(op, free_params))
        assert grad.shape == (op.num_free_params, 4, 2)
        assert names == op.free_param_names

    def test_grad_dict_matches_tensor(self):
        op = stacked()
        free_params = torch.tensor([0.5, 0.5, 0.5])
        by_tensor, _ = op.grad(free_params)
        by_dict, _ = op.grad(as_dict(op, free_params))
        assert torch.allclose(by_tensor, by_dict)

    def test_grad_follows_input_dtype(self):
        op = stacked()
        grad, _ = op.grad(torch.full((3,), 0.5, dtype=torch.float64))
        assert grad.dtype == torch.float64

    def test_grad_matches_autograd(self):
        op = stacked()
        free_params = torch.tensor([0.7, 0.3, 1.4], dtype=torch.float64)
        manual, _ = op.grad(free_params)
        op.grad_mode = "auto"
        auto, _ = op.grad(free_params)
        assert torch.allclose(manual, auto)

    def test_grad_places_operand_blocks_in_rows(self):
        op = stacked()
        grad, _ = op.grad(torch.tensor([0.5, 0.5, 0.5]))
        assert torch.count_nonzero(grad[0, :2, :]) > 0
        assert torch.count_nonzero(grad[0, 2:, :]) == 0
        assert torch.count_nonzero(grad[1, :2, :]) == 0
        assert torch.count_nonzero(grad[2, :2, :]) == 0

    def test_manual_grad_vs_auto_grad_rectangular(self):
        op = Stack(a=LowerTriangularMatrix(3, 2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        manual, manual_names = op.manual_grad(free_params)
        auto, auto_names = op.auto_grad(free_params)
        assert op(free_params).shape == (5, 2)
        assert manual.shape == (7, 5, 2)
        assert manual_names == auto_names
        assert torch.allclose(manual, auto)

    def test_grad_with_tensor_operand(self):
        op = Stack(a=ScalarMatrix(2), fixed=torch.eye(2))
        grad, names = op.grad(torch.tensor([0.5]))
        assert grad.shape == (1, 4, 2)
        assert names == ["a/sigma^2"]

    def test_grad_with_fixed_matrix_operand(self):
        op = Stack(a=ScalarMatrix(2), b=IdentityMatrix(2))
        grad, names = op.grad(torch.tensor([0.5]))
        assert names == ["a/sigma^2"]

    def test_all_fixed(self):
        op = Stack(a=IdentityMatrix(2), b=IdentityMatrix(2))
        grad, names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert names == []


class TestStackComposition:
    """Tests for Stack in compositions with other operators."""

    def test_three_operands(self):
        op = Stack(a=ScalarMatrix(2), b=ScalarMatrix(2), c=ScalarMatrix(2))
        result = op(torch.tensor([0.1, 0.2, 0.3]))
        assert result.shape == (6, 2)
        assert op.free_param_names == ["a/sigma^2", "b/sigma^2", "c/sigma^2"]

    def test_equals_transpose_augment_composition(self):
        free_params = torch.tensor([0.0, 0.5, 0.7])
        stacked_op = Stack(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        composed = Transpose(
            a=Augment(Transpose(a=ScalarMatrix(2)), Transpose(a=DiagonalMatrix(2)))
        )
        assert torch.allclose(stacked_op(free_params), composed(free_params))

    def test_nested_in_sum(self):
        op = Sum(
            s=Stack(a=ScalarMatrix(2), b=DiagonalMatrix(2)),
            d=DummyMatrix(["a", "b", "a", "b"]),
        )
        free_params = torch.tensor([0.5, 0.1, 0.2])
        grad, grad_names = op.manual_grad(free_params)
        assert op(free_params).shape == (4, 2)
        assert grad.shape == (3, 4, 2)
        assert grad_names == ["s/a/sigma^2", "s/b/sigma^2_0", "s/b/sigma^2_1"]
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **The axis-keeping trap versus `Augment`**: `Stack` is a near-copy of `Augment` with rows and columns swapped, and the two lines that differ are the ones that silently corrupt a gradient if mixed up. In `_get_or_build_intermediates`, `Augment` reads `vg.shape[1]` and stacks `col_offsets`; `Stack` reads `vg.shape[0]` and stacks `row_offsets`. In `manual_grad`, the buffer is `torch.zeros(k, total_rows, grad.shape[2])` with `tmp[:, r0:r1, :]` — note it *keeps* `grad.shape[2]` and replaces the middle axis, where `Augment` keeps `grad.shape[1]` and replaces the last. For an equal-sized square operand a wrong axis would still pass value tests in some orderings, so the rectangular test (`3 × 2` over `2 × 2`) is load-bearing: it is the only test where a transposed block placement would fail loudly. The row-block placement test is also asserted in both directions (nonzero inside the block, zero outside), so an all-zero gradient cannot pass it vacuously.
- **No shape validation**: the constructor accepts operands whose column counts disagree, and the mismatch only surfaces at `__call__` as `RuntimeError: Sizes of tensors must match except in dimension 0` from `torch.cat`. There is no test pinning that error, because the "no extra checks" convention makes the exact message torch's to change. The same convention means a `Stack` inside a `Sum` will produce gradients even when the enclosing shapes are inconsistent; `test_nested_in_sum` therefore asserts `op(free_params).shape` so the forward path is actually exercised. (This was caught while writing the tests: a first draft used `IdentityMatrix(4)` against a `4 × 2` stack, and `manual_grad` returned a valid-looking gradient while `__call__` failed.)
- **`cat`, not `torch.stack`**: the name matches PyTorch's `torch.cat(dim=0)` semantics — the output has `sum(rows_i)` rows and the operands keep their own dimensionality. A reader expecting `torch.stack`'s new-axis behaviour (`(n, m, k)`) will be surprised; the class docstring's `bmatrix` display is the disambiguator, and `test_call_shape` pins the concatenation contract.
- **The equivalence test compares values only**: `Stack(A, B)` and `Transpose(Augment(Transpose(A), Transpose(B)))` produce the same matrix but different parameter *names* (one namespacing level versus three) and the test uses fresh operand instances for each form, so it cannot detect a shared-instance parameter bug. Only the values are contractually identical; names are not.
- **No dedicated cache or tree tests**: `tree` / `grad_tree` paths are not asserted for this operator (unlike `Transpose`), and intermediate caching is covered only indirectly through dict/tensor reuse. The caching implementation is a copy of `Augment`'s against the same protocol, so a regression there would most likely show up in the sibling suites first; the module-specific caching risk — a stale `row_offsets` reused after a change in operand row counts — cannot arise, because operand shapes are fixed at construction.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor (positional, keyword, arity) | 3 |
| Namespacing, shape, row-wise values | 3 |
| Dict input, dtype, device, fixed-tensor device | 4 |
| `grad` default input, dict input, dict == tensor, dtype, autograd agreement | 5 |
| Gradient row-block placement | 1 |
| Rectangular `manual_grad` vs `auto_grad` | 1 |
| Fixed tensor operand, fixed matrix operand, all-fixed | 3 |
| Composition (three operands, equivalence, nested in `Sum`) | 3 |
| **Total** | **23** |
