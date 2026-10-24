# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.covariance.operator_block_diagonal`
**Purpose of Module:**
Provides a `BlockDiagonal` operator that arranges covariance matrices as blocks along the diagonal: `V = blockdiag(V₀, V₁, ...)`. Unlike `Sum`, operands can have different shapes. Off-diagonal block regions are all zeros. Parameter namespacing follows the standard `Operator` convention.

**Classes Covered:**

- `BlockDiagonal` (extends `Operator`) — block diagonal composite

**Testing Goal:**
Ensure correct block diagonal construction with operands of different sizes, correct gradient placement into the corresponding block regions, correct intermediate caching including row/col offsets, and proper zero-filling of non-block regions.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Class-Level Behavior

- Constructor requires ≥ 2 operands → `ValueError`
- Supports positional, keyword, and dict operand specification
- Instance is a subclass of `Matrix` (via `Operator`)
- Operands can have different shapes

#### B. `__call__(free_params)` Behavior

- Returns `torch.block_diag` of operand matrices
- Same-size operands: equivalent to placing each on diagonal
- Different-size operands: output shape is sum of individual dimensions
- Fixed Tensor operands included as-is
- Off-diagonal blocks are zero
- Intermediate caching works

#### C. Block Placement

- Block 0 occupies `[0:rows₀, 0:cols₀]`
- Block 1 occupies `[rows₀:rows₀+rows₁, cols₀:cols₀+cols₁]`
- Row and col offsets computed correctly for 2, 3 blocks
- Non-square operands handled correctly

#### D. `manual_grad(free_params)` Behavior

- Per-operand gradients padded with zeros to full output shape
- Each operand's gradient placed in correct block region
- Shape: `(num_free_params, total_rows, total_cols)`
- Fixed Tensor operands contribute nothing
- All-fixed → returns `(None, [])`
- Agrees with `auto_grad`

#### E. Parameter Namespacing

- Standard `"operand_name/param_name"` format
- Aggregated across all Matrix operands

#### F. Mixed Operands

- Mix of different Matrix types (ScalarMatrix, DiagonalMatrix)
- Mix of Matrix + fixed Tensor
- Same-size vs different-size operands

#### G. Intermediate Caching

- Caches v_groups, v, row_offsets, col_offsets
- Cache hit on second call
- Cache invalidated by `reset_intermediates`

#### H. REML Interface

- `map_theta_to_v` returns block diagonal
- `map_theta_to_dv` returns padded gradient

#### I. `inverse(free_params)`

`BlockDiagonal` overrides the base default with the closed form `V⁻¹ = blockdiag(V₀⁻¹, V₁⁻¹, …)`: each
operand is inverted through its own path and the results are reassembled block diagonally, instead of
inverting the whole composite densely.

- Agrees with `torch.linalg.inv(op(free_params))`
- Is exactly `torch.block_diag` of the operand inverses, in operand order
- `op.inverse(free_params) @ op(free_params)` is the identity, and the off-block entries of the inverse are
  exactly zero
- Three operands and a nested operand (`BlockDiagonal` inside `BlockDiagonal`) are handled the same way
- A fixed tensor operand is inverted directly
- The composite is never inverted densely: `torch.linalg.inv` is not called
- A rectangular operand raises (`RuntimeError` from the operand's own inverse)
- The argument contract is not bypassed: `None` means the defaults, a dict means the same as the flat
  tensor, and a non-tensor, a wrong-length vector and a bad dict each raise
- dtype follows the input, the result stays attached to `free_params`, and its gradient equals the gradient
  of `torch.linalg.inv(op(free_params))`

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Use operands of different sizes as the primary test (core differentiator from Sum)
- Also test same-size operands (degenerate case)
- Compare `manual_grad` vs `auto_grad`
- Verify gradient zero in non-block positions
- Verify gradient equals operand's standalone grad within block

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import torch
import math
import pytest
from torch_openreml.covariance import BlockDiagonal, ScalarMatrix, DiagonalMatrix, IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


class TestBlockDiagonal:
    """Tests for the BlockDiagonal operator."""

    def test_constructor(self):
        op = BlockDiagonal(ScalarMatrix(2), ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_two(self):
        with pytest.raises(ValueError, match="At least two operands"):
            BlockDiagonal(ScalarMatrix(3))

    def test_shape_different_sizes(self):
        op = BlockDiagonal(ScalarMatrix(2), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        result = op(free_params)
        assert result.shape == (5, 5)

    def test_shape_same_sizes(self):
        op = BlockDiagonal(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        result = op(free_params)
        assert result.shape == (6, 6)

    def test_call_block_structure(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([1.0, 2.0])
        result = op(free_params)
        # Block (2,3) and (3,2) should be zero (off-diagonal blocks)
        assert (result[0:2, 2:5] == 0).all()
        assert (result[2:5, 0:2] == 0).all()

    def test_call_diagonal_blocks(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.0, 1.0, 2.0])
        # a: sigma^2 = e^0 = 1
        # b: sigma^2 = [e^0, e^2, e^4] = [1, e^2, e^4]
        result = op(free_params)
        # Block a: 2x2 identity
        assert torch.allclose(result[0:2, 0:2], torch.eye(2))
        # Block b: 3x3 diagonal with [1, e^2, e^4]
        expected_b = torch.diag(torch.tensor([1.0, math.exp(2), math.exp(4)]))
        assert torch.allclose(result[2:5, 2:5], expected_b)

    def test_call_with_tensor(self):
        fixed = torch.tensor([[2.0]])
        op = BlockDiagonal(a=ScalarMatrix(2), fixed=fixed)
        free_params = torch.tensor([0.0])
        result = op(free_params)
        assert result.shape == (3, 3)
        assert result[2, 2] == 2.0

    def test_manual_grad_shape(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 5, 5)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_block_placement(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, _ = op.manual_grad(free_params)
        # grad[0] should be non-zero only in block (0:2, 0:2)
        assert (grad[0, 0:2, 0:2] != 0).any()
        assert (grad[0, 2:5, 2:5] == 0).all()
        assert (grad[0, 0:2, 2:5] == 0).all()
        assert (grad[0, 2:5, 0:2] == 0).all()
        # grad[1] should be non-zero only in block (2:5, 2:5)
        assert (grad[1, 2:5, 2:5] != 0).any()
        assert (grad[1, 0:2, 0:2] == 0).all()

    def test_manual_grad_equals_standalone(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad_op, _ = op.manual_grad(free_params)
        # Compare block a grad with standalone ScalarMatrix(2) grad
        a = ScalarMatrix(2)
        grad_a, _ = a.manual_grad(torch.tensor([0.0]))
        assert torch.allclose(grad_op[0, 0:2, 0:2], grad_a[0])

    def test_manual_grad_vs_auto_grad(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.1, 0.2, 0.3, 0.4])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = BlockDiagonal(IdentityMatrix(2), IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_three_operands(self):
        op = BlockDiagonal(ScalarMatrix(1), ScalarMatrix(2), ScalarMatrix(3))
        assert op.num_free_params == 3
        free_params = torch.tensor([0.0, 0.0, 0.0])
        result = op(free_params)
        assert result.shape == (6, 6)

    def test_intermediate_cache_hit(self):
        op = BlockDiagonal(ScalarMatrix(2), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "row_offsets" in cache
        assert "col_offsets" in cache

    def test_map_theta_to_v(self):
        op = BlockDiagonal(ScalarMatrix(2), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = BlockDiagonal(ScalarMatrix(2), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_inverse_matches_inverse_of_call(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_is_block_diagonal_of_operand_inverses(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        expected = torch.block_diag(ScalarMatrix(2).inverse(free_params[:1]), DiagonalMatrix(3).inverse(free_params[1:]))
        assert torch.allclose(op.inverse(free_params), expected)

    def test_inverse_product_with_call_is_identity(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params) @ op(free_params), torch.eye(5), atol=1e-6)

    def test_inverse_off_block_entries_zero(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        result = op.inverse(free_params)
        off_blocks = result[2:, :2]
        assert (off_blocks == 0.0).all()
        assert (result[:2, 2:] == 0.0).all()

    def test_inverse_three_operands(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(2), ScalarMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_nested_operand(self):
        op = BlockDiagonal(ScalarMatrix(2), BlockDiagonal(DiagonalMatrix(2), ScalarMatrix(2)))
        free_params = torch.tensor([0.5, 0.0, 1.0, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_with_tensor_operand(self):
        op = BlockDiagonal(ScalarMatrix(2), torch.eye(2) * 2.0)
        free_params = torch.tensor([0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        op.inverse(torch.tensor([0.5, 0.0, 0.5, 1.0]))
        assert calls == []

    def test_inverse_rectangular_operand_raises(self):
        op = BlockDiagonal(ScalarMatrix(2), torch.ones(2, 3))
        with pytest.raises(RuntimeError):
            op.inverse(torch.tensor([0.5]))

    def test_inverse_uses_default_free_params(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_validation_is_not_bypassed(self):
        op = BlockDiagonal(ScalarMatrix(2), DiagonalMatrix(3))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.5, 0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="length 4"):
            op.inverse(torch.tensor([0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=ScalarMatrix(3))
        r = repr(op)
        assert "BlockDiagonal" in r
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Different-sized operands**: This is the key differentiator from `Sum`. The `manual_grad` must correctly pad each operand's gradient into a zeros matrix of the full output shape, using `row_offsets` and `col_offsets`.
- **Non-square operands**: The code tracks `row_offsets` and `col_offsets` separately, allowing non-square blocks. Tests should verify this works correctly.
- **Gradient zeros outside block**: Each operand's gradient should only affect its own block region; all other regions must be exactly zero.
- **Offset calculation**: `v_groups` is computed in `build_operands` (which evaluates operands), then offsets are computed by iterating over the result shapes. The offsets might be recomputed each time `_get_or_build_intermediates` is called — but they're cached after first computation.
- **`inverse` requires square blocks, unlike the forward pass**: the forward pass tracks `row_offsets` and `col_offsets` separately and happily assembles non-square blocks, but a block diagonal matrix has an inverse only when every block is square (and invertible). The override does no shape check of its own, so `BlockDiagonal(ScalarMatrix(2), torch.ones(2, 3))` fails inside the rectangular block's own inverse with `RuntimeError: linalg.inv: A must be batches of square matrices...`. The test matches `RuntimeError` only, not the message wording, so it does not pin torch's exact phrasing.
- **The `inverse` path must not go through `__call__`**: a fallback that inverted `op(free_params)` would pass every value test, so the monkeypatched-`torch.linalg.inv` test is what pins the closed form; it asserts the dense inverse is not called at all. This matters more here than for the other operators, since the whole point of the block-diagonal form is to trade one `n³` inversion for a sum of `n_i³` ones.
- **Block order in the inverse follows `operands` iteration order**, the same order the forward pass and `manual_grad` use. The reverse-ordered comparison is implicitly covered by the operand-inverse test, which builds `torch.block_diag` from the operands in declaration order; a future change to operand ordering would break the forward pass first, so no separate ordering test is written.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Constructor | 2 |
| Shape (different/same sizes) | 2 |
| `__call__` structure & blocks | 3 |
| `manual_grad` structure & placement | 3 |
| `manual_grad` vs `auto_grad` | 1 |
| All-fixed | 1 |
| Three operands | 1 |
| Intermediate caching | 1 |
| REML interface | 2 |
| repr | 1 |
| `inverse` override | 15 |
| **Total** | **32** |
