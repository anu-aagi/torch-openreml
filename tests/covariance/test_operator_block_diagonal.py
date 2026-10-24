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
        assert (result[0:2, 2:5] == 0).all()
        assert (result[2:5, 0:2] == 0).all()

    def test_call_diagonal_blocks(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.0, 1.0, 2.0])
        result = op(free_params)
        assert torch.allclose(result[0:2, 0:2], torch.eye(2))
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
        assert (grad[0, 0:2, 0:2] != 0).any()
        assert (grad[0, 2:5, 2:5] == 0).all()
        assert (grad[0, 0:2, 2:5] == 0).all()
        assert (grad[0, 2:5, 0:2] == 0).all()
        assert (grad[1, 2:5, 2:5] != 0).any()
        assert (grad[1, 0:2, 0:2] == 0).all()

    def test_manual_grad_equals_standalone(self):
        op = BlockDiagonal(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad_op, _ = op.manual_grad(free_params)
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
