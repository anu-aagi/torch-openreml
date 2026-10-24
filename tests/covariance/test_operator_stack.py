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
