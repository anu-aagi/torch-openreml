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

    def test_constructor_positional(self):
        op = Augment(ScalarMatrix(2), DiagonalMatrix(2))
        assert isinstance(op, Matrix)
        assert list(op.operands.keys()) == ["op_0", "op_1"]

    def test_constructor_keyword(self):
        op = augmented()
        assert list(op.operands.keys()) == ["a", "b"]

    def test_constructor_requires_two_operands(self):
        with pytest.raises(ValueError, match="At least two operands"):
            Augment(a=ScalarMatrix(2))

    def test_param_names_namespaced(self):
        op = augmented()
        assert op.free_param_names == ["a/sigma^2", "b/sigma^2_0", "b/sigma^2_1"]

    def test_call_shape(self):
        op = augmented()
        assert op().shape == (2, 4)

    def test_call_with_dict_input(self):
        op = augmented()
        v = op(as_dict(op, torch.tensor([0.5, 0.5, 0.5])))
        assert v.shape == (2, 4)

    def test_call_follows_input_dtype(self):
        op = augmented()
        assert op(torch.full((3,), 0.5, dtype=torch.float64)).dtype == torch.float64

    def test_call_follows_input_device(self):
        op = augmented()
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
        op = Augment(a=ScalarMatrix(2), fixed=torch.eye(2, device=device))
        assert op(torch.tensor([0.5])).device == torch.device("cpu")


class TestAugmentGrad:
    """Tests for Augment gradients, including dict and default input."""

    def test_grad_without_input(self):
        op = augmented()
        grad, names = op.grad()
        assert grad.shape == (op.num_free_params, 2, 4)
        assert names == op.free_param_names

    def test_grad_with_dict_input(self):
        op = augmented()
        free_params = torch.tensor([0.5, 0.5, 0.5])
        grad, names = op.grad(as_dict(op, free_params))
        assert grad.shape == (op.num_free_params, 2, 4)
        assert names == op.free_param_names

    def test_grad_dict_matches_tensor(self):
        op = augmented()
        free_params = torch.tensor([0.5, 0.5, 0.5])
        by_tensor, _ = op.grad(free_params)
        by_dict, _ = op.grad(as_dict(op, free_params))
        assert torch.allclose(by_tensor, by_dict)

    def test_grad_follows_input_dtype(self):
        op = augmented()
        grad, _ = op.grad(torch.full((3,), 0.5, dtype=torch.float64))
        assert grad.dtype == torch.float64

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
        assert torch.count_nonzero(grad[2, :, :2]) == 0

    def test_grad_with_tensor_operand(self):
        op = Augment(a=ScalarMatrix(2), fixed=torch.eye(2))
        grad, names = op.grad(torch.tensor([0.5]))
        assert grad.shape == (1, 2, 4)
        assert names == ["a/sigma^2"]

    def test_grad_with_fixed_matrix_operand(self):
        op = Augment(a=ScalarMatrix(2), b=IdentityMatrix(2))
        grad, names = op.grad(torch.tensor([0.5]))
        assert names == ["a/sigma^2"]
