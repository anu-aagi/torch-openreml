import torch
import math
import pytest
from torch_openreml.covariance import Sum, ScalarMatrix, IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


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
        assert op.num_params == 1
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
        expected = 2.0 * torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_unequal_params(self):
        op = Sum(ScalarMatrix(3), ScalarMatrix(3))
        free_params = torch.tensor([0.0, 1.0])
        result = op(free_params)
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
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.0]))
        assert torch.allclose(grad[0], grad_a[0])
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
