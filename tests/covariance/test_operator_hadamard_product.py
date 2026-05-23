import torch
import math
import pytest
from torch_openreml.covariance import HadamardProduct, ScalarMatrix, IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


class TestHadamardProduct:
    """Tests for the HadamardProduct operator."""

    def test_constructor(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            HadamardProduct(ScalarMatrix(3))

    def test_call_element_wise(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        result = op(free_params)
        expected = math.exp(1) * torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_with_scalar_tensor(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=torch.tensor([5.0]))
        free_params = torch.tensor([0.0])
        result = op(free_params)
        expected = 5.0 * torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_shape(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        assert op(free_params).shape == (3, 3)

    def test_param_namespacing(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert op.num_free_params == 2
        assert sorted(op.free_param_names) == ["a/sigma^2", "b/sigma^2"]

    def test_param_namespacing_one_trainable(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op.num_free_params == 1
        assert op.free_param_names == ["a/sigma^2"]

    def test_manual_grad_shape(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 3, 3)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_a_equals_da_times_b(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        b = torch.eye(3)
        expected = grad_a * b
        assert torch.allclose(grad_op, expected)

    def test_manual_grad_with_scalar_tensor(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=torch.tensor([5.0]))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        assert grad_op.shape == (1, 3, 3)
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, grad_a * 5.0)

    def test_manual_grad_vs_auto_grad(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = HadamardProduct(a=IdentityMatrix(3), b=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "b" in cache

    def test_map_theta_to_v(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        op = HadamardProduct(a=ScalarMatrix(3), b=ScalarMatrix(3))
        r = repr(op)
        assert "HadamardProduct" in r
