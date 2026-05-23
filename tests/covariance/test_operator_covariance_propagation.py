import torch
import pytest
from torch_openreml.covariance import (
    CovariancePropagation,
    DummyMatrix,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestCovariancePropagation:
    """Tests for the CovariancePropagation operator."""

    def test_constructor(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            CovariancePropagation(ScalarMatrix(3))
        with pytest.raises(ValueError, match="Two operands"):
            CovariancePropagation(ScalarMatrix(3), ScalarMatrix(3), ScalarMatrix(3))

    def test_param_namespacing(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        assert op.num_free_params == 1
        assert op.free_param_names == ["g/sigma^2"]

    def test_call_fixed_z_trainable_g(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        result = op(free_params)
        expected = torch.eye(3)
        assert torch.allclose(result, expected)

    def test_call_dummy_z_trainable_g(self):
        z = DummyMatrix(["a", "b", "a"])
        op = CovariancePropagation(z=z, g=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        result = op(free_params)
        z_mat = z()
        g_mat = torch.diag(torch.exp(2.0 * free_params))
        expected = z_mat @ g_mat @ z_mat.T
        assert torch.allclose(result, expected)

    def test_shape(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = CovariancePropagation(z=z, g=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        result = op(free_params)
        assert result.shape == (4, 4)

    def test_manual_grad_g_only(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["g/sigma^2"]

    def test_manual_grad_g_equals_propagated(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        g = ScalarMatrix(3)
        grad_g, _ = g.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, grad_g)

    def test_manual_grad_z_and_g_both_trainable(self):
        op = CovariancePropagation(z=ScalarMatrix(2), g=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 2, 2)
        assert grad_names == ["z/sigma^2", "g/sigma^2"]

    def test_manual_grad_vs_auto_grad_fixed_z(self):
        z = DummyMatrix(["a", "b", "a"])
        op = CovariancePropagation(z=z, g=DiagonalMatrix(2))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_vs_auto_grad_both_trainable(self):
        op = CovariancePropagation(z=ScalarMatrix(2), g=ScalarMatrix(2))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "z" in cache
        assert "g" in cache

    def test_map_theta_to_v(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        op = CovariancePropagation(z=IdentityMatrix(3), g=ScalarMatrix(3))
        r = repr(op)
        assert "CovariancePropagation" in r
