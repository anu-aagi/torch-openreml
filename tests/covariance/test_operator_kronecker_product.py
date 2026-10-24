import torch
import math
import pytest
from torch_openreml.covariance import KroneckerProduct, ScalarMatrix, DiagonalMatrix, IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


class TestKroneckerProduct:
    """Tests for the KroneckerProduct operator."""

    def test_constructor(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            KroneckerProduct(ScalarMatrix(3))

    def test_shape(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        result = op(free_params)
        assert result.shape == (6, 6)

    def test_call_identity(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=IdentityMatrix(3))
        free_params = torch.tensor([])
        result = op(free_params)
        assert torch.equal(result, torch.eye(6))

    def test_call_values(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        result = op(free_params)
        expected_a = torch.eye(2)
        expected_b = math.exp(1) * torch.eye(3)
        expected = torch.kron(expected_a, expected_b)
        assert torch.allclose(result, expected)

    def test_param_namespacing(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        assert op.num_free_params == 2
        assert sorted(op.free_param_names) == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_shape(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 6, 6)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_a_only(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 6, 6)
        assert grad_names == ["a/sigma^2"]

    def test_manual_grad_b_only(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 6, 6)
        assert grad_names == ["b/sigma^2"]

    def test_manual_grad_a_equals_kron(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        a = ScalarMatrix(2)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        b = torch.eye(3)
        expected = torch.kron(grad_a, b)
        assert torch.allclose(grad_op, expected)

    def test_manual_grad_b_equals_kron(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, _ = op.manual_grad(free_params)
        a = torch.eye(2)
        b = ScalarMatrix(3)
        grad_b, _ = b.manual_grad(torch.tensor([0.5]))
        expected = torch.kron(a, grad_b)
        assert torch.allclose(grad_op, expected)

    def test_manual_grad_vs_auto_grad(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = KroneckerProduct(a=IdentityMatrix(2), b=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "b" in cache

    def test_map_theta_to_v(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        assert torch.allclose(op.map_theta_to_v(free_params), op(free_params))

    def test_map_theta_to_dv(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        expected, _ = op.grad(free_params)
        assert torch.allclose(op.map_theta_to_dv(free_params), expected)

    def test_inverse_matches_inverse_of_call(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_is_kron_of_operand_inverses(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        expected = torch.kron(ScalarMatrix(2).inverse(free_params[:1]), DiagonalMatrix(2).inverse(free_params[1:]))
        assert torch.allclose(op.inverse(free_params), expected)

    def test_inverse_product_with_call_is_identity(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert torch.allclose(op.inverse(free_params) @ op(free_params), torch.eye(4), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        op.inverse(torch.tensor([0.5, 0.0, 1.0]))
        assert calls == []

    def test_inverse_rectangular_operand_raises(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=torch.ones(2, 3))
        with pytest.raises(RuntimeError):
            op.inverse(torch.tensor([0.5]))

    def test_inverse_uses_default_free_params(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_validation_is_not_bypassed(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.5, 0.0, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = KroneckerProduct(a=ScalarMatrix(2), b=ScalarMatrix(3))
        r = repr(op)
        assert "KroneckerProduct" in r
