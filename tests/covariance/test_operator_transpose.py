import torch
import math
import pytest
from torch_openreml.covariance import (
    Transpose,
    Sum,
    Product,
    CovariancePropagation,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    DummyMatrix,
    LowerTriangularMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestTranspose:
    """Tests for the Transpose operator."""

    def test_constructor(self):
        op = Transpose(a=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_one(self):
        with pytest.raises(ValueError, match="One operand"):
            Transpose(ScalarMatrix(3), ScalarMatrix(3))

    def test_constructor_rejects_no_operand(self):
        with pytest.raises(TypeError):
            Transpose()

    def test_positional_operand_names(self):
        op = Transpose(DiagonalMatrix(2))
        assert op.free_param_names == ["op_0/sigma^2_0", "op_0/sigma^2_1"]

    def test_call_scales_by_operand_values(self):
        op = Transpose(a=ScalarMatrix(2))
        free_params = torch.tensor([0.5])
        assert torch.allclose(op(free_params), math.exp(1) * torch.eye(2))

    def test_call_square_matches_operand(self):
        op = Transpose(a=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        assert torch.allclose(op(free_params), DiagonalMatrix(2)(free_params))

    def test_call_rectangular_fixed(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = Transpose(a=z)
        result = op(torch.tensor([]))
        assert result.shape == (3, 4)
        assert torch.equal(result, z(torch.tensor([])).T)

    def test_call_rectangular_trainable(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        operand = LowerTriangularMatrix(3, 2)
        assert torch.allclose(op(free_params), operand(free_params).T)

    def test_call_with_dict_params(self):
        op = Transpose(a=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        expected = op(free_params)
        result = op({"a/sigma^2_0": free_params[:1], "a/sigma^2_1": free_params[1:]})
        assert torch.allclose(result, expected)

    def test_shape_rectangular(self):
        op = Transpose(a=DummyMatrix(["a", "b", "c", "a"]))
        assert op(torch.tensor([])).shape == (3, 4)
        assert op.shape == (3, 4)

    def test_param_namespacing(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        assert op.num_free_params == 5
        assert op.free_param_names == ["a/L_0_0", "a/L_1_0", "a/L_1_1", "a/L_2_0", "a/L_2_1"]

    def test_param_namespacing_fixed_operand(self):
        op = Transpose(a=IdentityMatrix(3))
        assert op.num_free_params == 0
        assert op.free_param_names == []

    def test_inverse_matches_inverse_of_call(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_is_operand_inverse_transposed(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5])
        expected = LowerTriangularMatrix(2, 2).inverse(free_params).T
        assert torch.allclose(op.inverse(free_params), expected)

    def test_inverse_product_with_call_is_identity(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5])
        assert torch.allclose(op(free_params) @ op.inverse(free_params), torch.eye(2), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = Transpose(a=DiagonalMatrix(3))
        op.inverse(torch.tensor([0.0, 0.5, 1.0]))
        assert calls == []

    def test_inverse_nested_operand(self):
        op = Transpose(a=Product(a=DiagonalMatrix(2), b=ScalarMatrix(2)))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_rectangular_operand_raises(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        with pytest.raises(RuntimeError):
            op.inverse(free_params)

    def test_inverse_uses_default_free_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = Transpose(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.5, 0.2, 0.5], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_validation_is_not_bypassed(self):
        op = Transpose(a=DiagonalMatrix(3))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.0, 0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = Transpose(a=ScalarMatrix(3))
        assert "Transpose" in repr(op)


class TestTransposeGrad:
    """Tests for Transpose gradient computation."""

    def test_manual_grad_shape(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (3, 3, 3)
        assert grad_names == ["a/sigma^2_0", "a/sigma^2_1", "a/sigma^2_2"]

    def test_manual_grad_equals_da_transpose(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        grad, grad_names = op.manual_grad(free_params)
        operand = LowerTriangularMatrix(3, 2)
        grad_operand, _ = operand.manual_grad(free_params)
        assert grad.shape == (5, 2, 3)
        assert torch.allclose(grad, grad_operand.mT)
        assert grad_names == ["a/L_0_0", "a/L_1_0", "a/L_1_1", "a/L_2_0", "a/L_2_1"]

    def test_manual_grad_vs_auto_grad(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.1])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_vs_auto_grad_rectangular(self):
        op = Transpose(a=LowerTriangularMatrix(3, 2))
        free_params = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        manual, _ = op.manual_grad(free_params)
        auto, _ = op.auto_grad(free_params)
        assert manual.shape == (5, 2, 3)
        assert torch.allclose(manual, auto)

    def test_all_fixed(self):
        op = Transpose(a=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_matches_manual_grad(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert torch.allclose(grad, manual)
        assert grad_names == manual_names == op.free_param_names


class TestTransposeCache:
    """Tests for Transpose intermediate caching."""

    def test_intermediate_cache_hit(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "v" in cache

    def test_reset_intermediates_clears_cache(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.reset_intermediates()
        assert op.get_intermediates(built) is None

    def test_disable_cache_clears_cache(self):
        op = Transpose(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.disable_cache()
        assert op.get_intermediates(built) is None

    def test_cache_recomputes_on_new_params(self):
        op = Transpose(a=DiagonalMatrix(3))
        first = torch.tensor([0.0, 0.5, 1.0])
        second = torch.tensor([1.0, 1.5, 2.0])
        v_first = op(first)
        v_second = op(second)
        assert not torch.allclose(v_first, v_second)
        assert torch.allclose(v_second, op(second))


class TestTransposeDtypeDevice:
    """Tests for dtype and device resolution for the operand."""

    def test_input_tensor_dtype_is_used(self):
        op = Transpose(a=ScalarMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_input_dict_dtype_is_used(self):
        op = Transpose(a=DiagonalMatrix(2))
        v = op(
            {
                "a/sigma^2_0": torch.tensor([0.5], dtype=torch.float64),
                "a/sigma^2_1": torch.tensor([0.5], dtype=torch.float64),
            }
        )
        assert v.dtype == torch.float64

    def test_input_device_is_used(self):
        op = Transpose(a=ScalarMatrix(3))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.tensor([0.5], device=device)).device.type == device

    def test_no_params_uses_torch_defaults(self):
        op = Transpose(a=IdentityMatrix(3))
        assert op.num_params == 0
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()


class TestTransposeTree:
    """Tests for Transpose integration with tree, free_param_tree and grad_tree."""

    def test_tree(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        results = op.tree(free_params)
        assert sorted(results) == ["/", "a"]
        assert torch.allclose(results["/"], op(free_params))

    def test_free_param_tree(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        free_params_by_path = op.free_param_tree(free_params)
        assert sorted(free_params_by_path) == ["/", "a"]
        assert free_params_by_path["a"].shape == (3,)

    def test_grad_tree(self):
        op = Transpose(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        grads = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a"]
        manual, names = op.manual_grad(free_params)
        assert torch.allclose(grads["/"][0], manual)
        assert grads["/"][1] == names == op.free_param_names

    def test_nested_in_sum(self):
        op = Sum(transposed=Transpose(a=ScalarMatrix(2)), extra=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["transposed/a/sigma^2"]


class TestTransposeComposition:
    """Tests for Transpose in compositions with other operators."""

    def test_transpose_of_transpose_returns_operand(self):
        operand = LowerTriangularMatrix(3, 2)
        op = Transpose(a=Transpose(a=operand))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        assert torch.allclose(op(free_params), operand(free_params))

    def test_sandwich_equals_covariance_propagation(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        g = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])

        sandwich = Product(a=z, b=Product(a=g, b=Transpose(a=z)))
        propagation = CovariancePropagation(z=z, g=g)

        assert torch.allclose(sandwich(free_params), propagation(free_params))

        grad_sandwich, _ = sandwich.manual_grad(free_params)
        grad_propagation, _ = propagation.manual_grad(free_params)
        assert torch.allclose(grad_sandwich, grad_propagation)
