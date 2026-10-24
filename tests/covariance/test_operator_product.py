import torch
import math
import pytest
from torch_openreml.covariance import (
    Product,
    Sum,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    DummyMatrix,
    LowerTriangularMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestProduct:
    """Tests for the Product operator."""

    def test_constructor(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_two(self):
        with pytest.raises(ValueError, match="Two operands"):
            Product(ScalarMatrix(3))

    def test_constructor_rejects_three(self):
        with pytest.raises(ValueError, match="Two operands"):
            Product(ScalarMatrix(3), ScalarMatrix(3), ScalarMatrix(3))

    def test_positional_operand_names(self):
        op = Product(ScalarMatrix(3), ScalarMatrix(3))
        assert op.free_param_names == ["op_0/sigma^2", "op_1/sigma^2"]

    def test_call_scales_by_operand_values(self):
        op = Product(a=ScalarMatrix(2), b=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        result = op(free_params)
        assert torch.allclose(result, math.exp(1) * torch.eye(2))

    def test_call_matches_matmul(self):
        op = Product(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 0.5])
        a = ScalarMatrix(2)
        b = DiagonalMatrix(2)
        assert torch.allclose(op(free_params), a(free_params[:1]) @ b(free_params[1:]))

    def test_call_identity_operands(self):
        op = Product(a=IdentityMatrix(3), b=IdentityMatrix(3))
        assert torch.equal(op(torch.tensor([])), torch.eye(3))

    def test_call_with_fixed_tensor_first(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = Product(a=x, b=IdentityMatrix(2))
        assert torch.allclose(op(torch.tensor([])), x)

    def test_call_with_fixed_tensor_second(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = Product(a=IdentityMatrix(2), b=x)
        assert torch.allclose(op(torch.tensor([])), x)

    def test_call_with_dict_params(self):
        op = Product(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.5, 0.0, 0.5])
        expected = op(free_params)
        result = op(
            {
                "a/sigma^2": free_params[:1],
                "b/sigma^2_0": free_params[1:2],
                "b/sigma^2_1": free_params[2:],
            }
        )
        assert torch.allclose(result, expected)

    def test_shape_square(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 0.5, 1.0])
        assert op(free_params).shape == (3, 3)
        assert op.shape == (3, 3)

    def test_shape_rectangular(self):
        op = Product(a=DummyMatrix(["a", "b", "c", "a"]), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op(free_params).shape == (4, 3)
        assert op.shape == (4, 3)

    def test_shape_rectangular_fixed_first(self):
        x = torch.ones(2, 4)
        op = Product(a=x, b=DummyMatrix(["a", "b", "c", "a"]))
        result = op(torch.tensor([]))
        assert result.shape == (2, 3)
        assert torch.allclose(result, x @ DummyMatrix(["a", "b", "c", "a"])(torch.tensor([])))

    def test_mismatched_shapes_raise_at_runtime(self):
        op = Product(a=IdentityMatrix(2), b=IdentityMatrix(3))
        with pytest.raises(RuntimeError):
            op(torch.tensor([]))

    def test_param_namespacing(self):
        op = Product(a=ScalarMatrix(2), b=DiagonalMatrix(2))
        assert op.num_free_params == 3
        assert op.free_param_names == ["a/sigma^2", "b/sigma^2_0", "b/sigma^2_1"]

    def test_param_namespacing_one_trainable(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op.num_free_params == 1
        assert op.free_param_names == ["a/sigma^2"]

    def test_param_namespacing_second_trainable(self):
        op = Product(a=IdentityMatrix(3), b=ScalarMatrix(3))
        assert op.num_free_params == 1
        assert op.free_param_names == ["b/sigma^2"]

    def test_inverse_matches_inverse_of_call(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_is_reversed_product_of_operand_inverses(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        expected = LowerTriangularMatrix(2, 2).inverse(free_params[2:]) @ DiagonalMatrix(2).inverse(free_params[:2])
        assert torch.allclose(op.inverse(free_params), expected)

    def test_inverse_is_not_the_forward_ordered_product(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        forward = DiagonalMatrix(2).inverse(free_params[:2]) @ LowerTriangularMatrix(2, 2).inverse(free_params[2:])
        assert not torch.allclose(op.inverse(free_params), forward)

    def test_inverse_product_with_call_is_identity(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5])
        assert torch.allclose(op.inverse(free_params) @ op(free_params), torch.eye(2), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        op.inverse(torch.tensor([0.0, 0.5, 1.0]))
        assert calls == []

    def test_inverse_with_fixed_tensor_operand(self):
        op = Product(a=DiagonalMatrix(2), b=torch.eye(2) * 2.0)
        free_params = torch.tensor([0.0, 0.5])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_rectangular_operand_raises(self):
        op = Product(a=LowerTriangularMatrix(3, 2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3, 1.0])
        with pytest.raises(RuntimeError):
            op.inverse(free_params)

    def test_inverse_uses_default_free_params(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.5], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_validation_is_not_bypassed(self):
        op = Product(a=DiagonalMatrix(2), b=ScalarMatrix(2))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.0, 0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        assert "Product" in repr(op)


class TestProductGrad:
    """Tests for Product gradient computation."""

    def test_manual_grad_shape(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (2, 3, 3)
        assert grad_names == ["a/sigma^2", "b/sigma^2"]

    def test_manual_grad_a_equals_da_matmul_b(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, grad_names = op.manual_grad(free_params)
        a = ScalarMatrix(3)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, grad_a @ torch.eye(3))
        assert grad_names == ["a/sigma^2"]

    def test_manual_grad_b_equals_a_matmul_db(self):
        op = Product(a=IdentityMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.5])
        grad_op, grad_names = op.manual_grad(free_params)
        b = ScalarMatrix(3)
        grad_b, _ = b.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, torch.eye(3) @ grad_b)
        assert grad_names == ["b/sigma^2"]

    def test_manual_grad_with_fixed_tensor_first(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = Product(a=x, b=ScalarMatrix(2))
        free_params = torch.tensor([0.5])
        grad_op, grad_names = op.manual_grad(free_params)
        b = ScalarMatrix(2)
        grad_b, _ = b.manual_grad(torch.tensor([0.5]))
        assert torch.allclose(grad_op, x @ grad_b)
        assert grad_names == ["b/sigma^2"]

    def test_manual_grad_rectangular_a(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = Product(a=ScalarMatrix(4), b=z)
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        a = ScalarMatrix(4)
        grad_a, _ = a.manual_grad(torch.tensor([0.5]))
        assert grad.shape == (1, 4, 3)
        assert torch.allclose(grad, grad_a @ z(torch.tensor([])))
        assert grad_names == ["a/sigma^2"]

    def test_manual_grad_rectangular_b(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = Product(a=z, b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.manual_grad(free_params)
        b = DiagonalMatrix(3)
        grad_b, _ = b.manual_grad(free_params)
        assert grad.shape == (3, 4, 3)
        assert torch.allclose(grad, z(torch.tensor([])) @ grad_b)
        assert grad_names == ["b/sigma^2_0", "b/sigma^2_1", "b/sigma^2_2"]

    def test_manual_grad_vs_auto_grad(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_vs_auto_grad_rectangular(self):
        op = Product(a=LowerTriangularMatrix(3, 2), b=DiagonalMatrix(2))
        free_params = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5])
        manual, _ = op.manual_grad(free_params)
        auto, _ = op.auto_grad(free_params)
        assert manual.shape == (7, 3, 2)
        assert torch.allclose(manual, auto)

    def test_manual_grad_vs_auto_grad_same_instance(self):
        m = ScalarMatrix(3)
        op = Product(a=m, b=m)
        free_params = torch.tensor([0.5, 0.5])
        assert op.num_free_params == 2
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert manual.shape == (2, 3, 3)
        assert torch.allclose(manual, auto)
        assert names_m == names_a == ["a/sigma^2", "b/sigma^2"]

    def test_all_fixed(self):
        op = Product(a=IdentityMatrix(3), b=torch.eye(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_matches_manual_grad(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 1.5])
        grad, grad_names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert torch.allclose(grad, manual)
        assert grad_names == manual_names == op.free_param_names


class TestProductCache:
    """Tests for Product intermediate caching."""

    def test_intermediate_cache_hit(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "b" in cache
        assert "v" in cache

    def test_reset_intermediates_clears_cache(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.reset_intermediates()
        assert op.get_intermediates(built) is None

    def test_disable_cache_clears_cache(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        free_params = torch.tensor([0.0, 0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.disable_cache()
        assert op.get_intermediates(built) is None

    def test_cache_recomputes_on_new_params(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        first = torch.tensor([0.0, 0.0])
        second = torch.tensor([0.5, 0.5])
        v_first = op(first)
        v_second = op(second)
        assert not torch.allclose(v_first, v_second)
        assert torch.allclose(v_second, op(second))


class TestProductDtypeDevice:
    """Tests for dtype and device resolution across the two operands."""

    def test_input_tensor_dtype_is_used(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_input_dict_dtype_is_used(self):
        op = Product(a=ScalarMatrix(3), b=ScalarMatrix(3))
        v = op(
            {
                "a/sigma^2": torch.tensor([0.5], dtype=torch.float64),
                "b/sigma^2": torch.tensor([0.5], dtype=torch.float64),
            }
        )
        assert v.dtype == torch.float64

    def test_input_device_is_used(self):
        op = Product(a=ScalarMatrix(3), b=IdentityMatrix(3))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.tensor([0.5], device=device)).device.type == device

    def test_tensor_operand_follows_input_dtype(self):
        op = Product(a=ScalarMatrix(3), b=torch.eye(3, dtype=torch.float64))
        assert op(torch.tensor([0.5], dtype=torch.float32)).dtype == torch.float32

    def test_tensor_operand_casts_up_to_input_dtype(self):
        op = Product(a=torch.eye(3), b=ScalarMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_no_params_uses_torch_defaults(self):
        op = Product(a=IdentityMatrix(3), b=IdentityMatrix(3))
        assert op.num_params == 0
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()


class TestProductTree:
    """Tests for Product integration with tree, free_param_tree and grad_tree."""

    def test_tree(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0, 1.5])
        results = op.tree(free_params)
        assert sorted(results) == ["/", "a", "b"]
        assert torch.allclose(results["/"], op(free_params))

    def test_free_param_tree(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0, 1.5])
        free_params_by_path = op.free_param_tree(free_params)
        assert sorted(free_params_by_path) == ["/", "a", "b"]
        assert free_params_by_path["a"].shape == (1,)
        assert free_params_by_path["b"].shape == (3,)

    def test_grad_tree(self):
        op = Product(a=ScalarMatrix(3), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0, 1.5])
        grads = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a", "b"]
        manual, names = op.manual_grad(free_params)
        assert torch.allclose(grads["/"][0], manual)
        assert grads["/"][1] == names == op.free_param_names

    def test_nested_in_sum(self):
        op = Sum(scaled=Product(a=ScalarMatrix(2), b=IdentityMatrix(2)), extra=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["scaled/a/sigma^2"]
