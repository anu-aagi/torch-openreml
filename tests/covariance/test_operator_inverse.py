import torch
import math
import pytest
from torch_openreml.covariance import (
    Inverse,
    Sum,
    Product,
    Transpose,
    ScalarMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    DummyMatrix,
    UnconstrainedMatrix,
    LowerTriangularMatrix,
)
from torch_openreml.covariance.matrix import Matrix


class TestInverse:
    """Tests for the Inverse operator."""

    def test_constructor(self):
        op = Inverse(a=ScalarMatrix(3))
        assert isinstance(op, Matrix)

    def test_constructor_requires_exactly_one(self):
        with pytest.raises(ValueError, match="One operand"):
            Inverse(ScalarMatrix(3), ScalarMatrix(3))

    def test_constructor_rejects_no_operand(self):
        with pytest.raises(TypeError):
            Inverse()

    def test_constructor_rejects_tensor_operand(self):
        with pytest.raises(TypeError):
            Inverse(a=torch.eye(3))

    def test_positional_operand_names(self):
        op = Inverse(DiagonalMatrix(2))
        assert op.free_param_names == ["op_0/sigma^2_0", "op_0/sigma^2_1"]

    def test_call_scalar_inverts_operand(self):
        op = Inverse(a=ScalarMatrix(2))
        free_params = torch.tensor([0.5])
        assert torch.allclose(op(free_params), math.exp(-1) * torch.eye(2))

    def test_call_diagonal_matches_linalg_inv(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op(free_params), torch.linalg.inv(DiagonalMatrix(3)(free_params)))

    def test_call_unconstrained_matches_linalg_inv(self):
        operand = UnconstrainedMatrix(2)
        op = Inverse(a=UnconstrainedMatrix(2))
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(op(free_params), torch.linalg.inv(operand(free_params)))

    def test_call_with_dict_params(self):
        op = Inverse(a=DiagonalMatrix(2))
        free_params = torch.tensor([0.0, 0.5])
        expected = op(free_params)
        result = op({"a/sigma^2_0": free_params[:1], "a/sigma^2_1": free_params[1:]})
        assert torch.allclose(result, expected)

    def test_call_non_square_raises(self):
        op = Inverse(a=DummyMatrix(["a", "b", "c", "a"]))
        with pytest.raises(RuntimeError, match="square"):
            op(torch.tensor([]))

    def test_shape(self):
        op = Inverse(a=DiagonalMatrix(3))
        assert op(torch.tensor([0.0, 0.5, 1.0])).shape == (3, 3)
        assert op.shape == (3, 3)

    def test_param_namespacing(self):
        op = Inverse(a=DiagonalMatrix(2))
        assert op.num_free_params == 2
        assert op.free_param_names == ["a/sigma^2_0", "a/sigma^2_1"]

    def test_param_namespacing_fixed_operand(self):
        op = Inverse(a=IdentityMatrix(3))
        assert op.num_free_params == 0
        assert op.free_param_names == []

    def test_inverse_matches_inverse_of_call(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_returns_operand(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), DiagonalMatrix(3)(free_params))

    def test_inverse_product_with_call_is_identity(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op(free_params) @ op.inverse(free_params), torch.eye(3), atol=1e-6)

    def test_inverse_does_not_invert_densely(self, monkeypatch):
        calls = []
        real_inv = torch.linalg.inv

        def counting_inv(a):
            calls.append(a)
            return real_inv(a)

        monkeypatch.setattr(torch.linalg, "inv", counting_inv)
        op = Inverse(a=DiagonalMatrix(3))
        op.inverse(torch.tensor([0.0, 0.5, 1.0]))
        assert calls == []

    def test_inverse_nested_operand(self):
        op = Inverse(a=Product(a=DiagonalMatrix(2), b=ScalarMatrix(2)))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(free_params), torch.linalg.inv(op(free_params)), atol=1e-6)

    def test_inverse_uses_default_free_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        assert torch.allclose(op.inverse(), op.inverse(op.free_param_defaults))

    def test_inverse_accepts_dict_of_free_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(op.inverse(op._to_free_param_dict(free_params)), op.inverse(free_params))

    def test_inverse_dtype_follows_input(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert op.inverse(free_params).dtype == torch.float32
        assert op.inverse(free_params.double()).dtype == torch.float64

    def test_inverse_gradients_flow_to_free_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)

    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        op.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        torch.linalg.inv(op(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)

    def test_inverse_rectangular_operand_is_not_checked(self):
        op = Inverse(a=DummyMatrix(["a", "b", "c", "a"]))
        assert op.inverse(torch.tensor([])).shape == (4, 3)

    def test_inverse_validation_is_not_bypassed(self):
        op = Inverse(a=DiagonalMatrix(3))
        with pytest.raises(TypeError, match="Torch tensor"):
            op.inverse([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="length 3"):
            op.inverse(torch.tensor([0.0, 0.5]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            op.inverse({})

    def test_repr(self):
        op = Inverse(a=ScalarMatrix(3))
        assert "Inverse" in repr(op)


class TestInverseGrad:
    """Tests for Inverse gradient computation."""

    def test_manual_grad_shape(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (3, 3, 3)
        assert grad_names == ["a/sigma^2_0", "a/sigma^2_1", "a/sigma^2_2"]

    def test_manual_grad_equals_neg_v_da_v(self):
        operand = LowerTriangularMatrix(2, 2)
        op = Inverse(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.2, -0.3, 0.5])
        grad, _ = op.manual_grad(free_params)
        da, _ = operand.manual_grad(free_params)
        v = torch.linalg.inv(operand(free_params))
        assert torch.allclose(grad, -v @ da @ v)

    def test_manual_grad_vs_auto_grad(self):
        op = Inverse(a=LowerTriangularMatrix(2, 2))
        free_params = torch.tensor([0.2, -0.3, 0.5])
        manual, names_m = op.manual_grad(free_params)
        auto, names_a = op.auto_grad(free_params)
        assert manual.shape == (3, 2, 2)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_all_fixed(self):
        op = Inverse(a=IdentityMatrix(3))
        grad, grad_names = op.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_matches_manual_grad(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert torch.allclose(grad, manual)
        assert grad_names == manual_names == op.free_param_names


class TestInverseCache:
    """Tests for Inverse intermediate caching."""

    def test_intermediate_cache_hit(self):
        op = Inverse(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        assert op.get_intermediates(built) is None
        op(free_params)
        cache = op.get_intermediates(built)
        assert cache is not None
        assert "a" in cache
        assert "v" in cache

    def test_reset_intermediates_clears_cache(self):
        op = Inverse(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.reset_intermediates()
        assert op.get_intermediates(built) is None

    def test_disable_cache_clears_cache(self):
        op = Inverse(a=ScalarMatrix(3))
        free_params = torch.tensor([0.0])
        built = op.build_params(free_params)
        op(free_params)
        assert op.get_intermediates(built) is not None
        op.disable_cache()
        assert op.get_intermediates(built) is None

    def test_cache_recomputes_on_new_params(self):
        op = Inverse(a=DiagonalMatrix(3))
        first = torch.tensor([0.0, 0.5, 1.0])
        second = torch.tensor([1.0, 1.5, 2.0])
        v_first = op(first)
        v_second = op(second)
        assert not torch.allclose(v_first, v_second)
        assert torch.allclose(v_second, torch.linalg.inv(DiagonalMatrix(3)(second)))


class TestInverseDtypeDevice:
    """Tests for dtype and device resolution for the operand."""

    def test_input_tensor_dtype_is_used(self):
        op = Inverse(a=ScalarMatrix(3))
        assert op(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64

    def test_input_dict_dtype_is_used(self):
        op = Inverse(a=DiagonalMatrix(2))
        v = op(
            {
                "a/sigma^2_0": torch.tensor([0.5], dtype=torch.float64),
                "a/sigma^2_1": torch.tensor([0.5], dtype=torch.float64),
            }
        )
        assert v.dtype == torch.float64

    def test_input_device_is_used(self):
        op = Inverse(a=ScalarMatrix(3))
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        assert op(torch.tensor([0.5], device=device)).device.type == device

    def test_no_params_uses_torch_defaults(self):
        op = Inverse(a=IdentityMatrix(3))
        assert op.num_params == 0
        v = op()
        assert v.dtype == torch.get_default_dtype()
        assert v.device == torch.get_default_device()


class TestInverseTree:
    """Tests for Inverse integration with tree, free_param_tree and grad_tree."""

    def test_tree(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        results = op.tree(free_params)
        assert sorted(results) == ["/", "a"]
        assert torch.allclose(results["/"], op(free_params))

    def test_free_param_tree(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        free_params_by_path = op.free_param_tree(free_params)
        assert sorted(free_params_by_path) == ["/", "a"]
        assert free_params_by_path["a"].shape == (3,)

    def test_grad_tree(self):
        op = Inverse(a=DiagonalMatrix(3))
        free_params = torch.tensor([0.5, 0.0, 1.0])
        grads = op.grad_tree(free_params)
        assert sorted(grads) == ["/", "a"]
        manual, names = op.manual_grad(free_params)
        assert torch.allclose(grads["/"][0], manual)
        assert grads["/"][1] == names == op.free_param_names

    def test_nested_in_sum(self):
        op = Sum(inverted=Inverse(a=ScalarMatrix(2)), extra=IdentityMatrix(2))
        free_params = torch.tensor([0.5])
        grad, grad_names = op.manual_grad(free_params)
        assert grad.shape == (1, 2, 2)
        assert grad_names == ["inverted/a/sigma^2"]


class TestInverseComposition:
    """Tests for Inverse in compositions with other operators."""

    def test_inverse_of_inverse_returns_operand(self):
        operand = UnconstrainedMatrix(2)
        op = Inverse(a=Inverse(a=UnconstrainedMatrix(2)))
        free_params = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(op(free_params), operand(free_params))

    def test_inverse_commutes_with_transpose(self):
        nested = Inverse(a=Transpose(a=LowerTriangularMatrix(2, 2)))
        commuted = Transpose(a=Inverse(a=LowerTriangularMatrix(2, 2)))
        free_params = torch.tensor([0.2, -0.3, 0.5])
        assert torch.allclose(nested(free_params), commuted(free_params))
        grad_nested, names_nested = nested.manual_grad(free_params)
        grad_commuted, names_commuted = commuted.manual_grad(free_params)
        assert names_nested == names_commuted
        assert torch.allclose(grad_nested, grad_commuted)

    def test_inverse_of_product_is_reversed_product(self):
        lhs = Inverse(a=Product(a=DiagonalMatrix(2), b=LowerTriangularMatrix(2, 2)))
        rhs = Product(
            a=Inverse(a=LowerTriangularMatrix(2, 2)), b=Inverse(a=DiagonalMatrix(2))
        )
        diag_params = torch.tensor([0.0, 0.5])
        lower_params = torch.tensor([0.2, -0.3, 0.4])
        assert torch.allclose(
            lhs(torch.cat([diag_params, lower_params])),
            rhs(torch.cat([lower_params, diag_params])),
        )

    def test_product_with_inverse_is_identity(self):
        op = Product(a=Inverse(a=DiagonalMatrix(3)), b=DiagonalMatrix(3))
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        assert torch.allclose(op(free_params), torch.eye(3), atol=1e-6)
