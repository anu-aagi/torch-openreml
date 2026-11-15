import pytest
import torch

from torch_openreml.covariance import (
    Gram,
    IdentityMatrix,
    KnownMatrix,
    LowerTriangularMatrix,
    ScalarMatrix,
)
from torch_openreml.covariance.operator import Operator


def _x():
    return torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64
    )


class TestGramConstructor:
    """Operand handling and the ``gram_type`` switch."""

    def test_is_operator_subclass(self):
        assert isinstance(Gram(LowerTriangularMatrix(3, 2)), Operator)

    def test_default_gram_type_is_xtx(self):
        assert Gram(LowerTriangularMatrix(3, 2)).gram_type == "xtx"

    def test_accepts_xxt(self):
        assert Gram(LowerTriangularMatrix(3, 2), gram_type="xxt").gram_type == "xxt"

    @pytest.mark.parametrize("gram_type", ["XTX", "xTx", "xxt ", "", None, 1])
    def test_rejects_invalid_gram_type(self, gram_type):
        with pytest.raises(ValueError, match="gram_type"):
            Gram(LowerTriangularMatrix(3, 2), gram_type=gram_type)

    def test_positional_operand_is_named_op_0(self):
        assert list(Gram(LowerTriangularMatrix(3, 2)).operands) == ["op_0"]

    def test_keyword_operand_keeps_its_name(self):
        assert list(Gram(x=LowerTriangularMatrix(3, 2)).operands) == ["x"]

    def test_dict_operand_keeps_its_name(self):
        op = Gram({"design": LowerTriangularMatrix(3, 2)})
        assert list(op.operands) == ["design"]

    def test_parameter_names_are_namespaced(self):
        assert Gram(x=LowerTriangularMatrix(3, 2)).param_names[0] == "x/L_0_0"

    def test_requires_exactly_one_operand(self):
        with pytest.raises(ValueError, match="One operand"):
            Gram(LowerTriangularMatrix(3, 2), LowerTriangularMatrix(3, 2))

    def test_zero_operands_requires_a_matrix(self):
        with pytest.raises(TypeError, match="at least one Matrix"):
            Gram()

    def test_fixed_tensor_alone_is_rejected(self):
        # The Operator contract requires at least one Matrix operand.
        with pytest.raises(TypeError, match="at least one Matrix"):
            Gram(torch.eye(3))

    def test_fixed_tensor_can_be_wrapped_in_known_matrix(self):
        op = Gram(KnownMatrix(_x()))
        assert torch.equal(op(), _x().T @ _x())

    def test_repr_contains_class_and_gram_type(self):
        text = repr(Gram(LowerTriangularMatrix(3, 2)))
        assert "Gram" in text
        assert "xtx" in text

    def test_repr_shows_the_operand(self):
        assert "LowerTriangularMatrix" in repr(Gram(LowerTriangularMatrix(3, 2)))


class TestGramValues:
    """``V = XᵀX`` and ``V = XXᵀ``."""

    def test_xtx_matches_explicit_product(self):
        op = Gram(KnownMatrix(_x()), gram_type="xtx")
        assert torch.allclose(op(torch.tensor([], dtype=torch.float64)), _x().T @ _x())

    def test_xxt_matches_explicit_product(self):
        op = Gram(KnownMatrix(_x()), gram_type="xxt")
        assert torch.allclose(op(torch.tensor([], dtype=torch.float64)), _x() @ _x().T)

    def test_xtx_shape_is_m_by_m(self):
        op = Gram(KnownMatrix(_x()), gram_type="xtx")
        assert op().shape == (2, 2)
        assert op.shape == (2, 2)

    def test_xxt_shape_is_n_by_n(self):
        op = Gram(KnownMatrix(_x()), gram_type="xxt")
        assert op().shape == (3, 3)
        assert op.shape == (3, 3)

    def test_square_input_xtx_and_xxt_generally_differ(self):
        k = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert not torch.allclose(Gram(KnownMatrix(k))(), Gram(KnownMatrix(k), gram_type="xxt")())

    def test_gram_of_scalar_matrix_scales_quadratically(self):
        free_params = torch.tensor([0.5])
        op = Gram(ScalarMatrix(3))
        sigma2 = ScalarMatrix(3)(free_params)
        assert torch.allclose(op(free_params), sigma2.T @ sigma2)

    def test_gram_is_symmetric(self):
        for gram_type in ("xtx", "xxt"):
            v = Gram(KnownMatrix(_x()), gram_type=gram_type)()
            assert torch.allclose(v, v.T)

    def test_gram_is_positive_semidefinite(self):
        k = torch.tensor([[1.0, 2.0], [2.0, 1.0], [0.5, -1.0]])
        for gram_type in ("xtx", "xxt"):
            v = Gram(KnownMatrix(k), gram_type=gram_type)()
            assert torch.linalg.eigvalsh(v).min() >= -1e-8

    def test_single_column_xtx_is_rank_one(self):
        k = torch.tensor([[1.0], [2.0], [3.0]])
        v = Gram(KnownMatrix(k))()
        assert v.shape == (1, 1)
        assert torch.allclose(v, torch.tensor([[14.0]]))

    def test_single_row_xxt_is_rank_one(self):
        k = torch.tensor([[1.0, 2.0, 3.0]])
        v = Gram(KnownMatrix(k), gram_type="xxt")()
        assert torch.allclose(v, torch.tensor([[14.0]]))

    def test_zero_matrix_gives_zero_gram(self):
        op = Gram(KnownMatrix(torch.zeros(3, 2)))
        assert torch.equal(op(), torch.zeros(2, 2))

    def test_repeated_calls_are_equal(self):
        op = Gram(KnownMatrix(_x()))
        assert torch.allclose(op(), op())

    def test_updates_shape_after_call(self):
        op = Gram(LowerTriangularMatrix(4, 2))
        assert op.shape == ()
        op(torch.zeros(7))
        assert op.shape == (2, 2)


class TestGramValuesFromParameters:
    """The operand is rebuilt from the free-parameter tensor."""

    def test_xtx_uses_parameterised_operand(self):
        x = LowerTriangularMatrix(3, 2)
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        op = Gram(x)
        expected = x(free_params)
        assert torch.allclose(op(free_params), expected.T @ expected)

    def test_xxt_uses_parameterised_operand(self):
        x = LowerTriangularMatrix(3, 2)
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        op = Gram(x, gram_type="xxt")
        expected = x(free_params)
        assert torch.allclose(op(free_params), expected @ expected.T)

    def test_dict_input(self):
        x = LowerTriangularMatrix(3, 2)
        free_params = {
            "op_0/L_0_0": torch.tensor([0.0]),
            "op_0/L_1_0": torch.tensor([0.5]),
            "op_0/L_1_1": torch.tensor([1.0]),
            "op_0/L_2_0": torch.tensor([0.2]),
            "op_0/L_2_1": torch.tensor([-0.3]),
        }
        op = Gram(x)
        expected = x(torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3]))
        assert torch.allclose(op(free_params), expected.T @ expected)

    def test_build_params_returns_the_free_tensor(self):
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        op = Gram(LowerTriangularMatrix(3, 2))
        assert torch.equal(op.build_params(free_params), free_params)


class TestGramGradients:
    """Manual and automatic Jacobians agree."""

    @pytest.mark.parametrize("gram_type", ["xtx", "xxt"])
    def test_manual_grad_matches_auto_grad(self, gram_type):
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        op = Gram(LowerTriangularMatrix(3, 2), gram_type=gram_type)
        manual, manual_names = op.manual_grad(free_params)
        auto, auto_names = op.auto_grad(free_params)
        assert manual_names == auto_names
        assert torch.allclose(manual, auto, atol=1e-6)

    @pytest.mark.parametrize("gram_type", ["xtx", "xxt"])
    def test_manual_grad_has_jacobian_shape(self, gram_type):
        op = Gram(LowerTriangularMatrix(3, 2), gram_type=gram_type)
        grad, names = op.manual_grad(torch.zeros(5))
        expected = (5, 2, 2) if gram_type == "xtx" else (5, 3, 3)
        assert grad.shape == expected
        assert len(names) == 5

    def test_manual_grad_names_match_parameter_names(self):
        op = Gram(LowerTriangularMatrix(3, 2))
        _, names = op.manual_grad(torch.zeros(5))
        assert names == op.param_names

    def test_grad_uses_manual_by_default(self):
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3])
        op = Gram(LowerTriangularMatrix(3, 2))
        grad, names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert names == manual_names
        assert torch.allclose(grad, manual)

    def test_grad_mode_manual(self):
        op = Gram(LowerTriangularMatrix(3, 2))
        op.grad_mode = "manual"
        grad, names = op.grad(torch.zeros(5))
        assert grad.shape == (5, 2, 2)
        assert names == op.param_names

    def test_grad_mode_auto(self):
        op = Gram(LowerTriangularMatrix(3, 2))
        op.grad_mode = "auto"
        grad, names = op.grad(torch.zeros(5))
        assert grad.shape == (5, 2, 2)
        assert names == op.param_names

    def test_fixed_operand_has_no_gradient(self):
        grad, names = Gram(KnownMatrix(_x())).manual_grad()
        assert grad is None
        assert names == []

    def test_grad_of_fixed_operand_is_none(self):
        grad, names = Gram(KnownMatrix(_x())).grad()
        assert grad is None
        assert names == []

    def test_partially_fixed_operand_gradient_has_only_free_params(self):
        specs = {
            "L_0_0": {"fixed": False, "default": torch.tensor([1.0]), "trans": None},
            "L_1_0": {"fixed": True, "default": torch.tensor([1.0]), "trans": None},
            "L_1_1": {"fixed": False, "default": torch.tensor([1.0]), "trans": None},
            "L_2_0": {"fixed": False, "default": torch.tensor([1.0]), "trans": None},
            "L_2_1": {"fixed": False, "default": torch.tensor([1.0]), "trans": None},
        }
        from torch_openreml.covariance.transform import TransformIdentity

        for spec in specs.values():
            spec["trans"] = TransformIdentity()
        op = Gram(LowerTriangularMatrix(3, 2, param_specs=specs))
        free_params = torch.tensor([0.0, 1.0, 0.2, -0.3])
        grad, names = op.grad(free_params)
        assert len(names) == 4
        assert "op_0/L_1_0" not in names
        assert grad.shape == (4, 2, 2)

    def test_grad_rejects_wrong_length(self):
        op = Gram(LowerTriangularMatrix(3, 2))
        with pytest.raises(ValueError):
            op.grad(torch.zeros(4))

    def test_grad_rejects_unknown_dict_key(self):
        op = Gram(LowerTriangularMatrix(3, 2))
        with pytest.raises(ValueError):
            op.grad({"op_0/nope": torch.tensor([1.0])})


class TestGramDtypeAndDevice:
    """The operand and the result follow the resolved dtype and device."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        op = Gram(LowerTriangularMatrix(3, 2))
        result = op(torch.zeros(5, dtype=dtype))
        assert result.dtype == dtype

    def test_follows_tensor_dtype(self):
        op = Gram(KnownMatrix(_x()))
        assert op(torch.tensor([], dtype=torch.float64)).dtype == torch.float64

    def test_uses_default_dtype_when_no_param_tensor_is_given(self):
        assert Gram(KnownMatrix(_x()))().dtype == torch.get_default_dtype()

    def test_default_dtype_when_omitted(self):
        assert Gram(LowerTriangularMatrix(3, 2))().dtype == torch.get_default_dtype()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        op = Gram(LowerTriangularMatrix(3, 2))
        result = op(torch.zeros(5, device=device))
        assert result.device.type == device

    def test_operand_device_follows_input(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        op = Gram(LowerTriangularMatrix(3, 2))
        op(torch.zeros(5, device=device))
        assert op.build_operands(torch.zeros(5, device=device))[0].device.type == device
