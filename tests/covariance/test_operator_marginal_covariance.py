import pytest
import torch

from torch_openreml.covariance import (
    DiagonalMatrix,
    DummyMatrix,
    KnownMatrix,
    LowerTriangularMatrix,
    MarginalCovariance,
    ScalarMatrix,
)
from torch_openreml.covariance.operator import Operator


def _op():
    return MarginalCovariance(
        Z=DummyMatrix(["a", "b", "c", "a"]),
        G=DiagonalMatrix(3),
        R=ScalarMatrix(4),
    )


def _fixed_op():
    return MarginalCovariance(
        Z=KnownMatrix(torch.eye(4)),
        G=KnownMatrix(torch.eye(4)),
        R=KnownMatrix(torch.eye(4)),
    )


class TestMarginalCovarianceConstructor:
    """Operand shapes and the two accepted calling conventions."""

    def test_is_operator_subclass(self):
        assert isinstance(_op(), Operator)

    def test_keyword_operands(self):
        op = _op()
        assert list(op.operands) == ["Z", "G", "R"]

    def test_positional_operands(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = MarginalCovariance(z, DiagonalMatrix(3), ScalarMatrix(4))
        assert list(op.operands) == ["Z", "G", "R"]

    def test_dict_operands(self):
        z = DummyMatrix(["a", "b", "c", "a"])
        op = MarginalCovariance({"Z": z, "G": DiagonalMatrix(3), "R": ScalarMatrix(4)})
        assert list(op.operands) == ["Z", "G", "R"]

    def test_positional_and_keyword_are_rejected_together(self):
        with pytest.raises(ValueError, match="not both"):
            MarginalCovariance(DummyMatrix(["a"]), G=DiagonalMatrix(1), R=ScalarMatrix(1))

    def test_requires_three_positional_operands(self):
        with pytest.raises(ValueError):
            MarginalCovariance()
        with pytest.raises(ValueError):
            MarginalCovariance(DummyMatrix(["a"]))
        with pytest.raises(ValueError):
            MarginalCovariance(DummyMatrix(["a"]), DiagonalMatrix(1))
        with pytest.raises(ValueError):
            MarginalCovariance(
                DummyMatrix(["a"]), DiagonalMatrix(1), ScalarMatrix(1), ScalarMatrix(1)
            )

    def test_missing_operand_is_rejected(self):
        with pytest.raises(ValueError, match="Z, G and R"):
            MarginalCovariance(Z=DummyMatrix(["a"]), G=DiagonalMatrix(1))

    def test_extra_operand_is_rejected(self):
        with pytest.raises(ValueError, match="Z, G and R"):
            MarginalCovariance(
                Z=DummyMatrix(["a"]),
                G=DiagonalMatrix(1),
                R=ScalarMatrix(1),
                Q=DiagonalMatrix(1),
            )

    def test_dict_with_wrong_keys_is_rejected(self):
        with pytest.raises(ValueError, match="Z, G and R"):
            MarginalCovariance({"Z": DummyMatrix(["a"]), "G": DiagonalMatrix(1)})

    def test_non_matrix_non_tensor_operand_is_rejected(self):
        with pytest.raises(TypeError):
            MarginalCovariance(Z=1, G=DiagonalMatrix(1), R=ScalarMatrix(1))

    def test_at_least_one_matrix_operand_is_required(self):
        with pytest.raises(TypeError, match="at least one Matrix"):
            MarginalCovariance(Z=torch.eye(4), G=torch.eye(2), R=torch.eye(4))

    def test_param_names_are_namespaced_by_operand(self):
        assert _op().param_names == [
            "G/sigma^2_0",
            "G/sigma^2_1",
            "G/sigma^2_2",
            "R/sigma^2",
        ]

    def test_z_parameters_come_first(self):
        op = MarginalCovariance(
            Z=LowerTriangularMatrix(4, 2),
            G=DiagonalMatrix(2),
            R=ScalarMatrix(4),
        )
        names = op.param_names
        assert names[0] == "Z/L_0_0"
        assert names[-1] == "R/sigma^2"
        assert names.index("G/sigma^2_0") < names.index("R/sigma^2")

    def test_free_param_names_match_param_names(self):
        op = _op()
        assert op.free_param_names == op.param_names

    def test_build_params_has_one_entry_per_free_parameter(self):
        op = _op()
        assert op.build_params(torch.zeros(4)).shape == (4,)

    def test_repr_contains_class_and_operands(self):
        text = repr(_op())
        assert "MarginalCovariance" in text
        assert "Z" in text or "DummyMatrix" in text


class TestMarginalCovarianceValues:
    """``V = R + Z G Zᵀ`` against explicit products."""

    def test_matches_explicit_product(self):
        z = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
        )
        g = DiagonalMatrix(2)
        r = ScalarMatrix(4)
        op = MarginalCovariance(Z=z, G=g, R=r)
        free_params = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
        expected = r(free_params[2:]) + z @ g(free_params[:2]) @ z.T
        assert torch.allclose(op(free_params), expected)

    def test_uses_the_one_hot_design(self):
        op = _op()
        free_params = torch.tensor([0.0, 0.1, 0.2, 0.3])
        z = DummyMatrix(["a", "b", "c", "a"])()
        g = DiagonalMatrix(3)(free_params[:3])
        r = ScalarMatrix(4)(free_params[3:])
        assert torch.allclose(op(free_params), r + z @ g @ z.T)

    def test_identity_z_with_two_scalar_terms_is_diagonal_sum(self):
        op = MarginalCovariance(
            Z=KnownMatrix(torch.eye(4)), G=ScalarMatrix(4), R=ScalarMatrix(4)
        )
        free_params = torch.tensor([0.1, -0.2])
        g = ScalarMatrix(4)(free_params[:1])
        r = ScalarMatrix(4)(free_params[1:])
        assert torch.allclose(op(free_params), g + r)

    def test_rank_one_g_contributes_a_multiple_of_ones(self):
        z = torch.ones(4, 1, dtype=torch.float64)
        op = MarginalCovariance(Z=z, G=ScalarMatrix(1), R=ScalarMatrix(4))
        free_params = torch.tensor([0.1, -0.2], dtype=torch.float64)
        g = ScalarMatrix(1)(free_params[:1])
        r = ScalarMatrix(4)(free_params[1:])
        expected = r + g * torch.ones(4, 4, dtype=torch.float64)
        assert torch.allclose(op(free_params), expected)

    def test_result_is_symmetric(self):
        v = _op()(torch.tensor([0.0, 0.1, 0.2, 0.3]))
        assert torch.allclose(v, v.T)

    def test_result_is_positive_definite(self):
        v = _op()(torch.tensor([0.0, 0.1, 0.2, 0.3]))
        assert torch.linalg.eigvalsh(v).min() > 0

    def test_default_parameters_match_explicit_zeros(self):
        op = _op()
        assert torch.allclose(op(), op(torch.zeros(4)))

    def test_shape_is_recorded_after_the_call(self):
        op = _op()
        result = op(torch.zeros(4))
        assert result.shape == (4, 4)
        assert op.shape == (4, 4)

    def test_repeated_calls_agree(self):
        op = _op()
        free_params = torch.tensor([0.0, 0.1, 0.2, 0.3])
        assert torch.allclose(op(free_params), op(free_params))

    def test_dict_input_matches_tensor_input(self):
        op = _op()
        free_params = torch.tensor([0.0, 0.1, 0.2, 0.3])
        as_dict = {
            "G/sigma^2_0": free_params[0:1],
            "G/sigma^2_1": free_params[1:2],
            "G/sigma^2_2": free_params[2:3],
            "R/sigma^2": free_params[3:4],
        }
        assert torch.allclose(op(free_params), op(as_dict))

    def test_changing_g_changes_the_result(self):
        op = _op()
        a = op(torch.tensor([0.0, 0.0, 0.0, 0.0]))
        b = op(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert not torch.allclose(a, b)


class TestMarginalCovarianceValidation:
    """Shape checks run on evaluation, not construction."""

    def _fixed_tensor_operand(self):
        # One Matrix operand is required; KnownMatrix supplies it.
        return KnownMatrix(torch.eye(4))

    def test_z_must_be_two_dimensional(self):
        op = MarginalCovariance(
            Z=torch.ones(4), G=torch.eye(2), R=self._fixed_tensor_operand()
        )
        with pytest.raises(ValueError, match="two-dimensional"):
            op()

    def test_g_must_be_two_dimensional(self):
        op = MarginalCovariance(
            Z=torch.ones(4, 2), G=torch.ones(2), R=self._fixed_tensor_operand()
        )
        with pytest.raises(ValueError, match="two-dimensional"):
            op()

    def test_r_must_be_two_dimensional(self):
        op = MarginalCovariance(
            Z=KnownMatrix(torch.ones(4, 2)), G=torch.eye(2), R=torch.ones(4)
        )
        with pytest.raises(ValueError, match="two-dimensional"):
            op()

    def test_r_must_be_square(self):
        op = MarginalCovariance(
            Z=torch.ones(4, 2), G=torch.eye(2), R=KnownMatrix(torch.ones(4, 3))
        )
        with pytest.raises(ValueError, match="R must be square"):
            op()

    def test_z_rows_must_match_r_rows(self):
        op = MarginalCovariance(
            Z=torch.ones(5, 2), G=torch.eye(2), R=KnownMatrix(torch.eye(4))
        )
        with pytest.raises(ValueError, match="same number of rows"):
            op()

    def test_z_columns_must_match_g(self):
        op = MarginalCovariance(
            Z=torch.ones(4, 3), G=torch.eye(2), R=KnownMatrix(torch.eye(4))
        )
        with pytest.raises(ValueError, match="both dimensions of G"):
            op()

    def test_g_must_be_square(self):
        op = MarginalCovariance(
            Z=torch.ones(4, 2), G=torch.ones(2, 3), R=KnownMatrix(torch.eye(4))
        )
        with pytest.raises(ValueError, match="both dimensions of G"):
            op()

    def test_validation_error_is_not_raised_at_construction(self):
        # Constructing must not evaluate the operands.
        MarginalCovariance(
            Z=torch.ones(5, 2), G=torch.eye(2), R=KnownMatrix(torch.eye(4))
        )


class TestMarginalCovarianceCache:
    """Intermediate caching keyed by parameter values, dtype and device."""

    def test_repeated_identical_parameters_reuse_the_result(self):
        op = _op()
        free_params = torch.tensor([0.0, 0.1, 0.2, 0.3])
        assert op(free_params) is op(free_params)

    def test_new_parameters_invalidate_the_cache(self):
        op = _op()
        first = op(torch.tensor([0.0, 0.0, 0.0, 0.0]))
        second = op(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert first is not second
        assert not torch.allclose(first, second)

    def test_dtype_change_invalidates_the_cache(self):
        op = _op()
        p32 = torch.zeros(4, dtype=torch.float32)
        p64 = torch.zeros(4, dtype=torch.float64)
        assert op(p32) is not op(p64)

    def test_disable_cache_forces_recomputation(self):
        op = _op()
        op.disable_cache()
        free_params = torch.zeros(4)
        assert op(free_params) is not op(free_params)

    def test_enable_cache_restores_reuse(self):
        op = _op()
        op.disable_cache()
        op.enable_cache()
        free_params = torch.zeros(4)
        assert op(free_params) is op(free_params)

    def test_grad_uses_the_cached_value(self):
        op = _op()
        free_params = torch.tensor([0.0, 0.1, 0.2, 0.3])
        v = op(free_params)
        op.manual_grad(free_params)
        assert op(free_params) is v


class TestMarginalCovarianceGradients:
    """Analytic Jacobian and the ``grad`` dispatch."""

    def test_manual_grad_matches_auto_grad(self):
        free_params = torch.tensor([0.0, 0.1, 0.2, 0.3])
        op = _op()
        manual, manual_names = op.manual_grad(free_params)
        auto, auto_names = op.auto_grad(free_params)
        assert manual_names == auto_names
        assert torch.allclose(manual, auto, atol=1e-6)

    def test_manual_grad_names_match_param_names(self):
        op = _op()
        _, names = op.manual_grad(torch.zeros(4))
        assert names == op.param_names

    def test_manual_grad_shape(self):
        grad, _ = _op().manual_grad(torch.zeros(4))
        assert grad.shape == (4, 4, 4)

    def test_grad_uses_manual_by_default(self):
        free_params = torch.tensor([0.0, 0.1, 0.2, 0.3])
        op = _op()
        grad, names = op.grad(free_params)
        manual, manual_names = op.manual_grad(free_params)
        assert names == manual_names
        assert torch.allclose(grad, manual)

    def test_grad_mode_auto(self):
        op = _op()
        op.grad_mode = "auto"
        grad, names = op.grad(torch.zeros(4))
        assert grad.shape == (4, 4, 4)
        assert names == op.param_names

    def test_grad_mode_manual(self):
        op = _op()
        op.grad_mode = "manual"
        grad, names = op.grad(torch.zeros(4))
        assert grad.shape == (4, 4, 4)
        assert names == op.param_names

    def test_g_only_gradient(self):
        op = MarginalCovariance(
            Z=torch.ones(4, 2),
            G=DiagonalMatrix(2),
            R=KnownMatrix(torch.eye(4)),
        )
        grad, names = op.grad(torch.zeros(2))
        assert names == ["G/sigma^2_0", "G/sigma^2_1"]
        assert grad.shape == (2, 4, 4)

    def test_r_only_gradient(self):
        op = MarginalCovariance(
            Z=torch.ones(4, 2),
            G=KnownMatrix(torch.eye(2)),
            R=ScalarMatrix(4),
        )
        grad, names = op.grad(torch.zeros(1))
        assert names == ["R/sigma^2"]
        assert grad.shape == (1, 4, 4)

    def test_z_only_gradient(self):
        op = MarginalCovariance(
            Z=LowerTriangularMatrix(4, 2),
            G=KnownMatrix(torch.eye(2)),
            R=KnownMatrix(torch.eye(4)),
        )
        grad, names = op.grad(torch.zeros(7))
        assert len(names) == 7
        assert names[0] == "Z/L_0_0"
        assert grad.shape == (7, 4, 4)

    def test_z_parameterised_gradient_matches_auto(self):
        op = MarginalCovariance(
            Z=LowerTriangularMatrix(4, 2),
            G=DiagonalMatrix(2),
            R=ScalarMatrix(4),
        )
        free_params = torch.tensor([0.0, 0.5, 1.0, 0.2, -0.3, 0.1, 0.0, 0.0, 0.0, 0.1])
        manual, manual_names = op.manual_grad(free_params)
        auto, auto_names = op.auto_grad(free_params)
        assert manual_names == auto_names
        assert torch.allclose(manual, auto, atol=1e-5)

    def test_all_fixed_operands_have_no_gradient(self):
        grad, names = _fixed_op().manual_grad()
        assert grad is None
        assert names == []

    def test_grad_of_all_fixed_operands_is_none(self):
        grad, names = _fixed_op().grad()
        assert grad is None
        assert names == []

    def test_grad_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            _op().grad(torch.zeros(3))

    def test_grad_rejects_unknown_dict_key(self):
        with pytest.raises(ValueError):
            _op().grad({"G/sigma^2_0": torch.tensor([1.0])})

    def test_grad_rejects_non_tensor_arguments(self):
        with pytest.raises(TypeError):
            _op().grad("not a tensor")


class TestMarginalCovarianceDtypeAndDevice:
    """One dtype and device for the composite, following the input tensor."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        result = _op()(torch.zeros(4, dtype=dtype))
        assert result.dtype == dtype

    def test_uses_default_dtype_without_input(self):
        assert _op()().dtype == torch.get_default_dtype()

    def test_fixed_operands_are_cast_to_the_resolved_dtype(self):
        op = MarginalCovariance(
            Z=KnownMatrix(torch.ones(4, 2, dtype=torch.float64)),
            G=DiagonalMatrix(2),
            R=ScalarMatrix(4),
        )
        result = op(torch.zeros(3, dtype=torch.float32))
        assert result.dtype == torch.float32

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        result = _op()(torch.zeros(4, device=device))
        assert result.device.type == device

    def test_gradient_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        grad, _ = _op().grad(torch.zeros(4, device=device))
        assert grad.device.type == device

    def test_shape_is_independent_of_dtype(self):
        assert _op()(torch.zeros(4, dtype=torch.float64)).shape == (4, 4)
