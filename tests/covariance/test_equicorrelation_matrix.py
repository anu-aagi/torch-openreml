import torch
import pytest
import pandas as pd
from math import log
from torch_openreml.covariance import EquicorrelationMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformChain,
    TransformScaleShift,
    TransformSigmoid,
)


class TestEquicorrelationMatrix:
    """Tests for the equicorrelation matrix."""

    def test_constructor(self):
        mat = EquicorrelationMatrix(3)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = EquicorrelationMatrix(3)
        assert mat.shape == (3, 3)

    def test_rho_min(self):
        mat = EquicorrelationMatrix(3)
        assert mat.rho_min == -0.5

    def test_shape_from_levels(self):
        assert EquicorrelationMatrix(["a", "b", "a", "c"]).shape == (3, 3)
        assert EquicorrelationMatrix(("a", "b", "a", "c")).shape == (3, 3)
        assert EquicorrelationMatrix(pd.Series(["a", "b", "a", "c"])).shape == (3, 3)

    def test_levels_equivalent_to_distinct_count(self):
        from_levels = EquicorrelationMatrix(["a", "b", "a", "c"])
        from_int = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5])
        assert from_levels.shape == from_int.shape
        assert from_levels.rho_min == from_int.rho_min
        assert from_levels.param_names == from_int.param_names
        assert torch.equal(from_levels(free_params), from_int(free_params))

    def test_rejects_empty_levels(self):
        with pytest.raises(ValueError, match="greater than 1"):
            EquicorrelationMatrix([])

    def test_rho_min_n2(self):
        mat = EquicorrelationMatrix(2)
        assert mat.rho_min == -1.0

    def test_default_params(self):
        mat = EquicorrelationMatrix(3)
        assert mat.num_params == 1
        assert mat.num_free_params == 1
        assert mat.num_fixed_params == 0

    def test_param_names(self):
        mat = EquicorrelationMatrix(3)
        assert mat.param_names == ["rho"]
        assert mat.free_param_names == ["rho"]

    def test_call_diagonal_is_one(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0))

    def test_call_off_diagonal_uniform(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        result = mat(free_params)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == off_diag[0]).all()

    def test_call_rho_zero_is_identity(self):
        mat = EquicorrelationMatrix(3)
        x = log(1 / 3 / (2 / 3))
        free_params_zero = torch.tensor([x])
        result = mat(free_params_zero)
        assert torch.allclose(result, torch.eye(3))

    def test_call_rho_in_bounds_negative(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([-10.0])
        result = mat(free_params)
        off_diag = result[0, 1]
        assert off_diag > mat.rho_min

    def test_call_rho_in_bounds_positive(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([10.0])
        result = mat(free_params)
        off_diag = result[0, 1]
        assert off_diag < 1.0

    def test_call_symmetric(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_n1(self):
        with pytest.raises(ValueError, match="'n' must be greater than 1!"):
            EquicorrelationMatrix(1)

    def test_manual_grad_shape(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["rho"]

    def test_manual_grad_diagonal_zero(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        grad, _ = mat.manual_grad(free_params)
        assert (grad[0].diag() == 0.0).all()

    def test_manual_grad_off_diagonal_uniform(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        grad, _ = mat.manual_grad(free_params)
        n = 3
        off_diag = grad[0][~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == off_diag[0]).all()

    def test_manual_grad_vs_auto_grad(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.2])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_grad_dispatches_to_manual(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.2])
        grad_default, _ = mat.grad(free_params)
        grad_manual, _ = mat.manual_grad(free_params)
        assert torch.allclose(grad_default, grad_manual)

    def test_fixed_rho(self):
        mat = EquicorrelationMatrix(3, param_specs={
            "rho": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(1.5, -0.5)]),
            }
        })
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 1
        assert mat.free_param_names == []

    def test_fixed_rho_grad(self):
        mat = EquicorrelationMatrix(3, param_specs={
            "rho": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(1.5, -0.5)]),
            }
        })
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_fixed_rho_call(self):
        mat = EquicorrelationMatrix(3, param_specs={
            "rho": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(1.5, -0.5)]),
            }
        })
        result = mat(torch.tensor([]))
        assert torch.allclose(result.diag(), torch.tensor(1.0))
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        expected_off = 0.25
        assert torch.allclose(off_diag, torch.tensor(expected_off))

    def test_intermediate_cache_hit(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        built = mat.build_params(free_params)
        assert mat.get_intermediates(built) is None
        mat(free_params)
        cache = mat.get_intermediates(built)
        assert cache is not None
        assert "v" in cache
        assert "i_n" in cache

    def test_intermediate_cache_reset(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        built = mat.build_params(free_params)
        mat(free_params)
        assert mat.get_intermediates(built) is not None
        mat.reset_intermediates()
        assert mat.get_intermediates(built) is None

    def test_dtype_float64(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_map_theta_to_v(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([1.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_inverse_matches_inverse_of_call(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.6931])
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)))
    def test_inverse_closed_form_structure(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.6931])
        rho = mat.build_params(free_params)
        expected_diag = 1 / (1 - rho) - rho / ((1 - rho) * (1 + 2 * rho))
        expected_off = -rho / ((1 - rho) * (1 + 2 * rho))
        result = mat.inverse(free_params)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert torch.allclose(result.diag(), expected_diag.expand(n))
        assert torch.allclose(off_diag, expected_off.expand_as(off_diag))
    def test_inverse_product_with_matrix_is_identity(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.6931])
        assert torch.allclose(mat(free_params) @ mat.inverse(free_params), torch.eye(3), atol=1e-6)
    def test_inverse_rho_zero_is_identity(self):
        mat = EquicorrelationMatrix(3)
        x = log(1 / 3 / (2 / 3))
        assert torch.allclose(mat.inverse(torch.tensor([x])), torch.eye(3))
    def test_inverse_n2_reduces_to_reciprocal_one_minus_rho_squared(self):
        mat = EquicorrelationMatrix(2)
        free_params = torch.tensor([0.3])
        rho = mat.build_params(free_params)
        result = mat.inverse(free_params)
        assert torch.allclose(result.diag(), (1 / (1 - rho ** 2)).expand(2))
        assert torch.allclose(result[0, 1], -rho / (1 - rho ** 2))
    def test_inverse_near_rho_min(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([-6.0])
        rho = mat.build_params(free_params)
        assert rho > mat.rho_min
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)), atol=1e-3)
    def test_inverse_uses_default_free_params(self):
        mat = EquicorrelationMatrix(3)
        assert torch.allclose(mat.inverse(), mat.inverse(mat.free_param_defaults))
    def test_inverse_accepts_dict_of_free_params(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5])
        assert torch.allclose(mat.inverse(mat._to_free_param_dict(free_params)), mat.inverse(free_params))
    def test_inverse_fixed_rho(self):
        mat = EquicorrelationMatrix(3, param_specs={
            "rho": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformChain([TransformSigmoid(), TransformScaleShift(1.5, -0.5)]),
            }
        })
        assert torch.allclose(mat.inverse(torch.tensor([])), torch.linalg.inv(mat(torch.tensor([]))))
    def test_inverse_dtype_float64(self):
        mat = EquicorrelationMatrix(3)
        assert mat.inverse(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64
    def test_inverse_follows_dtype_across_interleaved_calls(self):
        mat = EquicorrelationMatrix(3)
        mat(torch.tensor([0.5], dtype=torch.float64))
        assert mat.inverse(torch.tensor([0.5], dtype=torch.float32)).dtype == torch.float32
    def test_inverse_reads_the_cache_without_repopulating_it(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5])
        mat(free_params)
        cache = mat.get_intermediates(mat.build_params(free_params))
        mat.inverse(free_params)
        assert mat.get_intermediates(mat.build_params(free_params)) is cache
    def test_inverse_populates_the_cache_alone(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5])
        mat.inverse(free_params)
        cache = mat.get_intermediates(mat.build_params(free_params))
        assert cache is not None
        assert "i_n" in cache
        assert "j_n" in cache
    def test_inverse_gradients_flow_to_free_params(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)
    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        mat = EquicorrelationMatrix(3)
        free_params = torch.tensor([0.5], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        mat.reset_intermediates()
        torch.linalg.inv(mat(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)
    def test_inverse_validation_is_not_bypassed(self):
        mat = EquicorrelationMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.inverse([0.5])
        with pytest.raises(ValueError, match="length 1"):
            mat.inverse(torch.tensor([0.5, 1.0]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.inverse({})
    def test_repr(self):
        mat = EquicorrelationMatrix(3)
        r = repr(mat)
        assert "EquicorrelationMatrix" in r
