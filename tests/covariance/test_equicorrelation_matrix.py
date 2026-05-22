import torch
import pytest
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

    def test_repr(self):
        mat = EquicorrelationMatrix(3)
        r = repr(mat)
        assert "EquicorrelationMatrix" in r
