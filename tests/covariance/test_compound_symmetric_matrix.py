import torch
import pytest
import pandas as pd
from math import log
from torch_openreml.covariance import CompoundSymmetricMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExpPow2,
    TransformChain,
    TransformScaleShift,
    TransformSigmoid,
)


class TestCompoundSymmetricMatrix:
    """Tests for the compound symmetric covariance matrix."""

    def test_constructor(self):
        mat = CompoundSymmetricMatrix(3)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = CompoundSymmetricMatrix(3)
        assert mat.shape == (3, 3)

    def test_rho_min(self):
        mat = CompoundSymmetricMatrix(3)
        assert mat.rho_min == -0.5

    def test_shape_from_levels(self):
        assert CompoundSymmetricMatrix(["a", "b", "a", "c"]).shape == (3, 3)
        assert CompoundSymmetricMatrix(("a", "b", "a", "c")).shape == (3, 3)
        assert CompoundSymmetricMatrix(pd.Series(["a", "b", "a", "c"])).shape == (3, 3)

    def test_levels_equivalent_to_distinct_count(self):
        from_levels = CompoundSymmetricMatrix(["a", "b", "a", "c"])
        from_int = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.5, 0.0])
        assert from_levels.shape == from_int.shape
        assert from_levels.rho_min == from_int.rho_min
        assert from_levels.param_names == from_int.param_names
        assert torch.equal(from_levels(free_params), from_int(free_params))

    def test_rejects_empty_levels(self):
        with pytest.raises(ValueError, match="greater than 1"):
            CompoundSymmetricMatrix([])

    def test_rho_min_n2(self):
        mat = CompoundSymmetricMatrix(2)
        assert mat.rho_min == -1.0

    def test_default_params(self):
        mat = CompoundSymmetricMatrix(3)
        assert mat.num_params == 2
        assert mat.num_free_params == 2
        assert mat.num_fixed_params == 0
        assert mat.param_names == ["sigma^2", "rho"]
        assert mat.free_param_names == ["sigma^2", "rho"]

    def test_call_diagonal(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, -10.0])
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0), atol=1e-4)

    def test_call_off_diagonal(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        result = mat(free_params)
        sigma2 = torch.exp(torch.tensor(0.0))
        rho_min = -0.5
        rho = torch.sigmoid(torch.tensor(0.0)) * (1 - rho_min) + rho_min
        expected_off = sigma2 * rho
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert torch.allclose(off_diag, expected_off.expand_as(off_diag))

    def test_call_symmetric(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.5, 0.0])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 1.0])
        result = mat(free_params)
        assert result[0, 0] == result[1, 1] == result[2, 2]
        assert result[0, 1] == result[0, 2] == result[1, 2]

    def test_rho_in_bounds(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 100.0])
        result = mat(free_params)
        sigma2 = result[0, 0]
        off_diag = result[0, 1]
        rho = off_diag / sigma2
        assert rho > mat.rho_min
        assert rho <= 1.0

    def test_manual_grad_shape(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (2, 3, 3)
        assert grad_names == ["sigma^2", "rho"]

    def test_manual_grad_sigma_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        grad, _ = mat.manual_grad(free_params)
        sigma2 = mat.build_params(free_params)[0]
        grad_sigma = grad[0]
        assert torch.allclose(grad_sigma, mat(free_params) * 2)

    def test_manual_grad_rho_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        grad, _ = mat.manual_grad(free_params)
        grad_rho = grad[1]
        assert (grad_rho.diag() == 0.0).all()
        n = 3
        off_diag = grad_rho[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == off_diag[0]).all()

    def test_manual_grad_vs_auto_grad(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_only_sigma_free(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["sigma^2"]

    def test_only_sigma_free_grad(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        free_params = torch.tensor([0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["sigma^2"]

    def test_only_rho_free(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["rho"]

    def test_only_rho_free_grad(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        free_params = torch.tensor([0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["rho"]

    def test_both_fixed(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        built = mat.build_params(free_params)
        assert mat.get_intermediates(built) is None
        mat(free_params)
        cache = mat.get_intermediates(built)
        assert cache is not None
        assert "sigma2" in cache
        assert "rho_mat" in cache

    def test_intermediate_cache_reset(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        built = mat.build_params(free_params)
        mat(free_params)
        assert mat.get_intermediates(built) is not None
        mat.reset_intermediates()
        assert mat.get_intermediates(built) is None

    def test_dtype_float64(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_map_theta_to_v(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.0, 0.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_inverse_matches_inverse_of_call(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7])
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)))
    def test_inverse_closed_form_structure(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7])
        sigma2, rho = mat.build_params(free_params)
        expected_diag = (1 / sigma2) * (1 / (1 - rho) - rho / ((1 - rho) * (1 + 2 * rho)))
        expected_off = -(1 / sigma2) * rho / ((1 - rho) * (1 + 2 * rho))
        result = mat.inverse(free_params)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert torch.allclose(result.diag(), expected_diag.expand(n))
        assert torch.allclose(off_diag, expected_off.expand_as(off_diag))
    def test_inverse_product_with_matrix_is_identity(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7])
        assert torch.allclose(mat(free_params) @ mat.inverse(free_params), torch.eye(3), atol=1e-6)
    def test_inverse_scales_by_reciprocal_sigma_squared(self):
        mat = CompoundSymmetricMatrix(3)
        base = mat.inverse(torch.tensor([0.0, 0.7]))
        sigma2 = mat.build_params(torch.tensor([1.0, 0.7]))[0]
        assert torch.allclose(mat.inverse(torch.tensor([1.0, 0.7])), base / sigma2)
    def test_inverse_rho_zero_is_scaled_identity(self):
        mat = CompoundSymmetricMatrix(3)
        x = log(1 / 3 / (2 / 3))
        free_params = torch.tensor([0.4, x])
        sigma2 = mat.build_params(free_params)[0]
        assert torch.allclose(mat.inverse(free_params), torch.eye(3) / sigma2)
    def test_inverse_n2_reduces_to_reciprocal_one_minus_rho_squared(self):
        mat = CompoundSymmetricMatrix(2)
        free_params = torch.tensor([0.4, 0.7])
        sigma2, rho = mat.build_params(free_params)
        result = mat.inverse(free_params)
        assert torch.allclose(result.diag(), ((1 / sigma2) / (1 - rho ** 2)).expand(2))
        assert torch.allclose(result[0, 1], -(1 / sigma2) * rho / (1 - rho ** 2))
    def test_inverse_near_rho_min(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, -6.0])
        rho = mat.build_params(free_params)[1]
        assert rho > mat.rho_min
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)), atol=1e-3)
    def test_inverse_uses_default_free_params(self):
        mat = CompoundSymmetricMatrix(3)
        assert torch.allclose(mat.inverse(), mat.inverse(mat.free_param_defaults))
    def test_inverse_accepts_dict_of_free_params(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7])
        assert torch.allclose(mat.inverse(mat._to_free_param_dict(free_params)), mat.inverse(free_params))
    def test_inverse_only_sigma_free(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        free_params = torch.tensor([0.4])
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)))
    def test_inverse_only_rho_free(self):
        mat = CompoundSymmetricMatrix(3, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(1.5, -0.5)
            ])},
        })
        free_params = torch.tensor([0.7])
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)))
    def test_inverse_dtype_float64(self):
        mat = CompoundSymmetricMatrix(3)
        assert mat.inverse(torch.tensor([0.4, 0.7], dtype=torch.float64)).dtype == torch.float64
    def test_inverse_follows_dtype_across_interleaved_calls(self):
        mat = CompoundSymmetricMatrix(3)
        mat(torch.tensor([0.4, 0.7], dtype=torch.float64))
        assert mat.inverse(torch.tensor([0.4, 0.7], dtype=torch.float32)).dtype == torch.float32
    def test_inverse_reads_the_cache_without_repopulating_it(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7])
        mat(free_params)
        cache = mat.get_intermediates(mat.build_params(free_params))
        mat.inverse(free_params)
        assert mat.get_intermediates(mat.build_params(free_params)) is cache
    def test_inverse_populates_the_cache_alone(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7])
        mat.inverse(free_params)
        cache = mat.get_intermediates(mat.build_params(free_params))
        assert cache is not None
        assert "sigma2" in cache
        assert "i_n" in cache
        assert "j_n" in cache
    def test_inverse_gradients_flow_to_free_params(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)
    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        mat = CompoundSymmetricMatrix(3)
        free_params = torch.tensor([0.4, 0.7], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        mat.reset_intermediates()
        torch.linalg.inv(mat(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-6)
    def test_inverse_validation_is_not_bypassed(self):
        mat = CompoundSymmetricMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.inverse([0.4, 0.7])
        with pytest.raises(ValueError, match="length 2"):
            mat.inverse(torch.tensor([0.4]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.inverse({})
    def test_repr(self):
        mat = CompoundSymmetricMatrix(3)
        r = repr(mat)
        assert "CompoundSymmetricMatrix" in r
