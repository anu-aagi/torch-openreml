import torch
import pytest
import pandas as pd
from math import log
from torch_openreml.covariance import AR1Matrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import (
    TransformExpPow2,
    TransformChain,
    TransformScaleShift,
    TransformSigmoid,
)


class TestAR1Matrix:
    """Tests for the AR(1) covariance matrix."""

    def test_constructor(self):
        mat = AR1Matrix(4)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = AR1Matrix(4)
        assert mat.shape == (4, 4)

    def test_shape_from_levels(self):
        assert AR1Matrix(["a", "b", "a", "c"]).shape == (3, 3)
        assert AR1Matrix(("a", "b", "a", "c")).shape == (3, 3)
        assert AR1Matrix(pd.Series(["a", "b", "a", "c"])).shape == (3, 3)

    def test_levels_equivalent_to_distinct_count(self):
        from_levels = AR1Matrix(["a", "b", "a", "c"])
        from_int = AR1Matrix(3)
        free_params = torch.tensor([0.5, 0.0])
        assert from_levels.shape == from_int.shape
        assert from_levels.param_names == from_int.param_names
        assert torch.equal(from_levels(free_params), from_int(free_params))

    def test_rejects_empty_levels(self):
        with pytest.raises(ValueError, match="positive int"):
            AR1Matrix([])

    def test_default_params(self):
        mat = AR1Matrix(4)
        assert mat.num_params == 2
        assert mat.num_free_params == 2
        assert mat.num_fixed_params == 0
        assert mat.param_names == ["sigma^2", "rho"]
        assert mat.free_param_names == ["sigma^2", "rho"]

    def test_call_diagonal(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 0.0])
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0))

    def test_call_rho_zero_is_diagonal(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 0.0])
        result = mat(free_params)
        n = 4
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == 0.0).all()

    def test_call_geometric_decay(self):
        mat = AR1Matrix(4)
        x = log(0.75 / 0.25)  # logit(0.75) ≈ 1.099
        free_params = torch.tensor([0.0, x])
        result = mat(free_params)
        sigma2 = torch.exp(torch.tensor(0.0))
        rho = 0.5
        for i in range(4):
            for j in range(4):
                expected = sigma2 * (rho ** abs(i - j))
                assert torch.allclose(result[i, j], expected)

    def test_call_symmetric(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.5, 1.0])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_rho_positive_decay(self):
        mat = AR1Matrix(4)
        x = log(0.75 / 0.25)
        free_params = torch.tensor([0.0, x])
        result = mat(free_params)
        assert result[0, 1] > result[0, 2]
        assert result[0, 2] > result[0, 3]

    def test_call_n1(self):
        mat = AR1Matrix(1)
        free_params = torch.tensor([0.0, 0.0])
        result = mat(free_params)
        assert result.shape == (1, 1)
        assert result[0, 0] > 0

    def test_diff_matrix(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 0.0])
        mat(free_params)
        built = mat.build_params(free_params)
        diff = mat.get_intermediates(built)["diff"]
        for i in range(4):
            for j in range(4):
                assert diff[i, j] == abs(i - j)

    def test_manual_grad_shape(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (2, 4, 4)
        assert grad_names == ["sigma^2", "rho"]

    def test_manual_grad_sigma_structure(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        grad, _ = mat.manual_grad(free_params)
        grad_sigma = grad[0]
        result = mat(free_params)
        assert torch.allclose(grad_sigma, result * 2.0)

    def test_manual_grad_rho_diagonal_zero(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        grad, _ = mat.manual_grad(free_params)
        grad_rho = grad[1]
        assert (grad_rho.diag() == 0.0).all()

    def test_manual_grad_vs_auto_grad(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.1, 0.2])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_manual_grad_rho_zero(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 0.0])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        lag_one = 0.5 * torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        assert torch.isfinite(manual).all()
        assert torch.allclose(manual[1], lag_one)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_only_sigma_free(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["sigma^2"]

    def test_only_sigma_free_grad(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        free_params = torch.tensor([0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 4, 4)
        assert grad_names == ["sigma^2"]

    def test_only_rho_free(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        assert mat.num_free_params == 1
        assert mat.free_param_names == ["rho"]

    def test_only_rho_free_grad(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        free_params = torch.tensor([1.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 4, 4)
        assert grad_names == ["rho"]

    def test_both_fixed(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_intermediate_cache_hit(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        built = mat.build_params(free_params)
        assert mat.get_intermediates(built) is None
        mat(free_params)
        cache = mat.get_intermediates(built)
        assert cache is not None
        assert "rho_power" in cache
        assert "diff" in cache

    def test_intermediate_cache_reset(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        built = mat.build_params(free_params)
        mat(free_params)
        assert mat.get_intermediates(built) is not None
        mat.reset_intermediates()
        assert mat.get_intermediates(built) is None

    def test_dtype_float64(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_map_theta_to_v(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.0, 1.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_inverse_matches_inverse_of_call(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)), atol=1e-6)
    def test_inverse_closed_form_structure(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        sigma2, rho = mat.build_params(free_params)
        d = 1 / (sigma2 * (1 - rho ** 2))
        expected_diag = torch.stack([d, d * (1 + rho ** 2), d * (1 + rho ** 2), d]).squeeze(-1)
        result = mat.inverse(free_params)
        assert torch.allclose(result.diag(), expected_diag)
        assert torch.allclose(result[0, 1], -rho * d)
        assert torch.allclose(result[3, 2], -rho * d)
        assert result[0, 2] == 0.0
        assert result[0, 3] == 0.0
    def test_inverse_symmetric(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        assert torch.equal(mat.inverse(free_params), mat.inverse(free_params).T)
    def test_inverse_product_with_matrix_is_identity(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        assert torch.allclose(mat(free_params) @ mat.inverse(free_params), torch.eye(4), atol=1e-6)
    def test_inverse_negative_rho(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, -1.0])
        rho = mat.build_params(free_params)[1]
        result = mat.inverse(free_params)
        assert rho < 0
        assert result[0, 1] > 0
        assert torch.allclose(result, torch.linalg.inv(mat(free_params)), atol=1e-6)
    def test_inverse_n1(self):
        mat = AR1Matrix(1)
        free_params = torch.tensor([0.4, 1.0])
        sigma2 = mat.build_params(free_params)[0]
        assert torch.allclose(mat.inverse(free_params), torch.eye(1) / sigma2)
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)))
    def test_inverse_n2_both_diagonal_entries_are_corners(self):
        mat = AR1Matrix(2)
        free_params = torch.tensor([0.4, 1.0])
        sigma2, rho = mat.build_params(free_params)
        result = mat.inverse(free_params)
        assert torch.allclose(result.diag(), (1 / (sigma2 * (1 - rho ** 2))).expand(2))
        assert torch.allclose(result[0, 1], -rho / (sigma2 * (1 - rho ** 2)))
    def test_inverse_uses_default_free_params(self):
        mat = AR1Matrix(4)
        assert torch.allclose(mat.inverse(), mat.inverse(mat.free_param_defaults))
    def test_inverse_accepts_dict_of_free_params(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        assert torch.allclose(mat.inverse(mat._to_free_param_dict(free_params)), mat.inverse(free_params))
    def test_inverse_only_sigma_free(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        free_params = torch.tensor([0.4])
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)), atol=1e-6)
    def test_inverse_only_rho_free(self):
        mat = AR1Matrix(4, param_specs={
            "sigma^2": {"fixed": True, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "rho": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformChain([
                TransformSigmoid(), TransformScaleShift(2.0, -1.0)
            ])},
        })
        free_params = torch.tensor([1.0])
        assert torch.allclose(mat.inverse(free_params), torch.linalg.inv(mat(free_params)), atol=1e-6)
    def test_inverse_dtype_float64(self):
        mat = AR1Matrix(4)
        assert mat.inverse(torch.tensor([0.4, 1.0], dtype=torch.float64)).dtype == torch.float64
    def test_inverse_follows_dtype_across_interleaved_calls(self):
        mat = AR1Matrix(4)
        mat(torch.tensor([0.4, 1.0], dtype=torch.float64))
        assert mat.inverse(torch.tensor([0.4, 1.0], dtype=torch.float32)).dtype == torch.float32
    def test_inverse_reads_the_cache_without_repopulating_it(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        mat(free_params)
        cache = mat.get_intermediates(mat.build_params(free_params))
        mat.inverse(free_params)
        assert mat.get_intermediates(mat.build_params(free_params)) is cache
    def test_inverse_populates_the_cache_alone(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0])
        mat.inverse(free_params)
        cache = mat.get_intermediates(mat.build_params(free_params))
        assert cache is not None
        assert "sigma2" in cache
        assert "rho" in cache
    def test_inverse_gradients_flow_to_free_params(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        assert free_params.grad is not None
        assert torch.all(free_params.grad != 0.0)
    def test_inverse_gradient_matches_autograd_of_inverse_of_call(self):
        mat = AR1Matrix(4)
        free_params = torch.tensor([0.4, 1.0], requires_grad=True)
        mat.inverse(free_params).sum().backward()
        closed_form_grad = free_params.grad.clone()
        free_params.grad = None
        mat.reset_intermediates()
        torch.linalg.inv(mat(free_params)).sum().backward()
        assert torch.allclose(closed_form_grad, free_params.grad, atol=1e-5)
    def test_inverse_validation_is_not_bypassed(self):
        mat = AR1Matrix(4)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.inverse([0.4, 1.0])
        with pytest.raises(ValueError, match="length 2"):
            mat.inverse(torch.tensor([0.4]))
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat.inverse({})
    def test_repr(self):
        mat = AR1Matrix(4)
        r = repr(mat)
        assert "AR1Matrix" in r
