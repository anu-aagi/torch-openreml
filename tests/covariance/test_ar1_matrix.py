import torch
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

    def test_repr(self):
        mat = AR1Matrix(4)
        r = repr(mat)
        assert "AR1Matrix" in r
