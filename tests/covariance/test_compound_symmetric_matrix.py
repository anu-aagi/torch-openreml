import torch
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

    def test_repr(self):
        mat = CompoundSymmetricMatrix(3)
        r = repr(mat)
        assert "CompoundSymmetricMatrix" in r
