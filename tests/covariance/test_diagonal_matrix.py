import torch
import pytest
from torch_openreml.covariance import DiagonalMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformExp


class TestDiagonalMatrix:
    """Tests for the diagonal covariance matrix."""

    def test_constructor(self):
        mat = DiagonalMatrix(3)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = DiagonalMatrix(3)
        assert mat.shape == (3, 3)

    def test_default_params(self):
        mat = DiagonalMatrix(3)
        assert mat.num_params == 3
        assert mat.num_free_params == 3
        assert mat.num_fixed_params == 0
        assert mat.param_names == ["sigma^2_0", "sigma^2_1", "sigma^2_2"]
        assert mat.free_param_names == ["sigma^2_0", "sigma^2_1", "sigma^2_2"]

    def test_all_fixed_param_config(self):
        mat = DiagonalMatrix(3, param_specs={
            f"sigma^2_{i}": {
                "fixed": True,
                "default": torch.tensor([float(i)]),
                "trans": TransformExpPow2(),
            } for i in range(3)
        })
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 3
        assert mat.free_param_names == []

    def test_call_diagonal(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        result = mat(free_params)
        expected_diag = torch.exp(2.0 * free_params)
        assert torch.allclose(result.diag(), expected_diag)

    def test_call_off_diagonal_zero(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        result = mat(free_params)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == 0.0).all()

    def test_call_symmetric(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_positive_variances(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([-5.0, -10.0, 0.0])
        result = mat(free_params)
        assert (result.diag() > 0).all()

    def test_call_n1(self):
        mat = DiagonalMatrix(1)
        free_params = torch.tensor([0.0])
        result = mat(free_params)
        assert result.shape == (1, 1)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_call_shape_correct(self, n):
        mat = DiagonalMatrix(n)
        free_params = torch.zeros(n)
        assert mat(free_params).shape == (n, n)

    def test_manual_grad_shape(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (3, 3, 3)
        assert grad_names == ["sigma^2_0", "sigma^2_1", "sigma^2_2"]

    def test_manual_grad_nonzero_only_at_iii(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        grad, _ = mat.manual_grad(free_params)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == j == k:
                        assert grad[i, j, k] != 0.0
                    else:
                        assert grad[i, j, k] == 0.0

    def test_manual_grad_vs_auto_grad(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.1, 0.2, 0.3])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_grad_dispatches_to_manual(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.1, 0.2, 0.3])
        grad_default, _ = mat.grad(free_params)
        grad_manual, _ = mat.manual_grad(free_params)
        assert torch.allclose(grad_default, grad_manual)

    def test_manual_grad_all_fixed(self):
        mat = DiagonalMatrix(3, param_specs={
            f"sigma^2_{i}": {
                "fixed": True,
                "default": torch.tensor([0.0]),
                "trans": TransformExpPow2(),
            } for i in range(3)
        })
        grad, grad_names = mat.manual_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_mixed_free_fixed_call(self):
        mat = DiagonalMatrix(3, param_specs={
            "sigma^2_0": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "sigma^2_1": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExpPow2()},
            "sigma^2_2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
        })
        free_params = torch.tensor([0.0, 0.0])
        result = mat(free_params)
        assert torch.allclose(result.diag()[0], torch.tensor(1.0))
        assert torch.allclose(result.diag()[1], torch.exp(torch.tensor(2.0)))
        assert torch.allclose(result.diag()[2], torch.tensor(1.0))

    def test_mixed_free_fixed_param_properties(self):
        mat = DiagonalMatrix(3, param_specs={
            "sigma^2_0": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
            "sigma^2_1": {"fixed": True, "default": torch.tensor([1.0]), "trans": TransformExpPow2()},
            "sigma^2_2": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
        })
        assert mat.num_params == 3
        assert mat.num_free_params == 2
        assert mat.num_fixed_params == 1
        assert mat.free_param_names == ["sigma^2_0", "sigma^2_2"]
        assert mat.fixed_param_names == ["sigma^2_1"]
        assert mat.free_param_index == [0, 2]

    def test_custom_transforms(self):
        mat = DiagonalMatrix(2, param_specs={
            "sigma^2_0": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExp()},
            "sigma^2_1": {"fixed": False, "default": torch.tensor([0.0]), "trans": TransformExpPow2()},
        })
        free_params = torch.tensor([1.0, 1.0])
        result = mat(free_params)
        assert torch.allclose(result.diag()[0], torch.exp(torch.tensor(1.0)))
        assert torch.allclose(result.diag()[1], torch.exp(torch.tensor(2.0)))

    def test_build_params_tensor(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        built = mat.build_params(free_params)
        assert built.numel() == 3
        assert (built > 0).all()

    def test_build_params_dict(self):
        mat = DiagonalMatrix(2)
        free_params = torch.tensor([0.0, 0.5])
        result = mat.build_params(free_params, out_format="dict")
        assert "sigma^2_0" in result
        assert "sigma^2_1" in result

    def test_dtype_float32(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        assert mat(free_params).dtype == torch.float32

    def test_dtype_float64(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_map_theta_to_v(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(mat.map_theta_to_v(free_params), mat(free_params))

    def test_map_theta_to_dv(self):
        mat = DiagonalMatrix(3)
        free_params = torch.tensor([0.0, 0.5, 1.0])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        mat = DiagonalMatrix(3)
        r = repr(mat)
        assert "DiagonalMatrix" in r
