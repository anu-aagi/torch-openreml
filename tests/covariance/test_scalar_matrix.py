import torch
import pytest
from torch_openreml.covariance import ScalarMatrix
from torch_openreml.covariance.matrix import Matrix
from torch_openreml.covariance.transform import TransformExpPow2, TransformExp


class TestScalarMatrix:
    """Tests for the scaled identity covariance matrix."""

    def test_constructor(self):
        mat = ScalarMatrix(3)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = ScalarMatrix(3)
        assert mat.shape == (3, 3)

    def test_default_param_specs(self):
        mat = ScalarMatrix(3)
        assert "sigma^2" in mat.param_specs
        assert mat.param_specs["sigma^2"]["fixed"] is False
        assert torch.equal(
            mat.param_specs["sigma^2"]["default"], torch.tensor([0.0])
        )
        assert isinstance(mat.param_specs["sigma^2"]["trans"], TransformExpPow2)

    def test_num_params(self):
        mat = ScalarMatrix(3)
        assert mat.num_params == 1
        assert mat.num_free_params == 1
        assert mat.num_fixed_params == 0

    def test_param_names(self):
        mat = ScalarMatrix(3)
        assert mat.param_names == ["sigma^2"]
        assert mat.free_param_names == ["sigma^2"]
        assert mat.fixed_param_names == []

    def test_call_diagonal(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0))

    def test_call_off_diagonal_zero(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([1.0])
        result = mat(free_params)
        n = 3
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == 0.0).all()

    def test_call_symmetric(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        assert torch.equal(mat(free_params), mat(free_params).T)

    def test_call_positive_variance(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([-5.0])
        result = mat(free_params)
        assert (result.diag() > 0).all()

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_call_different_n(self, n):
        mat = ScalarMatrix(n)
        free_params = torch.tensor([0.0])
        assert mat(free_params).shape == (n, n)

    def test_manual_grad_shape(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        grad, grad_names = mat.manual_grad(free_params)
        assert grad.shape == (1, 3, 3)
        assert grad_names == ["sigma^2"]

    def test_manual_grad_diagonal_only(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        grad, _ = mat.manual_grad(free_params)
        assert torch.allclose(grad[0].diag(), torch.tensor(2.0))
        n = 3
        off_diag = grad[0][~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == 0.0).all()

    def test_manual_grad_vs_auto_grad(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        manual, names_m = mat.manual_grad(free_params)
        auto, names_a = mat.auto_grad(free_params)
        assert torch.allclose(manual, auto)
        assert names_m == names_a

    def test_grad_dispatches_to_manual(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        grad_default, _ = mat.grad(free_params)
        grad_manual, _ = mat.manual_grad(free_params)
        assert torch.allclose(grad_default, grad_manual)

    def test_grad_mode_auto(self):
        mat = ScalarMatrix(3)
        mat.grad_mode = "auto"
        free_params = torch.tensor([0.5])
        grad_auto, _ = mat.grad(free_params)
        expected, _ = mat.auto_grad(free_params)
        assert torch.allclose(grad_auto, expected)

    def test_fixed_parameter_config(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {
                "fixed": True,
                "default": torch.tensor([1.0]),
                "trans": TransformExpPow2(),
            }
        })
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 1

    def test_fixed_parameter_call(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {
                "fixed": True,
                "default": torch.tensor([1.0]),
                "trans": TransformExpPow2(),
            }
        })
        result = mat(torch.tensor([]))
        expected_val = torch.exp(torch.tensor(2.0))
        assert torch.allclose(result.diag(), expected_val)
        assert torch.allclose(result, expected_val * torch.eye(3))

    def test_fixed_parameter_grad(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {
                "fixed": True,
                "default": torch.tensor([1.0]),
                "trans": TransformExpPow2(),
            }
        })
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_custom_transform(self):
        mat = ScalarMatrix(3, param_specs={
            "sigma^2": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformExp(),
            }
        })
        free_params = torch.tensor([0.0])
        result = mat(free_params)
        assert torch.allclose(result.diag(), torch.tensor(1.0))

    def test_custom_param_name(self):
        mat = ScalarMatrix(3, param_specs={
            "var": {
                "fixed": False,
                "default": torch.tensor([0.0]),
                "trans": TransformExpPow2(),
            }
        })
        assert mat.param_names == ["var"]
        assert mat.free_param_names == ["var"]

    def test_build_params_output(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        built = mat.build_params(free_params)
        assert built.numel() == 1
        assert built.item() > 0

    def test_build_params_dict_format(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        result = mat.build_params(free_params, out_format="dict")
        assert isinstance(result, dict)
        assert "sigma^2" in result
        expected_val = mat.build_params(free_params)
        assert torch.allclose(result["sigma^2"], expected_val.unsqueeze(-1))

    def test_trans_grad(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        tg = mat.trans_grad(free_params)
        assert torch.allclose(tg, torch.tensor([2.0]))

    def test_dtype_float32(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0], dtype=torch.float32)
        assert mat(free_params).dtype == torch.float32

    def test_dtype_float64(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0], dtype=torch.float64)
        assert mat(free_params).dtype == torch.float64

    def test_map_theta_to_v(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.0])
        expected = mat(free_params)
        assert torch.allclose(mat.map_theta_to_v(free_params), expected)

    def test_map_theta_to_dv(self):
        mat = ScalarMatrix(3)
        free_params = torch.tensor([0.5])
        expected, _ = mat.grad(free_params)
        assert torch.allclose(mat.map_theta_to_dv(free_params), expected)

    def test_repr(self):
        mat = ScalarMatrix(3)
        r = repr(mat)
        assert "ScalarMatrix" in r
        assert "sigma^2" in r
