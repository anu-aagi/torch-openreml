import torch
import pytest
from torch_openreml.covariance import IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


class TestIdentityMatrix:
    """Tests for the fixed identity covariance matrix."""

    def test_constructor(self):
        mat = IdentityMatrix(3)
        assert isinstance(mat, Matrix)

    def test_shape(self):
        mat = IdentityMatrix(5)
        assert mat.shape == (5, 5)

    def test_call_returns_identity(self):
        mat = IdentityMatrix(3)
        expected = torch.eye(3)
        assert torch.equal(mat(), expected)

    def test_call_ignores_arguments(self):
        mat = IdentityMatrix(3)
        expected = torch.eye(3)
        assert torch.equal(mat(torch.tensor([1.0, 2.0, 3.0])), expected)
        assert torch.equal(mat(dummy=42), expected)

    def test_call_diagonal_is_one(self):
        mat = IdentityMatrix(4)
        result = mat()
        assert (result.diag() == 1.0).all()

    def test_call_off_diagonal_is_zero(self):
        mat = IdentityMatrix(4)
        result = mat()
        n = 4
        off_diag = result[~torch.eye(n, dtype=torch.bool)]
        assert (off_diag == 0.0).all()

    def test_call_symmetric(self):
        mat = IdentityMatrix(4)
        result = mat()
        assert torch.equal(result, result.T)

    def test_call_n1(self):
        mat = IdentityMatrix(1)
        assert torch.equal(mat(), torch.tensor([[1.0]]))

    def test_dtype_float32(self):
        mat = IdentityMatrix(3, dtype=torch.float32)
        assert mat().dtype == torch.float32

    def test_dtype_float64(self):
        mat = IdentityMatrix(3, dtype=torch.float64)
        assert mat().dtype == torch.float64

    def test_device_cpu(self):
        mat = IdentityMatrix(3, device="cpu")
        assert mat().device.type == "cpu"

    def test_param_specs_empty(self):
        mat = IdentityMatrix(5)
        assert mat.param_specs == {}

    def test_num_params_zero(self):
        mat = IdentityMatrix(5)
        assert mat.num_params == 0

    def test_num_free_params_zero(self):
        mat = IdentityMatrix(5)
        assert mat.num_free_params == 0

    def test_num_fixed_params_zero(self):
        mat = IdentityMatrix(5)
        assert mat.num_fixed_params == 0

    def test_free_param_names_empty(self):
        mat = IdentityMatrix(5)
        assert mat.free_param_names == []

    def test_fixed_param_names_empty(self):
        mat = IdentityMatrix(5)
        assert mat.fixed_param_names == []

    def test_grad_mode_default(self):
        mat = IdentityMatrix(3)
        assert mat.grad_mode == "default"

    def test_grad_returns_none_empty(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_auto_grad_returns_none_empty(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.auto_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_manual_grad_raises(self):
        mat = IdentityMatrix(3)
        with pytest.raises(NotImplementedError):
            mat.manual_grad(torch.tensor([]))

    def test_build_params_empty_tensor(self):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]))
        assert torch.equal(result, torch.tensor([]))

    def test_build_params_dict_format(self):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]), out_format="dict")
        assert result == {}

    def test_map_theta_to_v(self):
        mat = IdentityMatrix(3)
        expected = torch.eye(3)
        assert torch.equal(mat.map_theta_to_v(torch.tensor([])), expected)

    def test_map_theta_to_dv(self):
        mat = IdentityMatrix(3)
        assert mat.map_theta_to_dv(torch.tensor([])) is None

    def test_repr(self):
        mat = IdentityMatrix(3)
        r = repr(mat)
        assert "IdentityMatrix" in r
        assert "(3, 3)" in r

    def test_repr_dict(self):
        mat = IdentityMatrix(3)
        assert mat.repr_dict["shape"] == (3, 3)
        assert mat.repr_dict["param_specs"] == {}
