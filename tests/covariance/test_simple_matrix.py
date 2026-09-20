import torch
import pytest
from torch_openreml.covariance import SimpleMatrix
from torch_openreml.covariance.matrix import Matrix


def diagonal_call(free_params):
    return torch.diag(free_params)


class TestSimpleMatrix:
    """Tests for the function-backed covariance matrix."""

    def test_constructor(self):
        mat = SimpleMatrix(3, call=diagonal_call)
        assert isinstance(mat, Matrix)

    def test_call_requires_callable(self):
        with pytest.raises(ValueError, match="'call' must be provided"):
            SimpleMatrix(3, call=None)

    def test_call_with_tensor(self):
        mat = SimpleMatrix(3, call=diagonal_call)
        result = mat(torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(result, torch.diag(torch.tensor([1.0, 2.0, 3.0])))

    def test_call_with_defaults(self):
        """With no input, defaults are built into a tensor before the call."""
        mat = SimpleMatrix(3, call=diagonal_call, default=1.0)
        result = mat()
        assert torch.equal(result, torch.eye(3))

    def test_call_with_dict(self):
        """A parameter dict is resolved to a tensor before the call."""
        mat = SimpleMatrix(3, call=diagonal_call, default=1.0)
        result = mat({
            "theta_0": torch.tensor([1.0]),
            "theta_1": torch.tensor([2.0]),
            "theta_2": torch.tensor([3.0]),
        })
        assert torch.equal(result, torch.diag(torch.tensor([1.0, 2.0, 3.0])))

    def test_call_rejects_wrong_length(self):
        mat = SimpleMatrix(3, call=diagonal_call)
        with pytest.raises(ValueError, match="length 3"):
            mat(torch.tensor([1.0, 2.0]))

    def test_call_rejects_missing_dict_key(self):
        mat = SimpleMatrix(3, call=diagonal_call)
        with pytest.raises(ValueError, match="Missing free parameters"):
            mat({"theta_0": torch.tensor([1.0])})

    def test_call_follows_input_dtype(self):
        """Rule 1: dtype and device of the input win over the defaults."""
        mat = SimpleMatrix(3, call=diagonal_call, default=1.0)
        result = mat(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert result.dtype == torch.float64

    def test_call_follows_default_dtype(self):
        """Rule 2: with no input, the parameter defaults decide dtype."""
        mat = SimpleMatrix(3, call=diagonal_call, default=torch.tensor([1.0], dtype=torch.float64))
        assert mat().dtype == torch.float64

    def test_call_input_overrides_default_dtype(self):
        mat = SimpleMatrix(3, call=diagonal_call, default=torch.tensor([1.0], dtype=torch.float64))
        assert mat(torch.tensor([1.0, 2.0, 3.0])).dtype == torch.float32

    def test_grad_auto_with_tensor(self):
        mat = SimpleMatrix(3, call=diagonal_call)
        params = torch.tensor([1.0, 2.0, 3.0])
        grad, grad_names = mat.grad(params)
        expected = torch.stack([torch.diag(torch.eye(3)[i]) for i in range(3)])
        assert torch.equal(grad, expected)
        assert grad_names == ["theta_0", "theta_1", "theta_2"]

    def test_grad_auto_with_defaults(self):
        mat = SimpleMatrix(3, call=diagonal_call, default=1.0)
        grad, grad_names = mat.grad()
        assert grad.dtype == torch.float32
        assert grad_names == mat.free_param_names

    def test_grad_auto_with_dict(self):
        mat = SimpleMatrix(3, call=diagonal_call, default=1.0)
        grad, _ = mat.grad({
            "theta_0": torch.tensor([1.0]),
            "theta_1": torch.tensor([2.0]),
            "theta_2": torch.tensor([3.0]),
        })
        expected = torch.stack([torch.diag(torch.eye(3)[i]) for i in range(3)])
        assert torch.equal(grad, expected)

    def test_grad_follows_input_dtype(self):
        mat = SimpleMatrix(3, call=diagonal_call)
        grad, _ = mat.grad(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert grad.dtype == torch.float64

    def test_manual_grad(self):
        def manual_grad(free_params):
            return torch.eye(3).expand(3, 3, 3), ["theta_0", "theta_1", "theta_2"]

        mat = SimpleMatrix(3, call=diagonal_call, manual_grad=manual_grad)
        grad, grad_names = mat.grad(torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(grad, torch.eye(3).expand(3, 3, 3))
        assert grad_names == ["theta_0", "theta_1", "theta_2"]

    def test_manual_grad_with_defaults(self):
        def manual_grad(free_params):
            assert isinstance(free_params, torch.Tensor)
            return torch.eye(3).expand(3, 3, 3), ["theta_0", "theta_1", "theta_2"]

        mat = SimpleMatrix(3, call=diagonal_call, manual_grad=manual_grad)
        grad, _ = mat.grad()
        assert torch.equal(grad, torch.eye(3).expand(3, 3, 3))

    def test_missing_manual_grad_raises_in_manual_mode(self):
        """Without a manual_grad, the manual path raises and 'default' falls back."""
        mat = SimpleMatrix(3, call=diagonal_call)
        mat.grad_mode = "manual"
        with pytest.raises(NotImplementedError):
            mat.grad(torch.tensor([1.0, 2.0, 3.0]))

    def test_zero_free_params(self):
        mat = SimpleMatrix(0, call=lambda p: torch.eye(2))
        assert mat.num_free_params == 0
        assert mat().shape == (2, 2)
        assert mat.grad() == (None, [])
