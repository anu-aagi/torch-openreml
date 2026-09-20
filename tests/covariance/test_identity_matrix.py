import torch
import pytest
from torch_openreml.covariance import IdentityMatrix
from torch_openreml.covariance.matrix import Matrix


class TestIdentityMatrixConstructor:
    """Construction, shape, and the empty parameter surface."""

    def test_is_matrix_subclass(self):
        mat = IdentityMatrix(3)
        assert isinstance(mat, Matrix)

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_shape(self, n):
        mat = IdentityMatrix(n)
        assert mat.shape == (n, n)
        assert isinstance(mat.shape, tuple)

    def test_shape_is_stored_as_tuple(self):
        mat = IdentityMatrix(3)
        assert mat.shape == (3, 3)

    @pytest.mark.parametrize(
        "prop, expected",
        [
            ("param_specs", {}),
            ("param_names", []),
            ("free_param_names", []),
            ("fixed_param_names", []),
            ("free_param_index", []),
            ("fixed_param_index", []),
            ("param_defaults", {}),
            ("free_param_defaults", {}),
            ("fixed_param_defaults", {}),
            ("param_trans", {}),
            ("free_param_trans", {}),
            ("fixed_param_trans", {}),
        ],
    )
    def test_parameter_surface_is_empty(self, prop, expected):
        mat = IdentityMatrix(4)
        assert getattr(mat, prop) == expected

    def test_num_params_zero(self):
        mat = IdentityMatrix(4)
        assert mat.num_params == 0
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 0

    def test_grad_mode_default(self):
        mat = IdentityMatrix(3)
        assert mat.grad_mode == "default"

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="positive int"):
            IdentityMatrix(0)

    def test_rejects_negative_size(self):
        with pytest.raises(RuntimeError):
            IdentityMatrix(-1)

    @pytest.mark.parametrize("n", [3.0, "3", True, None])
    def test_rejects_non_int_size(self, n):
        with pytest.raises(TypeError):
            IdentityMatrix(n)

    def test_requires_size(self):
        with pytest.raises(TypeError):
            IdentityMatrix()

    def test_accepts_size_keyword(self):
        mat = IdentityMatrix(n=3)
        assert mat.shape == (3, 3)

    def test_rejects_tensor_size(self):
        with pytest.raises(ValueError, match="positive int"):
            IdentityMatrix(torch.tensor(3))

    def test_rejects_dtype_and_device_arguments(self):
        with pytest.raises(TypeError):
            IdentityMatrix(3, dtype=torch.float64)
        with pytest.raises(TypeError):
            IdentityMatrix(3, device="cpu")


class TestIdentityMatrixCallValues:
    """The matrix returned by ``__call__``."""

    def test_returns_identity(self):
        mat = IdentityMatrix(3)
        assert torch.equal(mat(), torch.eye(3))

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_returns_identity_of_matching_size(self, n):
        mat = IdentityMatrix(n)
        assert torch.equal(mat(), torch.eye(n))
        assert mat().shape == (n, n)

    def test_diagonal_is_one(self):
        mat = IdentityMatrix(4)
        assert (mat().diag() == 1.0).all()

    def test_off_diagonal_is_zero(self):
        mat = IdentityMatrix(4)
        result = mat()
        off_diag = result[~torch.eye(4, dtype=torch.bool)]
        assert (off_diag == 0.0).all()

    def test_is_symmetric(self):
        mat = IdentityMatrix(4)
        result = mat()
        assert torch.equal(result, result.T)

    def test_n1(self):
        mat = IdentityMatrix(1)
        assert torch.equal(mat(), torch.tensor([[1.0]]))

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_accepts_empty_free_params_of_every_form(self, free_params):
        mat = IdentityMatrix(3)
        assert torch.equal(mat(free_params), torch.eye(3))

    def test_accepts_free_params_keyword(self):
        mat = IdentityMatrix(3)
        assert torch.equal(mat(free_params=torch.tensor([])), torch.eye(3))

    def test_consecutive_calls_are_distinct_tensors(self):
        mat = IdentityMatrix(3)
        assert mat() is not mat()

    def test_result_can_be_mutated_without_affecting_later_calls(self):
        mat = IdentityMatrix(3)
        result = mat()
        result[0, 0] = 99.0
        result[0, 1] = 99.0
        assert torch.equal(mat(), torch.eye(3))

    def test_is_stable_across_repeated_calls(self):
        mat = IdentityMatrix(3)
        assert torch.equal(mat(), mat())


class TestIdentityMatrixDtypeAndDevice:
    """Rule 1: the dtype and the device of the input are followed.

    Rule 2: with no input, the Torch defaults are used.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        mat = IdentityMatrix(3)
        result = mat(torch.tensor([], dtype=dtype))
        assert result.dtype == dtype
        assert torch.equal(result, torch.eye(3, dtype=dtype))

    def test_follows_non_float_input_dtype(self):
        mat = IdentityMatrix(3)
        result = mat(torch.tensor([], dtype=torch.int64))
        assert result.dtype == torch.int64
        assert torch.equal(result, torch.eye(3, dtype=torch.int64))

    def test_default_dtype_when_omitted(self):
        mat = IdentityMatrix(3)
        assert mat().dtype == torch.get_default_dtype()

    def test_empty_dict_follows_torch_default_dtype(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = IdentityMatrix(3)
            assert mat({}).dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_follows_default_dtype_set_after_construction(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            mat = IdentityMatrix(3)
            torch.set_default_dtype(torch.float64)
            assert mat().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_cast_call_does_not_change_later_default_calls(self):
        mat = IdentityMatrix(3)
        mat(torch.tensor([], dtype=torch.float64))
        assert mat().dtype == torch.get_default_dtype()

    def test_build_params_follows_input_dtype(self):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([], dtype=torch.float64))
        assert result.dtype == torch.float64

    def test_trans_grad_follows_input_dtype(self):
        mat = IdentityMatrix(3)
        result = mat.trans_grad(torch.tensor([], dtype=torch.float64))
        assert result.dtype == torch.float64

    def test_default_device_when_omitted(self):
        mat = IdentityMatrix(3)
        assert mat().device == torch.get_default_device()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = IdentityMatrix(3)
        result = mat(torch.tensor([], device=device))
        assert result.device.type == device
        assert torch.equal(result.cpu(), torch.eye(3))

    def test_input_device_wins_over_default_device(self):
        mat = IdentityMatrix(3)
        result = mat(torch.tensor([], device="cpu"))
        assert result.device.type == "cpu"

    def test_follows_default_device_set_after_construction(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        original = torch.get_default_device()
        try:
            mat = IdentityMatrix(3)
            torch.set_default_device(device)
            assert mat().device.type == device
        finally:
            torch.set_default_device(original)

    def test_follows_dtype_and_device_together(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = IdentityMatrix(3)
        result = mat(torch.tensor([], dtype=torch.float16, device=device))
        assert result.dtype == torch.float16
        assert result.device.type == device
        assert torch.equal(result.cpu(), torch.eye(3, dtype=torch.float16))


class TestIdentityMatrixFreeParamsValidation:
    """``free_params`` carries no values: only dtype and device are read."""

    @pytest.mark.parametrize("length", [1, 3])
    def test_rejects_non_empty_tensor(self, length):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.ones(length))

    def test_rejects_non_empty_dict(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({"sigma^2": torch.tensor([1.0])})

    def test_rejects_dict_with_any_key(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({"x": torch.tensor([])})

    @pytest.mark.parametrize("free_params", [[], 1.0, "params", 0])
    def test_rejects_non_tensor(self, free_params):
        mat = IdentityMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat(free_params)

    @pytest.mark.parametrize("shape", [(0, 3), (2, 2)])
    def test_rejects_non_1d_tensor(self, shape):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.zeros(shape))

    def test_rejects_zero_dim_tensor(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.tensor(1.0))

    def test_rejects_unexpected_keyword(self):
        mat = IdentityMatrix(3)
        with pytest.raises(TypeError):
            mat(dummy=42)

    def test_rejects_non_empty_tensor_on_accelerator(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.ones(1, device=device))


class TestIdentityMatrixBuildParams:
    """``build_params`` on a matrix with no parameters at all."""

    def test_defaults_to_empty_tensor(self):
        mat = IdentityMatrix(3)
        result = mat.build_params()
        assert torch.equal(result, torch.tensor([]))
        assert result.shape == (0,)

    @pytest.mark.parametrize("include_fixed", [True, False])
    @pytest.mark.parametrize("trans", [True, False])
    def test_flags_do_not_change_the_empty_result(self, include_fixed, trans):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]), include_fixed=include_fixed, trans=trans)
        assert torch.equal(result, torch.tensor([]))

    @pytest.mark.parametrize("include_fixed", [True, False])
    def test_dict_format_is_empty(self, include_fixed):
        mat = IdentityMatrix(3)
        result = mat.build_params(torch.tensor([]), include_fixed=include_fixed, out_format="dict")
        assert result == {}

    def test_dict_input(self):
        mat = IdentityMatrix(3)
        assert torch.equal(mat.build_params({}), torch.tensor([]))

    @pytest.mark.parametrize("include_fixed", [True, False])
    def test_rejects_bad_out_format(self, include_fixed):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="out_format"):
            mat.build_params(torch.tensor([]), include_fixed=include_fixed, out_format="matrix")

    def test_rejects_non_empty_tensor(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="length 0"):
            mat.build_params(torch.ones(2))

    def test_rejects_non_empty_dict(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({"sigma^2": torch.tensor([1.0])})


class TestIdentityMatrixGrad:
    """No trainable parameters, so every gradient path returns ``(None, [])``."""

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_grad_returns_none_and_no_names(self, free_params):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.grad(free_params)
        assert grad is None
        assert grad_names == []

    def test_grad_without_arguments(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.grad()
        assert grad is None
        assert grad_names == []

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_auto_grad_returns_none_and_no_names(self, free_params):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.auto_grad(free_params)
        assert grad is None
        assert grad_names == []

    def test_manual_grad_raises(self):
        mat = IdentityMatrix(3)
        with pytest.raises(NotImplementedError):
            mat.manual_grad(torch.tensor([]))

    def test_manual_grad_without_arguments(self):
        mat = IdentityMatrix(3)
        with pytest.raises(NotImplementedError):
            mat.manual_grad()

    def test_auto_grad_does_not_differentiate(self):
        mat = IdentityMatrix(3)
        mat.jacobian_method = "bogus"
        grad, grad_names = mat.auto_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_mode_auto(self):
        mat = IdentityMatrix(3)
        mat.grad_mode = "auto"
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_mode_manual_raises(self):
        mat = IdentityMatrix(3)
        mat.grad_mode = "manual"
        with pytest.raises(NotImplementedError):
            mat.grad(torch.tensor([]))

    def test_grad_mode_unknown_raises(self):
        mat = IdentityMatrix(3)
        mat.grad_mode = "bogus"
        with pytest.raises(RuntimeError, match="Unknown grad mode"):
            mat.grad(torch.tensor([]))

    def test_grad_rejects_non_empty_tensor(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="length 0"):
            mat.grad(torch.ones(1))

    def test_grad_rejects_non_tensor(self):
        mat = IdentityMatrix(3)
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.grad(1.0)

    def test_grad_follows_input_dtype_without_error(self):
        mat = IdentityMatrix(3)
        grad, grad_names = mat.grad(torch.tensor([], dtype=torch.float64))
        assert grad is None
        assert grad_names == []

    def test_trans_grad_is_empty(self):
        mat = IdentityMatrix(3)
        result = mat.trans_grad(torch.tensor([]))
        assert result.shape == (0,)

    def test_trans_grad_without_arguments(self):
        mat = IdentityMatrix(3)
        assert mat.trans_grad().shape == (0,)

    def test_trans_grad_rejects_non_empty_tensor(self):
        mat = IdentityMatrix(3)
        with pytest.raises(ValueError, match="length 0"):
            mat.trans_grad(torch.ones(1))

    def test_get_default_dtype_device(self):
        mat = IdentityMatrix(3)
        device, dtype = mat.get_default_dtype_device()
        assert device == torch.get_default_device()
        assert dtype == torch.get_default_dtype()

    def test_get_default_dtype_device_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = IdentityMatrix(3)
            assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float64)
        finally:
            torch.set_default_dtype(original)


class TestIdentityMatrixIntermediates:
    """The intermediate cache can never hold anything without parameters."""

    def test_get_before_set_returns_none(self):
        mat = IdentityMatrix(3)
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_set_with_empty_params_is_a_noop(self):
        mat = IdentityMatrix(3)
        assert mat.set_intermediates(torch.tensor([]), {"eye": torch.eye(3)}) is None
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_set_then_get_is_still_none(self):
        mat = IdentityMatrix(3)
        mat.set_intermediates(torch.tensor([], dtype=torch.float64), "cached")
        assert mat.get_intermediates(torch.tensor([], dtype=torch.float64)) is None

    def test_reset_leaves_cache_empty(self):
        mat = IdentityMatrix(3)
        mat.set_intermediates(torch.tensor([]), "cached")
        mat.reset_intermediates()
        assert mat.get_intermediates(torch.tensor([])) is None

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_set_validates_params(self, params, error):
        mat = IdentityMatrix(3)
        with pytest.raises(error):
            mat.set_intermediates(params, "cached")

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_get_validates_params(self, params, error):
        mat = IdentityMatrix(3)
        with pytest.raises(error):
            mat.get_intermediates(params)


class TestIdentityMatrixRepr:
    def test_contains_class_name_and_shape(self):
        mat = IdentityMatrix(3)
        r = repr(mat)
        assert "IdentityMatrix" in r
        assert "(3, 3)" in r

    def test_is_single_line(self):
        mat = IdentityMatrix(3)
        assert "\n" not in repr(mat)

    def test_omits_empty_param_specs(self):
        mat = IdentityMatrix(3)
        assert "param_specs" not in repr(mat)

    def test_repr_dict(self):
        mat = IdentityMatrix(3)
        assert mat.repr_dict == {"shape": (3, 3), "param_specs": {}}
