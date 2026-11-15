import pytest
import torch

from torch_openreml.covariance import KnownMatrix
from torch_openreml.covariance.matrix import Matrix


def _known():
    return torch.tensor([[2.0, 0.5], [0.5, 1.0]])


class TestKnownMatrixConstructor:
    """Construction, shape, and the empty parameter surface."""

    def test_is_matrix_subclass(self):
        assert isinstance(KnownMatrix(_known()), Matrix)

    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (2, 3), (4, 2)])
    def test_shape(self, shape):
        mat = KnownMatrix(torch.zeros(shape))
        assert mat.shape == shape
        assert isinstance(mat.shape, tuple)

    def test_square_shape(self):
        assert KnownMatrix(_known()).shape == (2, 2)

    def test_matrix_property_returns_the_supplied_tensor(self):
        k = _known()
        assert KnownMatrix(k).matrix is k

    def test_matrix_property_keeps_values(self):
        k = _known()
        assert torch.equal(KnownMatrix(k).matrix, k)

    def test_parameter_surface_is_empty(self):
        mat = KnownMatrix(_known())
        assert mat.param_specs == {}
        assert mat.param_names == []
        assert mat.free_param_names == []
        assert mat.fixed_param_names == []
        assert mat.free_param_index == []
        assert mat.fixed_param_index == []

    def test_num_params_are_zero(self):
        mat = KnownMatrix(_known())
        assert mat.num_params == 0
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 0

    def test_grad_mode_default(self):
        assert KnownMatrix(_known()).grad_mode == "default"

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="Torch tensor"):
            KnownMatrix([[1.0, 0.0], [0.0, 1.0]])

    def test_rejects_scalar_tensor(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            KnownMatrix(torch.tensor(1.0))

    def test_rejects_one_dimensional_tensor(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            KnownMatrix(torch.ones(3))

    def test_rejects_three_dimensional_tensor(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            KnownMatrix(torch.ones(2, 2, 2))

    def test_rejects_empty_arguments(self):
        with pytest.raises(TypeError):
            KnownMatrix()


class TestKnownMatrixCall:
    """``__call__`` returns a cast copy of the supplied tensor."""

    def test_returns_the_known_matrix(self):
        k = _known()
        assert torch.equal(KnownMatrix(k)(), k)

    def test_returns_a_copy(self):
        mat = KnownMatrix(_known())
        assert mat() is not mat.matrix

    def test_mutating_the_result_does_not_change_the_source(self):
        k = _known()
        mat = KnownMatrix(k)
        result = mat()
        result[0, 0] = 99.0
        assert torch.equal(mat(), k)

    def test_rectangular_matrix(self):
        k = torch.arange(6.0).reshape(2, 3)
        assert torch.equal(KnownMatrix(k)(), k)

    def test_single_entry(self):
        k = torch.tensor([[3.0]])
        assert torch.equal(KnownMatrix(k)(), k)

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_accepts_empty_free_params_of_every_form(self, free_params):
        assert torch.equal(KnownMatrix(_known())(free_params), _known())

    def test_accepts_free_params_keyword(self):
        mat = KnownMatrix(_known())
        assert torch.equal(mat(free_params=torch.tensor([])), _known())

    def test_is_stable_across_repeated_calls(self):
        mat = KnownMatrix(_known())
        assert torch.equal(mat(), mat())

    def test_consecutive_calls_are_distinct_tensors(self):
        mat = KnownMatrix(_known())
        assert mat() is not mat()


class TestKnownMatrixDtypeAndDevice:
    """The tensor is cast to the dtype and device of the input."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    def test_follows_input_dtype(self, dtype):
        mat = KnownMatrix(_known())
        result = mat(torch.tensor([], dtype=dtype))
        assert result.dtype == dtype
        assert torch.allclose(result.float(), _known())

    def test_follows_non_float_input_dtype(self):
        mat = KnownMatrix(torch.tensor([[2, 0], [0, 3]]))
        result = mat(torch.tensor([], dtype=torch.int64))
        assert result.dtype == torch.int64
        assert torch.equal(result, torch.tensor([[2, 0], [0, 3]]))

    def test_default_dtype_when_omitted(self):
        assert KnownMatrix(_known())().dtype == torch.get_default_dtype()

    def test_follows_default_dtype_set_after_construction(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            mat = KnownMatrix(_known())
            torch.set_default_dtype(torch.float64)
            assert mat().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_cast_call_does_not_change_later_default_calls(self):
        mat = KnownMatrix(_known())
        mat(torch.tensor([], dtype=torch.float64))
        assert mat().dtype == torch.get_default_dtype()

    def test_build_params_follows_input_dtype(self):
        mat = KnownMatrix(_known())
        assert mat.build_params(torch.tensor([], dtype=torch.float64)).dtype == torch.float64

    def test_trans_grad_follows_input_dtype(self):
        mat = KnownMatrix(_known())
        assert mat.trans_grad(torch.tensor([], dtype=torch.float64)).dtype == torch.float64

    def test_default_device_when_omitted(self):
        assert KnownMatrix(_known())().device == torch.get_default_device()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = KnownMatrix(_known())
        result = mat(torch.tensor([], device=device))
        assert result.device.type == device
        assert torch.allclose(result.cpu(), _known())

    def test_input_device_wins_over_default_device(self):
        assert KnownMatrix(_known())(torch.tensor([], device="cpu")).device.type == "cpu"

    def test_follows_default_device_set_after_construction(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        original = torch.get_default_device()
        try:
            mat = KnownMatrix(_known())
            torch.set_default_device(device)
            assert mat().device.type == device
        finally:
            torch.set_default_device(original)


class TestKnownMatrixFreeParamsValidation:
    """The matrix has no free parameters, so every non-empty input is rejected."""

    @pytest.mark.parametrize("length", [1, 3])
    def test_rejects_non_empty_tensor(self, length):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.ones(length))

    def test_rejects_non_empty_dict(self):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({"sigma^2": torch.tensor([1.0])})

    @pytest.mark.parametrize("free_params", [[], 1.0, "params", 0])
    def test_rejects_non_tensor(self, free_params):
        mat = KnownMatrix(_known())
        with pytest.raises(TypeError, match="Torch tensor"):
            mat(free_params)

    @pytest.mark.parametrize("shape", [(0, 3), (2, 2)])
    def test_rejects_non_1d_tensor(self, shape):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.zeros(shape))

    def test_rejects_zero_dim_tensor(self):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.tensor(1.0))

    def test_rejects_unexpected_keyword(self):
        mat = KnownMatrix(_known())
        with pytest.raises(TypeError):
            mat(dummy=42)

    def test_build_params_rejects_non_empty_tensor(self):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="length 0"):
            mat.build_params(torch.ones(2))

    def test_build_params_rejects_non_empty_dict(self):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({"theta_0": torch.tensor([1.0])})

    def test_manual_inverse_rejects_non_empty_params(self):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="length 0"):
            mat.manual_inverse(torch.ones(1))


class TestKnownMatrixBuildParams:
    """``build_params`` on a matrix with no parameters at all."""

    def test_defaults_to_empty_tensor(self):
        mat = KnownMatrix(_known())
        result = mat.build_params()
        assert torch.equal(result, torch.tensor([]))
        assert result.shape == (0,)

    @pytest.mark.parametrize("include_fixed", [True, False])
    @pytest.mark.parametrize("trans", [True, False])
    def test_flags_do_not_change_the_empty_result(self, include_fixed, trans):
        mat = KnownMatrix(_known())
        result = mat.build_params(torch.tensor([]), include_fixed=include_fixed, trans=trans)
        assert torch.equal(result, torch.tensor([]))

    @pytest.mark.parametrize("include_fixed", [True, False])
    def test_dict_format_is_empty(self, include_fixed):
        mat = KnownMatrix(_known())
        assert mat.build_params(torch.tensor([]), include_fixed=include_fixed, out_format="dict") == {}

    def test_dict_input(self):
        mat = KnownMatrix(_known())
        assert torch.equal(mat.build_params({}), torch.tensor([]))

    def test_rejects_bad_out_format(self):
        mat = KnownMatrix(_known())
        with pytest.raises(ValueError, match="out_format"):
            mat.build_params(torch.tensor([]), out_format="matrix")


class TestKnownMatrixGrad:
    """No trainable parameters, so every gradient path returns ``(None, [])``."""

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_grad_returns_none_and_no_names(self, free_params):
        grad, names = KnownMatrix(_known()).grad(free_params)
        assert grad is None
        assert names == []

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_auto_grad_returns_none_and_no_names(self, free_params):
        grad, names = KnownMatrix(_known()).auto_grad(free_params)
        assert grad is None
        assert names == []

    def test_manual_grad_raises(self):
        with pytest.raises(NotImplementedError):
            KnownMatrix(_known()).manual_grad(torch.tensor([]))

    def test_grad_mode_auto(self):
        mat = KnownMatrix(_known())
        mat.grad_mode = "auto"
        assert mat.grad(torch.tensor([])) == (None, [])

    def test_grad_mode_manual_raises(self):
        mat = KnownMatrix(_known())
        mat.grad_mode = "manual"
        with pytest.raises(NotImplementedError):
            mat.grad(torch.tensor([]))

    def test_grad_mode_unknown_raises(self):
        mat = KnownMatrix(_known())
        mat.grad_mode = "bogus"
        with pytest.raises(RuntimeError, match="Unknown grad mode"):
            mat.grad(torch.tensor([]))

    def test_grad_rejects_non_empty_tensor(self):
        with pytest.raises(ValueError, match="length 0"):
            KnownMatrix(_known()).grad(torch.ones(1))

    def test_grad_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="Torch tensor"):
            KnownMatrix(_known()).grad(1.0)

    def test_trans_grad_is_empty(self):
        assert KnownMatrix(_known()).trans_grad(torch.tensor([])).shape == (0,)

    def test_get_default_dtype_device(self):
        device, dtype = KnownMatrix(_known()).get_default_dtype_device()
        assert device == torch.get_default_device()
        assert dtype == torch.get_default_dtype()


class TestKnownMatrixIntermediates:
    """The intermediate cache can never hold anything without parameters."""

    def test_get_before_set_returns_none(self):
        assert KnownMatrix(_known()).get_intermediates(torch.tensor([])) is None

    def test_set_with_empty_params_is_a_noop(self):
        mat = KnownMatrix(_known())
        assert mat.set_intermediates(torch.tensor([]), {"k": torch.eye(2)}) is None
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_reset_leaves_cache_empty(self):
        mat = KnownMatrix(_known())
        mat.set_intermediates(torch.tensor([]), "cached")
        mat.reset_intermediates()
        assert mat.get_intermediates(torch.tensor([])) is None

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_set_validates_params(self, params, error):
        with pytest.raises(error):
            KnownMatrix(_known()).set_intermediates(params, "cached")

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_get_validates_params(self, params, error):
        with pytest.raises(error):
            KnownMatrix(_known()).get_intermediates(params)


class TestKnownMatrixInverse:
    """The inverse is precomputed once at construction."""

    def test_matches_numeric_inverse(self):
        mat = KnownMatrix(_known())
        assert torch.allclose(mat.manual_inverse(), torch.linalg.inv(_known()))

    def test_identity_matrix_inverse(self):
        mat = KnownMatrix(torch.eye(4))
        assert torch.equal(mat.manual_inverse(), torch.eye(4))

    def test_single_entry_inverse(self):
        mat = KnownMatrix(torch.tensor([[4.0]]))
        assert torch.allclose(mat.manual_inverse(), torch.tensor([[0.25]]))

    def test_product_is_identity(self):
        mat = KnownMatrix(_known())
        assert torch.allclose(mat() @ mat.manual_inverse(), torch.eye(2), atol=1e-6)

    def test_inverse_is_symmetric_for_symmetric_input(self):
        inverse = KnownMatrix(_known()).manual_inverse()
        assert torch.allclose(inverse, inverse.T)

    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 4)])
    def test_rectangular_matrix_has_no_inverse(self, shape):
        with pytest.raises(NotImplementedError, match="square"):
            KnownMatrix(torch.zeros(shape)).manual_inverse()

    def test_singular_matrix_has_no_inverse(self):
        singular = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(NotImplementedError, match="non-singular"):
            KnownMatrix(singular).manual_inverse()

    def test_zero_matrix_has_no_inverse(self):
        with pytest.raises(NotImplementedError):
            KnownMatrix(torch.zeros(3, 3)).manual_inverse()

    def test_follows_input_dtype(self):
        mat = KnownMatrix(_known())
        inverse = mat.manual_inverse(torch.tensor([], dtype=torch.float64))
        assert inverse.dtype == torch.float64
        assert torch.allclose(inverse, torch.linalg.inv(_known().double()))

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = KnownMatrix(_known())
        inverse = mat.manual_inverse(torch.tensor([], device=device))
        assert inverse.device.type == device
        assert torch.allclose(inverse.cpu(), torch.linalg.inv(_known()))

    def test_returns_a_copy(self):
        mat = KnownMatrix(_known())
        first = mat.manual_inverse()
        second = mat.manual_inverse()
        assert first is not second
        first[0, 0] = 99.0
        assert torch.allclose(second, torch.linalg.inv(_known()))

    def test_accepts_empty_dict(self):
        mat = KnownMatrix(_known())
        assert torch.allclose(mat.manual_inverse({}), torch.linalg.inv(_known()))


class TestKnownMatrixRepr:
    """The representation names the class."""

    def test_repr_contains_class_name(self):
        assert "KnownMatrix" in repr(KnownMatrix(_known()))

    def test_str_contains_class_name(self):
        assert "KnownMatrix" in str(KnownMatrix(_known()))
