import numpy as np
import pytest
import torch
import pandas as pd
from torch_openreml.covariance import DummyMatrix
from torch_openreml.covariance.matrix import Matrix


class TestDummyMatrixConstructor:
    """Argument validation, shape, and the empty parameter surface."""

    def test_is_matrix_subclass(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert isinstance(mat, Matrix)

    def test_shape_single_factor(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert mat.shape == (3, 2)

    def test_shape_two_factors(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "y", "x"])
        assert mat.shape == (3, 4)

    def test_shape_three_factors(self):
        mat = DummyMatrix(["a", "b"], ["x", "y"], ["p", "q"])
        assert mat.shape == (2, 8)

    def test_shape_is_rectangular(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert mat.shape[0] != mat.shape[1]

    def test_single_observation(self):
        mat = DummyMatrix(["a"])
        assert mat.shape == (1, 1)

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
        mat = DummyMatrix(["a", "b"])
        assert getattr(mat, prop) == expected

    def test_num_params_zero(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.num_params == 0
        assert mat.num_free_params == 0
        assert mat.num_fixed_params == 0

    def test_grad_mode_default(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.grad_mode == "default"

    def test_accepts_tuple_input(self):
        mat = DummyMatrix(("a", "b"))
        assert mat.shape == (2, 2)

    def test_accepts_pandas_series(self):
        mat = DummyMatrix(pd.Series(["a", "b", "a"]))
        assert mat.shape == (3, 2)
        assert mat.colnames == ["a", "b"]

    def test_accepts_pandas_series_and_list(self):
        mat = DummyMatrix(pd.Series(["a", "b"]), ["x", "y"])
        assert mat.shape == (2, 4)

    @pytest.mark.parametrize("arg", [np.array(["a", "b"]), "ab", 1, None])
    def test_rejects_non_sequence(self, arg):
        with pytest.raises(TypeError, match="must be a list, a tuple or a pandas.Series"):
            DummyMatrix(arg)

    def test_rejects_non_sequence_in_second_position(self):
        with pytest.raises(TypeError, match="Argument 1 must be a list"):
            DummyMatrix(["a", "b"], "xy")

    def test_rejects_unequal_lengths(self):
        with pytest.raises(ValueError, match="same number of elements"):
            DummyMatrix(["a", "b"], ["x"])

    def test_requires_at_least_one_argument(self):
        with pytest.raises(IndexError):
            DummyMatrix()

    @pytest.mark.parametrize("args", [([],), ([], [])])
    def test_rejects_empty_input(self, args):
        with pytest.raises(ValueError, match="positive int"):
            DummyMatrix(*args)

    @pytest.mark.parametrize("arg", [[1, 2, 1], [1, "a"]])
    def test_rejects_non_string_levels(self, arg):
        with pytest.raises(TypeError):
            DummyMatrix(arg)

    def test_rejects_dtype_and_device_arguments(self):
        with pytest.raises(TypeError):
            DummyMatrix(["a", "b"], dtype=torch.float64)
        with pytest.raises(TypeError):
            DummyMatrix(["a", "b"], device="cpu")


class TestDummyMatrixEncoding:
    """The one-hot encoding produced by the constructor options."""

    def test_single_factor_encoding(self):
        mat = DummyMatrix(["a", "b", "a", "b"])
        result = mat()
        expected = torch.tensor([
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.],
        ])
        assert torch.equal(result, expected)

    def test_two_factor_encoding(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "x", "y"])
        result = mat()
        expected = torch.tensor([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
        ])
        assert torch.equal(result, expected)

    def test_three_factor_encoding(self):
        mat = DummyMatrix(["a", "b"], ["x", "y"], ["p", "q"])
        result = mat()
        assert tuple(result.shape) == (2, 8)
        assert result[0, 0] == 1.0
        assert result[1, 7] == 1.0
        assert result.sum() == 2.0

    def test_rows_are_one_hot(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "x", "y"])
        result = mat()
        assert torch.equal(result.sum(dim=1), torch.ones(3))

    def test_custom_levels(self):
        mat = DummyMatrix(["a", "b", "a"], levels=[["b", "a", "c"]])
        result = mat()
        assert mat.colnames == ["a", "b", "c"]
        expected = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
        ])
        assert torch.equal(result, expected)

    def test_custom_levels_keep_unused_level_column(self):
        mat = DummyMatrix(["a", "b"], levels=[["a", "b", "c"]])
        result = mat()
        assert mat.colnames == ["a", "b", "c"]
        assert (result[:, 2] == 0).all()

    def test_lex_order_true_sorts_combinations(self):
        mat = DummyMatrix(["a", "b"], ["y", "x"], levels=[["b", "a"], ["y", "x"]], lex_order=True)
        assert mat.colnames == ["a⋈x", "a⋈y", "b⋈x", "b⋈y"]
        assert torch.equal(mat()[0], torch.tensor([0., 1., 0., 0.]))

    def test_lex_order_false_keeps_product_order(self):
        mat = DummyMatrix(["a", "b"], ["y", "x"], levels=[["b", "a"], ["y", "x"]], lex_order=False)
        assert mat.colnames == ["b⋈y", "b⋈x", "a⋈y", "a⋈x"]
        assert torch.equal(mat()[0], torch.tensor([0., 0., 1., 0.]))

    def test_default_levels_are_sorted(self):
        mat = DummyMatrix(["b", "a"], lex_order=False)
        assert mat.colnames == ["a", "b"]

    def test_drop_first(self):
        mat = DummyMatrix(["a", "b", "c"], drop_first=True)
        assert mat.shape == (3, 2)
        assert mat.colnames == ["b", "c"]

    def test_drop_first_drops_only_the_first_combination(self):
        mat = DummyMatrix(["a", "b", "c"], ["x", "y", "z"], drop_first=True)
        assert mat.shape == (3, 8)
        assert mat.colnames == [
            "a⋈y", "a⋈z", "b⋈x", "b⋈y", "b⋈z", "c⋈x", "c⋈y", "c⋈z",
        ]

    def test_drop_empty_cols(self):
        mat = DummyMatrix(["a", "a"], levels=[["a", "b"]], drop_empty_cols=True)
        assert mat.shape == (2, 1)
        assert mat.colnames == ["a"]

    def test_drop_empty_cols_keeps_partial_columns(self):
        mat = DummyMatrix(["a", "b"], levels=[["a", "b", "c"]], drop_empty_cols=True)
        assert mat.shape == (2, 2)
        assert mat.colnames == ["a", "b"]

    def test_drop_first_and_drop_empty(self):
        mat = DummyMatrix(["a", "b", "a"], levels=[["a", "b", "c"]],
                          drop_first=True, drop_empty_cols=True)
        assert mat.colnames == ["b"]
        assert mat.shape == (3, 1)

    def test_dropping_every_column_is_rejected(self):
        with pytest.warns(RuntimeWarning):
            with pytest.raises(ValueError, match="positive int"):
                DummyMatrix(["z"], levels=[["a", "b"]], drop_empty_cols=True)

    def test_unknown_combination_warns(self):
        with pytest.warns(RuntimeWarning, match="Unknown combination"):
            DummyMatrix(["a", "b", "unknown"], levels=[["a", "b"]])

    def test_unknown_combination_row_is_zero(self):
        with pytest.warns(RuntimeWarning):
            mat = DummyMatrix(["a", "unknown", "b"], levels=[["a", "b"]])
        result = mat()
        assert (result[1] == 0).all()

    def test_unknown_combination_keeps_known_rows(self):
        with pytest.warns(RuntimeWarning):
            mat = DummyMatrix(["a", "unknown", "b"], levels=[["a", "b"]])
        result = mat()
        assert torch.equal(result[0], torch.tensor([1., 0.]))
        assert torch.equal(result[2], torch.tensor([0., 1.]))

    def test_unknown_combination_two_factors(self):
        with pytest.warns(RuntimeWarning, match="Unknown combination"):
            mat = DummyMatrix(["a", "z"], ["x", "x"], levels=[["a", "b"], ["x", "y"]])
        result = mat()
        assert torch.equal(result[0], torch.tensor([1., 0., 0., 0.]))
        assert (result[1] == 0).all()

    def test_warns_once_per_unknown_row(self):
        with pytest.warns(RuntimeWarning) as record:
            DummyMatrix(["z", "z"], levels=[["a", "b"]])
        assert len(record) == 2


class TestDummyMatrixColnames:
    """The ``colnames`` property."""

    def test_single_factor(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert mat.colnames == ["a", "b"]

    def test_two_factors(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "x", "y"])
        assert mat.colnames == ["a⋈x", "a⋈y", "b⋈x", "b⋈y"]

    def test_three_factors(self):
        mat = DummyMatrix(["a", "b"], ["x", "y"], ["p", "q"])
        assert mat.colnames == [
            "a⋈x⋈p", "a⋈x⋈q", "a⋈y⋈p", "a⋈y⋈q",
            "b⋈x⋈p", "b⋈x⋈q", "b⋈y⋈p", "b⋈y⋈q",
        ]

    def test_length_matches_number_of_columns(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "y", "x"])
        assert len(mat.colnames) == mat.shape[1]

    def test_reflects_drops(self):
        mat = DummyMatrix(["a", "b"], levels=[["a", "b", "c"]], drop_empty_cols=True)
        assert mat.colnames == ["a", "b"]
        assert len(mat.colnames) == mat.shape[1]


class TestDummyMatrixCallValues:
    """The matrix returned by ``__call__``."""

    def test_returns_stored_matrix(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert torch.equal(mat(), mat._matrix)

    def test_result_shape_matches_shape(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "y", "x"])
        assert tuple(mat().shape) == mat.shape

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_accepts_empty_free_params_of_every_form(self, free_params):
        mat = DummyMatrix(["a", "b"])
        assert torch.equal(mat(free_params), mat._matrix)

    def test_accepts_free_params_keyword(self):
        mat = DummyMatrix(["a", "b"])
        assert torch.equal(mat(free_params=torch.tensor([])), mat._matrix)

    def test_consecutive_calls_are_distinct_tensors(self):
        mat = DummyMatrix(["a", "b"])
        assert mat() is not mat()

    def test_result_can_be_mutated_without_affecting_later_calls(self):
        mat = DummyMatrix(["a", "b"])
        expected = torch.tensor([
            [1., 0.],
            [0., 1.],
        ])
        result = mat()
        result[0, 0] = 99.0
        result[1, 1] = 99.0
        assert torch.equal(mat(), expected)

    def test_is_stable_across_repeated_calls(self):
        mat = DummyMatrix(["a", "b"])
        assert torch.equal(mat(), mat())

    def test_colnames_survive_calls(self):
        mat = DummyMatrix(["a", "b"])
        mat(torch.tensor([], dtype=torch.float64))
        assert mat.colnames == ["a", "b"]


class TestDummyMatrixDtypeAndDevice:
    """Rule 1: the dtype and the device of the input are followed.

    Rule 2: with no input, the Torch defaults are used.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_follows_input_dtype(self, dtype):
        mat = DummyMatrix(["a", "b"])
        result = mat(torch.tensor([], dtype=dtype))
        assert result.dtype == dtype
        assert torch.equal(result, mat._matrix.to(dtype=dtype))

    def test_follows_non_float_input_dtype(self):
        mat = DummyMatrix(["a", "b"])
        assert mat(torch.tensor([], dtype=torch.int64)).dtype == torch.int64

    def test_default_dtype_when_omitted(self):
        mat = DummyMatrix(["a", "b"])
        assert mat().dtype == torch.get_default_dtype()

    def test_construction_follows_default_dtype(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = DummyMatrix(["a", "b"])
            assert mat._matrix.dtype == torch.float64
            assert mat().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_empty_dict_follows_torch_default_dtype(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = DummyMatrix(["a", "b"])
            assert mat({}).dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_follows_default_dtype_set_after_construction(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            mat = DummyMatrix(["a", "b"])
            torch.set_default_dtype(torch.float64)
            assert mat().dtype == torch.float64
        finally:
            torch.set_default_dtype(original)

    def test_cast_call_does_not_change_later_default_calls(self):
        mat = DummyMatrix(["a", "b"])
        mat(torch.tensor([], dtype=torch.float64))
        assert mat().dtype == torch.get_default_dtype()

    def test_build_params_follows_input_dtype(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.build_params(torch.tensor([], dtype=torch.float64)).dtype == torch.float64

    def test_trans_grad_follows_input_dtype(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.trans_grad(torch.tensor([], dtype=torch.float64)).dtype == torch.float64

    def test_default_device_when_omitted(self):
        mat = DummyMatrix(["a", "b"])
        assert mat().device == torch.get_default_device()

    def test_follows_input_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = DummyMatrix(["a", "b"])
        result = mat(torch.tensor([], device=device))
        assert result.device.type == device
        assert torch.equal(result.cpu(), mat._matrix)

    def test_input_device_wins_over_default_device(self):
        mat = DummyMatrix(["a", "b"])
        assert mat(torch.tensor([], device="cpu")).device.type == "cpu"

    def test_follows_default_device_set_after_construction(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        original = torch.get_default_device()
        try:
            mat = DummyMatrix(["a", "b"])
            torch.set_default_device(device)
            assert mat().device.type == device
        finally:
            torch.set_default_device(original)


class TestDummyMatrixFreeParamsValidation:
    """``free_params`` carries no values: only dtype and device are read."""

    @pytest.mark.parametrize("length", [1, 2])
    def test_rejects_non_empty_tensor(self, length):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.ones(length))

    def test_rejects_non_empty_dict(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({"sigma^2": torch.tensor([1.0])})

    def test_rejects_dict_with_any_key(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat({"x": torch.tensor([])})

    @pytest.mark.parametrize("free_params", [[], 1.0, "params", 0])
    def test_rejects_non_tensor(self, free_params):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(TypeError, match="Torch tensor"):
            mat(free_params)

    @pytest.mark.parametrize("shape", [(0, 3), (2, 2)])
    def test_rejects_non_1d_tensor(self, shape):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.zeros(shape))

    def test_rejects_zero_dim_tensor(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="1D tensor"):
            mat(torch.tensor(1.0))

    def test_rejects_unexpected_keyword(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(TypeError):
            mat(dummy=42)

    def test_rejects_non_empty_tensor_on_accelerator(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("no accelerator available")
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="length 0"):
            mat(torch.ones(1, device=device))


class TestDummyMatrixBuildParams:
    """``build_params`` on a matrix with no parameters at all."""

    def test_defaults_to_empty_tensor(self):
        mat = DummyMatrix(["a", "b"])
        result = mat.build_params()
        assert torch.equal(result, torch.tensor([]))
        assert result.shape == (0,)

    @pytest.mark.parametrize("include_fixed", [True, False])
    @pytest.mark.parametrize("trans", [True, False])
    def test_flags_do_not_change_the_empty_result(self, include_fixed, trans):
        mat = DummyMatrix(["a", "b"])
        result = mat.build_params(torch.tensor([]), include_fixed=include_fixed, trans=trans)
        assert torch.equal(result, torch.tensor([]))

    @pytest.mark.parametrize("include_fixed", [True, False])
    def test_dict_format_is_empty(self, include_fixed):
        mat = DummyMatrix(["a", "b"])
        result = mat.build_params(torch.tensor([]), include_fixed=include_fixed, out_format="dict")
        assert result == {}

    def test_dict_input(self):
        mat = DummyMatrix(["a", "b"])
        assert torch.equal(mat.build_params({}), torch.tensor([]))

    @pytest.mark.parametrize("include_fixed", [True, False])
    def test_rejects_bad_out_format(self, include_fixed):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="out_format"):
            mat.build_params(torch.tensor([]), include_fixed=include_fixed, out_format="matrix")

    def test_rejects_non_empty_tensor(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="length 0"):
            mat.build_params(torch.ones(2))

    def test_rejects_non_empty_dict(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="Unexpected free parameters"):
            mat.build_params({"sigma^2": torch.tensor([1.0])})


class TestDummyMatrixGrad:
    """No trainable parameters, so every gradient path returns ``(None, [])``."""

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_grad_returns_none_and_no_names(self, free_params):
        mat = DummyMatrix(["a", "b"])
        grad, grad_names = mat.grad(free_params)
        assert grad is None
        assert grad_names == []

    def test_grad_without_arguments(self):
        mat = DummyMatrix(["a", "b"])
        grad, grad_names = mat.grad()
        assert grad is None
        assert grad_names == []

    @pytest.mark.parametrize("free_params", [None, torch.tensor([]), {}])
    def test_auto_grad_returns_none_and_no_names(self, free_params):
        mat = DummyMatrix(["a", "b"])
        grad, grad_names = mat.auto_grad(free_params)
        assert grad is None
        assert grad_names == []

    def test_auto_grad_does_not_differentiate(self):
        mat = DummyMatrix(["a", "b"])
        mat.jacobian_method = "bogus"
        grad, grad_names = mat.auto_grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    @pytest.mark.parametrize("args", [(), (torch.tensor([]),)])
    def test_manual_grad_raises(self, args):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(NotImplementedError):
            mat.manual_grad(*args)

    def test_grad_mode_auto(self):
        mat = DummyMatrix(["a", "b"])
        mat.grad_mode = "auto"
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_grad_mode_manual_raises(self):
        mat = DummyMatrix(["a", "b"])
        mat.grad_mode = "manual"
        with pytest.raises(NotImplementedError):
            mat.grad(torch.tensor([]))

    def test_grad_mode_unknown_raises(self):
        mat = DummyMatrix(["a", "b"])
        mat.grad_mode = "bogus"
        with pytest.raises(RuntimeError, match="Unknown grad mode"):
            mat.grad(torch.tensor([]))

    def test_grad_rejects_non_empty_tensor(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="length 0"):
            mat.grad(torch.ones(1))

    def test_grad_rejects_non_tensor(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(TypeError, match="Torch tensor"):
            mat.grad(1.0)

    def test_trans_grad_is_empty(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.trans_grad(torch.tensor([])).shape == (0,)

    def test_trans_grad_without_arguments(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.trans_grad().shape == (0,)

    def test_trans_grad_rejects_non_empty_tensor(self):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(ValueError, match="length 0"):
            mat.trans_grad(torch.ones(1))

    def test_get_default_dtype_device(self):
        mat = DummyMatrix(["a", "b"])
        device, dtype = mat.get_default_dtype_device()
        assert device == torch.get_default_device()
        assert dtype == torch.get_default_dtype()

    def test_get_default_dtype_device_tracks_torch_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            mat = DummyMatrix(["a", "b"])
            assert mat.get_default_dtype_device() == (torch.get_default_device(), torch.float64)
        finally:
            torch.set_default_dtype(original)


class TestDummyMatrixIntermediates:
    """The intermediate cache can never hold anything without parameters."""

    def test_get_before_set_returns_none(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_set_with_empty_params_is_a_noop(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.set_intermediates(torch.tensor([]), {"matrix": torch.eye(2)}) is None
        assert mat.get_intermediates(torch.tensor([])) is None

    def test_set_then_get_is_still_none(self):
        mat = DummyMatrix(["a", "b"])
        mat.set_intermediates(torch.tensor([], dtype=torch.float64), "cached")
        assert mat.get_intermediates(torch.tensor([], dtype=torch.float64)) is None

    def test_reset_leaves_cache_empty(self):
        mat = DummyMatrix(["a", "b"])
        mat.set_intermediates(torch.tensor([]), "cached")
        mat.reset_intermediates()
        assert mat.get_intermediates(torch.tensor([])) is None

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_set_validates_params(self, params, error):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(error):
            mat.set_intermediates(params, "cached")

    @pytest.mark.parametrize("params, error", [(1.0, TypeError), (torch.zeros(2, 2), ValueError)])
    def test_get_validates_params(self, params, error):
        mat = DummyMatrix(["a", "b"])
        with pytest.raises(error):
            mat.get_intermediates(params)


class TestDummyMatrixRepr:
    def test_contains_class_name_and_shape(self):
        mat = DummyMatrix(["a", "b", "a"])
        r = repr(mat)
        assert "DummyMatrix" in r
        assert "(3, 2)" in r

    def test_is_single_line(self):
        mat = DummyMatrix(["a", "b"])
        assert "\n" not in repr(mat)

    def test_omits_empty_param_specs(self):
        mat = DummyMatrix(["a", "b"])
        assert "param_specs" not in repr(mat)

    def test_repr_dict_has_only_shape(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert mat.repr_dict == {"shape": (3, 2)}
