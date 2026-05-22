import torch
import pytest
import pandas as pd
from torch_openreml.covariance import DummyMatrix
from torch_openreml.covariance.matrix import Matrix


class TestDummyMatrix:
    """Tests for the fixed dummy matrix."""

    def test_constructor(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert isinstance(mat, Matrix)

    def test_shape_single_factor(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert mat.shape == (3, 2)

    def test_shape_two_factors(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "y", "x"])
        assert mat.shape == (3, 4)

    def test_no_params(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert mat.param_specs == {}
        assert mat.num_params == 0
        assert mat.num_free_params == 0

    def test_raises_type_error_on_non_list(self):
        with pytest.raises(TypeError):
            DummyMatrix("not_a_list")

    def test_raises_value_error_on_unequal_length(self):
        with pytest.raises(ValueError):
            DummyMatrix(["a", "b"], ["x"])

    def test_call_returns_matrix(self):
        mat = DummyMatrix(["a", "b", "a"])
        result = mat()
        assert torch.equal(result, mat._matrix)

    def test_call_ignores_arguments(self):
        mat = DummyMatrix(["a", "b", "a"])
        expected = mat._matrix
        assert torch.equal(mat(torch.tensor([1.0, 2.0])), expected)

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

    def test_colnames(self):
        mat = DummyMatrix(["a", "b", "a"])
        assert mat.colnames == ["a", "b"]

    def test_colnames_two_factors(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "x", "y"])
        assert mat.colnames == ["a⋈x", "a⋈y", "b⋈x", "b⋈y"]

    def test_two_factor_encoding(self):
        mat = DummyMatrix(["a", "b", "a"], ["x", "x", "y"])
        result = mat()
        expected = torch.tensor([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
        ])
        assert torch.equal(result, expected)

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

    def test_lex_order_false(self):
        mat = DummyMatrix(["a", "b", "a"], levels=[["b", "a"]], lex_order=False)
        assert mat.colnames == ["b", "a"]

    def test_drop_first(self):
        mat = DummyMatrix(["a", "b", "c"], drop_first=True)
        assert mat.shape == (3, 2)
        assert mat.colnames == ["b", "c"]

    def test_drop_empty_cols(self):
        mat = DummyMatrix(["a", "a"], levels=[["a", "b"]], drop_empty_cols=True)
        assert mat.shape == (2, 1)
        assert mat.colnames == ["a"]

    def test_drop_first_and_drop_empty(self):
        mat = DummyMatrix(["a", "b", "a"], levels=[["a", "b", "c"]],
                          drop_first=True, drop_empty_cols=True)
        assert mat.colnames == ["b"]
        assert mat.shape == (3, 1)

    def test_unknown_combination_warns(self):
        with pytest.warns(RuntimeWarning, match="Unknown combination"):
            DummyMatrix(["a", "b", "unknown"], levels=[["a", "b"]])

    def test_unknown_combination_row_zero(self):
        with pytest.warns(RuntimeWarning):
            mat = DummyMatrix(["a", "unknown", "b"], levels=[["a", "b"]])
        result = mat()
        assert (result[1] == 0).all()

    def test_pandas_series_input(self):
        s = pd.Series(["a", "b", "a"])
        mat = DummyMatrix(s)
        assert mat.shape == (3, 2)
        assert mat.colnames == ["a", "b"]

    def test_dtype(self):
        mat = DummyMatrix(["a", "b"], dtype=torch.float64)
        assert mat().dtype == torch.float64

    def test_device(self):
        mat = DummyMatrix(["a", "b"], device="cpu")
        assert mat().device.type == "cpu"

    def test_grad_returns_none(self):
        mat = DummyMatrix(["a", "b", "a"])
        grad, grad_names = mat.grad(torch.tensor([]))
        assert grad is None
        assert grad_names == []

    def test_map_theta_to_v(self):
        mat = DummyMatrix(["a", "b"])
        assert torch.equal(mat.map_theta_to_v(torch.tensor([])), mat())

    def test_map_theta_to_dv(self):
        mat = DummyMatrix(["a", "b"])
        assert mat.map_theta_to_dv(torch.tensor([])) is None

    def test_repr(self):
        mat = DummyMatrix(["a", "b", "a"])
        r = repr(mat)
        assert "DummyMatrix" in r
        assert "(3, 2)" in r
