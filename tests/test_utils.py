import torch
import pandas as pd
import pytest
from torch_openreml.utils import (
    get_device,
    get_dtype,
    numeric_to_design_matrix,
    augment,
    interaction,
    n_distinct,
)


class TestGetDevice:
    def test_all_same_device(self):
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        assert get_device(x, y) == x.device

    def test_single_tensor(self):
        x = torch.tensor([1.0])
        assert get_device(x) == x.device

    def test_empty_returns_default(self):
        assert get_device() == torch.get_default_device()

    def test_mismatch_raises(self):
        cpu = torch.tensor([1.0])
        if torch.cuda.is_available():
            cuda = torch.tensor([2.0], device="cuda")
            with pytest.raises(ValueError, match="Device mismatch"):
                get_device(cpu, cuda)


class TestGetDtype:
    def test_all_same_dtype(self):
        x = torch.tensor([1.0], dtype=torch.float32)
        y = torch.tensor([2.0], dtype=torch.float32)
        assert get_dtype(x, y) == torch.float32

    def test_single_tensor(self):
        x = torch.tensor([1.0], dtype=torch.float64)
        assert get_dtype(x) == torch.float64

    def test_empty_returns_default(self):
        assert get_dtype() == torch.get_default_dtype()

    def test_mismatch_raises(self):
        x = torch.tensor([1.0], dtype=torch.float32)
        y = torch.tensor([2.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="Dtype mismatch"):
            get_dtype(x, y)


class TestNumericToDesignMatrix:
    def test_two_tensors(self):
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([4.0, 5.0, 6.0])
        result = numeric_to_design_matrix(x1, x2)
        assert result.shape == (3, 2)
        assert torch.equal(result[:, 0], x1)
        assert torch.equal(result[:, 1], x2)

    def test_single_tensor(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = numeric_to_design_matrix(x)
        assert result.shape == (3, 1)

    def test_list_input(self):
        result = numeric_to_design_matrix([1.0, 2.0])
        assert result.shape == (2, 1)

    def test_tuple_input(self):
        result = numeric_to_design_matrix((1.0, 2.0, 3.0))
        assert result.shape == (3, 1)

    def test_pandas_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = numeric_to_design_matrix(s)
        assert result.shape == (3, 1)

    def test_dtype_param(self):
        x = torch.tensor([1, 2, 3])
        result = numeric_to_design_matrix(x, dtype=torch.float64)
        assert result.dtype == torch.float64

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one input"):
            numeric_to_design_matrix()

    def test_unequal_length_raises(self):
        x1 = torch.tensor([1.0, 2.0])
        x2 = torch.tensor([3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="Inconsistent lengths"):
            numeric_to_design_matrix(x1, x2)

    def test_non_tensor_list_raises(self):
        with pytest.raises(TypeError):
            numeric_to_design_matrix("invalid")


class TestAugment:
    def test_two_matrices(self):
        x1 = torch.ones(4, 2)
        x2 = torch.zeros(4, 3)
        result = augment(x1, x2)
        assert result.shape == (4, 5)
        assert torch.equal(result[:, :2], x1)
        assert torch.equal(result[:, 2:], x2)

    def test_three_matrices(self):
        x1 = torch.ones(3, 1)
        x2 = torch.ones(3, 2)
        x3 = torch.ones(3, 3)
        result = augment(x1, x2, x3)
        assert result.shape == (3, 6)


class TestInteraction:
    def test_two_lists(self):
        a = ["control", "treatment"]
        b = ["male", "female"]
        result = interaction(a, b)
        assert result == ["control⋈male", "treatment⋈female"]

    def test_custom_separator(self):
        a = ["a", "b"]
        b = ["x", "y"]
        result = interaction(a, b, sep=":")
        assert result == ["a:x", "b:y"]

    def test_three_lists(self):
        a = ["a", "b"]
        b = ["c", "d"]
        c = ["e", "f"]
        result = interaction(a, b, c)
        assert result == ["a⋈c⋈e", "b⋈d⋈f"]

    def test_single_list(self):
        a = ["x", "y"]
        result = interaction(a)
        assert result == ["x", "y"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one input"):
            interaction()

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            interaction("not_a_list")


class TestNDistinct:
    def test_with_duplicates(self):
        assert n_distinct(["a", "b", "a", "c"]) == 3

    def test_all_unique(self):
        assert n_distinct(["a", "b", "c"]) == 3

    def test_all_same(self):
        assert n_distinct(["a", "a", "a"]) == 1

    def test_empty(self):
        assert n_distinct([]) == 0

    def test_pandas_series(self):
        s = pd.Series(["a", "b", "a", "c"])
        assert n_distinct(s) == 3

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            n_distinct(42)
