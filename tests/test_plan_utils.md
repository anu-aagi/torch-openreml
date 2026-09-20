# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.utils`
**Purpose of Module:**
Provides general-purpose utility functions for device/dtype validation, design matrix construction, column-wise concatenation, interaction term construction, and unique element counting. All functions are pure and stateless.

**Functions Covered:**

- `get_device` — validate and return shared device
- `get_dtype` — validate and return shared dtype
- `numeric_to_design_matrix` — stack numeric vectors as columns
- `augment` — horizontally concatenate design matrices
- `interaction` — construct interaction term from categorical vectors
- `n_distinct` — count unique elements

**Testing Goal:**
Ensure each function correctly handles valid inputs, raises appropriate errors for invalid inputs, and handles edge cases (empty input, single input, mixed types).

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. `get_device`

- All tensors on same device → returns that device
- Single tensor → returns its device
- Empty call → returns default device
- Mismatched devices → `ValueError`

#### B. `get_dtype`

- All tensors with same dtype → returns that dtype
- Single tensor → returns its dtype
- Empty call → returns default dtype
- Mixed dtypes → `ValueError`

#### C. `numeric_to_design_matrix`

- Multiple tensors of same length → `(n, k)` matrix
- Single tensor → `(n, 1)` matrix
- Lists and tuples auto-converted to tensors
- `pd.Series` auto-converted
- `dtype` and `device` parameters respected
- Empty args → `ValueError`
- Unequal-length inputs → `ValueError`
- Non-tensor/list/tuple input → `TypeError`
- 2D tensor with shape `(n, 1)` should work (unsqueeze logic)

#### D. `augment`

- Two matrices → horizontal concatenation
- Three or more matrices
- Preserves row count
- Column count is sum of input columns
- Works with tensors of same row count but different column counts

#### E. `interaction`

- Two categorical lists → joined with default separator `⋈`
- Custom separator
- Three or more lists
- Single list → joins each element as string (trivially)
- Empty args → `ValueError`
- Non-list/tuple input → `TypeError`
- Result length matches input length

#### F. `n_distinct`

- List with duplicates → count of unique elements
- List with all unique elements → length of list
- List with all same element → 1
- Empty list → 0
- `pd.Series` input → convert and count
- Non-list/tuple/Series → `TypeError`

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Each function tested independently
- Use parametrize for edge case variations
- All functions are pure — no mocking needed
- Explicit value assertions

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
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

    def test_mismatch_with_different_cpu_tensors(self):
        # With one device type, should not raise
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        assert get_device(x, y) == x.device


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
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **`get_device` mismatch testing**: Requires either two different physical devices (CPU + CUDA) or is untestable without CUDA. The test can use `pytest.mark.skipif` for CUDA tests.
- **`numeric_to_design_matrix` and `len(x.shape) == 2`**: The code does `x.unsqueeze_(0)` if input is 2D — this is likely meant to handle `(1, n)` shapes by adding a leading dim, but `unsqueeze_(0)` adds a dim at position 0 which makes `(1, n)` → `(1, 1, n)`. This might be a bug (should be `squeeze` or reshaped differently). The test plan includes a test for 2D inputs to exercise this path.
- **`interaction` with lists of non-strings**: The function doesn't enforce string elements — it just calls `sep.join(parts)` which will fail at runtime if elements aren't strings.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| `get_device` | 4 |
| `get_dtype` | 4 |
| `numeric_to_design_matrix` | 9 |
| `augment` | 2 |
| `interaction` | 6 |
| `n_distinct` | 6 |
| **Total** | **31** |
