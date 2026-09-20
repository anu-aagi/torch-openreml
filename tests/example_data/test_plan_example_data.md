# Module Test Plan

## 1. Overview

**Module Name:** `torch_openreml.example_data`
**Purpose of Module:**
Provides example datasets for testing and demonstration. Currently contains `john_alpha` — a pandas DataFrame representing John's alpha design (an incomplete block design commonly used in plant breeding trials), with 72 observations across 3 reps, 6 blocks per rep, and 24 genotypes.

**Data Exported:**

- `john_alpha` — pandas DataFrame with John's alpha design

**Testing Goal:**
Ensure the dataset is correctly loaded, has the expected structure (columns, row count, dtypes), contains no missing values, and that known values at specific positions are correct.

---

## 2. Testing Scope

### 2.1 What Needs to Be Tested

#### A. Data Structure

- `john_alpha` is a `pandas.DataFrame`
- Has exactly 7 columns: `plot`, `rep`, `block`, `gen`, `yield`, `row`, `col`
- Has exactly 72 rows
- No missing (NaN) values in any column

#### B. Column Types

- `yield` dtype is `float32`
- Categorical columns (`rep`, `block`, `gen`) are strings (object dtype)
- Numeric columns (`plot`, `row`, `col`) are integers

#### C. Data Integrity

- Known values at specific positions match expected data
- `plot` column is 1..72 sequentially
- `rep` has exactly 3 levels: R1, R2, R3
- `block` has exactly 6 levels: B1..B6
- `gen` has exactly 24 levels: G01..G24
- Each rep-block combination contains exactly 4 genotypes (24 genotypes / 6 blocks = 4 per block)
- Yield values are all positive floats

#### D. Importability

- `from torch_openreml.example_data import john_alpha` works
- Module is importable without errors

---

## 3. How to Test

### 3.1 Unit Testing Strategy

- Simple data validation — check structure, types, known values
- No mocking needed

### 3.2 Test Structure

Each test should follow **Arrange → Act → Assert**.

Example structure:

```python
import pandas as pd
from torch_openreml.example_data import john_alpha


class TestJohnAlpha:
    """Tests for the John's alpha design example dataset."""

    def test_is_dataframe(self):
        assert isinstance(john_alpha, pd.DataFrame)

    def test_row_count(self):
        assert len(john_alpha) == 72

    def test_columns(self):
        expected = ["plot", "rep", "block", "gen", "yield", "row", "col"]
        assert list(john_alpha.columns) == expected

    def test_no_missing_values(self):
        assert not john_alpha.isnull().any().any()

    def test_yield_dtype(self):
        assert john_alpha["yield"].dtype == "float32"

    def test_yield_positive(self):
        assert (john_alpha["yield"] > 0).all()

    def test_plot_sequential(self):
        assert john_alpha["plot"].tolist() == list(range(1, 73))

    def test_rep_levels(self):
        assert set(john_alpha["rep"]) == {"R1", "R2", "R3"}

    def test_block_levels(self):
        expected = {"B1", "B2", "B3", "B4", "B5", "B6"}
        assert set(john_alpha["block"]) == expected

    def test_gen_count(self):
        assert john_alpha["gen"].nunique() == 24

    def test_first_row_values(self):
        row = john_alpha.iloc[0]
        assert row["plot"] == 1
        assert row["rep"] == "R1"
        assert row["block"] == "B1"
        assert row["gen"] == "G11"
        assert abs(row["yield"] - 4.1172) < 1e-4

    def test_last_row_values(self):
        row = john_alpha.iloc[-1]
        assert row["plot"] == 72
        assert row["rep"] == "R3"
        assert row["block"] == "B6"
        assert row["gen"] == "G07"
        assert abs(row["yield"] - 3.6096) < 1e-4
```

### 3.3 Test Execution

Do not need to execute.

---

## 4. Key Risks / Notes

- **Data immutability**: Tests should not modify the DataFrame. Each test reads independently.
- **No parametrization needed**: This is a fixed dataset — hardcoded expected values are appropriate.
- **yield precision**: float32 has ~7 decimal digits of precision. Use `atol=1e-4` for comparisons.

---

## 5. Expected Test Count Summary

| Category | Tests |
|---|---|
| Structure (type, rows, columns) | 3 |
| Data quality (missing, yield positive) | 2 |
| Column types & levels | 5 |
| Known values | 2 |
| **Total** | **12** |
