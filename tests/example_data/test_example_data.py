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
