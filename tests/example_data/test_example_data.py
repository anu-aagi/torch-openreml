import pandas as pd
from torch_openreml.example_data import fa_met, fa_met_params, john_alpha


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


class TestFaMet:
    """Tests for the simulated FA(2) multi-environment trial dataset."""

    def test_is_dataframe(self):
        assert isinstance(fa_met, pd.DataFrame)

    def test_row_count(self):
        assert len(fa_met) == 1200

    def test_columns(self):
        expected = [
            "genotype",
            "environment",
            "rep",
            "yield",
            "genetic",
        ]
        assert list(fa_met.columns) == expected

    def test_no_missing_values(self):
        assert not fa_met.isnull().any().any()

    def test_labels_are_strings(self):
        assert pd.api.types.is_string_dtype(fa_met["genotype"])
        assert pd.api.types.is_string_dtype(fa_met["environment"])

    def test_dtypes(self):
        assert fa_met["yield"].dtype == "float64"
        assert fa_met["genetic"].dtype == "float64"

    def test_design(self):
        assert fa_met["genotype"].nunique() == 100
        assert fa_met["environment"].nunique() == 6
        assert set(fa_met["environment"]) == {"01", "02", "03", "04", "05", "06"}
        assert set(fa_met["rep"]) == {1, 2}

    def test_two_replicates_per_cell(self):
        counts = fa_met.groupby(["genotype", "environment"]).size()
        assert (counts == 2).all()
        reps = fa_met.groupby(["genotype", "environment"])["rep"].apply(set)
        assert (reps == {1, 2}).all()

    def test_genetic_constant_within_cell(self):
        counts = fa_met.groupby(["genotype", "environment"])["genetic"].nunique()
        assert (counts == 1).all()

    def test_labels_zero_padded(self):
        assert fa_met["genotype"].iloc[0] == "001"
        assert fa_met["genotype"].iloc[-1] == "100"

    def test_first_row_values(self):
        row = fa_met.iloc[0]
        assert row["genotype"] == "001"
        assert row["environment"] == "01"
        assert row["rep"] == 1
        assert abs(row["yield"] - 4.779848473982807) < 1e-9
        assert abs(row["genetic"] - -0.2310389928947935) < 1e-12

    def test_last_row_values(self):
        row = fa_met.iloc[-1]
        assert row["genotype"] == "100"
        assert row["environment"] == "06"
        assert row["rep"] == 2
        assert abs(row["yield"] - 6.190447018808117) < 1e-9
        assert abs(row["genetic"] - 0.7592447668793535) < 1e-12


class TestFaMetParams:
    """Tests for the generating values of the simulated FA(2) trial."""

    def test_is_dict(self):
        assert isinstance(fa_met_params, dict)

    def test_keys(self):
        expected = {"means", "loadings", "uniqueness", "sigma2"}
        assert set(fa_met_params) == expected

    def test_means(self):
        assert fa_met_params["means"] == [5.0, 5.2, 4.8, 5.1, 5.3, 4.9]

    def test_loadings_shape(self):
        assert len(fa_met_params["loadings"]) == 6
        assert all(len(row) == 2 for row in fa_met_params["loadings"])

    def test_loadings_lower_triangular(self):
        assert fa_met_params["loadings"][0][1] == 0.0

    def test_uniqueness(self):
        assert fa_met_params["uniqueness"] == [0.5, 0.4, 0.5, 0.6, 0.4, 0.5]

    def test_sigma2(self):
        assert fa_met_params["sigma2"] == 0.3
