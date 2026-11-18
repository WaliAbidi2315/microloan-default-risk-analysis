import pandas as pd
from pathlib import Path
from src.train import fit_preprocessor

ROOT = Path(__file__).resolve().parents[1]


def test_sample_has_expected_shape_and_prevalence():
    data = pd.read_csv(ROOT / "data/synthetic_microloan_sample.csv", index_col="master_row_id")
    assert data.shape == (900, 9)
    assert int(data["default"].sum()) == 180


def test_preprocessing_learns_from_fit_and_removes_missing_values():
    fit = pd.DataFrame({"income": [10.0, None, 30.0], "distance": [1.0, 2.0, 3.0]})
    apply = pd.DataFrame({"income": [None, 50.0], "distance": [4.0, 5.0]})
    fit_imp, apply_imp, fit_scaled, apply_scaled = fit_preprocessor(fit, apply)
    assert not pd.isna(fit_imp).any()
    assert not pd.isna(apply_imp).any()
    # Fit-only median for income is 20, not influenced by apply values.
    assert fit_imp[1, 0] == 20.0
    assert apply_imp[0, 0] == 20.0
    assert fit_scaled.shape == (3, 2)
    assert apply_scaled.shape == (2, 2)
