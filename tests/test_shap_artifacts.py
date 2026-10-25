import json
from pathlib import Path
import pandas as pd

from src.config import PROJECT_ROOT


def test_shap_artifacts_exist_and_additivity_is_small():
    out = PROJECT_ROOT / "reports" / "explainability"
    summary = json.loads((out / "shap_summary.json").read_text(encoding="utf-8"))
    assert summary["background_rows"] >= 10
    assert summary["test_rows_in_explanation_sample"] > 0
    assert summary["max_absolute_shap_additivity_error"] < 5e-4
    assert (out / "global_shap_bar.png").is_file()
    assert (out / "global_shap_beeswarm.png").is_file()
    assert (out / "local" / "waterfall_true_positive.png").is_file()


def test_global_shap_importance_is_sorted_descending():
    out = PROJECT_ROOT / "reports" / "explainability" / "global_shap_importance.csv"
    df = pd.read_csv(out)
    assert len(df) == 30
    values = df["mean_abs_shap_value"].to_numpy()
    assert (values[:-1] >= values[1:]).all()
