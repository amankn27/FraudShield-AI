import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.autoencoder_anomaly import (
    TransactionAutoencoder,
    load_autoencoder_artifacts,
    prepare_features,
    reconstruction_errors,
    score_transactions,
)


def make_transactions(n=6):
    data = {f"V{i}": np.linspace(-1, 1, n) + i * 0.01 for i in range(1, 29)}
    data["Time"] = np.arange(n) * 10
    data["Amount"] = np.linspace(0, 100, n)
    data["Class"] = [0] * n
    return pd.DataFrame(data)


def test_prepare_features_excludes_time_and_target_and_log_transforms_amount():
    df = make_transactions(3)
    features = prepare_features(df)
    assert features.shape == (3, 29)
    assert "Time" not in features.columns
    assert "Class" not in features.columns
    assert features.loc[0, "Amount_log1p"] == 0.0
    assert np.isclose(features.loc[2, "Amount_log1p"], np.log1p(100))


def test_autoencoder_output_shape_matches_input():
    model = TransactionAutoencoder(input_dim=29, latent_dim=8)
    x = torch.randn(4, 29)
    y = model(x)
    assert y.shape == x.shape
    assert model.encoder(x).shape == (4, 8)


def test_reconstruction_errors_are_finite_and_nonnegative():
    model = TransactionAutoencoder(input_dim=29, latent_dim=8)
    x = np.random.default_rng(4).normal(size=(5, 29)).astype(np.float32)
    scores = reconstruction_errors(model, x, device=torch.device("cpu"), batch_size=2)
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_saved_artifacts_round_trip_scores(tmp_path):
    df = make_transactions(4)
    features = prepare_features(df)
    scaler = StandardScaler().fit(features.to_numpy(dtype=np.float32))
    model = TransactionAutoencoder(input_dim=29, latent_dim=8)
    model_path = tmp_path / "model.pt"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": 29,
        "latent_dim": 8,
        "feature_columns": list(features.columns),
        "architecture": "test",
        "seed": 42,
        "best_epoch": 1,
    }, model_path)
    joblib.dump({"scaler": scaler, "feature_columns": list(features.columns), "threshold": 1.5}, preprocessor_path)
    loaded_model, loaded_preprocessor, device = load_autoencoder_artifacts(model_path, preprocessor_path)
    assert loaded_model.training is False
    result = score_transactions(df, model_path=model_path, preprocessor_path=preprocessor_path)
    assert list(result.columns) == ["anomaly_score", "is_anomaly", "threshold"]
    assert len(result) == len(df)
    assert np.isfinite(result["anomaly_score"]).all()
    assert set(result["is_anomaly"].unique()).issubset({0, 1})
    assert (result["threshold"] == 1.5).all()
