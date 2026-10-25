"""Train and evaluate a PyTorch autoencoder for transaction anomaly detection.

Methodology:
- Chronological 60/20/20 split shared with other FraudShield phases.
- Train the autoencoder only on legitimate rows from the training partition.
- Exclude Time from the autoencoder inputs because a chronological split makes
  future Time values systematically larger; including it can flag time drift
  rather than unusual transaction patterns.
- Apply log1p to Amount, then fit StandardScaler on legitimate training rows only.
- Set the primary anomaly threshold to the 99th percentile of reconstruction
  error for legitimate validation rows (predefined 1% target validation FPR).
- Gradient updates use only legitimate training rows. Validation labels are used
  to select the checkpoint by Average Precision; the threshold itself is based
  on legitimate validation scores only. Test labels are reserved for final
  evaluation and a predeclared threshold-sensitivity report.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data.splitting import chronological_train_validation_test_split

FEATURE_COLUMNS = [*(f"V{i}" for i in range(1, 29)), "Amount_log1p"]
DEFAULT_THRESHOLD_QUANTILE = 0.99


class TransactionAutoencoder(nn.Module):
    """Small fully connected autoencoder for 29 scaled transaction features."""

    def __init__(self, input_dim: int = 29, latent_dim: int = 8) -> None:
        super().__init__()
        if input_dim < 2 or latent_dim < 1 or latent_dim >= input_dim:
            raise ValueError("Require input_dim >= 2 and 1 <= latent_dim < input_dim")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features without using target or the time index."""
    required = [*(f"V{i}" for i in range(1, 29)), "Amount"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing autoencoder features: {missing}")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount contains negative values; log1p feature requires non-negative amounts")
    features = df[[*(f"V{i}" for i in range(1, 29))]].copy()
    features["Amount_log1p"] = np.log1p(df["Amount"].astype(float).to_numpy())
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Autoencoder features contain NaN or infinite values")
    return features[FEATURE_COLUMNS]


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def reconstruction_errors(
    model: TransactionAutoencoder,
    features_scaled: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return per-row mean squared reconstruction error."""
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features_scaled), batch_size):
            batch = torch.as_tensor(
                features_scaled[start:start + batch_size], dtype=torch.float32, device=device
            )
            reconstruction = model(batch)
            errors = torch.mean(torch.square(reconstruction - batch), dim=1)
            outputs.append(errors.detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "n_legitimate": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "flagged_total": int(predicted.sum()),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
    }


def _save_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Predicted normal", "Predicted anomaly"],
        yticklabels=["Actual legitimate", "Actual fraud"],
        xlabel="Prediction", ylabel="Actual class", title=title,
    )
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_loss(history: list[dict[str, float]], path: Path) -> None:
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(h["epoch"], h["train_loss"], label="Train loss (legitimate only)")
    ax.plot(h["epoch"], h["validation_loss_legitimate_only"], label="Validation loss (legitimate only)")
    ax.set(title="Autoencoder reconstruction loss", xlabel="Epoch", ylabel="Mean squared error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_score_distribution(
    normal_scores: np.ndarray, fraud_scores: np.ndarray, threshold: float, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    # log1p display prevents a small number of very large errors from compressing
    # the main legitimate/fraud score distributions into a narrow strip.
    normal_display = np.log1p(np.asarray(normal_scores))
    fraud_display = np.log1p(np.asarray(fraud_scores))
    ax.hist(normal_display, bins=70, density=True, alpha=0.65, label="Legitimate test rows")
    if len(fraud_display):
        ax.hist(fraud_display, bins=50, density=True, alpha=0.65, label="Fraud test rows")
    ax.axvline(np.log1p(threshold), linestyle="--", linewidth=2, label=f"log1p(validation q99 threshold) = {np.log1p(threshold):.4f}")
    ax.set(title="Test reconstruction-error distribution (log1p display)", xlabel="log1p(mean squared reconstruction error)", ylabel="Density")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_precision_recall(y: np.ndarray, scores: np.ndarray, path: Path) -> None:
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(recall, precision, label="Autoencoder anomaly score")
    ax.axhline(float(np.mean(y)), linestyle="--", label=f"Random baseline = prevalence {np.mean(y):.3%}")
    ax.set(title="Autoencoder precision-recall curve (test)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fit_model(
    x_train: np.ndarray,
    x_val_normal: np.ndarray,
    x_val_all: np.ndarray,
    y_val: np.ndarray,
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[TransactionAutoencoder, list[dict[str, float]], int, float, float]:
    _set_seeds(seed)
    model = TransactionAutoencoder(input_dim=x_train.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss(reduction="mean")
    train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_tensor = torch.as_tensor(x_val_normal, dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    best_validation_ap = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_train_loss = 0.0
        seen = 0
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            n = len(batch)
            weighted_train_loss += float(loss.detach().cpu()) * n
            seen += n

        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_seen = 0
            for start in range(0, len(val_tensor), 4096):
                batch = val_tensor[start:start + 4096]
                loss = loss_fn(model(batch), batch)
                n = len(batch)
                val_total += float(loss.detach().cpu()) * n
                val_seen += n
        train_loss = weighted_train_loss / max(seen, 1)
        val_loss = val_total / max(val_seen, 1)
        # The model weights are trained on legitimate rows only. We use the
        # labeled validation partition only for checkpoint selection, a standard
        # model-selection decision; it never contributes gradients to training.
        val_scores = reconstruction_errors(model, x_val_all, device=device)
        val_ap = float(average_precision_score(y_val, val_scores))
        val_roc_auc = float(roc_auc_score(y_val, val_scores)) if len(np.unique(y_val)) == 2 else float("nan")
        history.append({
            "epoch": float(epoch), "train_loss": train_loss,
            "validation_loss_legitimate_only": val_loss,
            "validation_average_precision_label_informed_selection": val_ap,
            "validation_roc_auc": val_roc_auc,
        })

        if val_ap > best_validation_ap + 1e-7:
            best_validation_ap = val_ap
            best_validation_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Autoencoder training failed to produce a best state")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, best_epoch, float(best_validation_ap), float(best_validation_loss)


def run_autoencoder_experiment(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    *,
    epochs: int = 35,
    batch_size: int = 1024,
    latent_dim: int = 8,
    patience: int = 6,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    max_train_normal_rows: int = 60000,
) -> dict[str, Any]:
    """Run the whole anomaly-detection experiment and write reproducible artifacts."""
    project_root = Path(__file__).resolve().parents[2]
    config = load_config()
    data_path = Path(data_path) if data_path else project_root / config["data"]["raw_path"]
    output_dir = Path(output_dir) if output_dir else project_root / "reports" / "autoencoder"
    model_path = Path(model_path) if model_path else project_root / "models" / "autoencoder_state.pt"
    preprocessor_path = project_root / "models" / "autoencoder_preprocessor.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Put creditcard.csv in data/raw/.")
    if epochs < 1 or batch_size < 1 or patience < 1 or max_train_normal_rows < 1:
        raise ValueError("epochs, batch_size, patience, and max_train_normal_rows must be positive integers")
    if not 0.5 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be greater than 0.5 and less than 1.0")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    df = pd.read_csv(data_path)
    if "Class" not in df or "Time" not in df:
        raise ValueError("Dataset must contain Time and Class columns")
    if df["Class"].isna().any() or not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain non-missing binary labels 0/1")
    splits = chronological_train_validation_test_split(df, time_column="Time", train_fraction=0.60, validation_fraction=0.20)
    X_train_all = prepare_features(splits.train)
    X_val_all = prepare_features(splits.validation)
    X_test_all = prepare_features(splits.test)
    y_train = splits.train["Class"].astype(int).to_numpy()
    y_val = splits.validation["Class"].astype(int).to_numpy()
    y_test = splits.test["Class"].astype(int).to_numpy()
    x_train_normal_raw = X_train_all.loc[y_train == 0].to_numpy(dtype=np.float32)
    x_val_normal_raw = X_val_all.loc[y_val == 0].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    # Fit preprocessing on all legitimate training rows, then use a seeded
    # training subset for neural-net fitting to keep CPU runs practical.
    x_train_normal = scaler.fit_transform(x_train_normal_raw).astype(np.float32)
    if len(x_train_normal) > max_train_normal_rows:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(x_train_normal), size=max_train_normal_rows, replace=False))
        x_train_model = x_train_normal[chosen]
    else:
        x_train_model = x_train_normal
    x_val_normal = scaler.transform(x_val_normal_raw).astype(np.float32)
    x_val_all = scaler.transform(X_val_all.to_numpy(dtype=np.float32)).astype(np.float32)
    x_test_all = scaler.transform(X_test_all.to_numpy(dtype=np.float32)).astype(np.float32)

    start_time = time.time()
    model, history, best_epoch, best_validation_ap, best_validation_loss = _fit_model(
        x_train_model,
        x_val_normal,
        x_val_all,
        y_val,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
    training_seconds = time.time() - start_time

    val_errors = reconstruction_errors(model, x_val_all, device=device)
    val_errors_normal = val_errors[y_val == 0]
    threshold = float(np.quantile(val_errors_normal, threshold_quantile, method="higher"))
    test_errors = reconstruction_errors(model, x_test_all, device=device)

    val_metrics = _classification_metrics(y_val, val_errors, threshold)
    test_metrics = _classification_metrics(y_test, test_errors, threshold)
    # Threshold sensitivity is predeclared by normal-validation quantiles. The
    # 99th-percentile threshold is the primary policy; other rows are diagnostics.
    sensitivity_rows: list[dict[str, Any]] = []
    for q in (0.95, 0.975, 0.99, 0.995, 0.999):
        q_threshold = float(np.quantile(val_errors_normal, q, method="higher"))
        v = _classification_metrics(y_val, val_errors, q_threshold)
        t = _classification_metrics(y_test, test_errors, q_threshold)
        sensitivity_rows.append({
            "normal_validation_quantile": q,
            "threshold": q_threshold,
            "validation_fpr": v["false_positive_rate"],
            "validation_precision": v["precision"],
            "validation_recall": v["recall"],
            "validation_average_precision": v["average_precision"],
            "test_fpr": t["false_positive_rate"],
            "test_precision": t["precision"],
            "test_recall": t["recall"],
            "test_f1": t["f1"],
            "test_false_positive": t["false_positive"],
            "test_false_negative": t["false_negative"],
            "test_average_precision": t["average_precision"],
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    # Compare with the existing supervised model on exactly the same chronological test rows.
    xgb_metrics: dict[str, Any] | None = None
    xgb_model_path = project_root / "models" / "xgboost_best.joblib"
    xgb_policy_path = project_root / "models" / "decision_policy.json"
    if xgb_model_path.exists():
        xgb_model = joblib.load(xgb_model_path)
        xgb_feature_columns = [c for c in df.columns if c != "Class"]
        xgb_scores = xgb_model.predict_proba(splits.test[xgb_feature_columns])[:, 1]
        selected_xgb_threshold = 0.5
        if xgb_policy_path.exists():
            policy = json.loads(xgb_policy_path.read_text(encoding="utf-8"))
            selected_xgb_threshold = float(policy.get("decision", {}).get("threshold", 0.5))
        xgb_metrics = _classification_metrics(y_test, xgb_scores, selected_xgb_threshold)
        xgb_metrics.update({"model": "XGBoost supervised baseline", "threshold_method": "Phase 6 validation-selected threshold"})

    # Persist state_dict and enough metadata to reconstruct architecture safely.
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(len(FEATURE_COLUMNS)),
        "latent_dim": int(latent_dim),
        "feature_columns": FEATURE_COLUMNS,
        "architecture": "29 -> 64 -> 32 -> latent -> 32 -> 64 -> 29; ReLU hidden activations",
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_average_precision": float(best_validation_ap),
        "best_validation_reconstruction_loss_on_legitimate_rows": float(best_validation_loss),
        "checkpoint_selection": "maximize validation Average Precision using validation labels; model weight updates use legitimate training rows only",
    }
    torch.save(checkpoint, model_path)
    joblib.dump({
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "amount_transform": "numpy.log1p(Amount)",
        "time_feature_excluded": True,
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
        "threshold_source": "legitimate validation reconstruction errors only",
    }, preprocessor_path)

    # Export history and predictions.
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    prediction_df = pd.DataFrame({
        "Time": splits.test["Time"].to_numpy(),
        "actual_class": y_test,
        "reconstruction_error": test_errors,
        "is_anomaly_at_threshold": (test_errors >= threshold).astype(int),
    })
    if xgb_metrics is not None:
        prediction_df["xgboost_score"] = xgb_scores
        prediction_df["xgboost_flag_at_phase6_threshold"] = (xgb_scores >= selected_xgb_threshold).astype(int)
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Artifacts/plots.
    _plot_loss(history, output_dir / "training_loss.png")
    _plot_score_distribution(
        test_errors[y_test == 0], test_errors[y_test == 1], threshold,
        output_dir / "test_reconstruction_error_distribution.png",
    )
    _plot_precision_recall(y_test, test_errors, output_dir / "test_precision_recall_curve.png")
    test_pred = (test_errors >= threshold).astype(int)
    test_cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    _save_confusion_matrix(test_cm, output_dir / "confusion_matrix_test.png", "Autoencoder test confusion matrix (validation q99 threshold)")

    comparison_rows = [{
        "model": "PyTorch autoencoder (normal-only training)",
        "threshold_method": f"{threshold_quantile:.1%} quantile of legitimate validation errors",
        **test_metrics,
    }]
    if xgb_metrics is not None:
        comparison_rows.append(xgb_metrics)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "test_model_comparison.csv", index=False)

    metric_fields = ["average_precision", "roc_auc", "precision", "recall", "f1", "false_positive", "false_negative"]
    fig, axes = plt.subplots(1, len(metric_fields), figsize=(16, 4.8))
    plot_df = comparison_df.set_index("model")
    for ax, field in zip(axes, metric_fields):
        vals = plot_df[field].astype(float)
        ax.bar(range(len(vals)), vals.to_numpy())
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([s.replace(" (", "\n(") for s in vals.index], rotation=20, ha="right", fontsize=7)
        ax.set_title(field.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Test comparison: anomaly ranking vs supervised fraud scores")
    fig.tight_layout()
    fig.savefig(output_dir / "test_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    train_val_threshold_info = {
        "normal_train_rows": int(len(x_train_normal)),
        "model_training_normal_rows": int(len(x_train_model)),
        "max_train_normal_rows": int(max_train_normal_rows),
        "validation_normal_rows": int(len(x_val_normal)),
        "validation_fraud_rows": int((y_val == 1).sum()),
        "test_rows": int(len(y_test)),
        "test_fraud_rows": int(y_test.sum()),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_features": {"Time": "Excluded because chronological future time values shift mechanically; including it can create false anomalies unrelated to transaction pattern."},
        "amount_transform": "log1p(Amount), then StandardScaler fitted on legitimate training rows only",
        "split_cutoffs": splits.cutoffs,
        "device": str(device),
        "training_seconds": round(training_seconds, 3),
        "epochs_requested": int(epochs),
        "epochs_run": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_validation_average_precision_for_checkpoint_selection": float(best_validation_ap),
        "best_validation_reconstruction_loss_on_legitimate_rows": float(best_validation_loss),
        "checkpoint_selection_method": "validation Average Precision (validation labels used only for checkpoint/model selection; no validation gradients)",
        "batch_size": int(batch_size),
        "latent_dim": int(latent_dim),
        "learning_rate": float(learning_rate),
        "threshold_quantile_of_legitimate_validation_errors": float(threshold_quantile),
        "anomaly_threshold": threshold,
        "validation_metrics_at_selected_threshold": val_metrics,
        "test_metrics_at_selected_threshold": test_metrics,
        "xgboost_test_metrics_at_phase6_threshold": xgb_metrics,
        "training_loss_last_epoch": float(history[-1]["train_loss"]),
        "validation_loss_last_epoch": float(history[-1]["validation_loss_legitimate_only"]),
        "median_test_error_legitimate": float(np.median(test_errors[y_test == 0])),
        "median_test_error_fraud": float(np.median(test_errors[y_test == 1])) if (y_test == 1).any() else None,
        "test_error_q99_legitimate": float(np.quantile(test_errors[y_test == 0], 0.99)),
        "threshold_note": "Threshold chosen without fraud labels: 99th percentile of legitimate validation errors. Validation labels are used only for metrics; test labels are used only for final evaluation and sensitivity reporting.",
        "caveats": [
            "Reconstruction error is an anomaly score, not a calibrated fraud probability.",
            "Unusual legitimate transactions can be flagged, and fraud can be reconstructed well.",
            "This dataset is historical and anonymized; validation/test class prevalence and distributions differ over time.",
            "Sensitivity rows are diagnostics; do not select a production threshold from test labels.",
        ],
        "artifacts": {
            "model_checkpoint": str(model_path.relative_to(project_root)),
            "preprocessor": str(preprocessor_path.relative_to(project_root)),
            "training_history": str((output_dir / "training_history.csv").relative_to(project_root)),
            "threshold_sensitivity": str((output_dir / "threshold_sensitivity.csv").relative_to(project_root)),
            "test_predictions": str((output_dir / "test_predictions.csv").relative_to(project_root)),
            "model_comparison": str((output_dir / "test_model_comparison.csv").relative_to(project_root)),
        },
    }
    summary_path = output_dir / "autoencoder_summary.json"
    summary_path.write_text(json.dumps(train_val_threshold_info, indent=2), encoding="utf-8")
    _write_findings_report(train_val_threshold_info, sensitivity_df, comparison_df, output_dir / "PHASE8_AUTOENCODER_FINDINGS.md")
    return train_val_threshold_info



def load_autoencoder_artifacts(
    model_path: str | Path,
    preprocessor_path: str | Path,
    *,
    device_name: str = "cpu",
) -> tuple[TransactionAutoencoder, dict[str, Any], torch.device]:
    """Load model weights and preprocessor for inference or future hybrid scoring."""
    device = torch.device(device_name)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = TransactionAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor, device


def score_transactions(
    transactions: pd.DataFrame,
    *,
    model_path: str | Path,
    preprocessor_path: str | Path,
    device_name: str = "cpu",
) -> pd.DataFrame:
    """Score raw transaction rows and flag anomalies using the saved threshold.

    `transactions` must include V1–V28 and Amount. The target Class is ignored;
    Time is intentionally ignored by feature preparation.
    """
    model, preprocessor, device = load_autoencoder_artifacts(
        model_path, preprocessor_path, device_name=device_name
    )
    features = prepare_features(transactions)
    scaled = preprocessor["scaler"].transform(features.to_numpy(dtype=np.float32)).astype(np.float32)
    scores = reconstruction_errors(model, scaled, device=device)
    threshold = float(preprocessor["threshold"])
    result = pd.DataFrame({
        "anomaly_score": scores,
        "is_anomaly": (scores >= threshold).astype(int),
        "threshold": threshold,
    }, index=transactions.index)
    return result


def _write_findings_report(summary: dict[str, Any], sensitivity: pd.DataFrame, comparison: pd.DataFrame, path: Path) -> None:
    val = summary["validation_metrics_at_selected_threshold"]
    test = summary["test_metrics_at_selected_threshold"]
    lines = [
        "# Phase 8 — PyTorch Autoencoder Anomaly Detection Findings",
        "",
        "## Method",
        "",
        "- Chronological train/validation/test split using the same 60/20/20 strategy as earlier phases.",
        f"- Fit preprocessing using {summary['normal_train_rows']:,} legitimate training rows, and fit neural weights on a seeded sample of {summary['model_training_normal_rows']:,} of those rows; validation has {summary['validation_normal_rows']:,} legitimate rows and {summary['validation_fraud_rows']:,} fraud rows; test has {summary['test_rows']:,} rows including {summary['test_fraud_rows']} fraud cases.",
        f"- Features: {len(summary['feature_columns'])} values (V1–V28 and log1p(Amount)); Time is excluded to avoid the chronological time index dominating anomaly scores.",
        "- StandardScaler is fitted only on legitimate training data.",
        f"- Architecture: {summary['input_dim'] if 'input_dim' in summary else 29} → 64 → 32 → {summary['latent_dim']} latent dimensions → 32 → 64 → 29, with ReLU in hidden layers.",
        f"- The selected checkpoint is epoch {summary['best_epoch']} (validation Average Precision {summary['best_validation_average_precision_for_checkpoint_selection']:.4f}); best legitimate-validation reconstruction loss recorded for that checkpoint: {summary['best_validation_reconstruction_loss_on_legitimate_rows']:.4f}. Device: {summary['device']}; training time: {summary['training_seconds']:.1f}s.",
        f"- Threshold: {summary['anomaly_threshold']:.8g}, the {summary['threshold_quantile_of_legitimate_validation_errors']:.1%} quantile of reconstruction errors on legitimate validation rows. Labels were not used to set this threshold.",
        "",
        "## What reconstruction error means",
        "",
        "For each row, the model reconstructs the scaled feature vector. The anomaly score is the mean squared error across the 29 features. Larger errors mean the row was harder for this autoencoder to reconstruct; they do not prove fraud and are not fraud probabilities.",
        "",
        "## Validation metrics at the selected threshold",
        "",
        f"- Precision: {val['precision']:.4f}",
        f"- Recall: {val['recall']:.4f}",
        f"- F1: {val['f1']:.4f}",
        f"- ROC-AUC: {val['roc_auc']:.4f}",
        f"- Average Precision: {val['average_precision']:.4f}",
        f"- False positives: {val['false_positive']:,}; false negatives: {val['false_negative']:,}; validation legitimate FPR: {val['false_positive_rate']:.3%}",
        "",
        "## Held-out test metrics at the validation-selected threshold",
        "",
        f"- Precision: {test['precision']:.4f}",
        f"- Recall: {test['recall']:.4f}",
        f"- F1: {test['f1']:.4f}",
        f"- ROC-AUC: {test['roc_auc']:.4f}",
        f"- Average Precision: {test['average_precision']:.4f}",
        f"- Confusion counts: TN={test['true_negative']:,}, FP={test['false_positive']:,}, FN={test['false_negative']:,}, TP={test['true_positive']:,}",
        "",
        "## Comparison to supervised XGBoost",
        "",
    ]
    if summary.get("xgboost_test_metrics_at_phase6_threshold"):
        xgb = summary["xgboost_test_metrics_at_phase6_threshold"]
        lines += [
            f"- Autoencoder Average Precision: {test['average_precision']:.4f}; XGBoost Average Precision: {xgb['average_precision']:.4f}.",
            f"- Autoencoder recall at its validation-normal q{summary['threshold_quantile_of_legitimate_validation_errors']:.0%} threshold: {test['recall']:.4f}; XGBoost recall at Phase 6 threshold {xgb['threshold']:.6f}: {xgb['recall']:.4f}.",
            f"- Autoencoder false positives: {test['false_positive']:,}; XGBoost false positives: {xgb['false_positive']:,}.",
            "- The two decision thresholds have different selection rules. Compare score-ranking metrics and confusion counts, and do not claim apples-to-apples threshold optimality.",
        ]
    lines += [
        "",
        "## Threshold sensitivity (diagnostic only)",
        "",
        "| Normal-validation quantile | Threshold | Validation FPR | Validation recall | Test FPR | Test recall | Test FP | Test FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *["| " + " | ".join([
            f"{row.normal_validation_quantile:.3f}", f"{row.threshold:.6g}",
            f"{row.validation_fpr:.4f}", f"{row.validation_recall:.4f}",
            f"{row.test_fpr:.4f}", f"{row.test_recall:.4f}",
            str(int(row.test_false_positive)), str(int(row.test_false_negative)),
        ]) + " |" for row in sensitivity.itertuples(index=False)],
        "",
        "## Interpretation and limitations",
        "",
        "1. The autoencoder's gradient updates use only legitimate training rows. To choose the checkpoint/epoch, the experiment uses validation Average Precision with validation labels; this is label-informed model selection but does not train the model on fraud examples. The primary threshold is separately computed from legitimate validation reconstruction errors only.",
        "2. The threshold is based on legitimate validation reconstruction errors, not the best validation F1 and not test labels. Because data distributions change over time, test false-positive rate may differ from the target validation rate.",
        "3. Excluding Time is intentional: the later chronological periods naturally have larger Time values, which can become a spurious anomaly signal.",
        "4. Standardization and log1p(Amount) are learned/applied inside the training pipeline; scaler statistics come only from legitimate training rows.",
        "5. A high reconstruction error means unusual relative to this trained reconstruction model—not necessarily fraudulent. A fraud transaction may reconstruct well, and a legitimate outlier may reconstruct poorly.",
        "6. The public dataset is anonymized and historical. This is a technical prototype, not a production banking fraud detector.",
        "",
        "## Re-run",
        "",
        "```powershell",
        "python -m pip install -r requirements.txt",
        "python -m src.models.autoencoder_anomaly",
        "python -m pytest -q",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--max-train-normal-rows", type=int, default=60000)
    args = parser.parse_args()
    result = run_autoencoder_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        threshold_quantile=args.threshold_quantile,
        max_train_normal_rows=args.max_train_normal_rows,
    )
    print(json.dumps({
        "device": result["device"],
        "best_epoch": result["best_epoch"],
        "best_validation_average_precision_for_checkpoint_selection": result["best_validation_average_precision_for_checkpoint_selection"],
        "anomaly_threshold": result["anomaly_threshold"],
        "validation_metrics": result["validation_metrics_at_selected_threshold"],
        "test_metrics": result["test_metrics_at_selected_threshold"],
        "xgboost_test_metrics": result["xgboost_test_metrics_at_phase6_threshold"],
        "summary": str(Path(__file__).resolve().parents[2] / "reports" / "autoencoder" / "autoencoder_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
