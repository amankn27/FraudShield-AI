"""FraudShield AI Phase 9: leakage-aware XGBoost + autoencoder hybrid.

Protocol
--------
1. Use the project's fixed chronological 60/20/20 split.
2. Fit XGBoost on training only using the Phase 5 selected hyperparameters.
3. Train an autoencoder on legitimate training rows only. Select its epoch by
   legitimate-only validation reconstruction loss (no validation labels).
4. Use the first 70% of the chronological validation block to fit a regularized
   logistic stacking model on normalized base-model scores.
5. Use the final 30% of validation, legitimate rows only, to set a separate
   score threshold targeting approximately 1% validation false-positive rate.
6. Evaluate all approaches on the later test block exactly once in this run.

The hybrid score is an uncalibrated ranking score. Its 0-100 display transform
is explicitly NOT a real-world fraud probability. Probability calibration is
not fitted because this historical validation block has only 57 fraud cases.
"""
from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
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
from xgboost import XGBClassifier

from src.config import load_config
from src.data.splitting import chronological_train_validation_test_split
from src.models.autoencoder_anomaly import (
    FEATURE_COLUMNS as AE_FEATURE_COLUMNS,
    TransactionAutoencoder,
    prepare_features as prepare_ae_features,
    reconstruction_errors,
)

DEFAULT_TARGET_FPR = 0.01
DEFAULT_SEED = 42


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


def _logit_transform(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _normalize_signals(xgb_scores: np.ndarray, anomaly_errors: np.ndarray) -> np.ndarray:
    """Transform raw signals before z-score normalization.

    XGBoost probability is logit-transformed to spread scores near 0/1.
    Non-negative reconstruction error uses log1p to reduce right skew.
    """
    err = np.maximum(np.asarray(anomaly_errors, dtype=float), 0.0)
    return np.column_stack([_logit_transform(xgb_scores), np.log1p(err)])


def _threshold_index(time_values: pd.Series, fraction: float) -> int:
    """Find a chronological boundary without splitting equal-Time groups."""
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must be strictly between 0 and 1")
    n = len(time_values)
    idx = min(max(int(round(n * fraction)), 1), n - 1)
    while idx < n and time_values.iloc[idx] == time_values.iloc[idx - 1]:
        idx += 1
    if idx >= n:
        raise ValueError("Unable to create a non-empty temporal validation/calibration split")
    return idx


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    unique_classes = np.unique(y)
    return {
        "n_samples": int(len(y)),
        "n_fraud": int(y.sum()),
        "fraud_prevalence": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, s)) if len(unique_classes) == 2 else None,
        "average_precision": float(average_precision_score(y, s)) if len(unique_classes) == 2 else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "flagged_total": int(pred.sum()),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
    }


def _fit_normal_autoencoder(
    x_train_all: pd.DataFrame,
    y_train: np.ndarray,
    x_val_all: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    max_normal_rows: int,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    latent_dim: int,
) -> tuple[TransactionAutoencoder, StandardScaler, list[dict[str, float]], dict[str, Any]]:
    """Train normal-only and early-stop by legitimate-only validation MSE."""
    _set_seeds(seed)
    x_train_normal_raw = x_train_all.loc[y_train == 0].to_numpy(dtype=np.float32)
    x_val_normal_raw = x_val_all.loc[y_val == 0].to_numpy(dtype=np.float32)
    if len(x_train_normal_raw) == 0 or len(x_val_normal_raw) == 0:
        raise ValueError("Need legitimate rows in both training and validation partitions")

    scaler = StandardScaler()
    train_normal_scaled = scaler.fit_transform(x_train_normal_raw).astype(np.float32)
    val_normal_scaled = scaler.transform(x_val_normal_raw).astype(np.float32)
    if len(train_normal_scaled) > max_normal_rows:
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(len(train_normal_scaled), size=max_normal_rows, replace=False))
        train_for_model = train_normal_scaled[picked]
    else:
        train_for_model = train_normal_scaled

    model = TransactionAutoencoder(input_dim=train_for_model.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    train_tensor = torch.from_numpy(train_for_model)
    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_tensor = torch.as_tensor(val_normal_scaled, dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)

        model.eval()
        with torch.no_grad():
            vloss_sum, vseen = 0.0, 0
            for start in range(0, len(val_tensor), 4096):
                batch = val_tensor[start:start + 4096]
                loss = loss_fn(model(batch), batch)
                vloss_sum += float(loss.detach().cpu()) * len(batch)
                vseen += len(batch)
        train_loss = total_loss / max(seen, 1)
        val_loss = vloss_sum / max(vseen, 1)
        history.append({"epoch": float(epoch), "train_loss_normal_only": train_loss, "validation_loss_legitimate_only": val_loss})

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Autoencoder training failed to select a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    info = {
        "normal_training_rows_available": int(len(x_train_normal_raw)),
        "normal_training_rows_sampled_for_gradient_updates": int(len(train_for_model)),
        "legitimate_validation_rows_for_early_stopping": int(len(val_normal_scaled)),
        "epochs_requested": int(epochs),
        "epochs_run": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_legitimate_validation_mse": float(best_loss),
        "latent_dim": int(latent_dim),
        "device": str(device),
        "training_selection_method": "minimum reconstruction loss on legitimate validation rows only; validation fraud labels not used",
    }
    return model, scaler, history, info


def _save_plots(comparison: pd.DataFrame, test_records: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [("average_precision", "Average Precision"), ("roc_auc", "ROC-AUC"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    display_names = {"XGBoost": "XGBoost", "Autoencoder": "Autoencoder", "Hybrid (XGBoost + autoencoder)": "Hybrid"}
    labels = [display_names.get(x, x) for x in comparison["model"]]
    fig, axes = plt.subplots(1, len(metric_columns), figsize=(15.5, 4.8))
    for ax, (metric, title) in zip(axes, metric_columns):
        ax.bar(labels, comparison[metric])
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Phase 9 test comparison (thresholds target ~1% validation FPR)")
    fig.tight_layout()
    fig.savefig(output_dir / "test_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    y = test_records["actual_class"].to_numpy(dtype=int)
    for column, label in [("xgboost_score", "XGBoost"), ("anomaly_score", "Autoencoder"), ("hybrid_margin", "Hybrid")]:
        precision, recall, _ = precision_recall_curve(y, test_records[column].to_numpy(dtype=float))
        ax.plot(recall, precision, label=label)
    ax.axhline(float(y.mean()), linestyle="--", color="gray", label=f"Test prevalence ({y.mean():.3%})")
    ax.set(title="Test precision-recall curves", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "test_precision_recall_curves.png", dpi=160)
    plt.close(fig)

    # Coefficient plot for the two normalized meta-model signals.
    coefs = comparison.attrs.get("hybrid_coefficients")
    if coefs:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        labels = list(coefs)
        values = [coefs[k] for k in labels]
        ax.barh(labels, values)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(title="Hybrid Logistic Regression coefficients", xlabel="Coefficient on normalized signal")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "hybrid_coefficients.png", dpi=160)
        plt.close(fig)


def run_hybrid_experiment(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    target_fpr: float = DEFAULT_TARGET_FPR,
    max_normal_rows: int = 60000,
    ae_epochs: int = 30,
    ae_batch_size: int = 2048,
    ae_patience: int = 6,
    ae_latent_dim: int = 8,
    n_jobs: int = 2,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train base models, fit validation stacker, select validation threshold, test once."""
    project_root = Path(__file__).resolve().parents[2]
    config = load_config()
    data_path = Path(data_path) if data_path else project_root / config["data"]["raw_path"]
    output_dir = Path(output_dir) if output_dir else project_root / "reports" / "hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not (0 < target_fpr < 0.25):
        raise ValueError("target_fpr must be > 0 and < 0.25")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this PyTorch installation")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    df = pd.read_csv(data_path)
    if "Class" not in df or "Time" not in df:
        raise ValueError("Dataset must contain Time and Class columns")
    if df.isna().any().any() or not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Dataset contains missing values or invalid Class labels")
    splits = chronological_train_validation_test_split(df, time_column="Time", train_fraction=0.60, validation_fraction=0.20)
    train, val, test = splits.train.reset_index(drop=True), splits.validation.reset_index(drop=True), splits.test.reset_index(drop=True)
    y_train = train["Class"].astype(int).to_numpy()
    y_val = val["Class"].astype(int).to_numpy()
    y_test = test["Class"].astype(int).to_numpy()

    # Split validation chronologically into meta-training and threshold calibration.
    meta_train_end = _threshold_index(val["Time"], 0.70)
    meta_train_idx = np.arange(meta_train_end)
    threshold_idx = np.arange(meta_train_end, len(val))
    y_meta_train = y_val[meta_train_idx]
    y_threshold = y_val[threshold_idx]
    if np.unique(y_meta_train).size != 2 or np.unique(y_threshold).size != 2:
        raise ValueError("Both hybrid-training and threshold-calibration periods must contain both classes")

    # XGBoost trained on the training block only: all scored partitions use the
    # same fitted base model, avoiding train-only vs train+validation score shift.
    previous_metrics = json.loads((project_root / "reports" / "xgboost" / "xgboost_metrics.json").read_text(encoding="utf-8"))
    selected_params = previous_metrics["search"]["selected_params"]
    xgb_model = XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        random_state=seed, n_jobs=n_jobs, verbosity=0, **selected_params,
    )
    feature_columns = [col for col in train.columns if col != "Class"]
    xgb_train_x = train[feature_columns]
    xgb_val_x = val[feature_columns]
    xgb_test_x = test[feature_columns]
    xgb_model.fit(xgb_train_x, y_train)
    xgb_val_scores = xgb_model.predict_proba(xgb_val_x)[:, 1]
    xgb_test_scores = xgb_model.predict_proba(xgb_test_x)[:, 1]

    # Train a cleaner phase-9 autoencoder: no fraud labels enter gradient updates
    # or early-stopping/checkpoint selection.
    ae_train_x = prepare_ae_features(train)
    ae_val_x = prepare_ae_features(val)
    ae_test_x = prepare_ae_features(test)
    start = time.time()
    ae_model, ae_scaler, ae_history, ae_info = _fit_normal_autoencoder(
        ae_train_x, y_train, ae_val_x, y_val,
        seed=seed, device=device, max_normal_rows=max_normal_rows,
        epochs=ae_epochs, batch_size=ae_batch_size, patience=ae_patience,
        learning_rate=1e-3, latent_dim=ae_latent_dim,
    )
    ae_fit_seconds = time.time() - start
    x_ae_val = ae_scaler.transform(ae_val_x.to_numpy(dtype=np.float32)).astype(np.float32)
    x_ae_test = ae_scaler.transform(ae_test_x.to_numpy(dtype=np.float32)).astype(np.float32)
    ae_val_errors = reconstruction_errors(ae_model, x_ae_val, device=device)
    ae_test_errors = reconstruction_errors(ae_model, x_ae_test, device=device)

    # Normalize raw score signals. Fit the scaler only on the meta-training slice.
    raw_val_signal = _normalize_signals(xgb_val_scores, ae_val_errors)
    raw_test_signal = _normalize_signals(xgb_test_scores, ae_test_errors)
    score_scaler = StandardScaler().fit(raw_val_signal[meta_train_idx])
    z_val_signal = score_scaler.transform(raw_val_signal)
    z_test_signal = score_scaler.transform(raw_test_signal)

    # Regularized logistic stacking model. Balanced weights are used to learn
    # complementary signals; decision_function is a ranking margin, NOT a calibrated probability.
    hybrid_model = LogisticRegression(
        C=0.1, class_weight="balanced", solver="lbfgs", max_iter=2000,
        random_state=seed,
    )
    hybrid_model.fit(z_val_signal[meta_train_idx], y_meta_train)
    hybrid_val_margin = hybrid_model.decision_function(z_val_signal)
    hybrid_test_margin = hybrid_model.decision_function(z_test_signal)
    hybrid_test_display_score = 100.0 / (1.0 + np.exp(-np.clip(hybrid_test_margin, -30, 30)))

    # A fixed operational objective: calibrate each model's score threshold to
    # the same target FPR on later-validation legitimate rows (labels only used
    # to select legitimate calibration scores; fraud labels do not tune cutoffs).
    y_threshold_normals = y_threshold == 0
    if int(y_threshold_normals.sum()) < 100:
        raise ValueError("Too few legitimate rows in threshold calibration block")
    q = 1.0 - target_fpr
    threshold_xgb = float(np.quantile(xgb_val_scores[threshold_idx][y_threshold_normals], q, method="higher"))
    threshold_ae = float(np.quantile(ae_val_errors[threshold_idx][y_threshold_normals], q, method="higher"))
    threshold_hybrid = float(np.quantile(hybrid_val_margin[threshold_idx][y_threshold_normals], q, method="higher"))

    validation_calibration_metrics = {
        "xgboost": _metrics(y_threshold, xgb_val_scores[threshold_idx], threshold_xgb),
        "autoencoder": _metrics(y_threshold, ae_val_errors[threshold_idx], threshold_ae),
        "hybrid": _metrics(y_threshold, hybrid_val_margin[threshold_idx], threshold_hybrid),
    }

    test_metrics = {
        "XGBoost": _metrics(y_test, xgb_test_scores, threshold_xgb),
        "Autoencoder": _metrics(y_test, ae_test_errors, threshold_ae),
        "Hybrid (XGBoost + autoencoder)": _metrics(y_test, hybrid_test_margin, threshold_hybrid),
    }
    comparison_rows = []
    for name, metrics in test_metrics.items():
        comparison_rows.append({
            "model": name,
            "score_type": {"XGBoost": "raw predict_proba score", "Autoencoder": "reconstruction error", "Hybrid (XGBoost + autoencoder)": "logistic stacking decision margin"}[name],
            **metrics,
        })
    comparison_df = pd.DataFrame(comparison_rows)
    coeffs = {"normalized XGBoost logit": float(hybrid_model.coef_[0][0]), "normalized log1p anomaly error": float(hybrid_model.coef_[0][1])}
    comparison_df.attrs["hybrid_coefficients"] = coeffs
    comparison_df.to_csv(output_dir / "test_model_comparison.csv", index=False)

    # Row-level test scores for investigation and later Phase 10/12 integration.
    test_predictions = pd.DataFrame({
        "Time": test["Time"].to_numpy(),
        "actual_class": y_test,
        "xgboost_score": xgb_test_scores,
        "xgboost_flag_at_validation_fpr_threshold": (xgb_test_scores >= threshold_xgb).astype(int),
        "anomaly_score": ae_test_errors,
        "autoencoder_flag_at_validation_fpr_threshold": (ae_test_errors >= threshold_ae).astype(int),
        "hybrid_margin": hybrid_test_margin,
        "hybrid_score_0_100_not_probability": hybrid_test_display_score,
        "hybrid_flag_at_validation_fpr_threshold": (hybrid_test_margin >= threshold_hybrid).astype(int),
    })
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    # Validation threshold-calibration scores are exported for auditability.
    threshold_validation = pd.DataFrame({
        "Time": val.iloc[threshold_idx]["Time"].to_numpy(),
        "actual_class": y_threshold,
        "xgboost_score": xgb_val_scores[threshold_idx],
        "anomaly_score": ae_val_errors[threshold_idx],
        "hybrid_margin": hybrid_val_margin[threshold_idx],
    })
    threshold_validation.to_csv(output_dir / "threshold_calibration_validation.csv", index=False)

    # Test complements: does the anomaly signal catch fraud the XGB flag misses?
    xgb_flag = test_predictions["xgboost_flag_at_validation_fpr_threshold"].to_numpy(dtype=int).astype(bool)
    ae_flag = test_predictions["autoencoder_flag_at_validation_fpr_threshold"].to_numpy(dtype=int).astype(bool)
    hy_flag = test_predictions["hybrid_flag_at_validation_fpr_threshold"].to_numpy(dtype=int).astype(bool)
    fraud_mask = y_test == 1
    complementarity = {
        "test_fraud_cases": int(fraud_mask.sum()),
        "xgboost_caught_fraud": int((xgb_flag & fraud_mask).sum()),
        "autoencoder_caught_fraud": int((ae_flag & fraud_mask).sum()),
        "hybrid_caught_fraud": int((hy_flag & fraud_mask).sum()),
        "fraud_caught_by_autoencoder_but_missed_by_xgboost": int((ae_flag & ~xgb_flag & fraud_mask).sum()),
        "fraud_caught_by_xgboost_but_missed_by_autoencoder": int((xgb_flag & ~ae_flag & fraud_mask).sum()),
        "fraud_caught_by_hybrid_but_missed_by_xgboost": int((hy_flag & ~xgb_flag & fraud_mask).sum()),
        "false_positives_xgboost": int((xgb_flag & (y_test == 0)).sum()),
        "false_positives_autoencoder": int((ae_flag & (y_test == 0)).sum()),
        "false_positives_hybrid": int((hy_flag & (y_test == 0)).sum()),
    }

    # Save models and preprocessing, without overwriting Phase 5/8 artifacts.
    models_dir = project_root / "models"
    xgb_model.save_model(models_dir / "hybrid_xgboost_train_only.json")
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in ae_model.state_dict().items()},
        "input_dim": len(AE_FEATURE_COLUMNS), "latent_dim": ae_latent_dim,
        "feature_columns": AE_FEATURE_COLUMNS,
        "selection_method": "minimum legitimate-validation reconstruction loss",
        "seed": seed,
    }, models_dir / "hybrid_autoencoder_state.pt")
    joblib.dump({"scaler": ae_scaler, "feature_columns": AE_FEATURE_COLUMNS, "amount_transform": "log1p(Amount)", "time_excluded": True}, models_dir / "hybrid_autoencoder_preprocessor.joblib")
    joblib.dump({
        "signal_transforms": {"xgboost": "logit(clip(p,1e-6,1-1e-6))", "autoencoder": "log1p(reconstruction_error)"},
        "signal_scaler": score_scaler,
        "meta_model": hybrid_model,
        "meta_training_period": {"validation_rows": int(len(meta_train_idx)), "fraud_rows": int(y_meta_train.sum()), "time_min": float(val.iloc[meta_train_idx]["Time"].min()), "time_max": float(val.iloc[meta_train_idx]["Time"].max())},
        "threshold_calibration_period": {"validation_rows": int(len(threshold_idx)), "fraud_rows": int(y_threshold.sum()), "time_min": float(val.iloc[threshold_idx]["Time"].min()), "time_max": float(val.iloc[threshold_idx]["Time"].max())},
        "target_validation_fpr": target_fpr,
        "thresholds": {"xgboost": threshold_xgb, "autoencoder": threshold_ae, "hybrid_margin": threshold_hybrid},
        "hybrid_display_score_definition": "100 * sigmoid(decision_function); display score only, not a calibrated fraud probability",
        "class_weight": "balanced",
        "random_seed": seed,
    }, models_dir / "hybrid_stack.joblib")

    history_df = pd.DataFrame(ae_history)
    history_df.to_csv(output_dir / "hybrid_autoencoder_training_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(history_df["epoch"], history_df["train_loss_normal_only"], label="Train MSE (legitimate only)")
    ax.plot(history_df["epoch"], history_df["validation_loss_legitimate_only"], label="Validation MSE (legitimate only)")
    ax.set(title="Phase 9 autoencoder training", xlabel="Epoch", ylabel="Mean squared reconstruction error")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "hybrid_autoencoder_training_loss.png", dpi=160); plt.close(fig)
    _save_plots(comparison_df, test_predictions, output_dir)
    pd.DataFrame([{"signal": k, "coefficient": v} for k, v in coeffs.items()]).to_csv(output_dir / "hybrid_coefficients.csv", index=False)

    summary = {
        "phase": 9,
        "project": "FraudShield AI",
        "protocol": "Chronological 60/20/20; base models fit on train only; hybrid trained on first 70% of validation; thresholds calibrated on final 30% of validation legitimate scores; test evaluated after all choices.",
        "data_split": splits.cutoffs,
        "partition_counts": {
            "train": {"rows": int(len(train)), "fraud": int(y_train.sum())},
            "validation": {"rows": int(len(val)), "fraud": int(y_val.sum())},
            "hybrid_meta_train": {"rows": int(len(meta_train_idx)), "fraud": int(y_meta_train.sum())},
            "threshold_calibration": {"rows": int(len(threshold_idx)), "fraud": int(y_threshold.sum())},
            "test": {"rows": int(len(test)), "fraud": int(y_test.sum())},
        },
        "base_models": {
            "xgboost": {"fit_rows": int(len(train)), "selected_hyperparameters_source": "Phase 5 validation-selected configuration", "scale_pos_weight": selected_params.get("scale_pos_weight")},
            "autoencoder": {**ae_info, "fit_seconds": round(ae_fit_seconds, 3), "features": AE_FEATURE_COLUMNS},
        },
        "normalization": {
            "xgboost_transform": "logit(clipped predict_proba score)",
            "autoencoder_transform": "log1p(reconstruction error)",
            "normalizer": "StandardScaler fitted on hybrid meta-training rows only",
            "hybrid_model": "regularized LogisticRegression(C=0.1, class_weight=balanced)",
            "output_note": "Hybrid decision margin and its 0-100 sigmoid display are uncalibrated ranking scores, not verified fraud probabilities.",
        },
        "threshold_policy": {"target_fpr": target_fpr, "method": f"{1-target_fpr:.1%} quantile of each score on legitimate rows in the later validation calibration period", "thresholds": {"xgboost": threshold_xgb, "autoencoder": threshold_ae, "hybrid_margin": threshold_hybrid}},
        "validation_calibration_metrics": validation_calibration_metrics,
        "test_metrics": test_metrics,
        "hybrid_coefficients": coeffs,
        "test_complementarity": complementarity,
        "caveats": [
            "The meta-learner is trained using validation labels, so hybrid validation metrics are not presented as unbiased model-performance estimates.",
            "Autoencoder checkpoint selection in this phase uses legitimate-only validation reconstruction loss and does not inspect validation fraud labels.",
            "Only 57 fraud cases exist in the full validation partition; model and threshold estimates are uncertain.",
            "The hybrid score is not probability-calibrated; no independent calibration block with adequate fraud cases is available.",
            "The historical test period has been inspected in previous phases; a fresh later-time holdout is recommended for a final independent claim.",
            "This dataset is anonymized and historical; this is a technical portfolio prototype, not a production bank decision system.",
        ],
        "artifacts": {
            "test_metrics_csv": "reports/hybrid/test_model_comparison.csv",
            "test_predictions_csv": "reports/hybrid/test_predictions.csv",
            "threshold_calibration_csv": "reports/hybrid/threshold_calibration_validation.csv",
            "comparison_chart": "reports/hybrid/test_model_comparison.png",
            "precision_recall_chart": "reports/hybrid/test_precision_recall_curves.png",
            "hybrid_stack_model": "models/hybrid_stack.joblib",
            "xgboost_model": "models/hybrid_xgboost_train_only.json",
            "autoencoder_model": "models/hybrid_autoencoder_state.pt",
        },
    }
    (output_dir / "hybrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary, output_dir / "PHASE9_HYBRID_FINDINGS.md")
    return summary


def _write_report(summary: dict[str, Any], path: Path) -> None:
    rows = summary["test_metrics"]
    lines = [
        "# Phase 9 — Hybrid ML System Findings", "",
        "## Protocol", "", summary["protocol"], "",
        "The base XGBoost and autoencoder are fit using the chronological training block. The first 70% of validation trains the hybrid stacker; the later 30% of validation sets the model-specific threshold using legitimate rows only. The test partition is not used for fitting or threshold selection.", "",
        "## Test comparison", "",
        "| Model | Average Precision | ROC-AUC | Precision | Recall | F1 | FP | FN | Threshold |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['average_precision']:.4f} | {r['roc_auc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['false_positive']} | {r['false_negative']} | {r['threshold']:.6g} |")
    lines += ["", "## Threshold policy", "", f"All three score thresholds target approximately {summary['threshold_policy']['target_fpr']:.1%} false-positive rate on legitimate rows in the later validation calibration period. The actual test false-positive rate can differ under distribution shift.", "", "## Normalization and calibration", "", "- XGBoost score: clipped logit transform.", "- Autoencoder reconstruction error: `log1p` transform.", "- Both transformed signals are standardized using a scaler fitted only on the hybrid meta-training period.", "- The hybrid is regularized logistic stacking. Its decision margin and sigmoid-transformed 0–100 display score are ranking scores, not verified fraud probabilities.", "- A separate probability-calibration model is not fit because the available validation sample contains only 57 frauds, and the last validation block contains even fewer. Use a dedicated, later calibration period and reliability/Brier evaluation before making probabilistic claims.", "", "## Does the autoencoder add value?", "", f"Among the {summary['test_complementarity']['test_fraud_cases']} test fraud cases, XGBoost catches {summary['test_complementarity']['xgboost_caught_fraud']}, the autoencoder catches {summary['test_complementarity']['autoencoder_caught_fraud']}, and the hybrid catches {summary['test_complementarity']['hybrid_caught_fraud']} at their validation-calibrated thresholds.", f"The autoencoder flags {summary['test_complementarity']['fraud_caught_by_autoencoder_but_missed_by_xgboost']} fraud case(s) missed by XGBoost; the hybrid flags {summary['test_complementarity']['fraud_caught_by_hybrid_but_missed_by_xgboost']} fraud case(s) missed by XGBoost under this policy.", "", "## Caveats", ""]
    lines.extend([f"- {item}" for item in summary["caveats"]])
    lines += ["", "## Re-run", "", "```powershell", "python -m src.models.hybrid_system", "python -m pytest -q", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FraudShield AI Phase 9 hybrid experiment")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    parser.add_argument("--max-normal-rows", type=int, default=60000)
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-patience", type=int, default=6)
    parser.add_argument("--ae-latent-dim", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = run_hybrid_experiment(
        data_path=args.data_path, output_dir=args.output_dir, seed=args.seed,
        target_fpr=args.target_fpr, max_normal_rows=args.max_normal_rows,
        ae_epochs=args.ae_epochs, ae_batch_size=args.ae_batch_size,
        ae_patience=args.ae_patience, ae_latent_dim=args.ae_latent_dim,
        n_jobs=args.n_jobs, device_name=args.device,
    )
    print("Phase 9 hybrid experiment complete.")
    print(json.dumps({"test_metrics": summary["test_metrics"], "hybrid_coefficients": summary["hybrid_coefficients"], "test_complementarity": summary["test_complementarity"]}, indent=2))


if __name__ == "__main__":
    main()
