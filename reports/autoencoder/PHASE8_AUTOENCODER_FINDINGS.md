# Phase 8 — PyTorch Autoencoder Anomaly Detection Findings

## Method

- Chronological train/validation/test split using the same 60/20/20 strategy as earlier phases.
- Fit preprocessing using 170,528 legitimate training rows, and fit neural weights on a seeded sample of 60,000 of those rows; validation has 56,901 legitimate rows and 57 fraud rows; test has 56,961 rows including 75 fraud cases.
- Features: 29 values (V1–V28 and log1p(Amount)); Time is excluded to avoid the chronological time index dominating anomaly scores.
- StandardScaler is fitted only on legitimate training data.
- Architecture: 29 → 64 → 32 → 8 latent dimensions → 32 → 64 → 29, with ReLU in hidden layers.
- The selected checkpoint is epoch 18 (validation Average Precision 0.0700); best legitimate-validation reconstruction loss recorded for that checkpoint: 0.5510. Device: cpu; training time: 18.4s.
- Threshold: 2.4503756, the 99.0% quantile of reconstruction errors on legitimate validation rows. Labels were not used to set this threshold.

## What reconstruction error means

For each row, the model reconstructs the scaled feature vector. The anomaly score is the mean squared error across the 29 features. Larger errors mean the row was harder for this autoencoder to reconstruct; they do not prove fraud and are not fraud probabilities.

## Validation metrics at the selected threshold

- Precision: 0.0686
- Recall: 0.7368
- F1: 0.1256
- ROC-AUC: 0.9378
- Average Precision: 0.0700
- False positives: 570; false negatives: 15; validation legitimate FPR: 1.002%

## Held-out test metrics at the validation-selected threshold

- Precision: 0.1352
- Recall: 0.7067
- F1: 0.2270
- ROC-AUC: 0.9310
- Average Precision: 0.1579
- Confusion counts: TN=56,547, FP=339, FN=22, TP=53

## Comparison to supervised XGBoost

- Autoencoder Average Precision: 0.1579; XGBoost Average Precision: 0.7872.
- Autoencoder recall at its validation-normal q99% threshold: 0.7067; XGBoost recall at Phase 6 threshold 0.785300: 0.7733.
- Autoencoder false positives: 339; XGBoost false positives: 18.
- The two decision thresholds have different selection rules. Compare score-ranking metrics and confusion counts, and do not claim apples-to-apples threshold optimality.

## Threshold sensitivity (diagnostic only)

| Normal-validation quantile | Threshold | Validation FPR | Validation recall | Test FPR | Test recall | Test FP | Test FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.950 | 1.1752 | 0.0500 | 0.8421 | 0.0364 | 0.8000 | 2068 | 15 |
| 0.975 | 1.54087 | 0.0250 | 0.7895 | 0.0168 | 0.8000 | 953 | 15 |
| 0.990 | 2.45038 | 0.0100 | 0.7368 | 0.0060 | 0.7067 | 339 | 22 |
| 0.995 | 3.88783 | 0.0050 | 0.5614 | 0.0028 | 0.5200 | 160 | 36 |
| 0.999 | 14.0675 | 0.0010 | 0.0175 | 0.0005 | 0.0800 | 28 | 69 |

## Interpretation and limitations

1. The autoencoder's gradient updates use only legitimate training rows. To choose the checkpoint/epoch, the experiment uses validation Average Precision with validation labels; this is label-informed model selection but does not train the model on fraud examples. The primary threshold is separately computed from legitimate validation reconstruction errors only.
2. The threshold is based on legitimate validation reconstruction errors, not the best validation F1 and not test labels. Because data distributions change over time, test false-positive rate may differ from the target validation rate.
3. Excluding Time is intentional: the later chronological periods naturally have larger Time values, which can become a spurious anomaly signal.
4. Standardization and log1p(Amount) are learned/applied inside the training pipeline; scaler statistics come only from legitimate training rows.
5. A high reconstruction error means unusual relative to this trained reconstruction model—not necessarily fraudulent. A fraud transaction may reconstruct well, and a legitimate outlier may reconstruct poorly.
6. The public dataset is anonymized and historical. This is a technical prototype, not a production banking fraud detector.

## Re-run

```powershell
python -m pip install -r requirements.txt
python -m src.models.autoencoder_anomaly
python -m pytest -q
```
