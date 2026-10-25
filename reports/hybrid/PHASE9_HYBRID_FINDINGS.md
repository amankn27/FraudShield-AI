# Phase 9 — Hybrid ML System Findings

## Protocol

Chronological 60/20/20; base models fit on train only; hybrid trained on first 70% of validation; thresholds calibrated on final 30% of validation legitimate scores; test evaluated after all choices.

The base XGBoost and autoencoder are fit using the chronological training block. The first 70% of validation trains the hybrid stacker; the later 30% of validation sets the model-specific threshold using legitimate rows only. The test partition is not used for fitting or threshold selection.

## Test comparison

| Model | Average Precision | ROC-AUC | Precision | Recall | F1 | FP | FN | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.7867 | 0.9855 | 0.1400 | 0.8533 | 0.2406 | 393 | 11 | 0.298779 |
| Autoencoder | 0.0845 | 0.9216 | 0.0954 | 0.4933 | 0.1598 | 351 | 38 | 2.14366 |
| Hybrid (XGBoost + autoencoder) | 0.7371 | 0.9851 | 0.1477 | 0.8667 | 0.2524 | 375 | 10 | 1.81059 |

## Threshold policy

All three score thresholds target approximately 1.0% false-positive rate on legitimate rows in the later validation calibration period. The actual test false-positive rate can differ under distribution shift.

## Normalization and calibration

- XGBoost score: clipped logit transform.
- Autoencoder reconstruction error: `log1p` transform.
- Both transformed signals are standardized using a scaler fitted only on the hybrid meta-training period.
- The hybrid is regularized logistic stacking. Its decision margin and sigmoid-transformed 0–100 display score are ranking scores, not verified fraud probabilities.
- A separate probability-calibration model is not fit because the available validation sample contains only 57 frauds, and the last validation block contains even fewer. Use a dedicated, later calibration period and reliability/Brier evaluation before making probabilistic claims.

## Does the autoencoder add value?

Among the 75 test fraud cases, XGBoost catches 64, the autoencoder catches 37, and the hybrid catches 65 at their validation-calibrated thresholds.
The autoencoder flags 1 fraud case(s) missed by XGBoost; the hybrid flags 1 fraud case(s) missed by XGBoost under this policy.

## Caveats

- The meta-learner is trained using validation labels, so hybrid validation metrics are not presented as unbiased model-performance estimates.
- Autoencoder checkpoint selection in this phase uses legitimate-only validation reconstruction loss and does not inspect validation fraud labels.
- Only 57 fraud cases exist in the full validation partition; model and threshold estimates are uncertain.
- The hybrid score is not probability-calibrated; no independent calibration block with adequate fraud cases is available.
- The historical test period has been inspected in previous phases; a fresh later-time holdout is recommended for a final independent claim.
- This dataset is anonymized and historical; this is a technical portfolio prototype, not a production bank decision system.

## Re-run

```powershell
python -m src.models.hybrid_system
python -m pytest -q
```
