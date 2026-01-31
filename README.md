# Time-series Defect Detection & Explainability

---

## Project overview ✅
This project demonstrates a complete pipeline for multi-label defect detection on synthetic multivariate time-series data and provides per-defect explainability. It uses an LSTM-based classifier for detection and an LSTM autoencoder to identify and localize anomalous segments responsible for defects.

Key outcomes:
- High classification performance (binarized accuracy reported ~**99.832%**)
- Per-anomaly explainability that maps individual anomaly clusters to defect labels

---

## Files
- `ProjectData.ipynb` — Main notebook with data generation, model training/evaluation, and visualizations.
- `RNN_model.pt` — Saved classifier state (used by notebook when `retrain=False`).
- `lstm_autoencoder.pt` — Saved autoencoder state (used by notebook when `retrain_autoencoder=False`).

---

## Dataset & generation 💡
- Data is synthetic and generated in the notebook via the `createRow` function.
- Each sample is a multivariate time series (3 features), with 5 potential defect types injected probabilistically.
- Dataset size used for experiments: `n = 50000` (adjustable in notebook).

---

## Models & approach 🔧
- Classifier: `RNNClassifier` — two LSTM layers (64 → 32) + fully connected output with sigmoid nodes for multi-label classification. Optimized with Adam and `BCELoss`.
- Explainability: `LSTMAutoencoder` trained only on no-defect samples. Reconstruction error (MSE) is used to detect anomalies; thresholding and clustering isolate anomaly segments. The classifier is then used on series with single isolated anomaly clusters to determine which defect they represent.

---

## How to run (quick start) ▶️
1. Open `ProjectData.ipynb` in Jupyter or VS Code.
2. Install dependencies (example):

```bash
python -m pip install torch numpy pandas matplotlib scikit-learn pillow
```

3. Run cells from top to bottom. If you want to retrain models:
- Set `retrain = True` to train the classifier.
- Set `retrain_autoencoder = True` to train the autoencoder (autoencoder uses many epochs to overfit no-defect data).

Notes:
- If you prefer to use the pretrained models, keep `retrain=False` and ensure `RNN_model.pt` and `lstm_autoencoder.pt` are in the notebook directory.
- The notebook automatically handles padding variable-length sequences using `pad_sequence`.

---

## Important notebook functions & variables
- `createRow(...)` — synthetic generator of a single time series
- `RNNClassifier` — classifier model
- `LSTMAutoencoder` — explainability autoencoder
- `visualize_candy_defects(k, n)` — visualize examples of defect `k`
- `visualize_many(idx)` — visualize series with mapped defect labels
- `threshold` — reconstruction error threshold (default used: 0.19)

---

## Results & evaluation 📊
- Binarize predictions with threshold 0.5 to compute classification metrics.
- Example reported: Binarized Accuracy ≈ **0.99832**. Per-class precision & recall printed in the notebook.

---

## Reproducibility & tips
- Set seeds (`np.random.seed(42)` is used) for deterministic generation in examples.
- If you train on GPU make sure `torch.cuda.is_available()` and set device accordingly (handled in notebook).
- Adjust `threshold` if you change the autoencoder architecture or dataset.

---

## Possible extensions
- Replace synthetic generator with real sensor data and retrain.
- Use attention mechanisms for more detailed temporal explainability.
- Automate threshold selection (ROC/validation-based) for anomaly detection.


