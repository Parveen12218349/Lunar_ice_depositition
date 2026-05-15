# 🌙 Lunar Ice Deposit Detection

A Computer Vision pipeline that detects craters in lunar satellite imagery and scores each one for ice-deposit likelihood using a multi-signal approach.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## Features

- **Autonomous crater detection** via Circular Hough Transform — no annotations needed
- **59-feature vector** per crater: intensity stats, LBP texture, GLCM texture, edge density, shadow ratio, rim/floor contrast, Laplacian variance, histogram entropy
- **Multi-criteria labeling**: brightness + smoothness + shadow presence + rim/floor ratio
- **Ensemble classifier**: RandomForest + XGBoost + LightGBM soft-voting
- **Data augmentation**: flips, gamma jitter, Gaussian noise (6× training data)
- **Confidence tiers**: HIGH / MODERATE / LOW / UNLIKELY with probabilities
- **8-panel visualization**: original, CLAHE, Otsu, adaptive, heatmap, LBP, bright mask, sharpened
- **Streamlit web app**: upload any image, get instant results + downloadable CSV

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python ice_deposit_detection.py
```

### Launch the web app
```bash
streamlit run app.py
```

---

## Project Structure

```
├── ice_deposit_detection.py   # Core CV pipeline (importable)
├── app.py                     # Streamlit web application
├── requirements.txt           # Pinned dependencies
├── train/                     # Training images + _annotations.csv
├── valid/                     # Validation images + _annotations.csv
├── output/                    # Generated visualizations + model
│   ├── ice_detector_model.pkl
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── hough_demo.png
│   └── *_pipeline.png / *_density.png
└── .streamlit/
    └── config.toml            # Dark theme config
```

---

## Pipeline

```
Input Image
    ↓
Preprocessing
  ├── Grayscale conversion
  ├── CLAHE histogram equalization (clipLimit=2.5)
  ├── Gaussian blur (5×5, σ=1.0)
  ├── Canny edge detection
  └── Unsharp mask sharpening
    ↓
Segmentation
  ├── Otsu global thresholding
  ├── Adaptive Gaussian thresholding
  ├── High-reflectance mask (>200 intensity)
  └── Morphological cleanup + connected components
    ↓
Crater Detection
  └── Circular Hough Transform (autonomous)
    ↓
Feature Extraction (59 features per crater)
  ├── Intensity stats × 2 channels (12)
  ├── LBP histogram (26)
  ├── GLCM texture (10)
  ├── Edge density, bright ratio, shadow ratio (3)
  ├── Rim/floor ratio, roughness (2)
  ├── Shape features (2)
  └── Laplacian variance, entropy, percentile spread (4)
    ↓
Multi-Criteria Labeling
  ├── Brightness > 130 (high albedo)
  ├── Edge density < 0.08 (smooth surface)
  ├── Shadow ratio > 0.15 (cold-trap proxy)
  └── Rim/floor ratio > 1.25 (classic cold-trap signature)
  → Ice if ≥ 2 criteria met
    ↓
Ensemble Classification
  ├── RandomForest (300 trees)
  ├── XGBoost (300 estimators)
  └── LightGBM (300 estimators)
  → Soft-voting probability
    ↓
Confidence Tiers
  ├── HIGH     ≥ 75%
  ├── MODERATE ≥ 50%
  ├── LOW      ≥ 30%
  └── UNLIKELY  < 30%
    ↓
Visualization + Export
```

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Deploy — free, shareable link

> **Note:** The trained model (`output/ice_detector_model.pkl`) should be committed to the repo or the app will auto-train on first launch (~2 min).

---

## Tech Stack

| Library | Role |
|---|---|
| OpenCV | Image loading, CLAHE, blur, Canny, Hough, morphology |
| Scikit-image | LBP texture, GLCM texture |
| Scikit-learn | RandomForest, StandardScaler, cross-validation |
| XGBoost | Gradient boosting classifier |
| LightGBM | Gradient boosting classifier |
| NumPy / SciPy | Numerical operations |
| Pandas | Annotation CSV handling |
| Matplotlib | All static visualizations |
| Streamlit | Web application |
| Joblib | Model serialization |

---

## Results

| Metric | Value |
|---|---|
| 5-fold CV AUC | 0.998 ± 0.001 |
| Validation AUC | 0.999 |
| Validation Accuracy | 99% |
| Training samples (with augmentation) | ~20,000 |
| Features per crater | 59 |
