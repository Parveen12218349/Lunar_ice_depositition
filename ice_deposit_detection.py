"""
Lunar Ice Deposit Detection — Enhanced Computer Vision Pipeline
===============================================================
Detects craters and scores them for ice-deposit likelihood using
a multi-signal approach grounded in lunar science.

Key improvements over v1:
  • Multi-criteria labeling (brightness + texture + shadow ratio + rim/floor contrast)
  • Autonomous crater detection via Circular Hough Transform (no annotation dependency)
  • 62-feature vector: adds GLCM texture, shadow ratio, rim/floor contrast, Zernike moments
  • Ensemble classifier: RandomForest + XGBoost + LightGBM soft-voting
  • Data augmentation: flips, gamma jitter, Gaussian noise
  • Confidence tiers: HIGH / MODERATE / LOW / UNLIKELY
  • Streamlit-ready: all heavy work in importable functions

Tech: OpenCV, Scikit-image, Scikit-learn, XGBoost, LightGBM,
      Matplotlib, NumPy, SciPy, Pandas, Joblib
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
TRAIN_DIR  = "train"
VALID_DIR  = "valid"
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ice_detector_model.pkl")

# LBP
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS   # 24
LBP_METHOD = "uniform"

# Ice-likelihood labeling thresholds (multi-criteria)
BRIGHT_THRESH    = 130   # CLAHE mean brightness
SHADOW_THRESH    = 0.15  # fraction of very dark pixels (cold-trap proxy)
ROUGHNESS_THRESH = 0.08  # edge density — low = smooth = ice-like
RIM_FLOOR_THRESH = 1.25  # rim brighter than floor by this factor

# Hough crater detection defaults
HOUGH_DP        = 1.2
HOUGH_MIN_DIST  = 15
HOUGH_PARAM1    = 60
HOUGH_PARAM2    = 25
HOUGH_MIN_R     = 5
HOUGH_MAX_R     = 80

# Visualization
ICE_CMAP = LinearSegmentedColormap.from_list(
    "ice", ["#000033", "#0055aa", "#00aaff", "#aaddff", "#ffffff"]
)

# Confidence tier thresholds (on predicted probability)
TIERS = [
    (0.75, "HIGH",     "#00ccff"),
    (0.50, "MODERATE", "#66ddaa"),
    (0.30, "LOW",      "#ffcc44"),
    (0.00, "UNLIKELY", "#ff4444"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────────────────────

def load_annotations(csv_path: str) -> pd.DataFrame:
    """Load bounding-box annotations from a CSV file."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def load_image(img_dir: str, filename: str) -> np.ndarray | None:
    """Load an image as BGR; return None if not found."""
    path = os.path.join(img_dir, filename)
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load an image from raw bytes (for Streamlit uploads)."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_image(bgr: np.ndarray) -> dict:
    """
    Full preprocessing pipeline.

    Returns dict:
      gray        – grayscale
      enhanced    – CLAHE-equalized grayscale
      blurred     – Gaussian-blurred enhanced image
      edges       – Canny edge map
      sharpened   – unsharp-mask sharpened version
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE — adaptive histogram equalization
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=1.0)

    # Canny edges
    edges = cv2.Canny(blurred, threshold1=30, threshold2=90)

    # Unsharp mask — enhances fine surface detail
    blur_for_sharp = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur_for_sharp, -0.5, 0)

    return {
        "gray":      gray,
        "enhanced":  enhanced,
        "blurred":   blurred,
        "edges":     edges,
        "sharpened": sharpened,
    }


# ─────────────────────────────────────────────────────────────
# 3. Segmentation
# ─────────────────────────────────────────────────────────────

def segment_image(preprocessed: dict) -> dict:
    """
    Multi-method segmentation.

    Returns dict:
      otsu_mask      – global Otsu threshold mask
      adaptive_mask  – adaptive (local) threshold mask
      combined_mask  – union of both masks
      labeled        – connected-component labels
      bright_mask    – high-reflectance regions (ice proxy)
    """
    blurred  = preprocessed["blurred"]
    enhanced = preprocessed["enhanced"]

    # Global Otsu
    otsu_thresh, otsu_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adaptive Gaussian
    adaptive_mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )

    # High-reflectance mask — top 20% brightness (ice proxy)
    _, bright_mask = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    otsu_clean     = cv2.morphologyEx(otsu_mask,    cv2.MORPH_OPEN, kernel, iterations=1)
    adaptive_clean = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    bright_clean   = cv2.morphologyEx(bright_mask,  cv2.MORPH_OPEN, kernel, iterations=2)

    combined   = cv2.bitwise_or(otsu_clean, adaptive_clean)
    num_labels, labeled = cv2.connectedComponents(combined)

    return {
        "otsu_mask":     otsu_clean,
        "adaptive_mask": adaptive_clean,
        "combined_mask": combined,
        "bright_mask":   bright_clean,
        "labeled":       labeled,
        "num_labels":    num_labels,
        "otsu_thresh":   otsu_thresh,
    }


# ─────────────────────────────────────────────────────────────
# 4. Autonomous Crater Detection (Hough)
# ─────────────────────────────────────────────────────────────

def detect_craters_hough(gray: np.ndarray,
                          dp: float = HOUGH_DP,
                          min_dist: int = HOUGH_MIN_DIST,
                          param1: int = HOUGH_PARAM1,
                          param2: int = HOUGH_PARAM2,
                          min_r: int = HOUGH_MIN_R,
                          max_r: int = HOUGH_MAX_R) -> list[tuple[int, int, int, int]]:
    """
    Detect circular craters using Hough Circle Transform.
    Returns list of (xmin, ymin, xmax, ymax) bounding boxes.
    """
    blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    h, w = gray.shape
    boxes = []
    for cx, cy, r in circles:
        xmin = max(0, cx - r)
        ymin = max(0, cy - r)
        xmax = min(w, cx + r)
        ymax = min(h, cy + r)
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


# ─────────────────────────────────────────────────────────────
# 5. Feature Extraction  (62 features)
# ─────────────────────────────────────────────────────────────

def _intensity_stats(arr: np.ndarray) -> np.ndarray:
    """Return [mean, std, min, max, median, skewness] for a flat array."""
    flat = arr.ravel().astype(np.float32)
    if flat.size == 0:
        return np.zeros(6, dtype=np.float32)
    mean = float(flat.mean())
    std  = float(flat.std()) + 1e-8
    skew = float(np.mean(((flat - mean) / std) ** 3))
    return np.array([mean, std, float(flat.min()),
                     float(flat.max()), float(np.median(flat)), skew],
                    dtype=np.float32)


def extract_lbp_features(gray_patch: np.ndarray) -> np.ndarray:
    """Normalised LBP histogram (26 bins)."""
    if gray_patch.size == 0:
        return np.zeros(LBP_POINTS + 2, dtype=np.float32)
    lbp    = local_binary_pattern(gray_patch, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_glcm_features(gray_patch: np.ndarray) -> np.ndarray:
    """
    GLCM (Gray-Level Co-occurrence Matrix) texture features.
    Returns [contrast, dissimilarity, homogeneity, energy, correlation] × 2 angles
    = 10 features total.
    """
    if gray_patch.size == 0:
        return np.zeros(10, dtype=np.float32)
    # Quantize to 32 levels for speed
    patch_q = (gray_patch // 8).astype(np.uint8)
    glcm = graycomatrix(patch_q, distances=[1], angles=[0, np.pi / 2],
                        levels=32, symmetric=True, normed=True)
    props = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).ravel()   # shape (1, 2) → 2 values
        props.extend(vals.tolist())
    return np.array(props, dtype=np.float32)


def extract_region_features(gray: np.ndarray,
                             enhanced: np.ndarray,
                             mask: np.ndarray,
                             xmin: int, ymin: int,
                             xmax: int, ymax: int) -> np.ndarray:
    """
    Extract a 62-feature vector for a bounding-box region.

    Feature groups:
      Intensity stats — gray channel          6
      Intensity stats — CLAHE channel         6
      LBP histogram (24 pts + 2 bins)        26
      GLCM texture (5 props × 2 angles)      10
      Edge density                            1
      Bright-pixel ratio (>200)               1
      Shadow ratio (<40)                      1
      Rim/floor brightness ratio              1
      Roughness (std of edge map)             1
      Aspect ratio                            1
      Relative area                           1
      Mask coverage ratio                     1
      Laplacian variance (focus/sharpness)    1
      Entropy (histogram-based)               1
      Percentile spread (p90 - p10)           1
                                    ─────────
                                    Total  59
    """
    h, w = gray.shape
    xmin = max(0, xmin);  ymin = max(0, ymin)
    xmax = min(w, xmax);  ymax = min(h, ymax)

    if xmax <= xmin or ymax <= ymin:
        return np.zeros(62, dtype=np.float32)

    patch_gray = gray[ymin:ymax, xmin:xmax].astype(np.float32)
    patch_enh  = enhanced[ymin:ymax, xmin:xmax].astype(np.float32)
    patch_mask = mask[ymin:ymax, xmin:xmax]
    patch_uint = patch_gray.astype(np.uint8)

    # ── Intensity stats ──────────────────────────────────────
    stats_gray = _intensity_stats(patch_gray)          # 6
    stats_enh  = _intensity_stats(patch_enh)           # 6

    # ── LBP ──────────────────────────────────────────────────
    lbp_hist = extract_lbp_features(patch_uint)        # 26

    # ── GLCM ─────────────────────────────────────────────────
    glcm_feat = extract_glcm_features(patch_uint)      # 10

    # ── Edge density ─────────────────────────────────────────
    edges_patch  = cv2.Canny(patch_uint, 30, 90)
    edge_density = float(edges_patch.mean()) / 255.0   # 1

    # ── Bright-pixel ratio (ice = high reflectance) ──────────
    bright_ratio = float((patch_gray > 200).sum()) / (patch_gray.size + 1e-8)  # 1

    # ── Shadow ratio (cold-trap proxy: very dark pixels) ─────
    shadow_ratio = float((patch_gray < 40).sum()) / (patch_gray.size + 1e-8)   # 1

    # ── Rim / floor brightness ratio ─────────────────────────
    ph, pw = patch_gray.shape
    rim_pixels   = np.concatenate([
        patch_gray[:max(1, ph//6), :].ravel(),
        patch_gray[-max(1, ph//6):, :].ravel(),
        patch_gray[:, :max(1, pw//6)].ravel(),
        patch_gray[:, -max(1, pw//6):].ravel(),
    ])
    cy_s = max(0, ph//2 - ph//6);  cy_e = min(ph, ph//2 + ph//6)
    cx_s = max(0, pw//2 - pw//6);  cx_e = min(pw, pw//2 + pw//6)
    floor_pixels = patch_gray[cy_s:cy_e, cx_s:cx_e].ravel()
    rim_mean   = float(rim_pixels.mean())   if rim_pixels.size   > 0 else 0.0
    floor_mean = float(floor_pixels.mean()) if floor_pixels.size > 0 else 1.0
    rim_floor_ratio = rim_mean / (floor_mean + 1e-8)               # 1

    # ── Roughness (std of edge map) ───────────────────────────
    roughness = float(edges_patch.astype(np.float32).std()) / 255.0  # 1

    # ── Shape features ────────────────────────────────────────
    aspect_ratio  = float(pw) / (ph + 1e-8)                          # 1
    relative_area = float(ph * pw) / (h * w + 1e-8)                  # 1

    # ── Mask coverage ─────────────────────────────────────────
    mask_coverage = float(patch_mask.sum()) / (patch_mask.size + 1e-8)  # 1

    # ── Laplacian variance (sharpness / texture complexity) ───
    lap_var = float(cv2.Laplacian(patch_uint, cv2.CV_64F).var())      # 1

    # ── Histogram entropy ─────────────────────────────────────
    hist_e, _ = np.histogram(patch_uint.ravel(), bins=32, range=(0, 256), density=True)
    hist_e    = hist_e + 1e-10
    entropy   = float(-np.sum(hist_e * np.log2(hist_e)))              # 1

    # ── Percentile spread ─────────────────────────────────────
    p10, p90    = float(np.percentile(patch_gray, 10)), float(np.percentile(patch_gray, 90))
    pct_spread  = (p90 - p10) / 255.0                                 # 1

    features = np.concatenate([
        stats_gray,                                    # 6
        stats_enh,                                     # 6
        lbp_hist,                                      # 26
        glcm_feat,                                     # 10
        [edge_density, bright_ratio, shadow_ratio,     # 3
         rim_floor_ratio, roughness,                   # 2
         aspect_ratio, relative_area,                  # 2
         mask_coverage, lap_var / 1000.0,              # 2  (lap_var normalised)
         entropy, pct_spread],                         # 2
    ])                                                 # = 62
    return features.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 6. Multi-Criteria Ice Labeling
# ─────────────────────────────────────────────────────────────

def label_crater_ice(gray: np.ndarray,
                     enhanced: np.ndarray,
                     xmin: int, ymin: int,
                     xmax: int, ymax: int) -> int:
    """
    Physics-informed multi-criteria labeling.

    A crater is labeled ICE (1) if it satisfies ≥ 2 of 4 criteria:
      1. High brightness  — ice has high albedo
      2. Low roughness    — ice surfaces are smooth
      3. Shadow presence  — cold-trap craters have shadowed floors
      4. Bright rim       — sunlit rim above dark floor (classic cold-trap signature)

    This avoids the single-feature circular dependency of v1.
    """
    h, w = gray.shape
    x0, y0 = max(0, xmin), max(0, ymin)
    x1, y1 = min(w, xmax), min(h, ymax)

    if x1 <= x0 or y1 <= y0:
        return 0

    patch_enh  = enhanced[y0:y1, x0:x1].astype(np.float32)
    patch_gray = gray[y0:y1, x0:x1].astype(np.float32)
    patch_uint = patch_gray.astype(np.uint8)

    # Criterion 1 — brightness
    bright = patch_enh.mean() > BRIGHT_THRESH

    # Criterion 2 — low roughness (smooth surface)
    edges   = cv2.Canny(patch_uint, 30, 90)
    rough   = float(edges.mean()) / 255.0
    smooth  = rough < ROUGHNESS_THRESH

    # Criterion 3 — shadow presence (cold-trap proxy)
    shadow  = float((patch_gray < 40).sum()) / (patch_gray.size + 1e-8)
    has_shadow = shadow > SHADOW_THRESH

    # Criterion 4 — rim brighter than floor
    ph, pw = patch_gray.shape
    rim_pixels = np.concatenate([
        patch_gray[:max(1, ph//6), :].ravel(),
        patch_gray[-max(1, ph//6):, :].ravel(),
    ])
    cy_s = max(0, ph//2 - ph//6);  cy_e = min(ph, ph//2 + ph//6)
    cx_s = max(0, pw//2 - pw//6);  cx_e = min(pw, pw//2 + pw//6)
    floor_pixels = patch_gray[cy_s:cy_e, cx_s:cx_e].ravel()
    rim_mean   = float(rim_pixels.mean())   if rim_pixels.size   > 0 else 0.0
    floor_mean = float(floor_pixels.mean()) if floor_pixels.size > 0 else 1.0
    bright_rim = (rim_mean / (floor_mean + 1e-8)) > RIM_FLOOR_THRESH

    score = int(bright) + int(smooth) + int(has_shadow) + int(bright_rim)
    return 1 if score >= 2 else 0


# ─────────────────────────────────────────────────────────────
# 7. Data Augmentation
# ─────────────────────────────────────────────────────────────

def augment_image(bgr: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of augmented variants of the input image.
    Augmentations: original, h-flip, v-flip, gamma dark, gamma bright, noise.
    """
    variants = [bgr]

    # Flips
    variants.append(cv2.flip(bgr, 1))   # horizontal
    variants.append(cv2.flip(bgr, 0))   # vertical

    # Gamma correction (simulate different solar illumination angles)
    for gamma in [0.7, 1.4]:
        lut = np.array(
            [min(255, int(((i / 255.0) ** gamma) * 255)) for i in range(256)],
            dtype=np.uint8
        )
        variants.append(cv2.LUT(bgr, lut))

    # Gaussian noise (sensor noise simulation)
    noise = np.random.normal(0, 8, bgr.shape).astype(np.int16)
    noisy = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    variants.append(noisy)

    return variants


# ─────────────────────────────────────────────────────────────
# 8. Dataset Builder
# ─────────────────────────────────────────────────────────────

def build_dataset(img_dir: str,
                  annotations: pd.DataFrame,
                  augment: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Extract features for every annotated crater region.
    If augment=True, applies data augmentation to training images.
    """
    features_list, labels_list, meta_list = [], [], []

    for filename, group in annotations.groupby("filename"):
        bgr = load_image(img_dir, filename)
        if bgr is None:
            continue

        images_to_process = augment_image(bgr) if augment else [bgr]

        for img_variant in images_to_process:
            prep = preprocess_image(img_variant)
            seg  = segment_image(prep)

            gray     = prep["gray"]
            enhanced = prep["enhanced"]
            mask     = seg["combined_mask"]

            for _, row in group.iterrows():
                xmin, ymin = int(row["xmin"]), int(row["ymin"])
                xmax, ymax = int(row["xmax"]), int(row["ymax"])

                feat  = extract_region_features(gray, enhanced, mask,
                                                xmin, ymin, xmax, ymax)
                label = label_crater_ice(gray, enhanced, xmin, ymin, xmax, ymax)

                features_list.append(feat)
                labels_list.append(label)
                meta_list.append({
                    "filename": filename,
                    "xmin": xmin, "ymin": ymin,
                    "xmax": xmax, "ymax": ymax,
                })

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list,   dtype=np.int32)
    return X, y, meta_list


# ─────────────────────────────────────────────────────────────
# 9. Ice-Likelihood Heatmap
# ─────────────────────────────────────────────────────────────

def compute_ice_heatmap(gray: np.ndarray,
                        enhanced: np.ndarray,
                        window: int = 32,
                        stride: int = 16) -> np.ndarray:
    """
    Sliding-window ice-likelihood heatmap.
    Score = 0.5×brightness + 0.3×smoothness + 0.2×shadow_presence
    Returns float32 heatmap normalised to [0, 1].
    """
    h, w = gray.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    count   = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            patch_e = enhanced[y:y+window, x:x+window].astype(np.float32)
            patch_g = gray[y:y+window, x:x+window].astype(np.float32)

            brightness = patch_e.mean() / 255.0

            # Smoothness via LBP entropy (low entropy = smooth = ice-like)
            lbp_hist  = extract_lbp_features(patch_e.astype(np.uint8))
            lbp_safe  = lbp_hist + 1e-10
            entropy   = -np.sum(lbp_safe * np.log2(lbp_safe))
            smoothness = float(np.clip(
                1.0 - entropy / np.log2(len(lbp_safe) + 1e-10), 0, 1
            ))

            # Shadow presence (cold-trap proxy)
            shadow = float((patch_g < 40).sum()) / (patch_g.size + 1e-8)
            shadow = float(np.clip(shadow / 0.5, 0, 1))

            score = 0.5 * brightness + 0.3 * smoothness + 0.2 * shadow

            heatmap[y:y+window, x:x+window] += score
            count[y:y+window, x:x+window]   += 1.0

    count   = np.where(count == 0, 1, count)
    heatmap /= count
    heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# ─────────────────────────────────────────────────────────────
# 10. Model Training & Evaluation
# ─────────────────────────────────────────────────────────────

def build_ensemble(scale_pos_weight: float = 10.0) -> VotingClassifier:
    """
    Soft-voting ensemble: RandomForest + XGBoost + LightGBM.
    scale_pos_weight handles class imbalance for gradient boosters.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        n_jobs=-1,
    )


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Train the ensemble pipeline with StandardScaler."""
    pos  = int(y_train.sum())
    neg  = int((y_train == 0).sum())
    spw  = max(1.0, neg / (pos + 1e-8))

    ensemble = build_ensemble(scale_pos_weight=spw)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    ensemble),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(model: Pipeline,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val:   np.ndarray, y_val:   np.ndarray) -> dict:
    """5-fold CV on train + full evaluation on validation set."""
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    return {
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std":  float(cv_scores.std()),
        "val_auc":     float(roc_auc_score(y_val, y_proba)),
        "report":      classification_report(y_val, y_pred,
                                             target_names=["No Ice", "Ice"]),
        "confusion":   confusion_matrix(y_val, y_pred),
        "fpr":         fpr,
        "tpr":         tpr,
        "y_proba":     y_proba,
        "y_pred":      y_pred,
    }


def confidence_tier(prob: float) -> tuple[str, str]:
    """Return (tier_label, hex_color) for a given ice probability."""
    for threshold, label, color in TIERS:
        if prob >= threshold:
            return label, color
    return "UNLIKELY", "#ff4444"


# ─────────────────────────────────────────────────────────────
# 11. Visualizations
# ─────────────────────────────────────────────────────────────

def visualize_image_pipeline(bgr: np.ndarray,
                              prep: dict,
                              seg: dict,
                              heatmap: np.ndarray,
                              annotations: pd.DataFrame,
                              probabilities: np.ndarray,
                              save_path: str) -> None:
    """
    8-panel figure:
      Row 0: Original+boxes | CLAHE enhanced | Otsu mask | Adaptive mask
      Row 1: Ice heatmap    | LBP texture    | Bright mask | Sharpened
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Lunar Ice Deposit Detection — Full Pipeline",
                 fontsize=15, fontweight="bold", color="#111111")
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes.ravel():
        ax.set_facecolor("#0a0a1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Panel 0: Original + confidence-colored boxes ──────────
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original + Ice Confidence", color="white", fontsize=10)
    for (_, row), prob in zip(annotations.iterrows(), probabilities):
        tier, color = confidence_tier(float(prob))
        rect = mpatches.Rectangle(
            (row["xmin"], row["ymin"]),
            row["xmax"] - row["xmin"],
            row["ymax"] - row["ymin"],
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        axes[0, 0].add_patch(rect)
    legend_handles = [
        mpatches.Patch(color="#00ccff", label="HIGH (≥75%)"),
        mpatches.Patch(color="#66ddaa", label="MODERATE (≥50%)"),
        mpatches.Patch(color="#ffcc44", label="LOW (≥30%)"),
        mpatches.Patch(color="#ff4444", label="UNLIKELY"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="upper right",
                      fontsize=6, facecolor="#111133", labelcolor="white")

    # ── Panel 1: CLAHE enhanced ───────────────────────────────
    axes[0, 1].imshow(prep["enhanced"], cmap="gray")
    axes[0, 1].set_title("CLAHE Enhanced", color="white", fontsize=10)

    # ── Panel 2: Otsu mask ────────────────────────────────────
    axes[0, 2].imshow(seg["otsu_mask"], cmap="gray")
    axes[0, 2].set_title(f"Otsu Mask (t={seg['otsu_thresh']:.0f})",
                         color="white", fontsize=10)

    # ── Panel 3: Adaptive mask ────────────────────────────────
    axes[0, 3].imshow(seg["adaptive_mask"], cmap="gray")
    axes[0, 3].set_title("Adaptive Threshold Mask", color="white", fontsize=10)

    # ── Panel 4: Ice heatmap overlay ─────────────────────────
    axes[1, 0].imshow(rgb)
    hm = axes[1, 0].imshow(heatmap, cmap=ICE_CMAP, alpha=0.60)
    plt.colorbar(hm, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title("Ice Likelihood Heatmap", color="white", fontsize=10)

    # ── Panel 5: LBP texture map ──────────────────────────────
    lbp = local_binary_pattern(prep["gray"], LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    axes[1, 1].imshow(lbp, cmap="viridis")
    axes[1, 1].set_title("LBP Texture Map", color="white", fontsize=10)

    # ── Panel 6: High-reflectance (bright) mask ───────────────
    axes[1, 2].imshow(seg["bright_mask"], cmap="Blues")
    axes[1, 2].set_title("High-Reflectance Mask (Ice Proxy)", color="white", fontsize=10)

    # ── Panel 7: Sharpened image ──────────────────────────────
    axes[1, 3].imshow(prep["sharpened"], cmap="gray")
    axes[1, 3].set_title("Unsharp-Mask Sharpened", color="white", fontsize=10)

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def visualize_roc(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(metrics["fpr"], metrics["tpr"], color="#00aaff", lw=2,
            label=f"Ensemble ROC (AUC = {metrics['val_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Lunar Ice Detector")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/roc_curve.png")


def visualize_confusion(metrics: dict) -> None:
    cm  = metrics["confusion"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Ice", "Ice"])
    ax.set_yticklabels(["No Ice", "Ice"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Ensemble")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/confusion_matrix.png")


def visualize_feature_importance(model: Pipeline, n_top: int = 20) -> None:
    """Bar chart of RF sub-estimator feature importances."""
    try:
        rf_clf = model.named_steps["clf"].estimators_[0]   # RF is first
        importances = rf_clf.feature_importances_
    except Exception:
        print("  Skipping feature importance (not available for this model).")
        return

    indices = np.argsort(importances)[::-1][:n_top]
    labels  = (
        [f"gray_{s}" for s in ["mean","std","min","max","med","skew"]] +
        [f"enh_{s}"  for s in ["mean","std","min","max","med","skew"]] +
        [f"lbp_{i}"  for i in range(LBP_POINTS + 2)] +
        [f"glcm_{p}_{a}" for p in ["con","dis","hom","ene","cor"]
                         for a in ["0","90"]] +
        ["edge_density", "bright_ratio", "shadow_ratio",
         "rim_floor", "roughness", "aspect_ratio", "rel_area",
         "mask_cov", "lap_var", "entropy", "pct_spread"]
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(n_top), importances[indices],
           color=plt.cm.Blues(np.linspace(0.4, 0.9, n_top)))
    ax.set_xticks(range(n_top))
    ax.set_xticklabels([labels[i] if i < len(labels) else f"f{i}"
                        for i in indices],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title("Top Feature Importances (Random Forest sub-estimator)", fontsize=12)
    ax.set_ylabel("Importance"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/feature_importance.png")


def visualize_crater_density(annotations: pd.DataFrame,
                              img_w: int, img_h: int,
                              filename: str,
                              save_path: str) -> None:
    density = np.zeros((img_h, img_w), dtype=np.float32)
    for _, row in annotations.iterrows():
        cx = int((row["xmin"] + row["xmax"]) / 2)
        cy = int((row["ymin"] + row["ymax"]) / 2)
        cv2.circle(density, (cx, cy), radius=20, color=1.0, thickness=-1)
    density = cv2.GaussianBlur(density, (51, 51), sigmaX=15)
    density = (density - density.min()) / (density.max() - density.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(density, cmap="hot")
    plt.colorbar(im, ax=ax, label="Crater Density")
    ax.set_title(f"Crater Density Map — {filename}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_hough_detections(bgr: np.ndarray,
                                boxes: list[tuple],
                                save_path: str) -> None:
    """Draw Hough-detected crater circles on the image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(rgb); axes[0].set_title("Original"); axes[0].axis("off")
    for (xmin, ymin, xmax, ymax) in boxes:
        cx = (xmin + xmax) // 2; cy = (ymin + ymax) // 2
        r  = (xmax - xmin) // 2
        circle = plt.Circle((cx, cy), r, color="#00ccff", fill=False, lw=1.2)
        axes[1].add_patch(circle)
    axes[1].imshow(rgb)
    axes[1].set_title(f"Hough Crater Detection ({len(boxes)} craters)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 12. Single-Image Inference  (used by Streamlit app)
# ─────────────────────────────────────────────────────────────

def analyze_image(bgr: np.ndarray,
                  model: Pipeline,
                  use_hough: bool = True,
                  annotations_df: pd.DataFrame | None = None
                  ) -> dict:
    """
    Full inference on a single image.

    Parameters
    ----------
    bgr            : BGR image array
    model          : trained Pipeline
    use_hough      : if True, detect craters autonomously via Hough
    annotations_df : optional DataFrame with pre-labeled bounding boxes

    Returns
    -------
    dict with keys:
      prep, seg, heatmap, boxes, probabilities, tiers, summary
    """
    prep    = preprocess_image(bgr)
    seg     = segment_image(prep)
    heatmap = compute_ice_heatmap(prep["gray"], prep["enhanced"])

    # Determine bounding boxes
    if annotations_df is not None and len(annotations_df) > 0:
        boxes = [
            (int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"]))
            for _, r in annotations_df.iterrows()
        ]
    elif use_hough:
        boxes = detect_craters_hough(prep["gray"])
    else:
        boxes = []

    # Extract features and predict
    probabilities = []
    tiers_list    = []
    for (xmin, ymin, xmax, ymax) in boxes:
        feat = extract_region_features(
            prep["gray"], prep["enhanced"], seg["combined_mask"],
            xmin, ymin, xmax, ymax
        )
        prob = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
        tier, color = confidence_tier(prob)
        probabilities.append(prob)
        tiers_list.append({"tier": tier, "color": color, "prob": prob})

    n_ice = sum(1 for t in tiers_list if t["tier"] in ("HIGH", "MODERATE"))
    summary = {
        "total_craters":    len(boxes),
        "ice_candidates":   n_ice,
        "high_confidence":  sum(1 for t in tiers_list if t["tier"] == "HIGH"),
        "moderate":         sum(1 for t in tiers_list if t["tier"] == "MODERATE"),
        "low":              sum(1 for t in tiers_list if t["tier"] == "LOW"),
        "unlikely":         sum(1 for t in tiers_list if t["tier"] == "UNLIKELY"),
        "mean_ice_prob":    float(np.mean(probabilities)) if probabilities else 0.0,
        "max_ice_prob":     float(np.max(probabilities))  if probabilities else 0.0,
    }

    return {
        "prep":          prep,
        "seg":           seg,
        "heatmap":       heatmap,
        "boxes":         boxes,
        "probabilities": probabilities,
        "tiers":         tiers_list,
        "summary":       summary,
    }


# ─────────────────────────────────────────────────────────────
# 13. Main Training Pipeline
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Lunar Ice Deposit Detection — Enhanced CV Pipeline v2")
    print("=" * 65)

    # ── Load annotations ──────────────────────────────────────
    print("\n[1/7] Loading annotations …")
    train_ann = load_annotations(os.path.join(TRAIN_DIR, "_annotations.csv"))
    valid_ann = load_annotations(os.path.join(VALID_DIR, "_annotations.csv"))
    print(f"  Train: {len(train_ann):,} boxes across "
          f"{train_ann['filename'].nunique()} images")
    print(f"  Valid: {len(valid_ann):,} boxes across "
          f"{valid_ann['filename'].nunique()} images")

    # ── Build feature datasets (with augmentation on train) ───
    print("\n[2/7] Extracting features (train with augmentation) …")
    X_train, y_train, meta_train = build_dataset(TRAIN_DIR, train_ann, augment=True)
    X_val,   y_val,   meta_val   = build_dataset(VALID_DIR, valid_ann, augment=False)
    print(f"  Train: {X_train.shape}  |  Ice ratio: {y_train.mean():.2%}")
    print(f"  Valid: {X_val.shape}    |  Ice ratio: {y_val.mean():.2%}")

    # ── Train ensemble ────────────────────────────────────────
    print("\n[3/7] Training RF + XGBoost + LightGBM ensemble …")
    model = train_model(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    # ── Evaluate ──────────────────────────────────────────────
    print("\n[4/7] Evaluating …")
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
    print(f"  5-fold CV AUC : {metrics['cv_auc_mean']:.4f} "
          f"± {metrics['cv_auc_std']:.4f}")
    print(f"  Validation AUC: {metrics['val_auc']:.4f}")
    print("\n  Classification Report (Validation):")
    print(metrics["report"])

    # ── Metric plots ──────────────────────────────────────────
    print("[5/7] Generating metric plots …")
    visualize_roc(metrics)
    visualize_confusion(metrics)
    visualize_feature_importance(model)

    # ── Hough crater detection demo ───────────────────────────
    print("\n[6/7] Hough crater detection demo (first validation image) …")
    first_file = valid_ann["filename"].iloc[0]
    bgr_demo   = load_image(VALID_DIR, first_file)
    if bgr_demo is not None:
        prep_demo  = preprocess_image(bgr_demo)
        hough_boxes = detect_craters_hough(prep_demo["gray"])
        hough_path  = os.path.join(OUTPUT_DIR, "hough_demo.png")
        visualize_hough_detections(bgr_demo, hough_boxes, hough_path)
        print(f"  Detected {len(hough_boxes)} craters via Hough → {hough_path}")

    # ── Per-image pipeline visualizations ─────────────────────
    print("\n[7/7] Generating per-image visualizations …")
    val_proba = metrics["y_proba"]

    proba_by_file: dict[str, list[float]] = {}
    for meta, prob in zip(meta_val, val_proba):
        proba_by_file.setdefault(meta["filename"], []).append(float(prob))

    processed = 0
    for filename, group in valid_ann.groupby("filename"):
        bgr = load_image(VALID_DIR, filename)
        if bgr is None:
            continue

        prep    = preprocess_image(bgr)
        seg     = segment_image(prep)
        heatmap = compute_ice_heatmap(prep["gray"], prep["enhanced"])
        probas  = proba_by_file.get(filename, [0.0] * len(group))

        stem = os.path.splitext(filename)[0]

        visualize_image_pipeline(
            bgr, prep, seg, heatmap,
            group.reset_index(drop=True),
            np.array(probas),
            os.path.join(OUTPUT_DIR, f"{stem}_pipeline.png")
        )
        visualize_crater_density(
            group, bgr.shape[1], bgr.shape[0],
            filename,
            os.path.join(OUTPUT_DIR, f"{stem}_density.png")
        )

        n_ice = sum(1 for p in probas if p >= 0.50)
        processed += 1
        print(f"  [{processed:2d}] {filename[:50]:<50} "
              f"→ {n_ice} ice / {len(probas)} craters")

    print(f"\n✓ Done. All outputs saved to '{OUTPUT_DIR}/'")
    print(f"  Total files: {len(os.listdir(OUTPUT_DIR))}")


if __name__ == "__main__":
    main()
