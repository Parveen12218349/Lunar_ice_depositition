"""
Lunar Ice Deposit Detection — Streamlit Web App
================================================
Upload any lunar satellite image and get:
  • Autonomous crater detection (Hough Transform)
  • Per-crater ice likelihood score with confidence tiers
  • Color-coded heatmap overlay
  • Full 8-panel processing pipeline visualization
  • Downloadable results

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Community Cloud
"""

import io
import os
import tempfile

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from PIL import Image

from ice_deposit_detection import (
    MODEL_PATH,
    OUTPUT_DIR,
    ICE_CMAP,
    LBP_POINTS,
    LBP_RADIUS,
    LBP_METHOD,
    analyze_image,
    load_image_from_bytes,
    preprocess_image,
    segment_image,
    compute_ice_heatmap,
    detect_craters_hough,
    confidence_tier,
    train_model,
    build_dataset,
    load_annotations,
    TRAIN_DIR,
    VALID_DIR,
)
from skimage.feature import local_binary_pattern

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lunar Ice Detector",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a1a; color: #e0e8ff; }
    .stApp { background-color: #0a0a1a; }
    h1, h2, h3 { color: #00aaff; }
    .metric-card {
        background: #111133;
        border: 1px solid #223366;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .tier-high     { color: #00ccff; font-weight: bold; }
    .tier-moderate { color: #66ddaa; font-weight: bold; }
    .tier-low      { color: #ffcc44; font-weight: bold; }
    .tier-unlikely { color: #ff4444; font-weight: bold; }
    .stButton>button {
        background: linear-gradient(135deg, #0055aa, #0099ff);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-size: 16px;
    }
    .stButton>button:hover { background: #0077cc; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Auto-train if model not found
    st.warning("Model not found — training now. This takes ~2 minutes…")
    train_ann = load_annotations(os.path.join(TRAIN_DIR, "_annotations.csv"))
    X_train, y_train, _ = build_dataset(TRAIN_DIR, train_ann, augment=True)
    model = train_model(X_train, y_train)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


# ─────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────
def make_annotated_image(bgr: np.ndarray,
                          boxes: list,
                          probabilities: list) -> np.ndarray:
    """Return RGB image with confidence-colored bounding boxes."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    overlay = rgb.copy()
    for (xmin, ymin, xmax, ymax), prob in zip(boxes, probabilities):
        tier, hex_color = confidence_tier(float(prob))
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (r, g, b), 2)
        label = f"{tier} {prob:.0%}"
        cv2.putText(overlay, label, (xmin, max(ymin - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (r, g, b), 1, cv2.LINE_AA)
    return overlay


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)


def make_pipeline_figure(bgr, prep, seg, heatmap, boxes, probabilities) -> plt.Figure:
    """8-panel pipeline figure for display in Streamlit."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes.ravel():
        ax.set_facecolor("#0a0a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#223366")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Panel 0 — annotated
    axes[0, 0].imshow(make_annotated_image(bgr, boxes, probabilities))
    axes[0, 0].set_title("Detections + Confidence", color="white", fontsize=9)

    # Panel 1 — CLAHE
    axes[0, 1].imshow(prep["enhanced"], cmap="gray")
    axes[0, 1].set_title("CLAHE Enhanced", color="white", fontsize=9)

    # Panel 2 — Otsu
    axes[0, 2].imshow(seg["otsu_mask"], cmap="gray")
    axes[0, 2].set_title(f"Otsu Mask (t={seg['otsu_thresh']:.0f})",
                         color="white", fontsize=9)

    # Panel 3 — Adaptive
    axes[0, 3].imshow(seg["adaptive_mask"], cmap="gray")
    axes[0, 3].set_title("Adaptive Threshold", color="white", fontsize=9)

    # Panel 4 — Heatmap
    axes[1, 0].imshow(rgb)
    hm = axes[1, 0].imshow(heatmap, cmap=ICE_CMAP, alpha=0.60)
    plt.colorbar(hm, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title("Ice Likelihood Heatmap", color="white", fontsize=9)

    # Panel 5 — LBP
    lbp = local_binary_pattern(prep["gray"], LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    axes[1, 1].imshow(lbp, cmap="viridis")
    axes[1, 1].set_title("LBP Texture Map", color="white", fontsize=9)

    # Panel 6 — Bright mask
    axes[1, 2].imshow(seg["bright_mask"], cmap="Blues")
    axes[1, 2].set_title("High-Reflectance Mask", color="white", fontsize=9)

    # Panel 7 — Sharpened
    axes[1, 3].imshow(prep["sharpened"], cmap="gray")
    axes[1, 3].set_title("Sharpened Image", color="white", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("Full Processing Pipeline", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Lunar Ice Detector")
    st.markdown("---")
    st.markdown("### Detection Settings")

    use_hough = st.toggle("Autonomous Crater Detection (Hough)", value=True,
                          help="Detect craters automatically without annotations")

    if use_hough:
        st.markdown("**Hough Parameters**")
        hough_min_r = st.slider("Min crater radius (px)", 3, 30, 5)
        hough_max_r = st.slider("Max crater radius (px)", 20, 150, 80)
        hough_param2 = st.slider("Detection sensitivity", 10, 50, 25,
                                 help="Lower = more craters detected")
    else:
        hough_min_r, hough_max_r, hough_param2 = 5, 80, 25

    st.markdown("---")
    st.markdown("### Confidence Thresholds")
    st.markdown("""
    <span class='tier-high'>■ HIGH</span> ≥ 75% probability<br>
    <span class='tier-moderate'>■ MODERATE</span> ≥ 50% probability<br>
    <span class='tier-low'>■ LOW</span> ≥ 30% probability<br>
    <span class='tier-unlikely'>■ UNLIKELY</span> < 30% probability
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Detects potential ice deposits in lunar satellite imagery using:
    - **CLAHE** histogram equalization
    - **Otsu + Adaptive** thresholding
    - **LBP + GLCM** texture analysis
    - **RF + XGBoost + LightGBM** ensemble
    - **59-feature** vector per crater
    """)


# ─────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────
st.markdown("# 🌙 Lunar Ice Deposit Detection")
st.markdown(
    "Upload a lunar satellite image to detect craters and score them "
    "for ice-deposit likelihood using a multi-signal computer vision pipeline."
)

model = load_model()

# ── Upload ────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a lunar satellite image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Grayscale or color satellite imagery works best"
)

# ── Demo images ───────────────────────────────────────────────
st.markdown("**Or try a sample image:**")
sample_files = []
for d in [VALID_DIR, TRAIN_DIR]:
    if os.path.isdir(d):
        sample_files += [
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
sample_files = sample_files[:6]

if sample_files:
    cols = st.columns(min(6, len(sample_files)))
    for col, path in zip(cols, sample_files):
        with col:
            thumb = Image.open(path).resize((80, 60))
            if col.button(os.path.basename(path)[:12], key=path):
                with open(path, "rb") as f:
                    uploaded = io.BytesIO(f.read())
                    uploaded.name = os.path.basename(path)

# ── Analysis ──────────────────────────────────────────────────
if uploaded is not None:
    raw_bytes = uploaded.read() if hasattr(uploaded, "read") else open(uploaded, "rb").read()
    bgr = load_image_from_bytes(raw_bytes)

    if bgr is None:
        st.error("Could not decode image. Please upload a valid JPG or PNG.")
        st.stop()

    with st.spinner("Analyzing image…"):
        result = analyze_image(
            bgr, model,
            use_hough=use_hough,
        )
        # Re-run Hough with user params if needed
        if use_hough:
            from ice_deposit_detection import (
                HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1
            )
            boxes = detect_craters_hough(
                result["prep"]["gray"],
                dp=HOUGH_DP,
                min_dist=HOUGH_MIN_DIST,
                param1=HOUGH_PARAM1,
                param2=hough_param2,
                min_r=hough_min_r,
                max_r=hough_max_r,
            )
            # Re-predict for new boxes
            from ice_deposit_detection import extract_region_features
            probabilities = []
            for (xmin, ymin, xmax, ymax) in boxes:
                feat = extract_region_features(
                    result["prep"]["gray"],
                    result["prep"]["enhanced"],
                    result["seg"]["combined_mask"],
                    xmin, ymin, xmax, ymax,
                )
                prob = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
                probabilities.append(prob)
            result["boxes"]         = boxes
            result["probabilities"] = probabilities
            result["tiers"]         = [
                {"tier": confidence_tier(p)[0],
                 "color": confidence_tier(p)[1],
                 "prob": p}
                for p in probabilities
            ]
            n_ice = sum(1 for t in result["tiers"] if t["tier"] in ("HIGH", "MODERATE"))
            result["summary"].update({
                "total_craters":  len(boxes),
                "ice_candidates": n_ice,
                "high_confidence": sum(1 for t in result["tiers"] if t["tier"] == "HIGH"),
                "moderate":        sum(1 for t in result["tiers"] if t["tier"] == "MODERATE"),
                "low":             sum(1 for t in result["tiers"] if t["tier"] == "LOW"),
                "unlikely":        sum(1 for t in result["tiers"] if t["tier"] == "UNLIKELY"),
                "mean_ice_prob":   float(np.mean(probabilities)) if probabilities else 0.0,
                "max_ice_prob":    float(np.max(probabilities))  if probabilities else 0.0,
            })

    summary = result["summary"]

    # ── Summary metrics ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## Analysis Results")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Craters",    summary["total_craters"])
    c2.metric("Ice Candidates",   summary["ice_candidates"])
    c3.metric("HIGH Confidence",  summary["high_confidence"])
    c4.metric("Max Ice Prob",     f"{summary['max_ice_prob']:.1%}")
    c5.metric("Mean Ice Prob",    f"{summary['mean_ice_prob']:.1%}")

    # ── Tier breakdown bar ────────────────────────────────────
    if summary["total_craters"] > 0:
        st.markdown("**Confidence Tier Breakdown**")
        tier_data = {
            "HIGH":     summary["high_confidence"],
            "MODERATE": summary["moderate"],
            "LOW":      summary["low"],
            "UNLIKELY": summary["unlikely"],
        }
        tier_colors = ["#00ccff", "#66ddaa", "#ffcc44", "#ff4444"]
        fig_bar, ax_bar = plt.subplots(figsize=(8, 1.2))
        fig_bar.patch.set_facecolor("#111133")
        ax_bar.set_facecolor("#111133")
        left = 0
        for (label, val), color in zip(tier_data.items(), tier_colors):
            if val > 0:
                ax_bar.barh(0, val, left=left, color=color, height=0.6)
                if val / summary["total_craters"] > 0.05:
                    ax_bar.text(left + val / 2, 0, f"{label}\n{val}",
                                ha="center", va="center", fontsize=8,
                                color="black", fontweight="bold")
                left += val
        ax_bar.set_xlim(0, summary["total_craters"])
        ax_bar.axis("off")
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # ── Side-by-side: original vs annotated ──────────────────
    st.markdown("---")
    st.markdown("## Image Analysis")
    col_orig, col_ann = st.columns(2)

    with col_orig:
        st.markdown("**Original Image**")
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_ann:
        st.markdown("**Crater Detections + Confidence**")
        annotated = make_annotated_image(
            bgr, result["boxes"], result["probabilities"]
        )
        st.image(annotated, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Ice Likelihood Heatmap")
    fig_hm, ax_hm = plt.subplots(figsize=(10, 5))
    fig_hm.patch.set_facecolor("#0a0a1a")
    ax_hm.set_facecolor("#0a0a1a")
    ax_hm.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    hm_im = ax_hm.imshow(result["heatmap"], cmap=ICE_CMAP, alpha=0.60)
    plt.colorbar(hm_im, ax=ax_hm, label="Ice Likelihood Score")
    ax_hm.set_title("Ice Likelihood Heatmap", color="white")
    ax_hm.axis("off")
    st.pyplot(fig_hm, use_container_width=True)
    plt.close(fig_hm)

    # ── Full pipeline ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Full Processing Pipeline")
    fig_pipe = make_pipeline_figure(
        bgr,
        result["prep"],
        result["seg"],
        result["heatmap"],
        result["boxes"],
        result["probabilities"],
    )
    st.pyplot(fig_pipe, use_container_width=True)
    plt.close(fig_pipe)

    # ── Per-crater table ──────────────────────────────────────
    if result["boxes"] and len(result["boxes"]) <= 200:
        st.markdown("---")
        st.markdown("## Per-Crater Results")
        import pandas as pd
        rows = []
        for i, ((xmin, ymin, xmax, ymax), prob, tier_info) in enumerate(
            zip(result["boxes"], result["probabilities"], result["tiers"])
        ):
            rows.append({
                "Crater #":    i + 1,
                "X (center)":  (xmin + xmax) // 2,
                "Y (center)":  (ymin + ymax) // 2,
                "Width (px)":  xmax - xmin,
                "Height (px)": ymax - ymin,
                "Ice Prob":    f"{prob:.1%}",
                "Confidence":  tier_info["tier"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=300)

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Results CSV",
            data=csv_bytes,
            file_name="ice_detection_results.csv",
            mime="text/csv",
        )

    # ── Download pipeline image ───────────────────────────────
    buf = io.BytesIO()
    fig_pipe.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                     facecolor="#0a0a1a")
    buf.seek(0)
    st.download_button(
        "⬇ Download Pipeline Image",
        data=buf,
        file_name="pipeline_analysis.png",
        mime="image/png",
    )

else:
    # Landing state
    st.markdown("---")
    st.info(
        "👆 Upload a lunar satellite image above, or click one of the sample "
        "images to get started."
    )
    st.markdown("""
    ### How it works
    1. **Preprocessing** — CLAHE histogram equalization + Gaussian blur + Canny edges
    2. **Segmentation** — Otsu + Adaptive thresholding + morphological cleanup
    3. **Crater Detection** — Circular Hough Transform (autonomous, no annotations needed)
    4. **Feature Extraction** — 59 features per crater: intensity stats, LBP texture,
       GLCM texture, edge density, shadow ratio, rim/floor contrast, and more
    5. **Classification** — RF + XGBoost + LightGBM soft-voting ensemble
    6. **Scoring** — Each crater gets an ice probability + confidence tier
    7. **Visualization** — Color-coded heatmap + annotated detections + full pipeline view
    """)
