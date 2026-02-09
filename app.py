import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

from src.grad_cam import *
from src.config import *
from src.preprocess import *

# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(
    page_title="Rice Disease Detection",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model("/content/mydrive/MyDrive/rice_disease_models/rice_disease_model1.keras", compile=False)

model = load_trained_model()


# -----------------------------
# UI
# -----------------------------
st.title("🌾 Rice Disease Detection with Grad-CAM")

uploaded_file = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array, original_img = preprocess_image(image)

    # Prediction
    preds = model.predict(img_array, verbose=0)
    pred_class = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_class])

    # Grad-CAM
    conv_tensor = model.get_layer(LAST_CONV_LAYER).output
    heatmap = make_gradcam_heatmap(
        img_array, model, conv_tensor, pred_class
    )
    gradcam_img = overlay_gradcam(original_img, heatmap)

    # DISPLAY
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Grad-CAM Visualization")
        st.image(gradcam_img, channels="BGR", use_column_width=True)

    # RESULT
    st.markdown("---")
    st.subheader("Prediction Result")

    if confidence < 0.50:
        st.info("⚠️ Confidence below 50%. Please retake the photo.")
    else:
        st.success(f"**Disease:** {CLASS_NAMES[pred_class]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
