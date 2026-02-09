import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

from src.config import *
from src.preprocess import *
from src.grad_cam import *
from src.tflite_prediction import *

# -----------------------------
# CONFIG
# -----------------------------


st.set_page_config(
    page_title="Rice Disease Detection",
    layout="wide"
)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_tf_model():
    return load_model(
        "/content/mydrive/MyDrive/rice_disease_models/rice_disease_model1.keras",
        compile=False
    )

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(
        model_path="/content/mydrive/MyDrive/rice_disease_models/rice_disease_model1.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

tf_model = load_tf_model()
tflite_interpreter = load_tflite_model()






# -----------------------------
# UI
# -----------------------------
st.title("🌾 Rice Disease Detection (TFLite + CAM-Lite)")

uploaded_file = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array, original_img = preprocess_image(image)

    # ---- Prediction (TFLite)
    preds = tflite_predict(tflite_interpreter, img_array)
    pred_class = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_class])

    # ---- CAM-Lite Heatmap
    heatmap = make_cam_lite_heatmap(tf_model, img_array)
    cam_img = overlay_gradcam(original_img, heatmap)

    # ---- DISPLAY
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Localization (CAM-Lite)")
        st.image(cam_img, channels="BGR", use_column_width=True)

    # ---- RESULT
    st.markdown("---")
    st.subheader("Prediction Result")

    if confidence < 0.50:
        st.info("⚠️ Confidence below 50%. Please retake the photo.")
    else:
        st.success(f"**Disease:** {CLASS_NAMES[pred_class]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
