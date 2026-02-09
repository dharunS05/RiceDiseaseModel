import tensorflow as tf
import cv2
import numpy as np
from config import *

def make_cam_lite_heatmap(model, img_array):
    feature_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(LAST_CONV_LAYER).output
    )

    conv_features = feature_model(img_array, training=False)[0]

    heatmap = tf.reduce_mean(conv_features, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.45):
    heatmap = heatmap.astype(np.float32)

    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    if original_img.dtype != np.uint8:
        original_img = np.uint8(original_img)

    overlay = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay