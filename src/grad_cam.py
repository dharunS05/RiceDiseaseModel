import tensorflow as tf
import cv2
import numpy as np

def make_gradcam_heatmap(img_array, model, conv_tensor, pred_index):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [conv_tensor, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

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
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if original_img.dtype != np.uint8:
        original_img = np.uint8(original_img)

    overlay = cv2.addWeighted(
        original_img, 1 - alpha,
        heatmap_color, alpha, 0
    )

    return overlay