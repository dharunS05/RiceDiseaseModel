
import numpy as np
from config import *

# -----------------------------
# PREPROCESS IMAGE (VALIDATION STYLE)
# -----------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image)

    original_img = img.copy()  # for display

    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / \
          np.array([0.229, 0.224, 0.225])

    img = np.expand_dims(img, axis=0)
    return img, original_img