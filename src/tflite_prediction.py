import numpy as np

# -----------------------------
# TFLITE PREDICTION
# -----------------------------
def tflite_predict(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]['index'],
        img_array.astype(np.float32)
    )
    interpreter.invoke()

    preds = interpreter.get_tensor(
        output_details[0]['index']
    )
    return preds